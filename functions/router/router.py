"""
title: Router
description: Smart router that selects the best expert model for each query based on LLM classification (primary), semantic similarity, and keywords.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 2.1.0
license: MIT
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb
"""

import json
import logging
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple
import anyio
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

import sys

if "/app/backend/data/custom_lib" not in sys.path:
    sys.path.append("/app/backend/data/custom_lib")

try:
    from shared_resources import SQLiteCache, AsyncLRUCache

    _SHARED_RESOURCES_AVAILABLE = True
except ImportError:
    _SHARED_RESOURCES_AVAILABLE = False


def _build_cache(table: str, max_size: int, ttl: int, db_path: str):
    if _SHARED_RESOURCES_AVAILABLE:
        return SQLiteCache(db_path=db_path, table=table, max_size=max_size, ttl=ttl)

    import asyncio as _asyncio, time as _time

    class _FallbackCache:
        def __init__(self, max_size, ttl):
            self._store = {}
            self.max_size = max_size
            self.ttl = ttl
            self._lock = _asyncio.Lock()

        async def get(self, key):
            async with self._lock:
                e = self._store.get(key)
                if not e:
                    return None
                val, ts = e
                if self.ttl > 0 and _time.time() - ts > self.ttl:
                    del self._store[key]
                    return None
                self._store[key] = self._store.pop(key)
                return val

        async def set(self, key, val):
            async with self._lock:
                if (
                    self.max_size > 0
                    and len(self._store) >= self.max_size
                    and key not in self._store
                ):
                    del self._store[next(iter(self._store))]
                self._store[key] = (val, _time.time())
                self._store[key] = self._store.pop(key)

        async def clear(self):
            async with self._lock:
                self._store.clear()

    return _FallbackCache(max_size, ttl)


class Filter:
    class Valves(BaseModel):
        experts_json: str = Field(
            default="""[
        {
            "id": "experto-en-neovim",
            "name": "Neovim",
            "keywords": ["neovim", "nvim", "vimrc", "init.lua", "lazy.nvim", "vim-plug", "treesitter", "keymap", "which-key"],
            "description": "Configuration, plugins, and usage of Neovim editor",
            "examples": ["how to configure LSP in nvim", "best neovim plugins 2025", "treesitter setup init.lua"],
            "knowledge_base": "the official Neovim documentation and community plugins",
            "collection_name": "neovim_docs"
        },
        {
            "id": "experto-en-arch-linux",
            "name": "Arch Linux",
            "keywords": ["archlinux", "pacman", "yay", "aur", "systemd", "grub", "hyprland", "wayland"],
            "description": "Installation, maintenance, and troubleshooting of Arch Linux",
            "examples": ["how to install yay", "hyprland config", "pacman update error"],
            "knowledge_base": "the official Arch Linux wiki",
            "collection_name": "archwiki"
        },
        {
            "id": "arquitecto-de-codigo",
            "name": "Arquitecto de código",
            "keywords": [
                "architecture", "arquitectura", "design pattern", "patrón de diseño",
                "microservices", "microservicios", "system design", "diseño de sistemas",
                "scalability", "escalabilidad", "ddd", "domain driven design",
                "event sourcing", "c4 model", "modelo c4", "monolith", "monolito",
                "orchestration", "orquestación", "saga", "cqrs",
                "trade-off", "clean architecture", "arquitectura limpia",
                "hexagonal", "onion architecture", "architectural decision record", "adr",
                "component diagram", "diagrama de componentes"
            ],
            "description": "Maximum precision model for software architecture tasks",
            "examples": [
                "compara microservicios con monolito modular",
                "compare microservices vs modular monolith",
                "diseña la arquitectura de un sistema de reservas",
                "design the architecture of a booking system",
                "when to use event sourcing?",
                "draw a C4 diagram for an ecommerce app",
                "explica los componentes de una clean architecture"
            ]
        },
        {
            "id": "progrmador",
            "name": "Programador",
            "keywords": [
                "function", "función", "class", "clase",
                "algorithm", "algoritmo", "debug", "depurar", "bug",
                "implement", "implementar", "python", "javascript", "api", "script",
                "optimize", "optimizar", "refactor", "refactorizar",
                "snippet", "fragmento", "error", "excepción",
                "library", "librería", "framework", "endpoint", "test", "prueba unitaria"
            ],
            "description": "Precise model for code implementation tasks",
            "examples": [
                "escribe una función que calcule el factorial",
                "write a function to calculate factorial",
                "how to fix CORS error in express",
                "optimize this slow SQL query",
                "refactor this class using the strategy pattern"
            ]
        }
    ]""",
            description="JSON with expert definitions. 'model' field is no longer needed – the router respects the model already selected by the user.",
        )
        default_model: str = Field(default="generalista")
        default_model_name: str = Field(
            default="llamacpp/llama3.2:3b",
            description="Real model name for 'generalista' (with llamacpp/ prefix)",
        )
        change_threshold: int = Field(default=2)
        notify_change: bool = Field(default=True)
        notification_template: str = Field(default="🎯 Using expert: {expert}")

        LLM_BASE_URL: str = Field(default="http://localhost:8080")
        LLM_API_TOKEN: str = Field(default="")
        classifier_model: str = Field(
            default="llamacpp/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        classifier_temperature: float = Field(default=0.0)
        classifier_timeout: int = Field(default=15)
        default_llm_temperature: float = Field(default=0.3)
        default_llm_max_tokens: int = Field(default=256)
        default_llm_timeout: int = Field(default=30)
        enable_query_rewriting: bool = Field(default=True)
        enable_rag_injection: bool = Field(default=True)
        rewrite_cache_max_size: int = Field(default=500)
        string_cache_max_size: int = Field(default=1000)
        string_cache_ttl: int = Field(default=1800)
        rewrite_cache_ttl: int = Field(default=3600)
        USE_SEMANTIC_CLASSIFY: bool = Field(default=True)
        SEMANTIC_THRESHOLD: float = Field(default=0.55)
        CACHE_DB_PATH: str = Field(default="/app/backend/data/router_cache.db")
        DEBUG: bool = Field(default=True)

        # New valve: choose between chat and text completions for llama.cpp
        llamacpp_endpoint_type: str = Field(
            default="chat",
            description="Endpoint type for llama.cpp: 'chat' (default) or 'completion'.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._experts: List[dict] = []
        self._experts_json_hash: Optional[str] = None
        self._load_experts()
        db_path = self.valves.CACHE_DB_PATH
        self._string_cache = _build_cache(
            "classifications",
            self.valves.string_cache_max_size,
            self.valves.string_cache_ttl,
            db_path,
        )
        self._rewrite_cache = _build_cache(
            "rewrites",
            self.valves.rewrite_cache_max_size,
            self.valves.rewrite_cache_ttl,
            db_path,
        )
        self._embeddings_lock = asyncio.Lock()

    def _load_experts(self):
        try:
            new_json = self.valves.experts_json
            new_hash = hashlib.md5(new_json.encode()).hexdigest()
            if self._experts_json_hash == new_hash:
                if self.valves.DEBUG:
                    logger.info(
                        "[Router] Expert configuration unchanged, keeping previous."
                    )
                return
            parsed = json.loads(new_json)
            for exp in parsed:
                required = {"id", "name", "keywords"}
                if not required.issubset(exp.keys()):
                    raise ValueError(f"Expert missing required fields: {exp}")
            self._experts = parsed
            self._experts_json_hash = new_hash
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] Loaded {len(self._experts)} experts. JSON hash updated."
                )
        except Exception as e:
            logger.error(
                f"[Router] Error parsing experts_json: {e}. Keeping previous expert list."
            )

    async def _sync_cache_config(self):
        if (
            self._string_cache.max_size != self.valves.string_cache_max_size
            or self._string_cache.ttl != self.valves.string_cache_ttl
        ):
            self._string_cache.max_size = self.valves.string_cache_max_size
            self._string_cache.ttl = self.valves.string_cache_ttl
            if hasattr(self._string_cache, "_ram"):
                self._string_cache._ram.max_size = self.valves.string_cache_max_size
                self._string_cache._ram.ttl = self.valves.string_cache_ttl
        if (
            self._rewrite_cache.max_size != self.valves.rewrite_cache_max_size
            or self._rewrite_cache.ttl != self.valves.rewrite_cache_ttl
        ):
            self._rewrite_cache.max_size = self.valves.rewrite_cache_max_size
            self._rewrite_cache.ttl = self.valves.rewrite_cache_ttl
            if hasattr(self._rewrite_cache, "_ram"):
                self._rewrite_cache._ram.max_size = self.valves.rewrite_cache_max_size
                self._rewrite_cache._ram.ttl = self.valves.rewrite_cache_ttl

    async def _call_llm(
        self,
        prompt: str,
        system: str = "",
        provider: str = "",
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ) -> str:
        if temperature is None:
            temperature = self.valves.default_llm_temperature
        if max_tokens is None:
            max_tokens = self.valves.default_llm_max_tokens
        if timeout is None:
            timeout = self.valves.default_llm_timeout
        model_str = provider or self.valves.classifier_model

        # Normalize base URL: remove trailing /v1 if present
        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")

        # Detect Ollama by URL or port (legacy)
        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

        # Force OpenAI-compatible path if model has llamacpp/ prefix
        is_llamacpp = model_str.startswith("llamacpp/")
        if is_llamacpp:
            is_ollama = False

        # Extract real model name (strip provider prefix)
        if "/" in model_str and (
            model_str.startswith("ollama/") or model_str.startswith("llamacpp/")
        ):
            model_name = model_str.split("/", 1)[1]
        else:
            model_name = model_str

        # Determine endpoint type for llama.cpp (from valve)
        ep_type = "chat"
        if is_llamacpp:
            ep_type = self.valves.llamacpp_endpoint_type

        if _SHARED_RESOURCES_AVAILABLE:
            from shared_resources import call_llm

            try:
                return await call_llm(
                    prompt=prompt,
                    system=system,
                    base_url=self.valves.LLM_BASE_URL,
                    model=model_str,
                    api_token=self.valves.LLM_API_TOKEN,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    endpoint_type=ep_type,
                )
            except Exception as e:
                logger.warning(f"[Router] shared call_llm failed: {e}, using fallback")

        # Fallback HTTP
        import aiohttp
        from shared_resources import get_http_session

        api_token = (
            self.valves.LLM_API_TOKEN.strip() if self.valves.LLM_API_TOKEN else None
        )
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        if is_ollama:
            url = f"{base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        else:
            if ep_type == "completion":
                url = f"{base_url}/v1/completions"
                payload = {
                    "model": model_name,
                    "prompt": prompt if not system else f"{system}\n\n{prompt}",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            else:  # chat
                url = f"{base_url}/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

        session = await get_http_session(timeout=timeout)
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM HTTP {resp.status}: {text[:200]}")
                data = await resp.json()
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Router LLM connection error: {exc}") from exc

        if is_ollama:
            content = data.get("response", "")
            if "error" in data:
                logger.error(f"[Router] Ollama error: {data['error']}")
                return ""
        else:
            if ep_type == "completion":
                content = data["choices"][0].get("text", "")
            else:
                content = data["choices"][0]["message"]["content"]

        content = content.strip()
        if not content:
            logger.warning(f"[Router] LLM returned empty content for '{model_name}'.")
        if self.valves.DEBUG:
            logger.debug(f"[Router] LLM raw response: {content[:300]}")
        return content

    async def _classify_with_llm(self, user_query: str) -> Optional[str]:
        lines = []
        for exp in self._experts:
            line = f"- {exp['id']}: {exp['name']}"
            if exp.get("description"):
                line += f" ({exp['description']})"
            lines.append(line)
        expert_list = "\n".join(lines)
        examples_section = ""
        for exp in self._experts:
            if exp.get("examples"):
                for ex in exp["examples"]:
                    examples_section += f"User: {ex}\nExpert: {exp['id']}\n"
        system_prompt = (
            "You are a smart router. Choose the expert that best matches the user query.\n"
            "Reply with ONLY the expert ID (one word) or 'generalista'.\n\n"
            f"Available experts:\n{expert_list}\n"
        )
        if examples_section:
            system_prompt += f"\nExamples:\n{examples_section}\n"
        system_prompt += "\nExpert ID:"
        try:
            response = await self._call_llm(
                prompt=user_query,
                system=system_prompt,
                provider=self.valves.classifier_model,
                temperature=self.valves.classifier_temperature,
                max_tokens=10,
                timeout=self.valves.classifier_timeout,
            )
            if not response:
                return None
            first_word = response.strip().split()[0].lower()
            valid_ids = {exp["id"] for exp in self._experts}
            valid_ids.add("generalista")
            if first_word in valid_ids:
                return first_word
            else:
                logger.warning(
                    f"[Router] Unexpected LLM classifier output: '{first_word}'"
                )
                return None
        except Exception as e:
            logger.error(f"[Router] LLM classifier error: {e}")
            return None

    def _keyword_fallback(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for exp in self._experts:
            if any(kw in text_lower for kw in exp.get("keywords", [])):
                return exp["id"]
        return None

    def _get_cache_key(
        self, user_query: str, messages: list = None, context_window: int = 2
    ) -> str:
        base = user_query.lower().strip()
        if messages and len(messages) > 1:
            recent = [
                m
                for m in messages[-context_window - 1 : -1]
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            if recent:
                context_str = "|".join(
                    f"{m['role']}:{str(m.get('content', ''))[:80]}" for m in recent
                )
                ctx_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
                return f"{base}::{ctx_hash}"
        return base

    def _build_expert_example_embeddings(self, experts_config: list) -> dict:
        try:
            if _SHARED_RESOURCES_AVAILABLE:
                from shared_resources import get_embedder

                embedder = get_embedder()
            else:
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return {}
        import numpy as np

        result = {}
        for expert in experts_config:
            expert_id = expert.get("id") or expert.get("name", "")
            examples = expert.get("examples", [])
            if not examples or not expert_id:
                continue
            try:
                result[expert_id] = embedder.encode(
                    examples, convert_to_numpy=True, batch_size=32
                )
            except Exception:
                pass
        return result

    async def _semantic_classify(
        self, user_query: str, experts_config: list, threshold: float = None
    ) -> str:
        import numpy as np

        if threshold is None:
            threshold = self.valves.SEMANTIC_THRESHOLD
        if not experts_config:
            return ""
        current_hash = self._experts_json_hash
        async with self._embeddings_lock:
            if (
                not hasattr(self, "_expert_embeddings")
                or not hasattr(self, "_expert_embeddings_hash")
                or self._expert_embeddings_hash != current_hash
            ):
                import anyio

                self._expert_embeddings = await anyio.to_thread.run_sync(
                    lambda: self._build_expert_example_embeddings(experts_config)
                )
                self._expert_embeddings_hash = current_hash
        if not self._expert_embeddings:
            return ""
        try:
            if _SHARED_RESOURCES_AVAILABLE:
                from shared_resources import get_embedder

                embedder = get_embedder()
            else:
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return ""
        import anyio

        query_vec = await anyio.to_thread.run_sync(
            lambda: embedder.encode([user_query], convert_to_numpy=True)[0]
        )
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        best_id, best_score = "", -1.0
        for expert_id, example_vecs in self._expert_embeddings.items():
            norms = np.linalg.norm(example_vecs, axis=1, keepdims=True) + 1e-10
            scores = (example_vecs / norms) @ q_norm
            score = float(scores.max())
            if score > best_score:
                best_score, best_id = score, expert_id
        if best_score >= threshold:
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] Semantic classify: '{user_query[:50]}' → {best_id} (sim={best_score:.3f})"
                )
            return best_id
        return ""

    async def _classify_query(self, user_query: str, messages: list = None) -> str:
        cache_key = self._get_cache_key(user_query, messages)
        cached = await self._string_cache.get(cache_key)
        if cached is not None:
            if self.valves.DEBUG:
                logger.info(f"[Router] Cache hit: '{user_query[:50]}' → {cached}")
            return cached

        expert_id = ""

        # ── Sticky routing: if the active expert matches the semantic prediction,
        #    skip the LLM classifier entirely to avoid loading another model. ────
        if _SHARED_RESOURCES_AVAILABLE and self.valves.USE_SEMANTIC_CLASSIFY:
            try:
                from shared_resources import get_active_expert

                active_expert = get_active_expert()
                if active_expert and active_expert != "generalista":
                    sem_id = await self._semantic_classify(user_query, self._experts)
                    if sem_id == active_expert:
                        await self._string_cache.set(cache_key, sem_id)
                        if self.valves.DEBUG:
                            logger.info(
                                f"[Router] Sticky routing: keeping '{sem_id}' "
                                f"(already loaded, semantic confirms)"
                            )
                        return sem_id
            except Exception as e:
                if self.valves.DEBUG:
                    logger.info(f"[Router] Sticky routing check error: {e}")
        # ──────────────────────────────────────────────────────────────────────────

        # 1) LLM classifier (most accurate, covers edge cases)
        try:
            expert_id = await self._classify_with_llm(user_query) or ""
            if self.valves.DEBUG and expert_id:
                logger.info(f"[Router] LLM classifier → {expert_id}")
        except Exception as e:
            logger.warning(f"[Router] LLM classifier failed: {e}")
            expert_id = ""

        # 2) Semantic classification (fallback)
        if not expert_id and self.valves.USE_SEMANTIC_CLASSIFY:
            try:
                expert_id = await self._semantic_classify(user_query, self._experts)
                if self.valves.DEBUG and expert_id:
                    logger.info(f"[Router] Semantic classify → {expert_id}")
            except Exception as e:
                if self.valves.DEBUG:
                    logger.info(f"[Router] Semantic classify error: {e}")
                expert_id = ""

        # 3) Keyword fallback (last resort, only very specific terms)
        if not expert_id:
            expert_id = self._keyword_fallback(user_query) or ""
            if self.valves.DEBUG and expert_id:
                logger.info(f"[Router] Keyword fallback → {expert_id}")

        # 4) Default
        if not expert_id:
            expert_id = self.valves.default_model

        await self._string_cache.set(cache_key, expert_id)
        return expert_id

    async def _rewrite_query(self, original: str, expert_id: str) -> str:
        cache_key = f"{original.lower()}|{expert_id}"
        cached = await self._rewrite_cache.get(cache_key)
        if cached is not None:
            if self.valves.DEBUG:
                logger.info(f"[Router] Rewrite cache hit: '{cached}'")
            return cached
        kb_desc = None
        for exp in self._experts:
            if exp["id"] == expert_id:
                kb_desc = exp.get("knowledge_base")
                break
        rewritten = original
        prompt = (
            f"KB:{kb_desc}\nQ:{original}\nShort technical phrase (same language):"
            if kb_desc
            else f"Q:{original}\nShort technical phrase (same language):"
        )
        try:
            rewritten = await self._call_llm(
                prompt=prompt,
                provider=self.valves.classifier_model,
                temperature=0.0,
                max_tokens=25 if kb_desc else 50,
                timeout=10,
            )
            rewritten = rewritten.strip().split("\n")[0].strip('"').strip()
            if not rewritten:
                rewritten = original
        except Exception as e:
            logger.error(f"[Router] Query rewriting failed: {e}")
            rewritten = original
        await self._rewrite_cache.set(cache_key, rewritten)
        if self.valves.DEBUG:
            logger.info(
                f"[Router] Rewritten query: '{original[:60]}...' -> '{rewritten}'"
            )
        return rewritten

    def _inject_rag_guidance(self, expert_id: str, messages: list) -> bool:
        collection = kb_desc = None
        for exp in self._experts:
            if exp["id"] == expert_id:
                collection = exp.get("collection_name")
                kb_desc = exp.get("knowledge_base")
                break
        if not collection:
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] No collection for expert '{expert_id}', skipping RAG guidance."
                )
            return False
        guidance = (
            f"You have access to the knowledge base '{collection}' ({kb_desc or 'specialized documents'}). "
            "Use this knowledge base when relevant, but you may also use other available tools. "
            "Answer in the same language as the user."
        )
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if sys_msg:
            sys_msg["content"] = sys_msg["content"] + "\n\n" + guidance
        else:
            messages.insert(0, {"role": "system", "content": guidance})
        if self.valves.DEBUG:
            logger.info(
                f"[Router] Injected RAG guidance for collection '{collection}'."
            )
        return True

    def _is_tool_request(self, text: str) -> bool:
        tool_keywords = [
            "utiliza la herramienta",
            "usa la herramienta",
            "use the tool",
            "search_and_crawl",
            "ejecuta la herramienta",
            "llama a la herramienta",
            "run the tool",
            "call the tool",
            "utiliza la tool",
            "usa la tool",
            "utiliza la función",
            "usa la función",
            "ejecuta la función",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in tool_keywords)

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ) -> dict:
        __event_call__ = kwargs.get("__event_call__")
        event_sender = __event_emitter__ or __event_call__

        if self.valves.DEBUG:
            logger.info(
                f"[Router] emitter={__event_emitter__ is not None}, event_call={__event_call__ is not None}"
            )

        await self._sync_cache_config()
        self._load_experts()

        messages = body.get("messages", [])
        if not messages:
            return body

        metadata = body.get("metadata", {})
        current_expert = metadata.get("router_expert")
        current_query = messages[-1].get("content", "")

        new_expert = await self._classify_query(current_query, messages)
        new_name = self._get_expert_name(new_expert)

        if not current_expert:
            model_to_use = new_expert
            expert_name = new_name
            change = True
        else:
            if new_expert != current_expert:
                prev_words = set()
                for msg in messages[-3:-1]:
                    if msg.get("role") == "user":
                        prev_words.update(msg.get("content", "").lower().split())
                curr_words = set(current_query.lower().split())
                shared = curr_words.intersection(prev_words)
                if len(shared) < self.valves.change_threshold:
                    model_to_use = new_expert
                    expert_name = new_name
                    change = True
                else:
                    model_to_use = current_expert
                    expert_name = self._get_expert_name(current_expert)
                    change = False
            else:
                model_to_use = current_expert
                expert_name = self._get_expert_name(current_expert)
                change = False

        if self.valves.enable_query_rewriting and messages:
            original_query = messages[-1].get("content", "")
            if original_query and not self._is_tool_request(original_query):
                rewritten = await self._rewrite_query(original_query, model_to_use)
                messages[-1]["content"] = rewritten

        if (
            self.valves.enable_rag_injection
            and model_to_use != self.valves.default_model
        ):
            self._inject_rag_guidance(model_to_use, messages)

        if change:
            metadata["router_expert"] = model_to_use
            body["metadata"] = metadata
            if self.valves.DEBUG:
                logger.info(f"[Router] Switched to expert: {expert_name}")

        # ── Register the active expert in shared_resources for sticky routing ──
        if _SHARED_RESOURCES_AVAILABLE:
            try:
                from shared_resources import set_active_expert

                set_active_expert(model_to_use)
            except Exception:
                pass
        # ────────────────────────────────────────────────────────────────────────

        # Model is kept as originally selected by the user

        if self.valves.notify_change and change:
            notif = self.valves.notification_template.format(expert=expert_name)
            if event_sender is not None:
                try:
                    await event_sender(
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 Router: {notif}",
                                "done": True,
                            },
                        }
                    )
                    if self.valves.DEBUG:
                        logger.info(f"[Router] Notification emitted: {notif}")
                except Exception as e:
                    logger.error(f"[Router] Failed to emit notification: {e}")
            else:
                logger.warning("[Router] No event sender available for notification.")

        return body

    def _get_expert_name(self, expert_id: str) -> str:
        for exp in self._experts:
            if exp["id"] == expert_id:
                return exp["name"]
        return "General"
