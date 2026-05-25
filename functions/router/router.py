"""
title: Router
description: Smart router that selects the best expert model for each query based on keywords, LLM classification, and semantic similarity. Rewrites queries for better RAG retrieval and injects RAG guidance.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 1.0.0
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
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Path to shared library
import sys
if "/app/backend/data/custom_lib" not in sys.path:
    sys.path.append("/app/backend/data/custom_lib")

# Shared resources (persistent cache & shared LLM caller)
try:
    from shared_resources import SQLiteCache, AsyncLRUCache  # type: ignore
    _SHARED_RESOURCES_AVAILABLE = True
except ImportError:
    _SHARED_RESOURCES_AVAILABLE = False


def _build_cache(table: str, max_size: int, ttl: int, db_path: str):
    """
    Build cache: L1 RAM (LRU) + L2 SQLite (persistent) when shared_resources is available.
    Falls back to L1-only cache otherwise.
    """
    if _SHARED_RESOURCES_AVAILABLE:
        return SQLiteCache(db_path=db_path, table=table, max_size=max_size, ttl=ttl)

    # Minimal fallback with same async interface
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
        "keywords": ["neovim", "nvim", "vim", "lsp", "treesitter", "buffer", "init.lua", "keymap"],
        "description": "Configuration, plugins, and usage of Neovim editor",
        "examples": ["how to configure LSP in nvim", "best neovim plugins 2025", "treesitter setup init.lua"],
        "knowledge_base": "the official Neovim documentation and community plugins",
        "collection_name": "neovim_docs"
    },
    {
        "id": "experto-en-arch-linux",
        "name": "Arch Linux",
        "keywords": ["arch", "archlinux", "pacman", "yay", "aur", "systemd", "grub", "hyprland", "wayland"],
        "description": "Installation, maintenance, and troubleshooting of Arch Linux",
        "examples": ["how to install yay", "hyprland config", "pacman update error"],
        "knowledge_base": "the official Arch Linux wiki",
        "collection_name": "archwiki"
    }
]""",
            description="JSON with expert definitions. Each must have 'id', 'name', 'keywords' (list). Optional: 'description', 'examples', 'knowledge_base', 'collection_name'.",
        )
        default_model: str = Field(default="generalista")
        change_threshold: int = Field(default=2)
        notify_change: bool = Field(default=True)
        notification_template: str = Field(default="🎯 Using expert: {expert}")
        LLM_BASE_URL: str = Field(default="http://host.docker.internal:11434/")
        LLM_API_TOKEN: str = Field(default="")
        classifier_model: str = Field(default="ollama/llama3.2:3b")
        classifier_temperature: float = Field(default=0.0)
        classifier_timeout: int = Field(default=10)
        default_llm_temperature: float = Field(default=0.3)
        default_llm_max_tokens: int = Field(default=256)
        default_llm_timeout: int = Field(default=30)
        enable_query_rewriting: bool = Field(default=True)
        enable_rag_injection: bool = Field(default=True)
        rewrite_cache_max_size: int = Field(default=500)
        string_cache_max_size: int = Field(default=1000)
        string_cache_ttl: int = Field(default=1800)
        rewrite_cache_ttl: int = Field(default=3600)
        USE_SEMANTIC_CLASSIFY: bool = Field(
            default=True,
            description="Use embedding similarity before LLM for classification (faster)",
        )
        SEMANTIC_THRESHOLD: float = Field(
            default=0.55,
            description="Minimum cosine similarity to classify without LLM (0–1)",
        )
        CACHE_DB_PATH: str = Field(
            default="/app/backend/data/router_cache.db",
            description="SQLite path for persistent classification and rewrite cache",
        )
        DEBUG: bool = Field(default=True)

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
        db_path = self.valves.CACHE_DB_PATH
        if (
            self._string_cache.max_size != self.valves.string_cache_max_size
            or self._string_cache.ttl != self.valves.string_cache_ttl
        ):
            self._string_cache = _build_cache(
                "classifications",
                self.valves.string_cache_max_size,
                self.valves.string_cache_ttl,
                db_path,
            )
        if (
            self._rewrite_cache.max_size != self.valves.rewrite_cache_max_size
            or self._rewrite_cache.ttl != self.valves.rewrite_cache_ttl
        ):
            self._rewrite_cache = _build_cache(
                "rewrites",
                self.valves.rewrite_cache_max_size,
                self.valves.rewrite_cache_ttl,
                db_path,
            )

    # --- LLM call using shared caller or aiohttp fallback ---
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
                )
            except Exception as e:
                logger.warning(f"[Router] shared call_llm failed: {e}, using fallback")

        # Fallback: aiohttp without shared pool
        import aiohttp

        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        api_token = (
            self.valves.LLM_API_TOKEN.strip() if self.valves.LLM_API_TOKEN else None
        )
        model_name = model_str.split("/", 1)[1] if "/" in model_str else model_str
        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

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
            url = f"{base_url}/chat/completions"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        async with aiohttp.ClientSession() as session:
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

        if is_ollama:
            content = data.get("response", "")
            if "error" in data:
                logger.error(f"[Router] Ollama error: {data['error']}")
                return ""
        else:
            content = data["choices"][0]["message"]["content"]

        content = content.strip()
        if not content:
            logger.warning(f"[Router] LLM returned empty content for '{model_name}'.")
        if self.valves.DEBUG:
            logger.debug(f"[Router] LLM raw response: {content[:300]}")
        return content

    # --- Original classification methods (unchanged) ---
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

    # --- Contextual cache key ---
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

    # --- Semantic classification ---
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

        best_id = ""
        best_score = -1.0
        for expert_id, example_vecs in self._expert_embeddings.items():
            norms = np.linalg.norm(example_vecs, axis=1, keepdims=True) + 1e-10
            scores = (example_vecs / norms) @ q_norm
            score = float(scores.max())
            if score > best_score:
                best_score = score
                best_id = expert_id

        if best_score >= threshold:
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] Semantic classify: '{user_query[:50]}' → {best_id} (sim={best_score:.3f})"
                )
            return best_id
        return ""

    # --- Main classification pipeline ---
    async def _classify_query(self, user_query: str, messages: list = None) -> str:
        cache_key = self._get_cache_key(user_query, messages)

        cached = await self._string_cache.get(cache_key)
        if cached is not None:
            if self.valves.DEBUG:
                logger.info(f"[Router] Cache hit: '{user_query[:50]}' → {cached}")
            return cached

        expert_id = ""

        # Semantic classification
        if self.valves.USE_SEMANTIC_CLASSIFY:
            try:
                expert_id = await self._semantic_classify(user_query, self._experts)
            except Exception as e:
                if self.valves.DEBUG:
                    logger.info(f"[Router] Semantic classify error: {e}")
                expert_id = ""

        # LLM classifier
        if not expert_id:
            expert_id = await self._classify_with_llm(user_query) or ""

        # Keyword fallback
        if not expert_id:
            expert_id = self._keyword_fallback(user_query) or ""

        # Default
        if not expert_id:
            expert_id = self.valves.default_model

        await self._string_cache.set(cache_key, expert_id)
        return expert_id

    # --- Query rewriting (unchanged) ---
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
        if kb_desc:
            prompt = (
                f"KB:{kb_desc}\nQ:{original}\nShort technical phrase (same language):"
            )
        else:
            prompt = f"Q:{original}\nShort technical phrase (same language):"
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

    # --- RAG injection (unchanged) ---
    def _inject_rag_guidance(self, expert_id: str, messages: list) -> bool:
        collection = None
        kb_desc = None
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

    # --- Helper to detect tool requests ---
    def _is_tool_request(self, text: str) -> bool:
        """Return True if the user message explicitly asks to use a tool."""
        tool_keywords = [
            "utiliza la herramienta", "usa la herramienta", "use the tool",
            "search_and_crawl", "ejecuta la herramienta", "llama a la herramienta",
            "run the tool", "call the tool", "utiliza la tool", "usa la tool",
            "utiliza la función", "usa la función", "ejecuta la función",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in tool_keywords)

    # --- Inlet (now passes messages to classification) ---
    async def inlet(self, body: dict, **kwargs) -> dict:
        user = kwargs.get("user", {})
        __event_emitter__ = kwargs.get("__event_emitter__")

        await self._sync_cache_config()
        self._load_experts()

        messages = body.get("messages", [])
        if not messages:
            return body

        metadata = body.get("metadata", {})
        current_expert = metadata.get("router_expert")
        current_query = messages[-1].get("content", "")

        # Pass the full message history for contextual cache key
        new_expert = await self._classify_query(current_query, messages)
        new_name = self._get_expert_name(new_expert)

        # Decide whether to switch expert (with change threshold)
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

        # Optional query rewriting (skip if user is asking to use a tool)
        if self.valves.enable_query_rewriting and messages:
            original_query = messages[-1].get("content", "")
            if original_query and not self._is_tool_request(original_query):
                rewritten = await self._rewrite_query(original_query, model_to_use)
                messages[-1]["content"] = rewritten

        # Optional RAG injection
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

        body["model"] = model_to_use

        if __event_emitter__ and self.valves.notify_change and change:
            notif = self.valves.notification_template.format(expert=expert_name)
            await __event_emitter__(
                {"type": "notification", "data": {"type": "info", "content": notif}}
            )
        return body

    def _get_expert_name(self, expert_id: str) -> str:
        for exp in self._experts:
            if exp["id"] == expert_id:
                return exp["name"]
        return "General"
