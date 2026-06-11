"""
title: Shared Resources for OpenWebUI Plugins
description: Shared singletons (embedder, ChromaDB, HTTP session, tiktoken, LRU cache)
             to avoid resource duplication across plugins.
             Now includes a helper to safely unload all llama.cpp models.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 2.0.0
license: GPL3
requirements: aiohttp, chromadb, tiktoken, sentence-transformers
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. SentenceTransformer singleton (~80 MB VRAM, loaded once)
# ---------------------------------------------------------------------------
_EMBEDDER_INSTANCE = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(model_name: str = "intfloat/multilingual-e5-large"):
    """Return (or create) the SentenceTransformer singleton. Thread-safe."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER_INSTANCE is None:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _EMBEDDER_INSTANCE = SentenceTransformer(model_name)
    return _EMBEDDER_INSTANCE


# ---------------------------------------------------------------------------
# 2. ChromaDB PersistentClient — registry by path
# ---------------------------------------------------------------------------
_CHROMA_CLIENTS: Dict[str, Any] = {}
_CHROMA_LOCK = threading.Lock()


def get_chroma_client(path: str = "./chroma_cache"):
    """Return a PersistentClient for `path`. Each distinct path gets its own client."""
    import os
    norm = os.path.normpath(os.path.abspath(path))
    if norm not in _CHROMA_CLIENTS:
        with _CHROMA_LOCK:
            if norm not in _CHROMA_CLIENTS:
                import chromadb  # type: ignore
                _CHROMA_CLIENTS[norm] = chromadb.PersistentClient(path=norm)
    return _CHROMA_CLIENTS[norm]


# ---------------------------------------------------------------------------
# 3. tiktoken encoding cache
# ---------------------------------------------------------------------------
_TIKTOKEN_ENCODINGS: Dict[str, Any] = {}
_TIKTOKEN_LOCK = threading.Lock()


def get_tiktoken_encoding(model: str = "gpt-4"):
    """Return a cached tiktoken encoding for the given model."""
    if model not in _TIKTOKEN_ENCODINGS:
        with _TIKTOKEN_LOCK:
            if model not in _TIKTOKEN_ENCODINGS:
                import tiktoken  # type: ignore
                try:
                    _TIKTOKEN_ENCODINGS[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    _TIKTOKEN_ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODINGS[model]


# ---------------------------------------------------------------------------
# 4. aiohttp ClientSession with connection pool (shared)
# ---------------------------------------------------------------------------
_HTTP_SESSION: Optional[Any] = None
_HTTP_TIMEOUT_SECONDS: int = 120
_HTTP_SESSION_LOCK: Optional[asyncio.Lock] = None  # Lazy init to avoid errors before event loop


def _get_http_lock() -> asyncio.Lock:
    """Return the HTTP session lock, creating it lazily inside the event loop."""
    global _HTTP_SESSION_LOCK
    if _HTTP_SESSION_LOCK is None:
        _HTTP_SESSION_LOCK = asyncio.Lock()
    return _HTTP_SESSION_LOCK


async def get_http_session(timeout_seconds: int = 120):
    """
    Return (or create) the shared aiohttp ClientSession.
    If the requested timeout is larger than the current session's,
    the session is recreated to avoid cutting off long calls.
    Thread‑safe: guarded by an asyncio lock.
    """
    global _HTTP_SESSION, _HTTP_TIMEOUT_SECONDS
    import aiohttp  # type: ignore

    async with _get_http_lock():
        needs_recreate = (
            _HTTP_SESSION is None
            or _HTTP_SESSION.closed
            or timeout_seconds > _HTTP_TIMEOUT_SECONDS
        )
        if needs_recreate:
            if _HTTP_SESSION is not None and not _HTTP_SESSION.closed:
                await _HTTP_SESSION.close()
            connector = aiohttp.TCPConnector(
                limit=30,
                limit_per_host=10,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            _HTTP_SESSION = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            _HTTP_TIMEOUT_SECONDS = timeout_seconds
    return _HTTP_SESSION


async def close_http_session():
    """Close the shared session. Call on process shutdown if needed."""
    global _HTTP_SESSION
    async with _get_http_lock():
        if _HTTP_SESSION and not _HTTP_SESSION.closed:
            await _HTTP_SESSION.close()
            _HTTP_SESSION = None


# ---------------------------------------------------------------------------
# 5. Shared LLM caller
# ---------------------------------------------------------------------------
async def call_llm(
    prompt: str,
    *,
    system: str = "",
    base_url: str = "http://localhost:8080",
    model: Optional[str] = None,                        # No default model – must be provided
    api_token: str = "",
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
    endpoint_type: str = "chat",
) -> str:
    """
    Async LLM call. Reuses the shared HTTP session.
    Handles Ollama, llama.cpp and OpenAI-compatible endpoints.
    - base_url: may be given with or without trailing /v1 (it is normalized).
    - model: must be provided (e.g. 'llamacpp/llama3.2:3b' or 'gpt-4'). No default.
    - endpoint_type: 'chat' (default) or 'completion' for llama.cpp.
    - max_tokens: if None, no explicit limit is sent (server default used).
    """
    # Fail early if no model is supplied
    if model is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("LLM call requested but no model was provided. Please specify a model.")
        raise RuntimeError("No model provided for LLM call.")

    session = await get_http_session(timeout)

    # Normalise the base URL: strip trailing slash and remove any /v1 suffix
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

    # Force OpenAI-compatible path if model has llamacpp/ prefix
    is_llamacpp = model.startswith("llamacpp/")
    if is_llamacpp:
        is_ollama = False

    # Extract the real model name (everything after the first /, if present)
    model_str = model.split("/", 1)[1] if "/" in model else model

    headers = {"Content-Type": "application/json"}
    if api_token and api_token.strip():
        headers["Authorization"] = f"Bearer {api_token.strip()}"

    if is_ollama:
        url = f"{base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model_str,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
    else:
        if endpoint_type == "completion":
            url = f"{base_url}/v1/completions"
            payload: Dict[str, Any] = {
                "model": model_str,
                "prompt": prompt if not system else f"{system}\n\n{prompt}",
                "temperature": temperature,
            }
        else:  # default to chat completions
            url = f"{base_url}/v1/chat/completions"
            payload: Dict[str, Any] = {
                "model": model_str,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

    import aiohttp  # type: ignore

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM HTTP {resp.status}: {text[:300]}")
            data = await resp.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"LLM connection error: {exc}") from exc

    if is_ollama:
        content = data.get("response", "")
        if not content:
            err = data.get("error", "")
            if err:
                raise RuntimeError(f"Ollama model error: {err}")
    else:
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI response has no choices")
        if endpoint_type == "completion":
            content = choices[0].get("text", "")
        else:
            content = choices[0].get("message", {}).get("content", "")

            # ── Fallback for models that use thinking/chain-of-thought ──
            if not content:
                reasoning = choices[0].get("message", {}).get("reasoning_content", "")
                if reasoning:
                    content = reasoning.strip()
            # ─────────────────────────────────────────────────────────────

    # Diagnostic log – remove after fixing the empty content issue
    if not content:
        import logging
        log = logging.getLogger(__name__)
        log.warning("LLM response with empty content. Full response: %s", data)

    return content.strip()


# ---------------------------------------------------------------------------
# 6. Async-safe LRU Cache with TTL
# ---------------------------------------------------------------------------
class AsyncLRUCache:
    """Async-safe cache with LRU eviction and TTL."""

    def __init__(self, max_size: int = 1000, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if self.ttl > 0 and time.time() - ts > self.ttl:
                del self._store[key]
                return None
            # Move to end (LRU)
            self._store[key] = self._store.pop(key)
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if (
                self.max_size > 0
                and len(self._store) >= self.max_size
                and key not in self._store
            ):
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[key] = (value, time.time())
            self._store[key] = self._store.pop(key)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# 7. Persistent SQLite cache (L1 RAM + L2 SQLite)
# ---------------------------------------------------------------------------
class SQLiteCache:
    """
    Two-level cache: fast RAM (LRU) + persistent SQLite.
    Used for expert classifications and rewrite cache.
    """

    def __init__(self, db_path: str = "/app/backend/data/router_cache.db",
                 table: str = "cache", max_size: int = 500, ttl: int = 1800):
        self.table = table
        self.ttl = ttl
        self.max_size = max_size
        self._ram = AsyncLRUCache(max_size=max_size, ttl=ttl)
        self._db_path = db_path
        self._conn: Optional[Any] = None
        self._conn_lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize the database. Creates the persistent connection via _get_conn()."""
        self._get_conn()

    def _get_conn(self):
        """Return the persistent SQLite connection, creating it if needed. Thread‑safe."""
        import sqlite3, os
        with self._conn_lock:
            if self._conn is None:
                parent = os.path.dirname(os.path.abspath(self._db_path))
                os.makedirs(parent, exist_ok=True)
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                """)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA cache_size=-8000")
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ts ON {self.table}(ts)"
                )
                self._conn.commit()
            return self._conn

    async def get(self, key: str) -> Optional[str]:
        # L1 RAM
        hit = await self._ram.get(key)
        if hit is not None:
            return hit
        # L2 SQLite
        def _read():
            conn = self._get_conn()
            return conn.execute(
                f"SELECT value, ts FROM {self.table} WHERE key = ?", (key,)
            ).fetchone()

        import anyio
        row = await anyio.to_thread.run_sync(_read)
        if row is None:
            return None
        value, ts = row
        if self.ttl > 0 and time.time() - ts > self.ttl:
            await self._delete_from_db(key)
            return None
        await self._ram.set(key, value)
        return value

    async def set(self, key: str, value: str) -> None:
        await self._ram.set(key, value)
        def _write():
            conn = self._get_conn()
            conn.execute(
                f"REPLACE INTO {self.table} (key, value, ts) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            conn.commit()
        import anyio
        await anyio.to_thread.run_sync(_write)

    async def _delete_from_db(self, key: str) -> None:
        def _del():
            conn = self._get_conn()
            conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
            conn.commit()
        import anyio
        await anyio.to_thread.run_sync(_del)

    async def clear(self) -> None:
        await self._ram.clear()
        def _clr():
            conn = self._get_conn()
            conn.execute(f"DELETE FROM {self.table}")
            conn.commit()
        import anyio
        await anyio.to_thread.run_sync(_clr)


# ---------------------------------------------------------------------------
# 8. Active expert tracker (used for sticky routing)
# ---------------------------------------------------------------------------
_ACTIVE_EXPERT: Optional[str] = None
_ACTIVE_EXPERT_LOCK = threading.Lock()


def get_active_expert() -> Optional[str]:
    """
    Return the expert_id currently selected and loaded in the main conversation.
    This is set by the Router after choosing an expert.
    """
    return _ACTIVE_EXPERT


def set_active_expert(expert_id: str) -> None:
    """
    Update the active expert_id.
    Called by the Router after a routing decision, so that other plugins
    or subsequent requests can skip re-classification when the expert hasn't changed.
    """
    global _ACTIVE_EXPERT
    with _ACTIVE_EXPERT_LOCK:
        _ACTIVE_EXPERT = expert_id


# ---------------------------------------------------------------------------
# 9. Helper to safely unload all models from a llama.cpp server
#    (clears the single slot before switching to a different model)
# ---------------------------------------------------------------------------
async def unload_all_models(base_url: str) -> None:
    """
    List all currently loaded models on a llama.cpp server and unload each one.
    After completion the server slot should be empty.

    This function is designed to be called before switching to a different model
    on a server with a single slot, preventing "model failed to load" errors.
    """
    import aiohttp

    base_url = base_url.rstrip("/")
    # Remove /v1 if present – the unload endpoint is at the root
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    async with aiohttp.ClientSession() as sess:
        # 1. List models
        try:
            async with sess.get(
                f"{base_url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception:
            return

        models = []
        for m in data.get("data", []):
            # Only consider models that are currently loaded
            if m.get("status", {}).get("value") == "loaded":
                models.append(m["id"])

        # 2. Unload each loaded model
        for model_id in models:
            try:
                async with sess.post(
                    f"{base_url}/models/unload",
                    json={"model": model_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        import logging
                        logging.getLogger(__name__).debug(f"Unloaded model {model_id}")
            except Exception:
                pass

        # 3. Wait a moment for slots to be released
        await asyncio.sleep(0.5)
