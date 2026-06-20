"""
title: Shared Resources for OpenWebUI Plugins
description: Shared singletons (embedder, ChromaDB, HTTP session, tiktoken, LRU cache)
             to avoid resource duplication across plugins.
             Now includes a helper to safely unload all llama.cpp models,
             plus ConversationCompressor for v8 history compression.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 2.1.0
license: GPL3
requirements: aiohttp, chromadb, tiktoken, sentence-transformers, llmlingua>=0.2.0
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

# ============================================================================
# SECTION 1: Embedding & Vector DB (sentence-transformers + ChromaDB)
# ============================================================================

# ---------------------------------------------------------------------------
# 1.1 SentenceTransformer singleton (~80 MB VRAM, loaded once)
# ---------------------------------------------------------------------------
_EMBEDDER_INSTANCE = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    """
    Return the SentenceTransformer embedder singleton, loading it once.

    The embedder is used for semantic search, LTM retrieval, and RAPTOR
    clustering. It is loaded on demand and cached globally across all
    plugins that use this shared resource module.

    Args:
        model_name (str): The HuggingFace model name to load.
                         Defaults to "Qwen/Qwen3-Embedding-0.6B".

    Returns:
        SentenceTransformer: The loaded embedder instance.
    """
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER_INSTANCE is None:
                from sentence_transformers import SentenceTransformer
                # we now force CPU for the embedder
                _EMBEDDER_INSTANCE = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER_INSTANCE


# ---------------------------------------------------------------------------
# 1.2 ChromaDB PersistentClient — registry by path
# ---------------------------------------------------------------------------
_CHROMA_CLIENTS: Dict[str, Any] = {}
_CHROMA_LOCK = threading.Lock()


def get_chroma_client(path: str = "./chroma_cache"):
    """
    Return a PersistentClient for `path`. Each distinct path gets its own client.

    ChromaDB clients are expensive to create (they initialise the underlying
    database and embedding index). This function caches clients by their
    filesystem path, reusing them across calls.

    Args:
        path (str): The directory path where ChromaDB stores its data.
                   Defaults to "./chroma_cache".

    Returns:
        chromadb.PersistentClient: The ChromaDB client for the given path.
    """
    import os

    norm = os.path.normpath(os.path.abspath(path))
    if norm not in _CHROMA_CLIENTS:
        with _CHROMA_LOCK:
            if norm not in _CHROMA_CLIENTS:
                import chromadb

                _CHROMA_CLIENTS[norm] = chromadb.PersistentClient(path=norm)
    return _CHROMA_CLIENTS[norm]


# ============================================================================
# SECTION 2: Tokenization & Caching (tiktoken + LRU + SQLite)
# ============================================================================

# ---------------------------------------------------------------------------
# 2.1 tiktoken encoding cache
# ---------------------------------------------------------------------------
_TIKTOKEN_ENCODINGS: Dict[str, Any] = {}
_TIKTOKEN_LOCK = threading.Lock()


def get_tiktoken_encoding(model: str = "gpt-4"):
    """
    Return a cached tiktoken encoding for the given model.

    tiktoken encodings are model-specific and loading them repeatedly is
    expensive. This function caches them globally so that multiple plugins
    can share the same encoding instance.

    Args:
        model (str): The model name to get the encoding for.
                    Defaults to "gpt-4". If the model is not recognised,
                    falls back to "cl100k_base".

    Returns:
        tiktoken.Encoding: The tiktoken encoding for the specified model.
    """
    if model not in _TIKTOKEN_ENCODINGS:
        with _TIKTOKEN_LOCK:
            if model not in _TIKTOKEN_ENCODINGS:
                import tiktoken

                try:
                    _TIKTOKEN_ENCODINGS[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    _TIKTOKEN_ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODINGS[model]


# ---------------------------------------------------------------------------
# 2.2 Async-safe LRU Cache with TTL
# ---------------------------------------------------------------------------
class AsyncLRUCache:
    """Async-safe cache with LRU eviction and TTL.

    Provides thread-safe (asyncio-safe) in-memory caching with a maximum
    size limit and time-to-live expiration. LRU eviction ensures that
    the most recently accessed items are kept when the cache reaches its
    maximum size.

    Used for LLM response caching to avoid redundant inference calls.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 1800):
        """
        Initialize the async LRU cache.

        Args:
            max_size (int): Maximum number of items to store. 0 = unlimited.
            ttl (int): Time-to-live in seconds. 0 = never expire.
        """
        self.max_size = max_size
        self.ttl = ttl
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from the cache.

        If the item exists and has not expired, it is moved to the front of
        the LRU list and returned. Expired items are deleted.

        Args:
            key (str): The cache key.

        Returns:
            Optional[Any]: The cached value, or None if not found or expired.
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if self.ttl > 0 and time.time() - ts > self.ttl:
                del self._store[key]
                return None
            self._store[key] = self._store.pop(key)
            return value

    async def set(self, key: str, value: Any) -> None:
        """
        Store an item in the cache.

        If the cache is at its maximum size, the least recently used item is
        evicted. The new item is inserted and moved to the front.

        Args:
            key (str): The cache key.
            value (Any): The value to cache.
        """
        async with self._lock:
            if self.max_size > 0 and len(self._store) >= self.max_size and key not in self._store:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[key] = (value, time.time())
            self._store[key] = self._store.pop(key)

    async def clear(self) -> None:
        """Clear all items from the cache."""
        async with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        """Return the number of items currently in the cache."""
        return len(self._store)


# ---------------------------------------------------------------------------
# 2.3 Persistent SQLite cache (L1 RAM + L2 SQLite)
# ---------------------------------------------------------------------------
class SQLiteCache:
    """Two-level cache: fast RAM (LRU) + persistent SQLite.

    Provides a hybrid caching layer where the fastest lookups are served
    from RAM (LRU) and persistent storage is kept in SQLite. Items are
    automatically expired based on TTL.

    Used for router caches and other persistent key-value stores where
    data needs to survive process restarts.
    """

    def __init__(self, db_path: str = "/app/backend/data/router_cache.db",
                 table: str = "cache", max_size: int = 500, ttl: int = 1800):
        """
        Initialize the SQLite cache.

        Args:
            db_path (str): Path to the SQLite database file.
            table (str): Table name to store cache entries.
            max_size (int): Maximum RAM cache size (LRU eviction).
            ttl (int): Time-to-live in seconds for all entries.
        """
        self.table = table
        self.ttl = ttl
        self.max_size = max_size
        self._ram = AsyncLRUCache(max_size=max_size, ttl=ttl)
        self._db_path = db_path
        self._conn: Optional[Any] = None
        self._conn_lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialise the SQLite database and create the cache table if needed."""
        self._get_conn()

    def _get_conn(self):
        """Return the SQLite connection, creating it if necessary."""
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
                self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ts ON {self.table}(ts)")
                self._conn.commit()
            return self._conn

    async def get(self, key: str) -> Optional[str]:
        """
        Retrieve a value from the cache.

        First checks RAM cache, then falls back to SQLite. If found in
        SQLite and not expired, the value is promoted to RAM cache.

        Args:
            key (str): The cache key.

        Returns:
            Optional[str]: The cached value, or None if not found.
        """
        hit = await self._ram.get(key)
        if hit is not None:
            return hit
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
        """
        Store a value in the cache.

        Writes to both RAM and SQLite. If the RAM cache is at capacity,
        the least recently used item is evicted first.

        Args:
            key (str): The cache key.
            value (str): The value to store.
        """
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
        """Delete a single key from the SQLite database."""
        def _del():
            conn = self._get_conn()
            conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
            conn.commit()
        import anyio
        await anyio.to_thread.run_sync(_del)

    async def clear(self) -> None:
        """Clear all entries from both RAM and SQLite."""
        await self._ram.clear()
        def _clr():
            conn = self._get_conn()
            conn.execute(f"DELETE FROM {self.table}")
            conn.commit()
        import anyio
        await anyio.to_thread.run_sync(_clr)


# ============================================================================
# SECTION 3: HTTP & LLM (aiohttp session + LLM caller)
# ============================================================================

# ---------------------------------------------------------------------------
# 3.1 aiohttp ClientSession with connection pool (shared)
# ---------------------------------------------------------------------------
_HTTP_SESSION: Optional[Any] = None
_HTTP_TIMEOUT_SECONDS: int = 120
_HTTP_SESSION_LOCK: Optional[asyncio.Lock] = None


def _get_http_lock() -> asyncio.Lock:
    """Return the HTTP session lock, creating it lazily inside the event loop."""
    global _HTTP_SESSION_LOCK
    if _HTTP_SESSION_LOCK is None:
        _HTTP_SESSION_LOCK = asyncio.Lock()
    return _HTTP_SESSION_LOCK


async def get_http_session(timeout_seconds: int = 120):
    """
    Return (or create) the shared aiohttp ClientSession.

    If the requested timeout is larger than the current session's timeout,
    the session is recreated to avoid cutting off long calls. The session
    uses a connection pool with connection limits and keepalive.

    Args:
        timeout_seconds (int): Total timeout for requests in seconds.
                              Defaults to 120.

    Returns:
        aiohttp.ClientSession: The shared HTTP session.
    """
    global _HTTP_SESSION, _HTTP_TIMEOUT_SECONDS
    import aiohttp

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
# 3.2 Shared LLM caller
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    *,
    system: str = "",
    base_url: str = "http://localhost:8080",
    model: Optional[str] = None,
    api_token: str = "",
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
    endpoint_type: str = "chat",
) -> str:
    """
    Async LLM call. Reuses the shared HTTP session.

    Handles Ollama, llama.cpp and OpenAI-compatible endpoints. For Ollama,
    uses the `/api/generate` endpoint. For OpenAI-compatible endpoints,
    supports both `/v1/chat/completions` and `/v1/completions`.

    Args:
        prompt (str): The user prompt.
        system (str): System prompt for chat completions.
        base_url (str): The base URL of the LLM server.
        model (Optional[str]): The model to use. Required.
        api_token (str): API token for authenticated endpoints.
        temperature (float): Sampling temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Maximum tokens to generate.
        timeout (int): Request timeout in seconds. Defaults to 120.
        endpoint_type (str): 'chat' or 'completion'. Defaults to 'chat'.

    Returns:
        str: The LLM response content.

    Raises:
        RuntimeError: If the model is not provided, the HTTP request fails,
                      or the response is malformed.
    """
    if model is None:
        _logger.error("LLM call requested but no model was provided. Please specify a model.")
        raise RuntimeError("No model provided for LLM call.")

    session = await get_http_session(timeout)

    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    is_ollama = "ollama" in base_url.lower() or ":11434" in base_url
    is_llamacpp = model.startswith("llamacpp/")
    if is_llamacpp:
        is_ollama = False

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
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
    else:
        if endpoint_type == "completion":
            url = f"{base_url}/v1/completions"
            payload = {
                "model": model_str,
                "prompt": prompt if not system else f"{system}\n\n{prompt}",
                "temperature": temperature,
            }
        else:
            url = f"{base_url}/v1/chat/completions"
            payload = {
                "model": model_str,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

    import aiohttp

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
            if not content:
                reasoning = choices[0].get("message", {}).get("reasoning_content", "")
                if reasoning:
                    content = reasoning.strip()

    if not content:
        _logger.warning("LLM response with empty content. Full response: %s", data)

    return content.strip()


# ============================================================================
# SECTION 4: Conversation Compression (LLMLingua-2 for chat history)
# ============================================================================

# ---------------------------------------------------------------------------
# 4.1 ConversationCompressor — tiered LLMLingua-2 compression of chat history (v8)
# ---------------------------------------------------------------------------
class ConversationCompressor:
    """
    Tiered LLMLingua-2 compression of conversation history.

    Each turn is assigned a compression tier based on its age and whether its
    code is already indexed in the SymbolGraph:

        current      → never compressed (rate 1.0)
        recent       → light compression (recent_rate, e.g. 0.75)
        old          → aggressive compression (old_rate, e.g. 0.40)
        old_indexed  → very aggressive (indexed_rate, e.g. 0.20) — the code
                       bodies are recoverable via LOD, so prose can be stripped
                       hard and code blocks replaced with one-line stubs.

    Turns that contain a CodeAware protocol marker, or that look like an error
    / traceback (when preserve_errors=True), are kept verbatim regardless of
    tier. Original messages are never mutated; a new list is returned.
    """

    _PROTOCOL_MARKERS: frozenset = frozenset({
        "▶ CONTINÚA:",
        "▶ CONTINÚA EN LA SIGUIENTE PARTE",
        "🗜️ PARTE",
        "## Código — Parte",
        "## Código - Parte",
        "[🗜️ PARTE",
        "[CÓDIGO COMPRIMIDO",
    })

    _CODE_FENCE = re.compile(
        r"```(?P<lang>\w*)\s*\n(?P<body>.*?)```|```(?P<lang2>\w*)\s*\n(?P<body2>.*?)\Z",
        re.DOTALL,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.1.1 Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, compressor_singleton: Any) -> None:
        """
        Initialise the conversation compressor with an LLMLingua-2 instance.

        Args:
            compressor_singleton (Any): An instance of PromptCompressor
                                        from the llmlingua library.
        """
        self.raw = compressor_singleton

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.1.2 Main public method
    # ═══════════════════════════════════════════════════════════════════════════

    async def compress_messages(
        self,
        messages: List[dict],
        project_id: str,
        symbol_index: Any,
        current_msg_idx: int,
        recent_lookback: int = 4,
        old_rate: float = 0.40,
        recent_rate: float = 0.75,
        indexed_rate: float = 0.20,
        min_tokens_to_compress: int = 300,
        preserve_errors: bool = True,
        query: str = "",
    ) -> List[dict]:
        """
        Compress a list of conversation messages using tiered compression.

        Each turn is classified into one of four tiers based on recency and
        whether its code symbols are already indexed. Compression rates are
        applied accordingly.

        Args:
            messages (List[dict]): List of message dicts with 'role' and 'content'.
            project_id (str): The project ID for symbol index lookups.
            symbol_index (Any): The SymbolIndex instance for checking indexed code.
            current_msg_idx (int): Index of the current message in the conversation.
            recent_lookback (int): Number of recent turns to keep lightly compressed.
            old_rate (float): Compression rate for old turns (0.0-1.0).
            recent_rate (float): Compression rate for recent turns (0.0-1.0).
            indexed_rate (float): Compression rate for old indexed turns (0.0-1.0).
            min_tokens_to_compress (int): Minimum tokens before compression is attempted.
            preserve_errors (bool): Whether to preserve error/traceback messages verbatim.
            query (str): The user query for question-aware compression.

        Returns:
            List[dict]: A new list of messages with compressed content where applicable.
        """
        if self.raw is None or not messages:
            return messages

        conv_pos = 0
        pos_map = {}
        for i, m in enumerate(messages):
            if m.get("role") in ("user", "assistant"):
                pos_map[i] = conv_pos
                conv_pos += 1

        out = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role not in ("user", "assistant") or not content:
                out.append(msg)
                continue
            if self._has_protocol_marker(content) or self._should_preserve(content, preserve_errors):
                out.append(msg)
                continue
            if idx not in pos_map:
                out.append(msg)
                continue
            pos = pos_map[idx]
            last_conv_pos = conv_pos - 1
            tier = self._classify_turn(
                conv_pos=pos,
                last_conv_pos=last_conv_pos,
                recent_lookback=recent_lookback,
                content=content,
                symbol_index=symbol_index,
                project_id=project_id,
            )
            if tier == "current":
                out.append(msg)
                continue
            if self._estimate_tokens(content) < min_tokens_to_compress:
                out.append(msg)
                continue
            rate = {"recent": recent_rate, "old": old_rate, "old_indexed": indexed_rate}[tier]
            try:
                if tier == "old_indexed":
                    new_content = await self._compress_indexed_turn(content, rate, query, symbol_index, project_id)
                else:
                    new_content = await self._compress_turn(content, rate, query)
            except Exception:
                new_content = content
            out.append({**msg, "content": new_content})
        return out

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.1.3 Classification helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _classify_turn(self, conv_pos, last_conv_pos, recent_lookback, content, symbol_index, project_id):
        """
        Classify a conversation turn into a compression tier.

        Returns:
            str: 'current', 'recent', 'old', or 'old_indexed'
        """
        if conv_pos >= last_conv_pos:
            return "current"
        distance = last_conv_pos - conv_pos
        if distance <= recent_lookback:
            return "recent"
        if self._is_code_indexed(content, symbol_index, project_id):
            return "old_indexed"
        return "old"

    def _is_code_indexed(self, content, symbol_index, project_id):
        """
        Check if all code symbols in the content are indexed in the SymbolGraph.

        Returns True if all defined classes and functions in code fences are
        present in the symbol index. Returns False if any are missing or if
        the index is unavailable.
        """
        if symbol_index is None:
            return False
        defined = set()
        for m in self._CODE_FENCE.finditer(content):
            body = (m.group("body") or m.group("body2") or "")
            defined.update(re.findall(r"^class\s+([A-Za-z_]\w*)", body, re.MULTILINE))
            defined.update(re.findall(r"^(?:async )?def\s+([A-Za-z_]\w*)", body, re.MULTILINE))
        if not defined:
            return False
        try:
            indexed = symbol_index.get_all_names(project_id)
        except Exception:
            return False
        return defined.issubset(indexed)

    @classmethod
    def _has_protocol_marker(cls, text):
        """Check if the text contains a CodeAware protocol marker."""
        norm = text.replace("▶️", "▶").replace("►", "▶")
        return any(m in norm for m in cls._PROTOCOL_MARKERS)

    @staticmethod
    def _should_preserve(content, preserve_errors):
        """Check if a message should be preserved verbatim (e.g., tracebacks)."""
        if not preserve_errors:
            return False
        cl = content.lower()
        return (
            "traceback (most recent call last)" in cl
            or "traceback:" in cl
            or ("exception" in cl and 'file "' in cl)
            or bool(re.search(r'file ".+", line \d+', cl))
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.1.4 Compression helpers
    # ═══════════════════════════════════════════════════════════════════════════

    async def _compress_turn(self, content, rate, query):
        """
        Compress a turn that is not fully indexed (recent or old tier).

        Prose is compressed with LLMLingua. Code blocks are kept verbatim
        because they contain structural information that would be corrupted
        by language model compression.
        """
        segments = self._split_segments(content)
        rebuilt = []
        for kind, text, lang in segments:
            if not text.strip():
                rebuilt.append(text)
                continue
            if kind == "text":
                rebuilt.append(await self._llmlingua(text, rate, query))
            else:
                # NEVER run LLMLingua over code. It deletes "low-information"
                # tokens and eats identifiers (SubgraphExtractor -> "Subgr",
                # BaseModel -> "BaseM") — precisely the information a code query
                # needs intact. Large code blocks are already turned into clean
                # stubs upstream (lean_user_code, _compress_indexed_turn); any
                # code that reaches this "recent"/"old-unindexed" tier is kept
                # verbatim, since for those tiers compression has no safe
                # recovery path and corrupting the code is worse than keeping it.
                rebuilt.append(f"```{lang}\n{text}\n```")
        return "".join(rebuilt)

    async def _compress_indexed_turn(self, content, rate, query, symbol_index, project_id):
        """
        Compress a turn where all code symbols are already indexed.

        Prose is compressed with LLMLingua. Code blocks are replaced with
        stubs listing the defined symbols, since the full bodies are
        recoverable via LOD or `/expand`.
        """
        segments = self._split_segments(content)
        rebuilt = []
        for kind, text, lang in segments:
            if not text.strip():
                rebuilt.append(text)
                continue
            if kind == "text":
                rebuilt.append(await self._llmlingua(text, rate, query))
            else:
                tok = self._estimate_tokens(text)
                names = self._extract_defined_names(text)
                name_str = ", ".join(names[:8]) if names else "none"
                rebuilt.append(
                    f"```{lang}\n"
                    f"[CÓDIGO COMPRIMIDO — {tok} tokens — símbolos: {name_str}. "
                    f"Recuperar con /expand <nombre> o via LOD.]\n"
                    f"```"
                )
        return "".join(rebuilt)

    async def _llmlingua(self, text, rate, query, code_mode=False):
        """
        Apply LLMLingua-2 compression to a text segment.

        Args:
            text (str): The text to compress.
            rate (float): The compression rate (0.0-1.0).
            query (str): The user query for question-aware compression.
            code_mode (bool): Whether to treat the text as code.

        Returns:
            str: The compressed text.
        """
        import anyio
        kwargs = {"rate": rate}
        if code_mode:
            kwargs["force_tokens"] = ["\n", ":", "def ", "class ", "return ", "import "]
            kwargs["force_reserve_digit"] = True
        if query and query.strip():
            kwargs["question"] = query[:300]
        try:
            result = await anyio.to_thread.run_sync(lambda: self.raw.compress_prompt(text, **kwargs))
            return result.get("compressed_prompt", text)
        except Exception:
            return text

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.1.5 Segment splitting & extraction utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def _split_segments(self, content):
        """
        Split content into text and code segments.

        Returns:
            List[Tuple[str, str, str]]: List of (kind, text, language) where
                                        kind is 'text' or 'code'.
        """
        segments = []
        last = 0
        for m in self._CODE_FENCE.finditer(content):
            if m.start() > last:
                segments.append(("text", content[last:m.start()], ""))
            lang = m.group("lang") or m.group("lang2") or ""
            body = m.group("body") or m.group("body2") or ""
            segments.append(("code", body, lang))
            last = m.end()
        if last < len(content):
            segments.append(("text", content[last:], ""))
        return segments

    @staticmethod
    def _extract_defined_names(code):
        """Extract class and function names defined in a code block."""
        names = re.findall(r"^class\s+([A-Za-z_]\w*)", code, re.MULTILINE)
        names += re.findall(r"^(?:async )?def\s+([A-Za-z_]\w*)", code, re.MULTILINE)
        return names

    @staticmethod
    def _estimate_tokens(text):
        """Estimate token count for a text string (rough approximation)."""
        return len(text) // 4


# ---------------------------------------------------------------------------
# 4.2 Singleton factory for ConversationCompressor
# ---------------------------------------------------------------------------
_CONVERSATION_COMPRESSOR_INSTANCE: Optional[ConversationCompressor] = None
_CONVERSATION_COMPRESSOR_LOCK = None


def get_conversation_compressor() -> Optional[ConversationCompressor]:
    """
    Return the ConversationCompressor singleton, loading it once.

    The compressor uses LLMLingua-2 for history compression. It is loaded
    on demand and cached globally. Returns None if the llmlingua library
    is not available or initialisation fails.

    Returns:
        Optional[ConversationCompressor]: The compressor instance, or None
                                         if it could not be initialised.
    """
    global _CONVERSATION_COMPRESSOR_INSTANCE, _CONVERSATION_COMPRESSOR_LOCK
    import threading

    if _CONVERSATION_COMPRESSOR_LOCK is None:
        _CONVERSATION_COMPRESSOR_LOCK = threading.Lock()

    if _CONVERSATION_COMPRESSOR_INSTANCE is None:
        with _CONVERSATION_COMPRESSOR_LOCK:
            if _CONVERSATION_COMPRESSOR_INSTANCE is None:
                try:
                    from llmlingua import PromptCompressor

                    # FIX: use_llmlingua2 is the correct argument name for newer versions
                    raw = PromptCompressor(
                        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                        use_llmlingua2=True,
                        device_map="cpu",
                    )
                    _CONVERSATION_COMPRESSOR_INSTANCE = ConversationCompressor(raw)
                except ImportError:
                    pass
                except Exception as exc:
                    logging.getLogger(__name__).warning(f"ConversationCompressor init failed: {exc}")
    return _CONVERSATION_COMPRESSOR_INSTANCE


# ============================================================================
# SECTION 5: Utilities & Helpers
# ============================================================================

# ---------------------------------------------------------------------------
# 5.1 Active expert tracker (used for sticky routing)
# ---------------------------------------------------------------------------
_ACTIVE_EXPERT: Optional[str] = None
_ACTIVE_EXPERT_LOCK = threading.Lock()


def get_active_expert() -> Optional[str]:
    """
    Get the currently active expert ID for sticky routing.

    Returns:
        Optional[str]: The active expert ID, or None if no expert is active.
    """
    return _ACTIVE_EXPERT


def set_active_expert(expert_id: str) -> None:
    """
    Set the active expert ID for sticky routing.

    Args:
        expert_id (str): The expert ID to set as active.
    """
    global _ACTIVE_EXPERT
    with _ACTIVE_EXPERT_LOCK:
        _ACTIVE_EXPERT = expert_id


# ---------------------------------------------------------------------------
# 5.2 Helper to safely unload all models from a llama.cpp server
# ---------------------------------------------------------------------------
async def unload_all_models(base_url: str) -> None:
    """
    Unload all currently loaded models from a llama.cpp server.

    This is used when switching auxiliary models to free VRAM before
    loading a different model. The function queries the server for loaded
    models and unloads them one by one.

    Args:
        base_url (str): The base URL of the llama.cpp server.
    """
    import aiohttp
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.get(f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception:
            return
        models = []
        for m in data.get("data", []):
            if m.get("status", {}).get("value") == "loaded":
                models.append(m["id"])
        for model_id in models:
            try:
                async with sess.post(
                    f"{base_url}/models/unload",
                    json={"model": model_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logging.getLogger(__name__).debug(f"Unloaded model {model_id}")
            except Exception:
                pass
        await asyncio.sleep(0.5)
