"""
title: Shared Resources for OpenWebUI Plugins
description: Shared singletons (embedder, ChromaDB, HTTP session, tiktoken, LRU cache)
             to avoid resource duplication across plugins.
             Now includes a helper to safely unload all llama.cpp models,
             plus ConversationCompressor for v8 history compression.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 2.2.0
license: GPL3
requirements: aiohttp, chromadb, tiktoken, sentence-transformers, llmlingua>=0.2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import aiohttp

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
_ORPHANED_HTTP_SESSIONS: List[Any] = []
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

    async with _get_http_lock():
        needs_recreate = (
            _HTTP_SESSION is None
            or _HTTP_SESSION.closed
            or timeout_seconds > _HTTP_TIMEOUT_SECONDS
        )
        if needs_recreate:
            if _HTTP_SESSION is not None and not _HTTP_SESSION.closed:
                # Do NOT close: another coroutine may be mid-stream on this
                # session, and close() aborts its connection (the failure
                # then hides behind the caller's retry). Orphan it instead
                # and let in-flight requests drain; close_http_session()
                # sweeps orphans on shutdown. Bounded leak: one session per
                # distinct timeout upgrade in the process lifetime.
                _ORPHANED_HTTP_SESSIONS.append(_HTTP_SESSION)
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
    """Close the shared session and any orphaned predecessors."""
    global _HTTP_SESSION
    async with _get_http_lock():
        if _HTTP_SESSION and not _HTTP_SESSION.closed:
            await _HTTP_SESSION.close()
            _HTTP_SESSION = None
        while _ORPHANED_HTTP_SESSIONS:
            old = _ORPHANED_HTTP_SESSIONS.pop()
            if old and not old.closed:
                await old.close()


# ---------------------------------------------------------------------------
# 3.2 Shared LLM caller
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)


# 3.2.1 ── Typed errors ──────────────────────────────────────────────────────
# All subclass RuntimeError so existing callers that do `except RuntimeError`
# keep working unchanged, while new callers can branch on the specific type
# (and read `.status` on HTTP errors) instead of parsing exception strings.

class LLMError(RuntimeError):
    """Base class for every error raised by call_llm."""


class LLMConfigError(LLMError):
    """Invalid/missing configuration (no model, bad endpoint_type). Never retried."""


class LLMEmptyResponseError(LLMError):
    """Server returned an empty body / no content. Retried unless retry_on_empty=False."""


class LLMHTTPError(LLMError):
    """Non-2xx HTTP response. `.status` holds the code; 5xx and 429 are retryable."""

    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:300]}" if body else f"HTTP {status}")


# 3.2.2 ── Result type ───────────────────────────────────────────────────────
@dataclass
class LLMResult:
    """Structured result, returned by call_llm(..., return_meta=True).

    Lets callers track cost (token counts), detect truncation (finish_reason /
    truncated) and observe latency / retries without re-instrumenting each
    call site.
    """
    content: str
    backend: str
    model: str
    url: str
    attempts: int
    latency_s: float
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    truncated: bool = False


# 3.2.3 ── Helpers ───────────────────────────────────────────────────────────
_REDACT_KEYS = frozenset({"authorization", "api_key", "api_token", "token", "bearer", "key"})
_TRUNCATION_REASONS = frozenset({"length", "max_tokens"})


def _describe_output_tail(text: str, lines_back: int = 20) -> str:
    """
    Classify the line-level repetition at the end of a generation.

    Written for the one question a truncation log cannot otherwise
    answer: what SHAPE was the loop. A token count says a call ran
    to 57907 tokens; it cannot say whether the model repeated a
    block verbatim or walked a template substituting one identifier
    per line. The two call for opposite responses, because DRY's
    penalty grows with the length of the MATCHING run: a verbatim
    repeat lengthens its own match every iteration and is strangled
    as soon as it passes dry_allowed_length, while a varying line
    resets the match at the token that varies and stays invisible at
    any allowed_length. Lowering the threshold fixes the first and
    merely taxes legitimate code in the second.

    Deterministic and bounded: a fixed tail window, a fixed line
    count, and character comparison only.

    Args:
        text: The generation to inspect. Only its tail is read.
        lines_back: How many trailing non-blank lines to compare.

    Returns:
        A one-line description, or "" when the tail carries no
        line-level repetition worth reporting.
    """
    # Nothing this helper can hit is worth an exception escaping it.
    # It runs inside call_llm's truncation branch, whose surrounding
    # try catches only transport errors — a TypeError from a
    # non-string body or an IndexError from an unexpected shape would
    # propagate straight out of a call that had ALREADY SUCCEEDED and
    # merely hit a ceiling. A diagnostic must never be able to turn a
    # usable truncated response into a failed call.
    try:
        return _describe_output_tail_inner(text, lines_back)
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return ""


def _describe_output_tail_inner(text: str, lines_back: int) -> str:
    """Body of _describe_output_tail; see that function for intent."""
    # ── Step 1: the tail, as comparable non-blank lines ──
    rows = [r.rstrip() for r in (text or "")[-6000:].split("\n")]
    rows = [r for r in rows if r.strip()][-max(4, lines_back):]
    if len(rows) < 4:
        return ""

    # ── Step 2: the two signatures ──
    # Exact duplicates say 'verbatim repeat'. Adjacent lines that
    # share a long prefix without being equal say 'template walk',
    # which is the shape DRY cannot reach.
    exact = len(rows) - len(set(rows))
    prefixes = []
    for a, b in zip(rows, rows[1:]):
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        prefixes.append(n)
    shared = sorted(p for p in prefixes if p >= 12)
    if exact == 0 and not shared:
        return ""

    # ── Step 3: name the shape, from the number that decides it ──
    # The verdict follows the shared-prefix LENGTH, because that is what
    # DRY actually measures. An earlier version keyed off the duplicate
    # count alone and told the operator a template walk was beyond DRY's
    # reach at any setting — advice that was exactly backwards for the
    # first real runaway it saw, whose lines shared a 48-character
    # prefix (~13 tokens) under a threshold of 17. Lowering the
    # threshold was precisely the fix, and the instrument argued
    # against it while printing the number that proved it wrong.
    median = shared[len(shared) // 2] if shared else 0
    # Chars per token is a rough constant for prose and code alike;
    # the estimate only has to be good enough to compare against a
    # threshold expressed in tokens, and it is reported as approximate.
    approx_tokens = int(median / 3.7)
    if exact >= max(2, len(rows) // 3):
        shape = (
            "VERBATIM repeat — the match grows with every iteration, so "
            "DRY strangles this once it passes dry_allowed_length; a "
            "lower threshold bites sooner"
        )
    elif len(shared) >= max(2, (len(rows) - 1) // 2):
        if approx_tokens >= 4:
            shape = (
                "TEMPLATE walk (lines vary after a shared prefix) — the "
                "prefix is ~%d tokens, so DRY REACHES this whenever "
                "dry_allowed_length is below that; at or above it the "
                "match never crosses the threshold and nothing fires"
                % approx_tokens
            )
        else:
            shape = (
                "TEMPLATE walk with a ~%d token prefix — too short for "
                "DRY at any usable threshold, since lowering it far "
                "enough would penalise ordinary repeated code; the fix "
                "is not the sampler"
                % approx_tokens
            )
    else:
        shape = "partial repetition"
    return (
        "tail of %d lines: %d exact duplicate(s), %d adjacent pair(s) "
        "sharing >=12 chars (median %d chars ~%d tokens) — %s "
        "| last line: %r"
        % (
            len(rows),
            exact,
            len(shared),
            median,
            approx_tokens,
            shape,
            rows[-1][:120],
        )
    )


def _resolve_backend(
    backend: Optional[str], base_url: str, model: str
) -> Literal["ollama", "llamacpp", "openai"]:
    """Resolve the backend, honouring an explicit override before heuristics.

    The heuristic is best-effort: a model prefix `llamacpp/` wins, then an
    `ollama` / `:11434` hint in the URL, else assume an OpenAI-compatible
    server. Pass `backend=` explicitly whenever the heuristic might misfire
    (e.g. llama.cpp served on :11434, or "ollama" appearing in a proxy host).
    """
    if backend is not None:
        return backend  # type: ignore[return-value]
    if model.startswith("llamacpp/"):
        return "llamacpp"
    low = base_url.lower()
    if "ollama" in low or ":11434" in low:
        return "ollama"
    return "openai"


def _redact(obj: Any) -> Any:
    """Recursively mask credential-like values before logging a payload."""
    if isinstance(obj, dict):
        return {
            k: ("***REDACTED***" if str(k).lower() in _REDACT_KEYS else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def _strip_think_blocks(content: str) -> str:
    """
    Remove well-formed <think>…</think> blocks from a response.

    Servers running without a separated reasoning format ship the chain
    of thought inline in `content`; no downstream consumer of this module
    wants the raw tags. Only well-formed pairs are removed: an unclosed
    <think> means generation was truncated mid-reasoning, and stripping
    to the end would return an empty answer and trigger pointless
    retries — the truncation warning already covers that case. When
    stripping would leave nothing, the original text is returned intact.
    """
    if "<think>" not in content:
        return content
    stripped = _THINK_BLOCK_RE.sub("", content).strip()
    return stripped if stripped else content


def _is_retryable(exc: BaseException, retry_on_empty: bool) -> bool:
    """Decide whether an exception is worth another attempt.

    Decisions use the exception *type*, never its message text, so log/string
    changes can never silently flip retry behaviour. Specific subclasses are
    checked before the generic RuntimeError fallback.
    """
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, LLMHTTPError):
        return exc.status == 429 or 500 <= exc.status < 600
    if isinstance(exc, LLMEmptyResponseError):
        return retry_on_empty
    if isinstance(exc, LLMConfigError):
        return False
    if isinstance(exc, RuntimeError):
        # Unknown RuntimeError: preserve the original "retry transient" behaviour.
        return True
    return False


def _build_request(
    *,
    backend: str,
    base_url_clean: str,
    endpoint_type: str,
    model_str: str,
    prompt: str,
    system: str,
    temperature: float,
    forward_max_tokens: Optional[int],
    response_format: Optional[Dict[str, Any]],
    enable_thinking: bool,
    seed: Optional[int],
    stop: Optional[List[str]],
    top_p: Optional[float],
    extra_body: Optional[Dict[str, Any]],
    stream: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build the (url, payload) for the resolved backend.

    When stream is True the payload requests incremental token delivery
    (SSE for openai/llamacpp, newline-delimited JSON for ollama). Streaming
    is what makes client-side cancellation effective: the server only notices
    a dropped connection when it next writes a token, so a non-streaming
    request runs to completion server-side regardless of what the client does.
    """
    if backend == "ollama":
        url = f"{base_url_clean}/api/generate"
        options: Dict[str, Any] = {"temperature": temperature}
        if forward_max_tokens is not None:
            options["num_predict"] = forward_max_tokens
        if seed is not None:
            options["seed"] = seed
        if top_p is not None:
            options["top_p"] = top_p
        if stop:
            options["stop"] = stop
        if not enable_thinking:
            options["think"] = False
        payload: Dict[str, Any] = {
            "model": model_str,
            "prompt": prompt,
            "system": system,
            "stream": stream,
            "options": options,
        }
        if isinstance(response_format, dict):
            rf_type = response_format.get("type")
            if rf_type == "json_object":
                payload["format"] = "json"
            elif rf_type == "json_schema":
                schema = (response_format.get("json_schema") or {}).get("schema")
                if schema is not None:
                    payload["format"] = schema

    else:
        if endpoint_type == "completion":
            url = f"{base_url_clean}/v1/completions"
            payload = {
                "model": model_str,
                "prompt": prompt if not system else f"{system}\n\n{prompt}",
                "temperature": temperature,
            }
        else:
            url = f"{base_url_clean}/v1/chat/completions"
            payload = {
                "model": model_str,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
        if backend == "llamacpp":
            # Explicitly request slot prompt caching. Modern llama.cpp
            # defaults cache_prompt to true, but the whole KV strategy of
            # the context_manager plugin (slot save/restore, the pre_aligned
            # launchpad, --cache-reuse chunk reuse) depends on it, so the
            # request must not be at the mercy of a build's default. A
            # production run measured ZERO prefix reuse on the aligned
            # auxiliary calls (planner: 125.9s cold vs 123.1s with an
            # 80,693-token snapshot restored and a 61,294-token common
            # head) — this makes the client side of that contract explicit.
            payload["cache_prompt"] = True
        if forward_max_tokens is not None:
            payload["max_tokens"] = forward_max_tokens
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if not enable_thinking and backend == "llamacpp":
            if endpoint_type == "chat":
                payload["thinking"] = False
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                # /v1/completions applies no chat template, so neither
                # `thinking` nor `chat_template_kwargs` has any effect —
                # the request would silently think anyway. Warn instead
                # of pretending the opt-out was honored.
                _logger.warning(
                    "enable_thinking=False cannot be honored on the "
                    "completion endpoint (no chat template); the model "
                    "may still emit reasoning. Use endpoint_type='chat'."
                )
        payload["stream"] = stream

    if extra_body:
        for k, v in extra_body.items():
            if k == "options" and isinstance(v, dict) and isinstance(payload.get("options"), dict):
                payload["options"].update(v)
            else:
                payload[k] = v

    return url, payload


def _parse_response(
    *, backend: str, endpoint_type: str, data: Dict[str, Any]
) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    """Extract (content, finish_reason, prompt_tokens, completion_tokens).

    Raises LLMEmptyResponseError when no usable content is present.
    """
    if backend == "ollama":
        content = data.get("response", "")
        finish_reason = data.get("done_reason")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if not content:
            err = data.get("error", "")
            if err:
                raise LLMError(f"Ollama error: {err}")
            raise LLMEmptyResponseError("Empty response")
        return content, finish_reason, prompt_tokens, completion_tokens

    # openai-compatible
    choices = data.get("choices", [])
    if not choices:
        raise LLMEmptyResponseError("No choices")
    choice0 = choices[0] or {}
    finish_reason = choice0.get("finish_reason")
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    if endpoint_type == "completion":
        content = choice0.get("text", "")
        if not content:
            raise LLMEmptyResponseError("Empty content")
    else:
        msg = choice0.get("message") or {}
        content = msg.get("content", "")
        if not content:
            # Some reasoning models put the answer in reasoning_content when
            # content is empty — fall back to it rather than failing.
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                content = reasoning.strip()
            else:
                raise LLMEmptyResponseError("Empty content")

    return content, finish_reason, prompt_tokens, completion_tokens


async def _consume_stream(
    resp: aiohttp.ClientResponse,
    *,
    backend: str,
    endpoint_type: str,
) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    """
    Consume a streaming LLM response line by line, accumulating content.

    Reading token by token (instead of awaiting the whole body) is precisely
    what makes cancellation work: llama.cpp only checks whether the client is
    still connected when it next tries to write a token, so a non-streaming
    request generates to completion no matter what the client does. Under
    streaming, closing the socket between tokens lets the server observe the
    disconnect and abort the in-flight generation within roughly one token.

    Handles both wire formats:
        - openai / llamacpp: Server-Sent Events, one JSON object per `data:`
          line, terminated by `data: [DONE]`.
        - ollama: newline-delimited JSON, one object per line, final object
          carrying `done: true`.

    Returns:
        (content, finish_reason, prompt_tokens, completion_tokens). Token
        counts are None when the server does not report usage in the stream.

    Raises:
        LLMEmptyResponseError: the stream closed without any content.
    """
    chunks: List[str] = []
    reasoning_chunks: List[str] = []
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    async for raw_line in resp.content:
        line = raw_line.decode("utf-8", "ignore").strip()
        if not line:
            continue

        # ── Step 1: unwrap the SSE envelope for openai / llamacpp ──
        if backend != "ollama":
            if not line.startswith("data:"):
                continue
            line = line[len("data:"):].strip()
            if line == "[DONE]":
                break

        # ── Step 2: parse one event; skip anything unparseable ──
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # ── Step 3: accumulate the incremental content + bookkeeping ──
        # Thinking deltas (reasoning_content) go to a SEPARATE buffer. During
        # the whole thinking phase every delta has empty `content` and only
        # `reasoning_content`, so appending reasoning inline whenever content
        # is momentarily empty concatenates the entire chain-of-thought in
        # front of the answer — unmarked, budget-eating, and different from
        # the non-streaming path, whose reasoning fallback fires only when
        # the WHOLE response carried no content. The streaming path mirrors
        # that rescue below instead of interleaving per delta.
        if backend == "ollama":
            chunks.append(event.get("response", "") or "")
            if event.get("done"):
                finish_reason = event.get("done_reason") or finish_reason
                prompt_tokens = event.get("prompt_eval_count", prompt_tokens)
                completion_tokens = event.get("eval_count", completion_tokens)
                break
        else:
            choices = event.get("choices") or []
            if choices:
                choice0 = choices[0] or {}
                if endpoint_type == "completion":
                    chunks.append(choice0.get("text", "") or "")
                else:
                    delta = choice0.get("delta") or {}
                    piece = delta.get("content", "")
                    if piece:
                        chunks.append(piece)
                    reasoning = delta.get("reasoning_content", "")
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                if choice0.get("finish_reason"):
                    finish_reason = choice0["finish_reason"]
            usage = event.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

    content = "".join(chunks)
    if not content and reasoning_chunks:
        # Mirror of the non-streaming rescue: some reasoning models put the
        # entire answer in reasoning_content — fall back to it rather than
        # failing, but ONLY when no regular content arrived at all.
        content = "".join(reasoning_chunks).strip()
    if not content:
        raise LLMEmptyResponseError("Empty content")
    return content, finish_reason, prompt_tokens, completion_tokens

# 3.2.4 ── Public entry point ────────────────────────────────────────────────
# NOTE: get_model_backend / get_model_name are defined elsewhere in this same module.
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
    response_format: Optional[Dict[str, Any]] = None,
    enable_thinking: bool = True,
    # --- new: backend + sampling controls ---
    backend: Optional[Literal["ollama", "llamacpp", "openai"]] = None,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    top_p: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    # --- new: retry controls ---
    max_retries: int = 3,
    base_delay: float = 1.0,
    deadline_seconds: Optional[float] = None,
    retry_on_empty: bool = True,
    # --- new: observability ---
    return_meta: bool = False,
    log_raw_response: bool = False,
    label: str = "",
    # --- new: streaming + stall detection ---
    stream: bool = True,
    sock_read: Optional[int] = None,
) -> Union[str, LLMResult]:
    """
    Async LLM call with retries, typed errors, and optional response metadata.

    Supports Ollama (/api/generate), llama.cpp and OpenAI-compatible
    (/v1/chat/completions, /v1/completions) endpoints. Transient failures
    (network errors, timeouts, HTTP 5xx/429, empty responses) are retried with
    exponential backoff + jitter; 4xx (except 429) and configuration errors
    fail immediately.

    Backward compatible: by default returns the response text as a stripped
    str, and every error subclasses RuntimeError. Pass return_meta=True to get
    an LLMResult with token counts, finish_reason, latency and resolved backend.

    Streaming (stream=True, the default) is what makes client-side cancellation
    effective. A non-streaming request awaits the whole body: llama.cpp only
    checks whether the client is still connected when it next writes a token, so
    with nothing to write until the end it generates to completion server-side
    regardless of what the client does — which is exactly how a cancelled or
    abandoned call left an orphaned generation holding the single --parallel 1
    slot. Under streaming, the body is consumed token by token, so closing the
    socket mid-stream lets the server observe the disconnect and abort the
    in-flight generation within roughly one token.

    Args:
        prompt: The user prompt.
        system: System prompt (chat) or prepended to the prompt (completion).
        base_url: Base URL of the LLM server.
        model: Model name. Required. A vendor prefix is stripped before
               sending ("llamacpp/mistral" → "mistral") and is also used to
               auto-detect the llama.cpp backend.
        api_token: Bearer token for authenticated endpoints. Sent as a header,
                   never logged.
        temperature: Sampling temperature. Must be >= 0.
        max_tokens: Max tokens to generate. None or 0 means "no limit" — the
                    key is omitted so the server uses its own default. Passing
                    0 to llama.cpp would clamp generation to a single token, so
                    we guard against that here.
        timeout: Per-request timeout in seconds (applies to each attempt).
        endpoint_type: 'chat' or 'completion'. Anything else raises
                       LLMConfigError (previously it silently fell back to chat).
        response_format: Server-side output constraint. For OpenAI-compatible
                         backends it is passed through verbatim (e.g.
                         {"type": "json_object"}). For Ollama it is translated
                         to the `format` field ("json", or a JSON schema dict)
                         instead of being dropped.
        enable_thinking: Allow chain-of-thought before the answer. Set False
                         for deterministic structured-output tasks. The
                         llama.cpp opt-out fields are sent ONLY when the backend
                         is known to be llama.cpp; for Ollama {"think": false}
                         is used. They are never sent to a generic OpenAI
                         endpoint (which may reject unknown fields).
        backend: Force 'ollama' | 'llamacpp' | 'openai' instead of detecting it
                 from base_url / model prefix. The heuristic can misfire
                 (e.g. llama.cpp on :11434); set this when in doubt.
        seed: Optional sampling seed for reproducibility.
        stop: Optional list of stop sequences.
        top_p: Optional nucleus-sampling value.
        extra_body: Extra keys merged into the outgoing JSON (last, so they
                    win). For Ollama, a nested "options" dict is merged into
                    options rather than replacing it. Use this for
                    backend-specific knobs without editing this function.
        max_retries: Total attempts before giving up (default 3).
        base_delay: Base backoff seconds; actual delay is
                    base_delay * 2**(attempt-1) * jitter(0.5–1.5).
        deadline_seconds: Optional hard cap on total wall-clock across all
                          retries. Bounds the worst case
                          (max_retries * timeout + backoff).
        retry_on_empty: Whether an empty/unparseable response is retried
                        (default True). Set False for deterministic callers
                        where a retry cannot change the outcome.
        return_meta: If True, return an LLMResult instead of a bare str.
        log_raw_response: Log the outgoing request and raw response at INFO,
                          each line prefixed [RAW][label]. Payloads are
                          redacted for credential-like keys. Opt-in; do not
                          leave on for high-frequency callers.
        label: Identifier used only in log prefixes. Always set it when
               log_raw_response=True so the [RAW][label] prefix is meaningful.
        stream: Request incremental token delivery and consume the response as
                it arrives. Default True. This is the property that makes
                cancellation actually free the server-side slot; set False only
                for a backend that cannot stream.
        sock_read: Per-socket read timeout in seconds. Fires when the server
                   stops emitting tokens mid-stream, turning an indefinite
                   server-side stall (GPU idle, no tokens, forever) into a
                   bounded, retryable timeout. None means "use `timeout`", i.e.
                   no finer-grained stall detection than the overall cap.

    Returns:
        str (default) or LLMResult (when return_meta=True). Content is stripped.

    Raises:
        LLMConfigError: missing model / invalid endpoint_type (not retried).
        LLMHTTPError: non-2xx response (`.status` set); 4xx except 429 fail fast.
        LLMEmptyResponseError: empty / unparseable response.
        LLMError: retries exhausted or deadline exceeded.
        (All subclass RuntimeError for backward compatibility.)
    """
    # --- Validation (fail fast, before any network work) ---
    if model is None:
        _logger.error("LLM call requested but no model was provided.")
        raise LLMConfigError("No model provided for LLM call.")
    if endpoint_type not in ("chat", "completion"):
        raise LLMConfigError(
            f"Invalid endpoint_type: {endpoint_type!r} (expected 'chat' or 'completion')."
        )
    if temperature < 0:
        raise LLMConfigError(f"temperature must be >= 0, got {temperature}.")
    if temperature > 2:
        _logger.warning(
            "temperature=%s is unusually high; most backends expect 0–2.", temperature
        )
    if max_retries < 1:
        raise LLMConfigError(f"max_retries must be >= 1, got {max_retries}.")

    resolved_backend = _resolve_backend(backend, base_url, model)

    base_url_clean = base_url.rstrip("/")
    if base_url_clean.endswith("/v1"):
        base_url_clean = base_url_clean[:-3].rstrip("/")

    model_str = get_model_name(model)

    headers = {"Content-Type": "application/json"}
    if api_token and api_token.strip():
        headers["Authorization"] = f"Bearer {api_token.strip()}"

    # None or 0 means "no limit" — omit the key entirely (see max_tokens doc).
    forward_max_tokens: Optional[int] = (
        max_tokens if (max_tokens is not None and max_tokens > 0) else None
    )

    url, payload = _build_request(
        backend=resolved_backend,
        base_url_clean=base_url_clean,
        endpoint_type=endpoint_type,
        model_str=model_str,
        prompt=prompt,
        system=system,
        temperature=temperature,
        forward_max_tokens=forward_max_tokens,
        response_format=response_format,
        enable_thinking=enable_thinking,
        seed=seed,
        stop=stop,
        top_p=top_p,
        extra_body=extra_body,
        stream=stream,
    )

    start = time.monotonic()
    tag = label or "-"

    for attempt in range(1, max_retries + 1):
        try:
            session = await get_http_session(timeout)

            # Log the outgoing request before sending so the payload is visible
            # even if the call times out or the server crashes. Credentials are
            # masked even though the token normally lives only in the headers.
            if log_raw_response:
                _logger.info(
                    f"[RAW][{label}] → {url}\n"
                    f"payload: {json.dumps(_redact(payload), ensure_ascii=False, indent=2)}"
                )

            # ── Per-request timeout ──
            # `total` bounds the whole call; `sock_read` fires when the server
            # stops emitting tokens mid-stream. That converts an indefinite
            # server-side stall (GPU idle, no tokens, forever) into a bounded,
            # retryable timeout instead of a permanent hang — the amplifier
            # behind the multi-minute and 44-minute blocks. When sock_read is
            # None it falls back to `total`, i.e. no finer-grained detection.
            req_timeout = aiohttp.ClientTimeout(
                total=timeout,
                sock_read=(sock_read if sock_read is not None else timeout),
            )

            async with session.post(
                url, json=payload, headers=headers, timeout=req_timeout
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if log_raw_response:
                        _logger.info(
                            f"[RAW][{label}] ← HTTP {resp.status}\n{text[:2000]}"
                        )
                    raise LLMHTTPError(resp.status, text)

                # ── Read the body, streaming or whole ──
                # On cancel or a mid-stream stall, force the socket shut so the
                # server observes the disconnect and aborts the in-flight
                # generation. close() (not release()) is what sends the RST/FIN:
                # release() would return the connection to the pool while the
                # slot keeps generating to completion — the exact behaviour that
                # left orphaned generations blocking the single --parallel 1
                # slot.
                try:
                    if stream:
                        content, finish_reason, ptok, ctok = await _consume_stream(
                            resp,
                            backend=resolved_backend,
                            endpoint_type=endpoint_type,
                        )
                    else:
                        data = await resp.json()
                        content, finish_reason, ptok, ctok = _parse_response(
                            backend=resolved_backend,
                            endpoint_type=endpoint_type,
                            data=data,
                        )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    resp.close()
                    raise

            if log_raw_response:
                _logger.info(
                    f"[RAW][{label}] ← HTTP 200 (stream={stream}) ~{len(content)} chars"
                )

            truncated = (finish_reason or "") in _TRUNCATION_REASONS
            if truncated:
                # The shape of what was generated, not just how much.
                # Computed only on the truncation path, so a healthy
                # call pays nothing.
                _tail = _describe_output_tail(content)
                # A length stop means two very different things, and the
                # old message assumed the first. When a finite ceiling was
                # forwarded, the caller's own max_tokens cut the call and
                # raising it is the fix. When none was forwarded
                # (max_tokens None/0 — the caller deliberately disabled
                # every cap), nothing of ours cut anything: the generation
                # ran until it hit the server's context window. Advising
                # 'raise max_tokens' there is worse than useless — it
                # points the operator at a control they already turned off,
                # which reads as the disablement having failed. Observed
                # exactly this way: a 57907-token completion under an
                # intentionally uncapped valve, reported as if a cap had
                # fired.
                if forward_max_tokens is None:
                    _logger.warning(
                        "LLM output hit the CONTEXT WINDOW, not a configured "
                        "cap (finish_reason=%s, label=%s, model=%s, "
                        "completion_tokens=%s): no max_tokens was sent, so "
                        "the model generated until the server's n_ctx ran "
                        "out. Raising max_tokens cannot help; a generation "
                        "this long is a degenerate loop or a prompt that "
                        "invites unbounded output.%s",
                        finish_reason, tag, model_str, ctok,
                        (" " + _tail) if _tail else "",
                    )
                else:
                    _logger.warning(
                        "LLM output truncated by the configured cap "
                        "(finish_reason=%s, label=%s, model=%s, "
                        "max_tokens=%s, completion_tokens=%s); consider "
                        "raising max_tokens.%s",
                        finish_reason, tag, model_str,
                        forward_max_tokens, ctok,
                        (" " + _tail) if _tail else "",
                    )
            elif finish_reason == "content_filter":
                _logger.warning("LLM output filtered (content_filter, label=%s).", tag)

            if attempt > 1:
                _logger.debug(
                    "LLM call succeeded on attempt %d/%d (label=%s).",
                    attempt, max_retries, tag,
                )

            content = _strip_think_blocks(content)

            result = LLMResult(
                content=content.strip(),
                backend=resolved_backend,
                model=model_str,
                url=url,
                attempts=attempt,
                latency_s=time.monotonic() - start,
                finish_reason=finish_reason,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                truncated=truncated,
            )
            return result if return_meta else result.content

        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            # Non-retryable (4xx except 429, config errors): re-raise the actual
            # typed exception so callers keep `.status` etc. — no string parsing.
            if not _is_retryable(exc, retry_on_empty):
                _logger.warning(
                    "LLM call failed (non-retryable, label=%s): %s", tag, exc
                )
                raise

            # Last attempt — re-raise the original typed exception, preserving
            # its type/traceback rather than flattening to a generic message.
            if attempt == max_retries:
                _logger.error(
                    "LLM call failed after %d attempts (label=%s): %s",
                    max_retries, tag, exc,
                )
                raise

            # Exponential backoff with jitter (0.5x–1.5x) to avoid thundering
            # herd when many callers retry at once.
            delay = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())

            # Respect an overall deadline rather than running max_retries blind.
            if deadline_seconds is not None:
                elapsed = time.monotonic() - start
                if elapsed + delay >= deadline_seconds:
                    _logger.error(
                        "LLM call exceeded deadline of %.1fs (label=%s): %s",
                        deadline_seconds, tag, exc,
                    )
                    raise

            _logger.debug(
                "LLM call attempt %d/%d failed (label=%s), retrying in %.1fs: %s",
                attempt, max_retries, tag, delay, exc,
            )
            await asyncio.sleep(delay)

    # Defensive safety net — the loop always returns or raises above.
    raise LLMError(f"LLM call failed after {max_retries} attempts")


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
# 5.3 Helpers to parse a "backend/model" formatted model identifier
# ---------------------------------------------------------------------------
DEFAULT_BACKEND = "llamacpp"


def get_model_backend(model_id: str) -> str:
    """Return the backend part of a "backend/model" id (default: "llamacpp")."""
    return model_id.split("/", 1)[0] if "/" in model_id else DEFAULT_BACKEND


def get_model_name(model_id: str) -> str:
    """Return the model name without the backend prefix."""
    return model_id.split("/", 1)[1] if "/" in model_id else model_id
