"""
title: Code-Aware Context Manager with LTM & Summarization
description: Full-featured context manager for coding assistants.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 9.0.0
license: GPL3
requirements: loguru, tiktoken, sentence-transformers, chromadb, rapidfuzz, tree-sitter==0.25.2, tree-sitter-language-pack==1.8.1, llmlingua>=0.2.2
"""

import os
import time
import re
import anyio
import hashlib
import sqlite3
import ast
import contextvars
import json
import asyncio
import threading
import textwrap
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Union, Set, Iterable
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE = True
except ImportError:
    HAS_SENTENCE = False

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from rapidfuzz import fuzz

    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

try:
    from tree_sitter_language_pack import (
        get_language,
        detect_language_from_extension,
        process,
        ProcessConfig,
    )

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

try:
    import tree_sitter

    HAS_TREE_SITTER_CORE = True
except ImportError:
    HAS_TREE_SITTER_CORE = False

import sys

if "/app/backend/data/custom_lib" not in sys.path:
    sys.path.append("/app/backend/data/custom_lib")

from shared_resources import (
    get_embedder as _shared_get_embedder,
    get_chroma_client as _shared_get_chroma_client,
    AsyncLRUCache as _AsyncLRUCache,
    get_http_session as _shared_get_http_session,
    call_llm as _shared_call_llm,
    unload_all_models as _shared_unload_all_models,
    get_conversation_compressor as _shared_get_conversation_compressor,  # ← v8
)

_db_global_lock = threading.Lock()
import fcntl
import tempfile

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# Edge types and base weights
EDGE_WEIGHTS: Dict[str, float] = {
    "calls": 1.0,  # A function/method calls another
    "imports": 0.6,  # Module-level import relationship
    "reads": 0.7,  # Data is read from a variable or field
    "writes": 0.9,  # Data is written to a variable or field
    "inherits": 0.5,  # Class inheritance
    "references": 0.4,  # General reference (type annotation, parameter, etc.)
    "data_flow": 0.8,  # Data passes from a producer to a consumer via arguments
}

# CrossEncoder singleton (module‑level)
_CROSS_ENCODER = None
_CROSS_ENCODER_LOCK = threading.Lock()

# Global lock for SQLite operations (prevents "database is locked" errors)
_db_global_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Tree‑sitter fallback queries
# ---------------------------------------------------------------------------
# These S‑expression patterns tell tree‑sitter how to locate function / class
# definitions and call sites for each supported programming language.
# They are used by _extract_symbols_from_tree and _extract_calls_from_tree,
# which together produce the qualified CodeSymbol list and the call graph.
# If a language is missing from these maps, extraction is skipped with a
# warning — there is no legacy fallback that could inject unqualified data.

FALLBACK_LANGUAGE_QUERIES = {
    "python": """
        (function_definition name: (identifier) @name) @func
        (class_definition name: (identifier) @name) @class
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @func
        (class_declaration name: (identifier) @name) @class
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function)) @func)
    """,
    "tsx": """
        (function_declaration name: (identifier) @name) @func
        (class_declaration name: (identifier) @name) @class
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function)) @func)
    """,
    "go": """
        (function_declaration name: (identifier) @name) @func
        (type_declaration (type_spec name: (type_identifier) @name)) @class
    """,
    "rust": """
        (function_item name: (identifier) @name) @func
        (struct_item name: (type_identifier) @name) @class
        (enum_item name: (type_identifier) @name) @class
    """,
    "java": """
        (method_declaration name: (identifier) @name) @func
        (class_declaration name: (identifier) @name) @class
    """,
    "cpp": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name)) @func
        (class_specifier name: (type_identifier) @name) @class
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name)) @func
    """,
}

FALLBACK_CALL_QUERIES = {
    "python": """
        (function_definition
            body: (block
                (expression_statement
                    (call function: [(identifier) (attribute) @callee])))) @caller
    """,
    "javascript": """
        (function_declaration
            body: (statement_block
                (expression_statement
                    (call_expression function: [(identifier) (member_expression) @callee])))) @caller
        (lexical_declaration
            (variable_declarator
                name: (identifier) @caller_name
                value: (arrow_function body: (statement_block
                    (expression_statement
                        (call_expression function: [(identifier) (member_expression) @callee]))))))
    """,
    "tsx": """
        (function_declaration
            body: (statement_block
                (expression_statement
                    (call_expression function: [(identifier) (member_expression) @callee])))) @caller
        (lexical_declaration
            (variable_declarator
                name: (identifier) @caller_name
                value: (arrow_function body: (statement_block
                    (expression_statement
                        (call_expression function: [(identifier) (member_expression) @callee]))))))
    """,
    "go": """
        (function_declaration
            body: (block
                (expression_statement
                    (call_expression function: [(identifier) (selector_expression) @callee])))) @caller
    """,
    "rust": """
        (function_item
            body: (block
                (expression_statement
                    (call_expression function: [(identifier) (field_expression) @callee])))) @caller
        (function_item
            body: (block
                (macro_invocation macro: (identifier) @callee))) @caller
    """,
    "java": """
        (method_declaration
            body: (block
                (expression_statement
                    (method_invocation name: [(identifier) (field_access) @callee])))) @caller
    """,
    "cpp": """
        (function_definition
            body: (compound_statement
                (call_expression function: [(identifier) (field_expression) @callee]))) @caller
    """,
    "c": """
        (function_definition
            body: (compound_statement
                (call_expression function: [(identifier) (field_expression) @callee]))) @caller
    """,
}

# ---------------------------------------------------------------------------
# Global helper functions
# ---------------------------------------------------------------------------


class UseCase(str, Enum):
    """
    Use case categories for intent classification.

    Each member has a short internal key (value) and a human-readable label.
    The internal key is used for LOD profiles and logic; the label is used
    for logging and prompts.
    """

    ARCHITECTURE = "A"
    PLANNING = "B"
    PROGRAMMING = "C"
    REFACTORING = "D"
    SCAFFOLDING = "E"

    @property
    def label(self) -> str:
        """Human-readable label for the use case."""
        return {
            "A": "Architecture/Design",
            "B": "Planning/Roadmap",
            "C": "General Programming",
            "D": "Refactoring/Impact Analysis",
            "E": "Scaffolding/Boilerplate",
        }[self.value]


def qualify_symbol_name(
    name: str, parent_symbol: str, file_path: Optional[str] = None
) -> str:
    """
    Unique-within-project identity for a symbol.

    - Class-scoped symbols: 'ClassName.method'
    - Module-level functions: 'module.function' (derived from file_path)
    - Fallback: bare name when neither parent nor file_path is available

    This is the central fix for the docstring / call-graph collision bug:
    same-named methods in different classes, and same-named functions in
    different files, are now stored under distinct qualified ids so they
    never stomp on or fuse with each other.
    """
    if parent_symbol:
        return f"{parent_symbol}.{name}"
    if file_path:
        module = os.path.splitext(os.path.basename(file_path))[0]
        if module and module != name:
            return f"{module}.{name}"
    return name


def qualify_symbol(sym: "CodeSymbol") -> str:
    """
    Convenience wrapper: qualify a CodeSymbol using ALL the identity
    fields the indexing side already uses (name, parent_symbol, file_path).
    Prefer this over calling qualify_symbol_name(sym.name, sym.parent_symbol)
    directly — the two-arg form silently drops file_path, which only matters
    for module-level functions with a detected file path, but is exactly the
    inconsistency that caused them to go unmatched in several lookups.
    """
    return qualify_symbol_name(sym.name, sym.parent_symbol, sym.file_path)


def _get_cross_encoder(
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
) -> Optional[Any]:
    """
    Return the CrossEncoder singleton, loading it once. Thread‑safe.

    Used by `CommandRouter._predict_cross_encoder()` and other intent‑
    classification / reranking paths.  Returns None if the model cannot
    be loaded or `sentence_transformers` is not available.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        with _CROSS_ENCODER_LOCK:
            if _CROSS_ENCODER is None:
                try:
                    from sentence_transformers import CrossEncoder

                    _CROSS_ENCODER = CrossEncoder(model_name)
                except Exception:
                    return None
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# Models & Enums
# ---------------------------------------------------------------------------
class ContentType(str, Enum):
    """Classification of a code block's role in the conversation."""

    BASE_CODE = "base_code"  # Existing code provided by the user
    PROPOSED_CHANGE = "proposed_change"  # Suggested modification (diff or snippet)
    COMMITTED_CHANGE = "committed_change"  # Accepted / applied change
    GENERAL = "general"  # Plain conversation text
    TOOL_CALL = "tool_call"  # Structured tool / function call payload
    ERROR = "error"  # Traceback or error message


class CodeSymbol(BaseModel):
    """A single function, method, or class extracted from source code."""

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Symbol identity
    # ═══════════════════════════════════════════════════════════════════════════

    name: str
    kind: str  # "function", "class", or "method"
    signature: str

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Location in source
    # ═══════════════════════════════════════════════════════════════════════════

    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Relationships & context
    # ═══════════════════════════════════════════════════════════════════════════

    parent_block_hash: str = ""  # Hash of the CodeBlock that owns this symbol
    parent_symbol: str = ""  # Enclosing class name, or "" for top-level
    language: str = "unknown"

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Additional metadata (calls, docstring)
    # ═══════════════════════════════════════════════════════════════════════════

    calls: List[str] = Field(default_factory=list)  # Bare names called by this symbol
    docstring: str = ""


class CodeBlock(BaseModel):
    """A chunk of code managed by the context system.

    Every code block carries a content hash for deduplication, an importance
    score that decays over time, and a list of ``CodeSymbol`` entries
    extracted from its content.  The importance score is recalculated
    whenever ``_update_importance()`` is called.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Core content & identity
    # ═══════════════════════════════════════════════════════════════════════════

    content: str
    content_type: ContentType
    hash: str = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Location & timing
    # ═══════════════════════════════════════════════════════════════════════════

    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    timestamp: float = Field(default_factory=time.time)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. State flags
    # ═══════════════════════════════════════════════════════════════════════════

    is_active: bool = True
    generated_by_assistant: bool = False
    pinned: bool = False
    obsolete: bool = False
    is_raw: bool = False

    # REMOVED: potentially_affected: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Importance & recency
    # ═══════════════════════════════════════════════════════════════════════════

    importance_score: float = 1.0
    mention_count: int = 1
    last_mentioned: float = Field(default_factory=time.time)
    last_mentioned_msg_idx: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Extracted symbols & summary
    # ═══════════════════════════════════════════════════════════════════════════

    block_summary: str = ""
    symbols: List[CodeSymbol] = Field(default_factory=list)
    _cached_token_count: int = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, **data):
        super().__init__(**data)
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
        self._update_importance()

    def _update_importance(self):
        """Recalculate the importance score from content type, keywords,
        mention frequency, recency, and obsolete status."""
        base_score = {
            ContentType.BASE_CODE: 8.0,
            ContentType.ERROR: 7.0,
            ContentType.COMMITTED_CHANGE: 6.0,
            ContentType.PROPOSED_CHANGE: 5.0,
            ContentType.TOOL_CALL: 3.0,
            ContentType.GENERAL: 2.0,
        }.get(self.content_type, 2.0)
        keyword_boost = (
            2.0
            if re.search(
                r"\b(fix|bug|security|critical|important|todo)\b", self.content, re.I
            )
            else 0.0
        )
        if self.generated_by_assistant:
            base_score *= 0.8
        mention_boost = min(self.mention_count / 5, 3.0)
        recency_factor = 0.5 ** ((time.time() - self.last_mentioned) / 3600)
        penalty = 1.0
        if self.obsolete:
            penalty = 0.1
            self.is_active = False
        self.importance_score = (
            (base_score + keyword_boost) * mention_boost * recency_factor * penalty
        )


class Edge(BaseModel):
    """A directed relationship between two symbols in the call graph.

    The source (``src``) is always a qualified symbol id
    (``ClassName.method`` or ``module.function``).  The destination
    (``dst``) is a bare name — resolving it to a concrete class would
    require type inference, which is outside the scope of a static pass.
    Downstream components handle this by fanning out to every qualified
    symbol that shares the bare callee name.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Source & destination (identity)
    # ═══════════════════════════════════════════════════════════════════════════

    src: str
    dst: str

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Edge type & weight
    # ═══════════════════════════════════════════════════════════════════════════

    type: str
    weight: float = 1.0  # Base importance of this edge type
    confidence: float = 1.0  # 1.0 = confirmed, < 1.0 = inferred / provisional

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def effective_weight(self) -> float:
        """Effective weight used in activation propagation (PPR)."""
        return self.weight * self.confidence


# ---------------------------------------------------------------------------
# Activation Graph — query‑conditioned node activation
# ---------------------------------------------------------------------------
class ActivationState(BaseModel):
    """Snapshot of a single node's activation during PPR propagation."""

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Node identity
    # ═══════════════════════════════════════════════════════════════════════════

    node_id: str

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Activation score
    # ═══════════════════════════════════════════════════════════════════════════

    score: float

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Propagation metadata
    # ═══════════════════════════════════════════════════════════════════════════

    depth: int
    source: str


class ConversationState(BaseModel):
    """
    Persistent conversation state for a single project.
    """

    model_config = {"arbitrary_types_allowed": True}

    # ── Active code blocks ─────────────────────────────────────────────────
    active_blocks: Dict[str, "CodeBlock"] = Field(default_factory=dict)
    recent_changes: List["CodeBlock"] = Field(default_factory=list)
    committed_changes: List["CodeBlock"] = Field(default_factory=list)

    # ── Conversation counters ─────────────────────────────────────────────
    message_count: int = 0
    last_cot_level: int = 0

    # ── Feedback and suggestions ──────────────────────────────────────────
    feedback_history: List["AppliedChangeFeedback"] = Field(default_factory=list)
    last_compression_timestamp: float = 0.0
    last_suggestion_timestamp: float = 0.0
    last_cleanup_suggestion_msg_idx: int = 0

    # ── Call graph state ──────────────────────────────────────────────────
    has_any_calls: bool = False

    # ── Conversation summaries ────────────────────────────────────────────
    conversation_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    summarized_turn_hwm: int = 0

    # ── History compression tracker ──────────────────────────────────────
    history_blocked_age: Dict[str, int] = Field(default_factory=dict)

    # ── KV slot persistence ───────────────────────────────────────────────
    pending_slot_resave: bool = False

    # ── WindowManager instrumentation (persistent metrics) ───────────────
    wm_fired: bool = False
    wm_msgs_evicted: int = 0
    wm_turns_evicted: int = 0
    wm_summary_ok: bool = False
    wm_emergency_cap: bool = False
    wm_batch_too_small: bool = False
    wm_no_slot: bool = False
    wm_degradation_guard: bool = False

    # ── Hub‑Bodies Tier tracker (cross‑restart stability) ────────────────
    hub_tier_last_modified: Dict[str, int] = Field(default_factory=dict)
    hub_tier_body_hashes: Dict[str, str] = Field(default_factory=dict)
    hub_tier_query_heat: Dict[str, float] = Field(default_factory=dict)
    hub_tier_qids_persisted: List[str] = Field(default_factory=list)

    # ── Persistent compression stubs for large user messages ──────
    compressed_user_messages: Dict[str, str] = Field(default_factory=dict)
    # key: md5 hash of the original message content (16 hex chars)
    # value: the stub text that replaces it in the conversation history

    def reset_wm_metrics(self) -> None:
        """Reset all WindowManager instrumentation flags at the start of each turn."""
        self.wm_fired = False
        self.wm_msgs_evicted = 0
        self.wm_turns_evicted = 0
        self.wm_summary_ok = False
        self.wm_emergency_cap = False
        self.wm_batch_too_small = False
        self.wm_no_slot = False
        self.wm_degradation_guard = False


class ConversationStateManager:
    """
    Centralized manager for persistent conversation state.

    Parallel to ProjectStateManager (volatile per-project state):
    ConversationStateManager handles PERSISTENT state across sessions,
    backed by SQLite through StateStore's write queue.

    Responsibilities:
      - LRU cache of ConversationState per project_id.
      - Load from DB (_load_from_db) with CodeBlock reconstruction and docstring restoration.
      - Async save (_save_to_db) with 2-second debounce.
      - SymbolIndex rebuild after cold load.
      - LRU eviction when max_cached_projects is exceeded.

    Delegated to StateStore:
      - SQLite write queue       → StateStore._db_enqueue()
      - DDL table creation       → StateStore.init_db()
      - Per-project locks        → StateStore.get_project_lock()
      - Edges/path views/docstrings/CFG persistence → StateStore
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the manager with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        self._cache: OrderedDict[str, ConversationState] = OrderedDict()
        self._dirty: Set[str] = set()
        self._last_saved: Dict[str, float] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Public API
    # ═══════════════════════════════════════════════════════════════════════

    def get(self, project_id: str) -> ConversationState:
        """
        Return the ConversationState for the given project.

        Loads from DB if not cached; creates a fresh empty state if not in DB either.

        Replaces: StateStore.get_state(project_id)
        """
        if project_id in self._cache:
            self._cache.move_to_end(project_id)
            return self._cache[project_id]

        state = self._load_from_db(project_id)
        if state is None:
            state = ConversationState()

        self._cache[project_id] = state
        self._cache.move_to_end(project_id)
        self._evict_lru()

        if state.active_blocks:
            self._rebuild_symbol_index(state, project_id)

        return state

    def set(self, project_id: str, state: ConversationState) -> None:
        """
        Update the state in cache and mark it as dirty (pending save).

        Replaces: StateStore.set_state(project_id, state)

        Invariant: In existing code, set() is always preceded by get() for the
        same project_id in the same turn, so the project is already in cache.
        If future code adds a set() without a prior get(), call _evict_lru() here.
        """
        self._cache[project_id] = state
        self._cache.move_to_end(project_id)
        self.mark_dirty(project_id)

    def mark_dirty(self, project_id: str) -> None:
        """Mark the state for a project as modified (pending save)."""
        self._dirty.add(project_id)

    async def save_if_dirty(self, project_id: str) -> None:
        """
        Persist the state if dirty and at least 2 seconds have passed since last save.

        Replaces: StateStore.save_state_if_dirty(project_id)

        Debounce: if called within 2 seconds of the previous save, it returns
        without saving but leaves the dirty flag active, so the next turn will retry.
        To guarantee persistence on shutdown, call _save_to_db_async directly or
        ignore debounce.
        """
        if project_id not in self._dirty:
            return
        now = time.time()
        if now - self._last_saved.get(project_id, 0.0) < 2.0:
            return

        self._last_saved[project_id] = now
        self._dirty.discard(project_id)

        state = self._cache.get(project_id)
        if state is None:
            return

        try:
            await self._save_to_db_async(project_id, state)
        except Exception as e:
            import traceback

            self._f._log_debug(
                f"ConversationStateManager.save_if_dirty: failed for "
                f"'{project_id}': {e}\n{traceback.format_exc()}"
            )
            # Re-mark dirty to retry on the next turn.
            self._dirty.add(project_id)

    def clear_project(self, project_id: str) -> None:
        """Remove the cached state for a project (LRU eviction or project switch)."""
        self._cache.pop(project_id, None)
        self._dirty.discard(project_id)
        self._last_saved.pop(project_id, None)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Load from SQLite
    # ═══════════════════════════════════════════════════════════════════════

    def _load_from_db(self, project_id: str) -> Optional[ConversationState]:
        """
        Load ConversationState from SQLite.

        Migrated from StateStore._load_state_from_db().
        Differences:
          - Returns ConversationState instead of a dict.
          - Uses wm_* aliases for compatibility with pre-Phase-1 DBs (names with '_').
          - Reads history_blocked_age (migrated from pstate in Phase 1).
          - Reads hub_tier_* tracker fields for cross‑restart stability.
          - ALL SQLite reads are serialized with _db_global_lock.
        """
        try:
            with _db_global_lock:
                cur = self._f._db_conn.execute(
                    "SELECT state_json FROM conversation_state WHERE project_id = ?",
                    (project_id,),
                )
                row = cur.fetchone()
        except Exception as e:
            self._f._log_debug(
                f"ConversationStateManager._load_from_db: DB read error "
                f"for '{project_id}': {e}"
            )
            return None

        if not row:
            return None

        try:
            data = json.loads(row[0])
        except Exception as e:
            self._f._log_debug(
                f"ConversationStateManager._load_from_db: invalid JSON: {e}"
            )
            return None

        # Defaults for keys that may be absent in old DBs are handled by the
        # explicit data.get() fallbacks in the constructor below.

        # ── Validate active_blocks ─────────────────────────────────────────
        raw_active = data.get("active_blocks")
        if raw_active is None:
            self._f._log_debug(
                f"⚠️  CORRUPT STATE: 'active_blocks' missing for "
                f"'{project_id}'. Resetting. "
                f"If this persists, delete: {self._f.valves.state_db_path}"
            )
            raw_active = {}
        elif not isinstance(raw_active, dict):
            self._f._log_debug(
                f"⚠️  CORRUPT STATE: 'active_blocks' is "
                f"{type(raw_active).__name__} for '{project_id}'. Resetting."
            )
            raw_active = {}

        # ── Rebuild active_blocks ──────────────────────────────────────────
        active: Dict[str, CodeBlock] = {}
        for k, v in raw_active.items():
            try:
                content_field = v.get("content", "")
                if content_field.startswith("@@hash:"):
                    content_hash = content_field[7:]
                    with _db_global_lock:
                        cur2 = self._f._db_conn.execute(
                            "SELECT content FROM code_contents WHERE hash = ?",
                            (content_hash,),
                        )
                        row2 = cur2.fetchone()
                    v["content"] = (
                        row2[0]
                        if row2
                        else f"[Content not found for hash {content_hash}]"
                    )
                v["content_type"] = (
                    ContentType(v["content_type"])
                    if "content_type" in v
                    else ContentType.GENERAL
                )
                blk = CodeBlock(**v)
                if blk.last_mentioned_msg_idx is None:
                    blk.last_mentioned_msg_idx = data.get("message_count", 0)
                active[k] = blk
            except Exception:
                self._f._log_debug(f"Skipping corrupted block {k} in state DB")

        # ── Rebuild recent and committed lists ─────────────────────────────
        recent: List[CodeBlock] = []
        for b in data.get("recent_changes", []):
            try:
                b["content_type"] = (
                    ContentType(b["content_type"])
                    if "content_type" in b
                    else ContentType.GENERAL
                )
                recent.append(CodeBlock(**b))
            except Exception:
                pass

        committed: List[CodeBlock] = []
        for b in data.get("committed_changes", []):
            try:
                b["content_type"] = (
                    ContentType(b["content_type"])
                    if "content_type" in b
                    else ContentType.GENERAL
                )
                committed.append(CodeBlock(**b))
            except Exception:
                pass

        feedback: List[AppliedChangeFeedback] = []
        for fb in data.get("feedback_history", []):
            try:
                feedback.append(AppliedChangeFeedback(**fb))
            except Exception:
                self._f._log_debug(f"Skipping corrupt feedback entry in '{project_id}'")

        # ── Recalculate token counts ───────────────────────────────────────
        for blk in list(active.values()) + recent + committed:
            if self._f.tokenizer:
                blk._cached_token_count = len(self._f.tokenizer.encode(blk.content))
            else:
                blk._cached_token_count = len(blk.content) // 4

        # ── Restore docstrings from symbol_docstrings table ──────────────
        try:
            with _db_global_lock:
                cur = self._f._db_conn.execute(
                    "SELECT symbol_name, docstring FROM symbol_docstrings "
                    "WHERE project_id = ?",
                    (project_id,),
                )
                rows = cur.fetchall()
            if rows:
                doc_map = {row[0]: row[1] for row in rows}
                for block in active.values():
                    if block.obsolete:
                        continue
                    for sym in block.symbols:
                        if sym.docstring:
                            continue
                        qid = qualify_symbol_name(
                            sym.name, sym.parent_symbol, sym.file_path
                        )
                        doc = doc_map.get(qid) or doc_map.get(sym.name)
                        if doc:
                            sym.docstring = doc
                            self._f._symbol_index.update_docstring(qid, project_id, doc)
        except Exception as e:
            self._f._log_debug(f"_load_from_db: docstring restore failed: {e}")

        # ── Build and return ConversationState ─────────────────────────────
        return ConversationState(
            active_blocks=active,
            recent_changes=recent,
            committed_changes=committed,
            feedback_history=feedback,
            message_count=data.get("message_count", 0),
            last_compression_timestamp=data.get("last_compression_timestamp", 0.0),
            last_suggestion_timestamp=data.get("last_suggestion_timestamp", 0.0),
            has_any_calls=data.get("has_any_calls", False),
            last_cleanup_suggestion_msg_idx=data.get(
                "last_cleanup_suggestion_msg_idx", 0
            ),
            last_cot_level=data.get("last_cot_level", 0),
            conversation_summaries=data.get("conversation_summaries", []),
            summarized_turn_hwm=data.get("summarized_turn_hwm", 0),
            history_blocked_age=data.get("history_blocked_age", {}),
            # Aliases for compatibility with pre-Phase-1 DBs (fields with leading '_')
            wm_fired=data.get("wm_fired", data.get("_wm_fired", False)),
            wm_msgs_evicted=data.get(
                "wm_msgs_evicted", data.get("_wm_msgs_evicted", 0)
            ),
            wm_turns_evicted=data.get(
                "wm_turns_evicted", data.get("_wm_turns_evicted", 0)
            ),
            wm_summary_ok=data.get("wm_summary_ok", data.get("_wm_summary_ok", False)),
            wm_emergency_cap=data.get(
                "wm_emergency_cap", data.get("_wm_emergency_cap", False)
            ),
            wm_batch_too_small=data.get(
                "wm_batch_too_small", data.get("_wm_batch_too_small", False)
            ),
            wm_no_slot=data.get("wm_no_slot", data.get("_wm_no_slot", False)),
            wm_degradation_guard=data.get(
                "wm_degradation_guard",
                data.get("_wm_degradation_guard", False),
            ),
            pending_slot_resave=data.get(
                "pending_slot_resave",
                data.get("_pending_slot_resave", False),
            ),
            # ── Hub‑Bodies Tier tracker (cross‑restart) ──
            hub_tier_last_modified=data.get("hub_tier_last_modified", {}),
            hub_tier_body_hashes=data.get("hub_tier_body_hashes", {}),
            hub_tier_query_heat=data.get("hub_tier_query_heat", {}),
            hub_tier_qids_persisted=data.get("hub_tier_qids_persisted", []),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Save to SQLite
    # ═══════════════════════════════════════════════════════════════════════

    async def _save_to_db_async(
        self, project_id: str, state: ConversationState
    ) -> None:
        """Acquire the project lock and persist the state."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            await self._save_to_db(project_id, state)

    async def _save_to_db(self, project_id: str, state: ConversationState) -> None:
        """
        Serialize ConversationState and persist to SQLite.

        Migrated from StateStore._save_state_to_db().
        Differences:
          - Reads attributes (state.recent_changes) instead of dict keys.
          - Persists summarized_turn_hwm, history_blocked_age, and all wm_* fields.
          - Persists hub_tier_* tracker fields for cross‑restart stability.
          - The @@hash trick for code_contents remains unchanged.
        """
        # ── Serialize active_blocks, externalizing content ─────────────────
        active_blocks_meta: Dict[str, Any] = {}
        for k, v in state.active_blocks.items():
            d = v.dict()
            d["content_type"] = v.content_type.value
            content_hash = v.hash
            await self._f._state_store._db_enqueue(
                lambda ch=content_hash, ct=v.content: self._f._db_conn.execute(
                    "INSERT OR IGNORE INTO code_contents "
                    "(hash, content, created_at) VALUES (?, ?, ?)",
                    (ch, ct, time.time()),
                )
            )
            d["content"] = f"@@hash:{content_hash}"
            active_blocks_meta[k] = d

        serializable = {
            "active_blocks": active_blocks_meta,
            "recent_changes": [b.dict() for b in state.recent_changes],
            "committed_changes": [b.dict() for b in state.committed_changes],
            "feedback_history": [fb.dict() for fb in state.feedback_history],
            "message_count": state.message_count,
            "last_compression_timestamp": state.last_compression_timestamp,
            "last_suggestion_timestamp": state.last_suggestion_timestamp,
            "last_cleanup_suggestion_msg_idx": (state.last_cleanup_suggestion_msg_idx),
            "has_any_calls": state.has_any_calls,
            "last_cot_level": state.last_cot_level,
            "conversation_summaries": state.conversation_summaries,
            "summarized_turn_hwm": state.summarized_turn_hwm,
            "history_blocked_age": state.history_blocked_age,
            "wm_fired": state.wm_fired,
            "wm_msgs_evicted": state.wm_msgs_evicted,
            "wm_turns_evicted": state.wm_turns_evicted,
            "wm_summary_ok": state.wm_summary_ok,
            "wm_emergency_cap": state.wm_emergency_cap,
            "wm_batch_too_small": state.wm_batch_too_small,
            "wm_no_slot": state.wm_no_slot,
            "wm_degradation_guard": state.wm_degradation_guard,
            "pending_slot_resave": state.pending_slot_resave,
            # ── Hub‑Bodies Tier tracker (cross‑restart) ──
            "hub_tier_last_modified": state.hub_tier_last_modified,
            "hub_tier_body_hashes": state.hub_tier_body_hashes,
            "hub_tier_query_heat": state.hub_tier_query_heat,
            "hub_tier_qids_persisted": state.hub_tier_qids_persisted,
        }

        def _write() -> None:
            self._f._db_conn.execute(
                "REPLACE INTO conversation_state "
                "(project_id, state_json, updated_at) VALUES (?, ?, ?)",
                (project_id, json.dumps(serializable), time.time()),
            )
            self._f._db_conn.commit()

        await self._f._state_store._db_enqueue(_write)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. SymbolIndex rebuild after cold load
    # ═══════════════════════════════════════════════════════════════════════

    def _rebuild_symbol_index(self, state: ConversationState, project_id: str) -> None:
        """
        Rebuild SymbolIndex from active blocks.

        Migrated from StateStore._rebuild_symbol_index().
        No logic changes; only attribute access (state.active_blocks).

        Invariant: The index for project_id must be empty before calling this
        to avoid duplicates. get() guarantees this because it is only called
        on a cache miss (cold load), when the index for that project has not
        been populated yet. If called elsewhere, call clear_project() first.
        """
        for block in state.active_blocks.values():
            if block.obsolete:
                continue
            for sym in block.symbols:
                self._f._symbol_index.add(sym, block.hash, project_id)
                caller_qid = qualify_symbol_name(
                    sym.name, sym.parent_symbol, sym.file_path
                )
                for callee in sym.calls:
                    edge = Edge(
                        src=caller_qid,
                        dst=callee,
                        type="calls",
                        weight=EDGE_WEIGHTS["calls"],
                    )
                    self._f._symbol_index.add_edge(edge, project_id)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. LRU eviction
    # ═══════════════════════════════════════════════════════════════════════

    def _evict_lru(self) -> None:
        """
        Evict the least recently used projects when max_cached_projects is exceeded.

        Migrated from the while loop in StateStore.get_state().
        Centralised here so both get() and set() can trigger it.

        IMPORTANT: Flushes dirty state before eviction to avoid data loss.
        The blocking DB write is offloaded to the thread pool with a timeout
        to prevent event-loop stalls. If the write times out or fails, the
        state may be lost (logged) but eviction proceeds to keep the cache
        within bounds.
        """
        max_cached = self._f.valves.max_cached_projects
        while len(self._cache) > max_cached:
            oldest_pid, oldest_state = next(iter(self._cache.items()))

            # ── Flush dirty state before eviction ─────────────────────────
            if oldest_pid in self._dirty:
                try:
                    # Offload blocking DB write to the thread pool
                    # so the event loop is not stalled during eviction.
                    import concurrent.futures

                    future = self._f._db_executor.submit(
                        self._f._state_store._db_conn_write_sync,
                        oldest_pid,
                        oldest_state,
                    )
                    # Wait with a timeout to prevent indefinite blocking
                    future.result(timeout=10.0)
                    self._f._log_debug(
                        f"ConversationStateManager: flushed dirty state for "
                        f"'{oldest_pid}' before LRU eviction"
                    )
                except concurrent.futures.TimeoutError:
                    self._f._log_debug(
                        f"ConversationStateManager: flush timed out for "
                        f"'{oldest_pid}' — state may be lost"
                    )
                except AttributeError:
                    self._f._log_debug(
                        f"ConversationStateManager: StateStore._db_conn_write_sync "
                        f"not implemented; dirty state for '{oldest_pid}' may be lost."
                    )
                except Exception as e:
                    self._f._log_debug(
                        f"ConversationStateManager: flush before eviction failed "
                        f"for '{oldest_pid}': {e} — state may be lost"
                    )

            # ── Clear all per-project state ────────────────────────────────
            self._f._symbol_index.clear_project(oldest_pid)
            self._f._path_index.clear_project(oldest_pid)
            self._f._pager.clear_project(oldest_pid)
            self._f._project_state_manager.clear_project(oldest_pid)
            self.clear_project(oldest_pid)
            self._f._log_debug(
                f"ConversationStateManager: evicted LRU project '{oldest_pid}'"
            )


class ActivationGraph:
    """Personalised PageRank (PPR) engine for the symbol call graph.

    Seeds are set from user-query terms, tracebacks, and recent history.
    ``propagate()`` runs the PPR power iteration over the directed edges
    stored in ``SymbolIndex``, spreading activation to related symbols.
    The resulting scores determine the LOD tiers in Block B.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Constants & initialization
    # ═══════════════════════════════════════════════════════════════════════════

    DECAY_BASE: float = 0.7

    def __init__(self):
        self._activations: Dict[str, ActivationState] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Seed activation
    # ═══════════════════════════════════════════════════════════════════════════

    def seed(self, node_ids: List[str], initial_score: float = 1.0):
        """Insert activation seeds.  Called once before ``propagate()``."""
        for nid in node_ids:
            self._activations[nid] = ActivationState(
                node_id=nid,
                score=initial_score,
                depth=0,
                source="seed",
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. PPR propagation
    # ═══════════════════════════════════════════════════════════════════════════

    def propagate(
        self,
        edges_out: Dict[str, List[Edge]],
        max_steps: int = 20,
        min_score: float = 0.05,
        alpha: float = 0.85,
        tolerance: float = 1e-6,
    ):
        """
        Run the PPR power iteration until convergence or ``max_steps``.

        Seeds keep their original high score (they are not overwritten by the
        converged PPR score). Propagated nodes receive the PPR score as usual.
        """
        if not self._activations:
            return
        seed_total = sum(
            s.score for s in self._activations.values() if s.source == "seed"
        )
        if seed_total == 0:
            return
        personalization: Dict[str, float] = {
            nid: s.score / seed_total
            for nid, s in self._activations.items()
            if s.source == "seed"
        }
        out_weight_total: Dict[str, float] = {}
        for src, edges in edges_out.items():
            total_w = sum(e.effective_weight() for e in edges)
            out_weight_total[src] = total_w if total_w > 0 else 1.0
        r: Dict[str, float] = dict(personalization)
        for iteration in range(max_steps):
            r_new: Dict[str, float] = {}
            for node, score in personalization.items():
                r_new[node] = (1.0 - alpha) * score
            for src, edges in edges_out.items():
                src_score = r.get(src, 0.0)
                if src_score < min_score:
                    continue
                out_w = out_weight_total.get(src, 1.0)
                for edge in edges:
                    contribution = alpha * src_score * edge.effective_weight() / out_w
                    r_new[edge.dst] = r_new.get(edge.dst, 0.0) + contribution
            all_keys = set(r.keys()) | set(r_new.keys())
            delta = sum(abs(r_new.get(k, 0.0) - r.get(k, 0.0)) for k in all_keys)
            r = r_new
            if delta < tolerance:
                break

        # Update activations: seeds keep their original score, propagated nodes get the PPR score
        for node_id, score in r.items():
            if score < min_score:
                continue
            existing = self._activations.get(node_id)
            # Seeds keep their initial high score — PPR only lowers propagated neighbors
            final_score = (
                max(score, existing.score)
                if (existing and existing.source == "seed")
                else score
            )
            self._activations[node_id] = ActivationState(
                node_id=node_id,
                score=final_score,
                depth=existing.depth if existing else 99,
                source=existing.source if existing else "propagation",
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Query methods
    # ═══════════════════════════════════════════════════════════════════════════

    def get_score(self, node_id: str) -> float:
        """Return the final activation score of a node (0.0 if not activated)."""
        state = self._activations.get(node_id)
        return state.score if state else 0.0

    def get_activated_nodes(self, threshold: float = 0.1) -> Dict[str, float]:
        """Return {node_id: score} for nodes whose score >= threshold."""
        return {
            nid: s.score for nid, s in self._activations.items() if s.score >= threshold
        }

    def get_seed_nodes(self) -> List[str]:
        """
        Return node_ids that were seeded directly (source == 'seed'), as
        opposed to nodes that only received score via PPR propagation. Used
        by the case-D (refactor) caller pull-in: only the literal seeds —
        the symbols the user is actually asking about — should gain forced
        caller visibility, not every propagated neighbor.
        """
        return [nid for nid, s in self._activations.items() if s.source == "seed"]

    def aggregate_path_score(self, symbol_list: List[str]) -> float:
        """Mean activation score of a list of symbols (ignoring inactive ones)."""
        scores = [self.get_score(s) for s in symbol_list]
        active = [s for s in scores if s > 0]
        if not active:
            return 0.0
        return sum(active) / len(active)


# ---------------------------------------------------------------------------
# Query model and SubgraphExtractor skeleton
# ---------------------------------------------------------------------------
class SubgraphExtractor:
    """Extract a connected subgraph induced by activated nodes.

    Takes an ``ActivationGraph`` and the full edge set, then returns
    the subset of nodes that are activated (above threshold) and the
    edges among them.  Optionally expands the subgraph by one hop along
    high-confidence ``calls`` edges.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, activation_threshold: float = 0.1, expand_hops: int = 1):
        self.activation_threshold = activation_threshold
        self.expand_hops = expand_hops

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Extraction
    # ═══════════════════════════════════════════════════════════════════════════

    def extract(
        self,
        activation: ActivationGraph,
        edges_out: Dict[str, List[Edge]],
        edges_in: Dict[str, List[Edge]],
    ) -> Tuple[Set[str], List[Edge]]:
        """Return (activated_nodes, edges_among_them) after optional expansion."""
        activated = activation.get_activated_nodes(self.activation_threshold)
        included_nodes: Set[str] = set(activated.keys())
        if self.expand_hops > 0:
            expansion_candidates = []
            for node_id in list(included_nodes):
                for edge in edges_out.get(node_id, []):
                    if (
                        edge.dst not in included_nodes
                        and edge.effective_weight() >= 0.8
                        and edge.type == "calls"
                    ):
                        expansion_candidates.append(edge.dst)
            included_nodes.update(expansion_candidates)
        included_edges: List[Edge] = []
        for node_id in included_nodes:
            for edge in edges_out.get(node_id, []):
                if edge.dst in included_nodes:
                    included_edges.append(edge)
        return included_nodes, included_edges


# ---------------------------------------------------------------------------
# CodePathView — a cached projection of an activated subgraph
# ---------------------------------------------------------------------------
class CodePathView(BaseModel):
    """A cached snapshot of an activated subgraph, used for speculative
    prefetch and path tracking.  Holds the induced nodes (with scores),
    edges, and structural hashes so staleness can be detected cheaply."""

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Core identity & structural hashes
    # ═══════════════════════════════════════════════════════════════════════════

    path_id: str
    entry_point: str
    seed_nodes: List[str]

    structural_hash: str = ""  # Hash of block content for induced nodes
    call_graph_hash: str = ""  # Hash of call relationships among them

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Induced subgraph data
    # ═══════════════════════════════════════════════════════════════════════════

    induced_nodes: Dict[str, float]  # node_id → activation score
    induced_edges: List[Edge]
    activation_score: float

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Business metadata (LLM-generated labels)
    # ═══════════════════════════════════════════════════════════════════════════

    business_label: str = ""
    summary: str = ""
    label_confidence: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Timestamp
    # ═══════════════════════════════════════════════════════════════════════════

    last_built: float = Field(default_factory=time.time)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def is_stale(self, current_structural: str, current_call_graph: str) -> bool:
        """True if the code or call graph has changed since this view was built."""
        return (
            self.structural_hash != current_structural
            or self.call_graph_hash != current_call_graph
        )

    def top_symbols(self, n: int = 10) -> List[str]:
        """Return the *n* symbols with the highest activation score in this view."""
        return sorted(
            self.induced_nodes.keys(),
            key=lambda s: self.induced_nodes[s],
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# StaticEvidence – deterministic proof from the SymbolGraph
# ---------------------------------------------------------------------------
class StaticEvidence(BaseModel):
    """Deterministic evidence gathered from the SymbolGraph to validate a
    hypothesis during scientific Chain‑of‑Thought reasoning.

    All fields are derived without an LLM call — they come directly from the
    SymbolIndex, active blocks, and path index.  ``objective_score`` is the
    fraction of verifiable claims that hold true."""

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Symbol evidence & call relations
    # ═══════════════════════════════════════════════════════════════════════════

    symbols_found: Dict[str, bool]  # Is each mentioned symbol in the index?
    call_relations_valid: Dict[str, bool]  # Are claimed call edges actually present?

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Recent changes & entry points
    # ═══════════════════════════════════════════════════════════════════════════

    recent_changes: List[str]  # Mentioned symbols changed in the last hour
    entry_points_mentioned: List[str]  # Entry points referenced in the hypothesis

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Path memberships & data flow
    # ═══════════════════════════════════════════════════════════════════════════

    path_memberships: Dict[str, List[str]]  # Path views each symbol belongs to
    data_flow_upstream: Dict[str, List[str]] = Field(default_factory=dict)

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Objective score
    # ═══════════════════════════════════════════════════════════════════════════

    objective_score: float  # Fraction of verifiable claims that hold


# ---------------------------------------------------------------------------
# PathIndex — index of CodePathViews
# ---------------------------------------------------------------------------
class PathIndex:
    """Lightweight in‑memory index of ``CodePathView`` objects, organised by
    project and symbol.  Used for speculative prefetch, staleness detection,
    and entry‑point discovery.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self):
        self._views: Dict[str, CodePathView] = {}
        self._symbol_to_views: Dict[str, Set[str]] = defaultdict(set)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. View management (add, remove, get)
    # ═══════════════════════════════════════════════════════════════════════════

    def add(self, view: CodePathView, project_id: str):
        """Register a view and cross‑reference all its induced symbols."""
        key = f"{project_id}:{view.path_id}"
        self._views[key] = view
        for sym_name in view.induced_nodes:
            self._symbol_to_views[f"{project_id}:{sym_name}"].add(view.path_id)

    def remove(self, path_id: str, project_id: str):
        """Remove a view and its symbol cross‑references."""
        key = f"{project_id}:{path_id}"
        view = self._views.pop(key, None)
        if view:
            for sym_name in view.induced_nodes:
                sym_key = f"{project_id}:{sym_name}"
                self._symbol_to_views[sym_key].discard(path_id)

    def get(self, path_id: str, project_id: str) -> Optional[CodePathView]:
        """Return a single view by path_id, or None."""
        return self._views.get(f"{project_id}:{path_id}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Project-level queries & cleanup
    # ═══════════════════════════════════════════════════════════════════════════

    def get_all(self, project_id: str) -> List[CodePathView]:
        """All views for a project (order undefined)."""
        prefix = f"{project_id}:"
        return [v for k, v in self._views.items() if k.startswith(prefix)]

    def clear_project(self, project_id: str):
        """Drop every view and cross‑reference for a project."""
        prefix = f"{project_id}:"
        keys = [k for k in self._views if k.startswith(prefix)]
        for k in keys:
            del self._views[k]
        sym_keys = [k for k in self._symbol_to_views if k.startswith(prefix)]
        for k in sym_keys:
            del self._symbol_to_views[k]

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Symbol-level queries
    # ═══════════════════════════════════════════════════════════════════════════

    def mark_stale_for_symbol(self, symbol_name: str, project_id: str) -> List[str]:
        """Return path_ids that reference a given symbol (to invalidate them later)."""
        key = f"{project_id}:{symbol_name}"
        return list(self._symbol_to_views.get(key, set()))

    def find_entry_points(
        self, symbol_index: "SymbolIndex", project_id: str
    ) -> Set[str]:
        """Symbols with no known caller — used as activation seeds when a
        query matches nothing in the index.  Operates on QUALIFIED ids so
        that, for example, a class's only method that is called exclusively
        from outside is not confused with another class's same-named method
        that DOES have registered callers."""
        all_qids = symbol_index.get_all_qualified_names(project_id)
        result = set()
        for qid in all_qids:
            meta = symbol_index.get_symbol_meta(qid, project_id) or {}
            bare = meta.get("name", qid.rsplit(".", 1)[-1])
            if not symbol_index.get_callers(bare, project_id):
                result.add(qid)
        return result


# ---------------------------------------------------------------------------
# AppliedChangeFeedback
# ---------------------------------------------------------------------------
class AppliedChangeFeedback(BaseModel):
    """Feedback record for a change that was applied (or rejected) by the user.

    Kept in the conversation state so the system can learn from past changes
    and surface context in Block A when ``inject_feedback_context`` is enabled.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Change identity & description
    # ═══════════════════════════════════════════════════════════════════════════

    change_hash: str
    change_description: str

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Location
    # ═══════════════════════════════════════════════════════════════════════════

    file_path: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Status & metadata (original order: timestamp, success, user_comment, resolved)
    # ═══════════════════════════════════════════════════════════════════════════

    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


# ---------------------------------------------------------------------------
# Reranker singleton factory (module level)
# ---------------------------------------------------------------------------
def _get_cross_encoder(
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
) -> Optional[Any]:
    """Return the CrossEncoder singleton, loading it once. Thread‑safe."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        with _CROSS_ENCODER_LOCK:
            if _CROSS_ENCODER is None:
                try:
                    from sentence_transformers import CrossEncoder

                    _CROSS_ENCODER = CrossEncoder(model_name)
                except Exception:
                    return None
    return _CROSS_ENCODER


# ═══════════════════════════════════════════════════════════════════════════
# CLASSES — Module level, before class Filter
# ═══════════════════════════════════════════════════════════════════════════


class HubSymbolIndex:
    """
    Build the architecture map for Block A: a compact class→members outline,
    plus, for the hub symbols (top-N by call-graph centrality), a
    bidirectional view of who calls them and whom they call.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Public API
    # ═══════════════════════════════════════════════════════════════════════════

    def get_hub_names(self, centrality: dict, top_n: int) -> list:
        """Return symbol ids sorted by descending centrality, capped at *top_n*.
        Ties are broken alphabetically for deterministic, cache-stable output."""
        if not centrality or top_n <= 0:
            return []
        ranked = sorted(centrality.items(), key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in ranked[:top_n]]

    def is_hub(self, symbol_name: str, centrality: dict, top_n: int) -> bool:
        """True if *symbol_name* would appear in Block A for the given *top_n*."""
        return symbol_name in set(self.get_hub_names(centrality, top_n))

    def build(
        self,
        symbol_index: "SymbolIndex",
        centrality: dict,
        project_id: str,
        top_n: int = 30,
        valves=None,
        mode: str = "hubs_only",
    ) -> str:
        """
        Build the full Block A symbol text: class outline (unchanged across
        modes) + a call-graph section whose depth depends on *mode*:

          hubs_only      → existing behavior: top_n hubs by centrality only.
          expanded_hubs  → top_n hubs + every direct caller/callee of each hub
                           (depth 1, deduplicated, non-hub only).
          full_graph     → every qualified symbol with its direct
                           callers/callees, alphabetically sorted.

        *mode* is resolved by the CALLER (ContextBuilder._resolve_call_graph_mode)
        and passed in already-decided — this method does no query/intent logic,
        it only renders. Deterministic while the code is unchanged (alphabetical /
        centrality order), so llama.cpp's KV cache stays stable.
        """
        logger.debug(
            f"HubSymbolIndex.build: rendering mode='{mode}' for project='{project_id}'"
        )

        sections = []

        outline = self._build_class_outline(symbol_index, project_id, valves)
        if outline:
            sections.append(outline)

        if mode == "expanded_hubs":
            section = self._build_expanded_hubs_section(
                symbol_index, centrality, project_id, top_n, valves
            )
            if section:
                sections.append(section)
        elif mode == "full_graph":
            section = self._build_full_graph_section(symbol_index, project_id, valves)
            if section:
                sections.append(section)
        else:
            # "hubs_only" and any unexpected value: render the stable hub
            # section. Defensive fallback for an invalid valve never crashes
            # and never renders nothing.
            hub_qids = self.get_hub_names(centrality, top_n)
            if hub_qids:
                enable_callees = (
                    getattr(valves, "enable_hub_callees", True) if valves else True
                )
                sections.append(
                    self._build_hub_section(
                        hub_qids, centrality, symbol_index, project_id, enable_callees
                    )
                )

        return "\n\n".join(s for s in sections if s.strip())

    def _build_expanded_hubs_section(
        self, symbol_index, centrality, project_id, top_n, valves=None
    ) -> str:
        """
        Top-N hubs (rendered exactly as in hubs_only) PLUS a sub-section
        listing every direct neighbor (caller or callee) of any hub that is
        not itself a hub — one hop beyond the hub set without the full
        O(all symbols) cost.

        Neighbors are a flat qid list grouped by hub (NOT full
        _format_symbol_line entries — that would recursively pull in THEIR
        neighbors and balloon token cost, blowing the ~2k-5k budget).
        Truncated by expanded_hubs_max_tokens with an explicit notice.
        """
        enable_callees = getattr(valves, "enable_hub_callees", True) if valves else True
        hub_qids = self.get_hub_names(centrality, top_n)
        if not hub_qids:
            logger.debug("Expanded hubs: no hubs found, returning empty.")
            return ""

        hub_section = self._build_hub_section(
            hub_qids, centrality, symbol_index, project_id, enable_callees
        )

        hub_set = set(hub_qids)
        neighbor_lines = []
        for qid in hub_qids:
            callers = self._safe_callers(qid, project_id, symbol_index) - hub_set
            callees = self._safe_callees(qid, project_id, symbol_index) - hub_set
            extra = sorted(callers | callees)
            if extra:
                neighbor_lines.append(f"- `{qid}` neighbors: {', '.join(extra)}")

        if not neighbor_lines:
            logger.debug("Expanded hubs: no neighbor lines to add.")
            return hub_section

        budget_chars = self.valves_expanded_hubs_budget_chars(valves)
        kept_lines = []
        total = 0
        omitted = 0
        for idx, line in enumerate(neighbor_lines):
            total += len(line)
            if budget_chars and total > budget_chars:
                omitted = len(neighbor_lines) - idx
                kept_lines.append(
                    f"_(Neighbor list truncated to fit budget — "
                    f"{omitted} hub(s) omitted)_"
                )
                break
            kept_lines.append(line)

        logger.debug(
            f"Expanded hubs: {len(hub_qids)} hubs, {len(neighbor_lines)} neighbor entries total, "
            f"{len(kept_lines) - (1 if omitted else 0)} shown, {omitted} omitted due to budget."
        )

        neighbor_section = (
            "### Direct neighbors of hub symbols (depth 1, non-hub only)\n"
            + "\n".join(kept_lines)
        )
        return hub_section + "\n\n" + neighbor_section

    @staticmethod
    def valves_expanded_hubs_budget_chars(valves) -> int:
        """expanded_hubs_max_tokens → char budget (4 chars/token heuristic).
        Returns 0 (no limit) if valves is None or the field is absent."""
        if valves is None:
            return 0
        tok = getattr(valves, "expanded_hubs_max_tokens", 0)
        return tok * 4 if tok > 0 else 0

    def _build_full_graph_section(self, symbol_index, project_id, valves=None) -> str:
        """
        Every qualified symbol in the project, alphabetically sorted, each with
        its direct callers/callees (no centrality score — in full_graph every
        symbol is shown regardless of rank, so a score column is misleading
        noise).

        Alphabetical order (not centrality order) is deliberate: centrality
        ties at 0.0 for every leaf symbol, so centrality-based ordering is not
        stable across rebuilds and would cause spurious Block A cache misses
        from pure reordering. Alphabetical order is stable as long as the
        symbol set is unchanged — exactly what Block A's code_state_hash key
        already tracks.

        Hard-truncates at full_graph_max_tokens with an explicit notice. Never
        falls back to a different mode — truncation is the only degradation path.
        """
        all_qids = sorted(symbol_index.get_all_qualified_names(project_id))
        if not all_qids:
            logger.debug("Full graph: no symbols found, returning empty.")
            return ""

        max_tokens = (
            getattr(valves, "full_graph_max_tokens", 20000) if valves else 20000
        )
        budget_chars = max_tokens * 4 if max_tokens > 0 else None

        # Defense in depth: a manual override of full_graph bypasses the
        # auto-mode symbol-count ceiling entirely. Cap the candidate list
        # itself (not just the rendered text) so an oversized manually-forced
        # project never allocates per-symbol caller/callee sets it will
        # immediately discard via truncation. ~4 chars/token, ~80 chars/line
        # average for a symbol-with-edges line ⇒ budget_chars // 80 is a safe
        # upper bound on how many lines could possibly fit regardless of
        # actual edge density.
        if budget_chars is not None:
            max_renderable_lines = max(10, budget_chars // 80)
            if len(all_qids) > max_renderable_lines:
                original_count = len(all_qids)
                all_qids = all_qids[:max_renderable_lines]
                logger.debug(
                    f"Full graph: capped candidate list from {original_count} to "
                    f"{max_renderable_lines} (budget {max_tokens} tokens)"
                )

        logger.debug(
            f"Full graph: rendering {len(all_qids)} symbols, budget {max_tokens} tokens."
        )

        lines = [
            f"## Full Call Graph (all {len(all_qids)} symbols, direct edges only)",
            "_Every indexed symbol with its direct callers/callees. "
            "Use `/expand <name>` for full bodies._",
            "",
        ]
        total_chars = sum(len(l) for l in lines)

        truncated = False
        for idx, qid in enumerate(all_qids):
            line = self._format_symbol_line_no_score(qid, project_id, symbol_index)
            if budget_chars is not None and total_chars + len(line) > budget_chars:
                remaining = len(all_qids) - idx
                lines.append(
                    f"\n_(Truncated to fit {max_tokens}-token budget — "
                    f"{remaining} symbol(s) omitted. Consider expanded_hubs "
                    f"or raising full_graph_max_tokens.)_"
                )
                truncated = True
                logger.debug(
                    f"Full graph truncated: kept {idx} symbols out of {len(all_qids)}, "
                    f"omitted {remaining} due to budget {max_tokens} tokens."
                )
                break
            lines.append(line)
            total_chars += len(line)

        if not truncated:
            logger.debug(
                f"Full graph: successfully rendered all {len(all_qids)} symbols."
            )

        return "\n".join(lines)

    def _format_symbol_line_no_score(self, qid, project_id, symbol_index) -> str:
        """
        Same as _format_symbol_line but without the centrality score column.
        full_graph forces callees on: showing the bidirectional edge set is the
        whole point of this mode.

        Caller attribution caveat: get_edges_in() resolves by bare name —
        a method shared across multiple classes (every __init__, etc.) shows
        the union of callers of ANY same-named method, not specifically this
        occurrence. Lines affected by this are marked '(ambiguous: shared name)'
        so the model — and a human reading the dump — doesn't treat the
        caller list as precise for those symbols.

        All callers and callees are shown in full, without truncation.
        """
        bare_name = qid.rsplit(".", 1)[-1]
        is_ambiguous_name = (
            len(symbol_index.get_qualified_names_for(bare_name, project_id)) > 1
        )

        callers = self._safe_callers(qid, project_id, symbol_index)
        callees = self._safe_callees(qid, project_id, symbol_index)

        parts = [f"- `{qid}`"]

        if callers:
            tag = " (ambiguous: shared name)" if is_ambiguous_name else ""
            parts.append(f"\n  ← used by{tag}: {', '.join(sorted(callers))}")

        if callees:
            parts.append(f"\n  → calls: {', '.join(sorted(callees))}")

        return "".join(parts)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Class / function outline (Architecture Map)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_class_outline(self, symbol_index, project_id, valves=None) -> str:
        """Render the ``## Code Architecture Map`` section: one line per class
        listing its methods, plus module-level functions if any.  Respects
        ``architecture_map_max_tokens`` for the overall section budget.
        """
        if valves is not None and not getattr(valves, "enable_architecture_map", True):
            return ""
        max_tokens = (
            getattr(valves, "architecture_map_max_tokens", 3000) if valves else 3000
        )

        classes = sorted(symbol_index.get_classes(project_id))
        if not classes:
            return ""

        lines = [
            "## Code Architecture Map",
            "_Class → methods. See the hub section below for call relationships "
            "of the most central symbols. Use `/expand <name>` or "
            "`/expand Class.method` for full bodies._",
            "",
        ]
        total_chars = sum(len(l) for l in lines)
        budget_chars = max_tokens * 4 if max_tokens > 0 else None
        truncated = False

        for class_name in classes:
            member_qids = symbol_index.get_class_members(class_name, project_id)
            if not member_qids:
                continue
            bare_members = []
            for qid in member_qids:
                meta = symbol_index.get_symbol_meta(qid, project_id) or {}
                bare_members.append(meta.get("name", qid.rsplit(".", 1)[-1]))
            line = (
                f"- **{class_name}** ({len(bare_members)} methods): "
                f"{', '.join(bare_members)}"
            )
            if budget_chars is not None and total_chars + len(line) > budget_chars:
                lines.append(
                    f"_(Outline truncated to fit budget — {len(classes)} classes total)_"
                )
                truncated = True
                break
            lines.append(line)
            total_chars += len(line)

        if not truncated:
            top_level = sorted(
                qid
                for qid in symbol_index.get_all_qualified_names(project_id)
                if "." not in qid
                and (symbol_index.get_symbol_meta(qid, project_id) or {}).get("kind")
                == "function"
            )
            if top_level:
                line = f"- **(module-level functions)**: {', '.join(top_level)}"
                if budget_chars is None or total_chars + len(line) <= budget_chars:
                    lines.append(line)

        if len(lines) <= 3:
            return ""
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Hub symbols with bidirectional call graph
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_hub_section(
        self, hub_qids, centrality, symbol_index, project_id, enable_callees=True
    ) -> str:
        """Render the ``## Code Symbol Index`` section listing each hub symbol
        with its centrality score, incoming callers, and (when
        *enable_callees* is True) outgoing callees."""
        by_file: dict = {}
        for qid in hub_qids:
            file_path = self._file_for(qid, project_id, symbol_index)
            by_file.setdefault(file_path, []).append(qid)

        lines = [
            f"## Code Symbol Index — Hub Symbols (top {len(hub_qids)} by call-graph centrality)",
            "> Remaining symbols are available via LOD activation. "
            "Use /expand <name> for any symbol's full body.",
            "",
        ]

        if len(by_file) == 1 and None in by_file:
            for qid in sorted(hub_qids, key=lambda q: -centrality.get(q, 0)):
                lines.append(
                    self._format_symbol_line(
                        qid, centrality, symbol_index, project_id, enable_callees
                    )
                )
        else:
            for file_path in sorted(by_file.keys(), key=lambda fp: (fp is None, fp)):
                if file_path is None:
                    continue
                lines.append(f"### {file_path}")
                for qid in sorted(
                    by_file[file_path], key=lambda q: -centrality.get(q, 0)
                ):
                    lines.append(
                        self._format_symbol_line(
                            qid, centrality, symbol_index, project_id, enable_callees
                        )
                    )
                lines.append("")

        lines.append(
            "To see any symbol's full body, mention it in your message "
            "or use /expand <name>."
        )
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Private helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _file_for(self, qid, project_id, symbol_index):
        """Resolve a symbol's file path from the SymbolIndex, or None."""
        return symbol_index.get_file_for_symbol(qid, project_id)

    def _safe_callers(self, qid, project_id, symbol_index) -> set:
        """Callers of `qid`, looked up by its BARE name — edge destinations
        are best-effort bare strings (see SymbolIndex docstring), so this is
        the only reliable lookup direction, with the known limitation that a
        very common bare name (e.g. __init__) will show callers from any
        method of that same name, not only this one."""
        meta = symbol_index.get_symbol_meta(qid, project_id) or {}
        bare = meta.get("name", qid.rsplit(".", 1)[-1])
        fn = getattr(symbol_index, "get_edges_in", None)
        if not callable(fn):
            return set()
        try:
            edges = fn(bare, project_id)
            return {e.src for e in edges}
        except Exception:
            return set()

    def _safe_callees(self, qid, project_id, symbol_index) -> set:
        """Callees of `qid`, looked up by its qualified id — precise, since
        edge sources are always the exact symbol in whose body the call was
        found."""
        fn = getattr(symbol_index, "get_edges_out", None)
        if not callable(fn):
            return set()
        try:
            edges = fn(qid, project_id)
            return {e.dst for e in edges}
        except Exception:
            return set()

    def _format_symbol_line(
        self, qid, centrality, symbol_index, project_id, enable_callees=True
    ) -> str:
        """
        Format one hub-symbol line: ``- `qid` (centrality: score)``
        optionally followed by ``← used by:`` (all callers) and
        ``→ calls:`` (all callees, when *enable_callees* is True).

        All callers and callees are shown in full, without truncation.
        """
        score = centrality.get(qid, 0.0)
        callers = self._safe_callers(qid, project_id, symbol_index)

        parts = [f"- `{qid}` (centrality: {score:.2f})"]

        if callers:
            parts.append(f"\n  ← used by: {', '.join(sorted(callers))}")

        if enable_callees:
            callees = self._safe_callees(qid, project_id, symbol_index)
            if callees:
                parts.append(f"\n  → calls: {', '.join(sorted(callees))}")

        return "".join(parts)


class ContextPager:
    """
    Manages CodeBlock lifecycle between active_blocks (RAM) and ChromaDB (paged).
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization & state management
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, filter_ref: "Filter") -> None:
        # Back-reference to the parent Filter. Only purge_old_versions() needs
        # it (for valves + logging); the page-in/page-out paths are self-
        # contained. Kept as a deliberate back-reference rather than passing
        # the filter through every call.
        self._f = filter_ref
        # project_id → set of block hashes currently paged out.
        self._paged_hashes: dict = {}
        # Bounds concurrent background embedding tasks during page-out so a
        # mass eviction can't spawn one embedder.encode per block at once.
        # valves already exist at this point (Filter.__init__ sets self.valves
        # before constructing the pager). Semaphore creation does not bind to
        # an event loop until first use, so building it here is safe.
        self._page_out_semaphore = asyncio.Semaphore(
            max(
                1,
                getattr(
                    filter_ref.valves,
                    "block_paging_max_concurrent_embeddings",
                    2,
                ),
            )
        )

    def is_paged(self, block_hash: str, project_id: str) -> bool:
        """True if block_hash has been paged out to ChromaDB for this project."""
        return block_hash in self._paged_hashes.get(project_id, set())

    def clear_project(self, project_id: str) -> None:
        """Drop the in-memory paged registry for a project (on project switch)."""
        self._paged_hashes.pop(project_id, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Eviction candidate selection & page-out
    # ═══════════════════════════════════════════════════════════════════════════

    def get_eviction_candidates(
        self,
        state: ConversationState,
        project_id: str,
        activation_scores: dict,
        paging_threshold: int,
        min_activation: float,
    ) -> list:
        """
        Return block hashes eligible for page-out.

        Criteria: len(active_blocks) > paging_threshold AND
                  block_activation < min_activation.
        Pinned blocks are never candidates.

        activation_scores is keyed by SYMBOL name (from
        Filter._last_activation_scores). Block-level aggregation uses the MAX
        of its symbols' scores — any hot symbol keeps the whole block in RAM:
            block_activation = max(scores.get(s.name, 0.0)
                                   for s in block.symbols) or 0.0

        Args:
            state: The current conversation state (ConversationState).
            project_id: Current project identifier.
            activation_scores: Dict mapping qualified symbol ids to activation scores.
            paging_threshold: Active block count above which paging starts.
            min_activation: Minimum activation score to keep a block in RAM.

        Returns:
            A list of block hashes eligible for page-out, sorted by
            (activation, importance) ascending (coldest first).
        """
        active = state.active_blocks
        if len(active) <= paging_threshold:
            return []

        # ── Bug 5: fallback para tier_qids si pstate está vacío ──
        pstate = self._f._project_state_manager.get_pstate(project_id)
        tier_qids = set(pstate.get("hub_tier_qids", []))
        if not tier_qids:
            # Primer turno o cross‑restart: leer desde state (persistido)
            tier_qids = set(state.hub_tier_qids_persisted)

        candidates = []
        for h, block in active.items():
            if block.pinned or block.obsolete:
                continue

            # ── Proteger bloques que contienen hubs del tier ──
            if self._f.valves.hub_bodies_tier_protect_from_paging:
                if any(qualify_symbol(s) in tier_qids for s in block.symbols):
                    continue

            if block.symbols:
                # ── FIX 17.c: Use qualify_symbol to include file_path ──
                block_activation = max(
                    (
                        activation_scores.get(qualify_symbol(s), 0.0)
                        for s in block.symbols
                    ),
                    default=0.0,
                )
            else:
                block_activation = 0.0
            if block_activation < min_activation:
                candidates.append((h, block_activation, block.importance_score))

        # Page out the coldest first: lowest activation, then lowest importance.
        candidates.sort(key=lambda t: (t[1], t[2]))

        # Only page out enough to return under the threshold, leaving headroom.
        n_to_page = len(active) - paging_threshold
        selected = [h for h, _, _ in candidates[:n_to_page]]

        # Debug log if under-paging (FIX #8)
        if len(selected) < n_to_page:
            # The caller (Filter) will log this via self._log_debug when it
            # sees fewer blocks paged than expected.
            pass

        return selected

    async def page_out_block(
        self,
        block: "CodeBlock",
        project_id: str,
        state: dict,
        symbol_index: "SymbolIndex",
        chroma_collection,
        embedder,
    ) -> bool:
        """
        Soft‑evict `block` to ChromaDB **without blocking**.

        The block is removed from active_blocks synchronously; the actual
        embedding and ChromaDB upsert are offloaded to a background task.
        The full body stays in the SQLite code_contents table, so the block
        can always be reconstructed later.
        """
        if chroma_collection is None or embedder is None:
            return False

        # Capture the data needed for the background task
        entry_id = f"{project_id}_paged_{block.hash}"
        excerpt = block.content[:500]
        symbol_names = ",".join(s.name for s in block.symbols)
        safe_text = block.content
        if hasattr(self._f, "_tokens"):
            safe_text = self._f._tokens.truncate_text_to_tokens(block.content, 32768)
        metadata = {
            "project_id": project_id,
            "is_paged_block": True,
            "block_hash": block.hash,
            "file_path": block.file_path or "",
            "content_type": block.content_type.value,
            "importance_score": block.importance_score,
            "paged_at": time.time(),
            "symbol_names": symbol_names,
        }

        # Offload the heavy embedding + upsert
        asyncio.create_task(
            self._page_out_async(
                entry_id=entry_id,
                safe_text=safe_text,
                excerpt=excerpt,
                metadata=metadata,
                embedder=embedder,
                chroma_collection=chroma_collection,
            )
        )

        # Mark as paged immediately so the caller can remove the block
        self._paged_hashes.setdefault(project_id, set()).add(block.hash)
        return True

    async def _page_out_async(
        self,
        entry_id: str,
        safe_text: str,
        excerpt: str,
        metadata: dict,
        embedder,
        chroma_collection,
    ) -> None:
        """Background task for embedding and upserting a paged block.

        The embedding is serialized through _page_out_semaphore so a burst of
        evictions can't run many encodes concurrently and spike embedder
        memory. The block has already been removed from active_blocks by the
        synchronous caller; this only affects how fast cold storage catches up.
        """
        async with self._page_out_semaphore:
            try:
                embedding = await anyio.to_thread.run_sync(
                    lambda: embedder.encode(safe_text).tolist()
                )
                await anyio.to_thread.run_sync(
                    lambda: chroma_collection.upsert(
                        ids=[entry_id],
                        embeddings=[embedding],
                        documents=[excerpt],
                        metadatas=[metadata],
                    )
                )
            except Exception:
                # Best effort; the block content is still in SQLite
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Purge old versions (per-file version limit)
    # ═══════════════════════════════════════════════════════════════════════════

    async def purge_old_versions(
        self,
        project_id: str,
        state: dict,
        symbol_index: "SymbolIndex",
        chroma_collection,
        embedder,
        max_versions_per_file: int = 3,
    ) -> int:
        """
        Move code blocks older than the N most recent versions per file to cold storage.

        Returns the number of blocks purged.
        """
        from collections import defaultdict

        by_file = defaultdict(list)
        for h, block in state.active_blocks.items():
            if block.file_path and not block.pinned and not block.obsolete:
                by_file[block.file_path].append((h, block))

        purged = 0
        for file_path, versions in by_file.items():
            if len(versions) <= max_versions_per_file:
                continue

            # Keep the N most recent versions
            versions.sort(key=lambda x: x[1].timestamp, reverse=True)
            for h, block in versions[max_versions_per_file:]:
                if self._f.valves.enable_block_paging and chroma_collection is not None:
                    paged = await self.page_out_block(
                        block=block,
                        project_id=project_id,
                        state=state,
                        symbol_index=symbol_index,
                        chroma_collection=chroma_collection,
                        embedder=embedder,
                    )
                    if paged:
                        del state.active_blocks[h]
                        purged += 1
                        continue
                # Fallback: remove from active blocks without paging
                if h in state.active_blocks:
                    del state.active_blocks[h]
                    purged += 1

        if purged > 0:
            self._f._log_debug(
                f"Purged {purged} old code version(s) across " f"{len(by_file)} file(s)"
            )
        return purged

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Page-in (temporary reconstruction from ChromaDB)
    # ═══════════════════════════════════════════════════════════════════════════

    async def page_in_block(
        self,
        block_hash: str,
        project_id: str,
        chroma_collection,
        db_conn=None,
    ) -> "Optional[CodeBlock]":
        """
        Retrieve a paged block for temporary use THIS TURN ONLY.

        Reconstruction is lossless:
          1. Read the full body from SQLite code_contents WHERE hash = block_hash
             (populated by _save_state_to_db). If a db_conn is provided we use
             it; otherwise we fall back to the ChromaDB excerpt (degraded).
          2. Re-extract symbols deterministically via SignatureExtractor — this
             yields identical symbols to the original block.
          3. Rebuild the CodeBlock from content + ChromaDB metadata.

        Does NOT restore the block to active_blocks and does NOT remove it from
        the paged registry — the block stays cold; only this turn sees it.

        Returns None if the block cannot be reconstructed from either source.
        """
        if not self.is_paged(block_hash, project_id):
            return None

        entry_id = f"{project_id}_paged_{block_hash}"

        # Try ChromaDB for metadata + excerpt (FIX #4: ChromaDB is optional for reconstruction)
        meta = None
        excerpt = ""
        if chroma_collection is not None:
            try:
                result = await anyio.to_thread.run_sync(
                    lambda: chroma_collection.get(
                        ids=[entry_id], include=["metadatas", "documents"]
                    )
                )
                if result and result.get("ids"):
                    meta = result["metadatas"][0]
                    excerpt = result["documents"][0] if result.get("documents") else ""
            except Exception:
                pass

        # Recover the full body from code_contents (authoritative) – now via _db_read
        content = ""
        if db_conn is not None:
            try:
                row = await self._f._state_store._db_read(
                    lambda: self._f._db_conn.execute(
                        "SELECT content FROM code_contents WHERE hash = ?",
                        (block_hash,),
                    ).fetchone()
                )
                if row and row[0]:
                    content = row[0]
            except Exception:
                pass

        # If DB failed, fall back to excerpt
        if not content:
            content = excerpt

        # If both sources are empty, we cannot reconstruct
        if not content:
            return None

        # Extract metadata fields with safe defaults
        file_path = meta.get("file_path") if meta else None
        ctype_str = (
            meta.get("content_type", ContentType.GENERAL.value)
            if meta
            else ContentType.GENERAL.value
        )
        importance = meta.get("importance_score", 1.0) if meta else 1.0

        try:
            ctype = ContentType(ctype_str)
        except Exception:
            ctype = ContentType.GENERAL

        # Deterministic symbol re-extraction
        symbols = []
        try:
            symbols = await SignatureExtractor.extract_async(content, file_path)
        except Exception:
            pass

        # Reconstruct the CodeBlock
        block = CodeBlock(
            content=content,
            content_type=ctype,
            file_path=file_path,
            hash=block_hash,
            symbols=symbols,
            importance_score=importance,
        )
        for s in block.symbols:
            s.parent_block_hash = block_hash
        return block


class RaptorCodeIndex:
    """
    Hierarchical clustering of code symbols (RAPTOR adapted for code).

    Builds two levels:
    - L1: clusters raw symbols (functions/classes) by semantic + graph proximity.
    - L2: clusters L1 cluster summaries into broader subsystems.

    Provides retrieval of the most relevant cluster summaries for a query,
    with L2 summaries prioritised (broader context first).
    """

    _N_LANDMARKS: int = 8

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Public API
    # ═══════════════════════════════════════════════════════════════════════════

    async def rebuild(
        self,
        project_id: str,
        symbol_index: "SymbolIndex",
        edges_out: dict,
        n_clusters: int,
        summary_model: str,
        summary_max_tokens: int,
        chroma_collection,
        llm_caller,
        embedder,
        graph_weight: float = 0.5,
    ) -> None:
        """
        Full rebuild: build L1 (symbol clusters) and, if enough L1 clusters
        exist, L2 (clusters of L1 summaries). Uses upsert over stable ids —
        a failed rebuild leaves the prior set intact. After a successful
        build, prunes stale higher-numbered cluster ids (FIX #7).
        """
        names = []
        try:
            names = list(symbol_index.get_all_names(project_id))
        except Exception:
            return
        if len(names) < max(2 * n_clusters, 4):
            return

        # Build L1 from symbols
        n_l1 = await self.build_layer(
            project_id=project_id,
            level=1,
            symbol_index=symbol_index,
            edges_out=edges_out,
            n_clusters=n_clusters,
            summary_model=summary_model,
            summary_max_tokens=summary_max_tokens,
            chroma_collection=chroma_collection,
            llm_caller=llm_caller,
            embedder=embedder,
            graph_weight=graph_weight,
        )

        # Build L2 from L1 summaries if enough L1 clusters exist
        if n_l1 >= 4:
            await self.build_layer(
                project_id=project_id,
                level=2,
                symbol_index=symbol_index,
                edges_out=edges_out,
                n_clusters=max(2, n_l1 // 3),
                summary_model=summary_model,
                summary_max_tokens=summary_max_tokens,
                chroma_collection=chroma_collection,
                llm_caller=llm_caller,
                embedder=embedder,
                graph_weight=graph_weight,
            )

        # Prune stale cluster ids (FIX #7)
        await self._prune_stale_clusters(
            project_id=project_id,
            level=1,
            kept_count=n_l1,
            chroma_collection=chroma_collection,
        )
        if n_l1 >= 4:
            await self._prune_stale_clusters(
                project_id=project_id,
                level=2,
                kept_count=max(2, n_l1 // 3),
                chroma_collection=chroma_collection,
            )

    async def build_layer(
        self,
        project_id: str,
        level: int,
        symbol_index: "SymbolIndex",
        edges_out: dict,
        n_clusters: int,
        summary_model: str,
        summary_max_tokens: int,
        chroma_collection,
        llm_caller,
        embedder,
        graph_weight: float = 0.5,
    ) -> int:
        """
        Build one RAPTOR level and store its cluster summaries.

        level == 1: cluster raw symbols (now using qualified ids).
        level >= 2: cluster the previous level's summaries (read back from the
                    store); graph features are not used at L2 (summaries have no
                    direct call edges), so the augmented vector degrades to the
                    plain semantic embedding.

        Returns the number of clusters actually created.
        """
        import numpy as np

        # ── Gather items + embeddings for this level ──────────────────────
        if level == 1:
            # Use qualified ids so that every distinct symbol (e.g., each
            # class's __init__) gets its own embedding, rather than collapsing
            # all same‑named methods into a single point.
            names = list(symbol_index.get_all_qualified_names(project_id))
            texts = []
            for n in names:
                sig = self._safe(
                    getattr(symbol_index, "get_signature", None),
                    n,
                    project_id,
                    default=n,
                )
                doc = self._safe(
                    getattr(symbol_index, "get_docstring", None),
                    n,
                    project_id,
                    default="",
                )
                texts.append(f"{sig} — {doc}".strip(" —"))
            item_ids = names
        else:
            prev = await self._load_level_summaries(
                project_id, level - 1, chroma_collection
            )
            if len(prev) < max(2 * n_clusters, 4):
                return 0
            item_ids = [p["id"] for p in prev]
            texts = [p["text"] for p in prev]
            names = item_ids  # no graph identity at L2

        if len(texts) < max(2 * n_clusters, 4):
            return 0

        # Semantic embeddings
        try:
            sem = await anyio.to_thread.run_sync(
                lambda: np.asarray(embedder.encode(texts), dtype=np.float32)
            )
        except Exception:
            return 0

        # ── Augmented features (semantic | graph) ─────────────────────────
        if level == 1 and graph_weight > 0.0:
            graph_feats = self._build_graph_features(names, edges_out)
            sem_scaled = sem * (1.0 - graph_weight)
            graph_scaled = graph_feats * graph_weight
            features = np.hstack([sem_scaled, graph_scaled])
        else:
            features = sem

        # ── KMeans fit ────────────────────────────────────────────────────
        k = min(n_clusters, len(texts))
        try:
            from sklearn.cluster import KMeans

            labels = await anyio.to_thread.run_sync(
                lambda: KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(
                    features
                )
            )
        except Exception:
            return 0

        # ── Per-cluster summary + store ───────────────────────────────────
        clusters: dict = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(idx)

        created = 0
        for lab, member_idxs in clusters.items():
            member_texts = [texts[i] for i in member_idxs]
            member_names = [item_ids[i] for i in member_idxs]
            summary = await self._summarise_cluster(
                member_texts, level, summary_model, summary_max_tokens, llm_caller
            )
            if not summary:
                continue
            stored = await self._store_summary(
                project_id=project_id,
                level=level,
                cluster_id=lab,
                summary=summary,
                member_names=member_names,
                chroma_collection=chroma_collection,
                embedder=embedder,
            )
            if stored:
                created += 1
        return created

    async def retrieve(
        self,
        query: str,
        project_id: str,
        top_k: int,
        embedder,
        chroma_collection,
    ) -> list:
        """
        Return cluster summary texts most relevant to the query, highest level
        first (L2 subsystems before L1 function groups). Cheap: the RAPTOR
        summary set is bounded by configuration, so this is effectively O(1).
        """
        if chroma_collection is None:
            return []
        try:
            q_emb = await anyio.to_thread.run_sync(
                lambda: embedder.encode(query[:1000]).tolist()
            )
            res = await anyio.to_thread.run_sync(
                lambda: chroma_collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where={
                        "$and": [
                            {"project_id": project_id},
                            {"is_raptor_summary": True},
                        ]
                    },
                    include=["documents", "metadatas"],
                )
            )
        except Exception:
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        if not docs:
            return []

        # Sort by raptor_level desc (subsystems first), keep original order otherwise
        paired = list(zip(docs, metas))
        paired.sort(key=lambda dm: -int(dm[1].get("raptor_level", 1)))
        return [d for d, _ in paired]

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Graph & distance helpers (for clustering)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_adjacency(self, edges_out: dict) -> dict:
        """
        Build an undirected adjacency dict from the directed call graph.

        edges_out: {src_name: [Edge(dst=..., type=...), ...]}
        Returns:   {name: set(neighbour_names)}

        Undirected means callers and callees are treated symmetrically for
        proximity purposes — if A calls B, they are neighbours regardless of
        direction.
        """
        adj: dict = {}
        for src, edge_list in edges_out.items():
            adj.setdefault(src, set())
            for edge in edge_list:
                dst = getattr(edge, "dst", None) or getattr(edge, "target", None)
                if dst is None:
                    continue
                adj[src].add(dst)
                adj.setdefault(dst, set()).add(src)
        return adj

    def _bfs_from(self, start: str, adj: dict, max_depth: int = 3) -> dict:
        """
        BFS from `start` up to `max_depth` hops.

        Returns:
            {name: depth} for all reachable nodes within max_depth.
            depth is an integer in [0, max_depth]; unreachable nodes are
            absent from the dict.
        """
        from collections import deque

        visited = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj.get(node, ()):
                if nb not in visited:
                    visited[nb] = depth + 1
                    queue.append((nb, depth + 1))
        return visited

    def _graph_distance(
        self,
        sym_a: str,
        sym_b: str,
        adj: dict,
        max_depth: int = 3,
    ) -> float:
        """
        Normalised call-graph distance in [0, 1].

        0.0 = same symbol or direct call.
        1.0 = unreachable within max_depth.

        Uses precomputed undirected adjacency (FIX #2) — O(reachable) per call
        instead of O(V·E).
        """
        if sym_a == sym_b:
            return 0.0
        # Lightweight BFS up to max_depth
        from collections import deque

        visited = {sym_a}
        queue = deque([(sym_a, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj.get(node, ()):
                if nb == sym_b:
                    return (depth + 1) / (max_depth + 1)
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
        return 1.0

    def _combined_distance(
        self,
        emb_a,
        emb_b,
        sym_a: str,
        sym_b: str,
        edges_out: dict,
        graph_weight: float,
    ) -> float:
        """
        Combined clustering metric (used for validation, not for the KMeans fit).

        combined = (1 - graph_weight) * cosine_distance + graph_weight * graph_distance
        Lower = more similar.
        """
        import numpy as np

        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        cosine_dist = 1.0 - float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )
        adj = self._build_adjacency(edges_out)
        graph_dist = self._graph_distance(sym_a, sym_b, adj, max_depth=3)
        return (1.0 - graph_weight) * cosine_dist + graph_weight * graph_dist

    def _build_graph_features(self, names: list, edges_out: dict):
        """
        Build a graph-position feature matrix: for each symbol, its (inverse)
        BFS distance to each of N landmark hub symbols. Landmarks are the
        symbols with the highest out+in degree. The resulting vector encodes
        "where in the call graph" a symbol sits, so KMeans groups
        call-graph-adjacent symbols together.

        Returns an (len(names) x N_LANDMARKS) float32 matrix.

        Uses precomputed adjacency and per-landmark BFS (FIX #2) —
        O(N_LANDMARKS·(V+E)) instead of O(V²·E).
        """
        import numpy as np

        if not names:
            return np.zeros((0, self._N_LANDMARKS), dtype=np.float32)

        # ── Degree (out + in) for landmark selection ───────────────────
        out_deg = {n: len(edges_out.get(n, [])) for n in names}
        in_deg = {n: 0 for n in names}
        for src, elist in edges_out.items():
            for e in elist:
                tgt = getattr(e, "dst", None) or getattr(e, "target", None)
                if tgt in in_deg:
                    in_deg[tgt] += 1
        degree = {n: out_deg.get(n, 0) + in_deg.get(n, 0) for n in names}

        landmarks = sorted(names, key=lambda n: -degree[n])[: self._N_LANDMARKS]
        # Pad landmark list if fewer symbols than N_LANDMARKS.
        while len(landmarks) < self._N_LANDMARKS and landmarks:
            landmarks.append(landmarks[-1])

        # ── Build adjacency once for all BFS calls ─────────────────────
        adj = self._build_adjacency(edges_out)

        # ── Per-landmark BFS → closeness features ──────────────────────
        feats = np.zeros((len(names), self._N_LANDMARKS), dtype=np.float32)
        max_depth = 3
        for j, lm in enumerate(landmarks):
            bfs_result = self._bfs_from(lm, adj, max_depth=max_depth)
            for i, n in enumerate(names):
                if n in bfs_result:
                    d = bfs_result[n]
                    feats[i, j] = 1.0 - (d / (max_depth + 1))
                # else stays 0.0 (unreachable)
        return feats

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Summary generation & storage (LLM + ChromaDB)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _summarise_cluster(
        self,
        member_texts: list,
        level: int,
        summary_model: str,
        summary_max_tokens: int,
        llm_caller,
    ) -> str:
        """Generate an LLM summary describing the cluster's shared responsibility."""
        if not member_texts:
            return ""
        listing = "\n".join(f"- {t}" for t in member_texts[:30])
        if level == 1:
            prompt = (
                f"These {len(member_texts)} functions/classes are closely "
                f"related in a codebase:\n{listing}\n\n"
                f"In 2-3 sentences, describe the subsystem they form and its "
                f"single responsibility. Be specific and concise."
            )
        else:
            prompt = (
                f"These {len(member_texts)} subsystem summaries belong to one "
                f"larger module:\n{listing}\n\n"
                f"In 2-3 sentences, describe the module's overall purpose."
            )
        try:
            return (
                await llm_caller(
                    prompt,
                    system_prompt="You are a code summarisation assistant. Output only the summary.",
                    model_override=summary_model,
                    max_tokens=summary_max_tokens,
                    temperature=0.2,
                )
            ).strip()
        except Exception:
            return ""

    async def _store_summary(
        self,
        project_id: str,
        level: int,
        cluster_id: int,
        summary: str,
        member_names: list,
        chroma_collection,
        embedder,
    ) -> bool:
        """Upsert one cluster summary into memory_collection."""
        entry_id = f"{project_id}_raptor_L{level}_C{cluster_id}"
        metadata = {
            "project_id": project_id,
            "is_raptor_summary": True,
            "raptor_level": level,
            "level_tag": f"L{level}",  # ← FIX #10: string tag avoids int/float coercion
            "cluster_id": int(cluster_id),
            "member_count": len(member_names),
            "member_names": ",".join(member_names[:50]),
            "created_at": time.time(),
        }
        try:
            emb = await anyio.to_thread.run_sync(
                lambda: embedder.encode(summary).tolist()
            )
            await anyio.to_thread.run_sync(
                lambda: chroma_collection.upsert(
                    ids=[entry_id],
                    embeddings=[emb],
                    documents=[summary],
                    metadatas=[metadata],
                )
            )
            return True
        except Exception:
            return False

    async def _load_level_summaries(
        self, project_id: str, level: int, chroma_collection
    ) -> list:
        """
        Read back stored summaries for a given level.

        Returns [{id, text}].
        Filters by level_tag (FIX #10) to avoid int/float coercion issues.
        """
        try:
            res = await anyio.to_thread.run_sync(
                lambda: chroma_collection.get(
                    where={
                        "$and": [
                            {"project_id": project_id},
                            {"is_raptor_summary": True},
                            {"level_tag": f"L{level}"},
                        ]
                    },
                    include=["documents"],
                )
            )
        except Exception:
            return []
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        return [{"id": i, "text": d} for i, d in zip(ids, docs)]

    async def _prune_stale_clusters(
        self,
        project_id: str,
        level: int,
        kept_count: int,
        chroma_collection,
    ) -> None:
        """
        Remove cluster ids >= kept_count for the given level (FIX #7).
        Called after a successful build so the prior set stays intact on failure.
        """
        if chroma_collection is None:
            return
        # Read all current ids for this level
        try:
            res = await anyio.to_thread.run_sync(
                lambda: chroma_collection.get(
                    where={
                        "$and": [
                            {"project_id": project_id},
                            {"is_raptor_summary": True},
                            {"level_tag": f"L{level}"},
                        ]
                    },
                    include=["ids"],
                )
            )
            current_ids = res.get("ids") or []
        except Exception:
            return

        stale = []
        for id_ in current_ids:
            # id_ format: {project_id}_raptor_L{level}_C{cluster_id}
            parts = id_.split("_C")
            if len(parts) == 2:
                try:
                    cid = int(parts[1])
                    if cid >= kept_count:
                        stale.append(id_)
                except ValueError:
                    pass

        if stale:
            try:
                await anyio.to_thread.run_sync(
                    lambda: chroma_collection.delete(ids=stale)
                )
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Utilities
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe(fn, *args, default=None):
        """Call fn(*args), returning default on any exception."""
        try:
            result = fn(*args)
            return result if result is not None else default
        except Exception:
            return default


class ContextBuilder:
    """
    Builds Block A + Block B, owns KV slot lifecycle, and provides skeleton
    and inventory utilities for the system prompt.

    Block A is the static, KV‑cache‑anchoring part (hub symbols, architecture
    map, guidelines, feedback context).  Block B is the dynamic, per‑query
    part (LOD‑activated code, LTM, use‑case‑tuned policies).

    Also handles:
    * Skeleton tier (stable signatures) inside Block A.
    * Scaffolding / skeleton responses for intent queries.
    * Chain‑of‑Thought (CoT) expand resolution.
    * Slot persistence (save / restore of KV cache state).

    Docs 10–13 backported:
        B4 – `_warmup_tier_prefill` stub (prevents AttributeError).
        E1 – LOD‑2 hysteresis (entry/exit thresholds).
        E3 – stable ordering by (tier, -PPR, qid).
        E4 – ghost hub qid pruning in Block A.
        E5 – skip duplicate signatures for skeleton‑tier symbols.
        E6 – recency pointers with signature previews for symbols not in Block B.
        M6 – prev_seeds fallback to persisted qids on cold‑start.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

        # Fast-path trigger for inventory / structural queries.
        self._LIST_INTENTS = re.compile(
            r"\b(?:dame\s+la\s+)?lista\s*(?:de\s+)?(?:clases|funciones)|"
            r"lístame|listame|enumera|inventario|"
            r"todas las (clases|funciones)|qué (clases|funciones)|"
            r"estructura del código|overview|list all|all (classes|functions)|"
            r"muestra las clases|show classes|list classes",
            re.IGNORECASE,
        )
        # Fast-path for skeleton/scaffolding queries.
        self._SKELETON_INTENTS = re.compile(
            r"\b(esqueleto|skeleton|stubs?|scaffold(ing)?|"
            r"(solo|sólo)\s+firmas|signatures?\s+only|"
            r"plantilla\s+de\s+(clases|c[oó]digo))\b"
            r"|estructura[^.\n]{0,40}(completar|rellenar|esqueleto|stub)",
            re.IGNORECASE,
        )
        # Filtered skeleton: "esqueleto de ClassName" / "skeleton of FuncName".
        self._SKELETON_SYMBOL_RE = re.compile(
            r"\b(?:esqueleto|skeleton|stub|scaffolding?)\s+"
            r"(?:de|of|para|for)?\s*`?(?P<sym>[A-Za-z_]\w*)`?",
            re.IGNORECASE,
        )

        # ── LOD policy per use case ─────────────────────────────
        self.LOD_PROFILES: Dict[str, dict] = {
            "A": {
                "lod1_mult": 1.0,
                "lod2_mult": 1.4,
                "lod3_mult": 2.5,
                "activation_direction": "both",
            },
            "B": {
                "lod1_mult": 1.0,
                "lod2_mult": 1.0,
                "lod3_mult": 1.2,
                "activation_direction": "callees",
            },
            "C": {
                "lod1_mult": 1.0,
                "lod2_mult": 1.0,
                "lod3_mult": 1.0,
                "activation_direction": "callees",
            },
            "D": {
                "lod1_mult": 1.0,
                "lod2_mult": 1.0,
                "lod3_mult": 1.0,
                "activation_direction": "callers",
            },
            "E": {
                "lod1_mult": 1.0,
                "lod2_mult": 2.0,
                "lod3_mult": 4.0,
                "activation_direction": "callees",
            },
        }
        # Use-case detection. Refactor (D) is tested BEFORE architecture (A).
        self._UC_COMMAND_RE = re.compile(
            r"^\s*/(arch|plan|code|refactor|scaffold)\b", re.IGNORECASE
        )
        self._UC_SCAFFOLD_RE = re.compile(
            r"\b(esqueleto|skeleton|stubs?|scaffold(?:ing)?|solo\s+firmas|"
            r"signatures?\s+only|boilerplate|plantilla\s+de\s+(?:clase|c[oó]digo))\b",
            re.IGNORECASE,
        )
        self._UC_REFACTOR_RE = re.compile(
            r"\brefactor\w*|"
            r"\brenombr\w*|\brename\w*|"
            r"extrae\s+la\s+(?:validaci[oó]n|l[oó]gica)|"
            r"extrae\s+(?:el\s+)?(?:m[eé]todo|funci[oó]n)|"
            r"extract\s+(?:a\s+)?(?:method|function)|"
            r"mueve\s+\w+\s+a|move\s+\w+\s+to|\binline\b|deduplic\w*|"
            r"reorganiz\w*|restructur\w*|"
            r"split\s+(?:this|the|el|la)\b|"
            r"an[aá]lisis\s+de\s+impacto|"
            r"nueva\s+estructura|"
            r"demasiado\s+larg[oa]|"
            r"limpiar\s+(?:el\s+)?(?:m[eé]todo|c[oó]digo)|"
            r"separar\s+responsabilidades",
            re.IGNORECASE,
        )
        self._UC_ARCH_RE = re.compile(
            r"\b(arquitectura|architecture|dise[ñn]o|design|"
            r"c[oó]mo\s+(?:estructurar|organizar|dividir)|"
            r"qu[eé]\s+(?:clases|m[oó]dulos|componentes)\s+"
            r"(?:necesito|crear|a[ñn]adir)|"
            r"abstract\s+(?:base\s+)?class|interface\s+design|"
            r"propuesta\s+de\s+dise[ñn]o|API\s+surface)\b",
            re.IGNORECASE,
        )
        self._UC_PLAN_RE = re.compile(
            r"\b(plan\s+de\s+(?:implementaci[oó]n|cambios)|implementation\s+plan|"
            r"pasos\s+para|steps\s+to|roadmap|"
            r"c[oó]mo\s+implementar|how\s+to\s+implement)\b",
            re.IGNORECASE,
        )

    # ======================================================================
    # B4 – warmup stub
    # ======================================================================
    async def _warmup_tier_prefill(self, project_id: str) -> None:
        """Phase‑2 placeholder: pre‑warms the stable tier prefix into the KV slot
        immediately after silent ingestion, so the next inlet finds it hot.
        Currently a no‑op; full implementation deferred to Phase 2 (KV‑slot
        prediction). Removing this placeholder will cause AttributeError in the
        silent‑ingestion task — keep it until Phase 2 is wired."""
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Block A – static, KV‑cache‑anchoring content
    # ═══════════════════════════════════════════════════════════════════════════

    async def build_block_a(
        self,
        project_id: str,
        is_code_session: bool,
        is_continuation: bool,
    ) -> str:
        """
        Build Block A: stable, KV-cache-anchoring content.

        Returns "" when not a code session.
        """
        if not is_code_session:
            return ""

        # --- 1. Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # ── E4: prune ghost hub qids ──────────────────────────────────────
        hub_qids: List[str] = pstate.get("hub_tier_qids_persisted", [])
        if hub_qids:
            current_qids: Set[str] = set(
                self._f._symbol_index.get_all_qualified_names(project_id)
            )
            hub_set = set(hub_qids)
            ghost_qids = hub_set - current_qids
            if ghost_qids:
                deletion_ratio = len(ghost_qids) / max(len(hub_set), 1)
                if deletion_ratio > 0.30:
                    pruned = list(hub_set & current_qids)
                    self._f._log_debug(
                        f"Hub tier: pruned {len(ghost_qids)} ghost qids "
                        f"({deletion_ratio:.0%} deleted after large code change). "
                        f"Remaining: {len(pruned)}"
                    )
                    pstate["hub_tier_qids_persisted"] = pruned

        # --- 2. Compute structure-only hash (stable across docstring enrichment) ---
        structure_hash = self._f._symbol_index.compute_structure_hash(project_id)
        if not structure_hash:
            structure_hash = hashlib.md5("no_symbols".encode()).hexdigest()[:16]
        pstate["structure_hash_for_cache"] = structure_hash

        # --- 3. Resolved call graph mode ---
        mode = pstate.get("resolved_call_graph_mode") or "hubs_only"

        # --- 4. Build cache key using structure hash (not the rendered text) ---
        cache_key = f"{structure_hash}__{mode}"
        cached_text = pstate.get("block_a_cached")
        stored_key = pstate.get("block_a_cache_key")

        if stored_key and stored_key == cache_key and cached_text is not None:
            # --- 4a. Cache hit: same code + same mode ---
            return cached_text

        # --- 4b. Cache miss or continuation freeze ---
        if is_continuation and cached_text is not None:
            # Continuation: freeze Block A to prevent KV cache misses
            self._f._log_debug(
                "🧱 Block A: frozen for AutoContinue (KV cache stability)"
            )
            return cached_text

        # --- 5. Build the static block ---
        parts: List[str] = []

        # 5.1 Base instructions (completely static)
        if is_code_session and self._f.valves.enable_confidence_scoring:
            parts.append(self._f.valves.confidence_prompt.strip())

        if is_code_session and self._f.valves.enable_code_awareness:
            checklist = (
                "## Code review checklist (apply when reviewing or fixing code):\n"
                "1. Execute mentally with 3 different inputs including edge cases.\n"
                "2. Identify every assumption and verify each one.\n"
                "3. Test every regex or string match against 5 counter-examples.\n"
                "4. Test collections with empty, single-element, and large inputs.\n"
                "5. Consider the worst-case scenario.\n"
                "6. Reason step by step, then provide the corrected code."
            )
            parts.append(checklist)

            critical_guidelines = (
                "## Critical reasoning guidelines\n"
                "- Before diagnosing a bug, verify if the observed behavior matches the "
                "expected behavior defined in the specification or codebase. "
                "Do not confuse expected behavior with a bug.\n"
                "- When you propose a fix, explicitly explain **why** the change resolves "
                "the root cause and how it prevents the issue from recurring.\n"
                "- When you propose a change (refactor, addition), explain **why** "
                "the change is needed: what problem it solves, what benefits it brings, "
                "and any trade-offs involved.\n"
                "- Avoid magic numbers; define named constants with meaningful names "
                "and derive them from a single source of truth whenever possible."
            )
            parts.append(critical_guidelines)

        # 5.2 Symbol index (depth depends on call_graph_context_mode)
        symbol_section_rendered = False
        if is_code_session and self._f.valves.enable_code_awareness:
            state = self._f._conversation_state_manager.get(project_id)
            if state and state.active_blocks:
                centrality = pstate.get("node_centrality", {})
                resolved_mode = pstate.get("resolved_call_graph_mode") or "hubs_only"
                self._f._log_debug(
                    f"Building Block A symbol section with mode='{resolved_mode}' "
                    f"(project={project_id})"
                )
                symbol_section = self._f._hub_index.build(
                    symbol_index=self._f._symbol_index,
                    centrality=centrality,
                    project_id=project_id,
                    top_n=self._f.valves.symbol_index_max_in_block_a,
                    valves=self._f.valves,
                    mode=resolved_mode,
                )
                if symbol_section:
                    parts.append(symbol_section)
                    symbol_section_rendered = True

        # 5.3 Skeleton tier
        skeleton_rendered_this_turn = False
        if is_code_session and self._f.valves.enable_code_awareness:
            skeleton_tier = self._build_skeleton_tier(project_id)
            if skeleton_tier:
                parts.append(skeleton_tier)
                skeleton_rendered_this_turn = True

        # 5.4 Feedback context
        if (
            is_code_session
            and self._f.valves.enable_feedback_tracking
            and self._f.valves.inject_feedback_context
        ):
            feedback_ctx = self._f._enrichment.get_feedback_context(project_id)
            if feedback_ctx:
                parts.append(feedback_ctx)

        static_block = "\n\n".join(p for p in parts if p.strip())

        # --- 6. Store in cache with the mode-aware key (using structure hash) ---
        pstate["block_a_cache_key"] = cache_key
        pstate["block_a_cached"] = static_block

        # --- 7. Record whether skeleton was actually rendered (for suppression gating) ---
        pstate["skeleton_rendered_this_turn"] = skeleton_rendered_this_turn
        # Also record the render mode for diagnostic use
        pstate["skeleton_render_mode"] = mode

        # --- 8. Detect and log prefix changes (KV cache miss) ---
        # Use the structure hash as the stable prefix hash
        new_prefix_hash = structure_hash
        last_hash = pstate.get("last_static_prefix_hash")
        if last_hash and last_hash != new_prefix_hash:
            self._f._log_debug(
                f"⚠️  KV CACHE MISS detected: static block changed "
                f"({last_hash} → {new_prefix_hash}). "
                f"llama.cpp will do a full prefill on this request."
            )
        elif not last_hash:
            self._f._log_debug(
                f"KV Cache: first request for project, "
                f"static prefix established ({new_prefix_hash})."
            )
        else:
            self._f._log_debug(
                f"✓ KV Cache: static prefix stable ({new_prefix_hash}). "
                f"llama.cpp will reuse KV states for Block A."
            )
        pstate["last_static_prefix_hash"] = new_prefix_hash

        tokens = (
            len(self._f.tokenizer.encode(static_block))
            if self._f.tokenizer
            else len(static_block) // 4
        )
        self._f._log_debug(f"Static Context Block: ~{tokens} tokens")

        return static_block

    def invalidate_block_a_cache(
        self, project_id: str, reason: str = "", recompute_centrality: bool = False
    ) -> None:
        """
        Force Block A rebuild on the next request, optionally refreshing
        centrality scores.
        """
        # --- 1. Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # --- 2. Clear all per-project caches ---
        pstate["block_a_cache_key"] = None
        pstate["block_a_cached"] = None
        pstate["skeleton_cache_key"] = None
        pstate["skeleton_cached"] = None
        pstate["skeleton_tier_cache_key"] = None
        pstate["skeleton_tier_cached"] = None

        # --- 3. Optionally recompute centrality ---
        if recompute_centrality:
            try:
                pstate["node_centrality"] = self._f._symbol_index.precompute_centrality(
                    project_id
                )
            except Exception as e:
                self._f._log_debug(f"Centrality recomputation failed: {e}")

        # --- 4. Log the invalidation ---
        if reason:
            self._f._log_debug(f"Block A + skeleton cache invalidated ({reason})")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Skeleton tier (stable signatures inside Block A)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_skeleton_tier(self, project_id: str) -> str:
        """
        Render the project skeleton (signatures only) as a STABLE context tier,
        cached by structure_hash so body edits and docstring additions don't
        invalidate it. Returns "" when disabled, empty, or over the tier budget.
        """
        if not self._f.valves.enable_skeleton_tier:
            return ""

        pstate = self._f._project_state_manager.get_pstate(project_id)

        structure_hash = self._f._symbol_index.compute_structure_hash(project_id)
        if not structure_hash:
            return ""

        cached_hash = pstate.get("skeleton_tier_cache_key")
        cached_tier = pstate.get("skeleton_tier_cached")
        if cached_hash and cached_hash == structure_hash and cached_tier is not None:
            return cached_tier

        skel = self._format_skeleton(project_id)
        if not skel:
            pstate["skeleton_tier_cache_key"] = structure_hash
            pstate["skeleton_tier_cached"] = ""
            return ""

        budget = self._f.valves.skeleton_tier_max_tokens
        if budget > 0:
            tok = self._f._tokens.estimate_code_tokens(skel)
            if tok > budget:
                self._f._log_debug(
                    f"Skeleton tier skipped: {tok} tokens > budget {budget}. "
                    "Block B keeps signatures inline."
                )
                pstate["skeleton_tier_cache_key"] = structure_hash
                pstate["skeleton_tier_cached"] = ""
                return ""

        tier = (
            "## Project Skeleton (stable — signatures only)\n"
            "_Contracts for the whole project. Bodies are shown on demand below "
            "or via `/expand <name>`._\n\n"
            f"{skel}"
        )

        pstate["skeleton_tier_cache_key"] = structure_hash
        pstate["skeleton_tier_cached"] = tier

        self._f._log_debug(
            f"Skeleton tier rendered (structure_hash={structure_hash}, "
            f"~{self._f._tokens.estimate_code_tokens(tier)} tokens)"
        )

        return tier

    # ── M6 + E4: _build_hub_bodies_tier ──────────────────────────

    def _build_hub_bodies_tier(self, project_id: str) -> Tuple[str, str, List[str]]:
        """
        Build the Hub‑Bodies Tier: full bodies of top‑N hubs by centrality,
        ordered by stability (last_modified_turn), truncated by budget.
        """
        if not self._f.valves.enable_hub_bodies_tier:
            return "", "", []

        pstate = self._f._project_state_manager.get_pstate(project_id)
        centrality = pstate.get("node_centrality", {})
        if not centrality:
            return "", "", []

        state = self._f._conversation_state_manager.get(project_id)
        current_turn = state.message_count

        # ── M6: prev_seeds fallback on cold‑start ────────────────────────
        hub_seeds_this_turn = pstate.get("hub_tier_seeds_this_turn", [])
        if hub_seeds_this_turn:
            prev_seeds = pstate.get("hub_tier_prev_seeds", [])
        else:
            prev_seeds = list(pstate.get("hub_tier_qids_persisted", []))

        # Persistent trackers (survive restarts via state)
        last_mod = state.hub_tier_last_modified
        body_hashes = state.hub_tier_body_hashes
        heat = state.hub_tier_query_heat

        # Selection: top‑N with optional centrality floor
        ranked = self._f._symbol_index.get_hub_symbols(
            project_id, centrality, self._f.valves.hub_bodies_tier_top_n
        )
        floor = self._f.valves.hub_bodies_tier_min_centrality
        candidates = [qid for qid, c in ranked if (floor <= 0 or c >= floor)]

        resolved = {}
        for qid in candidates:
            body, lang = self._resolve_hub_body(qid, project_id, state)
            if not body:
                continue
            max_body = self._f.valves.hub_bodies_tier_max_body_tokens
            if max_body > 0 and self._f._tokens.estimate_code_tokens(body) > max_body:
                self._f._log_debug(f"Hub tier: skipping {qid} (body > {max_body} tok)")
                continue
            bh = hashlib.md5(body.encode()).hexdigest()[:16]
            if body_hashes.get(qid) != bh:
                last_mod[qid] = current_turn
            body_hashes[qid] = bh
            resolved[qid] = (body, bh, lang)

        if not resolved:
            return "", "", []

        alpha = 0.3
        for qid in candidates:
            was_seed = 1.0 if qid in prev_seeds else 0.0
            heat[qid] = alpha * was_seed + (1 - alpha) * heat.get(qid, 0.0)

        ordered = sorted(
            resolved,
            key=lambda q: (
                last_mod.get(q, current_turn),
                -heat.get(q, 0.0),
                q,
            ),
        )

        budget = self._f.valves.hub_bodies_tier_max_tokens
        if (
            self._f.valves.enable_multi_phase_response
            or self._f.valves.force_multi_phase_response
        ):
            budget = min(budget, 6000)
            self._f._log_debug(f"Hub tier: budget capped to 6000 (multi‑phase active)")

        lines = [
            "## Core Implementation (hub symbols — stable, cached)",
            "_Full bodies of the most central symbols, ordered from most to least stable. "
            "They rarely change and are anchored here for KV cache reuse._",
            "",
        ]
        total = self._f._tokens.estimate_code_tokens("\n".join(lines))
        kept = []
        excluded_by_cap = []

        for qid in ordered:
            body, bh, lang = resolved[qid]
            meta = self._f._symbol_index.get_symbol_meta(qid, project_id) or {}
            doc = meta.get("docstring", "")
            doc_line = f"_{doc}_\n" if doc else ""

            kept_set = set(kept)
            body_with_xrefs = self._inject_tier_xrefs(body, qid, kept_set | set([qid]))

            chunk = f"### `{qid}`\n{doc_line}```{lang}\n{body_with_xrefs}\n```\n"
            tok = self._f._tokens.estimate_code_tokens(chunk)
            if budget > 0 and total + tok > budget:
                excluded_by_cap.extend(ordered[ordered.index(qid) :])
                break
            lines.append(chunk)
            kept.append(qid)
            total += tok

        if excluded_by_cap:
            self._f._log_debug(
                f"Hub tier: {len(excluded_by_cap)} hub(s) excluded by budget cap "
                f"→ will be served via LoD: {excluded_by_cap}"
            )

        if not kept:
            return "", "", []

        tier_text = "\n".join(lines)

        config_prefix = (
            f"n={self._f.valves.hub_bodies_tier_top_n}|"
            f"floor={self._f.valves.hub_bodies_tier_min_centrality}"
        )
        tier_hash = hashlib.md5(
            f"{config_prefix}|"
            + "|".join(f"{q}:{resolved[q][1]}" for q in kept).encode()
        ).hexdigest()[:16]

        live = set(candidates)
        for d in (last_mod, body_hashes, heat):
            for stale in [k for k in list(d.keys()) if k not in live]:
                del d[stale]

        state.hub_tier_qids_persisted = kept
        self._f._conversation_state_manager.set(project_id, state)

        previous_tier_hash = pstate.get("last_tier_hash")
        if previous_tier_hash and previous_tier_hash != tier_hash:
            self._f._log_debug(
                f"⚠️ TIER CACHE MISS: tier_hash changed "
                f"({previous_tier_hash} → {tier_hash}). "
                f"Hub bodies will be re-prefilled."
            )
        elif not previous_tier_hash:
            self._f._log_debug(f"TIER CACHE: first tier built ({tier_hash})")
        else:
            self._f._log_debug(f"✓ TIER CACHE HIT: tier_hash stable ({tier_hash})")
        pstate["last_tier_hash"] = tier_hash

        # ── M6: persist prev_seeds ────────────────────────────────────────
        pstate["hub_tier_prev_seeds"] = hub_seeds_this_turn or prev_seeds

        return tier_text, tier_hash, kept

    def _resolve_hub_body(
        self, qid: str, project_id: str, state: "ConversationState"
    ) -> Tuple[str, str]:
        """
        Deterministically resolve the full body of a symbol.

        Returns (body, language). Returns ('', 'python') if not found.
        """
        candidates = []
        for bh in self._f._symbol_index.find_blocks(qid, project_id):
            block = state.active_blocks.get(bh)
            if block and not block.obsolete:
                body = CodeBlockManager.extract_symbol_body(block, qid)
                if body:
                    candidates.append((block.timestamp, body, block))
        if not candidates:
            return "", "python"
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, body, block = candidates[0]
        lang = block.symbols[0].language if block.symbols else "python"
        return body, lang

    def _inject_tier_xrefs(self, body: str, qid: str, kept_set: Set[str]) -> str:
        """Add inline comments where this symbol calls another hub in the tier."""
        lines = body.split("\n")
        result = []
        for line in lines:
            result.append(line)
            for other_qid in kept_set:
                if other_qid == qid:
                    continue
                bare = other_qid.rsplit(".", 1)[-1]
                if (
                    f"{bare}(" in line
                    or f"self.{bare}(" in line
                    or f"->{bare}(" in line
                ):
                    result.append(f"    # ↑ see `{other_qid}` in this tier")
                    break
        return "\n".join(result)

    # ── E6: recency pointers (new implementation) ────────────────────────────

    def _build_hub_recency_pointers(
        self,
        project_id: str,
        current_block_b_qids: Set[str],
    ) -> str:
        """
        Build the recency pointer section for Block B.

        E6: includes a one‑line signature preview for symbols NOT in Block B.
        Symbols that ARE in Block B are marked as active this turn.
        """
        if not self._f.valves.hub_bodies_tier_recency_pointers:
            return ""

        pstate = self._f._project_state_manager.get_pstate(project_id)
        prev_seeds = pstate.get("hub_tier_prev_seeds", [])
        if not prev_seeds:
            return ""

        in_ctx = [q for q in prev_seeds if q in current_block_b_qids]
        out_of_ctx = [q for q in prev_seeds if q not in current_block_b_qids]

        lines = ["**Recently Active Symbols** (recency pointers):"]

        for qid in in_ctx:
            lines.append(f"  · `{qid}`  ← active this turn (full body in Block B)")

        for qid in out_of_ctx[:3]:
            meta = self._f._symbol_index.get_symbol_meta(qid, project_id) or {}
            sig = meta.get("signature", qid)
            lines.append(f"  · `{sig}`  # signature preview (not in Block B this turn)")

        if len(out_of_ctx) > 3:
            lines.append(
                f"  · … and {len(out_of_ctx) - 3} more (not in context this turn)"
            )

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _build_instruction_tail(use_case: str = "C") -> str:
        """M2: instruction tail adapted to the active use case."""
        tails = {
            "A": "_[Architecture mode: focus on contracts, interfaces, and invariants. "
            "Reasoning guidelines from system above apply.]_",
            "D": "_[Refactor mode: identify all callers before proposing changes. "
            "Code review checklist from system above applies.]_",
            "E": "_[Scaffold mode: signatures only, no implementation. "
            "Guidelines from system above apply.]_",
        }
        default = (
            "_[Reasoning mode: code review checklist + critical reasoning "
            "guidelines from system above apply to this response]_"
        )
        return tails.get(use_case, default)

    def _is_skeleton_tier_active(self, project_id: str) -> bool:
        """True only if the skeleton tier was actually rendered into Block A THIS turn."""
        if not self._f.valves.enable_skeleton_tier:
            return False
        pstate = self._f._project_state_manager.get_pstate(project_id)
        return pstate.get("skeleton_rendered_this_turn", False)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.1 — Project skeleton rendering (signatures only)
    # ═══════════════════════════════════════════════════════════════════════════

    def _format_skeleton(self, project_id: str) -> str:
        """Render the project skeleton: signatures of all indexed symbols."""
        symbol_index = self._f._symbol_index
        qids = sorted(symbol_index.get_all_qualified_names(project_id))
        if not qids:
            return ""

        lines: List[str] = []
        include_docstrings = self._f.valves.skeleton_include_docstrings

        classes = sorted(symbol_index.get_classes(project_id))
        if classes:
            lines.append("## Classes")
            for cls_name in classes:
                members = symbol_index.get_class_members(cls_name, project_id)
                if not members:
                    continue
                lines.append(f"class {cls_name}:")
                for member_qid in members:
                    meta = symbol_index.get_symbol_meta(member_qid, project_id) or {}
                    sig = meta.get("signature", member_qid)
                    if include_docstrings:
                        docstring = meta.get("docstring", "")
                        if docstring:
                            first_line = docstring.split("\n")[0]
                            lines.append(f"    {sig}  # {first_line}")
                        else:
                            lines.append(f"    {sig}")
                    else:
                        lines.append(f"    {sig}")

        module_funcs = [
            qid
            for qid in qids
            if "." not in qid
            and (symbol_index.get_symbol_meta(qid, project_id) or {}).get("kind")
            == "function"
        ]
        if module_funcs:
            if lines:
                lines.append("")
            lines.append("## Module-level functions")
            for qid in module_funcs:
                meta = symbol_index.get_symbol_meta(qid, project_id) or {}
                sig = meta.get("signature", qid)
                if include_docstrings:
                    docstring = meta.get("docstring", "")
                    if docstring:
                        first_line = docstring.split("\n")[0]
                        lines.append(f"- `{sig}`  # {first_line}")
                    else:
                        lines.append(f"- `{sig}`")
                else:
                    lines.append(f"- `{sig}`")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.2 — Filtered skeleton for a single symbol
    # ═══════════════════════════════════════════════════════════════════════════

    async def _format_skeleton_for_symbol(
        self, symbol_name: str, project_id: str, query: str
    ) -> str:
        """Render a filtered skeleton for a single symbol and its direct neighbors."""
        symbol_index = self._f._symbol_index
        qids = symbol_index.get_qualified_names_for(symbol_name, project_id)
        if not qids:
            return ""

        target_qid = sorted(qids)[0]
        meta = symbol_index.get_symbol_meta(target_qid, project_id) or {}
        sig = meta.get("signature", target_qid)
        doc = meta.get("docstring", "")

        lines = [
            f"## Skeleton for `{target_qid}`",
            f"```\n{sig}\n```",
        ]
        if doc and self._f.valves.skeleton_include_docstrings:
            lines.append(f"# {doc.split(chr(10))[0]}")

        bare = meta.get("name", target_qid.rsplit(".", 1)[-1])
        callers = set()
        for edge in symbol_index.get_edges_in(bare, project_id):
            callers.add(edge.src)
        if callers:
            lines.append("\n### Callers")
            for c in sorted(callers)[:10]:
                cm = symbol_index.get_symbol_meta(c, project_id) or {}
                csig = cm.get("signature", c)
                lines.append(f"- `{csig}`")

        callees = set()
        for edge in symbol_index.get_edges_out(target_qid, project_id):
            callees.add(edge.dst)
        if callees:
            lines.append("\n### Callees")
            for c in sorted(callees)[:10]:
                q = symbol_index.get_qualified_names_for(c, project_id)
                if q:
                    qid = sorted(q)[0]
                    cm = symbol_index.get_symbol_meta(qid, project_id) or {}
                    csig = cm.get("signature", c)
                else:
                    csig = c
                lines.append(f"- `{csig}`")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.3 — Full symbol inventory
    # ═══════════════════════════════════════════════════════════════════════════

    async def _format_full_symbol_inventory(
        self, all_qids: List[str], project_id: str
    ) -> str:
        """Render a complete inventory of all symbols in the project."""
        if not all_qids:
            return ""

        symbol_index = self._f._symbol_index
        lines = ["## Code Inventory (all symbols)"]

        classes = sorted(symbol_index.get_classes(project_id))
        if classes:
            lines.append("\n### Classes")
            for cls in classes:
                members = symbol_index.get_class_members(cls, project_id)
                if not members:
                    continue
                lines.append(f"- **{cls}** ({len(members)} methods)")
                for m in members[:20]:
                    meta = symbol_index.get_symbol_meta(m, project_id) or {}
                    sig = meta.get("signature", m)
                    lines.append(f"    - `{sig}`")
                if len(members) > 20:
                    lines.append(f"    - ... and {len(members)-20} more")

        module_funcs = [
            qid
            for qid in all_qids
            if "." not in qid
            and (symbol_index.get_symbol_meta(qid, project_id) or {}).get("kind")
            == "function"
        ]
        if module_funcs:
            lines.append("\n### Module‑level functions")
            for qid in module_funcs[:50]:
                meta = symbol_index.get_symbol_meta(qid, project_id) or {}
                sig = meta.get("signature", qid)
                lines.append(f"- `{sig}`")
            if len(module_funcs) > 50:
                lines.append(f"- ... and {len(module_funcs)-50} more")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.4 — Skeleton for CoT (architecture reasoning context)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _get_skeleton_for_cot(self, project_id: str, query: str) -> str:
        """Retrieve the project skeleton (signatures only) for use as CoT context."""
        return self._format_skeleton(project_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.5 — Resolve /expand hints in CoT output
    # ═══════════════════════════════════════════════════════════════════════════

    async def _resolve_cot_expands(self, reasoning_text: str, project_id: str) -> str:
        """Replace `/expand <symbol>` placeholders with actual symbol bodies."""
        if not self._f.valves.enable_cot_expand_resolution:
            return reasoning_text

        pattern = re.compile(r"/expand\s+([A-Za-z_][\w.]*)")
        max_expands = self._f.valves.cot_expand_max_symbols
        max_tokens = self._f.valves.cot_expand_max_tokens

        matches = pattern.findall(reasoning_text)
        if not matches:
            return reasoning_text

        symbols = list(dict.fromkeys(matches))[:max_expands]
        expanded = reasoning_text
        total_chars_added = 0

        for sym in symbols:
            qids = self._f._symbol_index.get_qualified_names_for(sym, project_id)
            if not qids:
                continue
            qid = sorted(qids)[0]

            block_hashes = self._f._symbol_index.find_blocks(qid, project_id)
            if not block_hashes:
                continue
            state = self._f._conversation_state_manager.get(project_id)
            block = None
            for bh in block_hashes:
                blk = state.active_blocks.get(bh)
                if blk and not blk.obsolete:
                    block = blk
                    break
            if block is None:
                continue

            body = CodeBlockManager.extract_symbol_body(block, qid)
            if not body:
                continue

            if max_tokens > 0:
                est_tokens = self._f._tokens.estimate_code_tokens(body)
                if total_chars_added + est_tokens * 4 > max_tokens * 4:
                    truncated = (
                        body[: (max_tokens * 4 - total_chars_added)]
                        + "\n# ... [truncated]"
                    )
                    body = truncated
                    expanded = expanded.replace(
                        f"/expand {sym}", f"```\n{body}\n```", 1
                    )
                    break

            replacement = f"```\n{body}\n```"
            expanded = expanded.replace(f"/expand {sym}", replacement, 1)
            total_chars_added += len(body)

        return expanded

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.6 — Docstring provider
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_docstring_provider(self, project_id: str):
        """Return f(symbol_name, parent_class='') -> one-line docstring or ''."""
        if not self._f.valves.skeleton_include_docstrings:
            return lambda _name, _parent="": ""

        def _provider(symbol_name: str, parent_class: str = "") -> str:
            qid = qualify_symbol_name(symbol_name, parent_class)
            return self._f._symbol_index.get_docstring(qid, project_id) or ""

        return _provider

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Block B – dynamic, per‑query LOD‑activated context
    # ═══════════════════════════════════════════════════════════════════════════

    def get_effective_context_budget(self, project_id: str) -> int:
        """
        Tokens available for history + user message after Block A + Block B.
        """
        window = self._f.valves.context_window_tokens
        pstate = self._f._project_state_manager.get_pstate(project_id)
        used = pstate.get("last_system_tokens", 0)

        reserve = self._f.valves.response_reserve_tokens
        budget = max(0, window - used - reserve)

        self._f._log_debug(
            f"get_effective_context_budget: window={window}, used_system={used}, "
            f"reserve={reserve} → {budget} tokens available for history + user message"
        )
        return budget

    def classify_use_case(
        self, query: str, intent_vector: dict
    ) -> Tuple[str, dict, str]:
        """
        Classify the query into one of five use cases and return its LOD profile
        and a human-readable label.
        """
        q = query or ""

        if self._f.valves.lod_intent_explicit_override:
            m = self._UC_COMMAND_RE.match(q)
            if m:
                case_key = {
                    "arch": UseCase.ARCHITECTURE,
                    "plan": UseCase.PLANNING,
                    "code": UseCase.PROGRAMMING,
                    "refactor": UseCase.REFACTORING,
                    "scaffold": UseCase.SCAFFOLDING,
                }[m.group(1).lower()]
                case = case_key.value
                self._f._log_debug(
                    f"classify_use_case: detected '{case_key.label}' "
                    f"via explicit command '/{m.group(1)}'"
                )
                return case, dict(self.LOD_PROFILES[case]), case_key.label

        if self._UC_SCAFFOLD_RE.search(q):
            case = UseCase.SCAFFOLDING
            self._f._log_debug(f"classify_use_case: detected '{case.label}' via regex")
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        if self._UC_REFACTOR_RE.search(q):
            case = UseCase.REFACTORING
            self._f._log_debug(f"classify_use_case: detected '{case.label}' via regex")
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        if self._UC_ARCH_RE.search(q):
            case = UseCase.ARCHITECTURE
            self._f._log_debug(f"classify_use_case: detected '{case.label}' via regex")
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        if self._UC_PLAN_RE.search(q):
            case = UseCase.PLANNING
            self._f._log_debug(f"classify_use_case: detected '{case.label}' via regex")
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        iv = intent_vector or {}
        refactor_w = iv.get("refactor", 0.0)
        debug_w = iv.get("debug", 0.0)
        modify_w = iv.get("modify", 0.0)
        explain_w = iv.get("explain", 0.0)

        if refactor_w >= 0.30 and refactor_w >= max(debug_w, modify_w, explain_w):
            case = UseCase.REFACTORING
            self._f._log_debug(
                f"classify_use_case: detected '{case.label}' "
                f"via intent_vector tie-break (refactor={refactor_w:.2f})"
            )
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        if explain_w >= 0.5 and explain_w > modify_w:
            case = UseCase.ARCHITECTURE
            self._f._log_debug(
                f"classify_use_case: detected '{case.label}' "
                f"via intent_vector tie-break (explain={explain_w:.2f} > modify={modify_w:.2f})"
            )
            return case.value, dict(self.LOD_PROFILES[case.value]), case.label

        case = UseCase.PROGRAMMING
        self._f._log_debug(
            f"classify_use_case: default '{case.label}' - no specific signals"
        )
        return case.value, dict(self.LOD_PROFILES[case.value]), case.label

    def _resolve_call_graph_mode(
        self,
        query: str,
        intent_vector: dict,
        project_id: str,
    ) -> str:
        """Resolve the effective call-graph depth for Block A."""
        valve = self._f.valves.call_graph_context_mode
        self._f._log_debug(f"Resolving call graph mode: valve='{valve}'")

        if valve != "auto":
            self._f._log_debug(f"  manual override → '{valve}'")
            return valve

        use_case, _, _ = self.classify_use_case(query, intent_vector)

        total_symbols = len(self._f._symbol_index.get_all_qualified_names(project_id))
        free_tokens = self.get_effective_context_budget(project_id)

        self._f._log_debug(
            f"  auto resolution: use_case={use_case}, total_symbols={total_symbols}, "
            f"free_tokens={free_tokens}"
        )

        window = self._f.valves.context_window_tokens
        full_graph_floor = int(window * self._f.valves.full_graph_min_free_token_ratio)
        expanded_hubs_floor = int(
            window * self._f.valves.expanded_hubs_min_free_token_ratio
        )

        def _full_graph_allowed() -> bool:
            symbol_ok = (
                total_symbols
                <= self._f.valves.call_graph_auto_full_graph_symbol_ceiling
            )
            token_ok = free_tokens >= full_graph_floor
            self._f._log_debug(
                f"    full_graph guard: symbol_ok={symbol_ok} "
                f"({total_symbols} <= {self._f.valves.call_graph_auto_full_graph_symbol_ceiling}), "
                f"token_ok={token_ok} ({free_tokens} >= {full_graph_floor})"
            )
            return symbol_ok and token_ok

        def _expanded_hubs_allowed() -> bool:
            symbol_ok = (
                total_symbols
                <= self._f.valves.call_graph_auto_expanded_hubs_symbol_ceiling
            )
            token_ok = free_tokens >= expanded_hubs_floor
            self._f._log_debug(
                f"    expanded_hubs guard: symbol_ok={symbol_ok} "
                f"({total_symbols} <= {self._f.valves.call_graph_auto_expanded_hubs_symbol_ceiling}), "
                f"token_ok={token_ok} ({free_tokens} >= {expanded_hubs_floor})"
            )
            return symbol_ok and token_ok

        if use_case == "A":
            if _full_graph_allowed():
                self._f._log_debug(
                    "  resolved mode: full_graph (use_case A, guards passed)"
                )
                return "full_graph"
            if _expanded_hubs_allowed():
                self._f._log_debug(
                    "  resolved mode: expanded_hubs (use_case A, full_graph blocked, expanded passed)"
                )
                return "expanded_hubs"
            self._f._log_debug(
                "  resolved mode: hubs_only (use_case A, neither full nor expanded allowed)"
            )
            return "hubs_only"

        if use_case == "D":
            if _expanded_hubs_allowed():
                self._f._log_debug(
                    "  resolved mode: expanded_hubs (use_case D, guards passed)"
                )
                return "expanded_hubs"
            self._f._log_debug(
                "  resolved mode: hubs_only (use_case D, expanded blocked)"
            )
            return "hubs_only"

        self._f._log_debug(f"  resolved mode: hubs_only (use_case {use_case} default)")
        return "hubs_only"

    def prepare_call_graph_mode(
        self, project_id: str, query: str, intent_vector: dict
    ) -> str:
        """
        Resolve and apply the call-graph mode for this turn BEFORE Block A is
        built. Implements hysteresis: upgrades immediate, downgrades deferred.
        """
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # Global scope detection
        if hasattr(
            self._f, "_seed_inferencer"
        ) and self._f._seed_inferencer.is_global_scope(query):
            pstate["resolved_call_graph_mode"] = "full_graph"
            pstate["force_multi_phase_this_turn"] = True
            pstate["graph_mode_downgrade_streak"] = 0
            self._f._log_debug(
                "prepare_call_graph_mode: global scope detected → "
                "full_graph forced + multi‑phase activated this turn."
            )
            return "full_graph"

        raw_resolved_mode = self._resolve_call_graph_mode(
            query, intent_vector, project_id
        )

        _MODE_RANK = {"hubs_only": 0, "expanded_hubs": 1, "full_graph": 2}

        previous_mode = pstate.get("resolved_call_graph_mode")
        streak = pstate.get("graph_mode_downgrade_streak", 0)

        if previous_mode is None or _MODE_RANK.get(
            raw_resolved_mode, 0
        ) >= _MODE_RANK.get(previous_mode, 0):
            resolved_graph_mode = raw_resolved_mode
            pstate["graph_mode_downgrade_streak"] = 0
        else:
            streak += 1
            pstate["graph_mode_downgrade_streak"] = streak
            if streak >= self._f.valves.call_graph_mode_downgrade_after_turns:
                resolved_graph_mode = raw_resolved_mode
                pstate["graph_mode_downgrade_streak"] = 0
            else:
                resolved_graph_mode = previous_mode
                self._f._log_debug(
                    f"Call graph mode: downgrade to {raw_resolved_mode} deferred "
                    f"({streak}/{self._f.valves.call_graph_mode_downgrade_after_turns} "
                    f"turns) — keeping {previous_mode} to avoid KV-cache thrash"
                )

        if previous_mode != resolved_graph_mode:
            self._f._log_debug(
                f"Call graph mode: {previous_mode or '(none)'} → "
                f"{resolved_graph_mode} (resolved before Block A build this turn)"
            )
            pstate["resolved_call_graph_mode"] = resolved_graph_mode

        return resolved_graph_mode

    # ── E5: helper to render body only ──────────────────────────────────────

    def _render_symbol_body_only(self, qid: str, project_id: str) -> str:
        """Render a symbol's body without the signature header.

        Used when the signature is already present in Block A (skeleton tier).
        """
        state = self._f._conversation_state_manager.get(project_id)
        block_hashes = self._f._symbol_index.find_blocks(qid, project_id)
        for bh in block_hashes:
            block = state.active_blocks.get(bh)
            if block and not block.obsolete:
                body = CodeBlockManager.extract_symbol_body(block, qid)
                if body:
                    return f"# {qid} — body\n{body}\n"
        return ""

    async def build_block_b(
        self,
        project_id: str,
        query: str,
        messages: list,
        slot_free: bool,
        intent_vector: dict,
        is_continuation: bool,
    ) -> str:
        """
        Build Block B: dynamic per-query content with SWA-aware ordering.
        """
        if not self._f.valves.enable_path_analysis:
            active_ctx = self._f._activation.get_active_code_context(project_id, query)
            return active_ctx if active_ctx else ""

        state = self._f._conversation_state_manager.get(project_id)
        if not state or not state.active_blocks:
            return ""

        # Fast path: FILTERED skeleton
        if self._f.valves.enable_skeleton_intent:
            _sym_match = self._SKELETON_SYMBOL_RE.search(query)
            if _sym_match:
                _sym_name = _sym_match.group("sym")
                _all = self._f._symbol_index.get_all_names(project_id)
                if _sym_name in _all:
                    skel = await self._format_skeleton_for_symbol(
                        _sym_name, project_id, query
                    )
                    if skel:
                        return skel

        # Fast path: skeleton / scaffolding queries
        if self._f.valves.enable_skeleton_intent and self._SKELETON_INTENTS.search(
            query
        ):
            skel = self._format_skeleton(project_id)
            if skel:
                return skel

        # Fast path for inventory / listing queries
        if self._LIST_INTENTS.search(query):
            all_qids = self._f._symbol_index.get_all_qualified_names(project_id)
            if all_qids:
                return await self._format_full_symbol_inventory(all_qids, project_id)

        # Step 1a: Classify use case
        active_use_case, use_case_profile, _ = self.classify_use_case(
            query, intent_vector
        )

        # Step 1b: LLM‑guided seed inference
        inferred_seeds: Dict[str, float] = {}
        if self._f.valves.seed_inference_mode != "off":
            inferred_seeds = await self._f._seed_inferencer.infer_seeds(
                query=query,
                project_id=project_id,
                intent_vector=intent_vector,
                use_case=active_use_case,
                slot_free=slot_free,
            )

        # Step 1c: ActivationGraph
        ag = self._f._activation.build_activation_graph(
            query,
            project_id,
            messages=messages,
            inferred_seeds=inferred_seeds,
        )
        activated = ag.get_activated_nodes(
            threshold=self._f.valves.path_activation_threshold
        )
        if not activated:
            self._f._log_debug(
                "build_block_b: no activated nodes, falling back to full context"
            )
            return self._f._activation.get_active_code_context(project_id, query)

        if self._f.valves.debug and hasattr(self._f, "_write_counter"):
            if self._f._write_counter % 50 == 0:
                self._f._log_debug(self._f._activation._ppr_cache.stats)

        # Step 2: Adjust LOD thresholds by intent
        debug_weight = intent_vector.get("debug", 0.2)
        modify_weight = intent_vector.get("modify", 0.3)
        refactor_weight = intent_vector.get("refactor", 0.1)

        lod3 = self._f.valves.lod3_threshold
        lod2 = self._f.valves.lod2_threshold
        lod1 = self._f.valves.lod1_threshold

        if self._f.valves.enable_lod_by_intent:
            lod1 *= use_case_profile.get("lod1_mult", 1.0)
            lod2 *= use_case_profile.get("lod2_mult", 1.0)
            lod3 *= use_case_profile.get("lod3_mult", 1.0)
        else:
            if debug_weight + modify_weight > 0.6:
                scale = 0.7
            elif refactor_weight > 0.4:
                scale = 0.0
            else:
                scale = 1.0
            lod3 *= scale
            lod2 *= scale
            lod1 *= scale

        # Case D: pull in direct callers
        if (
            self._f.valves.enable_lod_by_intent
            and active_use_case == "D"
            and self._f.valves.lod_intent_refactor_callers_max != 0
        ):
            max_callers = self._f.valves.lod_intent_refactor_callers_max
            pulled = 0
            for seed_qid in ag.get_seed_nodes():
                meta = self._f._symbol_index.get_symbol_meta(seed_qid, project_id) or {}
                bare = meta.get("name", seed_qid.rsplit(".", 1)[-1])
                for edge in self._f._symbol_index.get_edges_in(bare, project_id):
                    if max_callers > 0 and pulled >= max_callers:
                        break
                    caller_qid = edge.src
                    if activated.get(caller_qid, 0.0) < lod1:
                        activated[caller_qid] = max(
                            activated.get(caller_qid, 0.0), lod1
                        )
                        pulled += 1
                if max_callers > 0 and pulled >= max_callers:
                    break
            if pulled:
                self._f._log_debug(
                    f"Case D: pulled {pulled} direct caller(s) of seed symbol(s) "
                    f"into Block B at LOD-1 (impact analysis)."
                )

        # Mode is resolved BEFORE Block A is built this turn
        pstate = self._f._project_state_manager.get_pstate(project_id)
        resolved_graph_mode = pstate.get("resolved_call_graph_mode")
        if resolved_graph_mode is None:
            resolved_graph_mode = self.prepare_call_graph_mode(
                project_id, query, intent_vector
            )

        # Step 3: Build LOD tiers
        total_tokens = 0
        budget = self._f.valves.active_context_max_tokens or 32000

        if (
            self._f.valves.auto_budget_context_for_parts
            and (
                self._f.valves.enable_multi_phase_response
                or self._f.valves.force_multi_phase_response
            )
            and self._f.valves.context_window_tokens > 0
        ):
            _SYSTEM_OVERHEAD = 2000
            _available_for_context = (
                self._f.valves.context_window_tokens
                - self._f.valves.multi_phase_effective_max_tokens
                - _SYSTEM_OVERHEAD
            )
            budget = min(budget, max(8000, _available_for_context))

        tier_qids = set(pstate.get("hub_tier_qids", []))
        injected_symbols: Set[str] = set(tier_qids)

        # ── E3: stable ordering ──────────────────────────────────────────────
        # First, compute LOD tiers for E1 and E3
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # ── E1: LOD‑2 hysteresis ────────────────────────────────────────────
        lod2_entry = self._f.valves.lod2_threshold
        lod2_exit = lod2_entry * self._f.valves.lod2_exit_ratio
        currently_lod2: Set[str] = set(pstate.get("lod2_active_qids_prev", []))

        lod2_qids: Set[str] = set()
        lod3_qids: Set[str] = set()
        lod1_qids: Set[str] = set()

        for qid, score in activated.items():
            if qid in currently_lod2:
                if score >= lod2_exit:
                    lod2_qids.add(qid)
            else:
                if score >= lod2_entry:
                    lod2_qids.add(qid)
            if score >= lod3:
                lod3_qids.add(qid)

        # Persist for next turn (E1)
        pstate["lod2_active_qids_prev"] = list(lod2_qids)

        # ── E5: retrieve skeleton tier qids to avoid duplicates ──────────────
        skeleton_qids: Set[str] = set(pstate.get("skeleton_tier_qids", []))

        # ── E3: stable ordering function ──────────────────────────────────────
        def _lod_tier(qid: str) -> int:
            if qid in lod3_qids:
                return 3
            if qid in lod2_qids:
                return 2
            return 1

        sorted_nodes = sorted(
            activated.keys(),
            key=lambda qid: (
                -_lod_tier(qid),  # tier DESC
                -activated.get(qid, 0.0),  # PPR DESC within tier
                qid,  # stable lexicographic tie‑breaker
            ),
        )

        # ── Centrality LOD bump ──────────────────────────────────────────────
        if self._f.valves.enable_centrality_lod_bump:
            centrality = pstate.get("node_centrality", {})
            threshold = self._f.valves.centrality_lod_bump_threshold
            adjusted = []
            for qid in sorted_nodes:
                cent = centrality.get(qid, 0.0)
                if cent >= threshold:
                    effective = min(
                        1.0,
                        activated.get(qid, 0.0)
                        + cent * self._f.valves.centrality_lod_bump_weight,
                    )
                    adjusted.append((qid, effective))
                else:
                    adjusted.append((qid, activated.get(qid, 0.0)))
            # Re‑sort with adjusted scores? Simpler: just update activated and re‑sort.
            # We'll keep sorted_nodes as is, but use activated_scores for sorting.
            # Actually, the existing logic uses activated dict. We'll just re‑apply the stable sort.
            activated_scores = dict(adjusted)
            sorted_nodes = sorted(
                activated_scores.keys(),
                key=lambda qid: (
                    -_lod_tier(qid),
                    -activated_scores.get(qid, 0.0),
                    qid,
                ),
            )
        else:
            activated_scores = activated

        # ── Batched LOD-2 docstring pre-resolution ─────────────────────────
        if self._f.valves.enable_auto_docstrings:
            lod2_candidates = [
                qid
                for qid in lod2_qids
                if not skeleton_qids or qid not in skeleton_qids
            ]
            if lod2_candidates:
                missing = []
                for qid in lod2_candidates:
                    has_doc = False
                    for bh in self._f._symbol_index.find_blocks(qid, project_id):
                        blk = state.active_blocks.get(bh)
                        if blk and any(
                            qualify_symbol_name(s.name, s.parent_symbol) == qid
                            and s.docstring
                            for s in blk.symbols
                        ):
                            has_doc = True
                            break
                    if not has_doc:
                        missing.append(qid)
                if missing:
                    await self._f._enrichment.ensure_docstrings_batch(
                        missing, project_id
                    )

        # ── Batched LOD-2.5 CFG pre-resolution ─────────────────────────────
        if self._f.valves.enable_cfg_skeletons and (
            active_use_case == "D"
            or intent_vector.get("debug", 0.0)
            >= self._f.valves.cfg_skeleton_debug_intent_threshold
        ):
            cfg_candidates = [
                qid
                for qid in lod2_qids
                if not skeleton_qids or qid not in skeleton_qids
            ]
            self._f._log_debug(
                f"CFG gate TRIGGERED: use_case={active_use_case}, "
                f"debug_intent={intent_vector.get('debug', 0.0):.2f}, "
                f"candidates={cfg_candidates}"
            )
            if cfg_candidates:
                await self._f._enrichment.ensure_cfg_batch(cfg_candidates, project_id)
        else:
            self._f._log_debug(
                f"CFG gate NOT triggered: use_case={active_use_case}, "
                f"debug_intent={intent_vector.get('debug', 0.0):.2f}, "
                f"enable_cfg_skeletons={self._f.valves.enable_cfg_skeletons}"
            )

        # ── Iterate over sorted_nodes and build LOD tiers ──────────────────
        _lod0_parts: List[str] = []
        _lod1_parts: List[str] = []
        _lod2_parts: List[str] = []
        _lod3_parts: List[str] = []

        for qid in sorted_nodes:
            if total_tokens >= budget:
                break

            if qid in injected_symbols:
                continue

            # ── E5: skip if in skeleton tier and LOD-2 ──────────────────────
            if qid in skeleton_qids:
                if _lod_tier(qid) == 2:
                    self._f._log_debug(
                        f"Block B: skipping LOD-2 for {qid} (in skeleton tier)"
                    )
                    continue
                if _lod_tier(qid) == 3:
                    # Render body only
                    body_only = self._render_symbol_body_only(qid, project_id)
                    if body_only:
                        tok = self._f._tokens.estimate_code_tokens(body_only)
                        if total_tokens + tok <= budget:
                            _lod3_parts.append(body_only)
                            total_tokens += tok
                            injected_symbols.add(qid)
                    continue

            block_hashes = self._f._symbol_index.find_blocks(qid, project_id)
            for bh in block_hashes:
                block = state.active_blocks.get(bh)

                if block is None and self._f._pager is not None:
                    if self._f._pager.is_paged(bh, project_id):
                        block = await self._f._pager.page_in_block(
                            block_hash=bh,
                            project_id=project_id,
                            chroma_collection=self._f.memory_collection,
                            db_conn=self._f._db_conn,
                        )

                if not block or block.obsolete:
                    continue

                score = activated_scores.get(qid, 0.0)

                # LOD-1: Signatures only
                if _lod_tier(qid) == 1:
                    sig = next(
                        (
                            sym.signature
                            for sym in block.symbols
                            if qualify_symbol(sym) == qid
                        ),
                        qid,
                    )
                    tok = len(sig) // 4 + 2
                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    _lod1_parts.append(f"- '{sig}'{loc} _(score: {score:.2f})_")
                    total_tokens += tok
                    injected_symbols.add(qid)

                # LOD-2: Signatures + docstrings
                elif _lod_tier(qid) == 2:
                    sig = next(
                        (
                            sym.signature
                            for sym in block.symbols
                            if qualify_symbol(sym) == qid
                        ),
                        qid,
                    )
                    docstring = next(
                        (
                            sym.docstring
                            for sym in block.symbols
                            if qualify_symbol(sym) == qid and sym.docstring
                        ),
                        "",
                    )

                    cfg_skeleton = ""
                    if self._f.valves.enable_cfg_skeletons and (
                        active_use_case == "D"
                        or intent_vector.get("debug", 0.0)
                        >= self._f.valves.cfg_skeleton_debug_intent_threshold
                    ):
                        cfg_skeleton = (
                            self._f._symbol_index.get_cfg(qid, project_id) or ""
                        )

                    if cfg_skeleton:
                        self._f._log_debug(f"💉 CFG injected for '{qid}' (LOD2)")
                        text = f"'{sig}'"
                        if docstring:
                            text += f": {docstring}"
                        text += f"\n```python\n{cfg_skeleton}\n```"
                    else:
                        text = f"- '{sig}': {docstring}" if docstring else f"- '{sig}'"

                    tok = self._f._tokens.estimate_code_tokens(text)
                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    _lod2_parts.append(f"{text}{loc} _(score: {score:.2f})_")
                    total_tokens += tok
                    injected_symbols.add(qid)

                # LOD-3: Full code body
                else:
                    content_to_inject = CodeBlockManager.extract_symbol_body(block, qid)
                    tok = self._f._tokens.estimate_code_tokens(content_to_inject)

                    _is_oversized = (
                        self._f.valves.max_code_block_tokens > 0
                        and tok > self._f.valves.max_code_block_tokens
                    )

                    if (
                        _is_oversized
                        and self._f.valves.code_block_overflow_action == "warn"
                    ):
                        content_to_inject = self._f.valves.code_block_warn_message
                        tok = self._f._tokens.estimate_code_tokens(content_to_inject)

                    elif (
                        _is_oversized
                        and self._f.valves.code_block_overflow_action == "summarize"
                    ):
                        if block.block_summary:
                            content_to_inject = (
                                f"[Summary of {tok}-token block]\n{block.block_summary}"
                            )
                        else:
                            content_to_inject = (
                                self._f._tokens.truncate_text_to_tokens(
                                    content_to_inject,
                                    self._f.valves.max_code_block_tokens,
                                )
                                + "\n# ... [truncated — use /expand for full body]"
                            )
                        tok = self._f._tokens.estimate_code_tokens(content_to_inject)

                    elif (
                        _is_oversized
                        and self._f.valves.code_block_overflow_action == "truncate"
                    ):
                        content_to_inject = self._f._tokens.truncate_text_to_tokens(
                            content_to_inject,
                            self._f.valves.max_code_block_tokens,
                        )
                        content_to_inject += (
                            "\n# ... [truncated — use /expand for full body]"
                        )
                        tok = self._f._tokens.estimate_code_tokens(content_to_inject)

                    if (
                        slot_free
                        and self._f.valves.enable_code_compression
                        and self._f._llmlingua_compressor
                        and tok > self._f.valves.code_compression_min_tokens
                    ):
                        content_to_inject = (
                            await self._f._history_compressor.compress_code_block(
                                content_to_inject,
                                language=(
                                    block.symbols[0].language
                                    if block.symbols
                                    else "unknown"
                                ),
                                rate=self._f.valves.code_compression_rate,
                                query=query,
                            )
                        )
                        tok = self._f._tokens.estimate_code_tokens(content_to_inject)

                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    _lod3_parts.append(
                        f"### '{qid}'{loc} [activation: {score:.2f}]\n"
                        f"```\n{content_to_inject}\n```\n"
                    )
                    total_tokens += tok
                    injected_symbols.add(qid)

                break

        # ── RAPTOR cluster summaries → LOD-2 tier ─────────────────────────
        if self._f.valves.enable_raptor and getattr(self._f, "_raptor", None):
            try:
                raptor_hits = await self._f._raptor.retrieve(
                    query=query,
                    project_id=project_id,
                    top_k=3,
                    embedder=self._f.embedder,
                    chroma_collection=self._f.memory_collection,
                )
            except Exception:
                raptor_hits = []
            if raptor_hits:
                raptor_section = (
                    "### Related subsystems (RAPTOR)\n"
                    + "\n".join(f"- {h}" for h in raptor_hits)
                    + "\n\n"
                )
                _lod2_parts.insert(0, raptor_section)

        # ── Step 4: SWA-aware assembly ────────────────────────────────────
        suppress_sigs = (
            self._f.valves.skeleton_tier_suppresses_block_b_signatures
            and active_use_case != "D"
            and self._is_skeleton_tier_active(project_id)
        )

        ordered = []
        ordered.append("## Code Context (activation-based LOD)\n")
        if _lod0_parts and not suppress_sigs:
            ordered.append(
                "**Known symbols** (minimal activation):\n" + ", ".join(_lod0_parts)
            )
        if _lod1_parts and not suppress_sigs:
            ordered.append(
                "\n**Signatures** (low activation):\n" + "\n".join(_lod1_parts)
            )
        if _lod2_parts:
            ordered.append(
                "\n**Signatures + docstrings** (medium activation):\n"
                + "\n".join(_lod2_parts)
            )
        if _lod3_parts:
            ordered.append(
                "\n### Directly relevant code (high activation)\n"
                + "\n".join(_lod3_parts)
            )

        # ── E6: recency pointers ──────────────────────────────────────────
        # Compute the set of qids actually rendered in Block B
        current_b_qids = set(injected_symbols)
        _ptr = self._build_hub_recency_pointers(project_id, current_b_qids)
        if _ptr:
            ordered.append(_ptr)

        # ── Instruction tail ──────────────────────────────────────────────
        ordered.append(self._build_instruction_tail(active_use_case))

        if len(ordered) <= 1:
            if self._f.valves.debug:
                self._f._log_debug("build_block_b: no activated nodes or empty context")
            if suppress_sigs:
                self._f._log_debug(
                    "build_block_b: skeleton tier active and suppress_sigs=True, "
                    "skipping fallback to avoid duplication"
                )
                return ""
            return self._f._activation.get_active_code_context(project_id, query)

        summary_line = (
            f"\n_(Context: {len(injected_symbols)} symbols, "
            f"~{total_tokens} tokens, "
            f"{len(activated)} nodes activated)_\n"
        )
        ordered.append(summary_line)

        # ── LOD tracking for adaptive feedback ──────────────────────────
        if self._f.valves.enable_lod_adaptive:
            lod_map: Dict[str, int] = {}
            for qid, score in activated.items():
                if score < lod1:
                    lod_map[qid] = 0
                elif score < lod2:
                    lod_map[qid] = 1
                elif score < lod3:
                    lod_map[qid] = 2
                else:
                    lod_map[qid] = 3
            pstate["last_lod_levels"] = lod_map

        return "\n".join(ordered)

    async def _cleanup_old_slot_files(self, project_id: str, keep: str) -> None:
        """Delete stale slot files, keeping only the current one."""
        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        if not os.path.isdir(slot_dir):
            return
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        prefix = f"slot{self._f.valves.slot_id}_{project_slug}_"
        try:
            for fname in os.listdir(slot_dir):
                if fname.startswith(prefix) and fname != keep:
                    os.remove(os.path.join(slot_dir, fname))
                    self._f._log_debug(f"Removed obsolete slot file: {fname}")
        except Exception as e:
            self._f._log_debug(f"Slot cleanup error: {e}")


# ---------------------------------------------------------------------------
# Utility – Reentrant async lock
# ---------------------------------------------------------------------------
class ReentrantAsyncLock:
    """Reentrant asyncio lock with optional timeout to prevent deadlocks."""

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, default_timeout: float = 60.0) -> None:
        """*default_timeout* applies to every ``acquire()`` call that doesn't
        specify its own timeout."""
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count = 0
        self._default_timeout = default_timeout

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Core synchronization methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def acquire(self, timeout: Optional[float] = None) -> None:
        """Acquire the lock, reentrantly if already held by the current task.
        *timeout* overrides the instance default."""
        task = asyncio.current_task()
        if self._owner is task:
            self._count += 1
            return
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout is not None:
            await asyncio.wait_for(self._lock.acquire(), timeout=effective_timeout)
        else:
            await self._lock.acquire()
        self._owner = task
        self._count = 1

    def release(self) -> None:
        """Release the lock once.  Raises ``RuntimeError`` if the current
        task does not own the lock."""
        task = asyncio.current_task()
        if self._owner is not task:
            raise RuntimeError("Lock not owned by current task")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Async context manager support
    # ═══════════════════════════════════════════════════════════════════════════

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        self.release()


# ---------------------------------------------------------------------------
# SymbolIndex – central name→block mapping and typed edges
# ---------------------------------------------------------------------------
class SymbolIndex:
    """Central index that stores every known symbol under a **qualified id**
    (``ClassName.method`` or ``module.function``) so that methods with the
    same bare name in different classes never collide.

    Provides:
    * Block‑hash lookup by qualified id, with bare‑name fallback that
      returns **all** matching symbols (inclusive, not last‑writer‑wins).
    * Typed call edges (``calls``, ``data_flow``, …) between symbols.
    * Per‑symbol metadata: signature, docstring, kind, file path, line span.
    * PageRank centrality over the qualified call graph.

    Use ``get_all_names()`` for coarse text matching, ``get_all_qualified_names()``
    when every distinct symbol must be visible (inventories, hashes, centrality).
    """

    MAX_ENTRIES = 10_000

    def __init__(self) -> None:
        # Primary storage, indexed by (project_id, qualified_id).
        self._name_to_blocks: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._stats: Counter = Counter()
        self._symbol_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Reverse index: (project_id, bare_name) -> {qualified_id, ...}.
        # Lets every bare‑name‑based consumer (query‑word matching,
        # /expand <bare>, traceback frame names, ...) resolve to the full
        # set of real symbols that share that name, instead of silently
        # picking whichever was indexed last.
        self._bare_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # Legacy call relationships, kept for compatibility with
        # find_entry_points() and any external consumer of get_callers().
        # Destinations remain bare (the callee's identity is generally not
        # resolvable without type inference); values are now caller qualified
        # ids instead of caller bare names.
        self._callee_to_callers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # Typed edges (v7+).  Sources are qualified ids (we always know
        # which symbol we are walking); destinations stay bare, best-effort
        # (see class docstring + qualify_symbol_name()).
        self._edges_out: Dict[str, List[Edge]] = defaultdict(list)
        self._edges_in: Dict[str, List[Edge]] = defaultdict(list)

        # Centrality cache (v8), now keyed by qualified id.
        self._centrality_cache: Dict[str, Dict[str, float]] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Symbol registration & removal (qualified id + bare index)
    # ═══════════════════════════════════════════════════════════════════════════

    def add(self, symbol: "CodeSymbol", block_hash: str, project_id: str) -> None:
        """Register *symbol* in the index, keyed by its qualified id.
        Updates the bare‑name reverse index and call‑relationship storage."""
        qid = qualify_symbol_name(symbol.name, symbol.parent_symbol, symbol.file_path)
        key = (project_id, qid)
        self._name_to_blocks[key].add(block_hash)
        self._stats[key] += 1
        self._bare_index[(project_id, symbol.name)].add(qid)

        prev = self._symbol_meta.get(key)
        self._symbol_meta[key] = {
            "name": symbol.name,
            "signature": symbol.signature,
            "docstring": symbol.docstring
            or (prev.get("docstring", "") if prev else ""),
            "file_path": symbol.file_path,
            "language": symbol.language,
            "kind": symbol.kind,
            "parent_symbol": symbol.parent_symbol
            or (prev.get("parent_symbol", "") if prev else ""),
            "line_start": symbol.line_start,
            "cfg_skeleton": "",  # NUEVO — lazy CFG skeleton (populated by ensure_cfg_batch)
            "cfg_body_hash": "",  # NUEVO — hash of the source body that generated cfg_skeleton
        }

        # Log para confirmar que los campos CFG se han inicializado
        logger.debug(
            f"[CFG] SymbolIndex.add: initialized CFG fields for '{qid}' "
            f"(kind={symbol.kind}, line_start={symbol.line_start}, line_end={symbol.line_end})"
        )

        for callee in symbol.calls:
            callee_key = (project_id, callee)
            self._callee_to_callers[callee_key].add(qid)
        self._evict_if_needed()

    def remove(self, symbol: "CodeSymbol", block_hash: str, project_id: str) -> None:
        """Remove *symbol* from the index.  If the block hash was the last
        reference to that qualified id, the entry is fully deleted from all
        internal structures."""
        qid = qualify_symbol_name(symbol.name, symbol.parent_symbol, symbol.file_path)
        key = (project_id, qid)
        s = self._name_to_blocks.get(key)
        if s:
            s.discard(block_hash)
            if not s:
                del self._name_to_blocks[key]
                self._stats.pop(key, None)
                self._symbol_meta.pop(key, None)
                bare_key = (project_id, symbol.name)
                bare_set = self._bare_index.get(bare_key)
                if bare_set:
                    bare_set.discard(qid)
                    if not bare_set:
                        del self._bare_index[bare_key]

    def remove_all_for_block(
        self, block_hash: str, symbols: List["CodeSymbol"], project_id: str
    ) -> None:
        """Remove every symbol belonging to *block_hash* and their edges."""
        for sym in symbols:
            self.remove(sym, block_hash, project_id)
            qid = qualify_symbol_name(sym.name, sym.parent_symbol, sym.file_path)
            self.remove_edges_for_symbol(qid, project_id)

    def _evict_if_needed(self) -> None:
        """
        Drop the least‑frequently‑added entry when the index exceeds
        ``MAX_ENTRIES``, keeping memory bounded.
        """
        while len(self._name_to_blocks) > self.MAX_ENTRIES:
            # Get the least common entry (by add count)
            least_common = self._stats.most_common()[-1][0]
            project_id, qid = least_common

            # --- 1. Remove all edges (in/out) that reference this symbol ---
            # This must be done before deleting the symbol entry, otherwise
            # edges would dangle. remove_edges_for_symbol uses qid (subsystem 04).
            self.remove_edges_for_symbol(qid, project_id)

            # --- 2. Remove the symbol entry itself ---
            meta = self._symbol_meta.get(least_common, {})
            bare = meta.get("name", qid)
            bare_key = (project_id, bare)
            bare_set = self._bare_index.get(bare_key)
            if bare_set:
                bare_set.discard(qid)
                if not bare_set:
                    del self._bare_index[bare_key]

            del self._name_to_blocks[least_common]
            self._stats.pop(least_common, None)
            self._symbol_meta.pop(least_common, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Name & symbol resolution (qualified, bare, and cross-reference)
    # ═══════════════════════════════════════════════════════════════════════════

    def find_blocks(self, name_or_qid: str, project_id: str) -> Set[str]:
        """Block hashes for a symbol.  Exact qualified-id match first; if
        not found, falls back to a bare‑name lookup that UNIONS all blocks
        of every symbol sharing that bare name (e.g. every __init__ of every
        class), instead of returning just one."""
        exact = self._name_to_blocks.get((project_id, name_or_qid))
        if exact is not None:
            return exact
        qids = self._bare_index.get((project_id, name_or_qid))
        if not qids:
            return set()
        result: Set[str] = set()
        for qid in qids:
            result |= self._name_to_blocks.get((project_id, qid), set())
        return result

    def get_all_names(self, project_id: str) -> Set[str]:
        """Bare names indexed in this project (deduplicated).  Use for
        coarse text/query matching, where the caller doesn't know — and
        doesn't need to know — which concrete class a name belongs to."""
        return {bare for (pid, bare) in self._bare_index if pid == project_id}

    def get_all_qualified_names(self, project_id: str) -> Set[str]:
        """Every distinct symbol identity in the project (one entry per real
        occurrence, e.g. each class's __init__ separately).  Use instead of
        get_all_names() where every real symbol must be visible —
        inventories, skeleton hash, centrality."""
        return {qid for (pid, qid) in self._symbol_meta if pid == project_id}

    def get_qualified_names_for(self, bare_name: str, project_id: str) -> Set[str]:
        """All qualified ids that share this bare name (e.g. every class's
        __init__).  If nothing is indexed under that bare name, returns
        {bare_name} as-is — defensive, and also makes passing an ALREADY-
        qualified id through here (not found as bare) a correct no-op,
        returning it unchanged."""
        qids = self._bare_index.get((project_id, bare_name))
        return set(qids) if qids else {bare_name}

    def get_symbol_meta(
        self, name_or_qid: str, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Full metadata dict for a symbol (signature, docstring, kind, …),
        or ``None`` if not found."""
        return self._resolve_meta(name_or_qid, project_id)

    def get_parent_symbol(self, name_or_qid: str, project_id: str) -> str:
        """Enclosing class name for a symbol, or ``""`` if it is top-level
        or the symbol is not found."""
        meta = self._resolve_meta(name_or_qid, project_id)
        return meta.get("parent_symbol", "") if meta else ""

    def get_class_members(self, class_name: str, project_id: str) -> List[str]:
        """Qualified ids of every member of `class_name`, ordered by
        source order (line_start) then by id.  Unlike before, this now
        returns ALL members correctly even when other classes in the project
        have methods sharing the same bare names."""
        members = []
        for (pid, qid), meta in self._symbol_meta.items():
            if pid == project_id and meta.get("parent_symbol") == class_name:
                members.append(qid)

        def _line_start(qid: str) -> int:
            meta = self._symbol_meta.get((project_id, qid), {})
            val = meta.get("line_start")
            return val if val is not None else 999999

        return sorted(members, key=lambda q: (_line_start(q), q))

    def get_classes(self, project_id: str) -> Set[str]:
        """Return every class name that has at least one indexed member,
        plus every symbol whose kind is ``"class"``."""
        classes = set()
        for (pid, qid), meta in self._symbol_meta.items():
            if pid != project_id:
                continue
            if meta.get("kind") == "class":
                classes.add(meta.get("name", qid))
            parent = meta.get("parent_symbol")
            if parent:
                classes.add(parent)
        return classes

    def get_signature(self, name_or_qid: str, project_id: str) -> Optional[str]:
        """Signature string for a symbol, or ``None``."""
        meta = self._resolve_meta(name_or_qid, project_id)
        return meta.get("signature") if meta else None

    def get_docstring(self, name_or_qid: str, project_id: str) -> str:
        """One-line docstring for a symbol, or ``""``."""
        meta = self._resolve_meta(name_or_qid, project_id)
        return meta.get("docstring", "") if meta else ""

    def update_cfg(
        self, qid: str, project_id: str, cfg_skeleton: str, body_hash: str
    ) -> None:
        """
        Store a symbol's control-flow skeleton and the body_hash it was derived
        from. Unlike update_docstring, this ONLY updates an exact qualified-id
        match — never a bare-name fallback. A CFG skeleton is specific to one
        concrete function body; applying it to "every symbol sharing this bare
        name" (as update_docstring does for resilience) would silently show the
        wrong control flow for a same-named method in a different class.
        """
        key = (project_id, qid)
        if key in self._symbol_meta:
            self._symbol_meta[key]["cfg_skeleton"] = cfg_skeleton
            self._symbol_meta[key]["cfg_body_hash"] = body_hash
            logger.debug(
                f"[CFG] SymbolIndex.update_cfg: stored CFG for '{qid}' "
                f"(body_hash={body_hash}, skeleton_len={len(cfg_skeleton)} chars)"
            )
        else:
            logger.debug(
                f"[CFG] SymbolIndex.update_cfg: key '{qid}' NOT found in _symbol_meta — "
                "CFG not stored (symbol may have been evicted)"
            )

    def get_cfg(self, qid: str, project_id: str) -> Optional[str]:
        """Return the cached CFG skeleton for an exact qualified id, or None.
        No bare-name fallback — see update_cfg()."""
        meta = self._symbol_meta.get((project_id, qid))
        if meta:
            skeleton = meta.get("cfg_skeleton")
            if skeleton:
                logger.debug(f"[CFG] SymbolIndex.get_cfg: found CFG for '{qid}'")
                return skeleton
            else:
                logger.debug(
                    f"[CFG] SymbolIndex.get_cfg: '{qid}' exists but cfg_skeleton is empty"
                )
        else:
            logger.debug(f"[CFG] SymbolIndex.get_cfg: no metadata for '{qid}'")
        return None

    def get_file_for_symbol(self, name_or_qid: str, project_id: str) -> Optional[str]:
        """File path for a symbol, or ``None``."""
        meta = self._resolve_meta(name_or_qid, project_id)
        return meta.get("file_path") if meta else None

    def update_docstring(
        self, name_or_qid: str, project_id: str, docstring: str
    ) -> None:
        """Update a symbol's docstring.  An exact qid match updates only that
        one occurrence — every new call site (background docstring
        generation, batch generation) now passes a qualified id and gets
        this precise behaviour.  A bare‑name call (legacy compatibility)
        updates every symbol sharing that name, which is safer than the old
        silent-overwrite-of-one but is still an approximation — prefer
        passing the qualified id when you have it."""
        key = (project_id, name_or_qid)
        if key in self._symbol_meta:
            self._symbol_meta[key]["docstring"] = docstring
            return
        qids = self._bare_index.get((project_id, name_or_qid))
        if qids:
            for qid in qids:
                meta = self._symbol_meta.get((project_id, qid))
                if meta is not None:
                    meta["docstring"] = docstring

    def _resolve_meta(
        self, name_or_qid: str, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Exact qualified-id match first; if missing, a deterministic
        choice among all symbols sharing that bare name.  Anyone who already
        knows the qualified id (e.g. anything iterating
        get_all_qualified_names(), or holding a CodeSymbol with its own
        parent_symbol) should pass the qid directly for an unambiguous
        answer.  Anyone with only a bare name (regex matches,
        /expand <bare> typed by the user, ...) gets a single representative
        entry — better than nothing, but inherently ambiguous when several
        classes share that method name."""
        key = (project_id, name_or_qid)
        meta = self._symbol_meta.get(key)
        if meta is not None:
            return meta
        qids = self._bare_index.get((project_id, name_or_qid))
        if not qids:
            return None
        chosen = sorted(qids)[0]
        return self._symbol_meta.get((project_id, chosen))

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Call relationships (legacy)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_callers(self, callee_name: str, project_id: str) -> Set[str]:
        """Qualified caller ids for a callee name (necessarily bare)."""
        return self._callee_to_callers.get((project_id, callee_name), set())

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Typed edges (v7+)
    # ═══════════════════════════════════════════════════════════════════════════

    def add_edge(self, edge: "Edge", project_id: str) -> None:
        """Register a typed edge.  `edge.src` is expected to be the
        qualified id of the symbol whose body contains the call (the caller
        must qualify it themselves with qualify_symbol_name() before
        building the Edge — SymbolIndex has no way to know a symbol's
        parent_symbol from a bare string alone).  `edge.dst` stays whatever
        the extractor produced (best-effort, normally bare)."""
        src_key = f"{project_id}:{edge.src}"
        dst_key = f"{project_id}:{edge.dst}"
        existing = self._edges_out.get(src_key, [])
        for e in existing:
            if e.dst == edge.dst and e.type == edge.type:
                return
        self._edges_out[src_key].append(edge)
        self._edges_in[dst_key].append(edge)

    def remove_edges_for_symbol(self, symbol_id: str, project_id: str) -> None:
        """Remove edges where `symbol_id` (qualified id for a class‑scoped
        symbol, bare name for a module‑level one) is source or destination."""
        src_key = f"{project_id}:{symbol_id}"
        for edge in self._edges_out.pop(src_key, []):
            dst_key = f"{project_id}:{edge.dst}"
            self._edges_in[dst_key] = [
                e for e in self._edges_in.get(dst_key, []) if e.src != symbol_id
            ]
        dst_key = f"{project_id}:{symbol_id}"
        for edge in self._edges_in.pop(dst_key, []):
            src_key_in = f"{project_id}:{edge.src}"
            self._edges_out[src_key_in] = [
                e for e in self._edges_out.get(src_key_in, []) if e.dst != symbol_id
            ]

    def get_edges_out(self, symbol_id: str, project_id: str) -> List["Edge"]:
        """Outgoing edges.  Pass a method's qualified id for precisely its
        own calls; a module‑level function's bare name works as-is."""
        return self._edges_out.get(f"{project_id}:{symbol_id}", [])

    def get_edges_in(self, callee_name: str, project_id: str) -> List["Edge"]:
        """Incoming edges for a callee name (necessarily bare, best-effort).
        May include callers from an unrelated symbol that shares that bare
        name — there is no general way to know which class's method a call
        `obj.method()` actually resolves to without type inference; this is
        the inherent and documented limitation."""
        return self._edges_in.get(f"{project_id}:{callee_name}", [])

    def get_all_edges_out(self, project_id: str) -> Dict[str, List["Edge"]]:
        prefix = f"{project_id}:"
        return {
            key[len(prefix) :]: edges
            for key, edges in self._edges_out.items()
            if key.startswith(prefix)
        }

    def get_all_edges_in(self, project_id: str) -> Dict[str, List["Edge"]]:
        prefix = f"{project_id}:"
        inverted: Dict[str, List["Edge"]] = defaultdict(list)
        for key, edges in self._edges_out.items():
            if not key.startswith(prefix):
                continue
            for e in edges:
                inverted[e.dst].append(
                    Edge(
                        src=e.dst,
                        dst=e.src,
                        type=e.type,
                        weight=e.weight,
                        confidence=e.confidence,
                    )
                )
        return dict(inverted)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Centrality (PageRank)
    # ═══════════════════════════════════════════════════════════════════════════

    def precompute_centrality(
        self,
        project_id: str,
        alpha: float = 0.85,
        max_steps: int = 30,
        tolerance: float = 1e-7,
    ) -> Dict[str, float]:
        """PageRank over QUALIFIED symbol identities, so e.g. fifteen
        __init__ with the same name are fifteen separate nodes instead of
        one node whose edges were silently fused.

        Each edge's destination (best-effort, normally bare) is resolved
        against every qualified node sharing that bare name — if a name is
        ambiguous (multiple classes with a method of that same name), the
        contribution is split among all candidates instead of being silently
        lost."""
        names = list(self.get_all_qualified_names(project_id))
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            scores = {names[0]: 1.0}
            self._store_centrality(project_id, scores)
            return scores

        idx = {name: i for i, name in enumerate(names)}

        bare_to_indices: Dict[str, List[int]] = {}
        for (pid, bare), qids in self._bare_index.items():
            if pid != project_id:
                continue
            indices = [idx[q] for q in qids if q in idx]
            if indices:
                bare_to_indices[bare] = indices

        out_links: list = [[] for _ in range(n)]
        for name in names:
            i = idx[name]
            for edge in self._edges_out.get(f"{project_id}:{name}", []):
                if edge.dst in idx:
                    out_links[i].append(idx[edge.dst])
                else:
                    for j in bare_to_indices.get(edge.dst, []):
                        out_links[i].append(j)

        rank = [1.0 / n] * n
        base = (1.0 - alpha) / n
        dangling_nodes = [i for i in range(n) if not out_links[i]]

        for _ in range(max_steps):
            new_rank = [base] * n
            dangling_sum = sum(rank[i] for i in dangling_nodes)
            if dangling_sum:
                share = alpha * dangling_sum / n
                for k in range(n):
                    new_rank[k] += share
            for i in range(n):
                links = out_links[i]
                if not links:
                    continue
                contrib = alpha * rank[i] / len(links)
                for j in links:
                    new_rank[j] += contrib
            delta = sum(abs(new_rank[k] - rank[k]) for k in range(n))
            rank = new_rank
            if delta < tolerance:
                break

        max_r = max(rank) if rank else 1.0
        if max_r <= 0:
            scores = {name: 0.0 for name in names}
        else:
            scores = {names[k]: rank[k] / max_r for k in range(n)}

        self._store_centrality(project_id, scores)
        return scores

    def get_hub_symbols(
        self, project_id: str, centrality: Dict[str, float], top_n: int
    ) -> List[Tuple[str, float]]:
        """Top‑N symbols by centrality, sorted by descending score.
        Falls back to the cached scores from the last
        ``precompute_centrality()`` call if *centrality* is empty."""
        if not centrality:
            centrality = getattr(self, "_centrality_cache", {}).get(project_id, {})
        if not centrality or top_n <= 0:
            return []
        ranked = sorted(centrality.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_n]

    def _store_centrality(self, project_id: str, scores: Dict[str, float]) -> None:
        """Cache centrality scores for cheap re-reads by ``get_hub_symbols()``."""
        self._centrality_cache[project_id] = scores

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Skeleton & signature hashes
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_signature_hash(self, project_id: str) -> str:
        """Stable hash of all symbol signatures (not bodies).  Changes only
        when symbols are added/removed/renamed or a signature changes."""
        qids = sorted(self.get_all_qualified_names(project_id))
        if not qids:
            return ""
        parts = []
        for qid in qids:
            meta = self._symbol_meta.get((project_id, qid), {})
            name = meta.get("name", qid)
            sig = meta.get("signature") or name
            parent = meta.get("parent_symbol", "")
            parts.append(f"{parent}\x1f{name}\x1f{sig}")
        blob = "\x1e".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def compute_skeleton_hash(self, project_id: str) -> str:
        """Stable hash of the skeleton tier (signatures + docstrings)."""
        qids = sorted(self.get_all_qualified_names(project_id))
        if not qids:
            return ""
        parts = []
        for qid in qids:
            meta = self._symbol_meta.get((project_id, qid), {})
            name = meta.get("name", qid)
            sig = meta.get("signature") or name
            parent = meta.get("parent_symbol", "")
            doc = meta.get("docstring", "")
            parts.append(f"{parent}\x1f{name}\x1f{sig}\x1f{doc}")
        blob = "\x1e".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def compute_structure_hash(self, project_id: str) -> str:
        """
        Hash of symbol signatures and structure only, excluding docstrings.
        Used for KV‑cache and slot persistence stability.

        This hash changes only when the symbol set or signatures change,
        not when docstrings are added/updated. This keeps the Block A
        prefix hash stable across the docstring-fill-in period, preventing
        spurious KV-cache misses and slot-restore failures.

        Args:
            project_id (str): The project identifier.

        Returns:
            str: A 16-character hex hash, or empty string if no symbols exist.
        """
        qids = sorted(self.get_all_qualified_names(project_id))
        if not qids:
            return ""
        parts = []
        for qid in qids:
            meta = self._symbol_meta.get((project_id, qid), {})
            name = meta.get("name", qid)
            sig = meta.get("signature") or name
            parent = meta.get("parent_symbol", "")
            # Include only structure: parent, name, signature.
            # Docstrings are excluded deliberately.
            parts.append(f"{parent}\x1f{name}\x1f{sig}")
        blob = "\x1e".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Project lifecycle & cleanup
    # ═══════════════════════════════════════════════════════════════════════════

    def clear_project(self, project_id: str) -> None:
        """Remove every symbol, edge, and metadata entry for *project_id*."""
        keys_to_remove = [key for key in self._name_to_blocks if key[0] == project_id]
        for key in keys_to_remove:
            del self._name_to_blocks[key]
            self._stats.pop(key, None)

        bare_keys = [key for key in self._bare_index if key[0] == project_id]
        for key in bare_keys:
            del self._bare_index[key]

        inv_keys = [key for key in self._callee_to_callers if key[0] == project_id]
        for key in inv_keys:
            del self._callee_to_callers[key]

        prefix = f"{project_id}:"
        for k in list(self._edges_out.keys()):
            if k.startswith(prefix):
                del self._edges_out[k]
        for k in list(self._edges_in.keys()):
            if k.startswith(prefix):
                del self._edges_in[k]

        self._centrality_cache.pop(project_id, None)

        meta_keys = [key for key in self._symbol_meta if key[0] == project_id]
        for key in meta_keys:
            del self._symbol_meta[key]

    def clear(self) -> None:
        """Drop all in‑memory data for all projects."""
        self._name_to_blocks.clear()
        self._bare_index.clear()
        self._callee_to_callers.clear()
        self._stats.clear()
        self._edges_out.clear()
        self._edges_in.clear()
        self._centrality_cache.clear()
        self._symbol_meta.clear()

    # ── Internal helpers (iteration) ─────────────────────────────────────

    def _iter_names(self, project_id: str):
        return iter(self.get_all_qualified_names(project_id))

    def _iter_out_edges(self, project_id: str, name: str):
        key = f"{project_id}:{name}"
        for edge in self._edges_out.get(key, []):
            yield edge.dst

    def _symbol_line_start(self, name_or_qid: str, project_id: str) -> int:
        meta = self._resolve_meta(name_or_qid, project_id)
        if meta is None:
            return 999999
        val = meta.get("line_start")
        return val if val is not None else 999999


# ---------------------------------------------------------------------------
# SignatureExtractor – tree‑sitter based symbol and call extraction
# ---------------------------------------------------------------------------
class SignatureExtractor:
    """
    Extracts ``CodeSymbol`` lists and call relationships from source code
    using tree‑sitter for precise, qualified results.

    Each returned symbol carries a **qualified identity** (``ClassName.method``
    or ``module.function``) via its ``parent_symbol`` field, so downstream
    indexing never confuses same‑named methods from different classes.

    When tree‑sitter is unavailable or the language cannot be detected, an
    empty list is returned and a warning is logged — no unqualified fallback
    data is ever produced.

    Docstrings are extracted statically for Python via the ``ast`` module;
    for other languages (or when the source lacks a docstring), they are
    filled in later by the LLM‑driven ``ensure_docstrings_batch`` path.

    Caching:
        Results are cached with a 1-hour TTL to avoid re-parsing the same
        code block multiple times within a session. The cache stores raw
        symbols (before parent enrichment) and returns deep copies to
        prevent mutation of cached entries.
    """

    MAX_PARSE_SIZE_BYTES = 5_000_000
    _LANG_MAP: Dict[str, str] = {
        "py": "python",
        "js": "javascript",
        "mjs": "javascript",
        "jsx": "tsx",
        "ts": "tsx",
        "tsx": "tsx",
        "go": "go",
        "rs": "rust",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "h": "cpp",
        "hpp": "cpp",
    }

    _parser_cache: Dict[str, Any] = {}
    _parser_cache_lock = threading.Lock()

    # ── extraction cache ──────────────────────────────────────────────
    _extraction_cache: Dict[str, Tuple[List["CodeSymbol"], float]] = {}
    _EXTRACTION_CACHE_MAXSIZE: int = 128
    _EXTRACTION_CACHE_TTL: int = 3600  # 1 hour
    _EXTRACTION_CACHE_LOCK = threading.Lock()

    @staticmethod
    def _cache_key(code: str, file_path: Optional[str], language: Optional[str]) -> str:
        """
        Generate a deterministic cache key for a code extraction request.
        """
        h = hashlib.md5(code.encode()).hexdigest()[:16]
        return f"{h}|{file_path or ''}|{language or ''}"

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Language detection
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _guess_language(file_path: Optional[str], code: str) -> str:
        """
        Heuristically determine the programming language of a code block.

        Resolution order (first match wins):
        1. tree-sitter extension detection if *file_path* is provided.
        2. Extension‑to‑language map (``_LANG_MAP``).
        3. Source‑code heuristics: Python keywords (def, class, import, from, async def)
           or JavaScript 'function' keyword.
        4. ``"unknown"`` — callers will skip tree-sitter extraction.
        """
        if file_path and HAS_TREE_SITTER:
            try:
                return detect_language_from_extension(
                    file_path.rsplit(".", 1)[-1].lower()
                )
            except Exception:
                pass
        if file_path:
            ext = file_path.rsplit(".", 1)[-1].lower()
            return SignatureExtractor._LANG_MAP.get(ext, "unknown")

        # ── Improved heuristics for code without an extension ──────────────
        # Look for multiple Python-specific patterns, not just 'def'
        if re.search(r"\b(?:def|class|import|from|async def)\s+\w+", code):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code):
            return "javascript"

        # Fallback: if it contains braces and semicolons, it could be C/Java,
        # but we leave it as unknown for now to avoid misdetection.
        return "unknown"

    @staticmethod
    def _infer_code_language(code_snippet: str) -> str:
        """Simple heuristic language detection for a code snippet."""
        if re.search(r"\bdef\s+\w+\s*\(", code_snippet):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code_snippet):
            return "javascript"
        return "unknown"

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Main extraction entry point (with caching)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    async def extract_async(
        code: str, file_path: Optional[str] = None, language: Optional[str] = None
    ) -> List["CodeSymbol"]:
        """
        Extract symbols and call relationships from source code using tree-sitter.

        Returns qualified symbols (``ClassName.method`` / ``module.function``).
        Falls back to an empty list when tree-sitter is unavailable or fails,
        logging a warning — no fallback extraction is attempted, to avoid
        corrupting the symbol index with unqualified data.

        Results are cached with a 1-hour TTL to avoid re-parsing the same
        code block multiple times. The cache stores raw symbols (before
        parent enrichment) and returns deep copies.
        """
        # ── Cache check (before any validation) ──────────────────────────
        cache_key = SignatureExtractor._cache_key(code, file_path, language)
        with SignatureExtractor._EXTRACTION_CACHE_LOCK:
            if cache_key in SignatureExtractor._extraction_cache:
                cached_symbols, ts = SignatureExtractor._extraction_cache[cache_key]
                if time.time() - ts < SignatureExtractor._EXTRACTION_CACHE_TTL:
                    # Return deep copies to prevent mutation of cached entries.
                    return [s.copy() for s in cached_symbols]
                else:
                    del SignatureExtractor._extraction_cache[cache_key]

        # ── Size validation ──────────────────────────────────────────────
        if len(code.encode()) > SignatureExtractor.MAX_PARSE_SIZE_BYTES:
            return []

        if not HAS_TREE_SITTER:
            logger.warning(
                "tree-sitter not available — skipping symbol extraction. "
                "Install tree-sitter-language-pack to enable code-aware features."
            )
            return []

        lang = language or SignatureExtractor._guess_language(file_path, code)
        if lang == "unknown":
            logger.warning(
                "Could not detect language for code block — skipping symbol extraction."
            )
            return []

        # ── Parse ──────────────────────────────────────────────────────────
        try:
            loop = asyncio.get_event_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(
                    None, SignatureExtractor._parse_sync, code.encode(), lang
                ),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"tree-sitter parse failed for language '{lang}': {e} — "
                "skipping symbol extraction to avoid corrupt fallback data."
            )
            return []

        # ── Extract symbols and calls ─────────────────────────────────────
        syms = SignatureExtractor._extract_symbols_from_tree(
            tree, lang, code, file_path
        )
        call_map = SignatureExtractor._extract_calls_from_tree(tree, lang, code)
        del tree

        # ── Populate call relationships ──────────────────────────────────
        for sym in syms:
            qid = qualify_symbol_name(sym.name, sym.parent_symbol, sym.file_path)
            calls = list(call_map.get(qid, []))
            if qid != sym.name:
                for c in call_map.get(sym.name, []):
                    if c not in calls:
                        calls.append(c)
            sym.calls = calls

        # ── Python docstring extraction (static) ─────────────────────────
        if lang == "python" or (file_path and file_path.endswith(".py")):
            SignatureExtractor._extract_docstrings_python(code, syms)

        # ── Cache store (after successful extraction) ────────────────────
        with SignatureExtractor._EXTRACTION_CACHE_LOCK:
            if (
                len(SignatureExtractor._extraction_cache)
                >= SignatureExtractor._EXTRACTION_CACHE_MAXSIZE
            ):
                # Evict oldest entry (by timestamp)
                oldest_key = min(
                    SignatureExtractor._extraction_cache,
                    key=lambda k: SignatureExtractor._extraction_cache[k][1],
                )
                del SignatureExtractor._extraction_cache[oldest_key]
            SignatureExtractor._extraction_cache[cache_key] = (syms, time.time())

        return syms

    @staticmethod
    def _parse_sync(code_bytes: bytes, lang: str):
        """
        Parse source-code bytes synchronously with a fresh tree-sitter parser.

        Creates a new parser instance on every call. This avoids thread-safety
        issues: tree-sitter Parser is not Send/Sync and cannot be shared across
        threads [2†L11]. Creating a parser is cheap compared to parsing.

        Supports both old API (set_language) and new API (language in constructor).
        See: https://tree-sitter.github.io/tree-sitter/ [3†L19-L22]

        Returns the root ``tree_sitter.Node`` of the concrete syntax tree.
        """
        from tree_sitter import Parser as TSParser
        from tree_sitter_language_pack import get_language

        lang_obj = get_language(lang)

        # New API (py-tree-sitter >= 0.23.0): language passed to constructor.
        try:
            parser = TSParser(lang_obj)
        except TypeError:
            # Old API (py-tree-sitter < 0.23.0): use set_language().
            parser = TSParser()
            if hasattr(parser, "set_language"):
                parser.set_language(lang_obj)
            else:
                raise RuntimeError(
                    f"Unsupported tree-sitter version: cannot set language '{lang}'"
                )

        return parser.parse(code_bytes)

    @staticmethod
    def enrich_symbols_with_parent_info(
        symbols: List["CodeSymbol"], full_code: str
    ) -> List["CodeSymbol"]:
        """Assign parent_symbol using AST line-range mapping.
        For symbols that come from the generic extractor (line_start=None),
        the line number is recovered by scanning the source code for the
        definition (def / class / async def) before matching.
        """
        import ast as ast_module

        # ── 1. Recover missing line numbers by scanning the source ──────
        lines = full_code.split("\n")
        for sym in symbols:
            if sym.line_start is not None:
                continue
            # Build a regex that matches the definition of this symbol
            if sym.kind in ("function", "method"):
                pattern = re.compile(
                    r"^\s*(?:async\s+)?def\s+" + re.escape(sym.name) + r"\b"
                )
            elif sym.kind == "class":
                pattern = re.compile(r"^\s*class\s+" + re.escape(sym.name) + r"\b")
            else:
                continue
            for i, line in enumerate(lines, start=1):
                if pattern.search(line):
                    sym.line_start = i
                    break

        # ── 2. Build the AST class‑line mapping using DFS (not BFS) ──
        try:
            tree = ast_module.parse(full_code)
        except SyntaxError:
            return symbols

        line_to_class: Dict[int, str] = {}

        def _visit(node, current_class: str) -> None:
            """Recursive DFS: current_class is the nearest enclosing class name
            (or "" at module level). Local functions nested inside a method keep
            the method's enclosing class — matching the heuristic tree-sitter
            already uses elsewhere in this file."""
            for child in ast_module.iter_child_nodes(node):
                if isinstance(child, ast_module.ClassDef):
                    end_lineno = getattr(child, "end_lineno", child.lineno)
                    for lineno in range(child.lineno, end_lineno + 1):
                        line_to_class[lineno] = current_class
                    _visit(child, child.name)
                elif isinstance(
                    child, (ast_module.FunctionDef, ast_module.AsyncFunctionDef)
                ):
                    end_lineno = getattr(child, "end_lineno", child.lineno)
                    for lineno in range(child.lineno, end_lineno + 1):
                        line_to_class[lineno] = current_class
                    _visit(child, current_class)
                else:
                    _visit(child, current_class)

        _visit(tree, "")

        # ── 3. Assign parent_symbol where possible ─────────────────────
        assigned = 0
        for sym in symbols:
            if sym.parent_symbol:
                continue  # tree-sitter already resolved this correctly — keep it
            if sym.line_start and sym.line_start in line_to_class:
                parent = line_to_class[sym.line_start]
                if parent and parent != sym.name:
                    sym.parent_symbol = parent
                    if sym.kind == "function":
                        sym.kind = "method"
                    assigned += 1

        return symbols

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Symbol extraction from tree-sitter
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_symbols_from_tree(
        tree, lang: str, code: str, file_path: Optional[str]
    ) -> List["CodeSymbol"]:
        """
        Extract symbols from the tree-sitter AST.

        MIGRATED (step 18): Now uses the parent function/class definition node
        to obtain `line_start` and `line_end`, ensuring that multi-line symbols
        (e.g., functions with bodies) have a range larger than one line.
        Previously the name-capture node (one line) was used, causing
        `line_start == line_end` for every symbol, breaking docstring and CFG
        generation.

        Args:
            tree: The tree-sitter parse tree.
            lang (str): The programming language.
            code (str): The source code.
            file_path (Optional[str]): The file path, if available.

        Returns:
            List[CodeSymbol]: The extracted symbols with correct line ranges.
        """
        query_str = FALLBACK_LANGUAGE_QUERIES.get(lang)
        if not query_str:
            logger.warning(
                f"No tree-sitter query defined for language '{lang}' — "
                "skipping symbol extraction to avoid corrupt fallback data."
            )
            return []
        try:
            lang_obj = get_language(lang)
            query = lang_obj.query(query_str)
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)  # dict: {capture_name: [nodes]}

            symbols = []
            func_types = (
                "function_definition",
                "function_declaration",
                "method_declaration",
                "function_item",
                "arrow_function",
                "function_expression",
            )
            class_types = (
                "class_definition",
                "class_declaration",
                "type_spec",
                "struct_item",
                "enum_item",
                "class_specifier",
            )
            # Combined types for the walk-up
            definition_types = func_types + class_types

            for cap_name, nodes in captures.items():
                if cap_name != "name":
                    continue
                for node in nodes:
                    parent = node.parent
                    kind = "unknown"
                    while parent:
                        if parent.type in func_types:
                            kind = "function"
                            break
                        elif parent.type in class_types:
                            kind = "class"
                            break
                        parent = parent.parent

                    parent_symbol = ""
                    walker = node.parent
                    if walker is not None:
                        walker = walker.parent
                    while walker:
                        if walker.type in class_types:
                            name_node = walker.child_by_field_name("name")
                            if name_node:
                                parent_symbol = name_node.text.decode("utf-8")
                            break
                        walker = walker.parent
                    if kind == "function" and parent_symbol:
                        kind = "method"

                    sig = (
                        parent.text.decode("utf-8").split("\n")[0].strip()[:200]
                        if parent
                        else node.text.decode("utf-8")
                    )
                    name = node.text.decode("utf-8")

                    # --- Find the definition node (function or class) for the line range ---
                    # Walk up from the name node until we hit a definition node.
                    # If none is found, fall back to the name node itself.
                    span_node = node
                    while (
                        span_node.parent is not None
                        and span_node.parent.type not in definition_types
                    ):
                        span_node = span_node.parent
                    if span_node.parent is not None:
                        span_node = span_node.parent  # Now it's the definition node

                    # Use the definition node's start/end points for the line range.
                    # tree-sitter points are 0-indexed; convert to 1-indexed for storage.
                    line_start = span_node.start_point[0] + 1
                    line_end = span_node.end_point[0] + 1

                    # Debug log to confirm the change
                    logger.debug(
                        f"[EXTRACT] Symbol '{name}' kind={kind}: "
                        f"line_start={line_start}, line_end={line_end} "
                        f"(span_node type={span_node.type})"
                    )

                    symbols.append(
                        CodeSymbol(
                            name=name,
                            kind=kind,
                            signature=sig,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            language=lang,
                            parent_symbol=parent_symbol,
                        )
                    )
            logger.debug(f"[EXTRACT] Extracted {len(symbols)} symbols from {lang} code")
            return symbols
        except Exception as e:
            logger.warning(
                f"tree-sitter symbol extraction failed for language '{lang}': {e} — "
                "skipping symbol extraction to avoid corrupt fallback data."
            )
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Call extraction from tree-sitter
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_calls_from_tree(tree, lang: str, code: str) -> Dict[str, List[str]]:
        query_str = FALLBACK_CALL_QUERIES.get(lang)
        if not query_str:
            logger.warning(
                f"No tree-sitter call query defined for language '{lang}' — "
                "returning empty call map to avoid corrupt fallback data."
            )
            return {}

        _class_node_types = (
            "class_definition",
            "class_declaration",
            "type_spec",
            "struct_item",
            "enum_item",
            "class_specifier",
        )

        try:
            lang_obj = get_language(lang)
            query = lang_obj.query(query_str)
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)  # dict: {capture_name: [nodes]}

            call_map: Dict[str, Set[str]] = defaultdict(set)
            current_arrow_caller = None

            for cap_name, nodes in captures.items():
                if cap_name == "caller_name":
                    if nodes:
                        current_arrow_caller = nodes[0].text.decode("utf-8")
                    continue
                if cap_name != "callee":
                    continue
                for node in nodes:
                    if node.type in (
                        "attribute",
                        "field_access",
                        "member_expression",
                        "selector_expression",
                        "field_expression",
                    ):
                        callee_name = (
                            node.text.decode("utf-8")
                            .split(".")[-1]
                            .split("->")[-1]
                            .strip()
                        )
                    else:
                        callee_name = node.text.decode("utf-8")

                    caller = None
                    caller_container = None
                    parent = node.parent
                    while parent:
                        if parent.type in (
                            "function_definition",
                            "function_declaration",
                            "method_declaration",
                            "function_item",
                        ):
                            name_node = parent.child_by_field_name("name")
                            if name_node:
                                caller = name_node.text.decode("utf-8")
                            caller_container = parent
                            break
                        elif parent.type == "arrow_function":
                            if current_arrow_caller:
                                caller = current_arrow_caller
                            else:
                                declarator = parent
                                while (
                                    declarator
                                    and declarator.type != "variable_declarator"
                                ):
                                    declarator = declarator.parent
                                if declarator:
                                    name_node = declarator.child_by_field_name("name")
                                    if name_node:
                                        caller = name_node.text.decode("utf-8")
                            caller_container = parent
                            break
                        parent = parent.parent

                    if caller:
                        caller_class = ""
                        class_walker = (
                            caller_container.parent if caller_container else None
                        )
                        while class_walker:
                            if class_walker.type in _class_node_types:
                                cname_node = class_walker.child_by_field_name("name")
                                if cname_node:
                                    caller_class = cname_node.text.decode("utf-8")
                                break
                            class_walker = class_walker.parent
                        caller_qid = qualify_symbol_name(caller, caller_class)
                        call_map[caller_qid].add(callee_name)

            return {k: list(v) for k, v in call_map.items()}
        except Exception as e:
            logger.warning(
                f"tree-sitter call extraction failed for language '{lang}': {e} — "
                "returning empty call map to avoid corrupt fallback data."
            )
            return {}

    @staticmethod
    def _extract_docstrings_python(code: str, symbols: List["CodeSymbol"]) -> None:
        """
        Extract docstrings from Python source code using AST with class context awareness.
        Uses a DFS visitor that tracks the current class name, so docstrings are keyed by qualified name
        (ClassName.method) to avoid collisions. Also handles line-wrapped docstrings by joining lines
        until sentence punctuation is found.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        doc_map: Dict[str, str] = {}

        def _first_complete_line(docstring: str) -> str:
            """Return the first complete sentence from a docstring, joining wrapped lines."""
            raw_lines = [l.strip() for l in docstring.strip().splitlines()]
            result = ""
            for line in raw_lines:
                if not line:
                    break
                result = (result + " " + line).strip() if result else line
                if result and result[-1] in ".!?:":
                    break
            return result[:200]

        def _visit(node: ast.AST, class_name: str) -> None:
            """DFS visitor that carries the current class name."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    _visit(child, child.name)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ds = ast.get_docstring(child)
                    if ds:
                        first = _first_complete_line(ds)
                        if first:
                            key = (
                                f"{class_name}.{child.name}"
                                if class_name
                                else child.name
                            )
                            # First definition wins (preserve the most representative one)
                            if key not in doc_map:
                                doc_map[key] = first
                    # Continue visiting nested functions (e.g., inside methods)
                    _visit(child, class_name)
                else:
                    _visit(child, class_name)

        _visit(tree, "")

        # Assign docstrings to symbols, preferring qualified key
        for sym in symbols:
            if sym.docstring:
                continue
            qid = qualify_symbol_name(sym.name, sym.parent_symbol)
            doc = doc_map.get(qid) or doc_map.get(sym.name)
            if doc:
                sym.docstring = doc


class ControlFlowExtractor:
    """
    Extracts a compressed control-flow skeleton for a single function or
    method, deterministically and without any LLM call.

    For a function body, this:
      1. Replaces straight-line statement runs with `...` (same spirit as
         ContextBuilder._skeletonize_node), but PRESERVES control structures
         (if/elif/else, try/except/finally, for, while) and the calls that
         sit directly inside them.
      2. Annotates branches with a deterministic role comment when a
         heuristic matches confidently (error path, fallback path, fast
         path). Branches that match nothing are left unannotated.
      3. Computes a body_hash from the exact source snippet
         [line_start, line_end] so the skeleton can be invalidated
         independently of sibling symbols in the same block and
         independently of the function's signature.

    Python only — see module-level discussion for why runtime
    instrumentation and bytecode introspection are not viable here
    (CodeAware never executes user-pasted code; many snippets are not even
    importable).
    """

    MAX_PARSE_SIZE_BYTES = SignatureExtractor.MAX_PARSE_SIZE_BYTES

    _CACHE_HIT_RE = re.compile(r"\b(cache|cached|memo)\w*\b", re.IGNORECASE)

    _CONTROL_LINE_RE = re.compile(
        r"^(\s*)(if\b|elif\b|else:|try:|except\b.*:|finally:|for\b|while\b|async for\b)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # 1. Public entry point
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_for_symbol(
        block_content: str,
        symbol: "CodeSymbol",
        max_lines: int = 40,
    ) -> Optional[Tuple[str, str]]:
        """
        Build the control-flow skeleton for one symbol.

        Returns (cfg_skeleton, body_hash), or None if:
          - the symbol isn't a function/method, or has no line_start/line_end;
          - the snippet exceeds max_lines (very long functions don't produce
            a "compressed" skeleton — they produce another wall of text);
          - the snippet doesn't parse as valid Python after dedent;
          - the function body has no control-flow nodes worth preserving.
        """
        if symbol.kind not in ("function", "method"):
            logger.debug(
                f"[CFG] SKIP: symbol kind is {symbol.kind}, not function/method"
            )
            return None
        if not symbol.line_start or not symbol.line_end:
            logger.debug(f"[CFG] SKIP: no line_start/line_end for {symbol.name}")
            return None
        if symbol.line_end - symbol.line_start > max_lines:
            logger.debug(
                f"[CFG] SKIP: line_count={symbol.line_end - symbol.line_start} > max_lines={max_lines}"
            )
            return None
        if len(block_content.encode()) > ControlFlowExtractor.MAX_PARSE_SIZE_BYTES:
            logger.debug(f"[CFG] SKIP: block_content exceeds MAX_PARSE_SIZE_BYTES")
            return None

        lines = block_content.split("\n")
        start_idx = max(0, symbol.line_start - 1)
        end_idx = min(len(lines), symbol.line_end)
        snippet = "\n".join(lines[start_idx:end_idx])

        if not snippet.strip():
            logger.debug(
                f"[CFG] SKIP: snippet is empty or whitespace only for {symbol.name}"
            )
            return None

        body_hash = hashlib.md5(snippet.encode()).hexdigest()[:16]

        # A method snippet sliced out of a class body starts indented
        # (e.g. "    def foo(...):"); dedent before parsing.
        dedented = textwrap.dedent(snippet)
        try:
            tree = ast.parse(dedented)
        except SyntaxError as e:
            logger.debug(f"[CFG] PARSE ERROR: {symbol.name} – {e}")
            return None

        func_node = next(
            (
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        if func_node is None:
            logger.debug(
                f"[CFG] SKIP: no FunctionDef/AsyncFunctionDef node found for {symbol.name}"
            )
            return None

        if not ControlFlowExtractor._has_control_flow(func_node):
            logger.debug(
                f"[CFG] SKIP: no control-flow (if/try/for/while) in body of {symbol.name}"
            )
            return None  # nothing to compress — LOD2 docstring already suffices

        role_queue: List[Optional[str]] = []
        func_node.body = ControlFlowExtractor._compress_stmt_list(
            func_node.body, role_queue
        )
        ast.fix_missing_locations(func_node)

        try:
            skeleton = ast.unparse(func_node)
        except Exception as e:
            logger.debug(f"[CFG] SKIP: ast.unparse failed for {symbol.name} – {e}")
            return None

        skeleton = ControlFlowExtractor._inject_role_comments(skeleton, role_queue)
        logger.debug(
            f"[CFG] SUCCESS: CFG generated for {symbol.name} (body_hash={body_hash})"
        )
        return skeleton, body_hash

    # ═══════════════════════════════════════════════════════════════════
    # 2. Control-flow detection
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _has_control_flow(func_node: "ast.AST") -> bool:
        """True if the function body contains at least one branch worth
        preserving. A straight-line function gets no CFG tier."""
        for node in ast.walk(func_node):
            if node is func_node:
                continue
            if isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.AsyncFor)):
                logger.debug(
                    f"[CFG] _has_control_flow: found control-flow node "
                    f"'{type(node).__name__}' in function"
                )
                return True
        logger.debug(
            "[CFG] _has_control_flow: no control-flow nodes found (straight-line function)"
        )
        return False

    # ═══════════════════════════════════════════════════════════════════
    # 3. Body compression (recursive AST rewrite)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compress_stmt_list(
        stmts: List["ast.stmt"], role_queue: List[Optional[str]]
    ) -> List["ast.stmt"]:
        """
        Compress a list of statements by collapsing straight‑line runs into `...`,
        while preserving control‑flow statements (if, try, for, while) and
        call statements that appear as direct expressions.

        The `role_queue` is appended to during traversal in the same order that
        control‑keyword lines will appear in the final text output, enabling
        correct role comment injection after unparse().

        Returns a new list of AST statements with the same structure but with
        all non‑control bodies replaced by `...` or call placeholders.
        """
        out: List[ast.stmt] = []
        straight_run: List[ast.stmt] = []
        total_stmts = len(stmts)
        control_count = 0
        call_count = 0
        terminal_count = 0

        logger.debug(f"[CFG] _compress_stmt_list: processing {total_stmts} statements")

        def _flush_run() -> None:
            if straight_run:
                logger.debug(
                    f"[CFG] _compress_stmt_list: flushing straight-run of "
                    f"{len(straight_run)} statements"
                )
                out.append(ControlFlowExtractor._placeholder_for(straight_run))
                straight_run.clear()

        for stmt in stmts:
            if isinstance(stmt, ast.If):
                _flush_run()
                control_count += 1
                logger.debug("[CFG] _compress_stmt_list: found If statement")
                out.append(ControlFlowExtractor._compress_if(stmt, role_queue))
            elif isinstance(stmt, ast.Try):
                _flush_run()
                control_count += 1
                logger.debug("[CFG] _compress_stmt_list: found Try statement")
                out.append(ControlFlowExtractor._compress_try(stmt, role_queue))
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                _flush_run()
                control_count += 1
                logger.debug("[CFG] _compress_stmt_list: found Loop statement")
                out.append(ControlFlowExtractor._compress_loop(stmt, role_queue))
            elif isinstance(stmt, (ast.Return, ast.Raise)):
                _flush_run()
                terminal_count += 1
                logger.debug(
                    "[CFG] _compress_stmt_list: found terminal statement (return/raise)"
                )
                out.append(stmt)  # terminal — keep verbatim
            elif ControlFlowExtractor._is_call_statement(stmt):
                _flush_run()
                call_count += 1
                logger.debug("[CFG] _compress_stmt_list: found call statement")
                out.append(ControlFlowExtractor._call_placeholder(stmt))
            else:
                straight_run.append(stmt)
        _flush_run()

        logger.debug(
            f"[CFG] _compress_stmt_list: completed - "
            f"{len(out)} output statements, "
            f"{control_count} control structures, "
            f"{call_count} calls, "
            f"{terminal_count} terminals"
        )

        return out or [ast.Expr(value=ast.Constant(value=Ellipsis))]

    @staticmethod
    def _compress_if(node: "ast.If", role_queue: List[Optional[str]]) -> "ast.If":
        """
        Compress an `if` statement:
          - The `if` line itself gets a role from `_classify_if_role`.
          - The body is compressed recursively.
          - `elif` chains are preserved as nested `if` nodes.
          - Plain `else` blocks are compressed without role annotation.
        """
        logger.debug(
            f"[CFG] _compress_if: compressing If node "
            f"(has_orelse={bool(node.orelse)}, "
            f"body_len={len(node.body)})"
        )

        role_queue.append(ControlFlowExtractor._classify_if_role(node))
        new_body = ControlFlowExtractor._compress_stmt_list(node.body, role_queue)

        new_orelse: List[ast.stmt] = []
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif chain — recurse; the nested call appends its own role.
                logger.debug(
                    "[CFG] _compress_if: elif chain detected (single If in orelse)"
                )
                new_orelse = [
                    ControlFlowExtractor._compress_if(node.orelse[0], role_queue)
                ]
            else:
                logger.debug(
                    f"[CFG] _compress_if: plain else block "
                    f"(orelse length {len(node.orelse)})"
                )
                role_queue.append(None)  # plain `else:` — never guess
                new_orelse = ControlFlowExtractor._compress_stmt_list(
                    node.orelse, role_queue
                )

        new_node = ast.If(test=node.test, body=new_body, orelse=new_orelse)
        ast.copy_location(new_node, node)
        logger.debug(
            f"[CFG] _compress_if: completed - "
            f"body={len(new_body)} stmts, "
            f"orelse={len(new_orelse)} stmts"
        )
        return new_node

    @staticmethod
    def _compress_try(node: "ast.Try", role_queue: List[Optional[str]]) -> "ast.Try":
        """
        Compress a `try` statement:
          - The `try:` line gets no role (queue None).
          - Each `except` handler gets a role from `_classify_except_role`.
          - `else:` and `finally:` blocks are compressed without role annotation.
        """
        logger.debug(
            f"[CFG] _compress_try: compressing Try node - "
            f"{len(node.handlers)} handlers, "
            f"has_orelse={bool(node.orelse)}, "
            f"has_finalbody={bool(node.finalbody)}"
        )

        role_queue.append(None)  # the `try:` line itself never gets a role
        new_body = ControlFlowExtractor._compress_stmt_list(node.body, role_queue)

        new_handlers: List[ast.ExceptHandler] = []
        for i, handler in enumerate(node.handlers):
            logger.debug(f"[CFG] _compress_try: processing handler {i+1}")
            role_queue.append(ControlFlowExtractor._classify_except_role(handler))
            new_handler_body = ControlFlowExtractor._compress_stmt_list(
                handler.body, role_queue
            )
            new_handler = ast.ExceptHandler(
                type=handler.type, name=handler.name, body=new_handler_body
            )
            ast.copy_location(new_handler, handler)
            new_handlers.append(new_handler)

        new_orelse: List[ast.stmt] = []
        if node.orelse:
            logger.debug("[CFG] _compress_try: processing else block")
            new_orelse = ControlFlowExtractor._compress_stmt_list(
                node.orelse, role_queue
            )

        new_finalbody: List[ast.stmt] = []
        if node.finalbody:
            logger.debug("[CFG] _compress_try: processing finally block")
            role_queue.append(None)
            new_finalbody = ControlFlowExtractor._compress_stmt_list(
                node.finalbody, role_queue
            )

        new_node = ast.Try(
            body=new_body,
            handlers=new_handlers,
            orelse=new_orelse,
            finalbody=new_finalbody,
        )
        ast.copy_location(new_node, node)
        logger.debug(
            f"[CFG] _compress_try: completed - "
            f"body={len(new_body)} stmts, "
            f"{len(new_handlers)} handlers, "
            f"orelse={len(new_orelse)} stmts, "
            f"finalbody={len(new_finalbody)} stmts"
        )
        return new_node

    @staticmethod
    def _compress_loop(node, role_queue: List[Optional[str]]):
        """
        Compress a `for`, `async for`, or `while` loop:
          - The loop header line gets no role (queue None).
          - The body is compressed recursively.
          - An `else:` clause on the loop (if present) is compressed without role.
        """
        loop_type = type(node).__name__
        logger.debug(
            f"[CFG] _compress_loop: compressing {loop_type} node "
            f"(has_orelse={bool(node.orelse)})"
        )

        role_queue.append(None)  # loops never get an automatic role in v1
        new_body = ControlFlowExtractor._compress_stmt_list(node.body, role_queue)
        new_orelse = (
            ControlFlowExtractor._compress_stmt_list(node.orelse, role_queue)
            if node.orelse
            else []
        )
        cls = type(node)
        kwargs = dict(body=new_body, orelse=new_orelse)
        if hasattr(node, "target"):
            kwargs["target"] = node.target
            kwargs["iter"] = node.iter
        if hasattr(node, "test"):
            kwargs["test"] = node.test
        new_node = cls(**kwargs)
        ast.copy_location(new_node, node)

        logger.debug(
            f"[CFG] _compress_loop: completed {loop_type} - "
            f"body={len(new_body)} stmts, "
            f"orelse={len(new_orelse)} stmts"
        )
        return new_node

    # ═══════════════════════════════════════════════════════════════════
    # 4. Call & straight-run placeholders
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_call_statement(stmt: "ast.stmt") -> bool:
        """Return True if `stmt` is an expression statement or assignment
        whose right‑hand side is a function call."""
        is_call = False
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            is_call = True
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            is_call = True

        if is_call:
            logger.debug("[CFG] _is_call_statement: detected call statement")
        else:
            logger.debug("[CFG] _is_call_statement: not a call statement")
        return is_call

    @staticmethod
    def _call_placeholder(stmt: "ast.stmt") -> "ast.stmt":
        """Keep the call's name, drop its arguments to `...` — the point is
        showing WHICH call happens in WHICH branch, not its exact inputs
        (that's LOD3 detail)."""
        call_node = stmt.value
        has_args = bool(call_node.args or call_node.keywords)

        logger.debug(
            f"[CFG] _call_placeholder: creating placeholder for call to "
            f"'{ast.unparse(call_node.func)[:50]}' "
            f"(has_args={has_args})"
        )

        new_call = ast.Call(
            func=call_node.func,
            args=[ast.Constant(value=Ellipsis)] if has_args else [],
            keywords=[],
        )
        ast.copy_location(new_call, call_node)
        if isinstance(stmt, ast.Assign):
            new_stmt = ast.Assign(targets=stmt.targets, value=new_call)
        else:
            new_stmt = ast.Expr(value=new_call)
        ast.copy_location(new_stmt, stmt)
        return new_stmt

    @staticmethod
    def _placeholder_for(straight_run: List["ast.stmt"]) -> "ast.stmt":
        """Collapse a run of non-control statements into a single `...`."""
        logger.debug(
            f"[CFG] _placeholder_for: collapsing straight-run of "
            f"{len(straight_run)} statements to '...'"
        )
        placeholder = ast.Expr(value=ast.Constant(value=Ellipsis))
        ast.copy_location(placeholder, straight_run[0])
        return placeholder

    # ═══════════════════════════════════════════════════════════════════
    # 5. Role comment injection (post-unparse text pass)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _inject_role_comments(
        skeleton_text: str, role_queue: List[Optional[str]]
    ) -> str:
        """
        ast.unparse() drops comments entirely (they aren't part of the AST).
        This walks the unparsed text in order and, for each control-keyword
        line, pops the next role from role_queue (collected during the same
        pre-order walk that produced the text) and appends it as a comment.

        Safe because we never reorder statements — only truncate bodies —
        so textual top-to-bottom order of control lines matches the
        pre-order AST walk order exactly.
        """
        lines = skeleton_text.split("\n")
        queue = list(role_queue)
        out_lines = []
        roles_assigned = 0

        logger.debug(
            f"[CFG] _inject_role_comments: {len(lines)} lines, "
            f"{len(queue)} roles in queue"
        )

        for line in lines:
            m = ControlFlowExtractor._CONTROL_LINE_RE.match(line)
            if m and queue:
                role = queue.pop(0)
                if role:
                    line = line.rstrip() + f"  # {role}"
                    roles_assigned += 1
                    logger.debug(
                        f"[CFG] _inject_role_comments: assigned role '{role}' "
                        f"to line: '{line[:60]}...'"
                    )
                else:
                    logger.debug("[CFG] _inject_role_comments: skipped None role")
            out_lines.append(line)

        logger.debug(
            f"[CFG] _inject_role_comments: injected {roles_assigned} role comment(s), "
            f"{len(queue)} roles remaining (should be 0 if queue matched lines)"
        )
        return "\n".join(out_lines)

    @staticmethod
    def _classify_if_role(if_node: "ast.If") -> Optional[str]:
        """'fast path' if the test mentions cache/memo AND the body returns
        directly. Any other `if` is left unlabeled — we never guess."""
        if not hasattr(ast, "unparse"):
            logger.debug("[CFG] _classify_if_role: ast.unparse not available")
            return None

        test_src = ast.unparse(if_node.test)
        body_has_return = any(isinstance(s, ast.Return) for s in if_node.body)

        logger.debug(
            f"[CFG] _classify_if_role: test='{test_src[:80]}', "
            f"body_has_return={body_has_return}"
        )

        if body_has_return and ControlFlowExtractor._CACHE_HIT_RE.search(test_src):
            logger.debug(
                f"[CFG] _classify_if_role: MATCH fast path (test mentions cache/memo)"
            )
            return "fast path"
        else:
            if not body_has_return:
                logger.debug(
                    "[CFG] _classify_if_role: no fast path (body does not return directly)"
                )
            elif not ControlFlowExtractor._CACHE_HIT_RE.search(test_src):
                logger.debug(
                    "[CFG] _classify_if_role: no fast path (test does not mention cache/memo)"
                )
        return None

    @staticmethod
    def _classify_except_role(handler: "ast.ExceptHandler") -> str:
        """'fallback path' if the handler returns an alternative value without
        re-raising; 'error path' in any other case (includes re-raise, log +
        pass, etc.). Unlike _classify_if_role, an except ALWAYS receives a role
        — there is no real ambiguity about whether an except is error handling."""
        has_raise = any(isinstance(s, ast.Raise) for s in handler.body)
        has_return_value = any(
            isinstance(s, ast.Return) and s.value is not None for s in handler.body
        )

        logger.debug(
            f"[CFG] _classify_except_role: has_raise={has_raise}, "
            f"has_return_value={has_return_value}"
        )

        if has_return_value and not has_raise:
            logger.debug("[CFG] _classify_except_role: MATCH fallback path")
            return "fallback path"
        else:
            logger.debug("[CFG] _classify_except_role: MATCH error path")
            return "error path"


class StateStore:
    """
    SQLite database infrastructure for the CodeAware filter.

    This class manages:
      - DDL (table creation, indexes)
      - Serialized write queue (prevents "database is locked" errors)
      - Per-project async locks
      - Persistence of symbol edges, path views, docstrings, CFG skeletons
      - Database checkpoints (WAL)
      - Purge of orphaned rows

    It does NOT manage conversation state (→ ConversationStateManager).
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the state store with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Database lifecycle (DDL, connection, worker)
    # ═══════════════════════════════════════════════════════════════════════

    def init_db(self) -> None:
        """
        Create all necessary tables and indexes (idempotent).

        Called once during Filter initialization. In addition to creating the
        schema, this method tunes SQLite's runtime behaviour to minimise
        contention:
          - `busy_timeout` – tells SQLite to wait up to `llm_per_call_timeout`
            seconds before giving up on a locked database.
          - `synchronous = NORMAL` – reduces the number of `fsync()` calls,
            increasing throughput under write-heavy workloads (background
            docstrings, edge persistence, etc.) without compromising crash
            safety, because WAL already guarantees durability.
        """
        db_path = self._f.valves.state_db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # ------------------------------------------------------------------
        # REGION 1 — Connect and set busy_timeout
        # ------------------------------------------------------------------
        self._f._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._f._db_conn.execute(
            f"PRAGMA busy_timeout = {self._f.valves.llm_per_call_timeout * 1000}"
        )

        # ------------------------------------------------------------------
        # REGION 2 — Reduce fsync pressure to lower lock contention
        # ------------------------------------------------------------------
        # 'NORMAL' syncs at critical moments (e.g., WAL checkpoint) but not
        # on every transaction commit. This is safe with WAL and dramatically
        # reduces the chance of "database is locked" under concurrent writes.
        self._f._db_conn.execute("PRAGMA synchronous = NORMAL")

        # ------------------------------------------------------------------
        # REGION 3 — Create all tables and indexes (idempotent)
        # ------------------------------------------------------------------
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                project_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS code_contents (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._f._db_conn.execute("PRAGMA journal_mode=WAL")
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS block_change_summaries (
                block_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS code_path_views (
                path_id             TEXT NOT NULL,
                project_id          TEXT NOT NULL,
                entry_point         TEXT NOT NULL,
                seed_nodes_json     TEXT NOT NULL DEFAULT '[]',
                induced_nodes_json  TEXT NOT NULL DEFAULT '{}',
                induced_edges_json  TEXT NOT NULL DEFAULT '[]',
                activation_score    REAL NOT NULL DEFAULT 0.0,
                business_label      TEXT NOT NULL DEFAULT '',
                summary             TEXT NOT NULL DEFAULT '',
                label_confidence    REAL NOT NULL DEFAULT 0.0,
                structural_hash     TEXT NOT NULL DEFAULT '',
                call_graph_hash     TEXT NOT NULL DEFAULT '',
                last_built          REAL NOT NULL,
                PRIMARY KEY (path_id, project_id)
            )
        """)
        self._f._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cpv_project "
            "ON code_path_views(project_id)"
        )
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_edges (
                project_id  TEXT NOT NULL,
                src         TEXT NOT NULL,
                dst         TEXT NOT NULL,
                type        TEXT NOT NULL,
                weight      REAL NOT NULL DEFAULT 1.0,
                confidence  REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (project_id, src, dst, type)
            )
        """)
        self._f._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_project_src "
            "ON symbol_edges(project_id, src)"
        )
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS raptor_clusters (
                cluster_id      TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL,
                level           INTEGER NOT NULL,
                member_ids_json TEXT NOT NULL,
                summary         TEXT NOT NULL,
                centroid_json   TEXT,
                created_at      REAL NOT NULL
            )
        """)
        self._f._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raptor_project_level "
            "ON raptor_clusters(project_id, level)"
        )
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_edges_meta (
                project_id      TEXT PRIMARY KEY,
                code_state_hash TEXT NOT NULL,
                edge_count      INTEGER NOT NULL DEFAULT 0,
                saved_at        REAL NOT NULL
            )
        """)
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_docstrings (
                project_id  TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                docstring    TEXT NOT NULL,
                updated_at  REAL NOT NULL,
                PRIMARY KEY (project_id, symbol_name)
            )
        """)
        self._f._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_cfg (
                project_id   TEXT NOT NULL,
                symbol_name  TEXT NOT NULL,
                cfg_skeleton TEXT NOT NULL,
                body_hash    TEXT NOT NULL,
                updated_at   REAL NOT NULL,
                PRIMARY KEY (project_id, symbol_name)
            )
        """)
        self._f._db_conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Database write queue (serialised, non-blocking writes)
    # ═══════════════════════════════════════════════════════════════════════

    async def _db_enqueue(self, fn, args=(), kwargs=None) -> None:
        """
        Enqueue a database write operation to the worker queue.

        Args:
            fn: Callable to execute.
            args: Positional arguments.
            kwargs: Keyword arguments (default: empty dict).
        """
        if kwargs is None:
            kwargs = {}
        await self._f._db_write_queue.put((fn, args, kwargs))

    async def db_worker(self) -> None:
        """Database write worker with automatic restart on failure."""
        while True:
            try:
                await self._db_worker_loop()
            except asyncio.CancelledError:
                self._f._log_debug("DB worker cancelled — shutting down.")
                break
            except Exception as e:
                self._f._log_debug(f"DB worker crashed: {e} — restarting in 2s")
                await asyncio.sleep(2)

    async def drain_writes(self, timeout: float = 5.0) -> None:
        """
        Blocks until all pending items in _db_write_queue have been processed
        (task_done called by the worker).

        Called at the start of each inlet to ensure that docstring writes from
        the previous turn finish BEFORE inlet_preprocess() starts reading from
        SQLite. Eliminates reader/writer overlap on _db_conn.

        Args:
            timeout: maximum seconds to wait. If timeout elapses, logs and
                     continues (soft degradation, no exception).
        """
        try:
            await asyncio.wait_for(self._f._db_write_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            self._f._log_debug(
                f"drain_writes: timeout {timeout}s; "
                "writes still in progress, continuing (soft degradation)"
            )

    async def _db_read(self, fn, *args, **kwargs):
        """
        Serialized read with the writer worker.

        While all access to _db_conn goes through _db_global_lock,
        two threads cannot touch the connection simultaneously, and
        "database is locked" disappears as a race condition.

        The lock is acquired INSIDE the thread (run_sync), not crossing
        an await, to avoid blocking the event loop.

        Usage:
            row = await self._f._state_store._db_read(
                lambda: self._f._db_conn.execute(sql, params).fetchone()
            )
        """

        def _run():
            with _db_global_lock:
                return fn(*args, **kwargs)

        return await anyio.to_thread.run_sync(_run)

    async def _db_worker_loop(self) -> None:
        """
        Single run of the DB write loop. Exits on CancelledError.

        Each job is executed with a retry loop that uses **exponential backoff**
        when encountering a `database is locked` error. This gives SQLite's WAL
        mechanism enough time to resolve contention.

        The lock is acquired INSIDE the thread (not crossing an await) to avoid
        blocking the event loop. On exhaustion of retries, the job is dropped
        (no raise) to prevent worker crashes and closure accumulation.
        """
        while True:
            try:
                job = await asyncio.wait_for(self._f._db_write_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            func, args, kwargs = job

            # Lock acquired inside the thread, not across an await
            def _run_batch(fn=func, a=args, kw=kwargs):
                with _db_global_lock:
                    fn(*a, **kw)

            for attempt in range(5):
                try:
                    await anyio.to_thread.run_sync(_run_batch)
                    break  # success
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        # Exponential backoff: 0.1, 0.2, 0.4, 0.8 seconds
                        backoff = 0.1 * (2**attempt)
                        self._f._log_debug(
                            f"DB worker: locked (attempt {attempt+1}/5), "
                            f"retrying in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                    else:
                        # Drop the job instead of crashing/restarting the worker
                        self._f._log_debug(
                            f"DB worker: job dropped after {attempt+1} attempts: {e}"
                        )
                        break

            # Mark the item as processed (enables queue.join())
            self._f._db_write_queue.task_done()

    def _db_conn_write_sync(self, project_id: str, state: ConversationState) -> None:
        """
        Synchronous write of ConversationState to SQLite (used by LRU eviction).

        This method bypasses the async queue and writes directly to the DB.
        It is only intended for emergency flushes during eviction to avoid data loss.

        Args:
            project_id: The project identifier.
            state: The ConversationState to persist.
        """
        # Serialize the state (logic mirrored from _save_to_db)
        active_blocks_meta = {}
        for k, v in state.active_blocks.items():
            d = v.dict()
            d["content_type"] = v.content_type.value
            content_hash = v.hash
            # Insert content if not present (synchronous)
            self._f._db_conn.execute(
                "INSERT OR IGNORE INTO code_contents (hash, content, created_at) VALUES (?, ?, ?)",
                (content_hash, v.content, time.time()),
            )
            d["content"] = f"@@hash:{content_hash}"
            active_blocks_meta[k] = d

        serializable = {
            "active_blocks": active_blocks_meta,
            "recent_changes": [b.dict() for b in state.recent_changes],
            "committed_changes": [b.dict() for b in state.committed_changes],
            "feedback_history": [fb.dict() for fb in state.feedback_history],
            "message_count": state.message_count,
            "last_compression_timestamp": state.last_compression_timestamp,
            "last_suggestion_timestamp": state.last_suggestion_timestamp,
            "last_cleanup_suggestion_msg_idx": state.last_cleanup_suggestion_msg_idx,
            "has_any_calls": state.has_any_calls,
            "last_cot_level": state.last_cot_level,
            "conversation_summaries": state.conversation_summaries,
            "summarized_turn_hwm": state.summarized_turn_hwm,
            "history_blocked_age": state.history_blocked_age,
            "wm_fired": state.wm_fired,
            "wm_msgs_evicted": state.wm_msgs_evicted,
            "wm_turns_evicted": state.wm_turns_evicted,
            "wm_summary_ok": state.wm_summary_ok,
            "wm_emergency_cap": state.wm_emergency_cap,
            "wm_batch_too_small": state.wm_batch_too_small,
            "wm_no_slot": state.wm_no_slot,
            "wm_degradation_guard": state.wm_degradation_guard,
            "pending_slot_resave": state.pending_slot_resave,
        }

        self._f._db_conn.execute(
            "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(serializable), time.time()),
        )
        self._f._db_conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Maintenance (checkpoints)
    # ═══════════════════════════════════════════════════════════════════════

    async def run_db_checkpoints(self) -> None:
        """Run SQLite WAL checkpoint and ChromaDB persist."""
        try:
            await anyio.to_thread.run_sync(
                lambda: self._f._db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            )
            self._f._log_debug("SQLite WAL checkpoint completed")
        except Exception as e:
            self._f._log_debug(f"SQLite checkpoint error: {e}")
        try:
            if self._f.chroma_client:
                await anyio.to_thread.run_sync(lambda: self._f.chroma_client.persist())
            self._f._log_debug("ChromaDB persist/checkpoint completed")
        except Exception as e:
            self._f._log_debug(f"ChromaDB checkpoint error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Edge persistence
    # ═══════════════════════════════════════════════════════════════════════

    async def save_symbol_edges_to_db(self, project_id: str) -> int:
        """
        Persist typed edges from the SymbolIndex to SQLite.

        Saves alongside the current code_state_hash for invalidation detection.
        Returns the number of edges saved.

        Args:
            project_id: The current project identifier.

        Returns:
            int: Number of edges persisted.
        """
        if not self._f.valves.enable_edge_persistence:
            return 0

        edges_out = self._f._symbol_index.get_all_edges_out(project_id)
        if not edges_out:
            return 0

        code_hash = self._f._activation.compute_code_state_hash(project_id)
        if not code_hash:
            return 0

        total_edges = sum(len(edges) for edges in edges_out.values())

        def _write():
            self._f._db_conn.execute(
                "DELETE FROM symbol_edges WHERE project_id = ?", (project_id,)
            )
            for src, edges in edges_out.items():
                for edge in edges:
                    self._f._db_conn.execute(
                        "INSERT OR REPLACE INTO symbol_edges "
                        "(project_id, src, dst, type, weight, confidence) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            project_id,
                            edge.src,
                            edge.dst,
                            edge.type,
                            edge.weight,
                            edge.confidence,
                        ),
                    )
            self._f._db_conn.execute(
                "INSERT OR REPLACE INTO symbol_edges_meta "
                "(project_id, code_state_hash, edge_count, saved_at) "
                "VALUES (?,?,?,?)",
                (project_id, code_hash, total_edges, time.time()),
            )
            self._f._db_conn.commit()

        await self._db_enqueue(_write)
        self._f._log_debug(
            f"Edge persistence: saved {total_edges} edges (code_hash={code_hash})"
        )
        return total_edges

    async def load_symbol_edges_from_db(self, project_id: str) -> int:
        """
        Restore typed edges from SQLite.

        Only restores if the saved code_state_hash matches the current state.
        Returns the number of edges restored (0 if stale or no data).

        Args:
            project_id: The current project identifier.

        Returns:
            int: Number of edges restored.
        """
        if not self._f.valves.enable_edge_persistence:
            return 0

        # Skip if edges are already loaded in memory
        existing = self._f._symbol_index.get_all_edges_out(project_id)
        if existing:
            return 0

        current_code_hash = self._f._activation.compute_code_state_hash(project_id)
        if not current_code_hash:
            return 0  # no active code

        meta_row = await self._db_read(
            lambda: self._f._db_conn.execute(
                "SELECT code_state_hash, edge_count FROM symbol_edges_meta "
                "WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        )

        if not meta_row:
            self._f._log_debug("Edge persistence: no saved edges for this project")
            return 0

        saved_hash, saved_count = meta_row
        if saved_hash != current_code_hash:
            self._f._log_debug(
                f"Edge persistence: stale edges detected "
                f"(saved={saved_hash}, current={current_code_hash}). "
                f"Edges will be rebuilt when code is processed."
            )
            return 0

        rows = await self._db_read(
            lambda: self._f._db_conn.execute(
                "SELECT src, dst, type, weight, confidence "
                "FROM symbol_edges WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        )

        count = 0
        for src, dst, etype, weight, confidence in rows:
            edge = Edge(
                src=src,
                dst=dst,
                type=etype,
                weight=weight,
                confidence=confidence,
            )
            self._f._symbol_index.add_edge(edge, project_id)
            count += 1

        self._f._log_debug(
            f"✓ Edge persistence: restored {count} edges "
            f"(code_hash={current_code_hash})"
        )
        return count

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Path view persistence
    # ═══════════════════════════════════════════════════════════════════════

    async def save_path_views_to_db(self, project_id: str, views: list) -> None:
        """
        Persist CodePathViews to SQLite, replacing any existing views for the project.

        Args:
            project_id: The current project identifier.
            views: List of CodePathView objects to persist.
        """

        def _write():
            self._f._db_conn.execute(
                "DELETE FROM code_path_views WHERE project_id = ?", (project_id,)
            )
            for v in views:
                self._f._db_conn.execute(
                    "INSERT INTO code_path_views VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        v.path_id,
                        project_id,
                        v.entry_point,
                        json.dumps(v.seed_nodes),
                        json.dumps(v.induced_nodes),
                        json.dumps([e.dict() for e in v.induced_edges]),
                        v.activation_score,
                        v.business_label,
                        v.summary,
                        v.label_confidence,
                        v.structural_hash,
                        v.call_graph_hash,
                        v.last_built,
                    ),
                )
            self._f._db_conn.commit()

        await self._db_enqueue(_write)

    async def load_path_views_from_db(self, project_id: str) -> list:
        """
        Load CodePathViews from SQLite for a project.

        Args:
            project_id: The current project identifier.

        Returns:
            list: List of CodePathView objects (may be empty).
        """
        rows = await self._db_read(
            lambda: self._f._db_conn.execute(
                "SELECT path_id, entry_point, seed_nodes_json, induced_nodes_json, "
                "induced_edges_json, activation_score, business_label, summary, "
                "label_confidence, structural_hash, call_graph_hash, last_built "
                "FROM code_path_views WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        )
        views = []
        for row in rows:
            try:
                induced_edges = [Edge(**e) for e in json.loads(row[4])]
                views.append(
                    CodePathView(
                        path_id=row[0],
                        entry_point=row[1],
                        seed_nodes=json.loads(row[2]),
                        induced_nodes=json.loads(row[3]),
                        induced_edges=induced_edges,
                        activation_score=row[5],
                        business_label=row[6],
                        summary=row[7],
                        label_confidence=row[8],
                        structural_hash=row[9],
                        call_graph_hash=row[10],
                        last_built=row[11],
                    )
                )
            except Exception as e:
                self._f._log_debug(f"Skipping corrupt CodePathView: {e}")
        return views

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Orphan data cleanup
    # ═══════════════════════════════════════════════════════════════════════

    async def purge_orphaned_data(self, project_id: str) -> int:
        """
        Remove rows from code_contents, symbol_docstrings, and symbol_cfg
        that no longer correspond to active blocks or symbols.

        Returns the total number of rows deleted.

        Args:
            project_id: The current project identifier.

        Returns:
            int: Total number of rows purged.
        """
        state = self._f._conversation_state_manager.get(project_id)
        if (
            not state.active_blocks
            and not self._f._symbol_index.get_all_qualified_names(project_id)
        ):
            return 0

        active_qids = self._f._symbol_index.get_all_qualified_names(project_id)
        active_hashes = set(state.active_blocks.keys())

        total_deleted = 0

        def _purge():
            nonlocal total_deleted
            conn = self._f._db_conn

            if active_hashes:
                placeholders = ",".join("?" * len(active_hashes))
                cursor = conn.execute(
                    f"DELETE FROM code_contents WHERE hash NOT IN ({placeholders})",
                    tuple(active_hashes),
                )
                total_deleted += cursor.rowcount

            if active_qids:
                placeholders = ",".join("?" * len(active_qids))
                cursor = conn.execute(
                    f"DELETE FROM symbol_docstrings WHERE project_id = ? "
                    f"AND symbol_name NOT IN ({placeholders})",
                    (project_id,) + tuple(active_qids),
                )
                total_deleted += cursor.rowcount

                cursor = conn.execute(
                    f"DELETE FROM symbol_cfg WHERE project_id = ? "
                    f"AND symbol_name NOT IN ({placeholders})",
                    (project_id,) + tuple(active_qids),
                )
                total_deleted += cursor.rowcount

            conn.commit()

        await self._db_enqueue(_purge)
        if total_deleted > 0:
            self._f._log_debug(f"Purged {total_deleted} orphaned rows from DB")
        return total_deleted

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Per‑project locks (reentrant async locks)
    # ═══════════════════════════════════════════════════════════════════════

    async def get_project_lock(self, project_id: str) -> ReentrantAsyncLock:
        """
        Return (or create) the reentrant async lock for the given project.

        Args:
            project_id: The project identifier.

        Returns:
            ReentrantAsyncLock: The lock instance.
        """
        async with self._f._lock_lock:
            if project_id not in self._f._project_locks:
                self._f._project_locks[project_id] = ReentrantAsyncLock()
            return self._f._project_locks[project_id]


class LongTermMemory:
    """Manages long‑term conversational and code memory using ChromaDB.

    Provides:
    * Storage and retrieval of user/assistant messages indexed by project
      and enriched with code symbols, file paths, and content type.
    * A response cache that avoids redundant LLM calls when semantically
      similar queries are repeated under the same code state.
    * Duplicate question detection using cosine similarity and optional
      CrossEncoder reranking.
    * Time‑bounded expiration of old memories.

    Docs 10–13 backported:
        M1 – strip <details> reasoning before embedding.
        M2 – skip partial multi‑phase assistant responses.
        C1 – apply similarity threshold on raw score, not decayed.
        C2 – deduplicate fragments by document ID.
        C3 – purge_project.
        C4 – validate embedding model dimension on startup.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """Initialize with a reference to the parent Filter."""
        self._f = filter_ref
        self._retrieval_disabled_reason: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def init(self) -> None:
        """Initialise ChromaDB, embedder, and response cache collection."""
        os.makedirs(self._f.valves.long_term_memory_dir, exist_ok=True)
        self._f.embedder = _shared_get_embedder()
        self._f._log_debug("Embedder: using Qwen/Qwen3-Embedding-0.6B")

        self._f.chroma_client = _shared_get_chroma_client(
            self._f.valves.long_term_memory_dir
        )
        self._f._log_debug("ChromaDB: using shared singleton")

        if self._f.chroma_client is None:
            self._f._log_debug("ChromaDB not available")
            return

        self._f.memory_collection = self._f.chroma_client.get_or_create_collection(
            name="conversation_memory", metadata={"hnsw:space": "cosine"}
        )
        self._f._log_debug(
            f"LTM collection ready – vector size = "
            f"{self._f.memory_collection.metadata.get('dimension', '?')}"
        )

        self._f._response_cache_collection = (
            self._f.chroma_client.get_or_create_collection(
                name=f"response_cache_{self._f.valves.project_id or 'default'}",
                metadata={"hnsw:space": "cosine"},
            )
        )
        self._f._log_debug("LTM ready")

        # ── C4: validate embedding model after collection is ready ──────────
        self._validate_embedding_model()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Response cache & duplicate detection
    # ═══════════════════════════════════════════════════════════════════════════

    async def find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        """Search the response cache for a semantically similar query.

        Returns a dict with ``response``, ``query``, and ``timestamp`` on a
        hit, or ``None`` if no valid cached entry is found.  Stale entries
        (code state changed or TTL expired) are deleted on the fly.
        """
        # ── Early exit: cache disabled or collection missing ──────────
        if not self._f.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        col = getattr(self._f, "_response_cache_collection", None)
        if col is None:
            return None

        # ── Embed query ───────────────────────────────────────────────
        query_vec = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )

        # ── Retrieve nearest neighbour ─────────────────────────────────
        results = await anyio.to_thread.run_sync(
            lambda: col.query(
                query_embeddings=[query_vec],
                n_results=1,
                where={"project_id": self._f.valves.project_id},
                include=["documents", "metadatas", "distances"],
            )
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return None

        # ── Validate similarity ────────────────────────────────────────
        dist = results["distances"][0][0]
        similarity = 1.0 - (dist / 2.0)
        if similarity < self._f.valves.response_cache_similarity_threshold:
            return None

        # ── Check staleness: code state hash ───────────────────────────
        meta = results["metadatas"][0][0]
        stored_code_state = meta.get("code_state_hash", "")
        if (
            stored_code_state
            and stored_code_state
            != self._f._activation.compute_code_state_hash(self._f.valves.project_id)
        ):
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        # ── Check staleness: TTL ───────────────────────────────────────
        ttl = self._f.valves.response_cache_ttl_hours * 3600
        ts = meta.get("timestamp", 0)
        if ttl > 0 and time.time() - ts > ttl:
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        doc = results["documents"][0][0]
        return {"response": doc, "query": meta.get("query", ""), "timestamp": ts}

    async def find_duplicate_question(
        self, query: str, project_id: str
    ) -> Optional[dict]:
        """Detect near‑duplicate user questions using cosine similarity and,
        when available, a CrossEncoder reranker.

        Returns ``{"sim": float, "doc": str}`` if a duplicate is found, or
        ``None`` otherwise.
        """
        # ── Early exit: prerequisites ──────────────────────────────────
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return None
        if not query or len(query.strip()) < 15:
            return None

        try:
            # ── Embed query ────────────────────────────────────────────
            q_emb = await anyio.to_thread.run_sync(
                lambda: self._f.embedder.encode(query[:1000]).tolist()
            )
            now = time.time()

            # ── Build time‑bounded filter ──────────────────────────────
            where = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"role": {"$eq": "user"}},
                    {
                        "timestamp": {
                            "$gt": time.time()
                            - self._f.valves.duplicate_question_lookback_hours * 3600
                        }
                    },
                ]
            }

            # ── Query ChromaDB ─────────────────────────────────────────
            results = await anyio.to_thread.run_sync(
                lambda: self._f.memory_collection.query(
                    query_embeddings=[q_emb],
                    n_results=self._f.valves.duplicate_question_lookback,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            )
            if not results or not results["ids"] or not results["ids"][0]:
                return None

            # ── Find best candidate, optionally via CrossEncoder ────────
            best_candidate = None
            best_sim = 0.0
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i]
                sim = 1.0 - (dist / 2.0)
                if sim >= self._f.valves.duplicate_question_threshold and doc != query:
                    pairs = [(query[:500], doc[:500])]
                    raw_score = await self._f._commands._predict_cross_encoder(pairs)
                    if raw_score is None:
                        self._f._log_debug(
                            "_find_duplicate_question: CrossEncoder not loaded, "
                            "using cosine similarity only (higher false positive risk)."
                        )
                        best_candidate = (sim, doc, None)
                        break
                    # Normalize CrossEncoder logit to [0,1] before comparing to threshold
                    ce_prob = self._f._commands._normalize_cross_encoder_score(
                        raw_score[0]
                    )
                    if ce_prob > 0.85:
                        best_candidate = (sim, doc, ce_prob)
                        break

            if best_candidate:
                sim, doc, ce = best_candidate
                log_msg = f"Duplicate question found (cosine={sim:.3f}"
                if ce is not None:
                    log_msg += f", crossencoder={ce:.3f}"
                log_msg += ")"
                self._f._log_debug(log_msg)
                return {"sim": sim, "doc": doc}

        except Exception as e:
            self._f._log_debug(f"Error in duplicate question detection: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Query helpers (parsing, symbol extraction, expansion)
    # ═══════════════════════════════════════════════════════════════════════════

    def _parse_forced_symbol_query(self, query: str) -> Tuple[Optional[str], str]:
        """Extract forced symbol from query (?symbol:Name). Returns (symbol, cleaned_query)."""
        if not self._f.valves.ltm_symbol_force_mode_enabled:
            return None, query
        match = re.match(r"\?symbol:(\S+)", query)
        if match:
            symbol = match.group(1)
            cleaned = query[match.end() :].strip()
            return symbol, cleaned if cleaned else symbol
        return None, query

    def _extract_query_symbols(self, query: str, project_id: str) -> Set[str]:
        """Return symbol names from the query that exist in the SymbolIndex."""
        if not query or not project_id:
            return set()
        words = set(re.findall(r"\b\w+\b", query))
        project_symbols = self._f._symbol_index.get_all_names(project_id)
        return words.intersection(project_symbols)

    def _is_symbol_indexable(self, symbol: "CodeSymbol") -> bool:
        """True if this symbol should be indexed in LTM metadata."""
        if symbol.kind not in ("function", "class", "method"):
            return False
        if len(symbol.name) < 3:
            return False
        blacklist = getattr(self._f, "_SYMBOL_BLACKLIST", set())
        if symbol.name in blacklist:
            return False
        return True

    async def _expand_query_for_retrieval(
        self,
        query: str,
        use_case_label: str = "General Programming",
        slot_free: bool = True,
    ) -> List[str]:
        """
        Generate alternative search queries for LTM retrieval.

        Forces structured output using a strong system prompt and numbered list.
        Extracts queries using regex to handle variations in formatting.
        """
        # ------------------------------------------------------------------
        # REGION 1: Early exits
        # ------------------------------------------------------------------
        if not self._f.valves.enable_multi_query_retrieval:
            return [query]
        if not slot_free:
            return [query]
        if len(query.strip()) < 15:
            return [query]

        # ------------------------------------------------------------------
        # REGION 2: Prompt with numbered list format
        # ------------------------------------------------------------------
        prompt = (
            f"User question: {query[:300]}\n\n"
            f"Generate {self._f.valves.multi_query_variants} alternative search queries.\n"
            "Output ONLY the queries, one per line, numbered 1 to N.\n"
            "Do not include any other text.\n\n"
            "1. "
        )

        # ------------------------------------------------------------------
        # REGION 3: Strong system prompt to enforce role
        # ------------------------------------------------------------------
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a search query reformulator. Your ONLY task is to output search queries. "
                "Never include analysis, reasoning, explanations, or meta-commentary. "
                "Output must be in the exact format: numbered queries, one per line. "
                "No other text is allowed."
            ),
            model_override=self._f.valves.llm_model,
            max_tokens=80,
            temperature=0.4,
            label="multi_query_expand",
        )

        # Log the raw response for debugging
        self._f._log_debug(f"Multi-query raw response: {response}")

        queries = [query]
        if response:
            # ------------------------------------------------------------------
            # REGION 4: Extract numbered lines using regex
            # ------------------------------------------------------------------
            import re

            pattern = re.compile(r"^\s*(?:\d+\.\s*|[-*]\s*)?(.+)$")
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    cleaned = match.group(1).strip()
                    # Accept if it's a valid query (not empty, not meta-commentary)
                    if (
                        cleaned
                        and len(cleaned) > 5
                        and "analysis" not in cleaned.lower()
                    ):
                        queries.append(cleaned)

            # Limit to configured variants (+1 for original)
            queries = queries[: self._f.valves.multi_query_variants + 1]

        self._f._log_debug(f"Multi-query expansion: {len(queries)} queries")
        return queries

    async def _rerank_results(
        self, query: str, documents: List[str], top_k: int
    ) -> List[str]:
        """Rerank documents using CrossEncoder and return top_k results."""
        if not self._f.valves.enable_reranking:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = await self._f._commands._predict_cross_encoder(pairs)
        if scores is None:
            self._f._log_debug(
                "_rerank_results: CrossEncoder not loaded, skipping reranking."
            )
            return documents[:top_k]
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    async def _build_retrieval_context(
        self,
        content: str,
        project_id: str,
        role: str,
        code_symbols: List[str],
        file_paths: List[str],
        content_type: str,
    ) -> str:
        """Build a short prefix that enriches a ChromaDB document for better
        retrieval.  Two modes are supported via ``contextual_retrieval_mode``:

        * ``"metadata"`` — fast, concatenates structured fields (project,
          files, symbols, type, excerpt).
        * ``"llm"`` — asks the LLM for a one‑sentence description of the
          content (slower, but captures nuance better).
        """
        if not self._f.valves.enable_contextual_retrieval:
            return ""

        # ── LLM mode ────────────────────────────────────────────────────
        if self._f.valves.contextual_retrieval_mode == "llm":
            return await self._build_retrieval_context_llm(content, project_id)

        # ── Metadata mode ───────────────────────────────────────────────
        parts: List[str] = [f"Project: {project_id}"]
        if file_paths:
            parts.append(f"Files: {', '.join(file_paths[:3])}")
        if code_symbols:
            parts.append(f"Functions: {', '.join(code_symbols[:6])}")
        if content_type and content_type != "general":
            parts.append(f"Type: {content_type}")
        role_label = "User question" if role == "user" else "Assistant response"
        parts.append(f"Role: {role_label}")
        excerpt = content[:120].replace("\n", " ").strip()
        if len(content) > 120:
            excerpt += "..."
        parts.append(f"Excerpt: {excerpt}")
        context_line = " | ".join(parts)
        return f"[{context_line}]\n\n"

    async def _build_retrieval_context_llm(self, content: str, project_id: str) -> str:
        """Use the LLM to generate a one‑sentence contextual description
        of *content* for improved long‑term memory retrieval."""
        prompt = (
            "In one sentence (10-20 words), describe what the following "
            "code/conversation excerpt is about, for search retrieval:\n\n"
            f"{content[:400]}"
        )
        context = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "Output only one descriptive sentence. "
                "Be specific about functions, files, or errors mentioned."
            ),
            model_override=self._f.valves.llm_model,
            max_tokens=40,
            temperature=0.2,
        )
        if context and context.strip():
            return f"[Context: {context.strip()}]\n\n"
        return ""

    async def _retrieve_by_symbol(
        self, symbol: str, cleaned_query: str, project_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve memories filtered by a specific code symbol."""
        now = time.time()
        where = {
            "$and": [
                {"project_id": {"$eq": project_id}},
                {"code_symbols": {"$contains": f",{symbol},"}},
            ]
        }

        # ── FIX 11: Expiration filter with OR for summaries ──
        # Apply the same OR logic here so that summaries are not excluded.
        if self._f.valves.long_term_memory_expiration_days > 0:
            where["$and"].append(
                {
                    "$or": [
                        {"expires_at": {"$gt": now}},
                        {"is_session_summary": {"$eq": True}},
                        {"is_turn_summary": {"$eq": True}},
                        {"is_raptor_summary": {"$eq": True}},
                        {"is_hierarchical_summary": {"$eq": True}},
                    ]
                }
            )

        q_emb = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode(cleaned_query[:1000]).tolist()
        )
        results = await anyio.to_thread.run_sync(
            lambda: self._f.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=self._f.valves.long_term_memory_top_k * 2,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        )
        docs_with_meta = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                sim = 1.0 - (results["distances"][0][i] / 2.0)
                ts = meta.get("timestamp")
                if ts is not None and ts < 1000000000:
                    ts = None
                if self._f.valves.ltm_time_decay_hours > 0 and ts is not None:
                    age_hours = (now - ts) / 3600
                    sim *= 0.5 ** (age_hours / self._f.valves.ltm_time_decay_hours)
                if sim >= self._f.valves.long_term_memory_similarity_threshold:
                    docs_with_meta.append((doc, sim, ts, meta))
        docs_with_meta.sort(key=lambda x: x[1], reverse=True)
        docs_with_meta = docs_with_meta[: self._f.valves.long_term_memory_top_k]
        if not docs_with_meta and self._f.valves.ltm_symbol_force_fallback_to_semantic:
            return await self.retrieve_memories_unified(cleaned_query, project_id)
        return [{"doc": doc, "timestamp": ts} for doc, _, ts, _ in docs_with_meta]

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Main retrieval methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def retrieve_memories_unified(
        self,
        query: str,
        project_id: str,
        use_case_label: str = "General Programming",
        slot_free: bool = True,
    ) -> list:
        """
        Retrieve relevant LTM entries, with multi‑query expansion and reranking.
        """
        # ── C4: early exit if retrieval disabled ──────────────────────────
        if self._retrieval_disabled_reason:
            self._f._log_debug(
                f"LTM retrieval skipped: {self._retrieval_disabled_reason}"
            )
            return []

        # ------------------------------------------------------------------
        # REGION 1: Early exits
        # ------------------------------------------------------------------
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return []

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            return await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )

        # ------------------------------------------------------------------
        # REGION 2: Build query variants (thematic expansion)
        # ------------------------------------------------------------------
        query_variants = await self._expand_query_for_retrieval(
            query, use_case_label=use_case_label, slot_free=slot_free
        )

        try:
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}

            # Expiration filter with OR for summaries
            if self._f.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append(
                    {
                        "$or": [
                            {"expires_at": {"$gt": now}},
                            {"is_session_summary": {"$eq": True}},
                            {"is_turn_summary": {"$eq": True}},
                            {"is_raptor_summary": {"$eq": True}},
                            {"is_hierarchical_summary": {"$eq": True}},
                        ]
                    }
                )

            # ── C2: deduplicate results by memory_id, keeping highest raw score ──
            all_raw_results: Dict[str, Tuple[str, float, Any, Any]] = {}

            # ------------------------------------------------------------------
            # REGION 3: Retrieve for each variant
            # ------------------------------------------------------------------
            for variant_query in query_variants:
                q_emb = await anyio.to_thread.run_sync(
                    lambda q=variant_query: self._f.embedder.encode(
                        self._f._tokens.truncate_text_to_tokens(q, 32768)
                    ).tolist()
                )
                try:
                    variant_results = await anyio.to_thread.run_sync(
                        lambda emb=q_emb: self._f.memory_collection.query(
                            query_embeddings=[emb],
                            n_results=self._f.valves.long_term_memory_top_k * 2,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"],
                        )
                    )
                except Exception as e:
                    self._f._log_debug(f"Multi-query retrieval failed for variant: {e}")
                    continue

                if not variant_results or not variant_results["documents"]:
                    continue

                for i, doc in enumerate(variant_results["documents"][0]):
                    meta = variant_results["metadatas"][0][i]
                    raw_sim = 1.0 - (variant_results["distances"][0][i] / 2.0)

                    # ── C1: apply threshold on RAW similarity ──────────────────
                    if raw_sim < self._f.valves.long_term_memory_similarity_threshold:
                        continue

                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None

                    mem_id = meta.get(
                        "memory_id",
                        hashlib.md5(doc.encode()).hexdigest()[:16],
                    )

                    # ── C2: deduplicate, keep highest raw score ─────────────────
                    if (
                        mem_id not in all_raw_results
                        or raw_sim > all_raw_results[mem_id][1]
                    ):
                        all_raw_results[mem_id] = (doc, raw_sim, ts, meta)

            # ── C1: apply time decay for ranking (not for filtering) ──────────
            docs_with_meta = []
            for mem_id, (doc, raw_sim, ts, meta) in all_raw_results.items():
                if self._f.valves.ltm_time_decay_hours > 0 and ts is not None:
                    age_hours = (now - ts) / 3600
                    effective_sim = raw_sim * (
                        0.5 ** (age_hours / self._f.valves.ltm_time_decay_hours)
                    )
                else:
                    effective_sim = raw_sim

                if meta.get("is_raptor_summary"):
                    raptor_level = meta.get("raptor_level", 1)
                    effective_sim *= 1.0 + 0.1 * raptor_level

                docs_with_meta.append((doc, effective_sim, ts, meta, raw_sim))

            if self._f.valves.preserve_error_context:
                new_docs = []
                for doc, eff_sim, ts, meta, raw_sim in docs_with_meta:
                    if meta.get("content_type") == ContentType.ERROR.value:
                        eff_sim *= 1.1
                    new_docs.append((doc, eff_sim, ts, meta, raw_sim))
                docs_with_meta = new_docs

            docs_with_meta.sort(key=lambda x: x[1], reverse=True)

            if self._f.valves.ltm_symbol_boost_enabled and query:
                query_symbols = self._extract_query_symbols(query, project_id)
                if query_symbols:
                    new_docs = []
                    for doc, eff_sim, ts, meta, raw_sim in docs_with_meta:
                        meta_symbols_str = meta.get("code_symbols", "")
                        if (
                            meta_symbols_str
                            and eff_sim
                            >= self._f.valves.ltm_symbol_boost_min_similarity
                        ):
                            meta_symbols = set(meta_symbols_str.split(","))
                            common = query_symbols.intersection(meta_symbols)
                            if common:
                                eff_sim *= self._f.valves.ltm_symbol_boost_factor
                        new_docs.append((doc, eff_sim, ts, meta, raw_sim))
                    new_docs.sort(key=lambda x: x[1], reverse=True)
                    docs_with_meta = new_docs

            if (
                self._f.valves.enable_reranking
                and self._f._cross_encoder
                and docs_with_meta
            ):
                rerank_k = min(
                    (
                        self._f.valves.reranker_top_k
                        if self._f.valves.reranker_top_k > 0
                        else self._f.valves.long_term_memory_top_k
                    ),
                    50,
                )
                docs_only = [d[0] for d in docs_with_meta[: rerank_k * 2]]
                reranked = await self._rerank_results(query, docs_only, rerank_k)
                doc_to_meta = {
                    d[0]: (d[1], d[2], d[3] if len(d) > 3 else {})
                    for d in docs_with_meta
                }
                docs_with_meta = [
                    (doc, *doc_to_meta.get(doc, (0.0, None, {}))) for doc in reranked
                ]

            docs_with_meta = docs_with_meta[: self._f.valves.long_term_memory_top_k]

            # ------------------------------------------------------------------
            # REGION 5: Normalize output
            # ------------------------------------------------------------------
            normalized = []
            for entry in docs_with_meta:
                if len(entry) == 5:
                    doc, score, ts, meta_dict, _ = entry
                elif len(entry) == 4:
                    doc, score, ts, meta_dict = entry
                elif len(entry) == 3:
                    doc, score, ts = entry
                    meta_dict = {}
                else:
                    doc, score, ts = entry[0], entry[1], entry[2]
                    meta_dict = entry[3] if len(entry) > 3 else {}
                normalized.append((doc, score, ts, meta_dict))

            return [
                {"doc": doc, "timestamp": ts, "meta": meta_dict}
                for doc, _, ts, meta_dict in normalized
            ]

        except Exception as e:
            logger.warning(f"Unified memory retrieval failed: {e}")
            return []

    async def retrieve_historical_messages(
        self, query: str, project_id: str, limit: int
    ) -> list:
        """Retrieve historically relevant messages from ChromaDB LTM."""
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return []

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            memories = await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )
            return [{"role": "user", "content": m["doc"]} for m in memories]

        try:
            # Requires an embedder supporting 32768 context or more.
            q_emb = await anyio.to_thread.run_sync(
                lambda: self._f.embedder.encode(
                    self._f._tokens.truncate_text_to_tokens(query, 32768)
                ).tolist()
            )
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}

            # ── FIX 11: Expiration filter with OR for summaries ──
            # Same as in retrieve_memories_unified: include summaries explicitly.
            if self._f.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append(
                    {
                        "$or": [
                            {"expires_at": {"$gt": now}},
                            {"is_session_summary": {"$eq": True}},
                            {"is_turn_summary": {"$eq": True}},
                            {"is_raptor_summary": {"$eq": True}},
                            {"is_hierarchical_summary": {"$eq": True}},
                        ]
                    }
                )

            results = await anyio.to_thread.run_sync(
                lambda: self._f.memory_collection.query(
                    query_embeddings=[q_emb],
                    n_results=limit * 3,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            )

            query_symbols = set()
            if self._f.valves.ltm_symbol_boost_enabled and query:
                query_symbols = self._extract_query_symbols(query, project_id)

            scored_regular = []
            scored_summaries = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    dist = results["distances"][0][i]
                    sim = 1.0 - (dist / 2.0)
                    if (
                        query_symbols
                        and sim >= self._f.valves.ltm_symbol_boost_min_similarity
                    ):
                        meta_symbols_str = meta.get("code_symbols", "")
                        if meta_symbols_str:
                            meta_symbols = set(meta_symbols_str.split(","))
                            common = query_symbols.intersection(meta_symbols)
                            if common:
                                sim *= self._f.valves.ltm_symbol_boost_factor
                    role = meta.get("role", "user")
                    is_summary = meta.get("is_hierarchical_summary", False) or meta.get(
                        "is_session_summary", False
                    )
                    entry = (sim, {"role": role, "content": doc})
                    if is_summary:
                        scored_summaries.append(entry)
                    else:
                        scored_regular.append(entry)

            scored_summaries.sort(key=lambda x: x[0], reverse=True)
            scored_regular.sort(key=lambda x: x[0], reverse=True)

            messages = [msg for _, msg in scored_summaries] + [
                msg for _, msg in scored_regular
            ]

            if (
                self._f.valves.enable_reranking
                and self._f._cross_encoder
                and len(messages) > 1
            ):
                docs = [m["content"] for m in messages]
                reranked = await self._rerank_results(query, docs, limit)
                doc_to_msg = {m["content"]: m for m in messages}
                messages = [doc_to_msg[doc] for doc in reranked if doc in doc_to_msg]

            return messages[:limit]
        except Exception as e:
            logger.warning(f"Historical message retrieval failed: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Message storage
    # ═══════════════════════════════════════════════════════════════════════════

    # ── M1: strip CoT scaffolding ──────────────────────────────────────────
    _LTM_STRIP_RE = re.compile(
        r"<details[^>]*>.*?</details>",
        re.DOTALL | re.IGNORECASE,
    )

    def _prepare_text_for_ltm(self, text: str, role: str) -> str:
        """Strip model CoT scaffolding before embedding into ChromaDB.
        Only applied to assistant messages. The original message content
        shown to the user is NOT modified."""
        if role != "assistant":
            return text
        stripped = self._LTM_STRIP_RE.sub("", text).strip()
        return stripped

    # ── M2: skip partial multi‑phase responses ─────────────────────────────
    _MULTI_PHASE_CONTINUATION_MARKER = "▶ CONTINÚA:"
    _MULTI_PHASE_PART_RE = re.compile(
        r"##\s+C[oó]digo\s*[—–-]\s*Parte\s+\d+/\d+", re.IGNORECASE
    )

    def _is_partial_multi_phase(self, text: str) -> bool:
        return self._MULTI_PHASE_CONTINUATION_MARKER in text or bool(
            self._MULTI_PHASE_PART_RE.search(text)
        )

    async def store_messages(
        self, project_id: str, messages: list, wait: bool = True
    ) -> None:
        """
        Store user/assistant messages in the LTM ChromaDB collection.

        If `wait` is False, the actual embedding and upsert run in a background
        task and the method returns immediately.
        """
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return

        # ── Filter valid messages (skip partial multi-phase) ──────────
        valid = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or len(content.strip()) < 15:
                continue

            role = msg.get("role", "")
            # M2: Skip partial multi-phase assistant responses
            if role == "assistant" and self._is_partial_multi_phase(content):
                self._f._log_debug(
                    "ltm: skipping partial multi-phase response (not embedding)"
                )
                continue

            # M1: Strip CoT scaffolding
            content = self._prepare_text_for_ltm(content, role)
            if not content:
                continue

            valid.append({"role": role, "content": content})

        if not valid:
            return

        if not wait:
            # Offload the heavy embedding to a background task
            asyncio.create_task(self._store_messages_async(project_id, valid))
            return

        # ── Original synchronous path ──────────────────────────────────────
        texts_for_embedding: List[str] = []
        documents_to_store: List[str] = []
        ids = []
        metadatas = []
        now = time.time()

        for i, msg in enumerate(valid):
            content = msg["content"]

            extracted, _ = await self._f._code_blocks.extract_code_blocks(content)
            content_type = self._f._code_blocks.classify_content(content, extracted)

            ctx_symbols: List[str] = []
            for blk in extracted[:3]:
                try:
                    syms = await SignatureExtractor.extract_async(
                        blk["code"], language=blk.get("language")
                    )
                    for sym in syms:
                        if self._is_symbol_indexable(sym):
                            ctx_symbols.append(sym.name)
                            if len(ctx_symbols) >= 10:
                                break
                except Exception:
                    pass

            ctx_file_paths: List[str] = []
            if self._f.valves.track_file_paths:
                ctx_file_paths = self._f._code_blocks.extract_file_paths(content)[:3]

            context_prefix = await self._build_retrieval_context(
                content=content,
                project_id=project_id,
                role=msg.get("role", "user"),
                code_symbols=ctx_symbols[:6],
                file_paths=ctx_file_paths,
                content_type=content_type.value,
            )

            contextual_doc = context_prefix + content
            texts_for_embedding.append(contextual_doc)
            documents_to_store.append(contextual_doc)

            msg_id = f"{project_id}_{int(now)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            ids.append(msg_id)

            expires_at = (
                now + (self._f.valves.long_term_memory_expiration_days * 86400)
                if self._f.valves.long_term_memory_expiration_days > 0
                else None
            )

            code_symbols_str = ""
            if self._f.valves.ltm_index_symbols_enabled:
                all_syms = set()
                for blk in extracted:
                    try:
                        syms = await SignatureExtractor.extract_async(
                            blk["code"], language=blk.get("language")
                        )
                        for sym in syms:
                            if self._is_symbol_indexable(sym):
                                all_syms.add(sym.name)
                                if (
                                    len(all_syms)
                                    >= self._f.valves.ltm_symbol_index_max_per_message
                                ):
                                    break
                    except Exception:
                        pass
                if all_syms:
                    code_symbols_str = "," + ",".join(sorted(all_syms)) + ","

            metadatas.append(
                {
                    "role": msg.get("role"),
                    "project_id": project_id,
                    "timestamp": now,
                    "expires_at": expires_at,
                    "content_type": content_type.value,
                    "has_code": len(extracted) > 0,
                    "code_symbols": code_symbols_str,
                    "memory_id": msg_id,
                }
            )

        # Requires an embedder supporting 32768 context or more.
        safe_texts = [
            self._f._tokens.truncate_text_to_tokens(t, 32768)
            for t in texts_for_embedding
        ]
        embeddings = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode(safe_texts, convert_to_numpy=True).tolist()
        )

        if ids:
            await anyio.to_thread.run_sync(
                lambda: self._f.memory_collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_to_store,
                )
            )

    async def _store_messages_async(self, project_id: str, valid: list) -> None:
        """Store each message one by one, offloading embedding to a thread."""
        for msg in valid:
            try:
                await self._store_single_message(project_id, msg)
            except Exception as e:
                self._f._log_debug(f"Async LTM store failed: {e}")

    async def _store_single_message(self, project_id: str, msg: dict) -> None:
        """Embed and insert a single message into ChromaDB (used by async path)."""
        content = msg["content"]
        extracted, _ = await self._f._code_blocks.extract_code_blocks(content)
        content_type = self._f._code_blocks.classify_content(content, extracted)

        ctx_symbols: List[str] = []
        for blk in extracted[:3]:
            try:
                # ── FIX 9: Pass language as keyword argument ──
                syms = await SignatureExtractor.extract_async(
                    blk["code"], language=blk.get("language")
                )
                for sym in syms:
                    if self._is_symbol_indexable(sym):
                        ctx_symbols.append(sym.name)
                        if len(ctx_symbols) >= 10:
                            break
            except Exception:
                pass

        ctx_file_paths: List[str] = []
        if self._f.valves.track_file_paths:
            ctx_file_paths = self._f._code_blocks.extract_file_paths(content)[:3]

        context_prefix = await self._build_retrieval_context(
            content=content,
            project_id=project_id,
            role=msg.get("role", "user"),
            code_symbols=ctx_symbols[:6],
            file_paths=ctx_file_paths,
            content_type=content_type.value,
        )

        contextual_doc = context_prefix + content
        safe_text = self._f._tokens.truncate_text_to_tokens(contextual_doc, 32768)
        now = time.time()

        embedding = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode(safe_text).tolist()
        )

        msg_id = (
            f"{project_id}_{int(now)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        )
        expires_at = (
            now + (self._f.valves.long_term_memory_expiration_days * 86400)
            if self._f.valves.long_term_memory_expiration_days > 0
            else None
        )

        code_symbols_str = ""
        if self._f.valves.ltm_index_symbols_enabled:
            all_syms = set()
            for blk in extracted:
                try:
                    # ── FIX 9: Pass language as keyword argument ──
                    syms = await SignatureExtractor.extract_async(
                        blk["code"], language=blk.get("language")
                    )
                    for sym in syms:
                        if self._is_symbol_indexable(sym):
                            all_syms.add(sym.name)
                            if (
                                len(all_syms)
                                >= self._f.valves.ltm_symbol_index_max_per_message
                            ):
                                break
                except Exception:
                    pass
            if all_syms:
                code_symbols_str = "," + ",".join(sorted(all_syms)) + ","

        metadata = {
            "role": msg.get("role"),
            "project_id": project_id,
            "timestamp": now,
            "expires_at": expires_at,
            "content_type": content_type.value,
            "has_code": len(extracted) > 0,
            "code_symbols": code_symbols_str,
            "memory_id": msg_id,
        }

        await anyio.to_thread.run_sync(
            lambda: self._f.memory_collection.upsert(
                ids=[msg_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[contextual_doc],
            )
        )

    async def store_response_in_cache(
        self,
        query: str,
        response: str,
        context_hash: str,
        state: dict,
        code_state_hash: str,
        wait: bool = True,
    ) -> None:
        """Store a response in the ChromaDB response cache for future reuse.
        If `wait` is False, the embedding and upsert are offloaded to a background task.
        """
        if not self._f.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return

        if not wait:
            asyncio.create_task(
                self._store_response_in_cache_async(
                    query, response, context_hash, state, code_state_hash
                )
            )
            return

        await self._store_response_in_cache_sync(
            query, response, context_hash, state, code_state_hash
        )

    async def _store_response_in_cache_sync(
        self,
        query: str,
        response: str,
        context_hash: str,
        state: dict,
        code_state_hash: str,
    ) -> None:
        """Synchronous version of response cache storage."""
        col = getattr(self._f, "_response_cache_collection", None)
        if col is None:
            return

        embedding = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
        entry_id = hashlib.md5(
            f"{self._f.valves.project_id}|{query}".encode()
        ).hexdigest()[:32]
        max_entries = self._f.valves.response_cache_max_entries
        project = self._f.valves.project_id

        pstate = self._f._project_state_manager.get_pstate(project)
        current_size = pstate.get("response_cache_count", 0)

        if current_size >= max_entries:
            to_delete_count = max(1, max_entries // 10)
            try:
                old_entries = await anyio.to_thread.run_sync(
                    lambda: col.get(
                        where={"project_id": project},
                        include=["metadatas"],
                        limit=to_delete_count,
                    )
                )
                if old_entries and old_entries["ids"]:
                    await anyio.to_thread.run_sync(
                        lambda: col.delete(ids=old_entries["ids"])
                    )
                    pstate["response_cache_count"] = max(
                        0, current_size - len(old_entries["ids"])
                    )
            except Exception:
                pass

        await anyio.to_thread.run_sync(
            lambda: col.upsert(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[response],
                metadatas=[
                    {
                        "query": query[:500],
                        "project_id": project,
                        "context_hash": "",
                        "code_state_hash": code_state_hash,
                        "timestamp": time.time(),
                    }
                ],
            )
        )
        pstate["response_cache_count"] = pstate.get("response_cache_count", 0) + 1

    async def _store_response_in_cache_async(
        self,
        query: str,
        response: str,
        context_hash: str,
        state: dict,
        code_state_hash: str,
    ) -> None:
        """Background wrapper for response cache storage."""
        try:
            await self._store_response_in_cache_sync(
                query, response, context_hash, state, code_state_hash
            )
        except Exception as e:
            self._f._log_debug(f"Async response cache store failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Maintenance (purge expired memories)
    # ═══════════════════════════════════════════════════════════════════════════

    async def purge_expired_memories(self) -> None:
        """Remove memories whose expires_at timestamp is in the past."""
        await asyncio.sleep(0)
        if not HAS_CHROMA or self._f.memory_collection is None:
            return
        if self._f.valves.long_term_memory_expiration_days <= 0:
            return
        try:
            await anyio.to_thread.run_sync(self._do_purge)
        except Exception as e:
            logger.warning(f"Purge failed: {e}")

    def _do_purge(self) -> None:
        now = time.time()
        expired = self._f.memory_collection.get(where={"expires_at": {"$lt": now}})
        if expired and expired["ids"]:
            self._f.memory_collection.delete(ids=expired["ids"])
            self._f._log_debug(f"Purged {len(expired['ids'])} expired memories")

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Project purge – NEW (C3)
    # ═══════════════════════════════════════════════════════════════════════════

    def purge_project(self, project_id: str) -> int:
        """Delete all ChromaDB documents for a project.

        Returns the number of documents deleted.
        """
        try:
            existing = self._f.memory_collection.get(
                where={"project_id": {"$eq": project_id}}
            )
            ids_to_delete: List[str] = existing.get("ids", [])
            if ids_to_delete:
                self._f.memory_collection.delete(ids=ids_to_delete)
                self._f._log_debug(
                    f"LTM: purged {len(ids_to_delete)} documents for project '{project_id}'"
                )
            return len(ids_to_delete)
        except Exception as e:
            self._f._log_debug(f"LTM: purge_project failed for '{project_id}': {e}")
            return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Embedding model validation – NEW (C4)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_embedding_dimension(self) -> int:
        """Return the output dimension of the current embedding model."""
        test_vec = self._f.embedder.encode(["test"])
        return len(test_vec[0]) if test_vec else 0

    def _validate_embedding_model(self) -> bool:
        """Validate that the current embedding model matches the stored one.

        Reads/writes model metadata from the ChromaDB collection.
        Returns False and sets _retrieval_disabled if a mismatch is found.
        """
        try:
            current_model: str = getattr(
                self._f.valves, "embedding_model_name", "unknown"
            )
            current_dim: int = self._get_embedding_dimension()

            coll_meta: dict = self._f.memory_collection.metadata or {}
            stored_model: Optional[str] = coll_meta.get("_codeaware_embedding_model")
            stored_dim: Optional[int] = coll_meta.get("_codeaware_embedding_dim")

            if stored_model is None:
                # First run: persist model fingerprint
                self._f.memory_collection.modify(
                    metadata={
                        **coll_meta,
                        "_codeaware_embedding_model": current_model,
                        "_codeaware_embedding_dim": current_dim,
                    }
                )
                self._f._log_debug(
                    f"LTM: embedding fingerprint stored (model={current_model}, dim={current_dim})"
                )
                return True

            if stored_model != current_model or stored_dim != current_dim:
                reason = (
                    f"LTM embedding mismatch — collection built with "
                    f"'{stored_model}' (dim={stored_dim}), "
                    f"current model is '{current_model}' (dim={current_dim}). "
                    f"LTM retrieval DISABLED. To fix: clear the ChromaDB collection."
                )
                self._f._log_debug(reason)
                self._retrieval_disabled_reason = reason
                return False

            return True

        except Exception as e:
            # Fail open to avoid breaking existing installs that lack the metadata
            self._f._log_debug(f"LTM: could not validate embedding model: {e}")
            return True


class LLMOrchestrator:
    """Centralised LLM caller with built‑in response cache, retry logic,
    and task deduplication.

    Provides:
    * ``call_llm(prompt, ...)`` — the single entry point for all LLM calls.
      Retries transient failures (429, 5xx) with a configurable total
      deadline and an exponential backoff.
    * In‑memory response cache (``AsyncLRUCache``) keyed by prompt hash,
      shared across the whole process.
    * Deduplication of concurrent identical prompts via per‑key futures,
      so parallel background tasks (docstrings, summaries) never fire
      duplicate LLM requests.
    * A concurrency semaphore (``_llm_semaphore``) that serialises
      inference for llama.cpp's ``--parallel 1`` mode.
    * ``should_keep_full_code(query)`` — lightweight CrossEncoder call
      that decides whether the user wants the full implementation or a
      summary.
    * ``wait_for_slot()`` / ``wait_for_llm_tasks()`` — coordination
      primitives used by the inlet and outlet to avoid dirtying the KV
      cache while auxiliary LLM work is in flight.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Cache & concurrency initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def init_cache(self) -> None:
        """Return the shared AsyncLRUCache instance for LLM response caching."""
        self._f._llm_cache = _AsyncLRUCache(
            max_size=self._f.valves.LLM_CACHE_MAX_SIZE,
            ttl=self._f.valves.LLM_CACHE_TTL,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Main LLM caller (with retries, cache, deduplication)
    # ═══════════════════════════════════════════════════════════════════════════

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        label: str = "",
        total_timeout: Optional[float] = None,
    ) -> Optional[str]:
        """
        Call the LLM with cache and deduplication. Retries are handled by
        shared_resources.call_llm internally.

        All calls to this method are serialized via `_llm_semaphore` to prevent
        concurrent LLM requests, which avoids cancellation issues when using
        `--parallel 1` in llama.cpp.
        """
        # ── Silent ingestion guard ──
        if getattr(self._f, "_is_silent_ingestion", False) and label not in (
            "bg_docstring",
        ):
            return None

        dedup_key = hashlib.md5(
            f"{prompt}|{system_prompt}|{temperature}|{max_tokens}|{model_override}".encode()
        ).hexdigest()
        async with self._f._pending_llm_lock:
            if dedup_key in self._f._pending_llm:
                future = self._f._pending_llm[dedup_key]
                is_producer = False
            else:
                future = asyncio.Future()
                self._f._pending_llm[dedup_key] = future
                is_producer = True

        if not is_producer:
            return await future

        t_start = time.monotonic()
        label_str = f" ({label})" if label else ""

        # ── SERIALIZATION WITH THE SEMAPHORE ──────────────────────────────────
        # All LLM calls (main response, CoT, docstrings, summaries...)
        # are serialized here. This guarantees that only one executes at a time,
        # preventing concurrency cancellations when the server uses --parallel 1.
        async with self._f._llm_semaphore:
            try:
                base_url = self._f.valves.LLM_BASE_URL.rstrip("/")
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3].rstrip("/")

                is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

                model = model_override or self._f.valves.llm_model
                if not model:
                    logger.warning(f"[LLM]{label_str} No model available")
                    future.set_result(None)
                    return None

                # ── Cache LLM ──
                cache_key = hashlib.md5(
                    f"{model}|{prompt}|{system_prompt}|{temperature}|{max_tokens}".encode()
                ).hexdigest()
                cached = await self._f._llm_cache.get(cache_key)
                if cached is not None:
                    future.set_result(cached)
                    self._f._log_debug(
                        f"[LLM] {model}{label_str} (cached) took {time.monotonic() - t_start:.3f}s"
                    )
                    return cached

                ep_type = "chat"
                if model.startswith("llamacpp/"):
                    ep_type = self._f.valves.llamacpp_endpoint_type

                if self._f.tokenizer:
                    prompt_tokens = len(self._f.tokenizer.encode(prompt))
                    self._f._log_debug(
                        f"LLM call to {model}{label_str} – prompt size: ~{prompt_tokens} tokens"
                    )

                # ── Real call (with internal retries in shared_resources) ──
                task = asyncio.current_task()
                async with self._f._active_llm_tasks_lock:
                    self._f._active_llm_tasks.add(task)
                try:
                    content = await _shared_call_llm(
                        prompt=prompt,
                        system=system_prompt,
                        base_url=self._f.valves.LLM_BASE_URL,
                        model=model,
                        api_token=self._f.valves.LLM_API_TOKEN,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self._f.valves.llm_request_timeout,
                        endpoint_type=ep_type,
                    )

                    if content:
                        await self._f._llm_cache.set(cache_key, content)
                        future.set_result(content)
                        async with self._f._model_lock:
                            self._f._last_used_model = model
                        in_tokens = (
                            len(self._f.tokenizer.encode(prompt))
                            if self._f.tokenizer
                            else "?"
                        )
                        out_tokens = (
                            len(self._f.tokenizer.encode(content))
                            if self._f.tokenizer
                            else "?"
                        )
                        self._f._log_debug(
                            f"[LLM] {model}{label_str} – in:{in_tokens} out:{out_tokens}"
                            f" took {time.monotonic() - t_start:.3f}s"
                        )
                        return content
                    else:
                        future.set_result(None)
                        return None

                finally:
                    async with self._f._active_llm_tasks_lock:
                        self._f._active_llm_tasks.discard(task)

            except Exception as e:
                future.set_exception(e)
                raise
            finally:
                async with self._f._pending_llm_lock:
                    self._f._pending_llm.pop(dedup_key, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Coordination primitives (slot & task waiting)
    # ═══════════════════════════════════════════════════════════════════════════

    async def wait_for_llm_tasks(self) -> None:
        """
        Wait until all LLM-using tasks have completed.

        Since all LLM calls are serialized via `_llm_semaphore` (limit 1),
        waiting for the semaphore to be fully available guarantees that
        no LLM calls are currently in progress and no tasks are pending.
        """
        async with self._f._llm_semaphore:
            pass

    async def wait_for_slot(self) -> None:
        """
        Wait until the inference slot is free.

        This is an alias for `wait_for_llm_tasks()` that provides semantic
        clarity when the caller only needs to know the slot is available.
        """
        async with self._f._llm_semaphore:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. CrossEncoder helper (keep full code decision)
    # ═══════════════════════════════════════════════════════════════════════════

    async def should_keep_full_code(self, user_question: str) -> bool:
        """
        Decide whether to keep the full code in context or provide only a summary.
        Uses the CrossEncoder for fast CPU inference.
        Returns True if full code should be kept.
        """
        if not user_question.strip():
            return False

        pairs = [
            (
                user_question[:500],
                "The user wants the full code, complete implementation, or exact details.",
            ),
            (
                user_question[:500],
                "The user only needs a summary, brief explanation, or high-level overview.",
            ),
        ]
        scores = await self._f._commands._predict_cross_encoder(pairs)
        if scores is None:
            self._f._log_debug(
                "_should_keep_full_code: CrossEncoder not loaded, keeping full code by default."
            )
            return True
        return scores[0] > scores[1]


class ReasoningEngine:
    """Detects when Chain‑of‑Thought reasoning is appropriate and generates
    reasoning chains using the LLM.

    Provides:
    * ``detect_cot_level(query)`` — returns 0‑3 (inconclusive → deep
      scientific reasoning) using either a CrossEncoder classifier or a
      keyword‑based heuristic.
    * ``generate_cot_reasoning(question, context)`` — produces a step‑by‑step
      reasoning block (Level 2).
    * ``generate_architecture_reasoning(question, skeleton, project_id)`` —
      Level 2 reasoning that works on the code skeleton (contracts only)
      instead of the full system prompt, producing /expand hints for
      implementation.
    * ``generate_scientific_reasoning_L3(question, context, project_id)`` —
      multi‑hypothesis reasoning validated against the SymbolGraph
      (StaticEvidence).
    * ``generate_scientific_architecture_reasoning(...)`` — Level 3
      architecture reasoning that evaluates competing design options
      against structural evidence.
    * ``is_architecture_query(query)`` — fast regex test to route design /
      refactor queries to the skeleton‑based reasoning path.
    * ``parse_cot_intent(content)`` — extracts the question and level from
      explicit ``/think`` commands.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ── Architecture/design intent detection (regex) ─────────────────────

    _ARCH_INTENT_RE = re.compile(
        r"\b(arquitectura|architecture|diseño|design|refactor(?:iza)?r?|"
        r"plan\s+de\s+(implementaci[oó]n|cambios)|dise[ñn]a|restructur|"
        r"reorganiz|breakdown|dependency|dependencias|"
        r"c[oó]mo\s+(estructurar|organizar|dividir)|"
        r"qu[eé]\s+(clases|m[oó]dulos|componentes)\s+(necesito|crear|a[ñn]adir)|"
        r"propuesta\s+de|propose\s+(a\s+)?design|"
        r"abstract\s+(base\s+)?class|interface\s+design|"
        r"contrato|contracts?|API\s+surface|surface\s+area)\b",
        re.IGNORECASE,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Query classification (architecture intent, /think command parsing)
    # ═══════════════════════════════════════════════════════════════════════════

    def is_architecture_query(self, user_content: str) -> bool:
        """True when the query targets design / architecture / refactoring."""
        return bool(self._ARCH_INTENT_RE.search(user_content))

    async def parse_cot_intent(self, user_content: str) -> Tuple[Optional[str], int]:
        """Parse /think command: returns (question, level) or (None, 2)."""
        content = user_content.strip()
        if not content.startswith("/think"):
            return None, 2
        rest = content[6:].strip()
        if not rest:
            return None, 2
        parts = rest.split(maxsplit=1)
        if parts[0].isdigit():
            level = int(parts[0])
            if level not in (1, 2, 3):
                level = 2
            question = parts[1] if len(parts) > 1 else ""
        else:
            level = 2
            question = rest
        return question, level

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CoT level detection (CrossEncoder / heuristic)
    # ═══════════════════════════════════════════════════════════════════════════

    async def detect_cot_level(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        """
        Determine CoT depth, optionally storing it in conversation state.

        Returns:
            0 — inconclusive, let LLM decide
            1 — simple, inject a think-step-by-step prompt
            2 — complex, generate a CoT reasoning chain
            3 — deep, generate CoT reasoning + self-reflection
        """
        if not user_content:
            return 0

        # ── v7 (PASO-15): force Level 3 Scientific CoT if valve is enabled ──
        if self._f.valves.enforce_scientific_method:
            self._f._log_debug("CoT: enforce_scientific_method=True → forcing Level 3")
            return 3

        if self._f.valves.enable_cot_llm_detection:
            level = await self._detect_cot_level_via_llm(
                user_content, is_code_session, state
            )
        else:
            level = self._detect_cot_level_heuristic(
                user_content, is_code_session, state
            )

        # Persist level for conversational continuity if feature is enabled
        if self._f.ENABLE_COT_STICKY:
            state.last_cot_level = level

        return level

    async def _detect_cot_level_via_llm(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        """
        Determine CoT depth using the CrossEncoder (instant CPU inference).
        Falls back to heuristic if CrossEncoder is not available.
        """
        session_type = "code" if is_code_session else "general"
        intent_hint = ""
        if (
            hasattr(self._f, "_user_intent_full_code")
            and self._f._user_intent_full_code is not None
        ):
            intent_hint = (
                "The user likely needs the full code."
                if self._f._user_intent_full_code
                else "The user likely needs only a summary of the code."
            )
        query = f"[Session: {session_type}] {intent_hint} {user_content[:500]}"

        pairs = [
            (query, "The user wants a simple, direct answer without reasoning."),
            (
                query,
                "The user asks a moderately complex question that requires step-by-step thinking.",
            ),
            (query, "The user asks a complex question that needs deep reasoning."),
            (
                query,
                "The user asks an extremely complex or open-ended question requiring exhaustive analysis.",
            ),
        ]
        scores = await self._f._commands._predict_cross_encoder(pairs)
        if scores is None:
            self._f._log_debug(
                "CoT detection via CrossEncoder unavailable, using heuristic."
            )
            return self._detect_cot_level_heuristic(
                user_content, is_code_session, state
            )
        import numpy as np

        best_level = int(np.argmax(scores))
        if best_level == 0:
            return 0
        elif best_level == 1:
            return 1
        elif best_level == 2:
            return 2
        else:
            return 3

    def _detect_cot_level_heuristic(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        """
        Determine the depth of Chain-of-Thought reasoning needed using heuristic keyword analysis.

        Returns:
            0 — inconclusive, let LLM decide
            1 — simple, inject a think-step-by-step prompt
            2 — complex, generate a CoT reasoning chain
            3 — deep, generate CoT reasoning + self-reflection
        """
        # ── Keyword sets ────────────────────────────────────────────────────
        complex_keywords_generic = {
            "explain how",
            "explica cómo",
            "how does",
            "cómo funciona",
            "how to implement",
            "cómo implementar",
            "why does",
            "por qué falla",
            "why is",
            "por qué es",
            "implement",
            "implementa",
            "system design",
            "diseño del sistema",
            "design a",
            "diseña un",
            "design the",
            "diseña el",
            "architecture",
            "arquitectura",
            "build a",
            "construye",
            "create a class",
            "crea una clase",
            "add support for",
            "añadir soporte para",
            "extend the",
            "extiende el",
            "debug",
            "depura",
            "fix the",
            "corrige el",
            "not working",
            "no funciona",
            "doesn't work",
            "no me funciona",
            "keeps failing",
            "sigue fallando",
            "throws an error",
            "lanza un error",
            "unexpected behavior",
            "comportamiento inesperado",
            "memory leak",
            "fuga de memoria",
            "bottleneck",
            "cuello de botella",
            "optimize",
            "optimiza",
            "refactor",
            "refactoriza",
            "migrate",
            "migra",
            "code review",
            "revisión de código",
            "audit",
            "audita",
            "integrate",
            "integra",
            "deploy",
            "despliega",
            "set up",
            "configura el entorno",
            "step by step",
            "paso a paso",
            "full implementation",
            "implementación completa",
            "best practices",
            "buenas prácticas",
            "comprehensive",
            "completo y detallado",
        }

        complex_keywords_code_only = {
            "test",
            "prueba",
            "configure",
            "configura",
            "scaffold",
            "fix",
            "corrige",
            "review",
            "revisa",
            "structure",
            "estructura",
            "compare",
            "compara",
            "improve",
            "mejora",
            "validate",
            "valida",
        }

        deep_keywords = {
            "deep review",
            "revisión profunda",
            "check every step",
            "comprueba cada paso",
            "razonamiento exhaustivo",
            "production ready",
            "listo para producción",
            "edge cases",
            "casos límite",
            "trade-offs",
            "ventajas y desventajas",
            "security review",
            "revisión de seguridad",
            "performance analysis",
            "análisis de rendimiento",
            "ensure correctness",
            "garantiza la corrección",
            "deep reflection",
            "reflexión profunda",
            "self-reflection",
            "auto-reflexión",
            "auto-evalúa",
            "itera varias veces",
            "exhaustive",
            "exhaustivo",
            "all edge cases",
            "todos los casos límite",
        }

        # ── Optional accent normalisation ───────────────────────────────────
        if self._f.ENABLE_ACCENT_NORMALIZATION:
            import unicodedata

            def _normalize(text: str) -> str:
                nfkd = unicodedata.normalize("NFKD", text)
                return nfkd.encode("ascii", "ignore").decode("ascii")

            content_lower = _normalize(user_content.lower())
            complex_keywords_generic = {_normalize(k) for k in complex_keywords_generic}
            complex_keywords_code_only = {
                _normalize(k) for k in complex_keywords_code_only
            }
            deep_keywords = {_normalize(k) for k in deep_keywords}
        else:
            content_lower = user_content.lower()

        word_count = len(user_content.split())
        has_code = "```" in user_content
        length_ok = len(user_content) >= self._f.valves.auto_cot_min_chars
        too_short = word_count < 5

        # ── Context: expand active set when in a code session ──────────────
        active_complex = set(complex_keywords_generic)
        if is_code_session:
            active_complex |= complex_keywords_code_only

        # ── Negation guard ─────────────────────────────────────────────────
        def _is_negated(text: str, kw: str) -> bool:
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx == -1:
                    break
                before = text[:idx].strip().split()[-3:]
                if not any(neg in before for neg in self._f._COT_NEGATION_PREFIXES):
                    return False
                start = idx + 1
            return True

        _sorted_deep = sorted(deep_keywords, key=len, reverse=True)
        _sorted_complex = sorted(active_complex, key=len, reverse=True)

        def _contains_any(text: str, sorted_kw: list) -> bool:
            for kw in sorted_kw:
                if kw in text and not _is_negated(text, kw):
                    return True
            return False

        if too_short and not has_code:
            return 0

        if _contains_any(content_lower, _sorted_deep):
            return 3

        has_complex_kw = _contains_any(content_lower, _sorted_complex)
        is_elaborate = length_ok or word_count > 30

        signals = 0
        if self._f.ENABLE_KEYWORD_COUNT_WEIGHT:
            kw_matches = sum(
                1
                for kw in _sorted_complex
                if kw in content_lower and not _is_negated(content_lower, kw)
            )
            signals += min(kw_matches + 2, 4)
        else:
            if has_complex_kw:
                signals += 3

        if has_code:
            signals += 1
        if is_elaborate:
            signals += 1
        if user_content.count("?") >= 2:
            signals += 1
        if is_code_session and has_complex_kw:
            signals += 1

        for phrase in ("in detail", "en detalle"):
            if phrase in content_lower and not _is_negated(content_lower, phrase):
                signals += 1
                break

        if is_code_session:
            has_stack_trace = (
                "traceback (most recent call last)" in content_lower
                or "traceback:" in content_lower
                or ("exception" in content_lower and "at line" in content_lower)
                or bool(re.search(r'file ".+", line \d+', content_lower))
                or bool(re.search(r"at \w+\.\w+\([\w.]+:\d+\)", content_lower))
            )
            if has_stack_trace:
                signals += 2

        code_block_count = user_content.count("```") // 2
        if code_block_count >= 2:
            signals += 1

        if self._f.ENABLE_COT_STICKY:
            prev_level = state.last_cot_level
            if prev_level >= 2 and has_complex_kw:
                signals += 1

        if signals >= 5:
            return 2
        elif signals >= 3:
            return 1
        else:
            return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Step‑back prompting (architectural context for better CoT)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _generate_step_back_context(
        self, question: str, code_context: str
    ) -> str:
        """
        Generate an architectural step-back for better CoT hypothesis quality.

        Asks: "What high-level principle governs this code?" before diving
        into the specific bug/question.

        Returns a formatted string to prepend to the CoT context,
        or empty string if disabled or the LLM call fails.
        """
        if not self._f.valves.enable_step_back_prompting:
            return ""
        if len(question.strip()) < 15:
            return ""

        debug_signals = (
            "error",
            "fail",
            "bug",
            "wrong",
            "exception",
            "traceback",
            "falla",
            "error",
            "excepción",
            "no funciona",
        )
        question_lower = question.lower()
        if not any(signal in question_lower for signal in debug_signals):
            if not self._f.valves.step_back_always:
                return ""

        step_back_prompt = (
            f"A programmer is debugging this specific issue:\n{question[:300]}\n\n"
            "What is the underlying architectural principle, design invariant, or "
            "general concept that governs correct behavior here? "
            "State it as an abstract question and answer it in 2-3 sentences. "
            "Focus on system-level understanding, not the specific bug."
        )

        step_back_response = await self._f._llm_orchestrator.call_llm(
            prompt=step_back_prompt,
            system_prompt=(
                "You are a senior software architect. "
                "Answer the abstract question concisely (2-3 sentences). "
                "Focus on principles, not the specific implementation."
            ),
            model_override=self._f.valves.cot_model_level2,
            max_tokens=self._f.valves.step_back_max_tokens,
            temperature=0.3,
            label="step_back",
        )

        if step_back_response and step_back_response.strip():
            self._f._log_debug(
                "Step-back context generated "
                f"({len(step_back_response.split())} words)"
            )
            return (
                "## Architectural Context (Step-Back)\n"
                f"{step_back_response.strip()}\n\n"
                "---\n\n"
            )
        return ""

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Level 2 reasoning generation (standard CoT + architecture mode)
    # ═══════════════════════════════════════════════════════════════════════════

    async def generate_cot_reasoning(
        self, question: str, context: str, label: str = ""
    ) -> str:
        """Generate a Chain‑of‑Thought reasoning chain for the given question."""
        effective_max_tokens = (
            self._f.valves.cot_max_tokens if self._f.valves.cot_max_tokens > 0 else None
        )

        step_back = await self._generate_step_back_context(question, context)
        enriched_context = step_back + context if step_back else context

        prompt = (
            f"Context:\n{enriched_context}\n\n"
            f"Question:\n{question}\n\n"
            "Think step by step and provide your reasoning:"
        )
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a helpful assistant that thinks step by step before answering."
            ),
            model_override=self._f.valves.cot_model_level2,
            max_tokens=effective_max_tokens,
            temperature=0.4,
            label=label,
        )
        if response:
            prefix = (
                "## 🔎 Automated Chain-of-Thought Reasoning (Level 2)\n"
                f"*Generated by {self._f.valves.cot_model_level2}.*"
            )
            if step_back:
                prefix += " *Includes step-back architectural context.*"
            return f"{prefix}\n\n{response}"
        return "Unable to generate reasoning."

    async def generate_architecture_reasoning(
        self,
        question: str,
        skeleton_context: str,
        project_id: str,
        label: str = "",
    ) -> str:
        """
        Architecture-mode CoT: reason on the code skeleton (contracts only).
        Falls back to standard CoT if skeleton is empty or LLM fails.
        """
        if not skeleton_context.strip():
            return await self.generate_cot_reasoning(question, "", label=label)

        effective_max_tokens = (
            self._f.valves.skeleton_cot_max_tokens
            if self._f.valves.skeleton_cot_max_tokens > 0
            else 600
        )

        prompt = (
            f"Code skeleton (contracts only — bodies as `...`):\n"
            f"{skeleton_context[:4000]}\n\n"
            f"Architecture question:\n{question[:500]}\n\n"
            "Reason step by step at the CONTRACT level:\n"
            "1. Which classes / methods are affected and why?\n"
            "2. What invariants or interfaces must be preserved or changed?\n"
            "3. What new signatures are needed? Write them as skeleton lines.\n"
            "4. Which existing bodies need to change? List as `/expand <name>`.\n"
            "Be specific about signatures; avoid implementation details."
        )

        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a software architect reasoning about code structure and "
                "contracts. Focus on interfaces, dependencies, and invariants. "
                "When a full implementation is needed, write `/expand <SymbolName>` "
                "instead of generating code."
            ),
            model_override=self._f.valves.cot_model_level2,
            max_tokens=effective_max_tokens,
            temperature=0.0,
            label=label or "arch_cot",
        )

        if not response or response.strip() == "Unable to generate reasoning.":
            self._f._log_debug("Architecture CoT failed — falling back to standard CoT")
            return await self.generate_cot_reasoning(
                question, skeleton_context, label=label
            )

        prefix = (
            "## 🏗️ Architecture Reasoning (skeleton-based CoT)\n"
            f"*Reasoning on contracts — use `/expand <name>` for implementations.*"
        )
        return f"{prefix}\n\n{response}"

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Level 3 scientific reasoning (multi‑hypothesis, evidence‑validated)
    # ═══════════════════════════════════════════════════════════════════════════

    async def generate_scientific_reasoning_L3(
        self, question: str, context: str, project_id: str, label: str = ""
    ) -> str:
        """
        Scientific Chain-of-Thought reasoning with structural validation.

        Flow:
        1. Generate N hypotheses about the answer.
        2. Score each hypothesis using StaticEvidence (deterministic) +
           LLM-expressed confidence (if available).
        3. If the best hypothesis passes the confidence threshold, stop.
        4. Otherwise, feed evidence back to the LLM to refine hypotheses.
        5. Iterate up to scientific_max_iterations times.
        6. Synthesize a final reasoning from the best hypothesis + evidence.
        """
        max_hypotheses = self._f.valves.scientific_hypotheses_count
        threshold = self._f.valves.scientific_confidence_threshold
        max_iters = self._f.valves.scientific_max_iterations

        def _parse_hypotheses_from_response(text: str) -> List[Tuple[str, float]]:
            results = []
            pattern = re.compile(
                r"Hypothesis\s*\d*\s*:\s*(.+?)\s*Confidence\s*:\s*([\d.]+)",
                re.IGNORECASE | re.DOTALL,
            )
            for match in pattern.finditer(text):
                hyp_text = match.group(1).strip().rstrip(".")
                try:
                    conf = float(match.group(2))
                    conf = max(0.0, min(1.0, conf))
                except ValueError:
                    conf = 0.5
                results.append((hyp_text, conf))
            return results

        # ── Step 1: Generate initial hypotheses ────────────────────
        prompt = (
            f"Context:\n{context[:3000]}\n\n"
            f"Question:\n{question[:500]}\n\n"
            f"Propose {max_hypotheses} distinct hypotheses that could explain the issue "
            f"or solve the problem. For each, state:\n"
            f"Hypothesis: <one concise sentence>\n"
            f"Confidence: <0.0-1.0>\n\n"
            f"Be specific: mention function names, files, or data flows if possible."
        )
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a scientific reasoning engine. Output exactly the requested "
                "hypotheses with confidence scores. No extra commentary."
            ),
            model_override=self._f.valves.cot_model_level3,
            max_tokens=600,
            temperature=0.4,
            label=label + "_gen_hypotheses" if label else "sci_gen_hypotheses",
        )

        if not response:
            return "Unable to generate hypotheses for scientific reasoning."

        hypotheses = _parse_hypotheses_from_response(response)
        if len(hypotheses) < 2:
            return await self.generate_cot_reasoning(question, context, label)

        best_hypothesis = ""
        best_combined_score = 0.0
        iteration = 0

        # ── Iterative refinement loop ──────────────────────────────
        while iteration < max_iters:
            iteration += 1
            scored = []
            for hyp_text, llm_conf in hypotheses:
                evidence = self._f._activation._gather_static_evidence(
                    hyp_text, project_id
                )
                obj_score = evidence.objective_score
                combined = 0.5 * obj_score + 0.5 * llm_conf
                scored.append((hyp_text, combined, obj_score, llm_conf, evidence))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[0]
            best_hypothesis, best_combined, best_obj, best_llm_conf, best_evidence = top

            self._f._log_debug(
                f"Scientific CoT iter {iteration}: best hypothesis "
                f"'{best_hypothesis[:80]}...' "
                f"score={best_combined:.3f} "
                f"(obj={best_obj:.3f}, llm_conf={best_llm_conf:.3f})"
            )

            if best_combined >= threshold or iteration >= max_iters:
                break

            evidence_feedback = (
                f"Previous best hypothesis (score {best_combined:.2f}):\n"
                f"{best_hypothesis}\n\n"
                f"Structural evidence:\n"
                f"- Symbols found: {best_evidence.symbols_found}\n"
                f"- Call relations valid: {best_evidence.call_relations_valid}\n"
                f"- Recent changes: {best_evidence.recent_changes}\n"
                f"- Data flow upstream: {best_evidence.data_flow_upstream}\n"
                f"- Objective score: {best_evidence.objective_score:.2f}\n\n"
                f"Based on this evidence, propose {max_hypotheses} improved hypotheses."
            )

            refine_prompt = (
                f"{evidence_feedback}\n\n"
                f"Output the same format as before: Hypothesis: ... Confidence: ..."
            )
            refine_response = await self._f._llm_orchestrator.call_llm(
                prompt=refine_prompt,
                system_prompt=(
                    "You are a scientific reasoning engine refining hypotheses "
                    "based on evidence."
                ),
                model_override=self._f.valves.cot_model_level3,
                max_tokens=600,
                temperature=0.4,
                label=label + "_refine" if label else "sci_refine",
            )
            if refine_response:
                new_hypotheses = _parse_hypotheses_from_response(refine_response)
                if len(new_hypotheses) >= 2:
                    hypotheses = new_hypotheses
                else:
                    break
            else:
                break

        # ── Step 6: Synthesize final reasoning ────────────────────
        final_prompt = (
            f"Context:\n{context[:3000]}\n\n"
            f"Question:\n{question[:500]}\n\n"
            f"The best validated hypothesis (score {best_combined:.3f}):\n"
            f"{best_hypothesis}\n\n"
            f"Structural evidence supporting it:\n"
            f"- Symbols found: {best_evidence.symbols_found}\n"
            f"- Call relations valid: {best_evidence.call_relations_valid}\n"
            f"- Recent changes: {best_evidence.recent_changes}\n"
            f"- Data flow upstream: {best_evidence.data_flow_upstream}\n\n"
            f"Provide a step-by-step reasoning to answer the question, "
            f"grounded in this evidence."
        )
        reasoning = await self._f._llm_orchestrator.call_llm(
            prompt=final_prompt,
            system_prompt=(
                "You are a helpful assistant that reasons step by step "
                "based on verified evidence."
            ),
            model_override=self._f.valves.cot_model_level3,
            max_tokens=(
                self._f.valves.cot_max_tokens
                if self._f.valves.cot_max_tokens > 0
                else None
            ),
            temperature=0.3,
            label=label + "_synthesize" if label else "sci_synthesize",
        )

        if not reasoning:
            return "Unable to synthesize scientific reasoning."

        return (
            f"## 🔬 Scientific Reasoning (Level 3)\n"
            f"*Validated against code structure. "
            f"Best hypothesis score: {best_combined:.2f} "
            f"(obj={best_obj:.2f}, llm_conf={best_llm_conf:.2f})*\n\n"
            f"{reasoning}"
        )

    async def generate_scientific_architecture_reasoning(
        self,
        question: str,
        skeleton_context: str,
        project_id: str,
        label: str = "",
    ) -> str:
        """
        Scientific-method architecture reasoning on skeleton contracts.

        Generates N competing design hypotheses, scores each against static
        evidence from the SymbolGraph, refines iteratively, and synthesises
        a final architecture proposal with concrete signatures and /expand hints.

        Key difference from generate_scientific_reasoning_L3:
          - Context is the skeleton (contracts), not compressed code.
          - Hypotheses are DESIGN OPTIONS (new interfaces, refactored signatures,
            new classes) not debugging explanations.
          - LLM confidence weighted higher (0.6) than objective score (0.4)
            because we are PROPOSING changes, not verifying existing behavior.
          - Output includes skeleton-style signature proposals and /expand hints.

        Falls back to generate_architecture_reasoning if skeleton is empty,
        hypothesis parsing fails, or the LLM is unavailable.
        """
        if not skeleton_context.strip():
            return await self.generate_architecture_reasoning(
                question, skeleton_context, project_id, label=label
            )

        max_hypotheses = self._f.valves.scientific_hypotheses_count
        threshold = self._f.valves.scientific_confidence_threshold
        max_iters = self._f.valves.scientific_max_iterations

        # ── Hypothesis parser (reuses same format as L3) ─────────────────
        def _parse_hypotheses(text: str):
            results = []
            pattern = re.compile(
                r"(?:Design\s+)?(?:Option|Hypothesis)\s*\d*\s*:\s*(.+?)"
                r"\s*Confidence\s*:\s*([\d.]+)",
                re.IGNORECASE | re.DOTALL,
            )
            for m in pattern.finditer(text):
                hyp = m.group(1).strip().rstrip(".")
                try:
                    conf = max(0.0, min(1.0, float(m.group(2))))
                except ValueError:
                    conf = 0.5
                results.append((hyp, conf))
            return results

        # ── Step 1: Generate design hypotheses from skeleton ─────────────
        prompt = (
            f"Code skeleton (contracts — bodies as `...`):\n"
            f"{skeleton_context[:3000]}\n\n"
            f"Architecture question:\n{question[:400]}\n\n"
            f"Propose {max_hypotheses} distinct design options to address this. "
            f"Each option should be concrete: name the classes/methods affected, "
            f"propose new signatures or interfaces as skeleton lines.\n\n"
            f"Format each option as:\n"
            f"Design Option N: <one sentence describing the approach>\n"
            f"Confidence: <0.0-1.0 — how well this fits the existing structure>\n\n"
            f"Consider reuse of existing symbols (higher confidence) vs "
            f"creating new ones (lower until validated)."
        )
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a software architect generating competing design options. "
                "Each option must be concrete and reference existing symbol names "
                "from the skeleton. Output exactly the requested format."
            ),
            model_override=self._f.valves.cot_model_level3,
            max_tokens=600,
            temperature=0.4,
            label=f"{label}_gen_options" if label else "sci_arch_gen",
        )
        if not response:
            return await self.generate_architecture_reasoning(
                question, skeleton_context, project_id, label=label
            )

        hypotheses = _parse_hypotheses(response)
        if len(hypotheses) < 2:
            self._f._log_debug(
                "Scientific arch: could not parse hypotheses — falling back to L2 arch"
            )
            return await self.generate_architecture_reasoning(
                question, skeleton_context, project_id, label=label
            )

        # ── Steps 2-3: Score + iterative refinement ──────────────────────
        best_hypothesis = ""
        best_combined = 0.0
        best_obj = 0.0
        best_llm_conf = 0.0
        best_evidence = None

        for iteration in range(max(1, max_iters)):
            scored = []
            for hyp_text, llm_conf in hypotheses:
                evidence = self._f._activation._gather_static_evidence(
                    hyp_text, project_id
                )
                obj = evidence.objective_score
                # Architecture weight: LLM confidence dominates (proposing, not verifying)
                combined = 0.4 * obj + 0.6 * llm_conf
                scored.append((hyp_text, combined, obj, llm_conf, evidence))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_hypothesis, best_combined, best_obj, best_llm_conf, best_evidence = (
                scored[0]
            )

            self._f._log_debug(
                f"Sci-arch iter {iteration + 1}: best='{best_hypothesis[:60]}…' "
                f"combined={best_combined:.3f} "
                f"(obj={best_obj:.3f}, llm={best_llm_conf:.3f})"
            )

            if best_combined >= threshold or iteration >= max_iters - 1:
                break

            # Refine: feed evidence back to the LLM
            symbols_found = {
                k: v for k, v in best_evidence.symbols_found.items() if not v
            }
            call_invalid = {
                k: v for k, v in best_evidence.call_relations_valid.items() if not v
            }
            feedback = (
                f"Best design option so far (score {best_combined:.2f}):\n"
                f"{best_hypothesis}\n\n"
                f"Structural feedback from the codebase:\n"
                f"- Symbols NOT yet in index (need creation): "
                f"{list(symbols_found.keys())[:5] or 'none'}\n"
                f"- Invalid call relationships proposed: "
                f"{list(call_invalid.keys())[:3] or 'none'}\n"
                f"- Recently changed symbols: {best_evidence.recent_changes[:3]}\n"
                f"- Objective feasibility score: {best_obj:.2f}\n\n"
                f"Revise or propose {max_hypotheses} improved options. "
                f"Prefer approaches that reuse existing symbols where possible."
            )
            refined = await self._f._llm_orchestrator.call_llm(
                prompt=feedback,
                system_prompt=(
                    "You are a software architect refining design options based on "
                    "structural evidence. Output exactly the same format as before."
                ),
                model_override=self._f.valves.cot_model_level3,
                max_tokens=600,
                temperature=0.3,
                label=f"{label}_refine" if label else "sci_arch_refine",
            )
            if refined:
                new_hyps = _parse_hypotheses(refined)
                if len(new_hyps) >= 2:
                    hypotheses = new_hyps
                else:
                    break
            else:
                break

        # ── Step 4: Synthesise final architecture proposal ────────────────
        evidence_summary = (
            (
                f"Symbols confirmed in index: "
                f"{[k for k, v in best_evidence.symbols_found.items() if v]}\n"
                f"Symbols to create: "
                f"{[k for k, v in best_evidence.symbols_found.items() if not v]}\n"
                f"Valid call relationships: "
                f"{[k for k, v in best_evidence.call_relations_valid.items() if v]}\n"
                f"Recent changes affecting this area: {best_evidence.recent_changes}\n"
            )
            if best_evidence
            else ""
        )

        synthesis_prompt = (
            f"Skeleton:\n{skeleton_context[:2000]}\n\n"
            f"Architecture question:\n{question[:400]}\n\n"
            f"Best design option (score {best_combined:.2f}):\n{best_hypothesis}\n\n"
            f"Structural evidence:\n{evidence_summary}\n\n"
            "Produce a final architecture proposal:\n"
            "1. Concrete signatures of new/modified classes and methods "
            "(as skeleton lines: `def foo(self, x: int) -> str: ...`).\n"
            "2. Which existing bodies need implementation changes "
            "(write `/expand <SymbolName>` for each — DO NOT write the implementation).\n"
            "3. Migration path: what changes in what order.\n"
            "Be specific. No vague descriptions."
        )
        synthesis = await self._f._llm_orchestrator.call_llm(
            prompt=synthesis_prompt,
            system_prompt=(
                "You are a software architect producing a final design proposal. "
                "Output concrete signatures and migration steps. "
                "Use `/expand <Name>` instead of writing full implementations."
            ),
            model_override=self._f.valves.cot_model_level3,
            max_tokens=(
                self._f.valves.skeleton_cot_max_tokens * 2
                if self._f.valves.skeleton_cot_max_tokens > 0
                else 1200
            ),
            temperature=0.25,
            label=f"{label}_synthesize" if label else "sci_arch_synth",
        )

        if not synthesis:
            return await self.generate_architecture_reasoning(
                question, skeleton_context, project_id, label=label
            )

        return (
            f"## 🔬🏗️ Scientific Architecture Reasoning (Level 3)\n"
            f"*{max_hypotheses} design options evaluated against the SymbolGraph. "
            f"Best option score: {best_combined:.2f} "
            f"(feasibility={best_obj:.2f}, design_fit={best_llm_conf:.2f})*\n\n"
            f"{synthesis}"
        )


class MultiPhasePlanner:
    """Generates the multi‑phase protocol instructions injected into the
    system prompt when the response budget is tight, and appends wrap‑up
    hints when the token window is critically low.

    Provides:
    * ``build_multi_phase_instructions(available_tokens, query, ...)`` —
      returns a complete multi‑phase protocol block (analysis → architecture
      → contract → code parts → verification) tailored to whether the task
      is code‑generation or a general long‑form answer.
    * ``append_critical_wrap_up_hint(messages)`` — appends a short (~25
      token) reminder to the last user message telling the model to stop
      cleanly and write a continuation marker, without consuming system
      prompt budget.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Build multi‑phase protocol instructions
    # ═══════════════════════════════════════════════════════════════════════════

    def build_multi_phase_instructions(
        self,
        available_tokens: int,
        user_query: str,
        cot_degraded_to_l1: bool = False,
        is_continuation: bool = False,
    ) -> str:
        """Build the multi‑phase protocol instructions injected into the system prompt."""
        _CODE_SIGNALS = {
            "refactor",
            "refactoriza",
            "implement",
            "implementa",
            "escribe",
            "write",
            "genera",
            "generate",
            "crea",
            "create",
            "código",
            "code",
            "clase",
            "class",
            "función",
            "function",
            "método",
            "method",
            "módulo",
            "module",
            "reescribe",
            "rewrite",
        }
        is_code_task = any(sig in user_query.lower() for sig in _CODE_SIGNALS)

        part_budget = min(
            self._f.valves.multi_phase_effective_max_tokens,
            max(500, available_tokens - 200),
        )

        fase1_suffix = (
            " *(razona paso a paso aquí — análisis de dependencias incluido)*"
            if cot_degraded_to_l1
            else ""
        )

        if is_continuation and is_code_task:
            header = (
                f"## 📋 CONTINUACIÓN MULTI-FASE — {part_budget} tokens por parte\n\n"
                "Fases 1-4 completadas. Continúa con el siguiente bloque del Plan."
            )
            phases = textwrap.dedent(f"""
                    **FASE 5...N — Código Parte K/M** (≤ {part_budget} tokens por parte)
                    Escribe el siguiente bloque del plan. REGLAS CRÍTICAS:
                      · Encabeza la parte: `## Código — Parte K/M: [nombre del bloque]`
                      · NUNCA cortes dentro de una función, clase o método.
                      · Antes de alcanzar el límite, cierra el bloque actual limpiamente
                        y escribe el marcador obligatorio:
                        `# ▶ CONTINÚA: Parte [K+1] — [nombre exacto del siguiente bloque]`
                        `# Pendiente: [lista de lo que falta]`
                      · Una clase puede partirse entre partes; un método, nunca.

                    **FASE FINAL — Verificación** (~150 tokens)
                    Lista los bloques del plan y marca: ✓ escrito | ✗ pendiente.
                """).strip()
            return f"{header}\n\n{phases}"

        if is_code_task:
            header = (
                f"## 📋 PROTOCOLO MULTI-FASE — {part_budget} tokens por parte\n\n"
                "Tu tarea genera más código del que cabe en un mensaje. "
                "Sigue **exactamente** este protocolo:"
            )
            phases = textwrap.dedent(f"""
                    **FASE 1 — Análisis{fase1_suffix}** (~300-400 tokens)
                    Qué existe, qué cambia, dependencias críticas. Sin código todavía.

                    **FASE 2 — Arquitectura** (~400-600 tokens) *(solo si el diseño es complejo)*
                    Decisiones de estructura: clases, inyección de dependencias, patrones.
                    Omite esta fase si el Plan la cubre suficientemente.

                    **FASE 3 — Contrato** (~300-500 tokens)
                    Firmas completas de todas las clases y métodos públicos, sin cuerpo.
                    Compromiso firme: no cambies estas firmas en fases posteriores.

                    **FASE 4 — Plan de Acción** (~300-400 tokens)
                    Lista numerada: bloque | tokens estimados | dependencias previas.
                    Última línea obligatoria:
                    "Total: ~X tokens → N partes de ≤ {part_budget} tokens c/u"

                    **FASE 5...N — Código Parte K/M** (≤ {part_budget} tokens por parte)
                    Escribe los bloques del plan en orden. REGLAS CRÍTICAS:
                      · Encabeza cada parte: `## Código — Parte K/M: [nombre del bloque]`
                      · NUNCA cortes dentro de una función, clase o método.
                      · Antes de alcanzar el límite, cierra el bloque actual limpiamente
                        y escribe el marcador obligatorio:
                        `# ▶ CONTINÚA: Parte [K+1] — [nombre exacto del siguiente bloque]`
                        `# Pendiente: [lista de lo que falta]`
                      · Una clase puede partirse entre partes; un método, nunca.

                    **FASE FINAL — Verificación** (~150 tokens)
                    Lista los bloques del plan y marca: ✓ escrito | ✗ pendiente.
                """).strip()
        else:
            cot_note = (
                "\nRazona paso a paso antes de continuar." if cot_degraded_to_l1 else ""
            )
            header = (
                f"## 📋 RESPUESTA LARGA — {available_tokens} tokens disponibles\n\n"
                "Tu respuesta probablemente excede el espacio en un mensaje. "
                "Divídela en partes lógicas:"
            )
            phases = textwrap.dedent(f"""
                    **Parte 1 — Resumen y plan** (~300 tokens)
                    Enumera los puntos que vas a desarrollar.{cot_note}

                    **Partes 2...N — Desarrollo** (≤ {part_budget} tokens por parte)
                    Al final de cada parte que no sea la última escribe:
                    `▶ CONTINÚA — [título de lo que sigue]`

                    **Parte Final — Conclusión** (~150 tokens)
                    Verifica que cubriste todos los puntos del plan.
                """).strip()

        return f"{header}\n\n{phases}"

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Wrap‑up hint (appended to user message when token window is critical)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def append_critical_wrap_up_hint(messages: list) -> list:
        """
        Append a short (~25-token) wrap-up reminder to the last user message
        when the response token budget is critically low.

        Appended to the user message (not system) so it is not deducted
        from the model's generation budget.
        """
        last_user_idx = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i].get("role") == "user"
            ),
            None,
        )
        if last_user_idx is None:
            return messages

        hint = (
            "\n\n⚠️ Tokens críticos. Cierra el bloque actual sin cortarlo, "
            "escribe el marcador de continuación y para."
        )
        messages[last_user_idx] = {
            **messages[last_user_idx],
            "content": messages[last_user_idx].get("content", "") + hint,
        }
        return messages


class CommandRouter:
    """Dispatches explicit slash‑commands, interprets natural‑language
    intents, and injects proactive suggestions into the conversation.

    Provides:
    * ``handle_explicit_commands(messages, ...)`` — processes ``/forget``,
      ``/status``, ``/clean``, and ``/expand`` commands, returning a
      fully‑formed assistant response when one is triggered.
    * ``handle_natural_intents(messages, ...)`` — detects forget / remember /
      obsolete requests expressed in natural language using the CrossEncoder
      and executes them without an explicit command prefix.
    * ``outlet_intercept_expand(assistant_content, project_id)`` — scans the
      assistant's response for ``/expand`` commands and replaces them inline
      with the actual symbol bodies from the SymbolIndex, so the user sees
      the code immediately.
    * ``classify_intent(query)`` — returns a probability distribution over
      explain / modify / debug / refactor using the CrossEncoder.
    * ``suggest_commands(project_id, state)`` — returns context‑management
      tips (``/forget``, ``/status``, ``/clean``) after the conversation
      reaches a threshold, with a cooldown between suggestions.
    * ``is_code_only_message(content)`` — detects messages that contain only
      code without a question, used by the inlet to trigger silent ingestion.
    """

    # ── Class constants ────────────────────────────────────────────────────

    _EXPAND_DOTTED = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)$")

    # ── Silent-ingestion code-only detection ───────────────────────────────
    # Lines that start a new statement/construct (multi-language, best-effort).
    _STRUCTURAL_LINE_START = re.compile(
        r"^\s*(?:"
        r"def |async def |class |import |from |@\w|"
        r"if |elif |else\b|for |while |try:|except|finally:|with |"
        r"return |yield |raise |pass\b|break\b|continue\b|#|"
        r"global |nonlocal |assert |del |lambda |"
        r"function |const |let |var |export |interface |type |enum |"
        r"func |struct |impl |use |pub |fn |trait |"
        r"package |public |private |protected |static |void |"
        r"#include |namespace |template "
        r")"
    )
    # Lines that continue a multi-line statement (dict/list entries, wrapped
    # calls, closing brackets, typed attributes...) rather than starting a
    # new construct. Without this, normally-formatted code (Field(...) calls,
    # multi-line regex, dataclass-style fields) gets misread as "prose".
    _CONTINUATION_OR_LITERAL = re.compile(
        r"^\s*(?:"
        r"[\)\]\}]|"
        r"[\"'`]|"
        r"[,\.]|"
        r"\*\*?\w|"
        r"-?\d|"
        r"[\w.\[\]]+\s*[:=\(\[{]"
        r")"
    )
    # Noise-stripping patterns: blanked out (newlines preserved) before
    # checking for a real '?' question, so docstrings and regex literals
    # (`(?:...)`, `\?`) never get mistaken for an explicit user question.
    _TRIPLE_QUOTE_RE = re.compile(r'("""|\'\'\')([\s\S]*?)\1')
    _BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
    _STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')
    _LINE_COMMENT_RE = re.compile(r"(#|//).*")

    # ── Intent keywords ──────────────────────────────────────────────────
    INTENT_KEYWORDS = {
        "forget",
        "olvida",
        "olvid",
        "remember",
        "recuerda",
        "pin",
        "fija",
        "guarda",
        "obsolete",
        "obsoleto",
        "deprecated",
        "ya no",
        "remove",
        "elimina",
        "borra",
        "quita",
        "keep",
        "mantén",
        "conserva",
    }

    _FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
    _INDENTED_CODE_RE = re.compile(r"(?m)^(?:    |\t).+$")
    _EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, filter_ref: "Filter") -> None:
        """Store a reference to the parent Filter for shared state."""
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CrossEncoder & ML helpers
    # ═══════════════════════════════════════════════════════════════════════════

    async def _predict_cross_encoder(self, pairs: list) -> Optional[list]:
        """Run the CrossEncoder on (text_a, text_b) pairs.
        Returns raw scores (logits) or None if the model is not loaded.
        """
        if self._f._cross_encoder is None:
            if not self._f._cross_encoder_unavailable_logged:
                self._f._log_debug(
                    "CrossEncoder not loaded – predictions will return None."
                )
                self._f._cross_encoder_unavailable_logged = True
            return None
        async with self._f._cross_encoder_lock:
            return await anyio.to_thread.run_sync(self._f._cross_encoder.predict, pairs)

    @staticmethod
    def _normalize_cross_encoder_score(raw_score: float) -> float:
        """
        Convert a raw CrossEncoder logit to a probability in [0,1] via sigmoid.

        sentence-transformers CrossEncoder returns logits by default.
        This normalization is needed when thresholds are calibrated for [0,1].
        """
        import math

        return 1.0 / (1.0 + math.exp(-raw_score))

    async def _detect_contradictions(self, messages: list) -> Optional[str]:
        """Check if the last user message contradicts recent conversation history.
        Returns a warning string if a contradiction is detected, else None.
        """
        if not self._f.valves.enable_contradiction_detection or len(messages) < 3:
            return None
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user:
            return None
        history = messages[:-1]
        if not history:
            return None
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in history if m.get("content")]
        )
        if not history_text.strip():
            return None

        new_msg = last_user["content"][:500]
        hist = history_text[-2000:]
        pairs = [
            (
                f"History: {hist}\n\nNew message: {new_msg}",
                "The new message contradicts a previous statement or decision.",
            ),
            (
                f"History: {hist}\n\nNew message: {new_msg}",
                "The new message is consistent with the history.",
            ),
        ]
        scores = await self._predict_cross_encoder(pairs)
        if scores is None:
            self._f._log_debug(
                "_detect_contradictions: CrossEncoder not loaded, "
                "skipping contradiction detection."
            )
            return None
        if scores[0] > scores[1]:
            return (
                "⚠️ **Contradiction detected**: The last message appears to contradict something established earlier. "
                "Please review and clarify if needed."
            )
        return None

    def _extract_text_for_classification(self, message: str) -> str:
        """
        Extract only non-code portions of the user message for intent classification.

        CrossEncoders are typically limited to 512 tokens. Passing full messages
        that contain pasted code causes silent truncation that often removes the
        actual question (usually at the end of the message).

        Code blocks are replaced with a '[CODE]' placeholder so the classifier
        can still infer 'there was code here' without ingesting the tokens.

        Args:
            message: The raw user message.

        Returns:
            A cleaned string with code blocks removed/replaced, suitable for
            feeding into the CrossEncoder.
        """
        if not message:
            return ""

        # Replace fenced code blocks (```...```) with a placeholder
        text = self._FENCED_CODE_BLOCK_RE.sub("[CODE]", message)

        # Replace indented code blocks (4 spaces or tab at start of line)
        text = self._INDENTED_CODE_RE.sub("", text)

        # Clean up excessive newlines
        text = self._EXCESS_NEWLINES_RE.sub("\n\n", text).strip()

        if not text or text == "[CODE]":
            # Pure code message: provide a placeholder so the classifier
            # can still decide (likely case B: implement/modify)
            return "[CODE ONLY — no natural language text]"

        return text

    async def classify_intent(self, user_query: str, project_id: str) -> dict:
        """
        Classify the user's intent using the CrossEncoder.

        Q1 fix: Strip code blocks from the input before classification to avoid
        silent truncation that removes the actual question.

        Returns a dict with probabilities for explain, modify, debug, refactor.
        """
        # ── Q1: Extract text without code for the classifier ──
        classifier_input = self._extract_text_for_classification(user_query)

        self._f._log_debug(
            f"classify_intent: input truncated from {len(user_query.split())} words "
            f"to {len(classifier_input.split())} (code stripped)"
        )

        pairs = [
            (classifier_input[:500], "The user wants to understand or explain code."),
            (classifier_input[:500], "The user wants to modify, fix, or create code."),
            (classifier_input[:500], "The user is debugging an error or exception."),
            (classifier_input[:500], "The user wants to refactor or restructure code."),
        ]
        raw = await self._predict_cross_encoder(pairs)
        if raw is None:
            self._f._log_debug(
                "Intent: CrossEncoder not available, using default distribution."
            )
            return {"explain": 0.25, "modify": 0.45, "debug": 0.2, "refactor": 0.1}
        exp_scores = [2.71828**s for s in raw]
        total_exp = sum(exp_scores)
        if total_exp > 0:
            result = {
                "explain": exp_scores[0] / total_exp,
                "modify": exp_scores[1] / total_exp,
                "debug": exp_scores[2] / total_exp,
                "refactor": exp_scores[3] / total_exp,
            }
            self._f._log_debug(
                f"Intent (CrossEncoder): {max(result, key=result.get)}="
                f"{max(result.values()):.2f}"
            )
            return result
        self._f._log_debug("Intent: softmax failed, using default distribution.")
        return {"explain": 0.25, "modify": 0.45, "debug": 0.2, "refactor": 0.1}

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Explicit command dispatch (called from inlet)
    # ═══════════════════════════════════════════════════════════════════════════

    async def handle_explicit_commands(
        self,
        messages: list,
        project_id: str,
        is_explicit_command: bool,
        last_user_msg: Optional[dict],
        __user__: Optional[dict],
    ) -> Tuple[bool, Optional[list]]:
        """Handle /forget, /status, /clean, /expand.
        Returns (handled, messages) if a command was processed, else (False, None).
        """
        if not last_user_msg:
            return False, None

        content = last_user_msg.get("content", "").strip()

        # /forget
        if self._f.valves.enable_forget_command and is_explicit_command:
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                return True, self._f._inlet_orch.ensure_last_message_is_user(
                    new_messages
                )

        # /status
        if (
            content == "/status"
            and self._f.valves.cleanup_status_command_enabled
            and self._f.valves.cleanup_suggestions_enabled
        ):
            candidates = self._f._activation.get_inactive_block_candidates(project_id)
            if not candidates:
                response = "✅ No inactive blocks detected."
            else:
                lines = [
                    f"⚠️ {len(candidates)} inactive block(s) (not mentioned in last "
                    f"{self._f.valves.cleanup_inactive_threshold_messages} messages):"
                ]
                state = self._f._conversation_state_manager.get(project_id)
                for h in candidates:
                    blk = state.active_blocks.get(h)
                    if blk:
                        snippet = blk.content[:80].replace("\n", " ")
                        file_info = f" ({blk.file_path})" if blk.file_path else ""
                        lines.append(f"- `{h[:8]}...`{file_info}: {snippet}...")
                response = "\n".join(lines)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._f._inlet_orch.ensure_last_message_is_user(messages)

        # /clean
        if (
            content.startswith("/clean")
            and self._f.valves.cleanup_command_enabled
            and self._f.valves.cleanup_suggestions_enabled
        ):
            response = await self._handle_clean_command(content, project_id)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._f._inlet_orch.ensure_last_message_is_user(messages)

        # /expand
        if content.startswith("/expand"):
            response = await self._handle_expand_command(content, project_id)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._f._inlet_orch.ensure_last_message_is_user(messages)

        return False, None

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Expand commands (/expand, outlet intercept)
    # ═══════════════════════════════════════════════════════════════════════════

    def _resolve_expand_target(self, token: str, project_id: str):
        """Classify an /expand target.
        Returns ('method', id) | ('class', name) | ('symbol', name)."""
        m = self._EXPAND_DOTTED.match(token)
        if m:
            cls, meth = m.group(1), m.group(2)
            qid = qualify_symbol_name(meth, cls)
            if self._f._symbol_index.find_blocks(qid, project_id):
                return ("method", qid)
            if meth in self._f._symbol_index.get_all_names(project_id):
                return ("method", meth)
        if token in self._f._symbol_index.get_classes(project_id):
            members = self._f._symbol_index.get_class_members(token, project_id)
            if members:
                return ("class", token)
        return ("symbol", token)

    async def _handle_expand_command(self, content: str, project_id: str) -> str:
        """
        Process /expand [depth] <symbol|Class|Class.method>.
        Returns the expanded code or an error message.

        Supports three target types:
        - Class: expands all methods of the class, each extracted individually.
        - Method: expands the method and its callees up to the specified depth.
        - Symbol: falls back to method expansion (legacy bare-name support).

        The class expansion now uses extract_symbol_body to show only the method
        bodies, not the entire file content.
        """
        parts = content.strip().split()
        if len(parts) < 2:
            return "Usage: /expand [depth] <name|Class|Class.method>"

        depth = self._f.valves.expand_default_depth
        token = parts[-1]
        # Only the last numeric argument is the depth, anything else is part of the token
        for i in range(1, len(parts)):
            if parts[i].isdigit():
                depth = int(parts[i])
            else:
                token = " ".join(parts[i:])
                break

        target_type, target_name = self._resolve_expand_target(token, project_id)

        # --- Branch 1: Class expansion ---
        if target_type == "class":
            members = self._f._symbol_index.get_class_members(target_name, project_id)
            if not members:
                return f"Class `{target_name}` has no indexed members."
            state = self._f._conversation_state_manager.get(project_id)
            parts_out = [f"## class `{target_name}` ({len(members)} methods)\n"]
            seen_pairs: Set[Tuple[str, str]] = set()  # (block_hash, method_qid)
            for mname in members:
                for bh in self._f._symbol_index.find_blocks(mname, project_id):
                    pair = (bh, mname)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    block = state.active_blocks.get(bh)
                    if block and not block.obsolete:
                        lang = block.symbols[0].language if block.symbols else ""
                        body = CodeBlockManager.extract_symbol_body(block, mname)
                        parts_out.append(f"### `{mname}`\n```{lang}\n{body}\n```\n")
            if len(parts_out) == 1:
                return f"Class `{target_name}` found but no code blocks available."
            return "\n".join(parts_out)

        # --- Branch 2: Method expansion (qualified id) ---
        elif target_type == "method":
            expanded = await self._expand_symbol_dependencies(
                target_name, depth, project_id
            )
            if expanded:
                return f"[Retrieved `{target_name}`]\n{expanded}"
            return f"Symbol `{target_name}` not found or has no code."

        # --- Branch 3: Bare symbol expansion (fallback) ---
        else:
            expanded = await self._expand_symbol_dependencies(
                target_name, depth, project_id
            )
            if expanded:
                return f"[Retrieved `{target_name}`]\n{expanded}"
            return f"Symbol `{target_name}` not found or has no code."

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Expand command dependencies (recursive symbol expansion)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _expand_symbol_dependencies(
        self, name: str, max_depth: int, project_id: str
    ) -> str:
        """
        Recursively expand a symbol and its callees up to max_depth.

        For each symbol found, this method extracts only the symbol's body
        (not the whole file) using CodeBlockManager.extract_symbol_body,
        ensuring that `/expand` returns the exact implementation of the
        requested function or method.

        Args:
            name: The symbol name (qualified id or bare name) to expand.
            max_depth: Maximum recursion depth for following callees.
            project_id: The current project identifier.

        Returns:
            A formatted string containing the expanded code blocks for the
            symbol and its callees, or an empty string if no blocks are found.
        """
        state = self._f._conversation_state_manager.get(project_id)
        if not state:
            return ""
        visited = set()
        lines = []

        async def recurse(current_name, current_depth):
            if current_depth > max_depth or current_name in visited:
                return
            visited.add(current_name)
            blocks = self._f._symbol_index.find_blocks(current_name, project_id)
            for h in blocks:
                block = state.active_blocks.get(h)
                if block and not block.obsolete:
                    loc = f" (file: {block.file_path})" if block.file_path else ""
                    # ── FIX 2c: Extract symbol body instead of whole block ──
                    body = CodeBlockManager.extract_symbol_body(block, current_name)
                    lines.append(
                        f"### `{current_name}` (depth {current_depth}){loc}\n"
                        f"```\n{body}\n```"
                    )
                    # Find the actual symbol in the block to follow its callees
                    for sym in block.symbols:
                        if (
                            sym.name == current_name
                            or qualify_symbol_name(sym.name, sym.parent_symbol)
                            == current_name
                        ):
                            for callee in sym.calls:
                                await recurse(callee, current_depth + 1)
                            break
                    break

        await recurse(name, 1)
        return "\n".join(lines)

    async def outlet_intercept_expand(
        self,
        assistant_content: str,
        project_id: str,
    ) -> Tuple[str, bool]:
        """
        Intercept /expand commands in the assistant's response and replace them
        with the actual expanded symbol code from the SymbolIndex.

        Supports:
        - `/expand Class` → expands all methods of the class, each with its own body.
        - `/expand Class.method` → expands the method and its callees up to depth.
        - `/expand symbol` (legacy) → expands the symbol as a method.

        When expanding a class, this method uses CodeBlockManager.extract_symbol_body
        to show only the method bodies, not the entire file content.

        Args:
            assistant_content: The raw assistant response text.
            project_id: Current project identifier.

        Returns:
            A tuple (modified_content, did_expand) indicating whether any
            expansion was performed.
        """
        if not self._f.valves.outlet_expand_intercept_enabled:
            return assistant_content, False

        EXPAND_RE = re.compile(r"/expand\s+(?:(\d+)\s+)?(\S+)", re.IGNORECASE)
        matches = list(EXPAND_RE.finditer(assistant_content))
        if not matches:
            return assistant_content, False

        replaced_content = assistant_content
        did_any = False
        state = self._f._conversation_state_manager.get(project_id)

        max_syms = self._f.valves.outlet_expand_intercept_max_symbols
        matches_to_process = matches if max_syms == 0 else matches[:max_syms]

        for match in matches_to_process:
            depth_str = match.group(1)
            token = match.group(2)
            depth = (
                int(depth_str)
                if depth_str
                else self._f.valves.outlet_expand_intercept_depth
            )
            if depth == 0:
                depth = 9999

            target_type, target_name = self._resolve_expand_target(token, project_id)

            # --- Class expansion: aggregate all methods ---
            if target_type == "class":
                members = self._f._symbol_index.get_class_members(
                    target_name, project_id
                )
                if not members:
                    continue
                seen_pairs: Set[Tuple[str, str]] = set()  # (block_hash, method_qid)
                buf = [f"## class `{target_name}` ({len(members)} methods)\n"]
                for mname in members:
                    for bh in self._f._symbol_index.find_blocks(mname, project_id):
                        pair = (bh, mname)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        block = state.active_blocks.get(bh)
                        if block and not block.obsolete:
                            lang = block.symbols[0].language if block.symbols else ""
                            body = CodeBlockManager.extract_symbol_body(block, mname)
                            buf.append(f"### `{mname}`\n```{lang}\n{body}\n```\n")
                if len(buf) > 1:
                    replacement = "\n".join(buf)
                else:
                    continue

            # --- Method or symbol expansion: follow callees ---
            elif target_type in ("method", "symbol"):
                expanded = await self._expand_symbol_dependencies(
                    target_name, depth, project_id
                )
                if not expanded:
                    continue
                replacement = f"[Retrieved `{target_name}`]\n{expanded}"

            # --- Apply replacement ---
            did_any = True
            replaced_content = replaced_content.replace(match.group(0), replacement, 1)

            # --- Pin the block if it exists (only for method/symbol) ---
            if target_type != "class":
                lock = await self._f._state_store.get_project_lock(project_id)
                async with lock:
                    block_hashes = self._f._symbol_index.find_blocks(
                        target_name, project_id
                    )
                    for h in block_hashes:
                        block = state.active_blocks.get(h)
                        if block and not block.obsolete:
                            block.is_raw = True
                            block.pinned = True
                            block.importance_score = 10.0
                            block.last_mentioned = time.time()
                            block.last_mentioned_msg_idx = state.message_count
                            break
                    self._f._activation.invalidate_lightweight_cache(project_id)
                    self._f._conversation_state_manager.set(project_id, state)

        return replaced_content, did_any

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Forget / remember / obsolete commands (explicit and natural)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _handle_forget_command(
        self, messages: List[dict], project_id: str, __user__: Optional[dict]
    ) -> Tuple[List[dict], bool]:
        """Handle /forget [all|last|<file_or_hash>].
        Returns (messages, was_handled).
        """
        if not (
            self._f.valves.enable_forget_command
            or self._f.valves.enable_natural_language_forget
        ):
            return messages, False
        if not messages:
            return messages, False
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return messages, False
        content = last_msg.get("content", "").strip()
        if self._f.valves.enable_forget_command and content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            state = self._f._conversation_state_manager.get(project_id)
            if not state:
                return messages, False
            if target == "all":
                for block in state.active_blocks.values():
                    self._f._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                state.active_blocks.clear()
                state.recent_changes.clear()
                state.committed_changes.clear()
                state.has_any_calls = False
                self._f._activation.invalidate_lightweight_cache(project_id)
                confirmation = "Forgotten all context."
            elif target == "last":
                if state.active_blocks:
                    last_hash = max(
                        state.active_blocks.keys(),
                        key=lambda h: state.active_blocks[h].timestamp,
                    )
                    block = state.active_blocks.get(last_hash)
                    if block:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state.active_blocks[last_hash]
                    self._f._activation.invalidate_lightweight_cache(project_id)
                    confirmation = "Forgotten the last context block."
                else:
                    confirmation = "No blocks to forget."
            else:
                to_remove = [
                    h
                    for h, blk in state.active_blocks.items()
                    if (blk.file_path and target in blk.file_path) or target in h
                ]
                for h in to_remove:
                    block = state.active_blocks.get(h)
                    if block:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state.active_blocks[h]
                self._f._activation.invalidate_lightweight_cache(project_id)
                confirmation = (
                    f"Forgotten {len(to_remove)} block(s) matching '{target}'."
                )
            self._f._conversation_state_manager.set(project_id, state)
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return messages, True
        return messages, False

    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
        """Execute a natural-language forget intent. Returns a user message."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            if not state:
                return "No active context to forget."

            action = intent.get("action")
            if action == "forget_all":
                return (
                    "⚠️ For safety, the natural language 'forget all' is disabled. "
                    "Please type `/forget all` explicitly to confirm."
                )

            if action == "forget_last":
                if state.active_blocks:
                    last_hash = max(
                        state.active_blocks.keys(),
                        key=lambda h: state.active_blocks[h].timestamp,
                    )
                    block = state.active_blocks.get(last_hash)
                    if block:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state.active_blocks[last_hash]
                    self._f._activation.invalidate_lightweight_cache(project_id)
                return "Forgotten the last context block."

            elif action == "forget_n":
                n = intent.get("n", 1)
                blocks_by_time = sorted(
                    state.active_blocks.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True,
                )
                removed = 0
                for h, block in blocks_by_time[:n]:
                    if h in state.active_blocks:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                        del state.active_blocks[h]
                        removed += 1
                if removed:
                    self._f._activation.invalidate_lightweight_cache(project_id)
                return f"Forgotten the last {removed} context block(s)."

            elif action == "forget_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_remove = [
                    h
                    for h, blk in state.active_blocks.items()
                    if blk.file_path and file_path in blk.file_path
                ]
                for h in to_remove:
                    block = state.active_blocks.get(h)
                    if block:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state.active_blocks[h]
                if to_remove:
                    self._f._activation.invalidate_lightweight_cache(project_id)
                return f"Forgotten {len(to_remove)} block(s) related to {file_path}."

            elif action == "forget_block":
                block_id = intent.get("hash") or intent.get("id") or ""
                if not block_id:
                    return "No block specified."
                if block_id in state.active_blocks:
                    block = state.active_blocks[block_id]
                    self._f._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                    del state.active_blocks[block_id]
                    self._f._activation.invalidate_lightweight_cache(project_id)
                    return f"Forgotten block {block_id}."
                matches = [h for h in state.active_blocks if block_id in h]
                if matches:
                    for h in matches:
                        block = state.active_blocks.get(h)
                        if block:
                            self._f._symbol_index.remove_all_for_block(
                                block.hash, block.symbols, project_id
                            )
                        del state.active_blocks[h]
                    self._f._activation.invalidate_lightweight_cache(project_id)
                    return f"Forgotten {len(matches)} block(s) matching {block_id}."
                return f"No block found for {block_id}."

            else:
                return "Unrecognized forget action."

    async def _execute_remember_intent(self, project_id: str, intent: Dict) -> str:
        """Execute a natural-language remember/pin intent."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            if not state:
                return "No active context to pin."

            def set_pinned(blocks, pinned_value):
                count = 0
                for blk in blocks:
                    blk.pinned = pinned_value
                    if pinned_value:
                        blk.importance_score = 10.0
                    else:
                        blk._update_importance()
                    count += 1
                return count

            action = intent.get("action", "")
            blocks = list(state.active_blocks.values())
            if not blocks:
                return "No blocks available."

            if action == "pin_last":
                last_block = max(blocks, key=lambda b: b.timestamp)
                set_pinned([last_block], True)
                return "Pinned last code block."
            elif action == "pin_n":
                n = intent.get("n", 1)
                blocks_by_time = sorted(blocks, key=lambda b: b.timestamp, reverse=True)
                to_pin = blocks_by_time[:n]
                count = set_pinned(to_pin, True)
                return f"Pinned {count} block(s)."
            elif action == "pin_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_pin = [
                    blk
                    for blk in blocks
                    if blk.file_path and file_path in blk.file_path
                ]
                count = set_pinned(to_pin, True)
                return f"Pinned {count} block(s) related to {file_path}."
            elif action == "pin_block":
                desc = intent.get("description", "") or intent.get("hash", "")
                if not desc:
                    return "No block identifier."
                matches = [
                    blk
                    for blk in blocks
                    if desc in blk.content
                    or (blk.hash and desc in blk.hash)
                    or (blk.file_path and desc in blk.file_path)
                ]
                count = set_pinned(matches, True)
                return f"Pinned {count} block(s) matching '{desc}'."
            elif action == "pin_all":
                count = set_pinned(blocks, True)
                return f"Pinned all {count} active blocks."
            elif action == "unpin_last":
                last_block = max(blocks, key=lambda b: b.timestamp)
                set_pinned([last_block], False)
                return "Unpinned last block."
            elif action == "unpin_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_unpin = [
                    blk
                    for blk in blocks
                    if blk.file_path and file_path in blk.file_path
                ]
                count = set_pinned(to_unpin, False)
                return f"Unpinned {count} block(s) related to {file_path}."
            elif action == "unpin_all":
                count = set_pinned(blocks, False)
                return f"Unpinned all {count} blocks."
            else:
                return "Unrecognized pin action."

    async def _execute_obsolete_intent(self, project_id: str, intent: Dict) -> str:
        """Execute a natural-language obsolete/revive intent."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            if not state:
                return "No active context to mark as obsolete."

            action = intent.get("action", "")
            if action == "obsolete_all":
                return (
                    "⚠️ For safety, the natural language 'obsolete all' is disabled. "
                    "Please type `/obsolete all` explicitly to confirm."
                )

            blocks = list(state.active_blocks.values())
            if not blocks:
                return "No blocks available."

            def set_obsolete(blks, val):
                for b in blks:
                    b.obsolete = val
                    b._update_importance()
                return len(blks)

            if action == "obsolete_last":
                last_block = max(blocks, key=lambda b: b.timestamp)
                set_obsolete([last_block], True)
                return "Marked last code block as obsolete."

            elif action == "obsolete_n":
                n = intent.get("n", 1)
                blocks_by_time = sorted(blocks, key=lambda b: b.timestamp, reverse=True)
                to_obsolete = blocks_by_time[:n]
                count = set_obsolete(to_obsolete, True)
                return f"Marked {count} block(s) as obsolete."

            elif action == "obsolete_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_obsolete = [
                    blk
                    for blk in blocks
                    if blk.file_path and file_path in blk.file_path
                ]
                count = set_obsolete(to_obsolete, True)
                return f"Marked {count} block(s) related to {file_path} as obsolete."

            elif action == "obsolete_block":
                desc = intent.get("description", "") or intent.get("hash", "")
                if not desc:
                    return "No block identifier."
                matches = [
                    blk
                    for blk in blocks
                    if desc in blk.content
                    or (blk.hash and desc in blk.hash)
                    or (blk.file_path and desc in blk.file_path)
                ]
                count = set_obsolete(matches, True)
                return f"Marked {count} block(s) matching '{desc}' as obsolete."

            elif action == "revive_last":
                last_block = max(blocks, key=lambda b: b.timestamp)
                set_obsolete([last_block], False)
                return "Removed obsolete mark from last block."

            elif action == "revive_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_revive = [
                    blk
                    for blk in blocks
                    if blk.file_path and file_path in blk.file_path
                ]
                count = set_obsolete(to_revive, False)
                return f"Removed obsolete mark from {count} block(s) related to {file_path}."

            elif action == "revive_all":
                count = set_obsolete(blocks, False)
                return f"Removed obsolete mark from all {count} block(s)."

            else:
                return "Unrecognized obsolete action."

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Natural language intents (forget, remember, obsolete)
    # ═══════════════════════════════════════════════════════════════════════════

    async def handle_natural_intents(
        self,
        messages: list,
        project_id: str,
        is_explicit_command: bool,
        last_user_msg: Optional[dict],
        slot_free: bool = True,
    ) -> Tuple[bool, Optional[list]]:
        """Handle natural language intents (forget, remember, obsolete).
        Returns (handled, messages) if an intent was processed, else (False, None).
        """
        if (
            not self._f.valves.enable_natural_language_forget
            or not last_user_msg
            or is_explicit_command
            or self.has_code_indicators(last_user_msg.get("content", ""))
        ):
            return False, None

        if not slot_free:
            self._f._log_debug(
                "⚡ COMMAND HANDLING – Natural intents skipped (no free slot)"
            )
            return False, None

        intents = await self._parse_all_intents(last_user_msg.get("content", ""))
        for intent_type in ("forget", "remember", "obsolete"):
            fi = intents.get(intent_type, {})
            if fi.get("action") in (None, "none"):
                continue
            if intent_type == "forget":
                confirmation = await self._execute_forget_intent(project_id, fi)
            elif intent_type == "remember":
                confirmation = await self._execute_remember_intent(project_id, fi)
            elif intent_type == "obsolete" and self._f.valves.enable_obsolete_marking:
                confirmation = await self._execute_obsolete_intent(project_id, fi)
            else:
                continue

            status_msg = f"[CodeAware] {confirmation}"
            messages.insert(0, {"role": "system", "content": status_msg})
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return True, self._f._inlet_orch.ensure_last_message_is_user(messages)

        return False, None

    async def _parse_all_intents(self, user_message: str) -> Dict[str, Any]:
        """
        Detect natural-language intents (forget, remember, obsolete) using the
        fast CrossEncoder instead of the main LLM.

        The CrossEncoder scores candidate action descriptions against the user's
        prose (code stripped).  The highest-scoring action above a confidence
        threshold wins; ties or low scores return "none".
        """
        if not self._f.valves.enable_natural_language_forget:
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

        # Strip code spans — only the user's own words matter.
        try:
            code_spans = await self._f._code_blocks.get_code_spans(user_message)
        except Exception:
            code_spans = []
        prose = (
            CodeBlockManager.remove_code_spans(user_message, code_spans).strip()
            if code_spans
            else user_message
        )
        if not prose or len(prose) < 3:
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

        # ── Action templates (multilingual — the CrossEncoder handles them) ──
        candidates: List[Tuple[str, str, dict]] = [
            # FORGET
            ("forget", "forget last", {"action": "forget_last"}),
            ("forget", "forget all", {"action": "forget_all"}),
            ("forget", "forget this", {"action": "forget_last"}),
            # REMEMBER / PIN
            ("remember", "pin last", {"action": "pin_last"}),
            ("remember", "pin this", {"action": "pin_last"}),
            ("remember", "unpin last", {"action": "unpin_last"}),
            ("remember", "unpin all", {"action": "unpin_all"}),
            # OBSOLETE
            ("obsolete", "mark last as obsolete", {"action": "obsolete_last"}),
            ("obsolete", "mark all as obsolete", {"action": "obsolete_all"}),
            ("obsolete", "revive last", {"action": "revive_last"}),
            ("obsolete", "revive all", {"action": "revive_all"}),
            # NONE
            ("none", "no action needed", {"action": "none"}),
        ]

        # Build pairs: (prose, candidate_description)
        pairs = [(prose, desc) for _, desc, _ in candidates]
        scores = await self._predict_cross_encoder(pairs)

        none = {"action": "none"}
        if scores is None:
            return {"forget": none, "remember": none, "obsolete": none}

        # Select the highest-scoring candidate per category
        best: Dict[str, Tuple[float, dict]] = {}
        for (category, _, action), score in zip(candidates, scores):
            if score > best.get(category, (-1.0, none))[0]:
                best[category] = (score, action)

        result: Dict[str, dict] = {
            "forget": none,
            "remember": none,
            "obsolete": none,
        }

        # Confidence threshold: scores below 0.6 are treated as "none"
        THRESHOLD = 0.6
        for category, (score, action) in best.items():
            if score >= THRESHOLD and action["action"] != "none":
                result[category] = action

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Proactive suggestions
    # ═══════════════════════════════════════════════════════════════════════════

    async def suggest_commands(self, project_id: str, state: dict) -> Optional[str]:
        """Suggest context management commands to the user after enough messages."""
        if not self._f.valves.enable_command_suggestions:
            return None
        now = time.time()
        last_sugg = state.last_suggestion_timestamp
        if now - last_sugg < self._f.valves.command_suggestion_cooldown_minutes * 60:
            return None
        if state.message_count > 15 and not state.has_any_calls:
            state.last_suggestion_timestamp = now
            return (
                "[CodeAware] Tip: You can manage context with commands like "
                "`/forget`, `/remember`, `/status`, `/clean`. Use `/help` for more info."
            )
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Utilities & helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_inactive_block_candidates(self, project_id: str) -> List[str]:
        """Return hashes of blocks that haven't been mentioned recently."""
        state = self._f._conversation_state_manager.get(project_id)
        if not state or not state.active_blocks:
            return []
        threshold = self._f.valves.cleanup_inactive_threshold_messages
        excluded_types = set(self._f.valves.cleanup_excluded_content_types)
        current_msg_idx = state.message_count
        candidates = []
        for h, block in state.active_blocks.items():
            if block.pinned or block.obsolete:
                continue
            if block.content_type.value in excluded_types:
                continue
            last_idx = block.last_mentioned_msg_idx
            if last_idx is None:
                last_idx = current_msg_idx
            if current_msg_idx - last_idx > threshold:
                candidates.append(h)
        return candidates

    async def _handle_clean_command(self, command_text: str, project_id: str) -> str:
        """Handle /clean [all|<hash>]. Lists or removes inactive blocks."""
        if (
            not self._f.valves.cleanup_suggestions_enabled
            or not self._f.valves.cleanup_command_enabled
        ):
            return "Cleanup is disabled."
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            candidates = self._get_inactive_block_candidates(project_id)
            parts = command_text.split(maxsplit=1)
            subcommand = parts[1].strip() if len(parts) > 1 else ""
            if not subcommand:
                if not candidates:
                    return "✅ No inactive blocks to clean."
                lines = [
                    f"⚠️ {len(candidates)} inactive block(s) (not mentioned in last {self._f.valves.cleanup_inactive_threshold_messages} messages):"
                ]
                for h in candidates:
                    blk = state.active_blocks.get(h)
                    if blk:
                        snippet = blk.content[:80].replace("\n", " ")
                        file_info = f" ({blk.file_path})" if blk.file_path else ""
                        lines.append(f"- `{h[:8]}...`{file_info}: {snippet}...")
                lines.append(
                    "\nUse `/clean all` to remove all, or `/clean <hash>` for a specific block."
                )
                return "\n".join(lines)
            if subcommand.lower() == "all":
                if not candidates:
                    return "✅ No inactive blocks to clean."
                for h in candidates:
                    block = state.active_blocks.pop(h, None)
                    if block:
                        self._f._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                state.recent_changes = [
                    c for c in state.recent_changes if c.hash not in candidates
                ]
                state.committed_changes = [
                    c for c in state.committed_changes if c.hash not in candidates
                ]
                self._f._activation.invalidate_lightweight_cache(project_id)
                self._f._conversation_state_manager.set(project_id, state)
                return f"✅ Cleaned {len(candidates)} inactive block(s)."
            target_hash = subcommand.strip()
            if target_hash in candidates:
                block = state.active_blocks.pop(target_hash, None)
                if block:
                    self._f._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                self._f._activation.invalidate_lightweight_cache(project_id)
                self._f._conversation_state_manager.set(project_id, state)
                return f"✅ Cleaned block `{target_hash[:8]}...`."
            else:
                matched = [h for h in state.active_blocks if target_hash in h]
                for h in matched:
                    if h in candidates:
                        block = state.active_blocks.pop(h, None)
                        if block:
                            self._f._symbol_index.remove_all_for_block(
                                block.hash, block.symbols, project_id
                            )
                        self._f._activation.invalidate_lightweight_cache(project_id)
                        self._f._conversation_state_manager.set(project_id, state)
                        return f"✅ Cleaned block `{h[:8]}...` (matched partial hash)."
                return "❌ Block not found among inactive candidates. Use `/status` to see candidates."

    async def is_code_only_message(self, content: str) -> bool:
        """
        Detect if a message contains only code without a question.

        Resolution order:
          1. Whole-message ast.parse: if it succeeds, this is unambiguously
             a complete, syntactically valid Python module — silent
             ingestion fires regardless of size, regardless of any '?' that
             happens to live inside a docstring, comment, or regex literal
             (ast.parse only succeeds on real code, so this can't misfire
             the way the old line-based '?' check did).
          2. Large pastes that aren't valid standalone Python (other
             languages, or an incomplete snippet): structural-line
             heuristic, with comments/strings blanked out first so a
             leftover '?' reflects genuine prose, not regex/docstring noise.
          3. Smaller pastes / fenced messages: tree-sitter-based code-span
             extraction (unchanged).
        """
        if not content or len(content.strip()) < 20:
            return False

        stripped = content.strip()

        # ── Step 1: unambiguous case — valid standalone Python ───────────
        # Avoid blocking event loop on giant paste.
        # if limit is surpassed, it falls to step 2 hueristic
        if len(stripped.encode()) <= SignatureExtractor.MAX_PARSE_SIZE_BYTES:
            try:
                ast.parse(stripped)
                self._f._log_debug(
                    "is_code_only_message: Step1 (ast.parse) succeeded → code-only"
                )
                return True
            except Exception:
                # Fall through to Step 2
                self._f._log_debug(
                    "is_code_only_message: Step1 (ast.parse) failed, falling back to Step2 heuristic"
                )
                pass
        else:
            # Fall through to Step 2
            self._f._log_debug(
                f"is_code_only_message: content size ({len(stripped.encode())} bytes) exceeds MAX_PARSE_SIZE ({SignatureExtractor.MAX_PARSE_SIZE_BYTES}), skipping ast.parse, falling back to Step2 heuristic"
            )

        estimated_tokens = self._f._tokens.estimate_code_tokens(content)

        # ── Step 2: large paste, not valid standalone Python ─────────────
        if estimated_tokens >= self._f.valves.lean_user_code_min_tokens:
            raw_lines = stripped.splitlines()
            cleaned_lines = self._strip_code_noise(stripped).splitlines()
            non_blank_idx = [i for i, l in enumerate(raw_lines) if l.strip()]
            total_lines = len(non_blank_idx)
            if total_lines == 0:
                return False

            structural_lines = 0
            prose_candidates: List[str] = []
            for i in non_blank_idx:
                raw_line = raw_lines[i]
                if self._STRUCTURAL_LINE_START.match(
                    raw_line
                ) or self._CONTINUATION_OR_LITERAL.match(raw_line):
                    structural_lines += 1
                    continue
                cleaned_line = (
                    cleaned_lines[i].strip() if i < len(cleaned_lines) else ""
                )
                if not cleaned_line:
                    # Entire line was a string/comment/docstring body — code.
                    structural_lines += 1
                    continue
                prose_candidates.append(cleaned_line)

            structural_ratio = structural_lines / total_lines
            if structural_ratio > 0.70:
                return True

            prose_text = " ".join(prose_candidates).strip()
            if not prose_text or len(prose_text) < 30:
                return True

            if "?" in prose_text:
                self._f._log_debug(
                    "_is_code_only_message: explicit question detected → not silent"
                )
                return False

            return True

        # ── Step 3: smaller pastes / fenced messages ──────────────────────
        code_blocks, _ = await self._f._code_blocks.extract_code_blocks(content)
        if not code_blocks:
            return False
        spans = await self._f._code_blocks.get_code_spans(content)
        if not spans:
            text_outside = re.sub(r"```[\s\S]*?```", "", content).strip()
        else:
            text_outside = CodeBlockManager.remove_code_spans(content, spans).strip()
        return len(text_outside) < 30

    @staticmethod
    def has_code_indicators(content: str) -> bool:
        """True if the content looks like it contains source code."""
        if "```" in content:
            return True
        if re.search(
            r"\b(def |class |import |from |function |const |let |var |"
            r"#include |package |fn |func )",
            content,
        ):
            return True
        if re.search(
            r"\b[\w\-/]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp)\b", content
        ):
            return True
        return False

    @classmethod
    def _strip_code_noise(cls, text: str) -> str:
        """
        Blank out triple-quoted strings, block/line comments, and quoted
        string literals while preserving every line break, so line counts
        stay aligned with the original text and any leftover '?' reflects
        real prose — never regex syntax, docstring text, or string content.
        """

        def _blank(m: "re.Match") -> str:
            return "\n".join(" " * len(part) for part in m.group(0).split("\n"))

        text = cls._TRIPLE_QUOTE_RE.sub(_blank, text)
        text = cls._BLOCK_COMMENT_RE.sub(_blank, text)
        text = cls._STRING_RE.sub(_blank, text)
        text = cls._LINE_COMMENT_RE.sub(_blank, text)
        return text


class CodeBlockManager:
    """
    Extract, classify, and manage code blocks throughout the conversation lifecycle.

    This class handles the full pipeline for code blocks:
    - Extraction from user/assistant messages (fenced or indented)
    - Language detection via tree-sitter with heuristics
    - Classification by content type (base code, diffs, errors, etc.)
    - Deduplication using AST similarity (Python) or fuzzy matching
    - Symbol body extraction via line-range slicing
    - Unified diff application onto existing blocks
    - Data-flow edge extraction for the call graph

    All methods are designed to be deterministic and avoid LLM calls unless
    explicitly noted. Caching is used where parsing is expensive.

    Attributes:
        _code_spans_cache: Cache of tree-sitter byte spans by content hash.
        _ast_signature_cache: Cache of AST signatures (dump + node-type Counter)
            by content hash for deduplication.
        _f: Reference to the parent Filter, for valves and shared state.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the CodeBlockManager with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        # Memoizes (stripped-AST dump, node-type Counter) by content hash
        self._ast_signature_cache: "OrderedDict[str, Tuple[str, Counter]]" = (
            OrderedDict()
        )
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Block extraction
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_code_spans(self, content: str) -> List[Tuple[int, int]]:
        """
        Return tree-sitter code spans for the given content, with caching.

        Uses tree-sitter to identify code regions in the text. Results are cached
        by MD5 hash of the content to avoid re-parsing the same text repeatedly.

        Args:
            content: The raw text to scan for code spans.

        Returns:
            A list of (start_byte, end_byte) tuples identifying code regions.
            Returns an empty list if tree-sitter is unavailable or parsing fails.
        """
        # ── 1. Early exit: tree-sitter unavailable ─────────────────────────
        if not HAS_TREE_SITTER:
            return []

        # ── 2. Check cache ───────────────────────────────────────────────────
        cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
        if cache_key in self._code_spans_cache:
            return self._code_spans_cache[cache_key]

        # ── 3. Parse with tree-sitter ──────────────────────────────────────
        try:
            config = ProcessConfig()
            blocks = process(content, config)
            spans = [(b.start_byte, b.end_byte) for b in blocks]
        except Exception:
            spans = []

        # ── 4. Store in cache with LRU eviction ─────────────────────────────
        if len(self._code_spans_cache) >= 200:
            keys_to_evict = list(self._code_spans_cache.keys())[:50]
            for key in keys_to_evict:
                del self._code_spans_cache[key]
        self._code_spans_cache[cache_key] = spans

        return spans

    @staticmethod
    def remove_code_spans(content: str, spans: List[Tuple[int, int]]) -> str:
        """
        Replace code regions with spaces, preserving line lengths.

        This blanking approach ensures that the surrounding text keeps its
        character positions, which is useful for line-number sensitive operations.

        Args:
            content: The original text containing code.
            spans: A list of (start_byte, end_byte) tuples identifying code regions.

        Returns:
            The text with code regions replaced by spaces.
        """
        chars = list(content)
        for start, end in spans:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        return "".join(chars)

    async def _extract_full_document_symbols(
        self,
        content: str,
        file_path: Optional[str],
        project_id: Optional[str] = None,
    ) -> Tuple[List["CodeSymbol"], str]:
        """
        Parse the entire document once to preserve nested class context.

        This is the key method that enables parent_symbol resolution. Unlike
        chunked parsing, this parses the whole document so that methods know
        which class they belong to, even when later split into separate blocks.

        Args:
            content (str): The full document source code.
            file_path (Optional[str]): Optional file path for language detection.
            project_id (Optional[str]): The project ID for per-project state.
                                        If None, uses self._f.valves.project_id.

        Returns:
            Tuple[List[CodeSymbol], str]: A tuple of (symbols_list, detected_language).
                                          symbols_list may be empty if language
                                          detection fails or tree-sitter is unavailable.
        """
        # --- 1. Resolve project state ---
        if project_id is None:
            project_id = self._f.valves.project_id
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # --- 2. Detect language ---
        lang = pstate.get("ingested_lang") or SignatureExtractor._guess_language(
            file_path, content
        )
        if lang == "unknown":
            return [], lang

        # --- 3. Extract symbols ---
        symbols = await SignatureExtractor.extract_async(
            content, file_path, language=lang
        )

        # --- 4. Enrich with parent symbol info ---
        if symbols:
            symbols = SignatureExtractor.enrich_symbols_with_parent_info(
                symbols, content
            )

        return symbols, lang

    def _assign_symbols_to_span(
        self,
        full_doc_symbols: List["CodeSymbol"],
        chunk_start_line: int,
        chunk_end_line: int,
    ) -> List["CodeSymbol"]:
        """
        Filter symbols to those whose definition falls within a line range.

        Used when splitting a document into chunks: we pre-parse the whole
        document (so class context is preserved), then assign each symbol to
        the chunk that contains its definition line.

        Args:
            full_doc_symbols: Symbols extracted from the full document.
            chunk_start_line: Start line of the chunk (1-indexed).
            chunk_end_line: End line of the chunk (1-indexed).

        Returns:
            A subset of symbols whose line_start falls within the chunk range.
        """
        # --- 1. Filter symbols by line range ---
        # Since both symbol line numbers and chunk boundaries are in document
        # coordinates, direct comparison is sufficient. No offset arithmetic
        # is needed.
        return [
            s
            for s in full_doc_symbols
            if s.line_start is not None
            and chunk_start_line <= s.line_start <= chunk_end_line
        ]

    async def extract_code_blocks(
        self, content: str, project_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        """
        Extract fenced and indented code blocks from message content.

        This method handles three extraction paths in order of preference:
        1. Pre-extracted symbols (silent ingestion path) — returns a single block.
        2. Tree-sitter processing (recommended) — uses AST for accurate symbol extraction.
        3. Regex fallback — handles fenced blocks and indented code.

        Args:
            content (str): The raw message content.
            project_id (Optional[str]): The project ID for per-project state.
                                        If None, uses self._f.valves.project_id.

        Returns:
            Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
                A tuple of (blocks_list, spans_list) where each block is a dict with:
                    - language: Detected programming language.
                    - code: The source code string.
                    - type: "fenced" or "indented".
                    - precomputed_symbols: Optional list of pre-extracted symbols.
                    - file_path: Optional file path associated with the block.
        """
        # --- 0. Resolve project ID and state ---
        if project_id is None:
            project_id = self._f.valves.project_id
        pstate = self._f._project_state_manager.get_pstate(project_id)

        blocks = []
        spans = []
        if not self._f.valves.auto_detect_code_blocks:
            return blocks, spans

        # ── Path 1: Pre-extracted symbols (silent ingestion) ──────────────
        raw = pstate.get("raw_ingested_symbols")
        if raw is not None:
            # Consume the symbols so they are not used again in the same turn.
            pstate["raw_ingested_symbols"] = None
            lang = pstate.get("ingested_lang") or "python"

            block = {
                "language": lang,
                "code": content,
                "type": "indented",
                "precomputed_symbols": raw,
            }

            # ── 1a. Extract file path ──────────────────────────────────────
            blk_file = None
            if self._f.valves.track_file_paths:
                blk_file = self.extract_file_paths(content)
                blk_file = blk_file[0] if blk_file else None

            # ── 1b. Filter internal code ──────────────────────────────────
            if self._f.valves.exclude_filter_internals and blk_file:
                if (
                    "/app/backend/data/functions/" in blk_file
                    or "open-webui/functions/" in blk_file
                ):
                    return [], []

            block["file_path"] = blk_file
            return [block], [(0, len(content))]

        # ── Path 2: Full-document parse with tree-sitter ───────────────────
        lines = content.split("\n")
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)

        # Parse the whole document once so class context is never lost.
        full_doc_symbols, full_doc_lang = await self._extract_full_document_symbols(
            content, None, project_id
        )

        ingested_lang = pstate.get("ingested_lang")

        # ── 2a. Tree-sitter processing ──────────────────────────────────────
        if HAS_TREE_SITTER:
            try:
                config = ProcessConfig()
                if ingested_lang:
                    try:
                        config.language = ingested_lang
                    except Exception:
                        pass

                ts_blocks = await anyio.to_thread.run_sync(
                    lambda: process(content, config)
                )
                if hasattr(ts_blocks, "blocks"):
                    ts_blocks = ts_blocks.blocks

                for tsb in ts_blocks:
                    start, end = tsb.start_byte, tsb.end_byte
                    raw_text = content[start:end].strip()

                    # ── 2a.i. Detect language ──────────────────────────────
                    lang = tsb.language or "text"
                    if ingested_lang:
                        lang = ingested_lang
                    elif lang in ("text", ""):
                        guessed = SignatureExtractor._guess_language(None, raw_text)
                        if guessed != "unknown":
                            lang = guessed
                        else:
                            lang = await self._infer_code_language(raw_text)

                    # ── 2a.ii. Extract code content ─────────────────────────
                    lines_in_block = raw_text.splitlines()
                    if lines_in_block and lines_in_block[0].startswith("```"):
                        lines_in_block = lines_in_block[1:]
                        if lines_in_block and lines_in_block[-1].startswith("```"):
                            lines_in_block = lines_in_block[:-1]
                        code = "\n".join(lines_in_block).strip()
                        block_type = "fenced"
                    else:
                        code = raw_text
                        block_type = "indented"

                    # ── 2a.iii. Assign symbols from full-document parse ────
                    start_line = next(
                        (i for i, off in enumerate(line_offsets) if off > start),
                        len(lines),
                    )
                    end_line = next(
                        (i for i, off in enumerate(line_offsets) if off >= end),
                        len(lines),
                    )
                    pre_syms = (
                        self._assign_symbols_to_span(
                            full_doc_symbols, start_line, end_line
                        )
                        if full_doc_symbols
                        else []
                    )

                    blocks.append(
                        {
                            "language": lang,
                            "code": code,
                            "type": block_type,
                            "precomputed_symbols": pre_syms,
                        }
                    )
                    spans.append((start, end))

                if pstate.get("ingested_lang") is not None:
                    pstate["ingested_lang"] = None

                # ── 2a.iv. Post-process blocks ─────────────────────────────
                if blocks:
                    processed_blocks = []
                    processed_spans = []
                    for idx, block in enumerate(blocks):
                        blk_file = None
                        if self._f.valves.track_file_paths and spans:
                            blk_file = self.extract_file_path_for_block(
                                content, spans[idx][0]
                            )
                        if not blk_file and len(blocks) == 1:
                            extracted_paths = self.extract_file_paths(content)
                            blk_file = extracted_paths[0] if extracted_paths else None

                        if self._f.valves.exclude_filter_internals and blk_file:
                            if (
                                "/app/backend/data/functions/" in blk_file
                                or "open-webui/functions/" in blk_file
                            ):
                                continue

                        block["file_path"] = blk_file
                        processed_blocks.append(block)
                        processed_spans.append(spans[idx])

                    return processed_blocks, processed_spans

            except Exception:
                # Fall through to regex fallback
                pass

        # ── Path 3: Regex fallback ──────────────────────────────────────────
        # 3a. Fenced blocks
        for match in self._f.code_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            start_line = next(
                (i for i, off in enumerate(line_offsets) if off > match.start()),
                len(lines),
            )
            end_line = next(
                (i for i, off in enumerate(line_offsets) if off >= match.end()),
                len(lines),
            )
            pre_syms = (
                self._assign_symbols_to_span(full_doc_symbols, start_line, end_line)
                if full_doc_symbols
                else []
            )
            blocks.append(
                {
                    "language": lang,
                    "code": code,
                    "type": "fenced",
                    "precomputed_symbols": pre_syms,
                }
            )
            spans.append((match.start(), match.end()))

        # 3b. Indented blocks
        indented = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(("    ", "\t")):
                indented.append(line.lstrip(" \t"))
                i += 1
            else:
                if len(indented) >= 3:
                    code = "\n".join(indented)
                    start_line = i - len(indented) + 1
                    end_line = i
                    pre_syms = (
                        self._assign_symbols_to_span(
                            full_doc_symbols, start_line, end_line
                        )
                        if full_doc_symbols
                        else []
                    )
                    blocks.append(
                        {
                            "language": "text",
                            "code": code,
                            "type": "indented",
                            "precomputed_symbols": pre_syms,
                        }
                    )
                    start_offset = line_offsets[i - len(indented)]
                    end_offset = line_offsets[i] - 1
                    spans.append((start_offset, end_offset))
                indented = []
                i += 1

        if len(indented) >= 3:
            code = "\n".join(indented)
            start_line = len(lines) - len(indented) + 1
            end_line = len(lines)
            pre_syms = (
                self._assign_symbols_to_span(full_doc_symbols, start_line, end_line)
                if full_doc_symbols
                else []
            )
            blocks.append(
                {
                    "language": "text",
                    "code": code,
                    "type": "indented",
                    "precomputed_symbols": pre_syms,
                }
            )
            start_offset = line_offsets[len(lines) - len(indented)]
            end_offset = line_offsets[-1] - 1 if line_offsets[-1] > 0 else len(content)
            spans.append((start_offset, end_offset))

        # ── 3c. Post-process fallback blocks ───────────────────────────────
        processed_blocks = []
        processed_spans = []
        for idx, block in enumerate(blocks):
            blk_file = None
            if self._f.valves.track_file_paths and spans:
                blk_file = self.extract_file_path_for_block(content, spans[idx][0])
            if not blk_file and len(blocks) == 1:
                extracted_paths = self.extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None

            if self._f.valves.exclude_filter_internals and blk_file:
                if (
                    "/app/backend/data/functions/" in blk_file
                    or "open-webui/functions/" in blk_file
                ):
                    continue

            block["file_path"] = blk_file
            processed_blocks.append(block)
            processed_spans.append(spans[idx])

        return processed_blocks, processed_spans

    async def _infer_code_language(self, code_snippet: str) -> str:
        """
        Simple heuristic language detection for a code snippet.

        Used as a fallback when tree-sitter language detection fails or
        returns "unknown". Only detects Python and JavaScript currently.

        Args:
            code_snippet: The source code to analyze.

        Returns:
            A language string ("python", "javascript", or "unknown").
        """
        if re.search(r"\bdef\s+\w+\s*\(", code_snippet):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code_snippet):
            return "javascript"
        return "unknown"

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Content classification
    # ═══════════════════════════════════════════════════════════════════════════

    def classify_content(self, content: str, extracted_blocks: list) -> "ContentType":
        """
        Classify a message's content into one of the ContentType categories.

        Detection order (first match wins):
        1. Diff pattern (unified diff) → PROPOSED_CHANGE
        2. Commit pattern (hash) → COMMITTED_CHANGE if "applied/committed/merged"
           in text, otherwise PROPOSED_CHANGE
        3. Traceback/error indicators → ERROR
        4. Tool/function call indicators → TOOL_CALL
        5. Structural code in extracted blocks (def/class/function) → BASE_CODE
        6. Default → GENERAL

        Args:
            content: The raw message content.
            extracted_blocks: List of code blocks extracted from the content.

        Returns:
            A ContentType enum value.
        """
        cl = content.lower()

        # ── 1. Diff pattern ──────────────────────────────────────────────────
        if self._f.diff_pattern.search(content) or "diff --git" in content:
            return ContentType.PROPOSED_CHANGE

        # ── 2. Commit pattern ───────────────────────────────────────────────
        if self._f.commit_pattern.search(content):
            if "applied" in cl or "committed" in cl or "merged" in cl:
                return ContentType.COMMITTED_CHANGE
            return ContentType.PROPOSED_CHANGE

        # ── 3. Error / traceback ────────────────────────────────────────────
        if (
            "traceback" in cl
            or ('file "' in cl and "line " in cl)
            or ("exception" in cl and ("traceback" in cl or 'file "' in cl))
        ):
            return ContentType.ERROR

        # ── 4. Tool call ────────────────────────────────────────────────────
        if '"tool_calls"' in content or '"function"' in content:
            return ContentType.TOOL_CALL

        # ── 5. Base code ────────────────────────────────────────────────────
        for blk in extracted_blocks:
            if blk["language"] in [
                "python",
                "javascript",
                "typescript",
                "go",
                "rust",
                "java",
                "cpp",
            ]:
                if (
                    "def " in blk["code"]
                    or "class " in blk["code"]
                    or "function " in blk["code"]
                ):
                    return ContentType.BASE_CODE

        # ── 6. Default ──────────────────────────────────────────────────────
        return ContentType.GENERAL

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. File path extraction
    # ═══════════════════════════════════════════════════════════════════════════

    def extract_file_paths(self, content: str) -> list:
        """
        Extract all file paths matching the configured pattern from content.

        Uses the regex pattern defined in `valves.file_path_pattern` to find
        file paths. Returns them as strings (unwraps tuple captures if present).

        Args:
            content: The text to scan for file paths.

        Returns:
            A list of file path strings (may be empty).
        """
        if not self._f.valves.track_file_paths:
            return []
        matches = re.findall(self._f.valves.file_path_pattern, content)
        return [m[0] if isinstance(m, tuple) else m for m in matches]

    def extract_file_path_for_block(
        self, content: str, block_start: int
    ) -> Optional[str]:
        """
        Try to find a file path associated with a code block by scanning backwards.

        Searches the text immediately preceding the block for a file path pattern.
        Used to infer the file that a code snippet belongs to.

        Args:
            content: The full text content.
            block_start: The byte offset where the block begins.

        Returns:
            The file path if found, otherwise None.
        """
        if block_start <= 0:
            return None

        # ── 1. Scan backwards from the block start ──────────────────────────
        before = content[:block_start]
        lines = before.splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # ── 1a. Try direct file path pattern ────────────────────────────
            match = re.search(self._f.valves.file_path_pattern, line)
            if match:
                return match.group(1) if match.lastindex else match.group(0)

            # ── 1b. Try line-range extraction ──────────────────────────────
            file_path, _, _ = self._extract_line_range(line)
            if file_path:
                return file_path

            break  # Only look at the nearest non-empty line

        return None

    def _extract_line_range(
        self, content: str
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Extract file path and line numbers from a reference string.

        Matches patterns like "file.py:42" or "file.py:10-20" to extract
        the file path, start line, and end line.

        Args:
            content: A string that may contain a file:line reference.

        Returns:
            A tuple of (file_path, start_line, end_line). Any field may be None.
        """
        if not self._f.valves.track_line_numbers:
            return None, None, None

        pattern = r"(?:^|\s)([^\s:]+\.\w+):(\d+)(?:-(\d+))?"
        match = re.search(pattern, content)
        if match:
            return (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)) if match.group(3) else int(match.group(2)),
            )
        return None, None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. General utilities
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Remove non-printable characters and replace backticks with single quotes.

        This prevents control characters from breaking the prompt formatting
        and ensures backticks don't interfere with fenced code blocks.

        Args:
            text: The text to sanitize.

        Returns:
            Sanitized text with no control characters and backticks replaced.
        """
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        cleaned = cleaned.replace("`", "'")
        return cleaned

    @staticmethod
    def format_block_context(block: "CodeBlock", is_latest: bool) -> str:
        """
        Render a single active CodeBlock for injection into the system prompt.

        Produces a formatted block with:
        - A header line with block hash, file path (if available), and a [LATEST] marker.
        - A fenced code block with the language tag and the full code body.

        Args:
            block: The CodeBlock to render.
            is_latest: Whether this is the most recent version of the file.

        Returns:
            A formatted string ready for inclusion in the system prompt.
        """
        latest_tag = " [LATEST]" if is_latest else ""
        location = f" `{block.file_path}`" if block.file_path else ""
        language = block.symbols[0].language if block.symbols else ""
        header = f"#### Block {block.hash[:8]}{location}{latest_tag}"
        return f"{header}\n```{language}\n{block.content}\n```"

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Symbol body extraction
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_symbol_body(
        block: "CodeBlock", node_id: str, max_chars: int = 0
    ) -> str:
        """
        Extract the source body of a symbol using its line range.

        This is the key method for precise symbol extraction: given a block
        and a symbol identifier, it finds the symbol's line range and slices
        the block content to return only that symbol's code.

        The method handles three matching strategies:
        1. Full qualified id (with file_path) — most precise.
        2. Qualified id (without file_path) — for cross-file consistency.
        3. Bare name — fallback for when only the name is known.

        Args:
            block: The CodeBlock containing the symbol.
            node_id: The qualified symbol id (e.g. "ClassName.method").
            max_chars: Optional character limit for the returned body.

        Returns:
            The sliced source body, or the whole block content if slicing fails.
        """
        # ── 1. Locate the symbol in the block ──────────────────────────────
        target = None
        for sym in block.symbols:
            # Try full qualified id (with file_path), then without, then bare name.
            if (
                qualify_symbol_name(sym.name, sym.parent_symbol, sym.file_path)
                == node_id
                or qualify_symbol_name(sym.name, sym.parent_symbol) == node_id
                or sym.name == node_id
            ):
                target = sym
                break

        # ── 2. Slice the block content by the symbol's line range ──────────
        body = block.content
        if target is not None and target.line_start and target.line_end:
            lines = block.content.split("\n")
            # Defensive bounds check to ensure line numbers are valid.
            if 1 <= target.line_start <= len(lines) and target.line_end <= len(lines):
                sliced = "\n".join(
                    lines[target.line_start - 1 : target.line_end]
                ).strip()
                if sliced:
                    body = sliced

        # ── 3. Apply character limit if requested ──────────────────────────
        if max_chars and len(body) > max_chars:
            body = body[:max_chars] + "\n# ... [truncated]"

        return body

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Similarity & deduplication
    # ═══════════════════════════════════════════════════════════════════════════

    def calculate_code_similarity(self, code1: str, code2: str) -> float:
        """
        Compute similarity between two code snippets.

        Uses two strategies in order:
        1. AST-based structural similarity for Python code (if enabled).
        2. Fuzzy matching (token sort ratio) as fallback.

        Args:
            code1: First code snippet.
            code2: Second code snippet.

        Returns:
            A similarity score in [0.0, 1.0]. Higher = more similar.
        """
        # ── 1. AST similarity (Python only) ─────────────────────────────────
        if (
            self._f.valves.enable_ast_deduplication
            and len(code1) > 30
            and len(code2) > 30
        ):
            ast_sim = self._ast_similarity(code1, code2)
            if ast_sim is not None:
                return ast_sim

        # ── 2. Fallback: token-sort ratio or character-level similarity ────
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))

        return fuzz.token_sort_ratio(code1, code2) / 100.0

    def _ast_similarity(self, code1: str, code2: str) -> Optional[float]:
        """
        Compute Jaccard similarity on AST node type distributions.

        For Python code only. Uses memoized AST signatures to avoid re-parsing
        the same code multiple times within a single deduplication pass.

        Args:
            code1: First Python code snippet.
            code2: Second Python code snippet.

        Returns:
            A similarity score in [0.0, 1.0], or None if either snippet
            is not Python or parsing fails.
        """
        # ── 1. Quick check: both snippets look like Python ──────────────────
        if not (
            re.search(r"\bdef\s+\w+\s*\(", code1) or re.search(r"\bclass\s+\w+", code1)
        ):
            return None

        # ── 2. Get memoized AST signatures ──────────────────────────────────
        sig1 = self._parsed_ast_signature(code1)
        sig2 = self._parsed_ast_signature(code2)
        if sig1 is None or sig2 is None:
            return None

        dump1, c1 = sig1
        dump2, c2 = sig2

        # ── 3. Quick exit: structurally identical ──────────────────────────
        if dump1 == dump2:
            return 1.0

        # ── 4. Compute Jaccard similarity on node types ─────────────────────
        all_types = set(c1.keys()) | set(c2.keys())
        if not all_types:
            return 0.0

        intersection = sum(min(c1.get(t, 0), c2.get(t, 0)) for t in all_types)
        union = sum(max(c1.get(t, 0), c2.get(t, 0)) for t in all_types)

        return intersection / union if union > 0 else 0.0

    def _parsed_ast_signature(self, code: str) -> Optional[Tuple[str, Counter]]:
        """
        Parse code into its AST signature: stripped dump + node-type Counter.

        Memoized by content hash so the same code is never parsed twice within
        a single deduplication pass. Docstrings are stripped from the AST dump
        because they don't affect structural similarity.

        Args:
            code: The source code to parse.

        Returns:
            A tuple of (stripped_ast_dump, node_type_counter), or None
            if parsing fails. The result is cached for future calls.
        """
        # ── 1. Check cache ───────────────────────────────────────────────────
        key = hashlib.md5(code.encode()).hexdigest()
        cached = self._ast_signature_cache.get(key)
        if cached is not None:
            self._ast_signature_cache.move_to_end(key)
            return cached

        # ── 2. Parse the code ───────────────────────────────────────────────
        try:
            tree = ast.parse(code)
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            return None

        # ── 3. Strip docstrings from the AST ────────────────────────────────
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                continue
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
                if not node.body:
                    node.body = [ast.Pass()]

        # ── 4. Build signature ──────────────────────────────────────────────
        result = (
            ast.dump(tree),
            Counter(type(node).__name__ for node in ast.walk(tree)),
        )

        # ── 5. Cache with LRU eviction ──────────────────────────────────────
        self._ast_signature_cache[key] = result
        if len(self._ast_signature_cache) > 200:
            self._ast_signature_cache.popitem(last=False)

        return result

    def remove_duplicate_blocks(self, state: dict, project_id: str) -> None:
        """
        Remove duplicate or near-duplicate code blocks from the active set.

        Uses three strategies:
        1. Pairwise similarity comparison (AST or fuzzy matching).
        2. Age-based: keep newer blocks if similarity threshold is met.
        3. Per-file version limiting: keep only the most recent version.

        Args:
            state: The conversation state containing active_blocks.
            project_id: The current project identifier.
        """
        if not self._f.valves.auto_remove_duplicate_blocks:
            return

        blocks = list(state.active_blocks.values())
        to_remove = set()

        # ── 1. Pairwise similarity comparison ──────────────────────────────
        for i, block in enumerate(blocks):
            if block.hash in to_remove or block.pinned or block.obsolete:
                continue

            for j, other in enumerate(blocks[i + 1 :], start=i + 1):
                if other.hash in to_remove or other.pinned or other.obsolete:
                    continue

                sim = self.calculate_code_similarity(block.content, other.content)
                if sim >= self._f.valves.code_similarity_threshold:
                    age_diff = abs(block.timestamp - other.timestamp) / 3600

                    # ── 1a. If age difference is significant ──────────────
                    if age_diff > self._f.valves.max_duplicate_age_hours:
                        if (
                            block.timestamp < other.timestamp
                            and block.importance_score < 5.0
                        ):
                            to_remove.add(block.hash)
                        elif (
                            other.timestamp < block.timestamp
                            and other.importance_score < 5.0
                        ):
                            to_remove.add(other.hash)
                        continue

                    # ── 1b. Compare by importance score ─────────────────────
                    score_diff = abs(block.importance_score - other.importance_score)
                    if score_diff < 1.0:
                        # If scores are similar, keep the newer one.
                        if block.timestamp >= other.timestamp:
                            to_remove.add(other.hash)
                        else:
                            to_remove.add(block.hash)
                    elif block.importance_score >= other.importance_score:
                        to_remove.add(other.hash)
                    else:
                        to_remove.add(block.hash)

        # ── 2. Per-file version limiting ────────────────────────────────────
        blocks_by_file = defaultdict(list)
        for b in blocks:
            if b.file_path and not b.pinned:
                blocks_by_file[b.file_path].append(b)

        for file_path, blks in blocks_by_file.items():
            if len(blks) > 1:
                blks.sort(key=lambda b: b.timestamp, reverse=True)
                for b in blks[1:]:
                    to_remove.add(b.hash)

        # ── 3. Apply removal ────────────────────────────────────────────────
        for h in to_remove:
            if h in state.active_blocks:
                block = state.active_blocks[h]
                self._f._symbol_index.remove_all_for_block(
                    block.hash, block.symbols, project_id
                )
                del state.active_blocks[h]

        # ── 4. Clean up dependent lists ─────────────────────────────────────
        state.recent_changes = [
            b for b in state.recent_changes if b.hash not in to_remove
        ]
        state.committed_changes = [
            b for b in state.committed_changes if b.hash not in to_remove
        ]

        # ── 5. Update state and invalidate cache ───────────────────────────
        if to_remove:
            state.has_any_calls = any(
                any(s.calls for s in b.symbols) for b in state.active_blocks.values()
            )
            self._f._activation.invalidate_lightweight_cache(project_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Proposed changes & diffs
    # ═══════════════════════════════════════════════════════════════════════════

    def has_conflicting_proposed_changes(
        self, state: dict, new_block: "CodeBlock"
    ) -> bool:
        """
        Check if a proposed change conflicts with an existing recent change.

        Conflict is detected when:
        - Two proposed changes affect the same file, OR
        - They have high content similarity (>80%).

        Args:
            state: The conversation state containing recent_changes.
            new_block: The proposed change block to check.

        Returns:
            True if there is a conflict with an existing proposed change.
        """
        if new_block.content_type != ContentType.PROPOSED_CHANGE:
            return False

        for existing in state.recent_changes:
            if existing.hash == new_block.hash:
                continue

            same_file = (
                existing.file_path
                and new_block.file_path
                and existing.file_path == new_block.file_path
            )

            if (
                same_file
                or self.calculate_code_similarity(existing.content, new_block.content)
                > 0.8
            ):
                return True

        return False

    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        """
        Apply a unified diff patch to original text.

        Parses the diff hunks and applies them in reverse order (so line
        numbers remain correct as earlier hunks are applied).

        Args:
            original: The original source code.
            diff_text: The unified diff content (from `git diff` or similar).

        Returns:
            The patched source code, or None if the diff cannot be applied.
        """
        if not self._f.valves.enable_diff_application:
            return None

        lines = original.splitlines(keepends=False)
        result_lines = lines[:]
        hunks = []

        # ── 1. Parse diff hunks ─────────────────────────────────────────────
        for match in re.finditer(
            r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*?)(?=@@|\Z)", diff_text, re.DOTALL
        ):
            old_start = int(match.group(1))
            old_count_str = match.group(2)
            old_count = int(old_count_str) if old_count_str else 1
            new_start = int(match.group(3))
            new_count_str = match.group(4)
            new_count = int(new_count_str) if new_count_str else 1
            hunk_body = match.group(5).strip("\n")

            old_lines, new_lines = [], []
            for line in hunk_body.split("\n"):
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])

            if old_count == 0:
                old_start_idx = old_start
            else:
                old_start_idx = old_start - 1

            if new_count == 0:
                new_lines = []

            hunks.append((old_start_idx, old_lines, new_lines))

        # ── 2. Apply hunks in reverse order ─────────────────────────────────
        applied_any = False
        for old_start_idx, old_lines, new_lines in reversed(hunks):
            # ── 2a. Bounds check ─────────────────────────────────────────────
            if old_start_idx < 0 or old_start_idx + len(old_lines) > len(result_lines):
                logger.warning(
                    f"Unified diff hunk out of bounds (start={old_start_idx}, "
                    f"lines={len(old_lines)}, total={len(result_lines)})"
                )
                continue

            # ── 2b. Verify context matches ──────────────────────────────────
            if (
                result_lines[old_start_idx : old_start_idx + len(old_lines)]
                != old_lines
            ):
                logger.warning(f"Unified diff hunk mismatch at line {old_start_idx}")
                continue

            # ── 2c. Apply hunk ──────────────────────────────────────────────
            result_lines = (
                result_lines[:old_start_idx]
                + new_lines
                + result_lines[old_start_idx + len(old_lines) :]
            )
            applied_any = True

        if not applied_any and hunks:
            logger.warning("No hunks were applied from the unified diff")
            return None

        return "\n".join(result_lines)

    async def apply_change_with_diff(
        self, base_block: "CodeBlock", proposed_block: "CodeBlock"
    ) -> bool:
        """
        Apply a unified diff from a proposed change onto a base block.

        If the diff applies successfully, the base block is updated with the
        new content, symbols are re-extracted, and the symbol index is updated.

        Args:
            base_block: The original code block to patch.
            proposed_block: The diff-containing proposed change.

        Returns:
            True if the diff was applied successfully, False otherwise.
        """
        # ── 1. Validate inputs ──────────────────────────────────────────────
        if proposed_block.content_type != ContentType.PROPOSED_CHANGE:
            return False

        if not (
            "@@" in proposed_block.content
            and ("-" in proposed_block.content or "+" in proposed_block.content)
        ):
            return False

        # ── 2. Apply the diff ───────────────────────────────────────────────
        new_code = self._apply_unified_diff(base_block.content, proposed_block.content)
        if not new_code or new_code == base_block.content:
            return False

        project_id = self._f.valves.project_id

        # ── 3. Remove old symbols from index ───────────────────────────────
        self._f._symbol_index.remove_all_for_block(
            base_block.hash, base_block.symbols, project_id
        )

        # ── 4. Update block content and hash ───────────────────────────────
        base_block.content = new_code
        base_block.hash = hashlib.md5(new_code.encode()).hexdigest()[:16]

        # ── 5. Re-extract symbols ───────────────────────────────────────────
        base_block.symbols = await SignatureExtractor.extract_async(
            new_code, base_block.file_path
        )

        # ── 6. Re-index symbols ─────────────────────────────────────────────
        for sym in base_block.symbols:
            sym.parent_block_hash = base_block.hash
            self._f._symbol_index.add(sym, base_block.hash, project_id)

        # ── 7. Update cached token count ───────────────────────────────────
        if self._f.tokenizer:
            base_block._cached_token_count = len(self._f.tokenizer.encode(new_code))
        else:
            base_block._cached_token_count = len(new_code) // 4

        # ── 8. Update metadata ─────────────────────────────────────────────
        base_block.timestamp = time.time()
        base_block.is_active = True
        base_block.potentially_affected = False
        base_block.importance_score = min(base_block.importance_score + 2.0, 10.0)

        # ── 9. Invalidate cache ─────────────────────────────────────────────
        self._f._activation.invalidate_lightweight_cache(project_id)

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Data flow edges
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_data_flow_edges_regex(
        self, code: str, project_id: str
    ) -> List["Edge"]:
        """
        Fallback data flow extraction for non-Python languages via regex.

        This is a best-effort heuristic that identifies:
        1. Assignments to variables from function calls.
        2. Subsequent usage of those variables as function arguments.

        Args:
            code: The source code to analyze.
            project_id: The current project identifier.

        Returns:
            A list of Edge objects representing data flow relationships.
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        edges: List[Edge] = []

        # ── 1. Find assignments from function calls ─────────────────────────
        pattern = re.compile(
            r"\b(\w+)\s*=\s*(" + "|".join(re.escape(n) for n in all_names) + r")\s*\("
        )

        for match in pattern.finditer(code):
            callee = match.group(2)
            var_name = match.group(1)

            # ── 2. Find subsequent usage of the variable ────────────────────
            use_pattern = re.compile(
                r"\b("
                + "|".join(re.escape(n) for n in all_names)
                + r")\s*\([^)]*\b"
                + re.escape(var_name)
                + r"\b"
            )

            for use_match in use_pattern.finditer(code):
                consumer = use_match.group(1)
                if consumer != callee:
                    edges.append(
                        Edge(
                            src=callee,
                            dst=consumer,
                            type="data_flow",
                            weight=EDGE_WEIGHTS["data_flow"],
                            confidence=0.5,
                        )
                    )

        return edges

    def extract_data_flow_edges(
        self, code: str, file_path: Optional[str], project_id: str
    ) -> List["Edge"]:
        """
        Extract data flow edges from Python code using AST, with regex fallback.

        For Python code, this uses AST analysis to track:
        1. Variable assignments.
        2. Which variables are passed as arguments to function calls.
        3. The flow of data from producers to consumers.

        The key insight is that if a variable is assigned from a function call
        and later passed to another function, there's a data flow edge from
        the producer to the consumer.

        Args:
            code: The source code to analyze.
            file_path: The file path (used to determine language).
            project_id: The current project identifier.

        Returns:
            A list of Edge objects representing data flow relationships.
        """
        # ── 1. Non-Python: use regex fallback ──────────────────────────────
        if not file_path or not file_path.endswith(".py"):
            return self._extract_data_flow_edges_regex(code, project_id)

        all_names = self._f._symbol_index.get_all_names(project_id)
        if not all_names:
            return []

        edges: List[Edge] = []

        # ── 2. Parse Python AST ─────────────────────────────────────────────
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # ── 3. Map line numbers to enclosing classes ────────────────────────
        line_to_class: Dict[int, str] = {}

        def _mark_classes(node, current_class: str) -> None:
            """Recursively mark lines with their enclosing class name."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    end = getattr(child, "end_lineno", child.lineno)
                    for ln in range(child.lineno, end + 1):
                        line_to_class[ln] = current_class
                    _mark_classes(child, child.name)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(child, "end_lineno", child.lineno)
                    for ln in range(child.lineno, end + 1):
                        line_to_class[ln] = current_class
                    _mark_classes(child, current_class)
                else:
                    _mark_classes(child, current_class)

        _mark_classes(tree, "")

        # ── 4. Analyze each function for data flow ──────────────────────────
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            caller_name = func_node.name
            if caller_name not in all_names:
                continue

            caller_class = line_to_class.get(func_node.lineno, "")
            caller_qid = qualify_symbol_name(caller_name, caller_class)

            # ── 4a. Collect variables assigned in this function ────────────
            assigned_vars: Set[str] = set()
            for child in ast.walk(func_node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)

            # ── 4b. Find calls that use assigned variables as arguments ────
            for child in ast.walk(func_node):
                if not isinstance(child, ast.Call):
                    continue

                callee_name = None
                if isinstance(child.func, ast.Name):
                    callee_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    callee_name = child.func.attr

                if callee_name not in all_names or callee_name == caller_name:
                    continue

                # Check if any argument is a variable assigned earlier.
                args_are_local_vars = any(
                    isinstance(arg, ast.Name) and arg.id in assigned_vars
                    for arg in child.args
                )

                if args_are_local_vars or child.args:
                    edges.append(
                        Edge(
                            src=caller_qid,
                            dst=callee_name,
                            type="data_flow",
                            weight=EDGE_WEIGHTS["data_flow"],
                            confidence=0.7,
                        )
                    )

        return edges


class ActivationEngine:
    """Builds activation graphs from query seeds and the symbol call graph,
    determining which code blocks are relevant to the current user message.

    Provides:
    * ``build_activation_graph(query, project_id)`` — produces an
      ``ActivationGraph`` with scores for every symbol reachable from
      query‑matched seeds, traceback frames, and recent history.
    * ``get_active_code_context(project_id, query)`` — returns a formatted
      string of all active blocks, sorted by relevance, for injection
      when path analysis is disabled.
    * ``rebuild_path_index(project_id)`` — reconstructs the ``PathIndex``
      from the current ``SymbolIndex`` after code changes.
    * ``resolve_dangling_edges(project_id)`` — upgrades provisional cross‑chunk
      call edges when the referenced symbol is later defined.
    * ``speculative_prefetch(project_id, last_activated)`` — pre‑builds
      ``CodePathView`` objects for symbols likely to be needed in the next
      request.
    * ``compute_code_state_hash(project_id)`` — returns a hash of the active
      blocks, used to detect KV‑cache invalidations and staleness.
    * ``invalidate_lightweight_cache(project_id)`` — clears cached context
      and centrality so the next request rebuilds them from scratch.

    Docs 10–13 backported:
        E7 – normalizes PPR scores to [0,1] within the project distribution.
    """

    # ── Q2: PPR cache (nested class) ──────────────────────────────────────────
    class _PPRCache:
        """LRU cache for PPR results.

        Key: (code_state_hash: str, seed_qids: frozenset[str])
        Value: dict[str, float]  — qid → PPR score

        Thread safety: not needed (single-threaded inlet/build_block_b flow).
        Invalidation: automatic via code_state_hash — any code change produces
        a new hash, naturally evicting all cached entries for that project.
        """

        def __init__(self, maxsize: int = 20):
            self._cache: OrderedDict[tuple, dict[str, float]] = OrderedDict()
            self._maxsize = maxsize
            self._hits = 0
            self._misses = 0

        def get(self, code_hash: str, seeds: frozenset) -> Optional[Dict[str, float]]:
            """Retrieve cached PPR scores if available."""
            key = (code_hash, seeds)
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

        def set(
            self, code_hash: str, seeds: frozenset, scores: Dict[str, float]
        ) -> None:
            """Store PPR scores in the cache."""
            key = (code_hash, seeds)
            self._cache[key] = scores
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

        @property
        def stats(self) -> str:
            """Return cache hit/miss statistics."""
            total = self._hits + self._misses
            rate = self._hits / total if total else 0
            return f"PPR cache: {self._hits}/{total} hits ({rate:.0%})"

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref
        # ── Q2: PPR cache ──
        self._ppr_cache = self._PPRCache(maxsize=20)

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Active code context (fallback when path analysis is disabled)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_active_code_context(self, project_id: str, user_query: str = "") -> str:
        """Return a formatted string with the currently active code context for the LLM."""
        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 1 — Load state & filter active blocks
        # ═══════════════════════════════════════════════════════════════════════════
        state = self._f._conversation_state_manager.get(project_id)
        if not state or not state.active_blocks:
            return ""

        now = time.time()
        active = []
        for block in state.active_blocks.values():
            if block.obsolete:
                continue
            if not block.is_active and self._f.valves.track_active_code_age:
                if (
                    now - block.timestamp
                    > self._f.valves.active_code_timeout_minutes * 60
                ):
                    continue
            active.append(block)

        if not active:
            return ""

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 2 — Recent activity & mentioned files
        # ═══════════════════════════════════════════════════════════════════════════
        recent_window = self._f.valves.recent_activity_window_minutes * 60
        recent_files = {}
        for b in active:
            if b.file_path:
                if (
                    b.file_path not in recent_files
                    or b.timestamp > recent_files[b.file_path]
                ):
                    recent_files[b.file_path] = b.timestamp

        recent_lines = []
        for file_path, ts in recent_files.items():
            age_seconds = now - ts
            if age_seconds <= recent_window:
                minutes_ago = int(age_seconds / 60)
                time_str = f"{minutes_ago} min ago" if minutes_ago > 0 else "just now"
                recent_lines.append(f"- `{file_path}` ({time_str})")

        recent_section = ""
        if recent_lines:
            recent_section = (
                f"## Recent Activity (last {self._f.valves.recent_activity_window_minutes} min)\n"
                + "\n".join(recent_lines)
                + "\n\n"
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 3 — Compute relevance boost (file + symbol mentions)
        # ═══════════════════════════════════════════════════════════════════════════
        mentioned_files = set()
        mentioned_symbols = set()
        if user_query:
            mentioned_files = set(
                re.findall(self._f.valves.file_path_pattern, user_query)
            )
            all_symbol_names = self._f._symbol_index.get_all_names(project_id)
            words = set(re.findall(r"\b[\w-]+\b", user_query))
            mentioned_symbols = all_symbol_names.intersection(words)

        BOOST = 5.0

        def relevance_boost(block: "CodeBlock") -> float:
            score = 0.0
            if block.file_path and block.file_path in mentioned_files:
                score += BOOST
            for sym in block.symbols:
                if sym.name in mentioned_symbols:
                    score += BOOST
                    break
            return score

        boosted_active = []
        for b in active:
            boost = relevance_boost(b)
            boosted_active.append((b, boost))

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 4 — Sort and group blocks by type
        # ═══════════════════════════════════════════════════════════════════════════
        boost_priority = self._f.valves.raw_file_priority_boost
        boosted_active.sort(
            key=lambda pair: (
                pair[1] > 0,
                pair[1],
                pair[0].importance_score + (boost_priority if pair[0].is_raw else 0),
            ),
            reverse=True,
        )

        latest_per_file = {}
        for b in active:
            if b.file_path:
                if (
                    b.file_path not in latest_per_file
                    or b.timestamp > latest_per_file[b.file_path].timestamp
                ):
                    latest_per_file[b.file_path] = b
        latest_hashes = {b.hash for b in latest_per_file.values()}

        base_codes = [
            b for b, _ in boosted_active if b.content_type == ContentType.BASE_CODE
        ][: self._f.valves.max_base_code_blocks]
        proposed = [
            b
            for b, _ in boosted_active
            if b.content_type == ContentType.PROPOSED_CHANGE
        ][: self._f.valves.max_proposed_changes]
        committed = [
            b
            for b, _ in boosted_active
            if b.content_type == ContentType.COMMITTED_CHANGE
        ][: self._f.valves.max_committed_changes]
        errors = (
            [b for b, _ in boosted_active if b.content_type == ContentType.ERROR][:3]
            if self._f.valves.preserve_error_context
            else []
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 5 — Build the context sections (Base, Proposed, Committed, Errors)
        # ═══════════════════════════════════════════════════════════════════════════
        parts = ["## Currently Active Code Context (by importance)\n"]
        parts.insert(
            1,
            "> **Note**: If multiple versions of a file appear, the one marked [LATEST] is the most recent and should be used.\n",
        )
        if recent_section:
            parts.insert(0, recent_section)

        if base_codes:
            parts.append("### Base Code (current work):")
            for b in base_codes:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(CodeBlockManager.format_block_context(b, is_latest) + tag)

        if proposed:
            parts.append("### Proposed Changes (pending review):")
            for b in proposed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(CodeBlockManager.format_block_context(b, is_latest) + tag)

        if committed:
            parts.append("### Recently Committed Changes:")
            for b in committed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(CodeBlockManager.format_block_context(b, is_latest) + tag)

        if errors:
            parts.append("### Recent Errors:")
            for b in errors:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(CodeBlockManager.format_block_context(b, is_latest) + tag)

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION 6 — Dynamic budget & truncation (uses pstate)
        # ═══════════════════════════════════════════════════════════════════════════
        pstate = self._f._project_state_manager.get_pstate(project_id)
        used_system = pstate.get("last_system_tokens", 0)

        effective_budget = max(
            4000,
            self._f.valves.context_window_tokens
            - used_system
            - self._f.valves.response_reserve_tokens,
        )
        max_tokens = min(
            self._f.valves.active_context_max_tokens or effective_budget,
            effective_budget,
        )

        # Convergent truncation with line-by-line fence detection.
        if max_tokens > 0 and self._f.tokenizer:
            part_tokens = [len(self._f.tokenizer.encode(p)) for p in parts]
            current_tokens = sum(part_tokens)
            truncation_done = False

            while current_tokens > max_tokens and len(parts) > 2:
                excess = current_tokens - max_tokens

                largest_idx = max(range(len(parts)), key=lambda i: part_tokens[i])
                largest_tok = part_tokens[largest_idx]

                if largest_tok >= excess + 100:
                    target = max(100, largest_tok - excess - 50)
                    truncated_text = self._f._tokens.truncate_text_to_tokens(
                        parts[largest_idx], target
                    )
                    if self._has_open_fence(truncated_text):
                        truncated_text += "\n```"
                    parts[largest_idx] = truncated_text + "\n[...truncado...]"
                    part_tokens[largest_idx] = len(
                        self._f.tokenizer.encode(parts[largest_idx])
                    )
                    current_tokens = sum(part_tokens)
                    truncation_done = True
                else:
                    parts.pop()
                    part_tokens.pop()
                    current_tokens = sum(part_tokens)

            if not truncation_done and current_tokens > max_tokens:
                parts.append(f"[Context truncated to fit token limit ({max_tokens})]")

        return "\n".join(parts)

    @staticmethod
    def _has_open_fence(text: str) -> bool:
        """True if the text ends inside an unclosed fenced code block."""
        inside = False
        for line in text.splitlines():
            if line.strip().startswith("```"):
                inside = not inside
        return inside

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Seed extraction helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_query_seeds(
        self, query: str, project_id: str
    ) -> Tuple[List[str], List[str]]:
        """
        Extract seed symbols from the query.
        Returns (exact_matches, partial_matches).
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        query_words = set(re.findall(r"\b\w+\b", query))

        exact = list(all_names.intersection(query_words))

        partial = []
        if len(exact) < 3:
            for word in query_words:
                if len(word) < 4:
                    continue
                for name in all_names:
                    if word.lower() in name.lower() and name not in exact:
                        partial.append(name)
                        break
            partial = partial[:5]

        return exact, partial

    def _extract_traceback_seeds(
        self, content: str, project_id: str
    ) -> List[Tuple[str, float]]:
        """
        Extract function names from a traceback with scores proportional
        to their depth in the call stack.
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        if not all_names:
            return []

        frames: List[str] = []

        # Python traceback
        py_pattern = re.compile(
            r'File\s+"[^"]+",\s+line\s+\d+,\s+in\s+(\w+)',
            re.MULTILINE,
        )
        for match in py_pattern.finditer(content):
            func = match.group(1)
            if func in all_names and func != "<module>":
                frames.append(func)

        # JavaScript / TypeScript
        js_pattern = re.compile(
            r"\bat\s+(\w+)\s*\([^)]*:\d+:\d+\)",
            re.MULTILINE,
        )
        for match in js_pattern.finditer(content):
            func = match.group(1)
            if func in all_names:
                frames.append(func)

        # Java / Kotlin
        java_pattern = re.compile(
            r"\bat\s+[\w.]+\.(\w+)\(\w+\.(?:java|kt):\d+\)",
            re.MULTILINE,
        )
        for match in java_pattern.finditer(content):
            func = match.group(1)
            if func in all_names:
                frames.append(func)

        if not frames:
            return []

        seen: Set[str] = set()
        unique_frames: List[str] = []
        for f in frames:
            if f not in seen:
                seen.add(f)
                unique_frames.append(f)

        n = len(unique_frames)
        results = []
        for i, func_name in enumerate(unique_frames):
            score = 0.5 + 0.5 * (i / max(n - 1, 1))
            specificity = self._compute_node_specificity(func_name, project_id)
            adjusted = min(1.0, score * min(specificity, 1.5))
            results.append((func_name, adjusted))

        self._f._log_debug(
            f"Traceback seeds: {len(results)} frame(s) detected "
            f"({[r[0] for r in results]})"
        )
        return results

    def _compute_node_specificity(self, symbol_name: str, project_id: str) -> float:
        """
        IDF-like specificity of a symbol.
        Symbols appearing in many blocks are less specific (like stop-words).
        Returns a multiplier in [0.1, 3.0] to adjust its weight as a seed.
        """
        import math

        all_names = self._f._symbol_index.get_all_names(project_id)
        total = max(len(all_names), 1)
        n_blocks = len(self._f._symbol_index.find_blocks(symbol_name, project_id))
        if n_blocks == 0:
            return 1.0
        specificity = math.log(total / n_blocks) + 1.0
        return max(0.1, min(3.0, specificity))

    def _extract_history_seeds(
        self, messages: Optional[List[dict]], project_id: str, lookback: int = 6
    ) -> Dict[str, float]:
        """
        Extract symbols with high mention frequency in recent messages.
        Returns {symbol_name: boost_score} where boost_score ∈ (0.0, 0.6].
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        if not all_names or not messages:
            return {}

        recent = messages[-lookback:] if len(messages) > lookback else messages
        mention_counts: Counter = Counter()

        for msg in recent:
            content = msg.get("content", "")
            if not content:
                continue
            words = set(re.findall(r"\b\w+\b", content))
            for sym in all_names.intersection(words):
                mention_counts[sym] += 1

        if not mention_counts:
            return {}

        max_count = max(mention_counts.values())
        return {
            sym: min(
                self._f.valves.history_seeds_max_boost,
                self._f.valves.history_seeds_max_boost * (count / max_count),
            )
            for sym, count in mention_counts.items()
            if count > 0
        }

    def _prepare_seed_symbols(
        self, query: str, project_id: str, messages: Optional[List[dict]]
    ) -> Tuple[List[str], List[str], List[Tuple[str, float]], Dict[str, float]]:
        """Extract exact, partial, traceback and historical seed symbols from the query."""
        exact_seeds, partial_seeds = self._extract_query_seeds(query, project_id)
        tb_seeds = (
            self._extract_traceback_seeds(query, project_id)
            if self._f.valves.enable_traceback_activation
            else []
        )
        history_boosts = (
            self._extract_history_seeds(
                messages, project_id, lookback=self._f.valves.history_seeds_lookback
            )
            if (self._f.valves.enable_history_seeds and messages)
            else {}
        )
        return exact_seeds, partial_seeds, tb_seeds, history_boosts

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. PPR computation with caching (Q2)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_or_compute_ppr_scores(
        self,
        seed_qids: List[str],
        project_id: str,
        code_state_hash: str,
        edges_out: Dict[str, List[Edge]],
        max_steps: int = 20,
        min_score: float = 0.05,
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """
        Compute PPR scores with LRU caching.

        Q2 fix: Caches results by (code_state_hash, frozenset(seed_qids)).
        Avoids recomputation when the same seeds are used on the same code.

        Args:
            seed_qids: List of qualified symbol ids to seed.
            project_id: Current project identifier.
            code_state_hash: Hash of the code state (cache invalidation key).
            edges_out: Call graph edges (outgoing).
            max_steps: PPR propagation steps.
            min_score: Minimum score to keep.
            alpha: Damping factor.

        Returns:
            Dict[str, float]: qid → PPR score.
        """
        seed_frozenset = frozenset(seed_qids)

        # ── Check cache ──
        cached = self._ppr_cache.get(code_state_hash, seed_frozenset)
        if cached is not None:
            self._f._log_debug(f"PPR: cache hit ({self._ppr_cache.stats})")
            return cached

        # ── Compute PPR ──
        ag = ActivationGraph()

        # Seed the graph
        total_seed_score = len(seed_qids)
        if total_seed_score == 0:
            # Fallback to entry points
            pstate = self._f._project_state_manager.get_pstate(project_id)
            centrality = pstate.get("node_centrality", {})
            entry_points = self._f._path_index.find_entry_points(
                self._f._symbol_index, project_id
            )
            if entry_points:
                sorted_eps = sorted(
                    entry_points, key=lambda ep: centrality.get(ep, 0.0), reverse=True
                )
                for sym_name in sorted_eps[:3]:
                    cent_score = centrality.get(sym_name, 0.0)
                    seed_score = 0.2 + 0.2 * cent_score
                    ag.seed([sym_name], initial_score=seed_score)
        else:
            # Distribute seed scores evenly
            init_score = 1.0 / total_seed_score
            ag.seed(seed_qids, initial_score=init_score)

        # Run PPR propagation
        ag.propagate(
            edges_out=edges_out,
            max_steps=max_steps,
            min_score=min_score,
            alpha=alpha,
        )

        # Extract scores
        scores = ag.get_activated_nodes(threshold=min_score)

        # ── Store in cache ──
        self._ppr_cache.set(code_state_hash, seed_frozenset, scores)
        self._f._log_debug(f"PPR: computed and cached ({self._ppr_cache.stats})")

        # Log stats periodically (every 50th computation)
        if hasattr(self._f, "_write_counter") and self._f._write_counter % 50 == 0:
            self._f._log_debug(self._ppr_cache.stats)

        return scores

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Activation graph builders
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_single_seed_graph(
        self,
        exact_seeds: List[str],
        partial_seeds: List[str],
        tb_seeds: List[Tuple[str, float]],
        history_boosts: Dict[str, float],
        edges_out: dict,
        project_id: str,
        inferred_seeds: Optional[Dict[str, float]] = None,
        cached_scores: Optional[Dict[str, float]] = None,  # ← Q2
    ) -> "ActivationGraph":
        """Build activation graph when multi‑seed activation is disabled.

        Each bare‑name seed is split across its qualified id(s) before being
        written into the graph, since edges_out is now keyed by qualified id.

        Q2: If cached_scores is provided, skip propagation and load scores directly.
        """
        symbol_index = self._f._symbol_index
        ag = ActivationGraph()
        lexical_seed_qids: Set[str] = set()

        # ── Lexical + traceback + history seeds ──
        if exact_seeds:
            for sym_name in exact_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(1.0, 0.5 + 0.5 * min(specificity, 1.0))
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                lexical_seed_qids.update(qids)
                share = score / len(qids) if qids else 0.0
                for qid in qids:
                    ag._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
        if partial_seeds:
            for sym_name in partial_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.6, 0.3 + 0.3 * min(specificity, 1.0))
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                lexical_seed_qids.update(qids)
                share = score / len(qids) if qids else 0.0
                for qid in qids:
                    ag._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
        for sym_name, tb_score in tb_seeds:
            qids = symbol_index.get_qualified_names_for(sym_name, project_id)
            lexical_seed_qids.update(qids)
            share = tb_score / len(qids) if qids else 0.0
            for qid in qids:
                existing = ag._activations.get(qid)
                if existing:
                    ag._activations[qid] = ActivationState(
                        node_id=qid,
                        score=min(1.0, existing.score + share * 0.4),
                        depth=0,
                        source="seed",
                    )
                else:
                    ag._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
        for sym_name, boost in history_boosts.items():
            qids = symbol_index.get_qualified_names_for(sym_name, project_id)
            lexical_seed_qids.update(qids)
            share = boost / len(qids) if qids else 0.0
            for qid in qids:
                existing = ag._activations.get(qid)
                if existing:
                    ag._activations[qid] = ActivationState(
                        node_id=qid,
                        score=min(1.0, existing.score + share),
                        depth=0,
                        source=existing.source,
                    )
                else:
                    ag._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )

        # ── Inferred seeds ──
        for qid, inf_score in (inferred_seeds or {}).items():
            lexical_seed_qids.add(qid)
            existing = ag._activations.get(qid)
            if existing:
                ag._activations[qid] = ActivationState(
                    node_id=qid,
                    score=min(1.0, max(existing.score, inf_score)),
                    depth=0,
                    source="seed",
                )
            else:
                ag._activations[qid] = ActivationState(
                    node_id=qid,
                    score=inf_score,
                    depth=0,
                    source="seed",
                )

        # ── Q2: Use cached scores if available, otherwise run PPR ──
        if cached_scores is not None:
            # Populate from cache (but preserve seed status for seeds that were already set)
            for qid, score in cached_scores.items():
                if score >= 0.01:
                    existing = ag._activations.get(qid)
                    final_score = max(score, existing.score) if existing else score
                    ag._activations[qid] = ActivationState(
                        node_id=qid,
                        score=final_score,
                        depth=0,
                        source="seed" if qid in lexical_seed_qids else "propagation",
                    )
            self._f._log_debug(f"PPR: loaded {len(cached_scores)} scores from cache")
        else:
            # ── Fallback to entry points if no seeds at all ──
            if not ag._activations:
                pstate = self._f._project_state_manager.get_pstate(project_id)
                centrality = pstate.get("node_centrality", {})
                entry_points = self._f._path_index.find_entry_points(
                    symbol_index, project_id
                )
                if entry_points:
                    sorted_eps = sorted(
                        entry_points,
                        key=lambda ep: centrality.get(ep, 0.0),
                        reverse=True,
                    )
                    for sym_name in sorted_eps[:3]:
                        cent_score = centrality.get(sym_name, 0.0)
                        seed_score = 0.2 + 0.2 * cent_score
                        ag._activations[sym_name] = ActivationState(
                            node_id=sym_name, score=seed_score, depth=0, source="seed"
                        )
                        lexical_seed_qids.add(sym_name)

            # Run PPR propagation
            ag.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.05,
                alpha=self._f.valves.ppr_alpha,
            )
        return ag

    def _build_multi_seed_graph(
        self,
        exact_seeds: List[str],
        partial_seeds: List[str],
        tb_seeds: List[Tuple[str, float]],
        history_boosts: Dict[str, float],
        edges_out: dict,
        project_id: str,
        inferred_seeds: Optional[Dict[str, float]] = None,
        cached_scores: Optional[Dict[str, float]] = None,  # ← Q2
    ) -> "ActivationGraph":
        """Build activation graph combining lexical, structural and historical
        seed vectors.

        Q2: If cached_scores is provided, use it directly instead of recomputing.
        """
        w_lex = self._f.valves.multi_seed_weight_lexical
        w_str = self._f.valves.multi_seed_weight_structural
        w_his = self._f.valves.multi_seed_weight_historical
        symbol_index = self._f._symbol_index

        # ── Vector 1: Lexical ──────────────────────────────────────────
        ag_lex = ActivationGraph()
        lexical_seed_qids: Set[str] = set()
        if exact_seeds:
            for sym_name in exact_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(1.0, 0.5 + 0.5 * min(specificity, 1.0))
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                lexical_seed_qids.update(qids)
                share = score / len(qids) if qids else 0.0
                for qid in qids:
                    ag_lex._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
        if partial_seeds:
            for sym_name in partial_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.6, 0.3 + 0.3 * min(specificity, 1.0))
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                lexical_seed_qids.update(qids)
                share = score / len(qids) if qids else 0.0
                for qid in qids:
                    ag_lex._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
        for sym_name, tb_score in tb_seeds:
            qids = symbol_index.get_qualified_names_for(sym_name, project_id)
            lexical_seed_qids.update(qids)
            share = tb_score / len(qids) if qids else 0.0
            for qid in qids:
                existing = ag_lex._activations.get(qid)
                if existing:
                    ag_lex._activations[qid] = ActivationState(
                        node_id=qid,
                        score=min(1.0, existing.score + share * 0.4),
                        depth=0,
                        source="seed",
                    )
                else:
                    ag_lex._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )

        # ── Inferred seeds → lexical vector ──
        for qid, inf_score in (inferred_seeds or {}).items():
            lexical_seed_qids.add(qid)
            existing = ag_lex._activations.get(qid)
            ag_lex._activations[qid] = ActivationState(
                node_id=qid,
                score=min(1.0, max(existing.score if existing else 0.0, inf_score)),
                depth=0,
                source="seed",
            )

        if ag_lex._activations:
            ag_lex.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.03,
                alpha=self._f.valves.ppr_alpha,
            )

        # ── Vector 2: Structural ───────────────────────────────────────
        ag_str = ActivationGraph()
        seed_qids_for_structural = set(lexical_seed_qids)
        structural_seeds: Set[str] = set()
        for view in self._f._path_index.get_all(project_id):
            for lex_seed in seed_qids_for_structural:
                if lex_seed in view.induced_nodes:
                    structural_seeds.add(view.entry_point)
                    break
        if structural_seeds:
            for sym_name in structural_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.8, 0.5 * min(specificity, 1.4))
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                share = score / len(qids) if qids else 0.0
                for qid in qids:
                    ag_str._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
            ag_str.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.03,
                alpha=self._f.valves.ppr_alpha,
            )

        # ── Vector 3: Historical ───────────────────────────────────────
        ag_his = ActivationGraph()
        if history_boosts:
            for sym_name, boost in history_boosts.items():
                qids = symbol_index.get_qualified_names_for(sym_name, project_id)
                share = boost / len(qids) if qids else 0.0
                for qid in qids:
                    ag_his._activations[qid] = ActivationState(
                        node_id=qid, score=share, depth=0, source="seed"
                    )
            ag_his.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.03,
                alpha=self._f.valves.ppr_alpha,
            )

        # ── Q2: If cached_scores is provided, combine them directly ──
        if cached_scores is not None:
            ag_final = ActivationGraph()
            # Use cached scores as the base
            for qid, score in cached_scores.items():
                if score >= 0.01:
                    source = "seed" if qid in lexical_seed_qids else "propagation"
                    ag_final._activations[qid] = ActivationState(
                        node_id=qid,
                        score=score,
                        depth=0,
                        source=source,
                    )
            self._f._log_debug(f"PPR: loaded {len(cached_scores)} scores from cache")
            return ag_final

        # ── No cached scores: combine the three vectors ──
        all_activated = (
            set(ag_lex.get_activated_nodes(0.01).keys())
            | set(ag_str.get_activated_nodes(0.01).keys())
            | set(ag_his.get_activated_nodes(0.01).keys())
        )

        ag_final = ActivationGraph()
        if not all_activated:
            pstate = self._f._project_state_manager.get_pstate(project_id)
            centrality = pstate.get("node_centrality", {})
            entry_points = self._f._path_index.find_entry_points(
                symbol_index, project_id
            )
            if entry_points:
                sorted_eps = sorted(
                    entry_points,
                    key=lambda ep: centrality.get(ep, 0.0),
                    reverse=True,
                )
                for sym_name in sorted_eps[:3]:
                    cent_score = centrality.get(sym_name, 0.0)
                    seed_score = 0.2 + 0.2 * cent_score
                    ag_final._activations[sym_name] = ActivationState(
                        node_id=sym_name, score=seed_score, depth=0, source="seed"
                    )
        else:
            for node in all_activated:
                combined = (
                    w_lex * ag_lex.get_score(node)
                    + w_str * ag_str.get_score(node)
                    + w_his * ag_his.get_score(node)
                )
                if combined >= 0.01:
                    source = "seed" if node in lexical_seed_qids else "propagation"
                    depth = min(
                        ag_lex._activations.get(
                            node,
                            ActivationState(
                                node_id=node, score=0, depth=99, source="propagation"
                            ),
                        ).depth,
                        ag_str._activations.get(
                            node,
                            ActivationState(
                                node_id=node, score=0, depth=99, source="propagation"
                            ),
                        ).depth,
                    )
                    ag_final._activations[node] = ActivationState(
                        node_id=node,
                        score=min(1.0, combined),
                        depth=depth,
                        source=source,
                    )

        activated_count = len(ag_final.get_activated_nodes(threshold=0.05))
        self._f._log_debug(
            f"Multi-seed ActivationGraph: {activated_count} nodes activated "
            f"(lex={len(ag_lex.get_activated_nodes(0.01))}, "
            f"str={len(ag_str.get_activated_nodes(0.01))}, "
            f"his={len(ag_his.get_activated_nodes(0.01))})"
        )

        return ag_final

    def _store_activation_scores(self, ag: ActivationGraph, project_id: str) -> None:
        """Save activation scores for speculative prefetch and LOD tracking."""
        activated = ag.get_activated_nodes(
            threshold=self._f.valves.path_activation_threshold
        )
        if not hasattr(self._f, "_last_activation_scores"):
            self._f._last_activation_scores: Dict[str, Dict[str, float]] = {}
        self._f._last_activation_scores[project_id] = activated

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Main entry point: build_activation_graph (with Q2 cache + E7 integration)
    # ═══════════════════════════════════════════════════════════════════════════

    def build_activation_graph(
        self,
        query: str,
        project_id: str,
        max_propagation_steps: int = 4,
        messages: Optional[List[dict]] = None,
        inferred_seeds: Optional[Dict[str, float]] = None,
    ) -> "ActivationGraph":
        """
        Build an ActivationGraph combining up to three independent seed vectors.

        Args:
            query: The user query string.
            project_id: Current project identifier.
            max_propagation_steps: Steps for PPR propagation.
            messages: Recent conversation messages (for historical seeds).
            inferred_seeds: Optional {qid: score} from LLM‑guided inference.

        Returns:
            ActivationGraph with propagated scores.

        Q2: Integrates PPR cache to avoid recomputation when seeds and code unchanged.
        E7: Normalizes PPR scores to [0,1] within the project distribution.
        """
        self._f._log_debug(
            f"[PPR] build_activation_graph: query='{query[:100]}', "
            f"project_id='{project_id}', "
            f"max_steps={max_propagation_steps}, "
            f"has_messages={bool(messages)}"
        )

        edges_out = self._f._symbol_index.get_all_edges_out(project_id)

        # 1. Extract all seed symbols from the query and history
        exact_seeds, partial_seeds, tb_seeds, history_boosts = (
            self._prepare_seed_symbols(query, project_id, messages)
        )

        self._f._log_debug(
            f"[PPR] Seeds extracted: exact={len(exact_seeds)} ({exact_seeds[:5] if exact_seeds else 'none'}), "
            f"partial={len(partial_seeds)} ({partial_seeds[:5] if partial_seeds else 'none'}), "
            f"tb={len(tb_seeds)}, history={len(history_boosts)}"
        )

        # ── Q2: Build seed_qids list for cache key ──
        seed_qids: List[str] = []
        all_qids = self._f._symbol_index.get_all_qualified_names(project_id)

        for sym in exact_seeds:
            qids = self._f._symbol_index.get_qualified_names_for(sym, project_id)
            seed_qids.extend(qids)
        for sym in partial_seeds:
            qids = self._f._symbol_index.get_qualified_names_for(sym, project_id)
            seed_qids.extend(qids)
        for sym, _ in tb_seeds:
            qids = self._f._symbol_index.get_qualified_names_for(sym, project_id)
            seed_qids.extend(qids)
        if inferred_seeds:
            seed_qids.extend(inferred_seeds.keys())
        # Deduplicate and filter to only existing qids
        seed_qids = list(set(seed_qids) & set(all_qids))

        # ── Q2: Get code_state_hash for cache invalidation ──
        code_state_hash = self.compute_code_state_hash(project_id)

        # ── Q2: Get cached PPR scores or compute ──
        cached_scores = self._get_or_compute_ppr_scores(
            seed_qids=seed_qids,
            project_id=project_id,
            code_state_hash=code_state_hash,
            edges_out=edges_out,
            max_steps=max_propagation_steps,
            min_score=0.05,
            alpha=self._f.valves.ppr_alpha,
        )

        # 2. Build the activation graph using cached scores if available
        _inferred = inferred_seeds or {}
        if not self._f.valves.enable_multi_seed_activation:
            self._f._log_debug("[PPR] Using SINGLE-SEED activation mode")
            ag = self._build_single_seed_graph(
                exact_seeds,
                partial_seeds,
                tb_seeds,
                history_boosts,
                edges_out,
                project_id,
                _inferred,
                cached_scores=cached_scores,  # ← Q2
            )
        else:
            self._f._log_debug("[PPR] Using MULTI-SEED activation mode")
            ag = self._build_multi_seed_graph(
                exact_seeds,
                partial_seeds,
                tb_seeds,
                history_boosts,
                edges_out,
                project_id,
                _inferred,
                cached_scores=cached_scores,  # ← Q2
            )

        # 3. Store scores for downstream consumers (LOD, prefetch, pager)
        self._store_activation_scores(ag, project_id)

        # ── E7: normalize scores before returning ──────────────────────────
        activated = ag.get_activated_nodes(
            threshold=self._f.valves.path_activation_threshold
        )
        activated = self._normalize_ppr_scores(activated)

        self._f._log_debug(
            f"[PPR] Activation complete: {len(activated)} nodes activated "
            f"(threshold={self._f.valves.path_activation_threshold})"
        )

        return ag

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Path index & cross‑chunk edges
    # ═══════════════════════════════════════════════════════════════════════════

    async def _build_view_from_activation(
        self, entry_point: str, ag: ActivationGraph, project_id: str
    ) -> Optional[CodePathView]:
        """Build a CodePathView from an ActivationGraph and register it in PathIndex."""
        edges_out = self._f._symbol_index.get_all_edges_out(project_id)
        edges_in_map: Dict[str, List[Edge]] = defaultdict(list)
        for sym, edge_list in edges_out.items():
            for e in edge_list:
                edges_in_map[e.dst].append(e)

        extractor = SubgraphExtractor(
            activation_threshold=self._f.valves.path_activation_threshold,
            expand_hops=1,
        )
        induced_nodes_set, induced_edges = extractor.extract(
            ag, edges_out, edges_in_map
        )

        if not induced_nodes_set:
            return None

        induced_nodes_scored = {node: ag.get_score(node) for node in induced_nodes_set}

        path_id = hashlib.md5(
            f"{entry_point}|{'|'.join(sorted(induced_nodes_set))}".encode()
        ).hexdigest()[:16]

        existing = self._f._path_index.get(path_id, project_id)
        structural_hash = self.compute_structural_hash(induced_nodes_set, project_id)
        call_graph_hash = self.compute_call_graph_hash(induced_nodes_set, project_id)

        view = CodePathView(
            path_id=path_id,
            entry_point=entry_point,
            seed_nodes=[entry_point],
            induced_nodes=induced_nodes_scored,
            induced_edges=induced_edges,
            activation_score=ag.aggregate_path_score(list(induced_nodes_set)),
            structural_hash=structural_hash,
            call_graph_hash=call_graph_hash,
        )

        if (
            existing
            and not existing.is_stale(structural_hash, call_graph_hash)
            and existing.business_label
        ):
            view.business_label = existing.business_label
            view.summary = existing.summary
            view.label_confidence = existing.label_confidence

        self._f._path_index.add(view, project_id)
        return view

    async def rebuild_path_index(self, project_id: str) -> None:
        """Reconstruct PathIndex from SymbolIndex for all entry points."""
        state = self._f._conversation_state_manager.get(project_id)
        if not state or not state.active_blocks:
            return
        entry_points = self._f._path_index.find_entry_points(
            self._f._symbol_index, project_id
        )
        for ep in entry_points:
            ag = self.build_activation_graph(ep, project_id)
            await self._build_view_from_activation(ep, ag, project_id)

        if self._f.valves.enable_centrality_prior:
            pstate = self._f._project_state_manager.get_pstate(project_id)
            pstate["node_centrality"] = self._f._symbol_index.precompute_centrality(
                project_id
            )

    async def resolve_dangling_edges(self, project_id: str) -> int:
        """
        Resolve cross-chunk symbol references.

        A 'dangling edge' is an edge whose destination is referenced in the
        call graph but has no code block yet. When a new chunk defines that
        symbol, the edge confidence is raised from 0.3 (provisional) to 1.0.

        Conversely, edges pointing to symbols that are referenced but not
        defined are marked with confidence 0.3.

        Returns the number of edges resolved.
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        resolved = 0

        for sym_name in all_names:
            has_definition = bool(
                self._f._symbol_index.find_blocks(sym_name, project_id)
            )
            edges_in = self._f._symbol_index.get_edges_in(sym_name, project_id)

            for edge in edges_in:
                if has_definition and edge.confidence < 1.0:
                    edge.confidence = 1.0
                    resolved += 1
                elif not has_definition and edge.confidence == 1.0:
                    edge.confidence = 0.3

        if resolved > 0:
            self._f._log_debug(
                f"Cross-chunk resolution: {resolved} edge(s) resolved "
                f"(references confirmed with definitions)"
            )
        return resolved

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Speculative prefetch
    # ═══════════════════════════════════════════════════════════════════════════

    async def speculative_prefetch(
        self,
        project_id: str,
        last_activated: dict,
    ) -> None:
        """
        Pre‑build CodePathViews for symbols likely to be relevant in the next query.

        Prediction: high‑confidence direct callees of the top‑N activated symbols.
        """
        if not self._f.valves.enable_speculative_prefetch:
            return
        if not last_activated:
            return

        top_syms = sorted(last_activated, key=last_activated.get, reverse=True)[:3]

        prefetch_candidates: Set[str] = set()
        for sym in top_syms:
            for edge in self._f._symbol_index.get_edges_out(sym, project_id):
                if (
                    edge.type == "calls"
                    and edge.effective_weight() >= 0.7
                    and edge.dst not in last_activated
                ):
                    prefetch_candidates.add(edge.dst)

        if not prefetch_candidates:
            return

        candidates = list(prefetch_candidates)[
            : self._f.valves.speculative_prefetch_max
        ]
        self._f._log_debug(
            f"Speculative prefetch: pre-building {len(candidates)} CodePathView(s) "
            f"for next likely query"
        )

        edges_out = self._f._symbol_index.get_all_edges_out(project_id)
        for sym_name in candidates:
            if not self._f._symbol_index.find_blocks(sym_name, project_id):
                continue
            qids = self._f._symbol_index.get_qualified_names_for(sym_name, project_id)
            ag = ActivationGraph()
            ag.seed(list(qids), initial_score=1.0)
            ag.propagate(edges_out, max_steps=2, min_score=0.1)
            await self._build_view_from_activation(sym_name, ag, project_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Hash utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_structural_hash(
        self, symbol_names: Iterable[str], project_id: str
    ) -> str:
        """Hash of the symbols' content blocks (changes when code changes)."""
        state = self._f._conversation_state_manager.get(project_id)
        hashes = []
        for name in sorted(symbol_names):
            for bh in sorted(self._f._symbol_index.find_blocks(name, project_id)):
                hashes.append(bh)
        return hashlib.md5("|".join(hashes).encode()).hexdigest()[:16] if hashes else ""

    def compute_call_graph_hash(
        self, symbol_names: Iterable[str], project_id: str
    ) -> str:
        """Hash of the call relationships (changes when the graph changes)."""
        edge_strs = []
        for name in sorted(symbol_names):
            for edge in self._f._symbol_index.get_edges_out(name, project_id):
                edge_strs.append(f"{edge.src}:{edge.type}:{edge.dst}")
        return (
            hashlib.md5("|".join(sorted(edge_strs)).encode()).hexdigest()[:16]
            if edge_strs
            else ""
        )

    def compute_code_state_hash(self, project_id: str) -> str:
        """Return a hash that changes when the set of active blocks changes."""
        pstate = self._f._project_state_manager.get_pstate(project_id)
        cached = pstate.get("cached_code_state_hash")
        if cached is not None:
            return cached
        state = self._f._conversation_state_manager.get(project_id)
        h = self._compute_code_state_hash_from_state(state)
        pstate["cached_code_state_hash"] = h
        return h

    def _compute_code_state_hash_from_state(self, state: dict) -> str:
        if not state or not state.active_blocks:
            return ""
        sorted_hashes = sorted(
            h for h, b in state.active_blocks.items() if not b.obsolete
        )
        return hashlib.md5("|".join(sorted_hashes).encode()).hexdigest()[:16]

    def _normalize_ppr_scores(
        self,
        raw_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalize PPR scores to [0, 1] within the project's distribution.

        E7: prevents threshold miscalibration across projects of different sizes.
        After normalization, lod2_threshold=0.10 means "top 10% by PPR rank",
        regardless of how many symbols the project has.
        """
        if not raw_scores:
            return raw_scores
        min_s = min(raw_scores.values())
        max_s = max(raw_scores.values())
        if max_s <= min_s:
            # All scores equal (e.g. isolated graph) — normalize to 0.5
            return {qid: 0.5 for qid in raw_scores}
        return {
            qid: (score - min_s) / (max_s - min_s) for qid, score in raw_scores.items()
        }

    def compute_context_hash(self, messages: list) -> str:
        """Hash of system message content, used for response cache keying."""
        if not self._f.valves.response_cache_include_context_hash:
            return ""
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        context_str = "\n".join([m.get("content", "") for m in sys_msgs])
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. Inactive block candidates & cache invalidation
    # ═══════════════════════════════════════════════════════════════════════════

    def get_inactive_block_candidates(self, project_id: str) -> list:
        """
        Return block hashes that haven't been mentioned in the last
        `cleanup_inactive_threshold_messages` messages, excluding pinned,
        obsolete, and content types listed in `cleanup_excluded_content_types`.
        """
        state = self._f._conversation_state_manager.get(project_id)
        if not state or not state.active_blocks:
            return []
        threshold = self._f.valves.cleanup_inactive_threshold_messages
        excluded_types = set(self._f.valves.cleanup_excluded_content_types)
        current_msg_idx = state.message_count
        candidates = []
        for h, block in state.active_blocks.items():
            if block.pinned or block.obsolete:
                continue
            if block.content_type.value in excluded_types:
                continue
            last_idx = block.last_mentioned_msg_idx
            if last_idx is None:
                last_idx = current_msg_idx
            if current_msg_idx - last_idx > threshold:
                candidates.append(h)
        return candidates

    def invalidate_lightweight_cache(self, project_id: str) -> None:
        """Clear cached lightweight context and centrality so the next request rebuilds them."""
        pstate = self._f._project_state_manager.get_pstate(project_id)
        pstate["cached_lightweight_context"] = ""
        pstate["cached_code_state_hash"] = None
        pstate["node_centrality"] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # 10. Static evidence (for scientific CoT)
    # ═══════════════════════════════════════════════════════════════════════════

    def _gather_static_evidence(
        self, hypothesis_text: str, project_id: str
    ) -> "StaticEvidence":
        """
        Gather deterministic evidence about the structural claims in a hypothesis.
        No LLM. No GPU. Instant.
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        state = self._f._conversation_state_manager.get(project_id)

        words = set(re.findall(r"\b\w+\b", hypothesis_text))
        mentioned = all_names.intersection(words)

        symbols_found = {
            name: bool(self._f._symbol_index.find_blocks(name, project_id))
            for name in mentioned
        }

        call_patterns = re.findall(
            r"`?(\w+)`?\s+(?:calls?|invokes?|uses?|depends on)\s+`?(\w+)`?",
            hypothesis_text,
            re.IGNORECASE,
        )
        call_relations_valid = {}
        for caller, callee in call_patterns:
            if caller not in all_names or callee not in all_names:
                continue
            key = f"{caller}_calls_{callee}"
            caller_edges = self._f._symbol_index.get_edges_out(caller, project_id)
            verified = any(
                e.dst == callee and e.type in ("calls", "reads", "writes")
                for e in caller_edges
            )
            call_relations_valid[key] = verified

        now = time.time()
        recent_window = 3600
        recent_changes = [
            name
            for name in mentioned
            if any(
                state.active_blocks.get(bh) is not None
                and (now - state.active_blocks[bh].timestamp) < recent_window
                for bh in self._f._symbol_index.find_blocks(name, project_id)
            )
        ]

        all_views = self._f._path_index.get_all(project_id)
        entry_points_mentioned = [
            v.entry_point for v in all_views if v.entry_point in mentioned
        ]

        path_memberships: Dict[str, List[str]] = {}
        for name in mentioned:
            path_memberships[name] = self._f._path_index.mark_stale_for_symbol(
                name, project_id
            )

        data_flow_upstream: Dict[str, List[str]] = {}
        if mentioned:
            for sym_name in mentioned:
                incoming_edges = self._f._symbol_index.get_edges_in(
                    sym_name, project_id
                )
                data_flow_sources = [
                    e.src for e in incoming_edges if e.type == "data_flow"
                ]
                if data_flow_sources:
                    data_flow_upstream[sym_name] = data_flow_sources

        verifiable = len(symbols_found) + len(call_relations_valid)
        if verifiable == 0:
            objective_score = 0.5
        else:
            verified_true = sum(1 for v in symbols_found.values() if v) + sum(
                1 for v in call_relations_valid.values() if v
            )
            objective_score = verified_true / verifiable

        return StaticEvidence(
            symbols_found=symbols_found,
            call_relations_valid=call_relations_valid,
            recent_changes=recent_changes,
            entry_points_mentioned=entry_points_mentioned,
            path_memberships=path_memberships,
            data_flow_upstream=data_flow_upstream,
            objective_score=objective_score,
        )


class HistoryCompressor:
    """
    Compresses conversation history and code blocks to keep the token
    budget under control without losing essential information.

    Provides:
    * ``compress_code_history(messages, project_id)`` — replaces old
      multi‑phase code parts with compact commit summaries when their
      symbols are safely indexed in the SymbolGraph.
    * ``lean_user_code_messages(messages, project_id)`` — replaces large
      code blocks in user messages with stubs when the SymbolGraph already
      covers every symbol, avoiding redundant token consumption.
    * ``compress_code_block(code, language, rate, query)`` — applies
      LLMLingua‑2 compression to a single code block, preserving structural
      tokens and optionally conditioning on the user's question.
    * ``summarize_messages(old_messages, is_code_context)`` — produces a
      single‑paragraph summary of trimmed conversation turns.
    * ``build_refactor_state_injection(messages)`` — builds a compact
      "Refactor Status" block from compressed code parts so the model knows
      what has already been written.
    * ``schedule_block_summary(block, project_id)`` — fire‑and‑forget
      background task that generates a summary for an oversized code block.
    * ``check_and_suggest_summarization(project_id, ...)`` — returns a
      proactive suggestion when the conversation is approaching the token
      window limit.

    Docs 10–13 backported:
        B5 – force‑compressed keys survive restarts via persistent set.
    """

    # ── Q4: Commit summary prompt ──────────────────────────────────────────
    COMMIT_SUMMARY_PROMPT = """\
You are summarizing a coding decision for long-term context compression.
Respond ONLY with a JSON object, no preamble, no markdown fences.

{
  "action": "<verb phrase: what was done, ≤15 words>",
  "rationale": "<why this approach: the constraint, tradeoff, or requirement that drove the decision, ≤20 words, or null if not stated>",
  "symbols": ["<fully_qualified_symbol_1>", "<fully_qualified_symbol_2>"]
}

User message to summarize:
\"\"\"
{user_message}
\"\"\"

Code context (recent symbols referenced):
{symbols_context}
"""

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the HistoryCompressor with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        # block.hash -> in-flight background summary task, to avoid firing
        # a second summary attempt for the same block while one is pending.
        self._pending_block_summaries: Set[str] = set()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Code history compression (multi‑phase code parts → summaries)
    # ═══════════════════════════════════════════════════════════════════════

    async def compress_code_history(self, messages: list, project_id: str) -> list:
        """
        Replace old assistant code-part messages with compact commit summaries.

        Compression pipeline:
          1. Find all assistant messages with multi-phase code part headers.
          2. Keep the last `code_history_keep_last_n_parts` in full.
          3. For each older part, verify symbols are indexed in the SymbolGraph.
          4. If indexed (ratio >= threshold): replace with commit summary.
          5. If NOT indexed: increment blocked age; if age exceeds
             code_history_force_compress_after_turns OR the message was already
             force-compressed in a previous session, force compression WITHOUT
             an /expand guarantee (marked '[🗜️ CÓDIGO COMPRIMIDO — sin índice]').
        """
        if not self._f.valves.enable_code_history_compression:
            return messages

        # ── Patterns ─────────────────────────────────────────────────────────
        _PART_HEADER = re.compile(r"##\s*Código\s*[—\-]\s*Parte\s*(\d+)/(\d+)")
        _ALREADY_COMPRESSED = re.compile(r"\[🗜️ PARTE \d+/\d+")

        _PHASE_HEADER = re.compile(
            r"^(?:Fase|Parte|Phase|Step)\s*(\d+)\s*[:—\-]\s*(.+)$",
            re.IGNORECASE,
        )

        keep = self._f.valves.code_history_keep_last_n_parts
        force_after = self._f.valves.code_history_force_compress_after_turns

        # ── Load persistent state ───────────────────────────────────────────
        pstate = self._f._project_state_manager.get_pstate(project_id)
        state = self._f._conversation_state_manager.get(project_id)
        blocked_age = state.history_blocked_age  # mutable dict, modified in-place

        # ── B5: load persistent force‑compressed keys ──────────────────────
        force_compressed_keys: Set[str] = set(pstate.get("force_compressed_keys", []))

        dirty = False  # track whether we modified blocked_age or force_compressed_keys

        # ── Collect indices of uncompressed code-part messages ─────────────
        code_part_indices: List[Tuple[int, int, int]] = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if _ALREADY_COMPRESSED.search(content):
                continue

            m = _PART_HEADER.search(content)
            if m:
                code_part_indices.append((i, int(m.group(1)), int(m.group(2))))
                continue

            # Try the broader phase/part pattern
            if _PHASE_HEADER.search(content):
                est_tokens = self._f._tokens.estimate_code_tokens(content)
                if est_tokens > 300:
                    max_phase = 0
                    for msg2 in messages:
                        if msg2.get("role") != "assistant":
                            continue
                        m2 = _PHASE_HEADER.search(msg2.get("content", ""))
                        if m2:
                            max_phase = max(max_phase, int(m2.group(1)))
                    m_phase = _PHASE_HEADER.search(content)
                    if m_phase:
                        part_num = int(m_phase.group(1))
                        total_parts = max_phase if max_phase >= part_num else part_num
                        code_part_indices.append((i, part_num, total_parts))
                    else:
                        code_part_indices.append((i, 1, 1))

        if len(code_part_indices) <= keep:
            # No old parts to compress; still save dirty if modified
            if dirty:
                self._f._conversation_state_manager.mark_dirty(project_id)
                pstate["force_compressed_keys"] = list(force_compressed_keys)
            return messages

        to_compress = code_part_indices[:-keep]
        new_messages = list(messages)
        compressed_n = 0
        blocked_by_ratio = 0
        forced_no_expand = 0

        for msg_idx, part_num, total_parts in to_compress:
            msg = new_messages[msg_idx]
            content = msg.get("content", "")

            # ── Generate a stable key for this message ─────────────────────
            msg_key = hashlib.md5(f"{msg.get('role')}|{content}".encode()).hexdigest()[
                :16
            ]

            # ── Verify symbols are indexed ─────────────────────────────────
            safe, ratio = self._verify_code_symbols_indexed(content, project_id)

            force_no_expand = False

            if not safe:
                # Increment blocked age
                blocked_age[msg_key] = blocked_age.get(msg_key, 0) + 1
                dirty = True
                age = blocked_age[msg_key]

                # Log the block with more detail
                part_label_for_log = (
                    f"Part {part_num}/{total_parts}"
                    if total_parts > 0
                    else f"message {msg_idx}"
                )
                self._f._log_debug(
                    f"Code history: WANTED to compress {part_label_for_log} but "
                    f"BLOCKED — symbol ratio {ratio:.0%} < threshold "
                    f"{self._f.valves.code_history_symbol_index_threshold:.0%}. "
                    f"Blocked age: {age} turn(s)."
                )
                blocked_by_ratio += 1

                # ── B5: force compression if age exceeds threshold OR key already force-compressed ──
                if force_after > 0 and (
                    age >= force_after or msg_key in force_compressed_keys
                ):
                    self._f._log_debug(
                        f"Code history: FORCING compression of {part_label_for_log} "
                        f"(force_after={force_after}, age={age}, "
                        f"already_force={msg_key in force_compressed_keys}). "
                        f"Compressing WITHOUT /expand guarantee."
                    )
                    force_no_expand = True
                    safe = True  # Proceed with compression
                    forced_no_expand += 1
                else:
                    continue

            # If safe (either verified or forced), compress
            summary = await self._build_code_commit_summary(
                content,
                project_id,
                part_num,
                total_parts,
                force_no_expand=force_no_expand,
            )
            tokens_before = self._f._tokens.estimate_tokens(content)
            tokens_after = self._f._tokens.estimate_tokens(summary)
            new_messages[msg_idx] = {**msg, "content": summary}
            compressed_n += 1

            # ── B5: record that this message was force-compressed ──────────
            if force_no_expand:
                force_compressed_keys.add(msg_key)
                dirty = True

            # Reset blocked age for this message since it was successfully compressed
            if msg_key in blocked_age:
                del blocked_age[msg_key]
                dirty = True

            self._f._log_debug(
                f"Code history: compressed Part {part_num}/{total_parts} — "
                f"{tokens_before:,} → {tokens_after:,} tokens "
                f"(ratio {ratio:.0%})"
                + (" [FORCED NO-EXPAND]" if force_no_expand else "")
            )

        # ── B5: evict old force‑compressed keys ─────────────────────────────
        if len(force_compressed_keys) > 500:
            # Keep the most recent 250 (ordered by insertion, but set order is stable)
            force_compressed_keys = set(list(force_compressed_keys)[-250:])
            dirty = True

        # ── B5: persist force‑compressed keys ──────────────────────────────
        if dirty:
            self._f._conversation_state_manager.mark_dirty(project_id)
            pstate["force_compressed_keys"] = list(force_compressed_keys)

        if compressed_n:
            self._f._log_debug(
                f"Code history: {compressed_n} part(s) compressed, "
                f"last {keep} kept in full. "
                f"(blocked_by_ratio={blocked_by_ratio}, forced_no_expand={forced_no_expand})"
            )
        elif blocked_by_ratio > 0:
            self._f._log_debug(
                f"Code history: {blocked_by_ratio} part(s) blocked by ratio, "
                f"none compressed (force_after={force_after})."
            )

        return new_messages

    def _verify_code_symbols_indexed(
        self, content: str, project_id: str
    ) -> Tuple[bool, float]:
        """
        Verify that code symbols in the content are indexed in the SymbolGraph.
        Returns (is_indexed, ratio).
        """
        _TOP_LEVEL = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
        expected = set(_TOP_LEVEL.findall(content))
        if not expected:
            _TOP_FN = re.compile(
                r"^(?:async )?def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
            )
            expected = set(_TOP_FN.findall(content))
        if not expected:
            return True, 1.0

        try:
            graph_symbols = self._f._symbol_index.get_all_names(project_id)
            ratio = len(expected & graph_symbols) / len(expected)
            return ratio >= self._f.valves.code_history_symbol_index_threshold, ratio
        except Exception as exc:
            self._f._log_debug(f"Symbol index check failed: {exc}")
            return False, 0.0

    # ── Q4: Commit summary builder with rationale ──────────────────────────

    def _parse_commit_summary_response(self, raw: str) -> dict:
        """
        Parse the LLM response for a commit summary.

        Strips markdown fences and any explanatory text before parsing JSON.
        Falls back to treating the whole response as the action if parsing fails.
        """
        try:
            # Strip potential reasoning blocks before parsing
            clean = re.sub(r"<details[^>]*>.*?</details>", "", raw, flags=re.DOTALL)
            clean = clean.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat the whole response as the action
            return {"action": raw[:100], "rationale": None, "symbols": []}

    def _render_commit_summary(self, summary: dict, turn: int) -> str:
        """
        Render a commit summary for insertion into compressed history.

        Format:
            [T{turn}] {action}
              Rationale: {rationale}
              Symbols: {symbols}
        """
        action = summary.get("action", "unknown action")
        rationale = summary.get("rationale")
        symbols = summary.get("symbols", [])

        lines = [f"[T{turn}] {action}"]
        if rationale:
            lines.append(f"  Rationale: {rationale}")
        if symbols:
            lines.append(f"  Symbols: {', '.join(symbols)}")

        return "\n".join(lines)

    async def _build_code_commit_summary(
        self,
        content: str,
        project_id: str,
        part_num: int,
        total_parts: int,
        force_no_expand: bool = False,
    ) -> str:
        """
        Generate a compact commit summary for a compressed code message.

        Q4: Uses LLM to produce structured summary including rationale.

        Args:
            content: The original code message content.
            project_id: The current project identifier.
            part_num: Part number in the multi-phase sequence.
            total_parts: Total number of parts in the sequence.
            force_no_expand: If True, omit /expand affordance and mark as
                non-indexed. Used when compression is forced despite the
                symbol-index ratio being too low.

        Returns:
            A formatted summary string.
        """
        # ── If forced no-expand, use the old format (no LLM call) ──
        if force_no_expand:
            return self._build_legacy_commit_summary(
                content, part_num, total_parts, force_no_expand=True
            )

        # ── Extract symbols from content ──────────────────────────────
        classes = re.findall(r"^class\s+([A-Za-z_]\w*)", content, re.MULTILINE)
        top_fns = re.findall(r"^(?:async )?def\s+([A-Za-z_]\w*)", content, re.MULTILINE)
        methods = re.findall(
            r"^\s{4,}(?:async )?def\s+([A-Za-z_]\w*)", content, re.MULTILINE
        )
        symbols = classes + top_fns + methods
        # Take up to 5 symbols
        symbols = symbols[:5]

        # ── Build the prompt ────────────────────────────────────────────
        prompt = self.COMMIT_SUMMARY_PROMPT.format(
            user_message=content[:2000],  # Truncate to avoid token overflow
            symbols_context=", ".join(symbols) if symbols else "none",
        )

        # ── Call LLM ────────────────────────────────────────────────────
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant. Output only valid JSON.",
            model_override=self._f.valves.summarization_model,
            max_tokens=250,
            temperature=0.2,
            label="commit_summary",
        )

        if not response:
            # Fallback to legacy summary
            return self._build_legacy_commit_summary(
                content, part_num, total_parts, force_no_expand=False
            )

        # ── Parse response ─────────────────────────────────────────────
        summary = self._parse_commit_summary_response(response)
        # Ensure action is not empty
        if not summary.get("action"):
            summary["action"] = "code change"
        # Add part info
        summary["part_num"] = part_num
        summary["total_parts"] = total_parts

        # ── Render with turn number (we use part_num as turn proxy) ──
        rendered = self._render_commit_summary(summary, turn=part_num)

        # ── If not fully indexed, add a note ────────────────────────────
        # (force_no_expand is handled above, but for normal case we check)
        return rendered

    def _build_legacy_commit_summary(
        self,
        content: str,
        part_num: int,
        total_parts: int,
        force_no_expand: bool = False,
    ) -> str:
        """
        Legacy commit summary builder (fallback when LLM is unavailable
        or when force_no_expand is True).
        """
        classes = re.findall(r"^class\s+([A-Za-z_]\w*)", content, re.MULTILINE)
        top_fns = re.findall(r"^(?:async )?def\s+([A-Za-z_]\w*)", content, re.MULTILINE)
        methods = re.findall(
            r"^\s{4,}(?:async )?def\s+([A-Za-z_]\w*)", content, re.MULTILINE
        )

        code_bodies = re.findall(r"```(?:\w*)\n(.*?)```", content, re.DOTALL)
        code_lines = sum(b.count("\n") for b in code_bodies)
        code_tokens = (
            self._f._tokens.estimate_tokens("\n".join(code_bodies))
            if code_bodies
            else 0
        )

        header_m = re.search(
            r"##\s*Código\s*[—\-]\s*Parte\s*\d+/\d+[:\s]+(.+?)(?:\n|$)", content
        )
        part_label = header_m.group(1).strip() if header_m else ""

        imports = re.findall(r"^from\s+\S+\s+import\s+(.+)$", content, re.MULTILINE)
        dep_symbols: List[str] = []
        for imp in imports[:3]:
            dep_symbols.extend(s.strip() for s in imp.split(","))
        dep_symbols = dep_symbols[:6]

        if force_no_expand:
            title = f"[🗜️ CÓDIGO COMPRIMIDO — sin índice]"
            lines = [title]
            if classes:
                lines.append(f"Clases:    {', '.join(classes)}")
            if top_fns:
                lines.append(f"Funciones: {', '.join(top_fns[:6])}")
            if methods:
                lines.append(f"Métodos:   {len(methods)} implementados")
            lines.append(f"Volumen:   ~{code_lines} líneas / ~{code_tokens} tokens")
            if dep_symbols:
                lines.append(f"Deps usadas: {', '.join(dep_symbols)}")
            lines.append(
                "[CÓDIGO NO INDEXADO — la implementación no es recuperable via /expand]"
            )
            return "\n".join(lines)

        title = f"[🗜️ PARTE {part_num}/{total_parts}"
        if part_label:
            title += f": {part_label}"
        title += " — COMPRIMIDO]"

        lines = [title]
        if classes:
            lines.append(f"Clases:    {', '.join(classes)}")
        if top_fns:
            lines.append(f"Funciones: {', '.join(top_fns[:6])}")
        if methods:
            lines.append(f"Métodos:   {len(methods)} implementados")
        lines.append(f"Volumen:   ~{code_lines} líneas / ~{code_tokens} tokens")
        if dep_symbols:
            lines.append(f"Deps usadas: {', '.join(dep_symbols)}")
        if classes:
            lines.append(f"Recuperar: /expand {classes[0]}")
        lines.append(
            "[Todos los símbolos indexados en SymbolGraph — accesibles via LOD]"
        )

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Lean user code compression (replace large user code blocks with stubs)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_user_stub(self, symbol_count: int) -> str:
        """
        Generate a stub message for a large user code block.

        This stub is used both during silent ingestion and in the normal
        message compression flow. It informs the model that the code is
        already indexed and available via /expand, avoiding redundant
        token consumption.

        Args:
            symbol_count (int): Number of symbols currently indexed in the SymbolGraph.

        Returns:
            str: The stub text, ready for injection into the conversation history.
        """
        return (
            f"_(The code is internally available; no need to repeat it here.)_\n\n"
            f"_[{symbol_count} symbols indexed in SymbolGraph. Use /expand <name> to see any implementation.]_"
        )

    def ensure_compressed_user_messages(
        self,
        messages: list,
        state: ConversationState,
        project_id: str,
    ) -> list:
        """
        Replace long user messages with compressed stubs, using persistent state.

        This method is called during message assembly (in MessageAssembler)
        and ensures that any user message exceeding lean_user_code_min_tokens
        is replaced by a compact stub. The stub is stored in ConversationState
        under compressed_user_messages, keyed by MD5 hash of the original content.

        If a stub already exists for a given message, it is reused. If not,
        a new stub is generated, stored, and the state is marked dirty for
        persistence in SQLite.

        Args:
            messages (list): The list of conversation messages (dicts).
            state (ConversationState): The persistent state for the current project.
            project_id (str): The current project identifier.

        Returns:
            list: The updated list of messages, with long user messages replaced by stubs.
        """
        if not self._f.valves.enable_lean_user_code:
            return messages

        min_tokens = self._f.valves.lean_user_code_min_tokens

        # Avoid compressing when the symbol index is too sparse (stub would be misleading)
        symbol_count = len(self._f._symbol_index.get_all_names(project_id))
        if symbol_count < 20:
            return messages

        new_messages = []
        for msg in messages:
            if msg.get("role") != "user":
                new_messages.append(msg)
                continue

            content = msg.get("content", "")

            # Skip messages that are already short enough
            if self._f._tokens.estimate_code_tokens(content) < min_tokens:
                new_messages.append(msg)
                continue

            # Compute a stable hash of the original content
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            # Reuse existing stub if available
            if content_hash in state.compressed_user_messages:
                stub = state.compressed_user_messages[content_hash]
                new_messages.append({**msg, "content": stub})
                continue

            # Generate new stub (no truncation needed – stub is naturally short)
            stub = self._build_user_stub(symbol_count)

            # Store in state and mark dirty for persistence
            state.compressed_user_messages[content_hash] = stub
            self._f._conversation_state_manager.mark_dirty(project_id)

            # Replace the message content with the stub
            new_messages.append({**msg, "content": stub})

        return new_messages

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Refactor state injection
    # ═══════════════════════════════════════════════════════════════════════

    def build_refactor_state_injection(self, messages: list) -> str:
        """
        Build a compact "Estado del Refactor" block from compressed code parts
        in conversation history.

        Injected into Block B when active multi-phase compression is detected.
        Gives the model awareness of what has been written without reading full messages.

        Returns empty string if no compressed parts found (no injection needed).
        """
        _COMPRESSED = re.compile(r"\[🗜️ PARTE (\d+)/(\d+)(?:: (.+?))? — COMPRIMIDO\]")

        completed: List[Tuple[int, str]] = []
        total_parts: Optional[int] = None

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            m = _COMPRESSED.search(msg.get("content", ""))
            if m:
                num = int(m.group(1))
                total = int(m.group(2))
                label = m.group(3) or f"Parte {num}"
                completed.append((num, label))
                total_parts = total

        if not completed or not total_parts:
            return ""

        done_nums = {n for n, _ in completed}
        pending = [i for i in range(1, total_parts + 1) if i not in done_nums]

        lines = ["## 📊 Estado del Refactor en Progreso"]
        for num, label in sorted(completed):
            lines.append(
                f"  ✓ Parte {num}/{total_parts}: {label} [indexado en SymbolGraph]"
            )
        for num in pending:
            lines.append(f"  ⏳ Parte {num}/{total_parts}: [pendiente]")
        lines.append(
            "Los símbolos de las partes completadas están disponibles via LOD "
            "(firmas en este prompt) o /expand para implementación completa."
        )

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. LLMLingua‑2 code block compression
    # ═══════════════════════════════════════════════════════════════════════

    async def compress_code_block(
        self,
        code: str,
        language: str = "python",
        rate: float = 0.5,
        query: str = "",
    ) -> str:
        """
        Compress a code block using LLMLingua-2.
        Preserves critical language structural tokens.

        If `query` is not empty and enable_question_aware_compression=True,
        uses LongLLMLingua conditioned on the query: the compressor preserves
        tokens relevant to the specific question.

        rate: fraction of tokens to KEEP (0.5 = keep 50%).
        Recommended range: 0.4 (aggressive) to 0.7 (conservative).

        Returns the original code if the compressor is unavailable
        or the block is below the minimum token threshold.
        """
        _raw = self._f._conv_compressor.raw if self._f._conv_compressor else None
        if not _raw:
            return code

        estimated_tokens = self._f._tokens.estimate_code_tokens(code)
        if estimated_tokens < self._f.valves.code_compression_min_tokens:
            return code

        FORCE_TOKENS_BY_LANG = {
            "python": [
                "\n",
                ":",
                "def ",
                "class ",
                "return ",
                "import ",
                "from ",
                "if ",
                "else:",
                "elif ",
                "for ",
                "while ",
                "try:",
                "except",
                "with ",
                "async ",
                "await ",
            ],
            "javascript": [
                "\n",
                ":",
                "function ",
                "const ",
                "let ",
                "var ",
                "return ",
                "class ",
                "import ",
                "export ",
                "=>",
                "if ",
                "else ",
                "for ",
                "while ",
            ],
            "typescript": [
                "\n",
                ":",
                "function ",
                "const ",
                "let ",
                "var ",
                "return ",
                "class ",
                "import ",
                "export ",
                "=>",
                "interface ",
                "type ",
                "if ",
                "else ",
                "for ",
            ],
            "go": [
                "\n",
                ":",
                "func ",
                "type ",
                "struct ",
                "import ",
                "return ",
                "if ",
                "else ",
                "for ",
                "package ",
            ],
            "rust": [
                "\n",
                ":",
                "fn ",
                "struct ",
                "impl ",
                "use ",
                "return ",
                "if ",
                "else ",
                "for ",
                "let ",
                "pub ",
            ],
        }
        force_tokens = FORCE_TOKENS_BY_LANG.get(
            language.lower(), ["\n", ":", "return "]
        )

        try:
            compress_kwargs = {
                "rate": rate,
                "force_tokens": force_tokens,
                "force_reserve_digit": True,
            }
            if query and self._f.valves.enable_question_aware_compression:
                compress_kwargs["question"] = query[:300]
                self._f._log_debug(
                    f"LLMLingua-2: question-aware mode active "
                    f"(query={query[:60]}...)"
                )

            result = await anyio.to_thread.run_sync(
                lambda: _raw.compress_prompt(code, **compress_kwargs)
            )
            compressed = result.get("compressed_prompt", code)
            compressed_tokens = self._f._tokens.estimate_code_tokens(compressed)
            self._f._log_debug(
                f"LLMLingua-2: {estimated_tokens} → {compressed_tokens} tokens "
                f"({100*(1-compressed_tokens/max(estimated_tokens,1)):.0f}% reduction)"
            )
            return compressed
        except Exception as e:
            self._f._log_debug(f"LLMLingua-2 compression failed: {e} — using original")
            return code

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Message summarization (for trimmed old messages)
    # ═══════════════════════════════════════════════════════════════════════

    async def summarize_messages(
        self, old_messages: list, is_code_context: bool = False
    ) -> Optional[str]:
        """Summarise a list of old conversation messages into a single paragraph."""
        if not old_messages:
            return None
        combined = "\n".join(
            [m.get("content", "") for m in old_messages if m.get("content")]
        )
        if not combined.strip():
            return None
        prompt = (
            f"Summarize the following conversation segment, preserving key decisions "
            f"and code changes:\n\n{combined[:4000]}"
        )
        system_prompt = (
            "You produce concise summaries of technical conversations."
            if is_code_context
            else "You produce concise summaries."
        )
        summary = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model_override=self._f.valves.summarization_model,
            max_tokens=500,
            temperature=0.3,
        )
        return summary.strip() if summary else None

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Proactive summarization suggestion
    # ═══════════════════════════════════════════════════════════════════════

    async def check_and_suggest_summarization(
        self,
        project_id: str,
        total_tokens: int,
        max_tokens: int,
    ) -> Optional[str]:
        """Return a proactive suggestion when the conversation is using a high ratio of the token window."""
        if max_tokens <= 0:
            return None
        ratio = total_tokens / max_tokens
        if ratio > self._f.valves.proactive_summary_threshold:
            return (
                f"[CodeAware] The conversation is using {total_tokens}/{max_tokens} tokens "
                f"(≈{int(ratio*100)}%). Consider using `/forget` or asking me to summarize "
                f"older parts."
            )
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Oversized block summary generation (fire-and-forget)
    # ═══════════════════════════════════════════════════════════════════════

    async def schedule_block_summary(self, block: "CodeBlock", project_id: str) -> None:
        """
        Fire-and-forget oversized-block summary generation.

        Returns immediately. Does nothing if the block already has a summary,
        is under the size threshold, or a background attempt is already
        in flight for this exact block hash.
        """
        if block.block_summary or block.hash in self._pending_block_summaries:
            return
        tok = block._cached_token_count or (len(block.content) // 4)
        if (
            self._f.valves.max_code_block_tokens <= 0
            or tok <= self._f.valves.max_code_block_tokens
        ):
            return

        self._pending_block_summaries.add(block.hash)

        async def _run() -> None:
            try:
                await self.maybe_generate_block_summary(block)
                if block.block_summary:
                    # Mark state dirty so the next save_state_if_dirty() persists it.
                    self._f._conversation_state_manager.mark_dirty(project_id)
            except Exception as exc:
                self._f._log_debug(f"Background block summary failed: {exc}")
            finally:
                self._pending_block_summaries.discard(block.hash)

        asyncio.create_task(_run())

    async def maybe_generate_block_summary(self, block: "CodeBlock") -> None:
        """Generate a summary for an oversized code block when overflow action is 'summarize'."""
        if not (
            self._f.valves.max_code_block_tokens > 0
            and self._f.valves.code_block_overflow_action == "summarize"
        ):
            return
        tok = block._cached_token_count or (len(block.content) // 4)
        if tok <= self._f.valves.max_code_block_tokens:
            return
        if block.block_summary:
            return

        sig = block.symbols[0].signature if block.symbols else ""
        prompt = (
            f"Summarize the following code block in 3-5 sentences. "
            f"Cover: main purpose, key classes/functions, and important dependencies.\n\n"
            f"{'Signature: ' + sig + chr(10) if sig else ''}"
            f"```\n{block.content[:self._f.valves.summary_code_max_chars]}\n```"
        )
        summary = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant. Be concise and technical.",
            model_override=self._f.valves.code_block_summary_model,
            max_tokens=self._f.valves.oversized_summary_max_tokens,
            temperature=0.2,
            label="block_summary",
        )
        if summary:
            block.block_summary = summary.strip()
            self._f._log_debug(
                f"Block {block.hash[:8]}: summary generated "
                f"({tok} tokens > {self._f.valves.max_code_block_tokens} limit)"
            )


class TokenUtils:
    """Token‑level utilities for budget management and text truncation.

    Provides:
    * ``estimate_tokens(messages)`` — total token count for a list of
      message dicts (content + role overhead).
    * ``estimate_code_tokens(code)`` — token count for a raw code string.
    * ``truncate_text_to_tokens(text, max_tokens)`` — truncates text to
      approximately *max_tokens* while preserving word boundaries.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Token estimation
    # ═══════════════════════════════════════════════════════════════════════════

    def estimate_tokens(self, messages: list) -> int:
        """Estimate total token count for a list of messages."""
        if self._f.tokenizer:
            total = 0
            for m in messages:
                content = str(m.get("content", ""))
                total += len(self._f.tokenizer.encode(content))
                total += 4  # role and formatting overhead
            return total
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        total_chars += sum(30 for _ in messages)
        return total_chars // 4

    def estimate_code_tokens(self, code: str) -> int:
        """Estimate token count for a code string."""
        if self._f.tokenizer:
            return len(self._f.tokenizer.encode(code))
        return len(code) // 4

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Text truncation
    # ═══════════════════════════════════════════════════════════════════════════

    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens while preserving word boundaries."""
        if not self._f.tokenizer:
            return text[: max_tokens * 4]
        tokens = self._f.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = self._f.tokenizer.decode(tokens[:max_tokens])
        for pattern in ("\n\n", "\n", ". ", " "):
            last = truncated.rfind(pattern)
            if last > max_tokens * 0.6:
                truncated = truncated[: last + len(pattern)]
                break
        return truncated.rstrip()


class EnrichmentTasks:
    """Runs post‑processing enrichment after each user or assistant message,
    keeping the symbol index, conversation state, and LOD thresholds aligned
    with the evolving conversation.

    Provides:
    * Change and session summaries persisted to LTM and SQLite.
    * Mention tracking that decays block importance over time.
    * Time‑ and turn‑based expiration of inactive blocks.
    * Adaptive LOD threshold tuning based on which symbols the LLM actually
      used in its response.
    * Feedback context injection into Block A.
    * Parallel contradiction detection, response cache lookup, and duplicate
      question detection.
    * Batched, lazy docstring generation for symbols that lack one,
      including a background loop that fills missing docstrings without
      blocking the request path.
    * Turn‑based conversation window management (summarise then evict old
      turns, with a no‑degradation guard).

    Docs 10–13 backported:
        M5 – dunder disambiguation in docstring batch parsing.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref
        self._lazy_docstrings_generated_this_turn: int = 0
        self._active_bg_task: Optional[asyncio.Task] = None
        self._bg_docstring_count: int = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Change & session summaries
    # ═══════════════════════════════════════════════════════════════════════════

    async def generate_change_summary(
        self,
        block_hash: str,
        prev_content: str,
        new_content: str,
    ) -> None:
        """Generate and persist a change summary immediately (no deferral)."""
        model = self._f.valves.llm_model
        prompt = (
            f"Summarise the code change in ONE short sentence (max 15 words).\n\n"
            f"Previous:\n```\n{prev_content[:1000]}\n```\n\n"
            f"New:\n```\n{new_content[:1000]}\n```\n\n"
            f"Change summary:"
        )
        summary = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a code change summariser. Output only one short sentence.",
            model_override=model,
            max_tokens=40,
            temperature=0.1,
        )
        if summary:
            now = time.time()
            self._f._block_change_summaries[block_hash] = (summary.strip(), now)
            if len(self._f._block_change_summaries) > self._f._MAX_CHANGE_SUMMARIES:
                self._f._block_change_summaries.popitem(last=False)

            def _write():
                self._f._db_conn.execute(
                    "INSERT OR REPLACE INTO block_change_summaries "
                    "(block_hash, summary, created_at) VALUES (?, ?, ?)",
                    (block_hash, summary.strip(), now),
                )
                self._f._db_conn.execute(
                    "DELETE FROM block_change_summaries WHERE block_hash NOT IN "
                    "(SELECT block_hash FROM block_change_summaries ORDER BY created_at DESC LIMIT ?)",
                    (self._f._MAX_CHANGE_SUMMARIES,),
                )
                self._f._db_conn.commit()

            await self._f._state_store._db_enqueue(_write)

    async def run_session_summary_task(self, params: dict, model: str) -> bool:
        """Generate an autobiographical session summary and store it in LTM."""
        project_id = params["project_id"]
        code_state_hash = params.get("code_state_hash", "")

        recent = await self._f._ltm.retrieve_historical_messages(
            query="recent conversation summary",
            project_id=project_id,
            limit=self._f.valves.session_summary_interval_messages,
        )
        if not recent:
            return False

        conversation_text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in recent
        )
        prompt = (
            "Summarise the following conversation segment in 2-3 sentences, "
            "capturing the main task, decisions made, files modified, "
            "and architectural changes:\n\n"
            f"{conversation_text[:3000]}"
        )
        summary = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a helpful assistant that produces concise autobiographical session summaries.",
            model_override=model,
            max_tokens=self._f.valves.session_summary_max_tokens,
            temperature=0.2,
            label="session_summary",
        )
        if not summary:
            return False

        msg_id = f"{project_id}_session_summary_{int(time.time())}"
        embedding = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode(summary).tolist()
        )
        await anyio.to_thread.run_sync(
            lambda: self._f.memory_collection.upsert(
                ids=[msg_id],
                embeddings=[embedding],
                metadatas=[
                    {
                        "role": "assistant",
                        "project_id": project_id,
                        "timestamp": time.time(),
                        "is_session_summary": True,
                        "code_state_hash": code_state_hash,
                        "content_type": ContentType.GENERAL.value,
                        "has_code": False,
                    }
                ],
                documents=[f"[Session summary]\n{summary}"],
            )
        )
        self._f._log_debug(f"Session summary stored in LTM (msg_id={msg_id})")
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Block maintenance (mentions, expiration)
    # ═══════════════════════════════════════════════════════════════════════════

    def update_mentions_from_message(
        self,
        state: dict,
        message_content: str,
        project_id: str,
    ) -> None:
        """
        Increment mention counts for symbols referenced in the message content,
        and mark the corresponding blocks as recently mentioned.
        """
        if not message_content:
            return
        all_symbol_names = self._f._symbol_index.get_all_names(project_id)
        words = set(re.findall(r"\b[\w-]+\b", message_content))
        mentioned_names = all_symbol_names.intersection(words)
        if not mentioned_names:
            return
        affected_blocks: Set[str] = set()
        for name in mentioned_names:
            affected_blocks.update(self._f._symbol_index.find_blocks(name, project_id))
        for block_hash in affected_blocks:
            block = state.active_blocks.get(block_hash)
            if block:
                block.mention_count += 1
                block.last_mentioned = time.time()
                block.last_mentioned_msg_idx = state.message_count
                block._update_importance()

    async def expire_blocks_by_time(self, project_id: str) -> None:
        """Remove blocks that have not been mentioned recently, based on configured timeouts."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            if not state:
                return
            now = time.time()
            expiration_seconds = self._f.valves.block_expiration_hours * 3600
            to_remove = []
            for h, block in state.active_blocks.items():
                if block.pinned or block.obsolete:
                    continue
                age = now - block.last_mentioned
                if (
                    block.content_type == ContentType.ERROR
                    and self._f.valves.error_retention_turns > 0
                ):
                    if age > max(
                        self._f.valves.error_retention_turns * 300, expiration_seconds
                    ):
                        to_remove.append(h)
                elif (
                    block.content_type == ContentType.PROPOSED_CHANGE
                    and self._f.valves.proposed_change_retention_turns > 0
                ):
                    if age > max(
                        self._f.valves.proposed_change_retention_turns * 300,
                        expiration_seconds,
                    ):
                        to_remove.append(h)
            for h in to_remove:
                if h in state.active_blocks:
                    block = state.active_blocks[h]
                    self._f._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                del state.active_blocks[h]
            if to_remove:
                state.has_any_calls = any(
                    any(s.calls for s in b.symbols)
                    for b in state.active_blocks.values()
                )
                self._f._activation.invalidate_lightweight_cache(project_id)
                self._f._conversation_state_manager.set(project_id, state)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. LOD adaptive adjustment
    # ═══════════════════════════════════════════════════════════════════════════

    async def update_lod_thresholds_from_response(
        self,
        project_id: str,
        response_text: str,
    ) -> None:
        """
        Adjust lod3_threshold based on which symbols appear in the LLM's
        response compared to the LOD level they received.
        """
        if not self._f.valves.enable_lod_adaptive:
            return

        pstate = self._f._project_state_manager.get_pstate(project_id)
        last_lod_map = pstate.get("last_lod_levels", {})
        if not last_lod_map:
            return

        bare_names = self._f._symbol_index.get_all_names(project_id)
        response_words = set(re.findall(r"\b\w+\b", response_text))
        bare_referenced = bare_names.intersection(response_words)

        referenced: Set[str] = set()
        for bare in bare_referenced:
            referenced |= self._f._symbol_index.get_qualified_names_for(
                bare, project_id
            )

        underserved = [sym for sym in referenced if last_lod_map.get(sym, 3) < 3]
        overserved = [
            sym
            for sym in last_lod_map
            if last_lod_map[sym] == 3 and sym not in referenced
        ]

        old_threshold = self._f.valves.lod3_threshold
        changed = False

        if len(underserved) >= self._f.valves.lod_adapt_underserved_min:
            self._f.valves.lod3_threshold = max(
                self._f.valves.lod_adapt_min,
                self._f.valves.lod3_threshold - self._f.valves.lod_adapt_rate,
            )
            changed = True
            self._f._log_debug(
                f"LOD adaptive ↓: threshold {old_threshold:.2f} → "
                f"{self._f.valves.lod3_threshold:.2f} "
                f"({len(underserved)} underserved: {underserved[:3]})"
            )

        elif len(overserved) >= self._f.valves.lod_adapt_overserved_min:
            self._f.valves.lod3_threshold = min(
                self._f.valves.lod_adapt_max,
                self._f.valves.lod3_threshold + self._f.valves.lod_adapt_rate * 0.5,
            )
            changed = True
            self._f._log_debug(
                f"LOD adaptive ↑: threshold {old_threshold:.2f} → "
                f"{self._f.valves.lod3_threshold:.2f} "
                f"({len(overserved)} overserved symbols)"
            )

        if not changed:
            self._f._log_debug(
                f"LOD adaptive: no adjustment needed "
                f"(threshold={self._f.valves.lod3_threshold:.2f})"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Feedback context
    # ═══════════════════════════════════════════════════════════════════════════

    def get_feedback_context(self, project_id: str) -> str:
        """Return a formatted string of recent feedback for the given project."""
        state = self._f._conversation_state_manager.get(project_id)
        feedback = state.feedback_history
        if not feedback:
            return ""
        recent = feedback[-self._f.valves.feedback_history_limit :]
        lines = ["## Previous Feedback"]
        for fb in recent:
            success = "✅" if fb.success else "❌"
            lines.append(f"- {success} {fb.change_description[:100]}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Parallel checks (contradiction, cache, duplicate)
    # ═══════════════════════════════════════════════════════════════════════════

    async def parallel_context_checks(
        self,
        messages: list,
        query: str,
        context_hash: str,
        project_id: str,
        state: dict,
        skip_contradiction: bool = False,
        skip_cache: bool = False,
        skip_duplicate: bool = False,
    ) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        """
        Run contradiction detection, response cache lookup, and duplicate question
        detection in parallel. Returns (contradiction_warning, cached_response,
        duplicate_match). All three can be None.
        """
        tasks = [
            (
                self._f._commands._detect_contradictions(messages)
                if (
                    self._f.valves.enable_contradiction_detection
                    and not skip_contradiction
                )
                else asyncio.sleep(0, result=None)
            ),
            (
                self._f._ltm.find_cached_response(query, context_hash, state)
                if (
                    self._f.valves.enable_response_cache
                    and HAS_SENTENCE
                    and not skip_cache
                )
                else asyncio.sleep(0, result=None)
            ),
            (
                self._f._ltm.find_duplicate_question(query, project_id)
                if (
                    self._f.valves.duplicate_question_threshold
                    and HAS_SENTENCE
                    and not skip_duplicate
                )
                else asyncio.sleep(0, result=None)
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        contradiction = results[0] if not isinstance(results[0], Exception) else None
        cached = results[1] if not isinstance(results[1], Exception) else None
        duplicate = results[2] if not isinstance(results[2], Exception) else None
        return contradiction, cached, duplicate

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Docstring generation (batch and background) – MODIFIED (M5)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_docstring_batch_prompt(self, items: List[Tuple[str, str, str]]) -> str:
        """items: list of (qid, signature, snippet).  `qid` may be a bare
        function name or a qualified 'ClassName.method' id — it is always
        repeated EXACTLY as given, dot included, so the response can be
        matched back to the correct symbol without ambiguity."""
        parts = []
        for qid, signature, snippet in items:
            parts.append(f"### {qid}\n```\n{signature}\n{snippet[:300]}\n```")
        listing = "\n\n".join(parts)
        return (
            f"For each of the following {len(items)} code symbols, write ONE "
            f"short sentence describing what it does.\n\n{listing}\n\n"
            f"Output exactly one line per symbol, in this exact format:\n"
            f"<identifier>: <one short sentence>\n"
            f"Use the EXACT identifier as given above, including any "
            f"'ClassName.' prefix — do not drop it, do not add numbering, "
            f"headers, or any other text."
        )

    _BATCH_DOCSTRING_LINE_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z_][\w.]*)\s*:\s*(.+)$")

    # ── M5: Resolve dunders with context ─────────────────────────────────────

    def _resolve_parsed_docstring_name(
        self,
        bare_name: str,
        project_id: str,
        context_symbol: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a bare name from the batch docstring response to a qid.

        M5: dunder disambiguation via context_symbol.
        Step 1: bare-name lookup (returns single qid if unambiguous).
        Step 2: for dunders (__init__, __str__, etc.), use context_symbol's
                parent class to construct the full qid.
        Step 3: if multiple bare-name matches and no context, return None.
        """
        # Step 1: bare-name lookup
        results = self._f._symbol_index.get_qualified_names_for(bare_name, project_id)
        if len(results) == 1:
            return next(iter(results))

        # Step 2: dunder disambiguation via context_symbol parent
        if bare_name.startswith("__") and bare_name.endswith("__") and context_symbol:
            # Extract parent class from context_symbol (e.g. "ContextBuilder.build" → "ContextBuilder")
            parent = (
                context_symbol.rsplit(".", 1)[0]
                if "." in context_symbol
                else context_symbol
            )
            candidate = f"{parent}.{bare_name}"
            # Verify the candidate exists in the index
            if candidate in self._f._symbol_index.get_all_qualified_names(project_id):
                return candidate

        # Step 3: ambiguous bare name
        if len(results) > 1:
            self._f._log_debug(
                f"_resolve_parsed_docstring_name: ambiguous bare name '{bare_name}' "
                f"→ {len(results)} candidates, skipping"
            )
            return None

        return None

    def _parse_docstring_batch_response(
        self,
        response: str,
        expected_names: Set[str],
        batch_qids: List[str],
    ) -> Dict[str, str]:
        """
        Parse the LLM response for batch docstrings.

        Each line is expected to be in the format:
            <identifier>: <one short sentence>

        The identifier is resolved to a qualified id using `_resolve_parsed_docstring_name`,
        with the corresponding qid from the batch providing context for dunder disambiguation.

        Returns a dict mapping qualified id -> docstring.

        Modified (M5): uses context_symbol to disambiguate dunders (__init__, __str__, etc.).
        """
        result: Dict[str, str] = {}
        pattern = re.compile(r"^\s*[-*]?\s*([A-Za-z_][\w.]*)\s*:\s*(.+)$")

        for idx, line in enumerate(response.splitlines()):
            m = pattern.match(line.strip())
            if not m:
                continue

            bare_name, docstring = m.group(1), m.group(2).strip()
            if not docstring:
                continue

            # ── M5: resolve with context hint ──────────────────────────────
            context_hint = batch_qids[idx] if idx < len(batch_qids) else None
            qid = self._resolve_parsed_docstring_name(
                bare_name, self._f.valves.project_id, context_hint
            )

            if qid and qid in expected_names:
                result[qid] = docstring[:200]
            else:
                self._f._log_debug(
                    f"_parse_docstring_batch_response: could not resolve '{bare_name}'"
                )

        return result

    async def ensure_docstrings_batch(
        self, qids: List[str], project_id: str
    ) -> Dict[str, str]:
        """
        Resolve docstrings for many symbols at once, identified by their
        QUALIFIED id (e.g. "ContextBuilder.__init__") — never by bare name.

        M5 fix: batch_qids are passed to `_parse_docstring_batch_response`
        to provide context for disambiguating dunders.
        """
        state = self._f._conversation_state_manager.get(project_id)
        resolved: Dict[str, str] = {}
        pending: List[str] = []

        _qid_index: Dict[str, Tuple["CodeSymbol", "CodeBlock"]] = {}
        for _block in state.active_blocks.values():
            for _sym in _block.symbols:
                _q = qualify_symbol_name(_sym.name, _sym.parent_symbol)
                if _q not in _qid_index:
                    _qid_index[_q] = (_sym, _block)

        def _find_symbol(qid: str):
            return _qid_index.get(qid, (None, None))

        for qid in qids:
            sym, _ = _find_symbol(qid)
            found = sym.docstring if sym and sym.docstring else ""
            if not found:
                try:
                    row = await self._f._state_store._db_read(
                        lambda: self._f._db_conn.execute(
                            "SELECT docstring FROM symbol_docstrings "
                            "WHERE project_id=? AND symbol_name=?",
                            (project_id, qid),
                        ).fetchone()
                    )
                except Exception:
                    row = None
                if row and row[0]:
                    found = row[0]
                    if sym is not None:
                        sym.docstring = found
                    self._f._symbol_index.update_docstring(qid, project_id, found)
            if found:
                resolved[qid] = found
            else:
                pending.append(qid)

        if not pending:
            return resolved

        budget = self._f.valves.lazy_docstring_max_per_turn
        if budget > 0:
            remaining = max(0, budget - self._lazy_docstrings_generated_this_turn)
            pending = pending[:remaining]
        if not pending:
            return resolved

        items: List[Tuple[str, str, str]] = []
        for qid in pending:
            sym, block = _find_symbol(qid)
            if sym is not None and block is not None:
                signature = sym.signature
                if sym.line_start:
                    lines = block.content.split("\n")
                    start_idx = max(0, sym.line_start - 1)
                    end_idx = min(len(lines), (sym.line_end or sym.line_start + 30))
                    snippet = "\n".join(lines[start_idx:end_idx])[:500]
                else:
                    snippet = block.content[:500]
            else:
                signature, snippet = qid, ""
            items.append((qid, signature, snippet))

        batch_size = max(1, self._f.valves.docstring_batch_size)
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            expected = {q for q, _, _ in batch}
            batch_qids = [qid for qid, _, _ in batch]

            prompt = self._build_docstring_batch_prompt(batch)
            response = await self._f._llm_orchestrator.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are a code summarization assistant. Output only the "
                    "requested lines, one per symbol, no extra commentary."
                ),
                model_override=self._f.valves.llm_model,
                max_tokens=min(60 * len(batch), 600),
                temperature=0.1,
                label="lazy_docstring_batch",
            )
            self._lazy_docstrings_generated_this_turn += len(batch)
            if not response:
                continue

            # ── M5: `_parse_docstring_batch_response` ahora recibe batch_qids ──
            # y usa `_resolve_parsed_docstring_name` para desambiguar dunders.
            parsed = self._parse_docstring_batch_response(
                response, expected, batch_qids
            )

            for qid, docstring in parsed.items():
                resolved[qid] = docstring
                sym, _ = _find_symbol(qid)
                if sym is not None:
                    sym.docstring = docstring
                self._f._symbol_index.update_docstring(qid, project_id, docstring)

            if parsed:
                rows = [
                    (project_id, qid, doc, time.time()) for qid, doc in parsed.items()
                ]

                def _write_batch(rows=rows):
                    self._f._db_conn.executemany(
                        "INSERT OR REPLACE INTO symbol_docstrings "
                        "(project_id, symbol_name, docstring, updated_at) "
                        "VALUES (?,?,?,?)",
                        rows,
                    )
                    self._f._db_conn.commit()

                await self._f._state_store._db_enqueue(_write_batch)

        return resolved

    async def ensure_cfg_batch(
        self, qids: List[str], project_id: str
    ) -> Dict[str, str]:
        """
        Resolve control-flow skeletons for many symbols, identified by their
        QUALIFIED id. Pure CPU, deterministic, no LLM call.
        """
        self._f._log_debug(f"CFG batch: invoked with {len(qids)} candidate(s): {qids}")
        state = self._f._conversation_state_manager.get(project_id)
        cfg_rows = []
        resolved: Dict[str, str] = {}

        _qid_index: Dict[str, Tuple["CodeSymbol", "CodeBlock"]] = {}
        for _block in state.active_blocks.values():
            for _sym in _block.symbols:
                _q = qualify_symbol_name(_sym.name, _sym.parent_symbol)
                if _q not in _qid_index:
                    _qid_index[_q] = (_sym, _block)

        def _find_symbol_and_block(qid: str):
            return _qid_index.get(qid, (None, None))

        for qid in qids:
            sym, block = _find_symbol_and_block(qid)
            if sym is None or block is None:
                self._f._log_debug(
                    f"CFG batch: '{qid}' NOT FOUND in active_blocks — skipping"
                )
                continue
            if not sym.line_start or not sym.line_end:
                self._f._log_debug(
                    f"CFG batch: '{qid}' has no line_start/line_end — skipping"
                )
                continue

            lines = block.content.split("\n")
            snippet = "\n".join(lines[max(0, sym.line_start - 1) : sym.line_end])
            current_hash = hashlib.md5(snippet.encode()).hexdigest()[:16]

            meta = self._f._symbol_index.get_symbol_meta(qid, project_id) or {}
            cached_skeleton = meta.get("cfg_skeleton", "")
            cached_hash = meta.get("cfg_body_hash", "")
            if cached_skeleton and cached_hash == current_hash:
                self._f._log_debug(f"CFG batch: '{qid}' cache HIT")
                resolved[qid] = cached_skeleton
                continue

            result = ControlFlowExtractor.extract_for_symbol(
                block.content, sym, max_lines=self._f.valves.cfg_skeleton_max_lines
            )
            if result is None:
                self._f._log_debug(f"CFG batch: '{qid}' extractor returned None")
                continue
            skeleton, body_hash = result
            self._f._log_debug(
                f"CFG batch: '{qid}' skeleton generated ({len(skeleton)} chars)"
            )
            self._f._symbol_index.update_cfg(qid, project_id, skeleton, body_hash)
            resolved[qid] = skeleton
            cfg_rows.append((project_id, qid, skeleton, body_hash, time.time()))

        if cfg_rows:

            def _write_cfg_batch(rows=cfg_rows):
                self._f._db_conn.executemany(
                    "INSERT OR REPLACE INTO symbol_cfg "
                    "(project_id, symbol_name, cfg_skeleton, body_hash, updated_at) "
                    "VALUES (?,?,?,?,?)",
                    rows,
                )
                self._f._db_conn.commit()

            await self._f._state_store._db_enqueue(_write_cfg_batch)
            self._f._log_debug(
                f"CFG batch: persisted {len(cfg_rows)} CFG entries in one batch"
            )

        self._f._log_debug(f"CFG batch: resolved {len(resolved)}/{len(qids)}")
        return resolved

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Background docstring loop (with Q5 prioritization)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Q5: Priorization of docstring targets ───────────────────────────────

    def _prioritize_docstring_targets(
        self,
        qids: List[str],
        project_id: str,
        pstate: dict,
    ) -> List[str]:
        """
        Sort symbols by docstring generation priority.

        Priority order:
        1. Symbols in skeleton tier (visible in Block A every turn)
        2. Symbols currently in Block B at LOD-2/3 (visible this turn)
        3. Hub symbols (high PPR — frequently referenced)
        4. Everything else (leaf symbols — rarely in context)
        """
        skeleton_qids: Set[str] = set(pstate.get("skeleton_tier_qids", []))
        block_b_qids: Set[str] = set(pstate.get("block_b_qids_this_turn", []))

        # Get cached PPR scores if available (from Q2 fix cache)
        ppr_scores: Dict[str, float] = (
            self._f._activation._ppr_cache.get(
                pstate.get("code_state_hash", ""),
                frozenset(pstate.get("hub_tier_qids_persisted", [])),
            )
            or {}
        )

        def priority_key(qid: str) -> tuple:
            in_skeleton = 0 if qid in skeleton_qids else 1  # 0 = highest priority
            in_block_b = 0 if qid in block_b_qids else 1
            ppr = -ppr_scores.get(qid, 0.0)  # negate for desc sort
            return (in_skeleton, in_block_b, ppr, qid)

        return sorted(qids, key=priority_key)

    async def _docstring_generation_loop(self, project_id: str) -> None:
        """
        Background loop that generates docstrings for all pending symbols
        (functions, methods, and classes) that lack one.

        Takes a snapshot of all symbols without docstrings at the start,
        then prioritizes them using `_prioritize_docstring_targets` so that
        skeleton-tier symbols are documented first.

        Processes in batches using `docstring_bg_batch_size` to reduce serial
        LLM calls from N to ceil(N / batch_size).

        After all batches complete, calls `slot_restore_for_continuity()`
        to return the KV slot to the stable prefix.
        """
        # ── 1. Snapshot: collect all symbols without docstrings ──────────
        state = self._f._conversation_state_manager.get(project_id)
        pstate = self._f._project_state_manager.get_pstate(project_id)

        pending_qids: List[str] = []

        for block in state.active_blocks.values():
            if block.obsolete:
                continue
            for sym in block.symbols:
                if sym.kind in ("function", "method", "class") and not sym.docstring:
                    qid = qualify_symbol_name(sym.name, sym.parent_symbol)
                    pending_qids.append(qid)

        if not pending_qids:
            self._f._log_debug("Background docstring loop: no pending symbols")
            return

        self._f._log_debug(
            f"Background docstring loop: {len(pending_qids)} symbol(s) to process"
        )

        # ── Q5: Prioritize symbols ──────────────────────────────────────────
        prioritized_qids = self._prioritize_docstring_targets(
            pending_qids, project_id, pstate
        )

        self._f._log_debug(
            f"bg_docstring: prioritized {len(prioritized_qids)} symbols, "
            f"skeleton-tier first"
        )

        # ── 2. Batch configuration ──────────────────────────────────────────
        batch_size = getattr(self._f.valves, "docstring_bg_batch_size", 5)
        batches = [
            prioritized_qids[i : i + batch_size]
            for i in range(0, len(prioritized_qids), batch_size)
        ]

        self._f._log_debug(
            f"bg_docstring: {len(prioritized_qids)} symbols → "
            f"{len(batches)} batches (batch_size={batch_size})"
        )

        # ── 3. Process each batch ──────────────────────────────────────────
        for batch_idx, batch in enumerate(batches):
            try:
                results: Dict[str, str] = await self.ensure_docstrings_batch(
                    batch, project_id
                )
                for qid, docstring in results.items():
                    if docstring:
                        self._f._symbol_index.update_docstring(
                            qid, docstring, project_id
                        )
                self._f._log_debug(
                    f"bg_docstring batch {batch_idx + 1}/{len(batches)}: "
                    f"got {len(results)} docstrings"
                )
            except Exception as e:
                self._f._log_debug(
                    f"bg_docstring batch {batch_idx + 1} failed: {e} — "
                    "continuing with remaining batches"
                )

        # ── 4. Restore slot to stable prefix after all batch LLM calls ────
        try:
            await self._f._project_state_manager.slot_restore_for_continuity(project_id)
            self._f._log_debug(
                "bg_docstring: slot restored to stable prefix after batches"
            )
        except Exception as e:
            self._f._log_debug(f"bg_docstring: slot restore failed (non-fatal): {e}")

        self._f._log_debug("Background docstring loop: finished")

    def start_docstring_loop(self, project_id: str) -> None:
        """Launch the background docstring generation loop (if not already running)."""
        if self._active_bg_task is not None and not self._active_bg_task.done():
            return

        self._active_bg_task = asyncio.create_task(
            self._docstring_generation_loop(project_id)
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Helper methods for background docstring (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    async def cancel_docstring_tasks(self) -> None:
        """Cancel the background docstring generation loop gracefully."""
        if self._active_bg_task is not None and not self._active_bg_task.done():
            self._active_bg_task.cancel()
            try:
                await asyncio.wait_for(self._active_bg_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._active_bg_task = None

        drain_deadline = time.monotonic() + 5.0
        while not self._f._db_write_queue.empty():
            if time.monotonic() > drain_deadline:
                self._f._log_debug(
                    "cancel_docstring_tasks: drain timeout (5s), proceeding anyway"
                )
                break
            await asyncio.sleep(0.05)

    async def _background_docstring(
        self,
        sym: "CodeSymbol",
        block: "CodeBlock",
        project_id: str,
    ) -> None:
        """
        Generate a one-line docstring for a symbol (function, method, or class) in the background,
        and persist it to the SymbolIndex and SQLite.
        """
        name = sym.name
        kind = sym.kind
        signature = sym.signature
        line_start = sym.line_start
        line_end = sym.line_end
        block_hash = block.hash

        state = self._f._conversation_state_manager.get(project_id)
        target_block = state.active_blocks.get(block_hash)
        if target_block is None:
            self._f._log_debug(
                f"Background docstring: block {block_hash} not found, skipping '{name}'"
            )
            return

        snippet = ""
        if kind == "class":
            members_qids = self._f._symbol_index.get_class_members(name, project_id)
            members_meta = []
            for qid in members_qids:
                meta = self._f._symbol_index.get_symbol_meta(qid, project_id)
                if meta:
                    members_meta.append(
                        {
                            "qid": qid,
                            "signature": meta.get("signature", qid),
                            "line_start": meta.get("line_start"),
                        }
                    )
            section_headers = self._extract_section_comments(block, sym)
            snippet = self._build_class_skeleton(
                class_name=name,
                members_meta=members_meta,
                section_headers=section_headers,
                block=block,
                line_start=line_start or 1,
                line_end=line_end or len(block.content.splitlines()),
            )
        else:
            if line_start and target_block:
                lines = target_block.content.split("\n")
                start_idx = max(0, line_start - 1)
                end_idx = min(len(lines), (line_end or line_start + 30))
                snippet = "\n".join(lines[start_idx:end_idx])[:500]
            else:
                snippet = target_block.content[:500]

        if kind == "class":
            prompt = f"In one sentence, describe the single responsibility of this class based on its method names and structure:\n\n```\n{snippet}\n```"
        else:
            prompt = f"Summarize in one short sentence what this code does:\n\n```\n{signature}\n{snippet}\n```"

        docstring_text = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant. Output only one concise sentence.",
            model_override=self._f.valves.llm_model,
            max_tokens=500,
            temperature=0.1,
            label="bg_docstring",
        )

        if not docstring_text or not docstring_text.strip():
            return

        docstring = self._clean_single_docstring(docstring_text, name)
        if not docstring:
            self._f._log_debug(
                f"Background docstring: no valid docstring extracted for '{name}'"
            )
            return

        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._conversation_state_manager.get(project_id)
            block = state.active_blocks.get(block_hash)
            if block:
                for s in block.symbols:
                    if s.name == name and s.line_start == line_start:
                        s.docstring = docstring
                        qid = qualify_symbol_name(s.name, s.parent_symbol)
                        self._f._symbol_index.update_docstring(
                            qid, project_id, docstring
                        )
                        await self._f._state_store._db_enqueue(
                            lambda q=qid, d=docstring, pid=project_id: self._f._db_conn.execute(
                                "INSERT OR REPLACE INTO symbol_docstrings (project_id, symbol_name, docstring, updated_at) VALUES (?,?,?,?)",
                                (pid, q, d, time.time()),
                            )
                        )
                        break
                self._f._conversation_state_manager.set(project_id, state)
            else:
                self._f._log_debug(
                    f"Background docstring: block {block_hash} disappeared, skipping '{name}'"
                )

    def _clean_single_docstring(
        self, raw_response: str, symbol_name: str
    ) -> Optional[str]:
        """Parse the LLM response to extract a single docstring sentence."""
        _BAD_PATTERNS = re.compile(
            r"(?:Analyze\s+the\s+Request|Task:|Step:|Goal:|Purpose:|Reasoning:)"
        )
        _DOCSTRING_LINE_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z_][\w.]*)\s*:\s*(.+)$")

        lines = raw_response.strip().splitlines()
        docstring = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _BAD_PATTERNS.search(stripped):
                continue
            m = _DOCSTRING_LINE_RE.match(stripped)
            if m:
                desc = m.group(2).strip()
                if desc and len(desc) > 5:
                    docstring = desc
                    break
            if (stripped[0].isupper() or stripped[0].isalpha()) and len(stripped) > 10:
                if not re.match(r"^\d+\.\s+", stripped) and "**" not in stripped:
                    docstring = stripped
                    break

        if not docstring:
            for line in reversed(lines):
                stripped = line.strip()
                if (
                    stripped
                    and not _BAD_PATTERNS.search(stripped)
                    and len(stripped) > 10
                    and "**" not in stripped
                ):
                    docstring = stripped[:200]
                    break

        if docstring:
            docstring = re.sub(r"^\s*[A-Za-z_][\w.]*\s*:\s*", "", docstring, count=1)
            docstring = re.split(r"[.!?]\s", docstring, maxsplit=1)[0] + "."
            docstring = docstring[:200]

        return docstring

    def _extract_section_comments(
        self, block: "CodeBlock", class_sym: "CodeSymbol"
    ) -> List[Tuple[int, str]]:
        """Extract section header comments from within a class body."""
        SECTION_RE = re.compile(
            r"^\s*#\s*[─━=\-]{2,}.*[─━=\-]{0,}\s*$|^\s*#\s*[─━=\-\s]*\w+[─━=\-\s]*$"
        )
        lines = block.content.split("\n")
        start = (class_sym.line_start or 1) - 1
        end = class_sym.line_end or len(lines)
        result = []
        for i, line in enumerate(lines[start:end], start=start):
            if SECTION_RE.match(line):
                result.append((i + 1, line.strip()))
        return result

    def _build_class_skeleton(
        self,
        class_name: str,
        members_meta: List[Dict],
        section_headers: List[Tuple[int, str]],
        block: "CodeBlock",
        line_start: int,
        line_end: int,
    ) -> str:
        """Build a structural skeleton for a class."""
        method_lines = {}
        for m in members_meta:
            if m.get("line_start"):
                method_lines[m["line_start"]] = m.get("signature", m.get("qid", ""))

        header_lines = {h[0] for h in section_headers}
        lines = block.content.split("\n")
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        buf = [f"class {class_name}:"]
        for i in range(start_idx, end_idx):
            line_num = i + 1
            line = lines[i].rstrip()
            if line_num in header_lines:
                buf.append("    " + line.strip())
            elif line_num in method_lines:
                buf.append("    " + method_lines[line_num])

        if len(buf) == 1:
            buf.append("    ...")

        skeleton = "\n".join(buf)
        return skeleton[:800]

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. Q6: Docstring migration on symbol rename
    # ═══════════════════════════════════════════════════════════════════════════

    async def _migrate_docstring(
        self,
        from_qid: str,
        to_qid: str,
        project_id: str,
    ) -> bool:
        """
        Move a docstring from one qid to another (rename migration).
        Returns True if migration was successful.
        """
        try:
            # First, get the existing docstring
            row = await self._f._state_store._db_read(
                lambda: self._f._db_conn.execute(
                    "SELECT docstring FROM symbol_docstrings "
                    "WHERE project_id=? AND symbol_name=?",
                    (project_id, from_qid),
                ).fetchone()
            )
            if not row or not row[0]:
                return False

            docstring = row[0]

            # Insert or replace into the new qid
            def _write():
                self._f._db_conn.execute(
                    "INSERT OR REPLACE INTO symbol_docstrings "
                    "(project_id, symbol_name, docstring, updated_at) VALUES (?,?,?,?)",
                    (project_id, to_qid, docstring, time.time()),
                )
                # Delete the old entry
                self._f._db_conn.execute(
                    "DELETE FROM symbol_docstrings "
                    "WHERE project_id=? AND symbol_name=?",
                    (project_id, from_qid),
                )
                self._f._db_conn.commit()

            await self._f._state_store._db_enqueue(_write)

            # Also update the in-memory SymbolIndex
            self._f._symbol_index.update_docstring(to_qid, project_id, docstring)
            # Clear the old docstring from memory (optional)
            # self._f._symbol_index.update_docstring(from_qid, project_id, "")

            self._f._log_debug(
                f"Docstring migrated on rename: '{from_qid}' → '{to_qid}'"
            )
            return True

        except Exception as e:
            self._f._log_debug(f"Docstring migration failed {from_qid} → {to_qid}: {e}")
            return False

    async def _detect_and_migrate_renames(
        self,
        deleted_qids: Set[str],
        added_qids: Set[str],
        project_id: str,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Detect symbol renames and migrate docstrings from old to new qid.

        A rename is detected when:
        - A symbol disappears from the graph (in deleted_qids)
        - A new symbol appears (in added_qids)
        - Both share the same body hash (same implementation, different name)

        Returns:
            (truly_deleted, truly_added): sets with renames removed from both.
            The caller should not re-enqueue migrated symbols for bg_docstring.
        """
        # ── Obtener estado una sola vez ──
        state = self._f._conversation_state_manager.get(project_id)
        active_blocks = state.active_blocks  # ← cache local

        # Compute body hashes for deleted symbols (only if they had docstrings)
        old_body_hashes: Dict[str, str] = {}  # body_hash -> old_qid
        for qid in deleted_qids:
            # Check if this symbol has a docstring worth migrating
            doc = self._f._symbol_index.get_docstring(qid, project_id)
            if not doc:
                continue
            # Get the body content from the active block
            block_hashes = self._f._symbol_index.find_blocks(qid, project_id)
            for bh in block_hashes:
                block = active_blocks.get(bh)  # ← usar cache local
                if block and not block.obsolete:
                    body = CodeBlockManager.extract_symbol_body(block, qid)
                    if body:
                        body_hash = hashlib.md5(body.encode()).hexdigest()
                        old_body_hashes[body_hash] = qid
                        break

        # Compute body hashes for added symbols
        new_body_hashes: Dict[str, str] = {}  # body_hash -> new_qid
        for qid in added_qids:
            block_hashes = self._f._symbol_index.find_blocks(qid, project_id)
            for bh in block_hashes:
                block = active_blocks.get(bh)  # ← usar cache local
                if block and not block.obsolete:
                    body = CodeBlockManager.extract_symbol_body(block, qid)
                    if body:
                        body_hash = hashlib.md5(body.encode()).hexdigest()
                        new_body_hashes[body_hash] = qid
                        break

        # Find renames: same body hash, different qid
        migrated_old: Set[str] = set()
        migrated_new: Set[str] = set()

        for body_hash, old_qid in old_body_hashes.items():
            new_qid = new_body_hashes.get(body_hash)
            if new_qid and new_qid != old_qid:
                # Rename detected: migrate docstring
                success = await self._migrate_docstring(old_qid, new_qid, project_id)
                if success:
                    migrated_old.add(old_qid)
                    migrated_new.add(new_qid)

        # Return filtered sets
        return (deleted_qids - migrated_old), (added_qids - migrated_new)


class ActiveCodeUpdater:
    """Processes a new user or assistant message through the full code‑aware
    pipeline, keeping the active block set and SymbolIndex in sync.

    Orchestrates, in order:
    * Extraction of code blocks and symbols via tree‑sitter.
    * Duplicate detection against the current active blocks.
    * Registration of new blocks (indexing symbols, call edges, data‑flow edges).
    * Handling of duplicate blocks (pinned/raw force‑update, obsolete marking).
    * Conflict detection for proposed changes and optional diff application.
    * Hard eviction when ``max_active_blocks`` is exceeded.
    * Post‑update tasks: mention tracking, expiration, duplicate removal,
      oversized‑block summaries, path‑index invalidation, and soft‑eviction
      via the ContextPager.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Main orchestration
    # ═══════════════════════════════════════════════════════════════════════════

    async def process(
        self, message: dict, project_id: str, is_continuation: bool = False
    ) -> None:
        """Orchestrate the full update pipeline for one message."""
        if not self._f.valves.enable_code_awareness:
            return

        content = message.get("content", "")
        role = message.get("role", "")

        # 1. Extract new blocks and symbols
        new_blocks_pending, symbols_list, content_to_syms, extracted_blocks = (
            await self._extract_and_prepare_new_blocks(content, role)
        )

        # 2. Get project lock and current state
        lock = await self._f._state_store.get_project_lock(project_id)
        state_before = self._f._conversation_state_manager.get(project_id)

        # 3. Detect duplicates
        duplicate_info = self._detect_duplicates(new_blocks_pending, state_before)

        async with lock:
            state = self._f._conversation_state_manager.get(project_id)

            # 4. Housekeeping
            self._f._enrichment.update_mentions_from_message(state, content, project_id)
            for block in state.active_blocks.values():
                if (
                    block.content
                    and self._f._code_blocks.calculate_code_similarity(
                        block.content[:200], content[:200]
                    )
                    > 0.7
                ):
                    block.mention_count += 1
                    block.last_mentioned = time.time()
                    block.last_mentioned_msg_idx = state.message_count
                    block._update_importance()

            if not content and not new_blocks_pending:
                return

            # 5. Process each new block
            for new_block, syms in zip(new_blocks_pending, symbols_list):
                if isinstance(syms, Exception):
                    syms = []

                new_block.content = CodeBlockManager.sanitize_text(new_block.content)
                if self._f.tokenizer:
                    new_block._cached_token_count = len(
                        self._f.tokenizer.encode(new_block.content)
                    )
                else:
                    new_block._cached_token_count = len(new_block.content) // 4

                is_dup, existing_hash = duplicate_info.get(
                    new_block.hash, (False, None)
                )
                existing = (
                    state.active_blocks.get(existing_hash) if existing_hash else None
                )

                if is_dup and existing:
                    await self._process_duplicate_block(
                        existing, new_block, syms, state, project_id
                    )
                else:
                    await self._process_new_block(new_block, syms, state, project_id)

            # 6. Update assistant base blocks
            if role == "assistant" and len(extracted_blocks) > 0:
                await self._update_assistant_base_blocks(
                    extracted_blocks, content_to_syms, state, project_id
                )

            # 7. Post-update tasks
            await self._post_update_tasks(
                state, project_id, new_blocks_pending, is_continuation
            )

            self._f._conversation_state_manager.set(project_id, state)

        # ── Invalidate session classification cache whenever active blocks change ──
        # (ensures that subsequent queries remain code sessions if code exists)
        if new_blocks_pending:
            self._f._session_classify_cache.clear()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Extraction & preparation of new blocks
    # ═══════════════════════════════════════════════════════════════════════════

    async def _extract_and_prepare_new_blocks(
        self, content: str, role: str
    ) -> Tuple[
        List["CodeBlock"], List[List["CodeSymbol"]], Dict[str, List["CodeSymbol"]]
    ]:
        """Extract code blocks from content, build CodeBlock objects, and extract symbols."""
        extracted_blocks, block_spans = await self._f._code_blocks.extract_code_blocks(
            content
        )
        new_blocks_pending = []
        for idx, block_info in enumerate(extracted_blocks):
            blk_file = None
            if self._f.valves.track_file_paths and block_spans:
                blk_file = self._f._code_blocks.extract_file_path_for_block(
                    content, block_spans[idx][0]
                )
            if not blk_file and len(extracted_blocks) == 1:
                extracted_paths = self._f._code_blocks.extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None
            content_type = self._f._code_blocks.classify_content(
                content, extracted_blocks
            )
            new_block = CodeBlock(
                content=block_info["code"],
                content_type=content_type,
                generated_by_assistant=(role == "assistant"),
                file_path=blk_file,
                line_range=None,
                timestamp=time.time(),
                is_active=True,
                mention_count=1,
                pinned=False,
                obsolete=False,
                is_raw=block_info.get("is_raw", False),
            )
            if "[KEEP]" in content:
                new_block.is_raw = True
            if "[KEEP]" in content or "#important" in content.lower():
                new_block.importance_score = 10.0
                new_block.pinned = True
            new_blocks_pending.append(new_block)

        symbols_list = []
        for idx, (blk, block_info) in enumerate(
            zip(new_blocks_pending, extracted_blocks)
        ):
            precomputed = block_info.get("precomputed_symbols")
            if precomputed:
                syms = precomputed
            else:
                syms = await SignatureExtractor.extract_async(
                    blk.content, blk.file_path
                )
            symbols_list.append(syms)

        content_to_syms: Dict[str, List[CodeSymbol]] = {
            blk.content: syms
            for blk, syms in zip(new_blocks_pending, symbols_list)
            if not isinstance(syms, Exception)
        }

        return new_blocks_pending, symbols_list, content_to_syms, extracted_blocks

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Duplicate detection
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_duplicates(
        self, new_blocks: List["CodeBlock"], state: dict
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Compare each new block against the current active blocks.
        Returns {new_block.hash: (is_duplicate, existing_block_hash)}.
        """
        duplicate_info: Dict[str, Tuple[bool, Optional[str]]] = {}
        if not state or not new_blocks:
            return duplicate_info

        existing_contents = {h: b.content for h, b in state.active_blocks.items()}

        for new_block in new_blocks:
            is_dup = False
            existing_dup = None
            for h, ex_content in existing_contents.items():
                ex_block = state.active_blocks.get(h)
                if (
                    ex_block
                    and self._f._code_blocks.calculate_code_similarity(
                        new_block.content, ex_content
                    )
                    >= self._f.valves.code_similarity_threshold
                ):
                    is_dup = True
                    existing_dup = h
                    break
            duplicate_info[new_block.hash] = (is_dup, existing_dup)

        return duplicate_info

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Processing duplicate blocks
    # ═══════════════════════════════════════════════════════════════════════════

    async def _process_duplicate_block(
        self,
        existing: "CodeBlock",
        new_block: "CodeBlock",
        syms: List["CodeSymbol"],
        state: dict,
        project_id: str,
    ) -> None:
        """
        Update an existing block with new content when a duplicate is detected.
        Handles both pinned/raw blocks (forced update) and prioritised recent code.
        Re‑indexes symbols and edges after the update.
        """
        if existing.pinned or new_block.is_raw:
            # Pinned or raw block: force update, keep pinned status
            self._f._symbol_index.remove_all_for_block(
                existing.hash, existing.symbols, project_id
            )
            prev_content = existing.content
            existing.content = new_block.content
            existing.hash = new_block.hash
            if new_block.file_path:
                existing.file_path = new_block.file_path
            existing.line_range = new_block.line_range
            existing.timestamp = time.time()
            existing.mention_count += 1
            existing.last_mentioned = time.time()
            existing.last_mentioned_msg_idx = state.message_count
            existing.pinned = True
            existing.is_raw = existing.is_raw or new_block.is_raw
            existing.importance_score = 10.0
            existing.symbols = syms

            # Re-index with background docstrings
            await self._reindex_block_symbols_with_docstrings(existing, project_id)

            if prev_content != new_block.content:
                await self._f._enrichment.generate_change_summary(
                    existing.hash, prev_content, new_block.content
                )
            return

        if self._f.valves.prioritize_recent_code:
            # Prioritise recent code: replace content with newest version
            self._f._symbol_index.remove_all_for_block(
                existing.hash, existing.symbols, project_id
            )
            prev_content = existing.content
            existing.content = new_block.content
            existing.hash = new_block.hash
            if new_block.file_path:
                existing.file_path = new_block.file_path
            existing.line_range = new_block.line_range
            existing.timestamp = time.time()
            existing.mention_count += 1
            existing.last_mentioned = time.time()
            existing.last_mentioned_msg_idx = state.message_count
            existing.symbols = syms

            # Re-index with background docstrings
            await self._reindex_block_symbols_with_docstrings(existing, project_id)

            if prev_content != new_block.content:
                await self._f._enrichment.generate_change_summary(
                    existing.hash, prev_content, new_block.content
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Reindexing helpers
    # ═══════════════════════════════════════════════════════════════════════════

    async def _reindex_block_symbols(self, block: "CodeBlock", project_id: str) -> None:
        """Re‑extract symbols for a block and register them + edges in the index."""
        for s in block.symbols:
            s.parent_block_hash = block.hash
            self._f._symbol_index.add(s, block.hash, project_id)
            caller_qid = qualify_symbol_name(s.name, s.parent_symbol, s.file_path)
            for callee_name in s.calls:
                edge = Edge(
                    src=caller_qid,
                    dst=callee_name,
                    type="calls",
                    weight=EDGE_WEIGHTS["calls"],
                    confidence=1.0,
                )
                self._f._symbol_index.add_edge(edge, project_id)

        # Data-flow edges: ONE full-file AST pass per block, not per symbol.
        if self._f.valves.enable_data_flow_analysis and block.file_path:
            df_edges = self._f._code_blocks.extract_data_flow_edges(
                block.content, block.file_path, project_id
            )
            for df_edge in df_edges:
                self._f._symbol_index.add_edge(df_edge, project_id)
            if df_edges:
                self._f._log_debug(
                    f"Data flow: {len(df_edges)} edge(s) extracted from {block.file_path}"
                )

        if self._f.tokenizer:
            block._cached_token_count = len(self._f.tokenizer.encode(block.content))
        else:
            block._cached_token_count = len(block.content) // 4
        block._update_importance()

    async def _reindex_block_symbols_with_docstrings(
        self, block: "CodeBlock", project_id: str
    ) -> None:
        """Re‑extract symbols for a block and register them + edges in the index."""
        for s in block.symbols:
            s.parent_block_hash = block.hash
            self._f._symbol_index.add(s, block.hash, project_id)
            caller_qid = qualify_symbol_name(s.name, s.parent_symbol, s.file_path)
            for callee_name in s.calls:
                edge = Edge(
                    src=caller_qid,
                    dst=callee_name,
                    type="calls",
                    weight=EDGE_WEIGHTS["calls"],
                    confidence=1.0,
                )
                self._f._symbol_index.add_edge(edge, project_id)

        # Data-flow edges: ONE full-file AST pass per block, not per symbol.
        if self._f.valves.enable_data_flow_analysis and block.file_path:
            df_edges = self._f._code_blocks.extract_data_flow_edges(
                block.content, block.file_path, project_id
            )
            for df_edge in df_edges:
                self._f._symbol_index.add_edge(df_edge, project_id)
            if df_edges:
                self._f._log_debug(
                    f"Data flow: {len(df_edges)} edge(s) extracted from {block.file_path}"
                )

        if self._f.tokenizer:
            block._cached_token_count = len(self._f.tokenizer.encode(block.content))
        else:
            block._cached_token_count = len(block.content) // 4
        block._update_importance()

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Processing new blocks
    # ═══════════════════════════════════════════════════════════════════════════

    async def _process_new_block(
        self,
        new_block: "CodeBlock",
        syms: List["CodeSymbol"],
        state: dict,
        project_id: str,
    ) -> None:
        """
        Register a brand‑new block: symbols, edges, conflict/obsolete checks,
        and hard eviction if max_active_blocks is exceeded.

        Q6: Migrates docstrings from renamed symbols before processing.
        """
        for sym in syms:
            sym.parent_block_hash = new_block.hash
        new_block.symbols = syms
        new_block.last_mentioned_msg_idx = state.message_count

        # --- 1. Index symbols and call-graph edges ---
        for sym in syms:
            self._f._symbol_index.add(sym, new_block.hash, project_id)
            caller_qid = qualify_symbol_name(sym.name, sym.parent_symbol, sym.file_path)
            for callee_name in sym.calls:
                edge = Edge(
                    src=caller_qid,
                    dst=callee_name,
                    type="calls",
                    weight=EDGE_WEIGHTS["calls"],
                    confidence=1.0,
                )
                self._f._symbol_index.add_edge(edge, project_id)

        # --- 2. Data-flow edges: ONE full-file AST pass per block ---
        if self._f.valves.enable_data_flow_analysis and new_block.file_path:
            df_edges = self._f._code_blocks.extract_data_flow_edges(
                new_block.content, new_block.file_path, project_id
            )
            for df_edge in df_edges:
                self._f._symbol_index.add_edge(df_edge, project_id)
            if df_edges:
                self._f._log_debug(
                    f"Data flow: {len(df_edges)} edge(s) extracted from {new_block.file_path}"
                )

        if any(s.calls for s in syms):
            state.has_any_calls = True

        # --- 3. Check for conflicting proposed changes ---
        is_conflicting = False
        if new_block.content_type == ContentType.PROPOSED_CHANGE:
            is_conflicting = self._f._code_blocks.has_conflicting_proposed_changes(
                state, new_block
            )
            if is_conflicting:
                new_block.importance_score = max(new_block.importance_score, 7.0)

        # --- 4. Insert the new block into active_blocks ---
        state.active_blocks[new_block.hash] = new_block

        # --- 5. Mark older blocks for the same file as obsolete (step 13) ---
        obsolete_hashes = []
        if new_block.file_path and self._f.valves.enable_obsolete_marking:
            for h, blk in list(state.active_blocks.items()):
                if h == new_block.hash:
                    continue
                if blk.file_path == new_block.file_path and not blk.pinned:
                    # Remove from symbol index
                    self._f._symbol_index.remove_all_for_block(
                        blk.hash, blk.symbols, project_id
                    )
                    # Mark obsolete
                    blk.obsolete = True
                    blk._update_importance()
                    obsolete_hashes.append(h)

            # --- Cap obsolete versions per file ---
            max_keep = self._f.valves.max_obsolete_versions_per_file
            if obsolete_hashes:
                if max_keep == 0:
                    for h in obsolete_hashes:
                        del state.active_blocks[h]
                    self._f._log_debug(
                        f"Removed {len(obsolete_hashes)} obsolete block(s) for '{new_block.file_path}' "
                        f"(max_obsolete_versions_per_file=0)."
                    )
                else:
                    obsolete_blocks = [
                        (h, state.active_blocks[h])
                        for h in obsolete_hashes
                        if h in state.active_blocks
                    ]
                    obsolete_blocks.sort(key=lambda x: x[1].timestamp, reverse=True)

                    to_remove = [h for h, _ in obsolete_blocks[max_keep:]]
                    for h in to_remove:
                        del state.active_blocks[h]

                    kept = len(obsolete_blocks) - len(to_remove)
                    self._f._log_debug(
                        f"Obsolete cap for '{new_block.file_path}': kept {kept} most recent "
                        f"out of {len(obsolete_blocks)} (max_obsolete_versions_per_file={max_keep})."
                    )

        # --- 6. Handle content-type specific actions ---
        if new_block.content_type == ContentType.PROPOSED_CHANGE:
            if new_block.file_path:
                state.recent_changes = [
                    c
                    for c in state.recent_changes
                    if not (
                        c.file_path
                        and c.file_path == new_block.file_path
                        and c.hash != new_block.hash
                    )
                ]
            state.recent_changes.append(new_block)
            if self._f.valves.enable_diff_application and not is_conflicting:
                for base in list(state.active_blocks.values()):
                    if (
                        base.content_type == ContentType.BASE_CODE
                        and base.file_path == new_block.file_path
                    ):
                        if await self._f._code_blocks.apply_change_with_diff(
                            base, new_block
                        ):
                            state.recent_changes = [
                                c
                                for c in state.recent_changes
                                if c.hash != new_block.hash
                            ]
                            state.committed_changes.append(new_block)
                            break
        elif new_block.content_type == ContentType.COMMITTED_CHANGE:
            state.committed_changes.append(new_block)
        elif (
            new_block.content_type == ContentType.ERROR
            and self._f.valves.preserve_error_context
        ):
            new_block.importance_score = min(new_block.importance_score + 3.0, 10.0)

        # --- 7. Hard eviction if too many active blocks ---
        if (
            self._f.valves.max_active_blocks > 0
            and len(state.active_blocks) > self._f.valves.max_active_blocks
        ):
            sorted_blocks = sorted(
                state.active_blocks.values(),
                key=lambda b: b.importance_score
                + (self._f.valves.raw_file_priority_boost if b.is_raw else 0),
                reverse=True,
            )
            keep_hashes = {
                b.hash for b in sorted_blocks[: self._f.valves.max_active_blocks]
            }
            to_remove_hard = [h for h in state.active_blocks if h not in keep_hashes]
            for h in to_remove_hard:
                block = state.active_blocks.get(h)
                if (
                    block
                    and self._f.valves.enable_block_paging
                    and self._f._pager is not None
                ):
                    paged = await self._f._pager.page_out_block(
                        block=block,
                        project_id=project_id,
                        state=state,
                        symbol_index=self._f._symbol_index,
                        chroma_collection=self._f.memory_collection,
                        embedder=self._f.embedder,
                    )
                    if paged:
                        del state.active_blocks[h]
                        continue
                if h in state.active_blocks:
                    del state.active_blocks[h]
            if to_remove_hard:
                self._f._log_debug(
                    f"Evicted {len(to_remove_hard)} blocks due to max_active_blocks limit. "
                    f"Their symbols remain in the index for lightweight context."
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Updating assistant base blocks
    # ═══════════════════════════════════════════════════════════════════════════

    async def _update_assistant_base_blocks(
        self,
        extracted: List[dict],
        content_to_syms: Dict[str, List["CodeSymbol"]],
        state: dict,
        project_id: str,
    ) -> None:
        """When the assistant replies with code that overlaps existing blocks, merge it in."""
        for block_info in extracted:
            best_base = None
            best_sim = 0.0
            for base in state.active_blocks.values():
                if base.content_type == ContentType.BASE_CODE:
                    sim = self._f._code_blocks.calculate_code_similarity(
                        base.content, block_info["code"]
                    )
                    if sim > best_sim and sim > 0.6:
                        best_sim = sim
                        best_base = base
            if best_base and best_sim > 0.5 and best_sim < 0.98:
                self._f._symbol_index.remove_all_for_block(
                    best_base.hash, best_base.symbols, project_id
                )
                prev_content = best_base.content
                best_base.content = CodeBlockManager.sanitize_text(block_info["code"])
                best_base.hash = hashlib.md5(block_info["code"].encode()).hexdigest()[
                    :16
                ]
                best_base.timestamp = time.time()
                best_base.is_active = True
                best_base.importance_score = min(best_base.importance_score + 1.0, 10.0)
                reused = content_to_syms.get(block_info["code"])
                if reused is not None:
                    best_base.symbols = [
                        s.copy(update={"parent_block_hash": best_base.hash})
                        for s in reused
                    ]
                else:
                    best_base.symbols = await SignatureExtractor.extract_async(
                        best_base.content, best_base.file_path
                    )
                await self._reindex_block_symbols(best_base, project_id)
                if any(s.calls for s in best_base.symbols):
                    state.has_any_calls = True
                if prev_content != block_info["code"]:
                    await self._f._enrichment.generate_change_summary(
                        best_base.hash, prev_content, block_info["code"]
                    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Post‑update tasks
    # ═══════════════════════════════════════════════════════════════════════════

    async def _post_update_tasks(
        self,
        state: dict,
        project_id: str,
        new_blocks_pending: List["CodeBlock"],
        is_continuation: bool,
    ) -> None:
        """Expiration, enrichment, oversized‑block summaries, path index, soft eviction."""
        if not is_continuation:
            state.message_count += 1
        if self._f.valves.auto_remove_duplicate_blocks:
            self._f._code_blocks.remove_duplicate_blocks(state, project_id)

        # Inline block expiration
        await self._f._enrichment.expire_blocks_by_time(project_id)

        # Missing docstrings are now generated reactively in _process_new_block
        # and _process_duplicate_block, right after a symbol is indexed.
        # No batch loop is needed anymore.

        if self._f.valves.enable_session_summary and not is_continuation:
            interval = self._f.valves.session_summary_interval_messages
            if (
                interval > 0
                and state.message_count % interval == 0
                and state.message_count > 0
            ):
                await self._f._enrichment.run_session_summary_task(
                    {
                        "project_id": project_id,
                        "message_count": state.message_count,
                        "code_state_hash": self._f._activation.compute_code_state_hash(
                            project_id
                        ),
                    },
                    self._f.valves.llm_model,
                )

        # Oversized block summaries (action="summarize")
        if (
            self._f.valves.max_code_block_tokens > 0
            and self._f.valves.code_block_overflow_action == "summarize"
        ):
            for block in state.active_blocks.values():
                if not block.obsolete:
                    await self._f._history_compressor.schedule_block_summary(
                        block, project_id
                    )

        self._f._activation.invalidate_lightweight_cache(project_id)

        # Path index invalidation
        if self._f.valves.enable_path_analysis:
            changed_symbols: Set[str] = set()
            for blk in new_blocks_pending:
                for sym in blk.symbols:
                    changed_symbols.add(sym.name)

            stale_path_ids: Set[str] = set()
            for sym_name in changed_symbols:
                for pid in self._f._path_index.mark_stale_for_symbol(
                    sym_name, project_id
                ):
                    stale_path_ids.add(pid)

            for pid in stale_path_ids:
                view = self._f._path_index.get(pid, project_id)
                if not view:
                    continue
                new_structural = self._f._activation.compute_structural_hash(
                    view.induced_nodes.keys(), project_id
                )
                new_call_graph = self._f._activation.compute_call_graph_hash(
                    view.induced_nodes.keys(), project_id
                )
                if view.is_stale(new_structural, new_call_graph):
                    view.structural_hash = new_structural
                    view.call_graph_hash = new_call_graph
                    view.summary = ""
                    view.business_label = ""
                    view.label_confidence = 0.0

            if stale_path_ids:
                self._f._log_debug(
                    f"Invalidated {len(stale_path_ids)} CodePathView(s) "
                    f"due to changes in {len(changed_symbols)} symbol(s)"
                )

        # Soft eviction via ContextPager
        if self._f.valves.enable_block_paging and self._f._pager is not None:
            candidates = self._f._pager.get_eviction_candidates(
                state=state,
                project_id=project_id,
                activation_scores=getattr(self._f, "_last_activation_scores", {}).get(
                    project_id, {}
                ),
                paging_threshold=self._f.valves.block_paging_threshold,
                min_activation=self._f.valves.block_paging_min_activation,
            )
            for hash_ in candidates:
                block = state.active_blocks.get(hash_)
                if block:
                    paged = await self._f._pager.page_out_block(
                        block=block,
                        project_id=project_id,
                        state=state,
                        symbol_index=self._f._symbol_index,
                        chroma_collection=self._f.memory_collection,
                        embedder=self._f.embedder,
                    )
                    if paged:
                        del state.active_blocks[hash_]
            if candidates:
                self._f._log_debug(
                    f"Soft-evicted {len(candidates)} block(s) via ContextPager "
                    f"(active_blocks now {len(state['active_blocks'])})"
                )


class InletOrchestrator:
    """Handles the early stages of request processing: extracting user
    information, classifying the session type, and preparing the code
    session for context assembly.

    Provides:
    * ``get_project_id()`` — returns the current project id from valves.
    * ``inlet_preprocess(body, project_id)`` — detects project switches,
      loads persisted edges and path views, and initiates KV‑slot restore.
    * ``inlet_extract_user_info(messages)`` — finds the last user message,
      strips code spans to isolate the question, and detects explicit
      slash‑commands.
    * ``inlet_prepare_code_session(messages, project_id, ...)`` —
      classifies whether the session involves code, triggers active‑block
      updates, and handles AutoContinue continuations.
    * ``classify_session(messages, project_id)`` — returns True if the
      conversation context indicates a coding session.
    * ``ensure_last_message_is_user(messages)`` — guarantees the message
      list ends with a user role.

    Docs 10–13 backported:
        E2 – detect short continuations and inherit use_case from previous turn.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """Initialize with a reference to the parent Filter."""
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Initialization & basic utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def get_project_id(self) -> str:
        """Return the current project id from the valves configuration."""
        return self._f.valves.project_id

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Preprocessing (project switch, cache load, slot restore)
    # ═══════════════════════════════════════════════════════════════════════════

    async def inlet_preprocess(self, body: dict, project_id: str) -> list:
        """Handle project switching, symbol cache loading, and KV slot restore."""
        messages = body.get("messages", [])

        if self._f._last_project_id and self._f._last_project_id != project_id:
            self._f._log_debug(
                f"Project changed from {self._f._last_project_id} to {project_id}"
            )
            self._f._conversation_state_manager.clear_project(self._f._last_project_id)
            self._f._symbol_index.clear_project(self._f._last_project_id)
            self._f._project_state_manager.clear_project(self._f._last_project_id)
            self._f._block_change_summaries.clear()

        self._f._last_project_id = project_id

        # ── load persisted CodePathViews if index is empty ──
        if self._f.valves.enable_path_analysis and HAS_TREE_SITTER:
            existing_views = self._f._path_index.get_all(project_id)
            all_names = self._f._symbol_index.get_all_names(project_id)
            if all_names and not existing_views:
                self._f._log_debug(
                    "PathIndex empty but symbols exist — loading from DB"
                )
                db_views = await self._f._state_store.load_path_views_from_db(
                    project_id
                )
                for view in db_views:
                    self._f._path_index.add(view, project_id)

        # ── restore typed edges from DB ─────────
        if self._f.valves.enable_edge_persistence:
            restored = await self._f._state_store.load_symbol_edges_from_db(project_id)
            if restored > 0:
                self._f._log_debug(
                    f"Cross-session: {restored} symbol edges restored from DB. "
                    f"No need to re-paste code."
                )

        return messages

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. User info extraction (last message, query, commands)
    # ═══════════════════════════════════════════════════════════════════════════

    async def inlet_extract_user_info(
        self,
        messages: list,
    ) -> Tuple[Optional[dict], str, str, bool, bool]:
        """Extract last user message, query, and detect explicit commands."""
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        user_query = last_user_msg.get("content", "") if last_user_msg else ""

        has_code_blocks = False
        user_question = user_query
        if last_user_msg and user_query:
            try:
                spans = await self._f._code_blocks.get_code_spans(user_query)
                if spans:
                    user_question = CodeBlockManager.remove_code_spans(
                        user_query, spans
                    ).strip()
                if "```" in user_query:
                    has_code_blocks = True
                if spans:
                    has_code_blocks = True
            except Exception:
                spans = None
            if not user_question or len(user_question) < 10:
                cleaned = re.sub(r"```.*?```", "", user_query, flags=re.DOTALL)
                cleaned = re.sub(r"`[^`]+`", "", cleaned)
                cleaned = re.sub(
                    r"\b(def |class |import |from |function |const |let |var )",
                    "",
                    cleaned,
                )
                cleaned = cleaned.strip()
                user_question = (
                    cleaned if (cleaned and len(cleaned) >= 10) else user_query
                )

        is_explicit_command = last_user_msg and last_user_msg.get(
            "content", ""
        ).startswith("/")

        # Capture every system message now, joined in order, before any
        # downstream compression/trim step can touch them.
        system_msgs_now = [m for m in messages if m.get("role") == "system"]
        original_system_prompt = "\n\n".join(
            m.get("content", "")
            for m in system_msgs_now
            if m.get("content", "").strip()
        )
        self._f._original_system_prompt = original_system_prompt

        return (
            last_user_msg,
            user_query,
            user_question,
            is_explicit_command,
            has_code_blocks,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Code session preparation (classification, active block update)
    # ═══════════════════════════════════════════════════════════════════════════

    async def inlet_prepare_code_session(
        self,
        messages: list,
        project_id: str,
        user_query: str,
        is_continuation: bool = False,
    ) -> Tuple[bool, str]:
        """Classify the session and update active code blocks."""
        is_code_session = await self.classify_session(messages, project_id)

        if self._f.valves.enable_code_awareness and is_code_session:
            last_idx = len(messages) - 1
            await self._f._update_active_code(
                messages[last_idx], project_id, is_continuation=is_continuation
            )
            extracted_blocks, block_spans = (
                await self._f._code_blocks.extract_code_blocks(user_query)
            )
            if block_spans:
                user_question = CodeBlockManager.remove_code_spans(
                    user_query, block_spans
                ).strip()
                if not user_question or len(user_question) < 10:
                    user_question = user_query
            else:
                user_question = user_query

            pstate = self._f._project_state_manager.get_pstate(project_id)
            pstate["last_processed_message_idx"] = last_idx
        else:
            user_question = user_query

        state = self._f._conversation_state_manager.get(project_id)
        if not isinstance(state.active_blocks, dict):
            self._f._log_debug(
                "CRITICAL: active_blocks corrupted even after load; resetting to empty. "
                "Delete %s if this recurs." % self._f.valves.state_db_path
            )
            state.active_blocks = {}
            self._f._conversation_state_manager.set(project_id, state)

        return is_code_session, user_question

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Session classification (cached CrossEncoder or heuristic)
    # ═══════════════════════════════════════════════════════════════════════════

    async def classify_session(self, messages: list, project_id: str) -> bool:
        """
        Determine whether the current session is a coding session.
        Uses cached results per project, then falls back to code indicators,
        block extraction, and finally a CrossEncoder check on the last user message.
        """
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        cache_key = None
        if last_user:
            content_key = last_user.get("content", "")[:200]
            cache_key = (
                f"{project_id}:{hashlib.md5(content_key.encode()).hexdigest()[:12]}"
            )
            cached = self._f._session_classify_cache.get(cache_key)
            if cached is not None:
                result, ts = cached
                if time.time() - ts < self._f._session_classify_ttl:
                    return result
                del self._f._session_classify_cache[cache_key]

        state = self._f._conversation_state_manager.get(project_id)
        if state and state.active_blocks:
            if cache_key:
                self._f._session_classify_cache[cache_key] = (True, time.time())
            return True

        for msg in reversed(messages[-10:]):
            if msg.get("role") != "user":
                continue
            if self._f._commands.has_code_indicators(msg.get("content", "")):
                if cache_key:
                    self._f._session_classify_cache[cache_key] = (True, time.time())
                return True

        if last_user and last_user.get("content", "").strip().startswith("/"):
            if cache_key:
                self._f._session_classify_cache[cache_key] = (True, time.time())
            return True

        if last_user and "```" in last_user.get("content", ""):
            if cache_key:
                self._f._session_classify_cache[cache_key] = (True, time.time())
            return True

        if (
            last_user
            and not state.active_blocks
            and not self._f._commands.has_code_indicators(last_user.get("content", ""))
        ):
            if len(last_user.get("content", "")) > 200:
                blocks, _ = await self._f._code_blocks.extract_code_blocks(
                    last_user.get("content", "")
                )
                if blocks:
                    if cache_key:
                        self._f._session_classify_cache[cache_key] = (True, time.time())
                    return True

        if not last_user or len(last_user.get("content", "")) < 20:
            result = False
        else:
            user_text = last_user.get("content", "")[:300]
            pairs = [
                (
                    user_text,
                    "This message is about programming, code, or software development.",
                ),
                (
                    user_text,
                    "This message is not about programming or code.",
                ),
            ]
            scores = await self._f._commands._predict_cross_encoder(pairs)
            if scores is None:
                self._f._log_debug(
                    "_classify_session: CrossEncoder not loaded, "
                    "falling back to keyword detection."
                )
                result = any(
                    kw in last_user.get("content", "").lower()
                    for kw in (
                        "code",
                        "function",
                        "def",
                        "class",
                        "error",
                        "bug",
                        "traceback",
                    )
                )
            else:
                result = scores[0] > scores[1]

        if cache_key:
            self._f._session_classify_cache[cache_key] = (result, time.time())
            if len(self._f._session_classify_cache) >= 500:
                items = sorted(
                    self._f._session_classify_cache.items(), key=lambda x: x[1][1]
                )
                self._f._session_classify_cache = dict(items[-400:])

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Message utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def ensure_last_message_is_user(self, messages: list) -> list:
        """Ensure the last message in the list is from the user."""
        if not messages:
            messages.append({"role": "user", "content": "continue"})
            return messages
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx == -1:
            while messages and messages[-1].get("role") != "user":
                messages.pop()
            messages.append({"role": "user", "content": "continue"})
        else:
            if last_user_idx + 1 < len(messages):
                messages = messages[: last_user_idx + 1]
        return messages

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Intent classification with continuation inheritance – NEW (E2)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── E2: continuation detection ──────────────────────────────────────────
    _CONTINUATION_RE = re.compile(
        r"^(ok|okay|sí|si|yes|yep|yeah|continúa|continua|continue|"
        r"go on|adelante|sigue|prosigue|hazlo|hazla|genéralo|generate it|"
        r"implementa|implement|de acuerdo|sounds good|great|perfecto|"
        r"bien|vale|listo|done|proceed)\W*$",
        re.IGNORECASE | re.UNICODE,
    )

    def _is_continuation_message(
        self, message: str, classifier_confidence: float
    ) -> bool:
        """E2: detect short continuations with no intent signal.

        Criteria:
        - No code blocks (pasted code always carries intent)
        - Fewer than 15 words in non-code portions
        - Classifier confidence below 0.80 (if classifier is certain, trust it)
        """
        if "```" in message:
            return False
        text_only = re.sub(r"```.*?```", "", message, flags=re.DOTALL).strip()
        if len(text_only.split()) > 15:
            return False
        if classifier_confidence >= 0.80:
            return False
        return True

    async def classify_intent_with_continuation(
        self,
        user_query: str,
        project_id: str,
        intent_vector: Optional[dict] = None,
    ) -> Tuple[str, dict, str]:
        """
        Classify the user intent and apply continuation inheritance (E2).

        If the message is a short continuation (no code, <15 words, low confidence),
        inherit the use_case from the previous turn instead of re-classifying.

        Returns:
            Tuple[str, dict, str]: (use_case_key, profile_copy, human_label)
        """
        # ── 1. Classify intent using the existing classifier ──────────────
        if intent_vector is None:
            intent_vector = await self._f._commands.classify_intent(
                user_query, project_id
            )

        # ── 2. Get the use_case from ContextBuilder ──────────────────────
        use_case_key, profile_copy, human_label = (
            self._f._ctx_builder.classify_use_case(user_query, intent_vector)
        )

        # ── 3. Compute classifier confidence (max probability) ────────────
        confidence = max(intent_vector.values()) if intent_vector else 0.5

        # ── 4. E2: inherit use_case for short continuations ──────────────
        pstate = self._f._project_state_manager.get_pstate(project_id)

        if self._is_continuation_message(user_query, confidence):
            inherited = pstate.get("last_use_case")
            if inherited:
                self._f._log_debug(
                    f"use_case: short continuation — inheriting '{inherited}' "
                    f"from previous turn (classifier had '{use_case_key}' @ {confidence:.2f})"
                )
                use_case_key = inherited
                # Update profile and label to match inherited use_case
                profile_copy = dict(
                    self._f._ctx_builder.LOD_PROFILES.get(inherited, {})
                )
                # Re-derive human label from UseCase enum
                try:
                    from enum import Enum

                    class UseCase(str, Enum):
                        ARCHITECTURE = "A"
                        PLANNING = "B"
                        PROGRAMMING = "C"
                        REFACTORING = "D"
                        SCAFFOLDING = "E"

                        @property
                        def label(self):
                            return {
                                "A": "Architecture/Design",
                                "B": "Planning/Roadmap",
                                "C": "General Programming",
                                "D": "Refactoring/Impact Analysis",
                                "E": "Scaffolding/Boilerplate",
                            }[self.value]

                    human_label = UseCase(inherited).label
                except Exception:
                    human_label = "General Programming"

        # ── 5. Persist the use_case for next turn ──────────────────────────
        pstate["last_use_case"] = use_case_key

        return use_case_key, profile_copy, human_label


class SystemPromptBuilder:
    """Assembles the complete system prompt from two layers: a stable,
    KV‑cache‑friendly Block A (built once per code state) and a dynamic,
    per‑query Block B that injects LTM, activated code, and proactive
    suggestions.

    Provides:
    * ``build(messages, project_id, ...)`` — orchestrates Block A + Block B
      construction, runs parallel checks (contradiction, cache, duplicate
      detection), and returns the static block, dynamic injection list,
      an optional cached response, and the assembled preliminary system
      prompt.
    * Internal helpers for each dynamic source: LTM retrieval, contradiction
      warning injection, activated code context via ``ContextBuilder``,
      proactive cleanup/summarisation suggestions, and persisted
      conversation summaries.

    Docs 10–13 backported:
        M7 – computes and stores block_a_rebuild_reason in pstate.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Main orchestration – MODIFIED (M7)
    # ═══════════════════════════════════════════════════════════════════════════

    async def build(
        self,
        messages: List[dict],
        project_id: str,
        user_query: str,
        user_question: str,
        is_code_session: bool,
        last_user_msg: Optional[dict],
        state: dict,
        slot_free: bool = True,
        intent_vector: Optional[dict] = None,
    ) -> Tuple[str, List[Tuple[str, str]], Optional[dict], str]:
        """
        Orchestrate the construction of the two-block system prompt.

        Returns (static_block, dynamic_injections, cached_response, prelim_system).

        Modified (M7): computes and stores block_a_rebuild_reason in pstate.
        """
        # ── REGION 1: Resolve per-project state ──────────────────────────────
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # ── M7: Get previous state for rebuild reason ──────────────────────────
        prev_block_a_hash = pstate.get("block_a_hash")
        prev_code_hash = pstate.get("code_state_hash")
        prev_graph_mode = pstate.get("resolved_call_graph_mode")

        # ── REGION 2: Build Block A (static) ─────────────────────────────────
        self._f._log_debug("🧱 Block A (static): building / retrieving from cache")
        static_block = await self._f._ctx_builder.build_block_a(
            project_id=project_id,
            is_code_session=is_code_session,
            is_continuation=not slot_free,
        )

        # ── M7: Compute Block A hash and rebuild reason ──────────────────────
        if static_block:
            new_block_a_hash = hashlib.md5(static_block.encode()).hexdigest()[:16]
        else:
            new_block_a_hash = ""

        # ── M7: Determine rebuild reason ──────────────────────────────────────
        if prev_block_a_hash is None:
            rebuild_reason = "first_build"
        elif new_block_a_hash != prev_block_a_hash:
            current_code_hash = self._f._activation.compute_code_state_hash(project_id)
            current_graph_mode = pstate.get("resolved_call_graph_mode")
            if current_code_hash != prev_code_hash:
                rebuild_reason = "code_changed"
            elif current_graph_mode != prev_graph_mode:
                rebuild_reason = "mode_changed"
            else:
                rebuild_reason = "other"  # docstring population, valve change, etc.
        else:
            rebuild_reason = None  # cache hit — no rebuild

        # ── M7: Store for evolution tracking ─────────────────────────────────
        pstate["block_a_hash"] = new_block_a_hash
        pstate["block_a_rebuild_reason"] = rebuild_reason

        # ── REGION 3: Build Hub‑Bodies Tier ──────────────────────────────────
        hub_tier_text, hub_tier_hash, hub_tier_qids = (
            self._f._ctx_builder._build_hub_bodies_tier(project_id)
        )
        pstate["hub_tier_text"] = hub_tier_text
        pstate["hub_tier_hash"] = hub_tier_hash
        pstate["hub_tier_qids"] = hub_tier_qids
        pstate["hub_tier_prev_seeds"] = pstate.get("hub_tier_seeds_this_turn", [])
        pstate["hub_tier_seeds_this_turn"] = list(hub_tier_qids)

        # ── REGION 4: Block B — Dynamic per-query injections ────────────────
        dynamic_injections: List[Tuple[str, str]] = []

        # 4a: Compute use_case label
        use_case, _, use_case_label = self._f._ctx_builder.classify_use_case(
            user_query, intent_vector or {}
        )

        # 4b: LTM retrieval
        self._f._log_debug("🔄 Block B – Step 1/5: LTM per-query retrieval")
        ltm_text = await self._build_ltm_injection(
            project_id,
            user_question,
            user_query,
            is_code_session,
            slot_free,
            use_case_label,
        )
        if ltm_text:
            dynamic_injections.append(("high", ltm_text))

        # 4c: Parallel checks (contradiction, cache, duplicate)
        self._f._log_debug("🔄 Block B – Step 2/5: Parallel checks")
        contradiction_warning, cached_response, duplicate_match = (
            await self._build_parallel_checks(
                messages, user_query, project_id, state, slot_free
            )
        )
        if cached_response:
            return static_block, [], cached_response, ""
        if contradiction_warning and self._f.valves.contradiction_inject_warning:
            dynamic_injections.append(("medium", contradiction_warning))
        if duplicate_match:
            dynamic_injections.append(
                (
                    "medium",
                    f"⚠️ **Note**: Similar question asked before "
                    f"(similarity {duplicate_match['sim']:.2f}).",
                )
            )

        # 4d: Activated code (per-query)
        self._f._log_debug("🔄 Block B – Step 3/5: Code activated by query")
        active_ctx = await self._build_activated_code(
            user_query, project_id, messages, is_code_session, slot_free
        )
        if active_ctx:
            dynamic_injections.append(("critical", active_ctx))

        # 4e: Proactive suggestions
        self._f._log_debug("🔄 Block B – Step 4/5: Proactive suggestions")
        for prio, text in await self._build_suggestions(
            state, project_id, messages, is_code_session
        ):
            dynamic_injections.append((prio, text))

        # 4f: Assemble prelim_system (budget-aware)
        self._f._log_debug("🔄 Block B – Step 5/5: Assemble prelim_system")
        prelim_system = self._assemble_prelim_system(
            static_block,
            hub_tier_text,
            dynamic_injections,
            messages,
        )

        self._f._log_debug("🔄 Block B: complete")
        return static_block, dynamic_injections, None, prelim_system

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Block B – Dynamic injections
    # ═══════════════════════════════════════════════════════════════════════════

    async def _build_ltm_injection(
        self,
        project_id: str,
        user_question: str,
        user_query: str,
        is_code_session: bool,
        slot_free: bool,
        use_case_label: str = "General Programming",
    ) -> Optional[str]:
        """
        Retrieve and format relevant LTM entries for the current query,
        using RAPTOR‑first and thematic expansions.

        The formatted section explicitly indicates that the retrieved content
        comes from long-term memory (past conversations, not the current chat
        history), so the LLM can correctly interpret it as external context.

        Formatting is delegated to _render_ltm_section for clarity and testability.

        Args:
            project_id (str): The current project identifier.
            user_question (str): The cleaned user question (without code spans).
            user_query (str): The full user query (with code).
            is_code_session (bool): Whether the session is a code session.
            slot_free (bool): Whether the LLM slot is free for auxiliary calls.
            use_case_label (str): Human-readable label of the resolved use case.

        Returns:
            Optional[str]: The formatted LTM context, or None if no relevant memories.
        """
        # ------------------------------------------------------------------
        # REGION 1: Early exits
        # ------------------------------------------------------------------
        if not (
            self._f.valves.enable_code_awareness
            and is_code_session
            and HAS_SENTENCE
            and HAS_CHROMA
        ):
            return None

        _ltm_query = user_question if user_question else user_query

        # ------------------------------------------------------------------
        # REGION 2: RAPTOR refinement (if enabled)
        # ------------------------------------------------------------------
        refined_query = _ltm_query
        if (
            self._f.valves.enable_raptor
            and getattr(self._f, "_raptor", None)
            and self._f.memory_collection is not None
        ):
            try:
                raptor_summaries = await self._f._raptor.retrieve(
                    query=_ltm_query,
                    project_id=project_id,
                    top_k=2,
                    embedder=self._f.embedder,
                    chroma_collection=self._f.memory_collection,
                )
                if raptor_summaries:
                    refined_query = _ltm_query + "\n" + "\n".join(raptor_summaries[:2])
            except Exception:
                pass  # fall through to plain query on any error

        # ------------------------------------------------------------------
        # REGION 3: Retrieve memories with thematic expansion
        # ------------------------------------------------------------------
        all_meta = await self._f._ltm.retrieve_memories_unified(
            refined_query,
            project_id,
            use_case_label=use_case_label,
            slot_free=slot_free,
        )

        if not all_meta:
            self._f._log_debug("LTM: no memories retrieved")
            return None

        # ------------------------------------------------------------------
        # REGION 4: Delegate formatting to dedicated renderer
        # ------------------------------------------------------------------
        return self._render_ltm_section(project_id, all_meta)

    def _render_ltm_section(self, project_id: str, memories: list) -> str:
        """
        Render the LTM section with a clear header and per-fragment labels.

        This method is separated from retrieval logic to make formatting
        easier to test and debug independently.

        Args:
            project_id (str): The current project identifier.
            memories (list): List of memory dicts with 'doc', 'timestamp', and 'meta'.

        Returns:
            str: The fully formatted LTM section, or empty string if no valid fragments.
        """
        # ------------------------------------------------------------------
        # REGION 1: Sort and deduplicate
        # ------------------------------------------------------------------
        memories.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
        seen = set()
        unique = []
        for m in memories:
            if m["doc"] not in seen:
                seen.add(m["doc"])
                unique.append(m)

        # ------------------------------------------------------------------
        # REGION 2: Build header with explicit LTM origin
        # ------------------------------------------------------------------
        header = (
            "## Relevant Past Context (long-term memory)\n\n"
            "> The following fragments were retrieved from past conversations "
            "(different sessions) and are provided as additional context. "
            "They are NOT part of the current chat history.\n\n"
        )

        # ------------------------------------------------------------------
        # REGION 3: Render each fragment with proper label
        # ------------------------------------------------------------------
        parts = []
        max_tokens = self._f.valves.ltm_retrieval_max_tokens
        current_tokens = 0

        for mem in unique:
            ts = mem.get("timestamp")

            # Build the label with timestamp if available
            if ts and ts > 1_000_000_000:
                time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%SZ"
                )
                # Check if this is a skeleton snapshot (special formatting)
                if mem.get("meta", {}).get("is_skeleton"):
                    saved_hash = mem.get("meta", {}).get("code_state_hash", "")
                    current_hash = self._f._activation.compute_code_state_hash(
                        project_id
                    )
                    fresh = (
                        "current ✓"
                        if saved_hash and saved_hash == current_hash
                        else "stale — code changed since"
                    )
                    text = f"[Skeleton snapshot {time_str} — {fresh}]\n{mem['doc']}"
                else:
                    text = f"[Past conversation — {time_str}]\n{mem['doc']}"
            else:
                text = f"[Past conversation — unknown date]\n{mem['doc']}"

            # Apply token budget
            tok = self._f._tokens.estimate_code_tokens(text)
            if max_tokens > 0 and current_tokens + tok > max_tokens:
                break
            parts.append(text)
            current_tokens += tok

        if not parts:
            self._f._log_debug("LTM: no parts after truncation")
            return ""

        # ------------------------------------------------------------------
        # REGION 4: Assemble and log
        # ------------------------------------------------------------------
        full_text = header + "\n---\n".join(parts)
        self._f._log_debug(
            f"LTM section rendered ({len(parts)} fragments, ~{current_tokens} tokens)"
        )
        return full_text

    async def _build_parallel_checks(
        self,
        messages: List[dict],
        user_query: str,
        project_id: str,
        state: dict,
        slot_free: bool,
    ) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        """Run contradiction detection, response cache lookup, and duplicate question detection."""
        if not messages:
            return None, None, None

        context_hash = self._f._activation.compute_context_hash(messages)
        contradiction_warning, cached_response, duplicate_match = (
            await self._f._enrichment.parallel_context_checks(
                messages,
                user_query,
                context_hash,
                project_id,
                state,
                skip_contradiction=not slot_free,
                skip_cache=not slot_free,
            )
        )
        return contradiction_warning, cached_response, duplicate_match

    async def _build_activated_code(
        self,
        user_query: str,
        project_id: str,
        messages: List[dict],
        is_code_session: bool,
        slot_free: bool,
    ) -> Optional[str]:
        """Obtain LOD-activated code context for the current query, if applicable."""
        if not (is_code_session and self._f.valves.enable_code_awareness):
            return None

        if self._f.valves.enable_path_analysis:
            intent_vector = await self._f._commands.classify_intent(
                user_query, project_id
            )
            active_ctx = await self._f._ctx_builder.build_block_b(
                project_id=project_id,
                query=user_query,
                messages=messages,
                slot_free=slot_free,
                intent_vector=intent_vector,
                is_continuation=not slot_free,
            )

            # --- Check suppression before falling back ---
            # Replicate the suppression check from build_block_b to avoid
            # falling back to full context when the skeleton tier is active.
            active_use_case, _, _ = self._f._ctx_builder.classify_use_case(
                user_query, intent_vector
            )
            suppress_sigs = (
                self._f.valves.skeleton_tier_suppresses_block_b_signatures
                and active_use_case != "D"
                and self._f._ctx_builder._is_skeleton_tier_active(project_id)
            )

            if active_ctx:
                return active_ctx
            elif suppress_sigs:
                # If suppression is active, don't fall back to full context
                self._f._log_debug(
                    "Skeleton tier active and suppress_sigs=True, "
                    "skipping fallback to active_code_context to avoid duplication"
                )
                return ""
            else:
                # Fallback to full context
                return self._f._activation.get_active_code_context(
                    project_id, user_query
                )

        return self._f._activation.get_active_code_context(project_id, user_query)

    async def _build_suggestions(
        self,
        state: dict,
        project_id: str,
        messages: List[dict],
        is_code_session: bool,
    ) -> List[Tuple[str, str]]:
        """Collect proactive cleanup, summarization, and command suggestions."""
        suggestions: List[Tuple[str, str]] = []

        # ── Cleanup suggestions ───────────────────────────────────────
        if (
            self._f.valves.cleanup_suggestions_enabled
            and self._f.valves.cleanup_proactive_suggestions
            and is_code_session
        ):
            candidates = self._f._activation.get_inactive_block_candidates(project_id)
            if candidates:
                last_sugg_idx = state.last_cleanup_suggestion_msg_idx
                if (
                    state.message_count - last_sugg_idx
                    >= self._f.valves.cleanup_suggestion_cooldown_messages
                ):
                    suggestions.append(
                        (
                            "low",
                            f"[CodeAware] {len(candidates)} inactive block(s). "
                            f"Use `/status` or `/clean`.",
                        )
                    )
                    state.last_cleanup_suggestion_msg_idx = state.message_count
                    self._f._conversation_state_manager.set(project_id, state)

        # ── Token pressure suggestion ──────────────────────────────────
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._f._tokens.estimate_tokens(sys_msgs + history_msgs)
        if self._f.valves.context_window_tokens > 0:
            suggestion = (
                await self._f._history_compressor.check_and_suggest_summarization(
                    project_id, total_tokens, self._f.valves.context_window_tokens
                )
            )
            if suggestion:
                suggestions.append(("low", suggestion))

        # ── Command suggestion ─────────────────────────────────────────
        cmd_suggestion = await self._f._commands.suggest_commands(project_id, state)
        if cmd_suggestion:
            suggestions.append(("low", cmd_suggestion))

        # ── Persisted conversation summaries ───────────────────────────
        summaries = state.conversation_summaries
        if summaries:

            def _summary_header(s: dict) -> str:
                ct = s.get("covers_turns")
                turns = f", turns {ct[0]}-{ct[1]}" if ct else ""
                tag = (
                    "Consolidated summary"
                    if s.get("level", 1) >= 2
                    else "Summary of earlier conversation"
                )
                ts = datetime.fromtimestamp(
                    s.get("created_at", 0), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M")
                return f"[{tag}{turns} — {ts}]"

            joined = "\n\n".join(
                f"{_summary_header(s)}\n{s['text']}" for s in summaries
            )
            suggestions.append(("medium", joined))

        return suggestions

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Final assembly (budget‑aware)
    # ═══════════════════════════════════════════════════════════════════════════

    def _assemble_prelim_system(
        self,
        static_block: str,
        hub_tier: str,
        dynamic_injections: List[Tuple[str, str]],
        messages: List[dict],
    ) -> str:
        """
        Assemble the preliminary system prompt with token budget constraints.

        The order is:
            1. static_block (Block A)
            2. hub_tier (stable full bodies of top hubs)
            3. dynamic_block (Block B: LOD, pointers, LTM, suggestions, etc.)
            4. user's original system prompt (captured separately)

        This order ensures that the stable prefix (A + tier) is KV-cacheable
        across turns, while the dynamic tail re-prefills as needed.
        """
        budget = self._f.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self._f.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected: List[str] = []
            used = 0
            static_tokens = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            hub_tier_tokens = len(self._f.tokenizer.encode(hub_tier)) if hub_tier else 0
            # User system prompt is never truncated
            user_prompt_tokens = (
                len(
                    self._f.tokenizer.encode(
                        getattr(self._f, "_original_system_prompt", "") or ""
                    )
                )
                if getattr(self._f, "_original_system_prompt", "")
                else 0
            )
            dyn_budget = max(
                0, budget - static_tokens - hub_tier_tokens - user_prompt_tokens
            )

            for prio, text in dynamic_injections:
                if not text:
                    continue
                tok = len(self._f.tokenizer.encode(text))
                if used + tok <= dyn_budget:
                    selected.append(text)
                    used += tok
                elif prio in ("critical", "high"):
                    avail = dyn_budget - used
                    if avail > 20:
                        selected.append(text[: avail * 4] + "\n[truncated]")
                        break
            dynamic_block = "\n\n".join(selected)
        else:
            dynamic_block = "\n\n".join(t for _, t in dynamic_injections if t)

        # ── Assemble with tier between static and dynamic ──
        separator = "\n\n---\n\n"
        parts = [p for p in [static_block, hub_tier, dynamic_block] if p.strip()]
        prelim_system = separator.join(parts)

        # Append user's original system prompt if present (captured once per turn)
        base_content = getattr(self._f, "_original_system_prompt", "") or ""
        if base_content.strip():
            prelim_system = (
                prelim_system + separator + "## User instructions\n" + base_content
            )

        return prelim_system


class WindowManager:
    """
    Single owner of the prose history window policy.

    Replaces (in Phase 1, step 3):
        MessageAssembler._apply_turn_based_window     (M1)
        MessageAssembler._trim_and_summarize           (M3 + M4)
        MessageAssembler._index_turns
        MessageAssembler._summary_sort_key

    Does not replace:
        MessageAssembler._apply_history_llmlingua      (M2, decoupled)
        HistoryCompressor.summarize_messages           (helper)
        HistoryCompressor._consolidate_summaries       (helper)
        HistoryCompressor._merge_summaries             (helper)
        MessageAssembler._persist_turn_summary_to_ltm  (helper)

    Complexity guarantee:
        Context history: O(K) bounded by history_max_tokens.
        Total session (T turns): O(T) linear — same as v9.0.0.

    Seams for future phases:
        _on_frontier_advance(old_hwm, new_hwm)  → Phase 2 (KV-freeze)
        _frontier_c_turn                        → Phase 3 (LLMLingua)
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the WindowManager with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        # Phase 3 seam: turn that separates raw band from compressible band.
        # In Phase 1 it is updated but nobody reads it.
        self._frontier_c_turn: int = 0

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Public entry point
    # ═══════════════════════════════════════════════════════════════════════

    async def apply(
        self,
        messages: List[dict],
        state: ConversationState,
        project_id: str,
        slot_free: bool,
    ) -> Tuple[List[dict], str]:
        """
        Apply the history window policy.

        Args:
            messages:    full list (system + history)
            state:       ConversationState for the project (persistent)
            project_id:  project identifier
            slot_free:   if False, no summaries are generated

        Returns:
            (messages_final, pending_summary)
            - messages_final: system messages at front + trimmed history
            - pending_summary: text of the most recent summary (or "")
              for injection into the system prompt — same contract
              as the pending_summary returned by _trim_and_summarize.
        """
        v = self._f.valves

        # ── 0. Clean orphan tool calls at front ──────────────────────────
        # (migrated from _trim_and_summarize preserve_tool_calls)
        if v.preserve_tool_calls:
            while messages and messages[0].get("role") == "tool":
                messages = messages[1:]
            if (
                messages
                and messages[0].get("role") == "assistant"
                and messages[0].get("tool_calls")
            ):
                tool_call_ids = {tc.get("id") for tc in messages[0]["tool_calls"]}
                tool_response_ids = {
                    m.get("tool_call_id")
                    for m in messages[1:]
                    if m.get("role") == "tool"
                }
                if not tool_call_ids.issubset(tool_response_ids):
                    messages = messages[1:]

        # ── 1. AutoContinue deferral ──────────────────────────────────────
        if v.compaction_defer_during_autocontinue and self._is_autocontinue_active(
            messages
        ):
            self._f._log_debug(
                "WindowManager: deferral — AutoContinue active, no changes"
            )
            return messages, ""

        # ── 2. Separate system / history ──────────────────────────────────
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        history = [m for m in messages if m.get("role") != "system"]
        if not history:
            return messages, ""

        # ── 3. Effective budget ──────────────────────────────────────────
        budget = self._effective_budget(project_id)

        # ── 4. Turn indexing ──────────────────────────────────────────────
        turns, _total_turns = self._index_turns(history)

        # ── 5. Compute frontier ───────────────────────────────────────────
        kept, old_msgs, cut_turn = self._compute_frontier(history, turns, budget)
        self._frontier_c_turn = cut_turn  # Phase 3 seam

        # ── 6. Everything fits ────────────────────────────────────────────
        if not old_msgs:
            return sys_msgs + kept, ""

        # ── 7. Emergency cap ──────────────────────────────────────────────
        kept, old_msgs = self._apply_emergency_cap(
            history, turns, kept, old_msgs, budget
        )
        if not old_msgs:
            return sys_msgs + kept, ""

        # ── 8. Minimum batch ──────────────────────────────────────────────
        old_turn_nums = {t for m, t in zip(history, turns) if m in old_msgs}
        if len(old_turn_nums) < v.summarize_batch_turns:
            self._f._log_debug(
                f"WindowManager: batch too small "
                f"({len(old_turn_nums)} < {v.summarize_batch_turns} turns), "
                "keeping raw"
            )
            return sys_msgs + history, ""

        # ── 9. Summarize batch ────────────────────────────────────────────
        if not slot_free:
            self._f._log_debug("WindowManager: no free slot, keeping raw history")
            return sys_msgs + history, ""

        has_code = any("```" in m.get("content", "") for m in old_msgs)
        summary_text = await self._f._history_compressor.summarize_messages(
            old_msgs, is_code_context=has_code
        )

        if not summary_text or not summary_text.strip():
            self._f._log_debug(
                "WindowManager: summary failed (no-degradation guard), "
                "keeping raw history"
            )
            return sys_msgs + history, ""

        # ── 10. Persist ──────────────────────────────────────────────────
        pending = await self._persist(
            summary_text=summary_text,
            old_msgs=old_msgs,
            turns=turns,
            history=history,
            state=state,
            project_id=project_id,
            slot_free=slot_free,
        )

        # ── 11. Return trimmed history ──────────────────────────────────
        return sys_msgs + kept, pending

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Calculation helpers (no side effects)
    # ═══════════════════════════════════════════════════════════════════════

    def _effective_budget(self, project_id: str) -> int:
        """
        Token budget for history.
        Token-primary: history_max_tokens, bounded by the actual window.
        Always returns a value >= 0.
        """
        v = self._f.valves
        pstate = self._f._project_state_manager.get_pstate(project_id)
        system_tokens = pstate.get("last_system_tokens", 0)
        budget = min(
            v.history_max_tokens,
            v.context_window_tokens - system_tokens - v.response_reserve_tokens,
        )
        return max(0, budget)

    def _token_count(self, text: str) -> int:
        """
        Count tokens with tokenizer if available, else 4 chars/token heuristic.
        """
        if self._f.tokenizer:
            return len(self._f.tokenizer.encode(text))
        return max(1, len(text) // 4)

    @staticmethod
    def _index_turns(history: List[dict]) -> Tuple[List[int], int]:
        """
        Assign a 1-based turn number to each message.
        Each user message increments the counter.
        Assistant/tool messages share the turn number of the preceding user.
        Returns (turns_per_message, total_turns).
        """
        turn = 0
        per_msg: List[int] = []
        for m in history:
            if m.get("role") == "user":
                turn += 1
            per_msg.append(max(turn, 1))
        return per_msg, turn

    def _compute_frontier(
        self,
        history: List[dict],
        turns: List[int],
        budget: int,
    ) -> Tuple[List[dict], List[dict], int]:
        """
        Compute the window frontier.
        Iterates from the most recent message backwards, accumulating tokens.
        The frontier is aligned to full turn boundaries (never cuts a turn in half).

        Returns (kept, old_msgs, cut_turn).
        cut_turn == 0 means everything fits in the budget.
        """
        accumulated = 0
        cut_turn = 0
        token_counts = [self._token_count(m.get("content", "")) for m in history]

        for i in range(len(history) - 1, -1, -1):
            tok = token_counts[i]
            if accumulated + tok > budget:
                # Align to the full turn boundary
                cut_turn = turns[i]
                break
            accumulated += tok

        if cut_turn == 0:
            return history, [], 0

        kept = [m for m, t in zip(history, turns) if t > cut_turn]
        old_msgs = [m for m, t in zip(history, turns) if t <= cut_turn]
        return kept, old_msgs, cut_turn

    def _apply_emergency_cap(
        self,
        history: List[dict],
        turns: List[int],
        kept: List[dict],
        old_msgs: List[dict],
        budget: int,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Emergency cap: if any turn in `kept` exceeds budget * 0.8
        (giant turn — e.g. a 5000-line file), keep only the last
        emergency_max_turns turns and move the rest to old_msgs.

        Replaces adaptive_trim with max_turns=8, with a more conservative
        default (4) and configurable via valve.
        """
        if not kept:
            return kept, old_msgs

        emergency_threshold = budget * 0.8
        kept_turn_set = {t for m, t in zip(history, turns) if m in kept}

        giant_found = False
        for turn_num in kept_turn_set:
            turn_tokens = sum(
                self._token_count(m.get("content", ""))
                for m, t in zip(history, turns)
                if t == turn_num
            )
            if turn_tokens > emergency_threshold:
                giant_found = True
                break

        if not giant_found:
            return kept, old_msgs

        emergency_max = getattr(self._f.valves, "emergency_max_turns", 4)
        sorted_kept_turns = sorted(kept_turn_set)
        turns_to_keep = set(sorted_kept_turns[-emergency_max:])
        turns_to_evict = kept_turn_set - turns_to_keep

        new_kept = [m for m, t in zip(history, turns) if t in turns_to_keep]
        extra_old = [m for m, t in zip(history, turns) if t in turns_to_evict]

        self._f._log_debug(
            f"WindowManager: emergency cap — giant turn detected "
            f"(>{emergency_threshold:.0f} tokens). "
            f"Keeping last {emergency_max} turns."
        )
        return new_kept, old_msgs + extra_old

    @staticmethod
    def _summary_sort_key(s: dict) -> float:
        """Chronological order by covered band start."""
        ct = s.get("covers_turns")
        return float(ct[0]) if ct else float(s.get("created_at", 0))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Persistence (side effects: state, LTM, consolidation)
    # ═══════════════════════════════════════════════════════════════════════

    async def _persist(
        self,
        summary_text: str,
        old_msgs: List[dict],
        turns: List[int],
        history: List[dict],
        state: ConversationState,
        project_id: str,
        slot_free: bool,
    ) -> str:
        """
        Persist the generated summary:
          a. Unified metadata → conversation_summaries.
          b. L1 cap.
          c. Update summarized_turn_hwm (manager owns the hwm).
          d. Persist to LTM (synchronous — closes the wait=False race).
          e. Consolidate L1→L2 if enough L1 summaries exist.
          f. Phase 2 seam.
          g. Persist state.

        Returns the formatted pending_summary for injection into the
        system prompt (same format as _trim_and_summarize returned).
        """
        v = self._f.valves
        old_hwm = state.summarized_turn_hwm

        old_turn_nums = [t for m, t in zip(history, turns) if m in old_msgs]
        new_hwm = max(old_turn_nums) if old_turn_nums else old_hwm

        # a. Unified metadata (covers_turns always present)
        summary_entry = {
            "text": summary_text,
            "created_at": time.time(),
            "covers_msgs": len(old_msgs),
            "covers_turns": [old_hwm + 1, new_hwm],
            "level": 1,
        }
        state.conversation_summaries.append(summary_entry)

        # b. L1 cap
        max_l1 = v.max_conversation_summaries
        if max_l1 > 0:
            l1 = [s for s in state.conversation_summaries if s.get("level", 1) == 1]
            if len(l1) > max_l1:
                keep_ids = {id(s) for s in l1[-max_l1:]}
                state.conversation_summaries = [
                    s
                    for s in state.conversation_summaries
                    if s.get("level", 1) != 1 or id(s) in keep_ids
                ]

        # c. Manager owns the hwm
        state.summarized_turn_hwm = new_hwm

        # ── Instrumentation: write metrics to state, not pstate ──
        state.wm_fired = True
        state.wm_summary_ok = True
        state.wm_msgs_evicted = len(old_msgs)
        state.wm_turns_evicted = new_hwm - old_hwm

        self._f._log_debug(
            f"WindowManager: L1 summary generated "
            f"(turns {old_hwm + 1}–{new_hwm}, {len(old_msgs)} msgs)"
        )

        # d. Persist to LTM (synchronous)
        await self._f._message_assembler._persist_turn_summary_to_ltm(
            summary_text, project_id, old_hwm + 1, new_hwm
        )

        # e. Consolidate L1→L2
        l1_count = sum(
            1 for s in state.conversation_summaries if s.get("level", 1) == 1
        )
        if (
            slot_free
            and v.enable_hierarchical_summaries
            and l1_count >= v.hierarchical_summary_group_size
        ):
            await self._f._message_assembler._consolidate_summaries(
                state, project_id, slot_free
            )

        # f. Phase 2 seam (no-op in Phase 1)
        self._on_frontier_advance(old_hwm, new_hwm)

        # g. Persist state using ConversationStateManager
        self._f._conversation_state_manager.set(project_id, state)

        return f"[Summary of earlier conversation]\n{summary_text}"

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Seams and static helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _on_frontier_advance(self, old_hwm: int, new_hwm: int) -> None:
        """
        Phase 2: KV-freeze after history eviction.

        When WindowManager evicts turns and generates a summary, the history
        before the frontier disappears from the context. If the server's KV
        cache still contains those pre-filled tokens, subsequent requests
        would have to re-prefill the entire new system+history sequence.

        By saving the slot immediately after eviction (force=True), the .bin
        file captures the KV state with the new stabilized prefix (Block A + summary).
        The next request restores from that checkpoint instead of re-prefilling
        from scratch.

        force=True bypasses the static hash guard (history changed, but Block A
        did not, so the hash doesn't change and without force the slot would
        not be saved). The slot_save_max_context_tokens guard is always respected.
        """
        if old_hwm >= new_hwm:
            return  # no real advance, nothing to freeze

        self._f._log_debug(
            f"Phase 2 KV-freeze: frontier advanced hwm {old_hwm}→{new_hwm}, "
            "scheduling slot_save(force=True)"
        )

        # asyncio.create_task works because _on_frontier_advance is called
        # from _persist(), which is a coroutine running inside the event loop.
        project_id = self._f._inlet_orch.get_project_id()
        asyncio.create_task(
            self._f._project_state_manager.slot_save(project_id, force=True)
        )

    @staticmethod
    def _is_autocontinue_active(messages: List[dict]) -> bool:
        """
        True if the last assistant message contains a multi-phase
        continuation marker.
        """
        _MARKERS = frozenset(
            {
                "▶ CONTINÚA:",
                "▶ CONTINÚA EN LA SIGUIENTE PARTE",
            }
        )
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"),
            None,
        )
        if not last_assistant:
            return False
        return any(marker in last_assistant.get("content", "") for marker in _MARKERS)


class MessageAssembler:
    """
    Final processing of the message list before it is sent to the LLM.

    Orchestrates, in order:
    * Chain‑of‑Thought detection and reasoning generation (Level 1‑3).
    * Code‑history compression and lean‑user‑code stubbing.
    * LLMLingua‑2 compression of conversation prose.
    * Turn‑based window management (summarise then evict old turns).
    * Multi‑phase protocol injection when the token budget is tight.
    * Adaptive trimming of old messages with optional summarisation.
    * Assembly of the final system prompt (Block A + Block B) and its
        injection as the first message.

    Docs 10–13 backported:
        M3 – enforce_scientific_method forces cot_any_used=True to guarantee generation.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the MessageAssembler with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        self._last_cot_degraded: bool = False
        # WindowManager: unified history window policy (replaces M1, M3, M4)
        self._window_manager = WindowManager(filter_ref)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Main orchestration
    # ═══════════════════════════════════════════════════════════════════════

    async def assemble(
        self,
        messages: List[dict],
        project_id: str,
        static_block: str,
        dynamic_injections: List[Tuple[str, str]],
        prelim_system: str,
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: ConversationState,
        __user__: Optional[dict],
        user_question: str,
        has_code_blocks: bool,
        slot_free: bool = True,
    ) -> List[dict]:
        """
        Orchestrate CoT, multi-phase, trimming, and final assembly.

        Args:
            messages: The current list of conversation messages.
            project_id: The project identifier.
            static_block: The rendered Block A (static, KV-cacheable).
            dynamic_injections: List of (priority, text) dynamic content.
            prelim_system: The preliminary system prompt (Block A + Block B).
            last_user_msg: The last user message, if any.
            is_code_session: Whether the session is code-aware.
            state: The ConversationState for the project.
            __user__: The user context from OpenWebUI.
            user_question: The extracted question from the user message.
            has_code_blocks: Whether the user message contained code fences.
            slot_free: Whether the LLM slot is free for auxiliary calls.

        Returns:
            List[dict]: The final message list ready for the LLM.
        """
        self._f._log_debug(
            "Assembling final messages (CoT, trimming, system prompt injection)"
        )

        # 1. CoT detection and generation (modifies dynamic_injections in-place)
        await self._detect_and_generate_cot(
            dynamic_injections,
            last_user_msg,
            is_code_session,
            state,
            user_question,
            prelim_system,
            project_id,
            slot_free,
            messages,
        )

        # 2. Code history compression + lean user code FIRST.
        messages = await self._compress_code_history_and_lean(
            messages, project_id, dynamic_injections
        )

        # 3. History LLMLingua compression (prose only — code is already stubbed
        #    here or skipped verbatim by ConversationCompressor, see FIX 4b).
        messages = await self._apply_history_llmlingua(
            messages, project_id, user_question
        )

        # 4. WindowManager: unified history window policy
        #    Replaces: _apply_turn_based_window (M1) + _trim_and_summarize (M3 + M4)
        #    ✅ state se pasa directamente (ya es ConversationState)
        messages, pending_summary = await self._window_manager.apply(
            messages, state, project_id, slot_free
        )

        # 5. Multi-phase instructions injection (token math is now accurate:
        #    history was leaned/compressed/windowed in steps 2-4).
        await self._inject_multi_phase_instructions(
            dynamic_injections,
            prelim_system,
            messages,
            user_question,
            slot_free,
            project_id,  # ← NEW: pass project_id for global-scope flag
        )

        # 6. Trim and summarize old messages (now handled by WindowManager)
        #    No action needed here; pending_summary is already populated.

        # 7. Assemble final system message and inject into message list
        messages = self._assemble_final_system_and_log(
            static_block, dynamic_injections, messages, project_id, pending_summary
        )

        return messages

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Chain‑of‑Thought (CoT) detection and generation – MODIFIED (M3)
    # ═══════════════════════════════════════════════════════════════════════

    async def _detect_and_generate_cot(
        self,
        dynamic_injections: List[Tuple[str, str]],
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: ConversationState,
        user_question: str,
        prelim_system: str,
        project_id: str,
        slot_free: bool,
        messages: List[dict],
    ) -> None:
        """
        Detect CoT level and generate reasoning.
        Modifies `dynamic_injections` in‑place.

        Modified (M3): enforce_scientific_method forces cot_any_used=True.
        """
        # ══════════════════════════════════════════════════════════════
        # REGION 1 — DETECT COT LEVEL
        # ══════════════════════════════════════════════════════════════
        self._f._log_debug("🧠 ENRICHMENT – CoT Step 1/3: Detect CoT level")
        manual_cot_used = False
        cot_any_used = False
        cot_level = 2
        reasoning = None
        cot_question = ""
        user_content = last_user_msg.get("content", "") if last_user_msg else ""

        # ── M3: enforce_scientific_method forces level 3 and cot_any_used ──
        if self._f.valves.enforce_scientific_method:
            self._f._log_debug("CoT: enforce_scientific_method=True → forcing Level 3")
            cot_level = 3
            cot_any_used = True  # ← M3: force generation path

        if self._f.valves.enable_cot_on_demand or self._f.valves.auto_cot_enabled:
            if (
                last_user_msg
                and self._f.valves.enable_cot_on_demand
                and user_content.strip().startswith("/think")
            ):
                cot_question, level = await self._f._reasoning.parse_cot_intent(
                    user_content
                )
                if cot_question:
                    manual_cot_used = True
                    cot_any_used = True
                    cot_level = level
                    if level == 1:
                        dynamic_injections.append(
                            (
                                "high",
                                "Please think step by step before answering. "
                                "Show your reasoning, then provide the final answer.",
                            )
                        )

        # Parallel CrossEncoder tasks (keep_full_code + auto CoT detection)
        if (
            slot_free
            and not manual_cot_used
            and not self._f.valves.enforce_scientific_method
        ):
            parallel_tasks = []
            _available_mp_pre = self._f.valves.context_window_tokens
            if (
                self._f.valves.enable_multi_phase_response
                and self._f.tokenizer
                and prelim_system
            ):
                _prelim_tok = len(self._f.tokenizer.encode(prelim_system))
                _hist_tok = self._f._tokens.estimate_tokens(
                    [m for m in messages if m.get("role") != "system"]
                )
                _available_mp_pre = max(
                    0, self._f.valves.context_window_tokens - _prelim_tok - _hist_tok
                )
            _skip_intent_llm = (
                self._f.valves.enable_multi_phase_response
                and _available_mp_pre < self._f.valves.multi_phase_response_threshold
            )

            if not _skip_intent_llm:
                parallel_tasks.append(
                    self._f._llm_orchestrator.should_keep_full_code(user_question)
                )
            else:
                parallel_tasks.append(asyncio.sleep(0, result=True))

            if self._f.valves.enable_cot_on_demand or self._f.valves.auto_cot_enabled:
                cot_detection_content = (
                    user_question
                    if (user_question and len(user_question) >= 10)
                    else user_content
                )
                parallel_tasks.append(
                    self._f._reasoning.detect_cot_level(
                        cot_detection_content, is_code_session, state
                    )
                )
            else:
                parallel_tasks.append(asyncio.sleep(0, result=0))

            results = await asyncio.gather(*parallel_tasks)
            self._f._user_intent_full_code = (
                results[0] if not _skip_intent_llm else True
            )
            detected_level = (
                results[1]
                if (
                    self._f.valves.enable_cot_on_demand
                    or self._f.valves.auto_cot_enabled
                )
                else 0
            )
            if detected_level > 0:
                cot_any_used = True
                cot_level = detected_level
                self._f._log_debug(
                    f"🧠 ENRICHMENT – CoT Step 1/3: Detected level {cot_level}"
                )
        else:
            self._f._user_intent_full_code = True
            if (
                not manual_cot_used
                and slot_free
                and not self._f.valves.enforce_scientific_method
            ):
                cot_detection_content = (
                    user_question
                    if (user_question and len(user_question) >= 10)
                    else user_content
                )
                cot_level = await self._f._reasoning.detect_cot_level(
                    cot_detection_content, is_code_session, state
                )
                if cot_level > 0:
                    cot_any_used = True
                    self._f._log_debug(
                        f"🧠 ENRICHMENT – CoT Step 1/3: Detected level {cot_level}"
                    )

        if not cot_any_used:
            self._f._log_debug("🧠 ENRICHMENT – CoT Step 1/3: No CoT needed")
            return
        if not slot_free:
            self._f._log_debug(
                "🧠 ENRICHMENT – CoT Step 1/3: CoT detection skipped (no free slot)"
            )
            return

        # Multi-phase pre-check: degrade CoT if context is tight
        _mp_cot_degraded = False
        _available_mp_pre = self._f.valves.context_window_tokens
        if (
            self._f.valves.enable_multi_phase_response
            and self._f.tokenizer
            and prelim_system
        ):
            _prelim_tok = len(self._f.tokenizer.encode(prelim_system))
            _hist_tok = self._f._tokens.estimate_tokens(
                [m for m in messages if m.get("role") != "system"]
            )
            _available_mp_pre = max(
                0, self._f.valves.context_window_tokens - _prelim_tok - _hist_tok
            )
        if (
            self._f.valves.enable_multi_phase_response
            and _available_mp_pre < self._f.valves.multi_phase_response_threshold
            and cot_any_used
            and cot_level >= 2
            and slot_free
        ):
            self._f._log_debug(
                f"🧠 Multi-phase pre-check: {_available_mp_pre} tokens available "
                f"< threshold {self._f.valves.multi_phase_response_threshold}. "
                f"Degrading CoT Level {cot_level} → 1 "
                f"(Fase 1 of the protocol absorbs the reasoning)."
            )
            cot_level = 1
            _mp_cot_degraded = True

        self._last_cot_degraded = _mp_cot_degraded

        # ══════════════════════════════════════════════════════════════
        # REGION 2 — GENERATE REASONING
        # ══════════════════════════════════════════════════════════════
        self._f._log_debug("🧠 ENRICHMENT – CoT Step 2/3: Generate reasoning")
        _model_ctx = self._f.valves.active_context_max_tokens or 28000
        _cot_context_limit = _model_ctx // 3

        # Architecture-mode: replace the full system prompt with the skeleton
        _is_arch = (
            self._f.valves.enable_skeleton_cot
            and self._f._reasoning.is_architecture_query(user_question)
        )
        _skeleton_ctx = ""
        if _is_arch:
            self._f._log_debug(
                "🏗️ Architecture intent detected — fetching skeleton for CoT prior"
            )
            try:
                _skeleton_ctx = await self._f._ctx_builder._get_skeleton_for_cot(
                    project_id, user_question
                )
            except Exception as _ske:
                self._f._log_debug(f"Skeleton for CoT failed: {_ske}")
            if _skeleton_ctx:
                prelim_for_cot = _skeleton_ctx
                self._f._log_debug(
                    f"🏗️ CoT context = skeleton "
                    f"(~{self._f._tokens.estimate_code_tokens(_skeleton_ctx)} tokens)"
                )
            else:
                _is_arch = False  # no skeleton available; fall back to normal CoT
                self._f._log_debug(
                    "🏗️ No skeleton available — falling back to standard CoT context"
                )

        if not _is_arch:
            if self._f.tokenizer:
                _prelim_tokens = len(self._f.tokenizer.encode(prelim_system))
                if _prelim_tokens > _cot_context_limit:
                    prelim_for_cot = self._f._tokens.truncate_text_to_tokens(
                        prelim_system, _cot_context_limit
                    )
                else:
                    prelim_for_cot = prelim_system
            else:
                prelim_for_cot = prelim_system[: _cot_context_limit * 4]

        if not manual_cot_used:
            question = user_question
            if (
                _is_arch
                and cot_level == 3
                and self._f.valves.enable_scientific_arch_reasoning
            ):
                reasoning = (
                    await self._f._reasoning.generate_scientific_architecture_reasoning(
                        question, prelim_for_cot, project_id, label="sci_arch_cot"
                    )
                )
            elif _is_arch and cot_level >= 2:
                reasoning = await self._f._reasoning.generate_architecture_reasoning(
                    question, prelim_for_cot, project_id, label="arch_cot"
                )
            elif cot_level == 2:
                reasoning = await self._f._reasoning.generate_cot_reasoning(
                    question, prelim_for_cot
                )
            elif cot_level == 3:
                reasoning = await self._f._reasoning.generate_scientific_reasoning_L3(
                    question, prelim_for_cot, project_id, label="scientific_cot"
                )
        else:
            if (
                _is_arch
                and cot_level == 3
                and self._f.valves.enable_scientific_arch_reasoning
            ):
                reasoning = (
                    await self._f._reasoning.generate_scientific_architecture_reasoning(
                        cot_question, prelim_for_cot, project_id, label="sci_arch_cot"
                    )
                )
            elif _is_arch and cot_level >= 2:
                reasoning = await self._f._reasoning.generate_architecture_reasoning(
                    cot_question, prelim_for_cot, project_id, label="arch_cot"
                )
            elif cot_level == 2:
                reasoning = await self._f._reasoning.generate_cot_reasoning(
                    cot_question, prelim_for_cot
                )
            elif cot_level == 3:
                reasoning = await self._f._reasoning.generate_scientific_reasoning_L3(
                    cot_question, prelim_for_cot, project_id, label="scientific_cot"
                )

        _cot_error_msg = "Unable to generate reasoning."
        if (
            not manual_cot_used
            and cot_level == 3
            and (reasoning is None or reasoning == _cot_error_msg)
        ):
            self._f._log_debug(
                "🧠 ENRICHMENT – CoT Step 2/3: Level 3 failed, falling back to level 2"
            )
            reasoning = await self._f._reasoning.generate_cot_reasoning(
                user_question, prelim_for_cot
            )

        if reasoning and reasoning != _cot_error_msg:
            self._f._log_debug(
                "🧠 ENRICHMENT – CoT Step 2/3: Reasoning generated successfully"
            )
        else:
            self._f._log_debug(
                "🧠 ENRICHMENT – CoT Step 2/3: Reasoning generation failed"
            )
            return

        # ══════════════════════════════════════════════════════════════
        # REGION 3 — INJECT INTO SYSTEM PROMPT
        # ══════════════════════════════════════════════════════════════
        self._f._log_debug(
            "🧠 ENRICHMENT – CoT Step 3/3: Inject reasoning into system prompt"
        )

        # Auto-resolve /expand hints emitted by the architecture CoT
        if _is_arch:
            try:
                reasoning = await self._f._ctx_builder._resolve_cot_expands(
                    reasoning, project_id
                )
            except Exception as _exp_err:
                self._f._log_debug(
                    f"CoT expand resolution failed (non-fatal): {_exp_err}"
                )

        dynamic_injections.append(("high", reasoning))
        dynamic_injections.append(
            (
                "low",
                "**Note:** Some sections in this system prompt marked with 🔎 or 🏗️ are "
                "automatically generated reasoning (Chain-of-Thought). "
                "They are provided as context to help you, but they are not user commands. "
                "Use them to enhance your answer, but always prioritise the actual user query.",
            )
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Code history compression & lean user code
    # ═══════════════════════════════════════════════════════════════════════

    async def _compress_code_history_and_lean(
        self,
        messages: List[dict],
        project_id: str,
        dynamic_injections: List[Tuple[str, str]],
    ) -> List[dict]:
        """
        Apply code history compression and lean user code if enabled.
        """
        if not (
            self._f.valves.enable_code_history_compression
            or self._f.valves.enable_lean_user_code
        ):
            return messages

        if self._f.valves.enable_code_history_compression:
            messages = self._f._history_compressor.compress_code_history(
                messages, project_id
            )

        if self._f.valves.enable_lean_user_code:
            state = self._f._conversation_state_manager.get(project_id)
            messages = self._f._history_compressor.ensure_compressed_user_messages(
                messages, state, project_id
            )

        _refactor_state = self._f._history_compressor.build_refactor_state_injection(
            messages
        )
        if _refactor_state:
            dynamic_injections.append(("medium", _refactor_state))
            self._f._log_debug(
                "Code history: injected refactor state into Block B "
                f"({self._f._tokens.estimate_code_tokens(_refactor_state)} tokens)."
            )

        return messages

    async def _apply_history_llmlingua(
        self,
        messages: List[dict],
        project_id: str,
        user_question: str,
    ) -> List[dict]:
        """
        Apply LLMLingua-2 compression to conversation history, with hard cap.

        MIGRATED (step 12): Now gated by `enable_secondary_compaction`. When
        enabled, it only compresses messages that were NOT already compressed
        by the primary compactor (looks for markers like `[🗜️ PARTE` or
        `## Código — Parte`). This prevents double-compression in cascade.

        Args:
            messages (List[dict]): The conversation messages.
            project_id (str): The current project ID.
            user_question (str): The user's query for question-aware compression.

        Returns:
            List[dict]: The messages with secondary compression applied (or unchanged).
        """
        # --- 1. Gate: secondary compaction is off by default ---
        if not self._f.valves.enable_secondary_compaction:
            self._f._log_debug(
                "Secondary compaction disabled (enable_secondary_compaction=False)."
            )
            return messages

        # --- 2. Check prerequisites ---
        if not (
            self._f.valves.enable_history_llmlingua
            and self._f._conv_compressor is not None
        ):
            return messages

        # --- 3. Filter out messages already compressed by the primary compactor ---
        # Primary compactor marks messages with patterns like:
        #   "[🗜️ PARTE ..." or "## Código — Parte ..."
        _PRIMARY_COMPACTED_MARKERS = (
            "[🗜️ PARTE",
            "## Código — Parte",
            "## Código - Parte",
        )

        def _is_primary_compacted(content: str) -> bool:
            return any(marker in content for marker in _PRIMARY_COMPACTED_MARKERS)

        # Separate messages into those already compacted and those to compress.
        to_compress = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                if not _is_primary_compacted(content):
                    to_compress.append(msg)

        if not to_compress:
            self._f._log_debug(
                "Secondary compaction: no eligible messages (all already primary-compacted)."
            )
            return messages

        # --- 4. Apply secondary compression (LLMLingua) on eligible messages ---
        # We need to preserve the original order, so we rebuild the list.
        # The compressor expects a full list, so we pass only the eligible ones
        # and then merge back.
        compressed = await self._f._conv_compressor.compress_messages(
            messages=to_compress,
            project_id=project_id,
            symbol_index=self._f._symbol_index,
            current_msg_idx=len(messages) - 1,
            recent_lookback=self._f.valves.history_compress_recent_lookback,
            old_rate=self._f.valves.history_compress_old_rate,
            recent_rate=self._f.valves.history_compress_recent_rate,
            indexed_rate=self._f.valves.history_compress_indexed_rate,
            query=user_question,
        )

        # --- 5. Merge back: replace the original eligible messages with the compressed ones ---
        # Build a mapping from original index to compressed message.
        # Since we only compressed a subset, we need to iterate and replace.
        comp_iter = iter(compressed)
        out = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if (
                role in ("user", "assistant")
                and content
                and not _is_primary_compacted(content)
            ):
                out.append(next(comp_iter))
            else:
                out.append(msg)

        self._f._log_debug(
            f"Secondary compaction: compressed {len(to_compress)} message(s) "
            f"(skipped {len(messages) - len(to_compress)} already primary-compacted)."
        )

        return out

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Turn‑based window management (now handled by WindowManager)
    # ═══════════════════════════════════════════════════════════════════════

    # Los siguientes métodos auxiliares se mantienen sin cambios, ya que
    # no tocan el estado persistente directamente (usan LTM o helpers).

    async def _persist_turn_summary_to_ltm(
        self, summary: str, project_id: str, turn_start: int, turn_end: int
    ) -> None:
        """Store a turn‑range summary in LTM."""
        if not (HAS_SENTENCE and HAS_CHROMA and self._f.memory_collection is not None):
            return
        try:
            text = f"[Conversation summary, turns {turn_start}-{turn_end}]\n{summary}"
            embedding = await anyio.to_thread.run_sync(
                lambda: self._f.embedder.encode(text).tolist()
            )
            msg_id = (
                f"{project_id}_turnsummary_{turn_start}_{turn_end}_{int(time.time())}"
            )
            await anyio.to_thread.run_sync(
                lambda: self._f.memory_collection.upsert(
                    ids=[msg_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[
                        {
                            "role": "assistant",
                            "project_id": project_id,
                            "timestamp": time.time(),
                            "is_session_summary": True,
                            "is_turn_summary": True,
                            "covers_turn_start": turn_start,
                            "covers_turn_end": turn_end,
                            "content_type": ContentType.GENERAL.value,
                            "has_code": False,
                        }
                    ],
                )
            )
        except Exception as e:
            self._f._log_debug(f"Turn summary LTM persist failed: {e}")

    async def _merge_summaries(self, group_summaries: List[dict]) -> Optional[dict]:
        """Fuse several L1 turn‑range summaries into one L2 summary via the LLM."""
        texts: List[str] = []
        starts: List[int] = []
        ends: List[int] = []
        total_msgs = 0
        for s in group_summaries:
            t = s.get("text", "")
            if t:
                texts.append(t)
            ct = s.get("covers_turns")
            if ct:
                starts.append(ct[0])
                ends.append(ct[1])
            total_msgs += s.get("covers_msgs", 0)

        combined = "\n\n".join(f"- {t}" for t in texts)
        if not combined.strip():
            return None

        prompt = (
            "Consolidate these conversation summaries into ONE higher-level "
            "summary (3-5 sentences). Preserve key decisions, files modified, and "
            "architectural changes; drop redundancy and chit-chat.\n\n"
            f"{combined[:4000]}"
        )
        merged = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You produce concise hierarchical summaries of technical "
                "conversations. Output only the summary."
            ),
            model_override=self._f.valves.summarization_model,
            max_tokens=self._f.valves.hierarchical_summary_max_tokens,
            temperature=0.2,
            label="hierarchical_summary",
        )
        if not merged or not merged.strip():
            return None

        return {
            "text": merged.strip(),
            "created_at": time.time(),
            "covers_msgs": total_msgs,
            "covers_turns": [min(starts) if starts else 0, max(ends) if ends else 0],
            "level": 2,
        }

    async def _consolidate_summaries(
        self,
        state: ConversationState,
        project_id: str,
        slot_free: bool,
    ) -> None:
        """
        Consolidate L1 summaries into L2 and apply level-aware cap.

        Args:
            state: The ConversationState for the project.
            project_id: The project identifier.
            slot_free: Whether the LLM slot is free for auxiliary calls.
        """
        summaries = state.conversation_summaries
        if not summaries:
            return

        # Use WindowManager._summary_sort_key for consistent ordering
        if slot_free and self._f.valves.enable_hierarchical_summaries:
            group = self._f.valves.hierarchical_summary_group_size
            l1 = sorted(
                (s for s in summaries if s.get("level", 1) == 1),
                key=WindowManager._summary_sort_key,
            )
            l2plus = [s for s in summaries if s.get("level", 1) >= 2]
            if len(l1) >= group:
                oldest = l1[:group]
                merged = await self._merge_summaries(oldest)
                if merged:
                    summaries = l2plus + [merged] + l1[group:]
                    self._f._log_debug(
                        f"Hierarchical: folded {group} L1 summaries into one L2 "
                        f"covering turns {merged['covers_turns'][0]}–"
                        f"{merged['covers_turns'][1]}."
                    )
                else:
                    self._f._log_debug(
                        "Hierarchical: L2 merge failed; L1 summaries kept "
                        "(no-degradation guard)."
                    )

        max_l1 = self._f.valves.max_conversation_summaries
        max_l2 = self._f.valves.max_hierarchical_summaries
        l1 = sorted(
            (s for s in summaries if s.get("level", 1) == 1),
            key=WindowManager._summary_sort_key,
        )
        l2 = sorted(
            (s for s in summaries if s.get("level", 1) >= 2),
            key=WindowManager._summary_sort_key,
        )
        if max_l1 > 0:
            l1 = l1[-max_l1:]
        if max_l2 > 0:
            l2 = l2[-max_l2:]
        # Reassign to state attribute
        state.conversation_summaries = sorted(
            l2 + l1,
            key=WindowManager._summary_sort_key,
        )
        # ✅ Persistir el estado mediante ConversationStateManager
        self._f._conversation_state_manager.set(project_id, state)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Multi‑phase instructions injection
    # ═══════════════════════════════════════════════════════════════════════

    async def _inject_multi_phase_instructions(
        self,
        dynamic_injections: List[Tuple[str, str]],
        prelim_system: str,
        messages: List[dict],
        user_question: str,
        slot_free: bool,
        project_id: str,  # ← NEW
    ) -> None:
        """
        Inject multi‑phase protocol if the token budget is tight or a global‑scope
        query demands it.
        """
        if not (
            self._f.valves.enable_multi_phase_response
            and self._f.tokenizer
            and prelim_system
        ):
            return

        _prelim_tok: int = len(self._f.tokenizer.encode(prelim_system))
        _hist_tok: int = self._f._tokens.estimate_tokens(
            [m for m in messages if m.get("role") != "system"]
        )
        _mp_available: int = max(
            0, self._f.valves.context_window_tokens - _prelim_tok - _hist_tok
        )

        # --- Critical budget warning: append a wrap-up hint to the user message ---
        if _mp_available < self._f.valves.multi_phase_response_budget_warn:
            self._f._log_debug(
                f"Multi-phase CRITICAL ({_mp_available} tokens): "
                "wrap-up hint appended to user message (0 system tokens used)."
            )
            self._f._multi_phase.append_critical_wrap_up_hint(messages)
            return

        # --- Always evaluate the budget branch; force is an additive override ---
        budget_tight = _mp_available < self._f.valves.multi_phase_response_threshold

        # ── NEW: read the one‑shot global‑scope flag ──
        pstate = self._f._project_state_manager.get_pstate(project_id)
        force_global_scope = pstate.pop("force_multi_phase_this_turn", False)

        use_multi_phase = (
            self._f.valves.force_multi_phase_response
            or budget_tight
            or force_global_scope
        )
        if force_global_scope and not budget_tight:
            self._f._log_debug(
                "Multi‑phase: activated by global‑scope query (full_graph active this turn)."
            )

        if use_multi_phase:
            _INSTRUCTION_OVERHEAD = 450
            _mp_budget_reported = max(500, _mp_available - _INSTRUCTION_OVERHEAD)
            _mp_instructions = self._f._multi_phase.build_multi_phase_instructions(
                available_tokens=_mp_budget_reported,
                user_query=user_question,
                cot_degraded_to_l1=False,
                is_continuation=not slot_free,
            )
            dynamic_injections.append(("critical", _mp_instructions))
            self._f._log_debug(
                f"Multi-phase injected (priority=critical): "
                f"{_mp_available} available, reporting {_mp_budget_reported} to model "
                f"(overhead={_INSTRUCTION_OVERHEAD}). "
                f"force={self._f.valves.force_multi_phase_response}, budget_tight={budget_tight}"
            )
        else:
            self._f._log_debug(
                f"Multi-phase: not needed ({_mp_available} tokens > threshold "
                f"{self._f.valves.multi_phase_response_threshold} and force=False)."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Final system assembly & logging
    # ═══════════════════════════════════════════════════════════════════════

    def _assemble_final_system_and_log(
        self,
        static_block: str,
        dynamic_injections: List[Tuple[str, str]],
        messages: List[dict],
        project_id: str,
        pending_summary: str,
    ) -> List[dict]:
        """
        Assemble final system prompt, inject it, and log token breakdown.

        The stable prefix (Block A + hub-bodies tier) is placed FIRST in the
        system prompt to maximize KV cache reusability. User instructions
        (base_content) are appended LAST — they are the dynamic tail and do
        not affect the stable prefix.
        """
        budget = self._f.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        # --- Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # ── Build dynamic_block from injections (budget-aware) ──
        if budget > 0 and self._f.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_dynamic: List[str] = []
            used_dyn = 0
            static_tokens = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            user_prompt_tokens = (
                len(
                    self._f.tokenizer.encode(
                        getattr(self._f, "_original_system_prompt", "") or ""
                    )
                )
                if getattr(self._f, "_original_system_prompt", "")
                else 0
            )
            # User system prompt is never truncated — only CodeAware's
            # dynamic injections are rationed.
            dyn_budget = max(0, budget - static_tokens - user_prompt_tokens)

            for prio, text in dynamic_injections:
                if not text:
                    continue
                tok = len(self._f.tokenizer.encode(text))
                if used_dyn + tok <= dyn_budget:
                    selected_dynamic.append(text)
                    used_dyn += tok
                elif prio in ("critical", "high"):
                    avail = dyn_budget - used_dyn
                    if avail > 20:
                        selected_dynamic.append(text[: avail * 4] + "\n[truncated]")
                        break
            dynamic_block = "\n\n".join(selected_dynamic)
        else:
            dynamic_block = "\n\n".join(t for _, t in dynamic_injections if t)

        separator = "\n\n---\n\n" if static_block and dynamic_block else ""

        # ── Stable prefix: Block A + tier ────────────────────────────
        # This is placed FIRST so it is KV-cacheable across turns.
        codeaware_block = static_block + separator + dynamic_block

        # ── User instructions (dynamic tail) ─────────────────────────
        # Appended LAST so changes to it do not shift the stable prefix.
        base_content = getattr(self._f, "_original_system_prompt", "") or ""

        # ── Assemble final system ────────────────────────────────────
        final_system_parts = []
        if codeaware_block.strip():
            final_system_parts.append(codeaware_block)
        final_system = "\n\n---\n\n".join(final_system_parts)

        # User instructions go LAST — they are the dynamic tail.
        if base_content.strip():
            if final_system.strip():
                final_system = (
                    final_system
                    + "\n\n---\n\n## User instructions\n"
                    + base_content.strip()
                )
            else:
                final_system = "## User instructions\n" + base_content.strip()

        # Append pending summary if any
        if pending_summary:
            final_system = final_system + "\n\n" + pending_summary

        # Inject final system message
        if final_system.strip():
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": final_system})

        # Ensure last message is from user
        if messages and messages[-1].get("role") != "user":
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                messages = messages[: last_user_idx + 1]
            else:
                messages.append({"role": "user", "content": "continue"})

        # ── Token breakdown log ─────────────────────────────────────────
        if self._f.valves.debug and self._f.tokenizer and final_system.strip():
            static_tok = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            dynamic_tok = (
                len(self._f.tokenizer.encode(dynamic_block)) if dynamic_block else 0
            )
            base_tok = (
                len(
                    self._f.tokenizer.encode(
                        getattr(self._f, "_original_system_prompt", "") or ""
                    )
                )
                if getattr(self._f, "_original_system_prompt", "")
                else 0
            )
            total_system_tok = len(self._f.tokenizer.encode(final_system))

            # --- store in pstate ---
            pstate["last_system_tokens"] = total_system_tok

            prefix_hash = pstate.get("last_static_prefix_hash", "N/A")
            self._f._log_debug("─" * 60)
            self._f._log_debug("TOKEN BREAKDOWN — system prompt")
            self._f._log_debug(f"  BLOCK A (static, cacheable):  ~{static_tok} tokens")
            self._f._log_debug(f"  BLOCK B (dynamic, per-query): ~{dynamic_tok} tokens")
            self._f._log_debug(
                f"  User instructions (tail):      ~{base_tok} tokens (LAST)"
            )
            self._f._log_debug(
                f"  TOTAL system tokens:          ~{total_system_tok} tokens"
            )
            self._f._log_debug(f"  Prefix hash (Block A):        {prefix_hash}")
            self._f._log_debug(
                f"  → If hash matches previous:   KV cache HIT in llama.cpp"
            )
            self._f._log_debug(
                f"  → If hash changed:            KV cache MISS, full prefill"
            )
            if self._f.valves.enable_multi_phase_response:
                self._f._log_debug("  Multi-phase:                  (see earlier log)")
            if (
                self._f.valves.enable_code_history_compression
                or self._f.valves.enable_lean_user_code
            ):
                _compressed_parts = sum(
                    1
                    for m in messages
                    if m.get("role") == "assistant"
                    and re.search(r"\[🗜️ PARTE \d+/\d+", m.get("content", ""))
                )
                _leaned_msgs = sum(
                    1
                    for m in messages
                    if m.get("role") == "user"
                    and "[CÓDIGO COMPRIMIDO" in m.get("content", "")
                )
                self._f._log_debug(
                    f"  Code history:                 "
                    f"{_compressed_parts} part(s) compressed, "
                    f"{_leaned_msgs} user msg(s) leaned"
                )
            self._f._log_debug("─" * 60)

        # ── Context dump (evolution tracking) ─────────────────────────
        if self._f.valves.enable_context_dump:
            try:
                self._f._context_dumper.schedule_inlet_snapshot(
                    project_id=project_id,
                    static_block=static_block,
                    dynamic_block=dynamic_block,
                    final_system=final_system,
                    messages=messages,
                )
            except Exception as _dump_err:
                self._f._log_debug(f"Context dump scheduling failed: {_dump_err}")

        return messages


# ---------------------------------------------------------------------------
# ContextDumper — per-turn context snapshots for evolution tracking
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ContextDumper — per-turn context snapshots for evolution tracking
# ---------------------------------------------------------------------------
class ContextDumper:
    """
    Captures per‑turn context snapshots and writes them to disk for
    offline evolution tracking — the operator can follow how the context
    grows and when the Block‑A KV‑cache prefix changes across a
    conversation.

    Provides:
    * ``schedule_inlet_snapshot(project_id, static_block, dynamic_block,
      final_system, messages)`` — captures the current payload synchronously
      (cheap string copies) and offloads the disk write to a background task.
    * Markdown snapshots (one per turn) and a rolling ``latest.md`` for
      quick inspection.
    * A compact JSONL metrics log (``evolution.jsonl``) with token counts
      and hashes, suitable for plotting context growth over time.
    * Automatic pruning of old snapshots per project.

    Writes are best‑effort and fully decoupled from the request path: any
    failure is swallowed with a debug log, so nothing here can break the
    inlet or outlet.

    Docs 10–13 backported:
        M7 – includes block_a_rebuild_reason in the JSONL record.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the ContextDumper with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        self._tasks: Set[asyncio.Task] = set()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Public API – schedule snapshot capture
    # ═══════════════════════════════════════════════════════════════════════

    def schedule_inlet_snapshot(
        self,
        *,
        project_id: str,
        static_block: str,
        dynamic_block: str,
        final_system: str,
        messages: List[dict],
    ) -> None:
        """
        Capture the snapshot payload now and offload the write to a task.

        Called from a synchronous context inside the async inlet, so a running
        loop exists; if it does not (unexpected), fall back to a blocking write.

        Args:
            project_id: The project identifier.
            static_block: The rendered Block A (static, KV-cacheable).
            dynamic_block: The rendered Block B (dynamic, per-query).
            final_system: The fully assembled system prompt.
            messages: The final message list sent to the LLM.
        """
        if not self._f.valves.enable_context_dump:
            return

        self._f._log_debug(f"📸 Scheduling context dump for project '{project_id}'")

        payload = self._capture_payload(
            project_id, static_block, dynamic_block, final_system, messages
        )
        try:
            task = asyncio.create_task(self._write_async(payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError:
            # No running event loop — write inline (best effort).
            self._f._log_debug("No event loop, writing context dump inline")
            try:
                self._write_sync(payload)
            except Exception as exc:
                self._f._log_debug(f"Context dump inline write failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Payload capture (sync, cheap, mutation‑safe) – MODIFIED (M7)
    # ═══════════════════════════════════════════════════════════════════════

    def _capture_payload(
        self,
        project_id: str,
        static_block: str,
        dynamic_block: str,
        final_system: str,
        messages: List[dict],
    ) -> dict:
        """
        Snapshot strings + metadata immediately so later mutation can't race.

        Returns:
            dict: A complete payload dictionary with all context data and metrics.

        M7: Includes block_a_rebuild_reason from pstate.
        """
        max_chars = self._f.valves.context_dump_message_max_chars
        msg_copy: List[Tuple[str, str]] = []
        if self._f.valves.context_dump_include_messages:
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "") or ""
                if max_chars > 0 and len(content) > max_chars:
                    content = (
                        content[:max_chars]
                        + f"\n[...truncated {len(content) - max_chars} chars...]"
                    )
                msg_copy.append((role, content))

        # ── Resolve per-project state ────────────────────────────────────────
        pstate = self._f._project_state_manager.get_pstate(project_id)

        try:
            code_state_hash = self._f._activation.compute_code_state_hash(project_id)
        except Exception:
            code_state_hash = ""

        # ── Get hashes from pstate ──────────────────────────────────────────
        block_a_hash = pstate.get("last_static_prefix_hash", "")
        slot_hash = pstate.get("last_saved_slot_hash", "")

        # ── Get persistent state via ConversationStateManager ───────────────
        try:
            state = self._f._conversation_state_manager.get(project_id)
            turn = state.message_count
            n_active_blocks = len(state.active_blocks)
        except Exception:
            turn = 0
            n_active_blocks = 0

        try:
            n_symbols = len(self._f._symbol_index.get_all_names(project_id))
        except Exception:
            n_symbols = 0

        # ── Class membership metrics ────────────────────────────────────────
        try:
            all_names = self._f._symbol_index.get_all_names(project_id)
            n_with_parent = sum(
                1
                for n in all_names
                if self._f._symbol_index.get_parent_symbol(n, project_id)
            )
            n_classes = len(self._f._symbol_index.get_classes(project_id))
        except Exception:
            n_with_parent, n_classes = 0, 0

        # ── WindowManager metrics (now persistent in ConversationState) ────
        try:
            state = self._f._conversation_state_manager.get(project_id)
            wm_fired = state.wm_fired
            wm_msgs_evicted = state.wm_msgs_evicted
            wm_turns_evicted = state.wm_turns_evicted
            wm_summary_ok = state.wm_summary_ok
            wm_emergency_cap = state.wm_emergency_cap
            wm_batch_too_small = state.wm_batch_too_small
            wm_no_slot = state.wm_no_slot
            wm_degradation_guard = state.wm_degradation_guard
            frontier_hwm = state.summarized_turn_hwm
            summaries = state.conversation_summaries
            n_summaries_l1 = sum(1 for s in summaries if s.get("level", 1) == 1)
            n_summaries_l2 = sum(1 for s in summaries if s.get("level", 1) >= 2)
        except Exception:
            wm_fired = wm_summary_ok = wm_emergency_cap = False
            wm_batch_too_small = wm_no_slot = wm_degradation_guard = False
            wm_msgs_evicted = wm_turns_evicted = 0
            frontier_hwm = n_summaries_l1 = n_summaries_l2 = 0

        now = time.time()
        return {
            # ── Existing fields ──────────────────────────────────────────────
            "project_id": project_id,
            "turn": turn,
            "timestamp": now,
            "iso": datetime.fromtimestamp(now, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "static_block": static_block or "",
            "dynamic_block": dynamic_block or "",
            "final_system": final_system or "",
            "messages": msg_copy,
            "block_a_hash": block_a_hash,
            # ── M7: rebuild reason ─────────────────────────────────────────────
            "block_a_rebuild_reason": pstate.get("block_a_rebuild_reason"),
            "code_state_hash": code_state_hash,
            "slot_saved_hash": slot_hash,
            "n_active_blocks": n_active_blocks,
            "n_symbols": n_symbols,
            "n_symbols_with_parent": n_with_parent,
            "n_classes": n_classes,
            # ── WindowManager metrics ──────────────────────────────────────
            "wm_fired": wm_fired,
            "wm_msgs_evicted": wm_msgs_evicted,
            "wm_turns_evicted": wm_turns_evicted,
            "wm_summary_ok": wm_summary_ok,
            "wm_emergency_cap": wm_emergency_cap,
            "wm_batch_too_small": wm_batch_too_small,
            "wm_no_slot": wm_no_slot,
            "wm_degradation_guard": wm_degradation_guard,
            # ── HWM and summaries ───────────────────────────────────────────
            "frontier_hwm": frontier_hwm,
            "n_summaries_l1": n_summaries_l1,
            "n_summaries_l2": n_summaries_l2,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Writing (async + sync)
    # ═══════════════════════════════════════════════════════════════════════

    async def _write_async(self, payload: dict) -> None:
        """
        Asynchronously write the context snapshot to disk.

        Args:
            payload: The payload dictionary from _capture_payload.
        """
        self._f._log_debug(f"📝 Writing context dump (turn {payload['turn']})...")
        try:
            await anyio.to_thread.run_sync(self._write_sync, payload)
            self._f._log_debug(f"✅ Context dump written (turn {payload['turn']})")
        except Exception as exc:
            self._f._log_debug(f"❌ Context dump write failed: {exc}")

        # 3. Append compact metrics to the evolution log.
        if self._f.valves.context_dump_write_jsonl:
            record = {
                # ── Existing metrics ──────────────────────────────────────
                "ts": payload["timestamp"],
                "iso": payload["iso"],
                "turn": payload["turn"],
                "block_a_tokens": block_a_tokens,
                "block_b_tokens": block_b_tokens,
                "system_tokens": system_tokens,
                "history_tokens": history_tokens,
                "n_messages": len(payload["messages"]),
                "n_active_blocks": payload["n_active_blocks"],
                "n_symbols": payload["n_symbols"],
                "block_a_hash": payload["block_a_hash"],
                # ── NEW: rebuild reason ──────────────────────────────────
                "block_a_rebuild_reason": payload.get("block_a_rebuild_reason"),
                "code_state_hash": payload["code_state_hash"],
                "slot_saved_hash": payload["slot_saved_hash"],
                # ── WindowManager metrics ──────────────────────────────────
                "wm_fired": payload.get("wm_fired", False),
                "wm_msgs_evicted": payload.get("wm_msgs_evicted", 0),
                "wm_turns_evicted": payload.get("wm_turns_evicted", 0),
                "wm_summary_ok": payload.get("wm_summary_ok", False),
                "wm_emergency_cap": payload.get("wm_emergency_cap", False),
                "wm_batch_too_small": payload.get("wm_batch_too_small", False),
                "wm_no_slot": payload.get("wm_no_slot", False),
                "wm_degradation_guard": payload.get("wm_degradation_guard", False),
                # ── HWM and summaries ──────────────────────────────────────
                "frontier_hwm": payload.get("frontier_hwm", 0),
                "n_summaries_l1": payload.get("n_summaries_l1", 0),
                "n_summaries_l2": payload.get("n_summaries_l2", 0),
            }
            with open(
                os.path.join(project_dir, "evolution.jsonl"), "a", encoding="utf-8"
            ) as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_sync(self, payload: dict) -> None:
        """
        Synchronously write the context snapshot to disk (Markdown + JSONL).

        Modified (M7): adds block_a_rebuild_reason to evolution.jsonl.
        """
        project_id = payload["project_id"]
        project_dir = self._project_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)

        # ── 1. Compute token counts ──────────────────────────────────────────
        if self._f.tokenizer:
            static_block = payload.get("static_block", "")
            dynamic_block = payload.get("dynamic_block", "")
            final_system = payload.get("final_system", "")
            block_a_tokens = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            block_b_tokens = (
                len(self._f.tokenizer.encode(dynamic_block)) if dynamic_block else 0
            )
            system_tokens = (
                len(self._f.tokenizer.encode(final_system)) if final_system else 0
            )
            # History tokens: sum of all non-system messages
            history_tokens = 0
            for role, content in payload.get("messages", []):
                if role != "system":
                    history_tokens += len(self._f.tokenizer.encode(content))
        else:
            block_a_tokens = len(payload.get("static_block", "")) // 4
            block_b_tokens = len(payload.get("dynamic_block", "")) // 4
            system_tokens = len(payload.get("final_system", "")) // 4
            history_tokens = 0
            for role, content in payload.get("messages", []):
                if role != "system":
                    history_tokens += len(content) // 4

        # ── 2. Write Markdown snapshot ───────────────────────────────────────
        md_content = self._render_markdown(
            payload, block_a_tokens, block_b_tokens, system_tokens, history_tokens
        )
        turn = payload.get("turn", 0)
        timestamp_str = datetime.fromtimestamp(
            payload["timestamp"], tz=timezone.utc
        ).strftime("%Y%m%d_%H%M%S")
        md_filename = f"{timestamp_str}_turn_{turn:04d}.md"
        md_path = os.path.join(project_dir, md_filename)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # ── 3. Update latest.md symlink (or copy) ──────────────────────────
        latest_path = os.path.join(project_dir, "latest.md")
        try:
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(md_filename, latest_path)
        except Exception:
            # If symlink fails (e.g., Windows without admin), copy instead
            try:
                import shutil

                shutil.copy2(md_path, latest_path)
            except Exception:
                pass

        # ── 4. Write JSONL evolution record (M7) ─────────────────────────────
        if self._f.valves.context_dump_write_jsonl:
            # ── M7: include rebuild reason in the JSONL record ──────────────
            record = {
                "ts": payload["timestamp"],
                "iso": payload["iso"],
                "turn": turn,
                "block_a_tokens": block_a_tokens,
                "block_b_tokens": block_b_tokens,
                "system_tokens": system_tokens,
                "history_tokens": history_tokens,
                "n_messages": len(payload.get("messages", [])),
                "n_active_blocks": payload.get("n_active_blocks", 0),
                "n_symbols": payload.get("n_symbols", 0),
                "block_a_hash": payload.get("block_a_hash", ""),
                # ── M7: rebuild reason (string or null) ──────────────────────
                "block_a_rebuild_reason": payload.get("block_a_rebuild_reason"),
                "code_state_hash": payload.get("code_state_hash", ""),
                "slot_saved_hash": payload.get("slot_saved_hash", ""),
                # ── WindowManager metrics ──────────────────────────────────
                "wm_fired": payload.get("wm_fired", False),
                "wm_msgs_evicted": payload.get("wm_msgs_evicted", 0),
                "wm_turns_evicted": payload.get("wm_turns_evicted", 0),
                "wm_summary_ok": payload.get("wm_summary_ok", False),
                "wm_emergency_cap": payload.get("wm_emergency_cap", False),
                "wm_batch_too_small": payload.get("wm_batch_too_small", False),
                "wm_no_slot": payload.get("wm_no_slot", False),
                "wm_degradation_guard": payload.get("wm_degradation_guard", False),
                "frontier_hwm": payload.get("frontier_hwm", 0),
                "n_summaries_l1": payload.get("n_summaries_l1", 0),
                "n_summaries_l2": payload.get("n_summaries_l2", 0),
            }
            jsonl_path = os.path.join(project_dir, "evolution.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── 5. Prune old snapshots ───────────────────────────────────────────
        self._prune(project_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Rendering
    # ═══════════════════════════════════════════════════════════════════════

    def _render_markdown(
        self,
        payload: dict,
        block_a_tokens: int,
        block_b_tokens: int,
        system_tokens: int,
        history_tokens: int,
    ) -> str:
        """
        Render the context snapshot as Markdown for human inspection.

        Args:
            payload: The payload dictionary.
            block_a_tokens: Token count of Block A.
            block_b_tokens: Token count of Block B.
            system_tokens: Total system prompt tokens.
            history_tokens: History tokens (non-system).

        Returns:
            str: The rendered Markdown text.
        """
        lines: List[str] = []
        lines.append(
            f"# Context Snapshot — `{payload['project_id']}` — "
            f"turn {payload['turn']} — {payload['iso']}"
        )
        lines.append("")
        lines.append("## Metadata")
        lines.append(f"- turn (message_count): {payload['turn']}")
        lines.append(f"- Block A tokens (static):  ~{block_a_tokens}")
        lines.append(f"- Block B tokens (dynamic): ~{block_b_tokens}")
        lines.append(f"- system prompt tokens:     ~{system_tokens}")
        lines.append(f"- history tokens:           ~{history_tokens}")
        lines.append(f"- Block A prefix hash: `{payload['block_a_hash'] or 'N/A'}`")
        lines.append(f"- code_state_hash:     `{payload['code_state_hash'] or 'N/A'}`")
        lines.append(f"- slot saved hash:     `{payload['slot_saved_hash'] or 'N/A'}`")
        lines.append(f"- active blocks: {payload['n_active_blocks']}")
        lines.append(f"- indexed symbols: {payload['n_symbols']}")
        lines.append(
            f"- symbols with class resolved: "
            f"{payload['n_symbols_with_parent']}/{payload['n_symbols']}"
        )
        lines.append(f"- classes detected: {payload['n_classes']}")

        # ── WindowManager metrics ────────────────────────────────────────────
        lines.append("")
        lines.append("### WindowManager metrics (this turn)")
        lines.append(
            f"- wm_fired={payload.get('wm_fired', False)}"
            f"  msgs_evicted={payload.get('wm_msgs_evicted', 0)}"
            f"  turns_evicted={payload.get('wm_turns_evicted', 0)}"
        )
        lines.append(
            f"- summary_ok={payload.get('wm_summary_ok', False)}"
            f"  emergency_cap={payload.get('wm_emergency_cap', False)}"
        )
        lines.append(
            f"- batch_too_small={payload.get('wm_batch_too_small', False)}"
            f"  no_slot={payload.get('wm_no_slot', False)}"
            f"  degradation_guard={payload.get('wm_degradation_guard', False)}"
        )
        lines.append(
            f"- frontier_hwm={payload.get('frontier_hwm', 0)}"
            f"  summaries_L1={payload.get('n_summaries_l1', 0)}"
            f"  summaries_L2={payload.get('n_summaries_l2', 0)}"
        )

        lines.append("")
        lines.append("## Block A — static (KV-cache prefix)")
        lines.append("```text")
        lines.append(payload["static_block"] or "(empty)")
        lines.append("```")
        lines.append("")

        lines.append("## Block B — dynamic (per-query)")
        lines.append("```text")
        lines.append(payload["dynamic_block"] or "(empty)")
        lines.append("```")
        lines.append("")

        if self._f.valves.context_dump_include_messages:
            lines.append("## Message window (non-system, sent to model)")
            idx = 0
            for role, content in payload["messages"]:
                if role == "system":
                    continue  # already shown as Block A + Block B above
                lines.append(f"### [{idx}] {role}")
                lines.append("```text")
                lines.append(content or "(empty)")
                lines.append("```")
                idx += 1
            lines.append("")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Directory & pruning
    # ═══════════════════════════════════════════════════════════════════════

    def _project_dir(self, project_id: str) -> str:
        """
        Return the project-specific dump directory path.

        Args:
            project_id: The project identifier.

        Returns:
            str: The directory path.
        """
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:40] or "default"
        return os.path.join(self._f.valves.context_dump_dir.rstrip("/"), slug)

    def _prune(self, project_dir: str) -> None:
        """
        Prune old snapshots, keeping only the most recent ones.

        Args:
            project_dir: The project directory containing snapshots.
        """
        keep = self._f.valves.context_dump_max_files_per_project
        if keep <= 0:
            return
        try:
            # Find all snapshots with the new format: XXXX_turn_...md
            snapshots = sorted(
                f
                for f in os.listdir(project_dir)
                if re.match(r"^\d{4}_turn_\d+\.md$", f)
            )
        except Exception:
            return
        excess = len(snapshots) - keep
        for fname in snapshots[: max(0, excess)]:
            try:
                os.remove(os.path.join(project_dir, fname))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ProjectStateManager — per‑project volatile state (SRP)
# ---------------------------------------------------------------------------
class ProjectStateManager:
    """
    Manages per‑project volatile state that lives in memory only.

    This class holds all attributes that vary by project (call‑graph mode,
    ingested language, cache keys, invalidation flags, etc.) in a dictionary
    keyed by project_id. It provides a clean accessor and factory, and will
    support project‑state cleanup on eviction (subsystem 05).

    This decouples project‑scoped state from the singleton Filter instance,
    preventing cross‑project corruption when alternating between projects.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance (provides valves, logger, etc.).
        """
        self._f = filter_ref
        self._store: dict[str, dict] = {}

    def _new_pstate(self) -> dict:
        """
        Factory for a fresh per-project state bag with all defaults.
        """
        return {
            # Call-graph mode
            "resolved_call_graph_mode": None,
            "current_call_graph_mode": None,
            # Silent ingestion
            "ingested_lang": None,
            "raw_ingested_symbols": None,
            # Block A / skeleton cache keys and invalidation flags
            "block_a_cache_key": None,
            "block_a_cached": None,
            "skeleton_cache_key": None,
            "skeleton_cached": None,
            "skeleton_tier_cache_key": None,
            "skeleton_tier_cached": None,
            "skeleton_invalidated": False,
            "block_a_invalidated": False,
            # Centrality and lightweight context
            "node_centrality": {},
            "cached_lightweight_context": "",
            "cached_code_state_hash": None,
            # Token accounting
            "last_system_tokens": 0,
            "last_total_context_tokens": 0,
            # KVCache persistence
            "last_saved_slot_hash": "",
            "slot_restored": False,
            "slot_restore_attempted": False,
            # Misc
            "last_processed_message_idx": -1,
            "response_cache_count": 0,
            "summarize_inactive_in_progress": False,
            # Graph mode downgrade streak
            "graph_mode_downgrade_streak": 0,
            # LOD adaptive tracking
            "last_activation_scores": {},
            "last_lod_levels": {},
            # structure_hash_for_cache is set by ContextBuilder.build_block_a
            "structure_hash_for_cache": None,
            # ── one‑shot flag for global‑scope queries ──
            "force_multi_phase_this_turn": False,
        }

    def get_pstate(self, project_id: str) -> dict:
        """
        Return the mutable per‑project state bag, creating it on first access.

        Args:
            project_id (str): The project identifier.

        Returns:
            dict: The per‑project state bag for the given project.
        """
        st = self._store.get(project_id)
        if st is None:
            st = self._new_pstate()
            self._store[project_id] = st
        return st

    def clear_project(self, project_id: str) -> None:
        """Remove the state bag for a project (called on eviction)."""
        self._store.pop(project_id, None)

    # ═══════════════════════════════════════════════════════════════════════
    # 3 — KVCache persistence
    # ═══════════════════════════════════════════════════════════════════════

    async def slot_save(self, project_id: str, force: bool = False) -> bool:
        """
        Save the KV slot after a turn.

        force=True ignores the static-hash guard (used after monotonic
        compaction, when the history prefix changed but Block A did not).
        The token-threshold guard (P5) is always respected.

        Uses the structural hash (signatures only) for the filename so
        docstring population does not cause slot file proliferation.

        Args:
            project_id: The project identifier.
            force: Whether to force save even if the hash hasn't changed.

        Returns:
            bool: True if the slot was saved successfully.
        """
        if not self._f.valves.enable_slot_persistence:
            return False

        # --- 1. Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # --- 2. Token threshold guard (skip oversized KV writes) ---
        _max_ctx = self._f.valves.slot_save_max_context_tokens
        if _max_ctx > 0:
            _ctx_tok = pstate.get("last_total_context_tokens", 0)
            if _ctx_tok > _max_ctx:
                self._f._log_debug(
                    f"Slot save skipped: context {_ctx_tok} tokens > threshold "
                    f"{_max_ctx} (avoids large KV write under mutex)"
                )
                return False

        # --- 3. Get the structural hash from pstate (set by build_block_a) ---
        static_hash = pstate.get("structure_hash_for_cache")
        if not static_hash:
            # Fallback: compute from cache if available (should not happen in normal flow)
            cached = pstate.get("block_a_cached")
            if cached:
                static_hash = hashlib.md5(cached.encode()).hexdigest()[:16]
            else:
                return False

        # --- 4. Skip if already saved and not forced ---
        if not force and pstate.get("last_saved_slot_hash") == static_hash:
            return False

        # --- 5. Build filename and call llama.cpp API ---
        filename = self._slot_filename(project_id, static_hash)
        base = self._f.valves.LLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        try:
            session = await _shared_get_http_session(
                timeout_seconds=self._f.valves.llm_per_call_timeout
            )
            async with session.post(
                f"{base}/slots/{self._f.valves.slot_id}",
                params={"action": "save"},
                json={"filename": filename, "model": self._f.valves.llm_model},
            ) as resp:
                if resp.status == 200:
                    pstate["last_saved_slot_hash"] = static_hash
                    data = await resp.json()
                    self._f._log_debug(
                        f"✓ Slot saved → {filename} "
                        f"({data.get('n_saved', '?')} tokens, "
                        f"{data.get('timings', {}).get('save_ms', '?'):.0f}ms)"
                    )
                    await self._cleanup_old_slot_files(project_id, filename)
                    return True
                else:
                    body = await resp.text()
                    self._f._log_debug(f"Slot save failed: HTTP {resp.status} — {body}")
                    return False
        except Exception as e:
            self._f._log_debug(f"Slot save error: {e}")
            return False

    async def slot_restore(self, project_id: str) -> bool:
        """
        Restore the KV slot at session start.

        Uses the structural hash (signatures only) to locate the correct
        slot file, ensuring that docstring population does not cause a miss.

        Args:
            project_id: The project identifier.

        Returns:
            bool: True if the slot was restored successfully.
        """
        if not self._f.valves.enable_slot_persistence:
            return False

        # --- 1. Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # --- 2. Skip if already attempted ---
        if pstate.get("slot_restore_attempted", False):
            return pstate.get("slot_restored", False)

        pstate["slot_restore_attempted"] = True

        # --- 3. Get the structural hash from pstate ---
        static_hash = pstate.get("structure_hash_for_cache")
        if not static_hash:
            # Fallback: compute from cache if available
            cached = pstate.get("block_a_cached")
            if cached:
                static_hash = hashlib.md5(cached.encode()).hexdigest()[:16]
            else:
                return False

        filename = self._slot_filename(project_id, static_hash)

        # --- 4. Check if file exists ---
        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        if not os.path.exists(os.path.join(slot_dir, filename)):
            self._f._log_debug(f"Slot restore: no file found for {filename}")
            return False

        # --- 5. Call llama.cpp API to restore ---
        base = self._f.valves.LLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        try:
            session = await _shared_get_http_session(
                timeout_seconds=self._f.valves.llm_per_call_timeout
            )
            async with session.post(
                f"{base}/slots/{self._f.valves.slot_id}",
                params={"action": "restore"},
                json={"filename": filename, "model": self._f.valves.llm_model},
            ) as resp:
                if resp.status == 200:
                    pstate["slot_restored"] = True
                    data = await resp.json()
                    self._f._log_debug(
                        f"✓ Slot restored ← {filename} "
                        f"({data.get('n_restored', '?')} tokens)"
                    )
                    return True
                else:
                    body = await resp.text()
                    self._f._log_debug(
                        f"Slot restore failed: HTTP {resp.status} — {body}"
                    )
                    return False
        except Exception as e:
            self._f._log_debug(f"Slot restore error: {e}")
            return False

    async def slot_restore_for_continuity(self, project_id: str) -> bool:
        """
        Restore KV cache after auxiliary LLM calls (CoT, contradiction) have
        dirtied the slot due to SWA architecture. Called at the end of every
        inlet when slot_free=True.

        Uses the structural hash for consistency with the slot filename.

        Args:
            project_id: The project identifier.

        Returns:
            bool: True if the slot was restored successfully.
        """
        if not self._f.valves.enable_slot_persistence:
            self._f._log_debug(
                "slot_restore_for_continuity: disabled (enable_slot_persistence=False)"
            )
            return False

        # --- 1. Resolve per-project state ---
        pstate = self._f._project_state_manager.get_pstate(project_id)

        # --- 2. Get the structural hash ---
        static_hash = pstate.get("structure_hash_for_cache")
        if not static_hash:
            cached = pstate.get("block_a_cached")
            if cached:
                static_hash = hashlib.md5(cached.encode()).hexdigest()[:16]
            else:
                self._f._log_debug(
                    "slot_restore_for_continuity: no static hash available, skipping"
                )
                return False

        filename = self._slot_filename(project_id, static_hash)

        # --- 3. Check if file exists ---
        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        file_path = os.path.join(slot_dir, filename)
        self._f._log_debug(
            f"slot_restore_for_continuity: looking for {file_path} (hash={static_hash})"
        )
        if not os.path.exists(file_path):
            self._f._log_debug(
                f"slot_restore_for_continuity: file not found: {file_path}, skipping"
            )
            return False

        # --- 4. Call llama.cpp API to restore ---
        base = self._f.valves.LLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        self._f._log_debug(
            f"slot_restore_for_continuity: attempting restore for '{project_id}' "
            f"with hash {static_hash} -> {filename}"
        )

        try:
            session = await _shared_get_http_session(
                timeout_seconds=self._f.valves.llm_per_call_timeout
            )
            async with session.post(
                f"{base}/slots/{self._f.valves.slot_id}",
                params={"action": "restore"},
                json={"filename": filename, "model": self._f.valves.llm_model},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._f._log_debug(
                        f"✓ KV cache restored post-aux ← {filename} "
                        f"({data.get('n_restored', '?')} tokens)"
                    )
                    return True

                body = await resp.text()
                self._f._log_debug(
                    f"slot_restore_for_continuity: restore failed: "
                    f"HTTP {resp.status} — {body}"
                )
                return False

        except Exception as e:
            self._f._log_debug(f"slot_restore_for_continuity: error: {e}")
            return False

    def _slot_filename(self, project_id: str, static_hash: str) -> str:
        """
        Deterministic slot file name.
        Encodes: project + static block hash + model hash.
        If any of the three changes → different name → no stale restore.

        The static_hash must be the structural hash (signatures only, no docstrings)
        to ensure slot persistence survives docstring population.

        Args:
            project_id (str): The project identifier.
            static_hash (str): The structural hash of the code state.

        Returns:
            str: The filename for the slot file.
        """
        model_hash = hashlib.md5(self._f.valves.llm_model.encode()).hexdigest()[:8]
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        return f"slot{self._f.valves.slot_id}_{project_slug}_{static_hash}_{model_hash}.bin"

    async def _cleanup_old_slot_files(self, project_id: str, keep: str) -> None:
        """
        Delete stale slot files, keeping only the current one.

        Args:
            project_id: The project identifier.
            keep: The filename to keep (current slot).
        """
        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        if not os.path.isdir(slot_dir):
            return
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        prefix = f"slot{self._f.valves.slot_id}_{project_slug}_"
        try:
            for fname in os.listdir(slot_dir):
                if fname.startswith(prefix) and fname != keep:
                    os.remove(os.path.join(slot_dir, fname))
                    self._f._log_debug(f"Removed obsolete slot file: {fname}")
        except Exception as e:
            self._f._log_debug(f"Slot cleanup error: {e}")


class SemanticSeedInferencer:
    """
    LLM‑guided seed inference for the PPR activation graph.

    PROBLEM IT SOLVES
    ─────────────────
    `_extract_query_seeds()` only seeds symbols whose name appears literally in
    the query. For implicit requests ("add OAuth2", "make login use JWT", "how
    would a new notification type affect the system?") the relevant symbols are
    never named → PPR gets no seeds → the model doesn't see the bodies it needs
    → shallow or incorrect answers.

    SOLUTION
    ────────
    Given the project skeleton (already built and cached for Block A) and the
    query, ask the LLM which qualified ids need their full body. The returned
    ids are injected as high‑confidence seeds before PPR.

    COST AND KV CACHE
    ─────────────────
    - ONE auxiliary call bounded by seed_inference_skeleton_max_tokens.
    - Gated by slot_free (does not run during AutoContinue continuations).
    - Dirties the KV slot exactly like CoT/contradiction.
    - Covered by slot_restore_for_continuity at the end of the inlet (see patch K).
      No separate restore is needed.

    GLOBAL SCOPE
    ─────────────
    "check every call for orphans" → the whole project is relevant; seeding subsets
    doesn't help. is_global_scope() detects these cases and inference returns {}
    → the request is routed to the multi‑phase protocol with full_graph in Block A
    (see patch I).

    Docs 10–13 backported:
        B2 – dotted‑name decomposition (e.g. 'Filter.inlet') resolves correctly.
    """

    # Queries that demand traversing the ENTIRE project, not a subgraph.
    _GLOBAL_SCOPE_RE = re.compile(
        r"\b("
        r"todos?\s+los\s+(?:m[eé]todos|s[ií]mbolos|calls?|llamadas?|funciones?)|"
        r"cada\s+(?:m[eé]todo|funci[oó]n|call|llamada)|"
        r"l[ií]nea\s+a\s+l[ií]nea|"
        r"call(?:s)?\s+hu[eé]rfan|hu[eé]rfan[oa]s?\s+(?:call|llamada)|orphan\s+call|"
        r"todo\s+el\s+(?:c[oó]digo|proyecto)|whole\s+(?:codebase|project)|"
        r"every\s+(?:method|function|call|symbol)|all\s+(?:methods|functions|calls|symbols)|"
        r"recorre\s+(?:todo|cada)|traverse\s+(?:all|every)"
        r")\b",
        re.IGNORECASE,
    )

    # A line from the LLM that contains a qualified id (with or without backticks/bullets).
    _ID_LINE_RE = re.compile(r"^[\s\-*\d.]*`?([A-Za-z_][\w.]*)`?\s*(?:#.*)?$")

    def __init__(self, filter_ref: "Filter") -> None:
        """
        Initialize the inferencer with a reference to the parent Filter.

        Args:
            filter_ref: The parent Filter instance.
        """
        self._f = filter_ref

    # ── Global scope detection ────────────────────────────────────────────────

    def is_global_scope(self, query: str) -> bool:
        """
        Return True if the request demands a traversal of the whole project.
        """
        return bool(self._GLOBAL_SCOPE_RE.search(query or ""))

    # ── Gate: is it worth spending an LLM call? ──────────────────────────────

    def _should_infer(
        self,
        query: str,
        project_id: str,
        intent_vector: dict,
        use_case: str,
    ) -> bool:
        """
        Decide whether to spend an LLM call on seed inference.

        Modes (valve seed_inference_mode):
          'off'    → never.
          'always' → always, provided there is a skeleton and a non‑trivial query.
          'auto'   → (default) when lexical seeds are insufficient OR
                     the use case is A/D (architecture/refactor), where the
                     impact reasoning needs bodies the user didn't name.
        """
        mode = self._f.valves.seed_inference_mode
        if mode == "off":
            return False

        # Query too short to infer anything useful.
        if not query or len(query.strip()) < self._f.valves.seed_inference_min_chars:
            return False

        # Empty project: no skeleton to send.
        if not self._f._symbol_index.get_all_qualified_names(project_id):
            return False

        if mode == "always":
            return True

        # 'auto': infer when lexical seeds are scarce OR the use case is
        # architecture/refactor (implicit impact on many unnamed symbols).
        exact, _ = self._f._activation._extract_query_seeds(query, project_id)
        if len(exact) < self._f.valves.seed_inference_min_lexical:
            return True
        if use_case in ("A", "D"):
            return True

        return False

    # ── Main inference ────────────────────────────────────────────────────────

    async def infer_seeds(
        self,
        query: str,
        project_id: str,
        intent_vector: dict,
        use_case: str,
        slot_free: bool = True,
    ) -> Dict[str, float]:
        """
        Return {qualified_id: seed_score} of symbols the LLM judges relevant.

        Returns {} when:
          - slot is not free (AutoContinue active: no inference on parts).
          - global scope detected (routed to multi‑phase + full_graph).
          - the gate _should_infer fails.
          - skeleton is empty or LLM call fails.
          Always without exception for the caller — PPR simply uses lexical
          seeds in those cases.
        """
        if not slot_free:
            return {}

        # Global scope → multi‑phase, no subgraph expansion.
        if self.is_global_scope(query):
            self._f._log_debug(
                "SemanticSeedInferencer: global scope detected → "
                "inference skipped, delegated to multi‑phase + full_graph."
            )
            return {}

        if not self._should_infer(query, project_id, intent_vector, use_case):
            return {}

        # Reuse the skeleton already built/cached for Block A.
        # _get_skeleton_for_cot returns the same text as _format_skeleton,
        # cached by structure_hash. Cost: O(1) if already in cache.
        skeleton = await self._f._ctx_builder._get_skeleton_for_cot(project_id, query)
        if not skeleton.strip():
            self._f._log_debug("SemanticSeedInferencer: empty skeleton, cannot infer.")
            return {}

        # Trim skeleton to the configured budget.
        max_sk = self._f.valves.seed_inference_skeleton_max_tokens
        if max_sk > 0:
            skeleton = self._f._tokens.truncate_text_to_tokens(skeleton, max_sk)

        n = self._f.valves.seed_inference_max_symbols
        prompt = (
            f"Project skeleton (signatures only — no bodies):\n"
            f"```\n{skeleton}\n```\n\n"
            f'User request:\n"{query[:600]}"\n\n'
            f"List the qualified symbol identifiers whose FULL implementation body "
            f"must be read to fulfill this request correctly. "
            f"Use the exact identifiers from the skeleton (e.g. `ClassName.method` "
            f"or `module_function`). Include direct dependencies implied by the "
            f"request even if not named explicitly by the user. "
            f"Output ONLY identifiers, one per line, no explanations, no numbering. "
            f"Maximum {n} identifiers."
        )

        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a code retrieval planner. Given a project skeleton and a "
                "user request, output only the qualified symbol identifiers whose "
                "full bodies must be read. One identifier per line, nothing else."
            ),
            model_override=(
                self._f.valves.seed_inference_model or self._f.valves.llm_model
            ),
            max_tokens=self._f.valves.seed_inference_max_tokens,
            temperature=0.0,
            label="seed_inference",
        )

        if not response:
            self._f._log_debug("SemanticSeedInferencer: LLM returned no response.")
            return {}

        seeds = self._parse_and_resolve(response, project_id)
        return seeds

    # ── Fuzzy fallback (original, unchanged) ─────────────────────────────────

    def _fuzzy_resolve(self, token: str, project_id: str) -> List[str]:
        """
        Fallback fuzzy resolution for hallucinated tokens.

        Uses rapidfuzz token_set_ratio against all qualified ids in the project.
        """
        if not HAS_FUZZ:
            return []
        from rapidfuzz import fuzz

        all_qids = self._f._symbol_index.get_all_qualified_names(project_id)
        best_match = None
        best_ratio = 0.0
        token_lower = token.lower()
        for qid in all_qids:
            ratio = fuzz.token_set_ratio(token_lower, qid.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = qid
        if best_match and best_ratio >= self._f.valves.seed_inference_fuzzy_threshold:
            self._f._log_debug(
                f"SemanticSeedInferencer: fuzzy matched '{token}' → '{best_match}' "
                f"(ratio={best_ratio:.0f}%)"
            )
            return [best_match]
        return []

    # ── Parsing and resolution ── MODIFIED (B2) ──────────────────────────────

    def _parse_and_resolve(self, response: str, project_id: str) -> Dict[str, float]:
        """
        Parse one‑id‑per‑line and resolve each token to actual qualified ids.

        Resolution rules (B2):
          - Token in all_qids → exact match, full score.
          - (NEW B2) Dotted name (e.g. 'Filter.inlet') → split on last dot,
            construct the qualified id via qualify_symbol_name(method, parent).
          - Bare token (no '.') → get_qualified_names_for → all qids sharing
            that bare name; score divided among ambiguities.
          - Hallucinated token → fuzzy matching with rapidfuzz (token_set_ratio)
            threshold 0.85, score penalty 0.8.
          - If fuzzy fails, discard silently.
        """
        score = self._f.valves.seed_inference_score
        max_syms = self._f.valves.seed_inference_max_symbols
        all_qids = self._f._symbol_index.get_all_qualified_names(project_id)

        seeds: Dict[str, float] = {}
        for line in response.splitlines():
            if len(seeds) >= max_syms:
                break
            m = self._ID_LINE_RE.match(line)
            if not m:
                continue
            token = m.group(1).strip()
            if not token:
                continue

            # ── Step 1: Exact match ──────────────────────────────────────────────
            if token in all_qids:
                seeds[token] = max(seeds.get(token, 0.0), score)
                continue

            # ── Step 2 (B2): Dotted‑name decomposition ──────────────────────────
            # The LLM often returns qualified names like "Filter.inlet" or
            # "ContextBuilder.build_block_b". Split on the last dot and try to
            # resolve the method against its parent class via qualify_symbol_name.
            if "." in token:
                parent_name, method_part = token.rsplit(".", 1)
                # Try qualify_symbol_name(method, parent) with the global helper
                constructed_qid = qualify_symbol_name(method_part, parent_name)
                if constructed_qid in all_qids:
                    seeds[constructed_qid] = max(seeds.get(constructed_qid, 0.0), score)
                    continue
                # Fallback: maybe the parent is also compound ("a.b.C") —
                # try just the method part via bare‑name lookup
                by_method = self._f._symbol_index.get_qualified_names_for(
                    method_part, project_id
                )
                if by_method:
                    share = score / len(by_method)
                    for q in by_method:
                        if q in all_qids:
                            seeds[q] = max(seeds.get(q, 0.0), share)
                    continue

            # ── Step 3: Bare name resolution (original) ──────────────────────────
            qids = {
                q
                for q in self._f._symbol_index.get_qualified_names_for(
                    token, project_id
                )
                if q in all_qids
            }
            if qids:
                share = score / len(qids)
                for q in qids:
                    seeds[q] = max(seeds.get(q, 0.0), share)
                continue

            # ── Step 4: Fuzzy matching (existing) ──────────────────────────────
            fuzzy_matches = self._fuzzy_resolve(token, project_id)
            if fuzzy_matches:
                fuzzy_score = score * self._f.valves.seed_inference_fuzzy_penalty
                for q in fuzzy_matches:
                    seeds[q] = max(seeds.get(q, 0.0), fuzzy_score)
                continue

            # If no match, discard silently
            self._f._log_debug(f"SemanticSeedInferencer: no match for '{token}'")

        if seeds:
            sample = sorted(seeds)[:8]
            ellipsis = "..." if len(seeds) > 8 else ""
            self._f._log_debug(
                f"SemanticSeedInferencer: {len(seeds)} symbol(s) seeded "
                f"→ {sample}{ellipsis}"
            )
        else:
            self._f._log_debug(
                "SemanticSeedInferencer: LLM responded but no id "
                "matches the index (possible hallucinations)."
            )
        return seeds


# ---------------------------------------------------------------------------
# Valves
# ---------------------------------------------------------------------------
class Filter:
    """OpenWebUI pipeline filter that implements the full CodeAware context
    manager.  Owns all configuration valves, persistent state, long‑term
    memory, and every subsystem class.

    The ``inlet()`` and ``outlet()`` methods are the entry points called by
    the OpenWebUI runtime at the start and end of each request.
    """

    # ── Class constants ────────────────────────────────────────────────────

    INTENT_KEYWORDS = {
        "forget",
        "olvida",
        "olvid",
        "remember",
        "recuerda",
        "pin",
        "fija",
        "guarda",
        "obsolete",
        "obsoleto",
        "deprecated",
        "ya no",
        "remove",
        "elimina",
        "borra",
        "quita",
        "keep",
        "mantén",
        "conserva",
    }

    _COT_NEGATION_PREFIXES: frozenset = frozenset(
        {
            "don't",
            "do not",
            "dont",
            "no need to",
            "without",
            "not",
            "never",
            "avoid",
            "skip",
            "no",
            "sin",
            "no hace falta",
            "no es necesario",
        }
    )

    _MULTI_PHASE_MARKERS: frozenset = frozenset(
        {
            "▶ CONTINÚA:",
            "▶ CONTINÚA EN LA SIGUIENTE PARTE",
        }
    )

    # Read via getattr(self._f, "_SYMBOL_BLACKLIST", set()) in
    # LongTermMemory._is_symbol_indexable() — was referenced but never
    # defined, so the blacklist check was a permanent no-op. Minimal
    # starting point: low-retrieval-value dunders every class has, which
    # add noise to LTM "code_symbols" metadata without being meaningful
    # call-graph targets. __init__ deliberately excluded — constructors
    # carry real signal for symbol-boosted retrieval. Adjust freely; this
    # only affects what counts as a "symbol mention" in LTM, not the
    # SymbolIndex itself (those symbols stay fully indexed there).
    _SYMBOL_BLACKLIST: frozenset = frozenset(
        {
            "__repr__",
            "__str__",
            "__eq__",
            "__hash__",
            "__len__",
            "__iter__",
        }
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Configuration valves (nested class)
    # ═══════════════════════════════════════════════════════════════════════════

    class Valves(BaseModel):
        """Pydantic model holding every user‑facing configuration valve for
        the filter, with descriptions, defaults, and constraints.
        """

        # ═══════════════════════════════════════════════════════════════════
        #  Context window budgets
        # ═══════════════════════════════════════════════════════════════════
        context_window_tokens: int = Field(
            default=262000,
            description="Total token capacity of the LLM server. Must match the llama.cpp --ctx-size exactly.",
        )
        active_context_max_tokens: int = Field(
            default=15000,
            description="Maximum tokens for code context injected in Block B (LOD‑activated code).",
        )
        history_max_tokens: int = Field(
            default=24000,
            description=(
                "Maximum tokens for conversation history (non‑system messages). "
                "Only operates over conversation messages, not code. "
                "Enforced after LLMLingua compression. 0 = disabled."
            ),
        )
        ltm_retrieval_max_tokens: int = Field(
            default=6000,
            description="Maximum tokens for long‑term memory retrieved per request. 0 = unlimited.",
        )
        cot_max_tokens: int = Field(
            default=4000,
            description="Maximum tokens for Chain‑of‑Thought reasoning responses. 0 = unlimited.",
        )
        response_reserve_tokens: int = Field(
            default=4096,
            ge=256,
            le=16384,
            description="Minimum tokens reserved for the LLM's response when computing the effective context budget.",
        )
        global_injection_token_budget: int = Field(
            default=120000,
            description="Hard cap for ALL system injections combined (Block A + Block B). 0 = disabled.",
        )
        # ── Per‑block limits ───────────────────────────────────────
        max_code_block_tokens: int = Field(
            default=6000,
            description="Maximum tokens per individual code block. 0 = unlimited. See code_block_overflow_action.",
        )
        code_block_overflow_action: str = Field(
            default="summarize",
            description="Action when a code block exceeds max_code_block_tokens: 'warn', 'truncate', or 'summarize'.",
        )
        code_block_truncate_keep_head: int = Field(default=50)
        code_block_truncate_keep_tail: int = Field(default=50)
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]"
        )
        # ── Oversized block summaries ──────────────────────────────
        summary_code_max_chars: int = Field(
            default=20000,
            description="Maximum characters of source code sent to the LLM when generating a summary for an oversized code block.",
        )
        oversized_summary_max_tokens: int = Field(
            default=350,
            description="Maximum tokens allowed for the generated summary of an oversized code block.",
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Long‑term memory (ChromaDB + RAPTOR)
        # ═══════════════════════════════════════════════════════════════════
        # ── ChromaDB infrastructure ─────────────────────────────────
        long_term_memory_dir: str = Field(default="/app/backend/data/long_term_memory")
        long_term_memory_expiration_days: int = Field(default=30)
        long_term_memory_top_k: int = Field(default=10)
        long_term_memory_similarity_threshold: float = Field(default=0.65)
        ltm_time_decay_hours: float = Field(default=12.0)
        ltm_store_only_code_sessions: bool = Field(default=True)
        # ── Symbol indexing in LTM ──────────────────────────────────
        ltm_index_symbols_enabled: bool = Field(default=True)
        ltm_symbol_index_max_per_message: int = Field(default=20)
        ltm_symbol_boost_enabled: bool = Field(default=True)
        ltm_symbol_boost_factor: float = Field(default=1.5)
        ltm_symbol_boost_min_similarity: float = Field(default=0.5)
        ltm_symbol_force_mode_enabled: bool = Field(default=False)
        ltm_symbol_force_fallback_to_semantic: bool = Field(default=True)
        # ── Reranking ───────────────────────────────────────────────
        enable_reranking: bool = Field(default=True)
        reranker_model: str = Field(
            default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        reranker_top_k: int = Field(default=5)
        # ── RAPTOR ──────────────────────────────────────────────────
        enable_raptor: bool = Field(
            default=True,
            description="Enable RAPTOR hierarchical clustering of code symbols for faster LTM retrieval.",
        )
        raptor_clusters_per_level: int = Field(default=5, ge=2, le=20)
        raptor_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        raptor_summary_max_tokens: int = Field(default=150)
        raptor_rebuild_interval: int = Field(default=20)
        raptor_use_call_graph_proximity: bool = Field(
            default=True,
            description="Weight call‑graph distance alongside semantic similarity when clustering symbols.",
        )
        raptor_graph_weight: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="0.0 = semantic only, 1.0 = graph only.",
        )
        # ── Augmented retrieval ─────────────────────────────────────
        enable_contextual_retrieval: bool = Field(default=True)
        contextual_retrieval_mode: str = Field(default="metadata")
        enable_multi_query_retrieval: bool = Field(default=True)
        multi_query_variants: int = Field(default=2, ge=1, le=4)

        # ═══════════════════════════════════════════════════════════════════
        #  Context compression (conversation + code)
        # ═══════════════════════════════════════════════════════════════════
        # ── History compression with LLMLingua ──────────────────────
        enable_history_llmlingua: bool = Field(
            default=True,
            description="Apply LLMLingua‑2 compression to conversation history.",
        )
        history_compress_recent_rate: float = Field(
            default=0.75,
            ge=0.3,
            le=1.0,
            description="Compression rate for the last `history_compress_recent_lookback` turns.",
        )
        history_compress_old_rate: float = Field(
            default=0.40,
            ge=0.1,
            le=1.0,
            description="Compression rate for turns older than recent_lookback.",
        )
        history_compress_indexed_rate: float = Field(
            default=0.20,
            ge=0.05,
            le=0.5,
            description="Compression rate for old turns whose code is fully indexed (safe to be aggressive).",
        )
        history_compress_recent_lookback: int = Field(
            default=4,
            ge=1,
            le=20,
            description="Number of recent turns exempt from aggressive compression.",
        )

        # ── Secondary compaction gate ─────────────────────
        enable_secondary_compaction: bool = Field(
            default=True,
            description=(
                "If True, run the secondary ConversationCompressor (LLMLingua) after the "
                "primary compactor, restricted to prose the primary did not summarize. "
                "Off by default to avoid double-compaction in cascade."
            ),
        )

        # ── Code compression with LLMLingua ─────────────────────────
        enable_code_compression: bool = Field(
            default=False,
            description="Apply LLMLingua‑2 compression to individual code blocks in Block B (LOD‑3).",
        )
        code_compression_rate: float = Field(
            default=0.5,
            ge=0.3,
            le=0.8,
            description="Fraction of tokens to KEEP when compressing a code block.",
        )
        code_compression_min_tokens: int = Field(
            default=150,
            description="Minimum tokens a code block must have before compression is attempted.",
        )
        enable_question_aware_compression: bool = Field(
            default=True,
            description="Preserve tokens relevant to the user's question during code compression.",
        )
        # ── Multi‑phase code history ────────────────────────────────
        enable_code_history_compression: bool = Field(
            default=True,
            description="Replace old multi‑phase code parts with compact commit summaries.",
        )
        code_history_force_compress_after_turns: int = Field(
            default=8,
            ge=0,
            description=(
                "If a code-bearing history message stays compression-eligible "
                "but blocked by the symbol-index ratio for more than this many "
                "turns, compress it anyway WITHOUT an /expand guarantee "
                "(marked '[🗜️ CÓDIGO COMPRIMIDO — sin índice]'). Prevents the "
                "history anti-growth guarantee from being silently disabled "
                "when assistant-code indexing degrades or the ratio threshold "
                "is set high. 0 = never force (legacy: ratio gate is absolute)."
            ),
        )
        code_history_keep_last_n_parts: int = Field(
            default=3,
            ge=1,
            le=5,
            description="Steps to remember in multi phase processes.",
        )
        code_history_symbol_index_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
        # ── User code in history ────────────────────────────────────
        enable_lean_user_code: bool = Field(
            default=True,
            description=(
                "If enabled, large code blocks in user messages are replaced with a compact "
                "stub. The full code is stored in LTM and SymbolIndex, and remains recoverable "
                "via `/expand` or LOD activation. Only the stub stays in the conversation "
                "history, saving context tokens.\n\n"
                "If disabled, all user code blocks are kept in the conversation history "
                "verbatim (may increase token usage and context size)."
            ),
        )
        lean_user_code_min_tokens: int = Field(
            default=12000,
            description=(
                "After this many tokens, compress a code block"
                "into a stub in the expanded code context."
                "In real life applies to LOD3 mostly."
            ),
        )
        # ── Conversation summaries ──────────────────────────────────
        summarize_old_messages: bool = Field(
            default=True,
            description="Summarise conversation messages that are trimmed from the history.",
        )
        max_conversation_summaries: int = Field(
            default=3,
            ge=0,
            description="Maximum conversation summary blocks kept and re‑injected each request. 0 = keep all.",
        )
        summarization_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        summary_fallback_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )

        # ═══════════════════════════════════════════════════════════════════
        #  SymbolGraph & active code
        # ═══════════════════════════════════════════════════════════════════
        # ── Extraction & detection ──────────────────────────────────
        enable_code_awareness: bool = Field(default=True)
        auto_detect_code_blocks: bool = Field(default=True)
        code_block_pattern: str = Field(default="```(\\w*)\\n(.*?)```")
        track_file_paths: bool = Field(default=True)
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b"
        )
        track_line_numbers: bool = Field(default=True)
        exclude_filter_internals: bool = Field(default=True)
        # ── Call‑graph extraction ───────────────────────────────────
        enable_call_graph_extraction: bool = Field(default=True)
        enable_data_flow_analysis: bool = Field(default=True)
        # ── Generate missing symbol docstrings ──────────────────────
        enable_auto_docstrings: bool = Field(
            default=True,
            description="Automatically generate missing docstrings for functions and methods using the LLM.",
        )
        enable_cfg_skeletons: bool = Field(
            default=True,
            description=(
                "Generate and inject compressed control-flow skeletons (branches "
                "preserved, straight-line bodies elided) for LOD2-tier symbols when "
                "the use-case is refactor (/refactor, case D) or the query has high "
                "debug intent. Deterministic, no LLM call."
            ),
        )
        cfg_skeleton_debug_intent_threshold: float = Field(
            default=0.4,
            ge=0.0,
            le=1.0,
            description=(
                "Minimum intent_vector['debug'] weight that triggers CFG skeleton "
                "injection outside of the explicit refactor use-case (D)."
            ),
        )
        cfg_skeleton_max_lines: int = Field(
            default=40,
            ge=5,
            description=(
                "Skip CFG skeleton generation for functions whose source snippet "
                "exceeds this many lines — very long functions produce skeletons "
                "too large to count as 'compressed'."
            ),
        )
        # Ideally, use the same value.
        lazy_docstring_max_per_turn: int = Field(
            default=25,
            ge=0,
            description="Maximum number of docstrings generated on-demand (lazy) per turn. 0 = unlimited.",
        )
        docstring_batch_size: int = Field(
            default=25,
            ge=1,
            le=20,
            description=(
                "How many symbols to bundle into a single lazy-docstring LLM "
                "call. Higher = fewer round trips (important under "
                "--parallel 1, where calls cannot run concurrently) at the "
                "cost of a larger prompt/response per call."
            ),
        )
        docstring_bg_batch_size: int = Field(
            default=5,
            ge=1,
            le=20,
            description=(
                "Number of symbols per batch in the background docstring loop. "
                "Smaller batches produce shorter prompts and reduce per-call latency. "
                "Foreground batching (during LOD-2 pre-resolution) uses docstring_batch_size."
            ),
        )
        enable_auto_docstrings_background: bool = Field(
            default=True,
            description=(
                "Launch background tasks from the outlet to generate missing docstrings. "
                "Requires --parallel > 1 on the server to avoid blocking the next user turn. "
                "When disabled, docstrings are only generated on-demand (lazy)."
            ),
        )
        # ── Block deduplication ─────────────────────────────────────
        code_similarity_threshold: float = Field(default=0.85)
        enable_ast_deduplication: bool = Field(default=True)
        auto_remove_duplicate_blocks: bool = Field(default=True)
        max_duplicate_age_hours: float = Field(default=6.0)
        # ── Active block management ─────────────────────────────────
        max_active_blocks: int = Field(default=0, ge=0)
        max_base_code_blocks: int = Field(default=3)
        max_proposed_changes: int = Field(default=5)
        max_committed_changes: int = Field(default=10)
        prioritize_recent_code: bool = Field(default=True)
        enable_obsolete_marking: bool = Field(default=True)
        # ── Obsolete version cap ──────────────────────────
        # increase to improve backtracing over code changes
        max_obsolete_versions_per_file: int = Field(
            default=3,
            ge=0,
            description=(
                "Maximum number of obsolete code versions to keep per file. "
                "0 (default) removes them immediately. Values > 0 keep the "
                "N most recent obsolete versions for potential rollback, "
                "evicting older ones."
            ),
        )
        # ── Diffs & commits ─────────────────────────────────────────
        enable_diff_application: bool = Field(default=True)
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@"
        )
        commit_pattern: str = Field(default="commit\\s+([a-f0-9]{7,40})")
        # ── Hub symbols in Block A ──────────────────────────────────
        symbol_index_max_in_block_a: int = Field(
            default=30,
            ge=5,
            le=200,
            description="Maximum hub symbols (top‑N by centrality) kept in Block A.",
        )
        # ── Call graph depth in Block A ─────────────────────────────
        call_graph_context_mode: str = Field(
            default="full_graph",
            description=(
                "Control the depth of call graph injected into Block A.\n"
                "- 'auto': resolved per-query from use_case/intent/project size/token budget (default).\n"
                "- 'hubs_only': only top-N hub symbols (~300-500 tokens).\n"
                "- 'expanded_hubs': top-N hubs + all direct callers/callees, depth 1 (~2k-5k tokens).\n"
                "- 'full_graph': every symbol in the project with direct callers/callees (~8k-20k+ tokens)."
            ),
        )
        full_graph_max_tokens: int = Field(
            default=60000,
            ge=1000,
            description=(
                "Token budget for full_graph mode. If the rendered graph would exceed "
                "this, it is truncated (alphabetical cutoff) with an explicit notice — "
                "never silently downgraded to a different mode."
            ),
        )
        expanded_hubs_max_tokens: int = Field(
            default=10000,
            ge=500,
            description="Token budget for expanded_hubs mode. Same truncation behavior as full_graph_max_tokens.",
        )
        call_graph_auto_full_graph_symbol_ceiling: int = Field(
            default=300,
            ge=20,
            description=(
                "In auto mode, full_graph is only considered when get_all_qualified_names() "
                "count is at or below this ceiling. Above it, auto caps at expanded_hubs."
            ),
        )
        # ── Window-relative guard floors ────────────
        full_graph_min_free_token_ratio: float = Field(
            default=0.38,
            ge=0.05,
            le=0.95,
            description=(
                "Minimum fraction of context_window_tokens that must remain "
                "free for 'full_graph' call-graph mode to be allowed. "
                "Replaces a previously hardcoded absolute floor so the guard "
                "scales with the window instead of becoming unreachable on "
                "small-window models. 0.38 * 262000 ≈ 100k, matching prior "
                "behavior at the default window."
            ),
        )
        expanded_hubs_min_free_token_ratio: float = Field(
            default=0.076,
            ge=0.01,
            le=0.95,
            description=(
                "Minimum fraction of context_window_tokens that must remain "
                "free for 'expanded_hubs' mode. 0.076 * 262000 ≈ 20k, matching "
                "prior behavior at the default window."
            ),
        )
        call_graph_auto_expanded_hubs_symbol_ceiling: int = Field(
            default=600,
            ge=50,
            description=(
                "In auto mode, expanded_hubs is only considered when symbol count is at or "
                "below this ceiling. Above it, auto caps at hubs_only regardless of intent."
            ),
        )
        call_graph_auto_min_free_tokens_for_full: int = Field(
            default=100000,
            ge=0,
            description=(
                "In auto mode, full_graph additionally requires this many free tokens in "
                "the effective context budget. Prevents auto from injecting 20k tokens "
                "into an already-tight window."
            ),
        )
        call_graph_auto_min_free_tokens_for_expanded: int = Field(
            default=20000,
            ge=0,
            description="Same guard as above, applied to expanded_hubs in auto mode.",
        )
        call_graph_mode_downgrade_after_turns: int = Field(
            default=3,
            ge=1,
            le=20,
            description=(
                "In auto mode, a resolved downgrade (e.g. full_graph → hubs_only) "
                "only commits after this many consecutive turns that would "
                "resolve lower than the currently active mode. Upgrades always "
                "apply immediately. Prevents Block A from flip-flopping (and "
                "forcing a full KV-cache prefill) on every topic switch."
            ),
        )
        call_graph_mode_recompute_pagerank: bool = Field(
            default=False,
            description=(
                "If True, recompute PageRank when the call-graph mode changes. "
                "Mode does not alter graph topology, so this is normally unnecessary; "
                "kept as an opt-in safety/debug switch."
            ),
        )
        call_graph_downgrade_turns: int = Field(
            default=3,
            ge=1,
            le=20,
            description=(
                "Number of consecutive turns in which the resolved mode would be lower "
                "than the current mode before a downgrade is applied. "
                "Upgrades (adding context) are always immediate."
            ),
        )

        # ── Inventory (structural / listing queries) ────────────────
        enable_hierarchical_inventory: bool = Field(
            default=True,
            description="For large codebases, serve an architecture-first inventory (classes-with-responsibility + grouped functions) instead of a flat list.",
        )
        inventory_hierarchical_threshold: int = Field(
            default=80,
            ge=20,
            description="Symbol count above which the inventory switches to hierarchical mode.",
        )
        # ── Soft‑eviction ───────────────────────────────────────────
        enable_block_paging: bool = Field(
            default=True,
            description="Soft‑evict low‑activation code blocks to ChromaDB instead of dropping them.",
        )
        block_paging_threshold: int = Field(
            default=15,
            ge=5,
            le=100,
            description="active_blocks count above which soft paging starts.",
        )
        block_paging_min_activation: float = Field(
            default=0.15,
            ge=0.01,
            le=0.5,
            description="PPR activation score below which a block becomes a paging candidate.",
        )
        block_paging_max_concurrent_embeddings: int = Field(
            default=2,
            ge=1,
            le=16,
            description=(
                "Max concurrent background embedding tasks during block page-out. "
                "Mass eviction (e.g. a large paste overflowing max_active_blocks) "
                "would otherwise spawn one embedder.encode per evicted block at "
                "once — an unbounded RAM/VRAM spike on the embedder. The block is "
                "removed from active_blocks immediately regardless; only the "
                "background embedding is rate-limited."
            ),
        )
        # ── Purge old versions ──────────────────────────────────────
        purge_old_code_versions_enabled: bool = Field(
            default=True,
            description="Move code versions beyond the N most recent per file to cold storage.",
        )
        purge_old_code_versions_max_per_file: int = Field(
            default=3,
            ge=1,
            le=20,
            description="Number of recent code versions per file to keep in active context.",
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Activation graph (PPR, LOD, seeds)
        # ═══════════════════════════════════════════════════════════════════
        # ── Path activation ─────────────────────────────────────────
        enable_path_analysis: bool = Field(default=True)
        path_activation_threshold: float = Field(default=0.3, ge=0.01, le=1.0)
        path_relevance_high_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        path_propagation_steps: int = Field(default=6, ge=1, le=8)
        path_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        path_summary_max_tokens: int = Field(default=80)
        # ── LOD thresholds ──────────────────────────────────────────
        lod1_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
        lod2_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
        lod3_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
        # ── LOD by use case ───────────────────────────────────
        enable_lod_by_intent: bool = Field(
            default=True,
            description=(
                "Tune Block B LOD policy and activation direction per use case "
                "(A architecture, B plans, C programming, D refactor, "
                "E scaffolding) instead of the flat intent-scaling. "
                "Off = legacy scale() behaviour."
            ),
        )
        lod_intent_explicit_override: bool = Field(
            default=True,
            description=(
                "Allow an explicit command prefix (/arch, /plan, /code, "
                "/refactor, /scaffold) at the start of the message to force the "
                "use case, overriding auto-detection."
            ),
        )
        lod_intent_refactor_callers_max: int = Field(
            default=12,
            ge=0,
            description=(
                "Max DIRECT callers pulled into Block B at LOD-1 for refactor "
                "(case D) impact analysis. 0 = unlimited."
            ),
        )
        lod2_exit_ratio: float = Field(
            default=0.60,
            ge=0.3,
            le=0.9,
            description=(
                "Fraction of lod2_threshold used as the exit threshold for LOD-2 "
                "hysteresis. A symbol that entered LOD-2 will stay until its PPR "
                "drops below lod2_threshold * lod2_exit_ratio. "
                "Lower values = symbols stay in LOD-2 longer."
            ),
        )
        # ── Centrality ──────────────────────────────────────────────
        enable_centrality_prior: bool = Field(default=True)
        enable_centrality_lod_bump: bool = Field(default=True)
        centrality_lod_bump_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        centrality_lod_bump_weight: float = Field(default=0.15, ge=0.0, le=0.5)
        # ── Seeds ───────────────────────────────────────────────────
        enable_traceback_activation: bool = Field(default=True)
        enable_history_seeds: bool = Field(default=True)
        history_seeds_lookback: int = Field(default=6, ge=2, le=20)
        history_seeds_max_boost: float = Field(default=0.6, ge=0.1, le=0.9)
        enable_multi_seed_activation: bool = Field(default=True)
        multi_seed_weight_lexical: float = Field(default=0.5, ge=0.0, le=1.0)
        multi_seed_weight_structural: float = Field(default=0.3, ge=0.0, le=1.0)
        multi_seed_weight_historical: float = Field(default=0.2, ge=0.0, le=1.0)
        ppr_alpha: float = Field(default=0.90, ge=0.5, le=0.99)
        # ── LOD adaptation ──────────────────────────────────────────
        enable_lod_adaptive: bool = Field(default=True)
        lod_adapt_rate: float = Field(default=0.05, ge=0.01, le=0.2)
        lod_adapt_min: float = Field(default=0.25, ge=0.1, le=0.5)
        lod_adapt_max: float = Field(default=0.75, ge=0.5, le=0.95)
        lod_adapt_underserved_min: int = Field(default=2, ge=1, le=10)
        lod_adapt_overserved_min: int = Field(default=3, ge=1, le=10)

        # ═══════════════════════════════════════════════════════════════════
        #  Reasoning (Chain‑of‑Thought)
        # ═══════════════════════════════════════════════════════════════════
        # ── Architecture-mode CoT ────────────────────────────────────
        enable_skeleton_cot: bool = Field(
            default=True,
            description=(
                "For architecture / design / refactor queries, use the code "
                "skeleton (contracts only) as the CoT reasoning context instead "
                "of the full system prompt. Produces cleaner hypotheses and plans."
            ),
        )
        skeleton_cot_max_tokens: int = Field(
            default=1600,
            ge=200,
            le=2000,
            description="Token budget for the architecture reasoning chain.",
        )
        enable_skeleton_ltm: bool = Field(
            default=True,
            description="Store the generated skeleton in LTM so future sessions can retrieve the architecture without re-deriving it.",
        )
        skeleton_ltm_expiration_days: int = Field(
            default=14,
            ge=0,
            description="How long to keep skeleton snapshots in LTM. 0 = never expire.",
        )
        enable_cot_expand_resolution: bool = Field(
            default=True,
            description=(
                "Auto-resolve /expand <Name> hints emitted by the architecture "
                "CoT: retrieve the full symbol body and annotate it directly in "
                "the reasoning output before the main model sees it."
            ),
        )
        cot_expand_max_symbols: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Maximum number of /expand hints resolved per CoT turn.",
        )
        cot_expand_max_tokens: int = Field(
            default=3000,
            ge=200,
            description="Token budget for all auto-resolved expansions combined.",
        )
        enable_scientific_arch_reasoning: bool = Field(
            default=True,
            description=(
                "For architecture queries at CoT level 3, use multi-hypothesis "
                "scientific reasoning on the skeleton (design options evaluated "
                "against static evidence) instead of single-chain L2 arch reasoning."
            ),
        )
        # ── Detection & generation ──────────────────────────────────
        enable_cot_on_demand: bool = Field(default=True)
        auto_cot_enabled: bool = Field(default=False)
        auto_cot_min_chars: int = Field(default=200)
        cot_model: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level2: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level3: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        enable_cot_llm_detection: bool = Field(default=True)
        # ── Scientific method ───────────────────────────────────────
        enforce_scientific_method: bool = Field(default=False)
        scientific_hypotheses_count: int = Field(default=3, ge=2, le=6)
        scientific_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
        scientific_max_iterations: int = Field(default=2, ge=1, le=4)
        # ── Step‑back prompting ─────────────────────────────────────
        enable_step_back_prompting: bool = Field(default=True)
        step_back_always: bool = Field(default=False)
        step_back_max_tokens: int = Field(default=150, ge=50, le=400)
        # ── Contradictions ──────────────────────────────────────────
        enable_contradiction_detection: bool = Field(default=True)
        contradiction_detection_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        contradiction_inject_warning: bool = Field(default=True)
        # ── Confidence scoring ──────────────────────────────────────
        enable_confidence_scoring: bool = Field(default=True)
        confidence_prompt: str = Field(
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]'..."
        )

        # ═══════════════════════════════════════════════════════════════════
        #  LLM & orchestration
        # ═══════════════════════════════════════════════════════════════════
        # ── Endpoint & model ────────────────────────────────────────
        LLM_BASE_URL: str = Field(default="http://host.docker.internal:8080")
        LLM_API_TOKEN: str = Field(default="")
        llm_model: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        llm_request_timeout: int = Field(default=900)
        llm_per_call_timeout: int = Field(default=900, ge=1)
        llm_retry_total_timeout: int = Field(default=950, ge=10)
        LLM_CACHE_TTL: int = Field(default=300)
        LLM_CACHE_MAX_SIZE: int = Field(default=100)
        llamacpp_endpoint_type: str = Field(default="chat")
        # ── Auxiliary models ────────────────────────────────────────
        code_block_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        session_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        natural_language_forget_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        # ── Multi‑phase response ────────────────────────────────────
        enable_multi_phase_response: bool = Field(default=True)
        force_multi_phase_response: bool = Field(
            default=False,
            description="Force multi-phase response protocol even when budget is not tight.",
        )
        multi_phase_effective_max_tokens: int = Field(default=8000, ge=1000, le=200000)
        multi_phase_response_threshold: int = Field(default=7000, ge=0, le=200000)
        multi_phase_response_budget_warn: int = Field(default=800, ge=500, le=40000)
        auto_budget_context_for_parts: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════════
        #  Interaction & commands
        # ═══════════════════════════════════════════════════════════════════
        # ── Explicit commands ───────────────────────────────────────
        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=True)
        outlet_expand_intercept_enabled: bool = Field(default=True)
        outlet_expand_intercept_max_symbols: int = Field(default=0, ge=0)
        outlet_expand_intercept_depth: int = Field(default=5, ge=0)
        expand_default_depth: int = Field(default=2)
        enable_skeleton_intent: bool = Field(
            default=True,
            description="Serve a copy-pasteable signature-only code skeleton (bodies as ...) for scaffolding queries (esqueleto / skeleton / stubs / solo firmas).",
        )
        # ── Proactive suggestions ───────────────────────────────────
        enable_command_suggestions: bool = Field(default=True)
        command_suggestion_cooldown_minutes: int = Field(default=10)
        proactive_summary_threshold: float = Field(default=0.95)
        # ── Context cleanup ─────────────────────────────────────────
        cleanup_suggestions_enabled: bool = Field(default=True)
        cleanup_inactive_threshold_messages: int = Field(default=30)
        cleanup_excluded_content_types: list = Field(
            default_factory=lambda: ["BASE_CODE"]
        )
        cleanup_status_command_enabled: bool = Field(default=True)
        cleanup_proactive_suggestions: bool = Field(default=True)
        cleanup_suggestion_cooldown_messages: int = Field(default=20)
        cleanup_command_enabled: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════════
        #  Session & state
        # ═══════════════════════════════════════════════════════════════════
        # ── Session management ──────────────────────────────────────
        project_id: str = Field(default="default")
        max_cached_projects: int = Field(default=10)
        state_db_path: str = Field(default="/app/backend/data/conversation_state.db")
        preserve_tool_calls: bool = Field(default=True)
        # ── Session summaries ───────────────────────────────────────
        enable_session_summary: bool = Field(default=True)
        session_summary_interval_messages: int = Field(default=8)
        session_summary_max_tokens: int = Field(default=200)
        # ── Turn-based window (summarize@N / evict@M) ───────────────
        summarize_batch_turns: int = Field(
            default=5,
            ge=1,
            le=30,
            description="Minimum number of unsummarized turns to accumulate before generating one summary (limits fragmentation).",
        )
        # ── Hierarchical (L1 → L2) consolidation ────────────────────
        enable_hierarchical_summaries: bool = Field(
            default=True,
            description="Fold the oldest L1 turn-range summaries into a single L2 summary so the cap retains broad coverage.",
        )
        hierarchical_summary_group_size: int = Field(
            default=4,
            ge=2,
            le=12,
            description="Number of oldest L1 summaries folded into one L2 summary.",
        )
        max_hierarchical_summaries: int = Field(
            default=2,
            ge=0,
            description="Maximum L2 summaries kept in the direct re-injection. 0 = keep all.",
        )
        hierarchical_summary_max_tokens: int = Field(
            default=250,
            ge=80,
            le=800,
            description="Token budget for an L2 consolidated summary.",
        )
        # ── Feedback tracking ───────────────────────────────────────
        enable_feedback_tracking: bool = Field(default=True)
        feedback_history_limit: int = Field(default=10)
        inject_feedback_context: bool = Field(default=True)
        feedback_importance_penalty_for_failure: float = Field(default=2.0)
        preserve_error_context: bool = Field(default=True)
        # ── Response cache ──────────────────────────────────────────
        enable_response_cache: bool = Field(default=True)
        response_cache_similarity_threshold: float = Field(default=0.92)
        response_cache_ttl_hours: float = Field(default=24.0)
        response_cache_max_entries: int = Field(default=100)
        response_cache_include_context_hash: bool = Field(default=True)
        # ── Duplicate detection ─────────────────────────────────────
        duplicate_question_threshold: float = Field(default=0.92)
        duplicate_question_lookback: int = Field(default=20)
        duplicate_question_lookback_hours: float = Field(default=24.0)

        # ═══════════════════════════════════════════════════════════════════
        #  Performance & persistence
        # ═══════════════════════════════════════════════════════════════════
        # ── KV cache ────────────────────────────────────────────────
        enable_kv_cache_stability: bool = Field(default=True)
        enable_slot_persistence: bool = Field(default=True)
        slot_save_path: str = Field(default="/kvcache")
        slot_id: int = Field(default=0, ge=0)
        # ── Slot save threshold guard ────────────────────────────────
        slot_save_max_context_tokens: int = Field(
            default=0,
            ge=0,
            description=(
                "Skip slot save when the total context exceeds this many tokens. "
                "Saving writes the whole KV state to disk under the server mutex, "
                "stalling all inference for huge contexts. 0 = no guard."
            ),
        )
        # ── Volatility-tiered context ─────────────────────────
        enable_skeleton_tier: bool = Field(
            default=True,
            description=(
                "Inject the project skeleton (signatures) as a stable cache tier "
                "inside Block A. Cached by signature_hash; survives body edits."
            ),
        )
        skeleton_tier_max_tokens: int = Field(
            default=0,
            ge=0,
            description=(
                "Max tokens for the skeleton tier. 0 = unlimited. Over budget → "
                "tier skipped, Block B keeps inline signatures."
            ),
        )
        skeleton_tier_suppresses_block_b_signatures: bool = Field(
            default=True,
            description=(
                "When the skeleton tier is active, Block B emits only bodies "
                "(LOD-3) and summaries (LOD-2), not bare signatures (LOD-0/LOD-1) "
                "— they are already in the stable tier. Case D (refactor) is "
                "exempt: its caller signatures are impact signal, not duplication."
            ),
        )
        skeleton_include_docstrings: bool = Field(
            default=True,
            description=(
                "Include one-line docstrings (source + LLM-generated) in the skeleton "
                "tier. Improves LLM comprehension. Cost: the skeleton tier cache key "
                "becomes docstring-aware, so Block A re-renders as background "
                "docstrings land, then stabilizes. Set False for a strictly stable, "
                "signature-only Block A prefix if KV-cache churn is observed."
            ),
        )
        emergency_max_turns: int = Field(
            default=4,
            ge=1,
            le=20,
            description=(
                "Turns to keep when an individual turn exceeds budget * 0.8 (emergency cap). "
                "Replaces max_turns=8 from adaptive_trim. Default is conservative (4) because "
                "in coding sessions a single turn may contain a whole file."
            ),
        )
        # ── Monotonic compaction (#16) ──────────────────────────────
        compaction_defer_during_autocontinue: bool = Field(
            default=True,
            description=(
                "Skip turn-based summarize/evict while an AutoContinue multi-part "
                "session is active, to avoid breaking the KV cache mid-generation."
            ),
        )
        # ── Graph persistence ───────────────────────────────────────
        enable_edge_persistence: bool = Field(default=True)
        # ── Speculative prefetch ────────────────────────────────────
        enable_speculative_prefetch: bool = Field(default=True)
        speculative_prefetch_max: int = Field(default=5, ge=1, le=20)
        # ── Silent ingestion ────────────────────────────────────────
        enable_silent_ingestion: bool = Field(default=True)
        # ── DB orphans cleanup ────────────────────────────────────────
        purge_orphaned_data_interval: int = Field(
            default=10,
            ge=0,
            description="Number of turns between automatic purges of orphaned DB rows (0 = disabled).",
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Utilities & tuning
        # ═══════════════════════════════════════════════════════════════════
        debug: bool = Field(default=True)
        # ── Context dump (evolution tracking) ───────────────────────
        enable_context_dump: bool = Field(
            default=True,
            description=(
                "Dump the assembled per-turn context (Block A, Block B, message "
                "window) to disk for evolution tracking. Off by default; writes "
                "are best-effort and fully decoupled from the request path."
            ),
        )
        context_dump_dir: str = Field(
            default="/app/backend/data/context_dumps",
            description="Directory for per-turn context snapshots (one subdir per project).",
        )
        context_dump_max_files_per_project: int = Field(
            default=200,
            ge=0,
            description="Max Markdown snapshots kept per project (oldest pruned). 0 = keep all.",
        )
        context_dump_include_messages: bool = Field(
            default=True,
            description="Include the non-system message window in each snapshot.",
        )
        context_dump_message_max_chars: int = Field(
            default=8000,
            ge=0,
            description="Truncate each captured message body to this many chars. 0 = no truncation.",
        )
        context_dump_write_jsonl: bool = Field(
            default=True,
            description="Append a compact metrics line per turn to evolution.jsonl (token counts + hashes).",
        )
        priority: int = Field(default=0)
        use_tiktoken: bool = Field(default=True)
        # ── Weighting & decay ───────────────────────────────────────
        raw_file_priority_boost: float = Field(default=2.0)
        importance_mention_boost: float = Field(default=0.2)
        importance_recency_half_life_hours: float = Field(default=2.0)
        block_expiration_hours: float = Field(default=24.0)
        proposed_change_retention_turns: int = Field(default=20)
        error_retention_turns: int = Field(default=15)
        track_active_code_age: bool = Field(default=True)
        active_code_timeout_minutes: int = Field(default=45)
        recent_activity_window_minutes: int = Field(default=15)
        max_change_summaries: int = Field(default=1000)
        frequency_weight_factor: float = Field(default=0.3)
        min_mentions_for_boost: int = Field(default=3)
        frequency_decay_hours: float = Field(default=12.0)

        # ═══════════════════════════════════════════════════════════════════
        #  Architecture Map
        # ═══════════════════════════════════════════════════════════════════
        enable_architecture_map: bool = Field(
            default=True,
            description=(
                "Inject a compact class→methods outline into Block A, on top "
                "of the existing hub-symbols section. Cheap, deterministic, "
                "cache-stable while code is unchanged."
            ),
        )
        architecture_map_max_tokens: int = Field(
            default=0,
            ge=0,
            description="Token budget for the class outline section. 0 = unlimited.",
        )
        enable_hub_callees: bool = Field(
            default=True,
            description=(
                "Show outgoing calls ('→ calls:') for hub symbols, alongside "
                "the existing incoming-callers line ('← used by:')."
            ),
        )

        # ═══════════════════════════════════════════════════════════════════════
        #  Hub‑Bodies Tier (stable full bodies of top hubs)
        # ═══════════════════════════════════════════════════════════════════════
        # This region controls the amount of permanent code permanently expanded.
        # This provides better visibility and prevent excessive filling.
        enable_hub_bodies_tier: bool = Field(
            default=True,
            description=(
                "Enable the stable hub‑bodies tier in Block A. "
                "When enabled, the full bodies of the top‑N most central symbols "
                "are injected as a cacheable tier between Block A and Block B. "
                "This dramatically improves recall for queries about core symbols "
                "without increasing LoD pressure.\n\n"
                "Off by default until validated on hardware. Estimated cost: "
                "+3.3s cold start, 0s on read‑only turns (slot hit), "
                "+1.7s on first edit of a cold hub."
            ),
        )

        hub_bodies_tier_top_n: int = Field(
            default=7,
            ge=1,
            le=20,
            description=(
                "Number of top hubs to include in the tier. "
                "PageRank centrality is heavy‑tailed; the top‑7 are genuinely "
                "central. N > 7 dilutes attention without proportional gain.\n\n"
                "Recommended: 7 (default). Range 1‑20."
            ),
        )

        hub_bodies_tier_min_centrality: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description=(
                "Minimum PageRank centrality score to qualify for the tier. "
                "0.0 = no floor (only top‑N applies). Values > 0 automatically "
                "prune low‑centrality hubs even if they are in the top‑N.\n\n"
                "Useful for very large codebases where even the 7th hub is "
                "peripheral. Default 0.0 (rely on top‑N only)."
            ),
        )

        hub_bodies_tier_max_tokens: int = Field(
            default=10000,
            ge=500,
            description=(
                "Token budget for the entire tier. If the rendered tier exceeds "
                "this, hubs are dropped from the bottom (most volatile) until "
                "the budget is met.\n\n"
                "Auto‑capped to 6000 if multi‑phase response is active "
                "(see enable_multi_phase_response / force_multi_phase_response).\n\n"
                "Recommended: 10000 (MP‑OFF), 6000 (MP‑ON)."
            ),
        )

        hub_bodies_tier_max_body_tokens: int = Field(
            default=1500,
            ge=200,
            description=(
                "Maximum tokens allowed for an individual hub body. "
                "Hubs larger than this are skipped (they go via LoD).\n\n"
                "Rationale: very large bodies (e.g., 200‑line functions) have "
                "poor re‑prefill cost when edited and cause attention dilution "
                "when irrelevant. LoD injects them only when activated.\n\n"
                "Recommended: 1500 (default). Range 200‑4000."
            ),
        )

        hub_bodies_tier_protect_from_paging: bool = Field(
            default=True,
            description=(
                "Prevent code blocks that contain hubs in the tier from being "
                "paged out by ContextPager.\n\n"
                "Required for tier correctness. Keep enabled unless debugging."
            ),
        )

        hub_bodies_tier_recency_pointers: bool = Field(
            default=True,
            description=(
                "Include recency pointers for hub seeds in Block B.\n\n"
                "CRITICAL for attention: the full bodies are in the tier "
                "(zone media, ~22k‑27k), but the recency pointer creates an "
                "anchor near the query, enabling effective recall.\n\n"
                "Keep enabled unless you measure that the model recovers well "
                "without it. Default True."
            ),
        )

        hub_bodies_tier_warmup_on_ingestion: bool = Field(
            default=False,
            description=(
                "Background prefill of the stable prefix (Block A + tier) "
                "after silent ingestion.\n\n"
                "When enabled, the tier is built and a dummy 1‑token request "
                "is sent to the LLM, populating the KV cache. The first real "
                "query then does a slot restore instead of a cold prefill.\n\n"
                "Hides the +3.3s cold start cost. Default False (Phase 2)."
            ),
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Semantic seed inference (LLM-guided LOD-3 selection)
        # ═══════════════════════════════════════════════════════════════════
        seed_inference_mode: str = Field(
            default="auto",
            description=(
                "LLM‑guided seed inference before PPR.\n"
                "'auto': infer when lexical seeds are scarce "
                "(< seed_inference_min_lexical) or the use case is A/D.\n"
                "'always': always in code sessions.\n"
                "'off': disabled."
            ),
        )
        seed_inference_model: str = Field(
            default="",
            description=(
                "Model for seed inference. " "Empty = use llm_model (the main model)."
            ),
        )
        seed_inference_min_lexical: int = Field(
            default=2,
            ge=0,
            description=(
                "In 'auto' mode: infer if the query names fewer than N symbols "
                "from the index literally."
            ),
        )
        seed_inference_min_chars: int = Field(
            default=15,
            ge=0,
            description="Minimum query length to trigger inference.",
        )
        seed_inference_max_symbols: int = Field(
            default=12,
            ge=1,
            le=40,
            description="Maximum symbols seeded by inference.",
        )
        seed_inference_score: float = Field(
            default=0.85,
            ge=0.1,
            le=1.0,
            description=(
                "Seed score assigned to LLM‑validated symbols. "
                "High value (> lod3_threshold) guarantees LOD‑3 (full body)."
            ),
        )
        seed_inference_skeleton_max_tokens: int = Field(
            default=6000,
            ge=500,
            description=(
                "Skeleton token cap sent to the planner LLM. "
                "0 = no cap (not recommended)."
            ),
        )
        seed_inference_max_tokens: int = Field(
            default=200,
            ge=50,
            description="Token cap for the planner's response.",
        )
        # ── Fuzzy matching for rapidfuzz ──────────────────────────────
        seed_inference_fuzzy_threshold: float = Field(
            default=0.85,
            ge=0.6,
            le=1.0,
            description="Minimum token_set_ratio for fuzzy matching of hallucinated ids.",
        )
        seed_inference_fuzzy_penalty: float = Field(
            default=0.8,
            ge=0.5,
            le=1.0,
            description="Score multiplier for symbols found via fuzzy matching.",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self):
        # Valves and basic objects
        self.valves = self.Valves()

        self.tokenizer = None
        self._db_conn = None
        self._cross_encoder = None
        self._cross_encoder_unavailable_logged = False
        self._cross_encoder_lock = asyncio.Lock()

        self._conv_compressor = _shared_get_conversation_compressor()
        self._llmlingua_compressor = (
            self._conv_compressor.raw if self._conv_compressor else None
        )

        self._state_store = StateStore(self)
        self._conversation_state_manager = ConversationStateManager(self)

        self._ltm = LongTermMemory(self)
        self._llm_orchestrator = LLMOrchestrator(self)
        self._reasoning = ReasoningEngine(self)
        self._multi_phase = MultiPhasePlanner(self)
        self._commands = CommandRouter(self)
        self._code_blocks = CodeBlockManager(self)
        self._activation = ActivationEngine(self)
        self._history_compressor = HistoryCompressor(self)
        self._tokens = TokenUtils(self)
        self._enrichment = EnrichmentTasks(self)
        self._inlet_orch = InletOrchestrator(self)
        self._active_code_updater = ActiveCodeUpdater(self)
        self._system_prompt_builder = SystemPromptBuilder(self)
        self._message_assembler = MessageAssembler(self)
        self._context_dumper = ContextDumper(self)
        self._seed_inferencer = SemanticSeedInferencer(self)

        self._hub_index = HubSymbolIndex()
        self._ctx_builder = ContextBuilder(self)
        self._pager = ContextPager(self)
        self._raptor = RaptorCodeIndex()

        # Patterns
        self.code_pattern = re.compile(self.valves.code_block_pattern, re.DOTALL)
        self.diff_pattern = re.compile(self.valves.diff_pattern)
        self.commit_pattern = re.compile(self.valves.commit_pattern, re.IGNORECASE)

        # Tokenizer
        if HAS_TIKTOKEN and self.valves.use_tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._log_debug("Tiktoken initialized")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")

        # State database
        self._state_store.init_db()

        # Long‑term memory (ChromaDB + embeddings)
        self.embedder = None
        self.chroma_client = None
        self.memory_collection = None
        self._response_cache_collection = None
        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._ltm.init()
        else:
            logger.warning("Long‑term memory or code awareness disabled")

        # Reranker (module‑level singleton)
        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._cross_encoder = _get_cross_encoder(self.valves.reranker_model)
        else:
            self._cross_encoder = None

        # HTTP session and locks
        self._project_locks: Dict[str, ReentrantAsyncLock] = {}
        self._lock_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

        # Semaphores
        self._llm_semaphore = asyncio.Semaphore(1)
        self._pending_llm: Dict[str, asyncio.Future] = {}
        self._pending_llm_lock = asyncio.Lock()
        self._llm_orchestrator.init_cache()
        self._last_used_model: Optional[str] = None

        # ── Tracking of active LLM tasks ──
        self._active_llm_tasks: Set[asyncio.Task] = set()
        self._active_llm_tasks_lock = asyncio.Lock()

        # ── Database write queue ──
        self._db_write_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._db_worker_task = asyncio.create_task(self._state_store.db_worker())

        # Session classification cache
        self._session_classify_cache: Dict[str, Tuple[bool, float]] = {}
        self._session_classify_ttl: float = 1800.0
        self._project_state_manager = ProjectStateManager(self)

        # ── Project tracking ──
        self._last_project_id: str = ""

        # Symbol index and path index
        self._symbol_index = SymbolIndex()
        self._path_index = PathIndex()

        # Block change summaries LRU
        self._block_change_summaries: OrderedDict = OrderedDict()
        self._MAX_CHANGE_SUMMARIES = self.valves.max_change_summaries

        # Thread pools
        import concurrent.futures

        self._db_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="codeaware_db"
        )
        self._chroma_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="codeaware_chroma"
        )

        # CoT heuristic feature flags
        self.ENABLE_ACCENT_NORMALIZATION = True
        self.ENABLE_KEYWORD_COUNT_WEIGHT = True
        self.ENABLE_COT_STICKY = False

        # ── Write counter for periodic tasks ──
        self._write_counter = 0

        # ── Silent ingestion guard ──
        self._is_silent_ingestion = False

        # ── Original user system prompt ──
        self._original_system_prompt: str = ""

        # ── C6: LTM store completion event ──────────────────────────────
        self._ltm_store_complete: asyncio.Event = asyncio.Event()
        self._ltm_store_complete.set()  # initially "complete"

        # --- Validate valve coherence at startup ---
        self._validate_valve_coherence()

        print("[CodeAware] Filter loaded")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Logging utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def _log_debug(self, msg: str):
        if self.valves.debug:
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{timestamp}] [CodeAware] {msg}")

    def _log_timing(self, step_name: str, elapsed_since_start: float, duration: float):
        if self.valves.debug:
            self._log_debug(
                f"[Timing] {step_name}: +{elapsed_since_start:.3f}s (dur={duration:.3f}s)"
            )

    def _log_section(self, title: str, duration: float = None):
        if not self.valves.debug:
            return
        line_len = 70
        title_text = f"  {title}  "
        if duration is not None:
            title_text += f"(dur={duration:.3f}s)  "
        if len(title_text) > line_len - 2:
            title_text = title_text[: line_len - 5] + "..."
        remaining = line_len - len(title_text)
        left = remaining // 2
        right = remaining - left
        line = f"{'=' * left}{title_text}{'=' * right}"
        print(f"[CodeAware] {line}")

    def _validate_valve_coherence(self) -> None:
        """
        Warn about individually-legal valve combinations that silently
        disable or degrade a context guarantee. Pure diagnostics: never
        mutates behavior, only logs. Each check names the guarantee at risk.

        Called once at startup from __init__ to surface misconfigurations
        early. Does not raise exceptions.
        """
        v = self.valves
        warnings: List[str] = []

        window = v.context_window_tokens

        # ------------------------------------------------------------------
        # 1. Mode guards unreachable for the window (Section 1)
        # ------------------------------------------------------------------
        if hasattr(v, "full_graph_min_free_token_ratio"):
            fg_floor = int(window * v.full_graph_min_free_token_ratio)
        else:
            fg_floor = int(window * 0.38)  # fallback default

        if hasattr(v, "expanded_hubs_min_free_token_ratio"):
            eh_floor = int(window * v.expanded_hubs_min_free_token_ratio)
        else:
            eh_floor = int(window * 0.076)  # fallback default

        usable_window = window - v.response_reserve_tokens

        if fg_floor >= usable_window:
            warnings.append(
                f"full_graph mode is effectively unreachable: floor {fg_floor} "
                f">= usable window ({usable_window}). Architecture queries will "
                f"never get the full call graph. Lower full_graph_min_free_token_ratio "
                f"or raise context_window_tokens."
            )

        if eh_floor >= usable_window:
            warnings.append(
                f"expanded_hubs mode is effectively unreachable: floor {eh_floor} "
                f">= usable window ({usable_window}). Refactor queries may be "
                f"limited to hubs_only. Lower expanded_hubs_min_free_token_ratio "
                f"or raise context_window_tokens."
            )

        # ------------------------------------------------------------------
        # 2. Budget arithmetic can underflow
        # ------------------------------------------------------------------
        if hasattr(v, "active_context_max_tokens") and hasattr(
            v, "global_injection_token_budget"
        ):
            if v.active_context_max_tokens > v.global_injection_token_budget:
                warnings.append(
                    "active_context_max_tokens > global_injection_token_budget: "
                    "per-query active context can exceed the total injection "
                    "budget; Block B may be trimmed unpredictably."
                )

        if v.response_reserve_tokens >= window:
            warnings.append(
                f"response_reserve_tokens ({v.response_reserve_tokens}) >= "
                f"context_window_tokens ({window}): effective budget is negative; "
                f"mode guards will always fail."
            )

        # ------------------------------------------------------------------
        # 3. Suppression without guaranteed coverage (Section 3)
        # ------------------------------------------------------------------
        if hasattr(v, "skeleton_tier_suppresses_block_b_signatures") and hasattr(
            v, "call_graph_context_mode"
        ):
            if (
                v.skeleton_tier_suppresses_block_b_signatures
                and v.call_graph_context_mode == "hubs_only"
            ):
                warnings.append(
                    "skeleton_tier_suppresses_block_b_signatures=True with "
                    "call_graph_context_mode='hubs_only': Block A renders no full "
                    "skeleton, so suppressing Block B signatures can hide symbols "
                    "entirely. (Mitigated at runtime by the render-gated "
                    "suppression, but the static combination is still suspicious.)"
                )

        # ------------------------------------------------------------------
        # 4. Compression anti-growth disabled (Section 4)
        # ------------------------------------------------------------------
        if hasattr(v, "code_history_symbol_index_threshold") and hasattr(
            v, "code_history_force_compress_after_turns"
        ):
            if (
                v.code_history_symbol_index_threshold >= 0.95
                and v.code_history_force_compress_after_turns == 0
            ):
                warnings.append(
                    "code_history_symbol_index_threshold is very high and "
                    "code_history_force_compress_after_turns=0: if assistant-code "
                    "indexing dips below the threshold, history compression never "
                    "fires and history grows unbounded. Consider setting "
                    "code_history_force_compress_after_turns > 0."
                )

        # ------------------------------------------------------------------
        # 5. Lazy docstrings vs Block A stability (informational hint)
        # ------------------------------------------------------------------
        if hasattr(v, "skeleton_include_docstrings") and hasattr(
            v, "enable_auto_docstrings"
        ):
            if v.skeleton_include_docstrings and v.enable_auto_docstrings:
                self._log_debug(
                    "Note: docstrings are included in the skeleton and generated "
                    "lazily. The Block A prefix hash is computed over a "
                    "docstring-stripped projection (structure hash), so KV cache "
                    "and slot restore remain stable across docstring population."
                )

        # ------------------------------------------------------------------
        # 6. Slot persistence guard threshold sanity
        # ------------------------------------------------------------------
        if (
            hasattr(v, "slot_save_max_context_tokens")
            and v.slot_save_max_context_tokens > 0
        ):
            if v.slot_save_max_context_tokens < v.context_window_tokens // 2:
                warnings.append(
                    f"slot_save_max_context_tokens ({v.slot_save_max_context_tokens}) "
                    f"is less than half the window ({v.context_window_tokens}). "
                    f"KV slot saves will be skipped even when there is plenty "
                    f"of room, reducing cross-session performance."
                )

        # ------------------------------------------------------------------
        # 7. Multi-phase response thresholds sanity
        # ------------------------------------------------------------------
        if hasattr(v, "multi_phase_response_threshold") and hasattr(
            v, "multi_phase_response_budget_warn"
        ):
            if v.multi_phase_response_budget_warn >= v.multi_phase_response_threshold:
                warnings.append(
                    f"multi_phase_response_budget_warn ({v.multi_phase_response_budget_warn}) "
                    f">= multi_phase_response_threshold ({v.multi_phase_response_threshold}). "
                    f"The critical warning may never fire because the tight budget "
                    f"threshold triggers first."
                )

        # ------------------------------------------------------------------
        # 8. Block paging threshold vs max_active_blocks
        # ------------------------------------------------------------------
        if hasattr(v, "block_paging_threshold") and hasattr(v, "max_active_blocks"):
            if (
                v.max_active_blocks > 0
                and v.block_paging_threshold >= v.max_active_blocks
            ):
                warnings.append(
                    f"block_paging_threshold ({v.block_paging_threshold}) >= "
                    f"max_active_blocks ({v.max_active_blocks}). Paging will never "
                    f"activate because the hard eviction cap is reached first."
                )

        # Log all warnings
        for w in warnings:
            self._log_debug(f"⚠️ VALVE COHERENCE: {w}")

        if not warnings:
            self._log_debug("Valve coherence check: no issues detected.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Code update helper
    # ═══════════════════════════════════════════════════════════════════════════

    async def _update_active_code(
        self, message: dict, project_id: str, is_continuation: bool = False
    ) -> None:
        """Update active blocks and SymbolIndex from a new message."""
        await self._active_code_updater.process(message, project_id, is_continuation)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Inlet helpers
    # ═══════════════════════════════════════════════════════════════════════════

    async def _inlet_build_system_injections(
        self,
        messages,
        project_id,
        user_query,
        user_question,
        is_code_session,
        last_user_msg,
        state,
        slot_free=True,
        intent_vector=None,
    ):
        """
        Build system injections (Block A + Block B) via SystemPromptBuilder.
        """
        return await self._system_prompt_builder.build(
            messages=messages,
            project_id=project_id,
            user_query=user_query,
            user_question=user_question,
            is_code_session=is_code_session,
            last_user_msg=last_user_msg,
            state=state,
            slot_free=slot_free,
            intent_vector=intent_vector,
        )

    async def _inlet_assemble_final_messages(
        self,
        messages,
        project_id,
        static_block,
        dynamic_injections,
        prelim_system,
        last_user_msg,
        is_code_session,
        state,
        __user__,
        user_question,
        has_code_blocks,
        slot_free=True,
    ):
        """
        Delegate final message assembly to MessageAssembler.

        This method is a thin wrapper that forwards all arguments to
        `MessageAssembler.assemble`, which orchestrates the final steps of
        message preparation before sending to the LLM:
          - Chain‑of‑Thought detection and reasoning generation
          - Code‑history compression and lean‑user‑code stubbing
          - LLMLingua‑2 compression of conversation prose
          - Turn‑based window management (summarise/evict)
          - Multi‑phase protocol injection when the token budget is tight
          - Adaptive trimming of old messages with optional summarisation
          - Assembly of the final system prompt (Block A + Block B) and injection

        This separation keeps `inlet` focused on orchestration, delegating the
        complex message‑assembly pipeline to a dedicated component.

        Args:
            messages (list): The current list of conversation messages.
            project_id (str): The project identifier.
            static_block (str): The rendered Block A (static, KV‑cacheable).
            dynamic_injections (list): List of (priority, text) dynamic content.
            prelim_system (str): The preliminary system prompt (Block A + Block B).
            last_user_msg (dict|None): The last user message, if any.
            is_code_session (bool): Whether the session is code‑aware.
            state (dict): The conversation state for the project.
            __user__ (dict|None): The user context from OpenWebUI.
            user_question (str): The extracted question from the user message.
            has_code_blocks (bool): Whether the user message contained code fences.
            slot_free (bool): Whether the LLM slot is free for auxiliary calls.

        Returns:
            list: The final message list ready for the LLM.
        """
        return await self._message_assembler.assemble(
            messages,
            project_id,
            static_block,
            dynamic_injections,
            prelim_system,
            last_user_msg,
            is_code_session,
            state,
            __user__,
            user_question,
            has_code_blocks,
            slot_free,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # INLET – orchestrated entry point
    # ═══════════════════════════════════════════════════════════════════════════
    # Value categories (see project documentation):
    #   🔥 STATE MANAGEMENT    – Critical steps that maintain conversation state
    #   ⚡ COMMAND HANDLING    – User‑initiated context control commands
    #   🧠 ENRICHMENT          – Features that add information to the system prompt
    #   📦 COMPRESSION         – Features that reduce context size to fit the window
    #   🚀 RESOURCE OPTIMISATION – Features that improve speed / avoid conflicts
    # ═══════════════════════════════════════════════════════════════════════════
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Pre‑process the request before the LLM sees it.

        Orchestrates seven sequential steps:
        1. Project‑switch detection, cache loading, and KV‑slot restore.
        2. User‑info extraction (last message, question, explicit commands).
        3. Explicit command dispatch (/forget, /status, /clean, /expand).
        4. Natural‑language intent dispatch (forget, remember, obsolete).
        5. Silent ingestion when the message is a large code‑only paste.
        6. Session classification and active‑code update.
        7. System‑prompt assembly (Block A + Block B) with CoT, compression,
           multi‑phase, and adaptive trimming.

        Returns the modified body with the final message list ready for the LLM.
        """
        self._log_debug("inlet called")
        inlet_start = time.monotonic()
        self._log_section("CONTEXT MANAGER - INLET START")

        def _inlet_timing(step_name: str, start: float, end: float = None):
            if end is None:
                end = time.monotonic()
            self._log_timing(step_name, start - inlet_start, end - start)

        project_id = self._inlet_orch.get_project_id()

        # ── Get state early to decide slot_free ──
        state = self._conversation_state_manager.get(project_id)

        # ── slot_free logic ─────────────────────────────────────────────
        slot_free = True
        # If no model has been used yet and this is the first turn,
        # disable slot_free so we don't try to restore a non‑existent slot.
        if self._last_used_model is None and state.message_count <= 1:
            slot_free = False

        await self._enrichment.cancel_docstring_tasks()
        self._enrichment._lazy_docstrings_generated_this_turn = 0

        # ── Phase A: Write barrier ──────────────────────────────────
        # Wait for all pending writes from the previous turn to finish
        # BEFORE we start reading SQLite.
        await self._state_store.drain_writes(timeout=5.0)

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   1. Preprocess (project switch, cache load)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        messages = await self._inlet_orch.inlet_preprocess(body, project_id)
        _inlet_timing("Step 1/7: Preprocess (project switch, cache load)", step_start)
        if not messages:
            return body

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   2. Extract user info
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        (
            last_user_msg,
            user_query,
            user_question,
            is_explicit_command,
            has_code_blocks,
        ) = await self._inlet_orch.inlet_extract_user_info(messages)
        _inlet_timing("Step 2/7: Extract user info", step_start)

        # ── C6: Wait for previous LTM store to complete ──────────────────
        if not self._ltm_store_complete.is_set():
            self._log_debug("LTM: store pending from previous turn — waiting up to 3s")
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._ltm_store_complete.wait()),
                    timeout=3.0,
                )
                self._log_debug("LTM: store completed — proceeding with retrieval")
            except asyncio.TimeoutError:
                self._log_warning(
                    "LTM: store timeout (>3s) — retrieval may miss previous turn"
                )

        # ── Detect AutoContinue continuation ──────────────────────────────
        _last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        _hint = ""
        _is_continuation = False
        if _last_assistant:
            _ac = _last_assistant.get("content", "")
            for _marker in self._MULTI_PHASE_MARKERS:
                if _marker in _ac:
                    _is_continuation = True
                    _idx = _ac.find(_marker)
                    _hint_line = _ac[_idx:].split("\n")[0]
                    _hint = re.sub(
                        r"▶\s*CONTINÚA[:\s]+(?:Parte\s*\d+[/\d]*\s*[—\-]?\s*)?",
                        "",
                        _hint_line,
                        flags=re.IGNORECASE,
                    ).strip()
                    if _hint:
                        user_question = _hint
                        self._log_debug(
                            f"AutoContinue detected — LOD query: '{user_question}'"
                        )
                    break
        if _is_continuation:
            slot_free = False
            user_query = user_query + " código"

        # ─────────────────────────────────────────────────────────────────
        # ⚡ COMMAND HANDLING (High value)
        #   3. Explicit commands (/forget, /status, /clean, /expand)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        handled, handled_messages = await self._commands.handle_explicit_commands(
            messages, project_id, is_explicit_command, last_user_msg, __user__
        )
        _inlet_timing("Step 3/7: Handle explicit commands", step_start)
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ─────────────────────────────────────────────────────────────────
        # ⚡ COMMAND HANDLING (High value)
        #   4. Natural language intents (forget, remember, obsolete)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        handled, handled_messages = await self._commands.handle_natural_intents(
            messages,
            project_id,
            is_explicit_command,
            last_user_msg,
            slot_free=slot_free,
        )
        _inlet_timing("Step 4/7: Handle natural language intents", step_start)
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ── Silent Ingestion (Modo B: chunked paste) ────────────────────
        if (
            self.valves.enable_silent_ingestion
            and last_user_msg is not None
            and not is_explicit_command
        ):
            if await self._commands.is_code_only_message(user_query):
                self._log_section("SILENT INGESTION MODE")

                pstate = self._project_state_manager.get_pstate(project_id)

                _guessed_lang = SignatureExtractor._guess_language(None, user_query)
                _lang = _guessed_lang if _guessed_lang != "unknown" else "python"
                pstate["ingested_lang"] = _lang

                raw_symbols = []
                if HAS_TREE_SITTER:
                    try:
                        raw_symbols = await SignatureExtractor.extract_async(
                            user_query, None, language=_lang
                        )
                    except Exception:
                        raw_symbols = []

                if raw_symbols:
                    raw_symbols = SignatureExtractor.enrich_symbols_with_parent_info(
                        raw_symbols, user_query
                    )

                pstate["raw_ingested_symbols"] = raw_symbols

                _msg_to_index = last_user_msg

                self._is_silent_ingestion = True
                try:
                    await self._update_active_code(_msg_to_index, project_id)
                finally:
                    pass

                await self._activation.resolve_dangling_edges(project_id)

                if self.valves.enable_path_analysis:
                    await self._activation.rebuild_path_index(project_id)

                self._ctx_builder.invalidate_block_a_cache(
                    project_id, "new chunk ingested", recompute_centrality=True
                )

                try:
                    static_block = await self._ctx_builder.build_block_a(
                        project_id, is_code_session=True, is_continuation=False
                    )
                    self._log_debug(
                        "🧱 Block A scaffold (hub symbols + skeleton tier) "
                        "pre-built after silent ingestion"
                    )
                except Exception as _scaffold_err:
                    static_block = ""
                    self._log_debug(
                        f"Eager Block A scaffold build failed (non-fatal): {_scaffold_err}"
                    )

                # ── M7: Pre‑compute tier during silent ingestion ──
                if (
                    self.valves.enable_hub_bodies_tier
                    and self.valves.hub_bodies_tier_warmup_on_ingestion
                ):
                    tier_text, tier_hash, tier_qids = (
                        self._ctx_builder._build_hub_bodies_tier(project_id)
                    )
                    pstate["hub_tier_text"] = tier_text
                    pstate["hub_tier_hash"] = tier_hash
                    pstate["hub_tier_qids"] = tier_qids
                    state.hub_tier_qids_persisted = tier_qids
                    self._conversation_state_manager.set(project_id, state)

                    # ── B4: warmup tier prefill ──────────────────────────────
                    asyncio.create_task(
                        self._ctx_builder._warmup_tier_prefill(project_id)
                    )

                # Mark dirty and save via ConversationStateManager
                self._conversation_state_manager.mark_dirty(project_id)
                await self._conversation_state_manager.save_if_dirty(project_id)

                # Get final counts after indexing
                state = self._conversation_state_manager.get(project_id)
                num_blocks = len(state.active_blocks)
                num_symbols = len(self._symbol_index.get_all_names(project_id))
                num_classes = len(self._symbol_index.get_classes(project_id))

                # ── Generate and store stub (no truncation) ──
                stub = self._history_compressor._build_user_stub(num_symbols)

                # Store stub in state so it persists for future turns
                content_hash = hashlib.md5(user_query.encode()).hexdigest()[:16]
                state.compressed_user_messages[content_hash] = stub
                self._conversation_state_manager.mark_dirty(project_id)

                # Replace the current user message with the stub
                messages[-1] = {**messages[-1], "content": stub}

                # Build the assistant response
                response = (
                    f"✅ {num_symbols} symbols in {num_classes} classes "
                    f"({num_blocks} active blocks). Code is in the SymbolGraph. "
                    f"Use `/expand <Class>` or `/expand <Class>.<method>` to see implementations."
                )
                messages.append({"role": "assistant", "content": response})

                # ── Context dump (if enabled) ────────────────────────────────────
                if self.valves.enable_context_dump:
                    try:
                        self._context_dumper.schedule_inlet_snapshot(
                            project_id=project_id,
                            static_block=static_block,
                            dynamic_block="",
                            final_system=static_block,
                            messages=messages,
                        )
                    except Exception as _dump_err:
                        self._log_debug(
                            f"Context dump scheduling failed (silent ingestion): {_dump_err}"
                        )

                body["messages"] = messages
                _inlet_timing("total_inlet (end-to-end)", inlet_start)
                self._log_section(
                    "CONTEXT MANAGER - INLET END",
                    duration=time.monotonic() - inlet_start,
                )
                return body

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   5. Prepare code session (classify, update code blocks)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        is_code_session, user_question = (
            await self._inlet_orch.inlet_prepare_code_session(
                messages, project_id, user_query, is_continuation=_is_continuation
            )
        )
        _inlet_timing("Step 5/7: Prepare code session", step_start)

        # ─────────────────────────────────────────────────────────────────
        # 🧠 ENRICHMENT (Critical for call‑graph mode resolution)
        #   Resolve call‑graph mode BEFORE Block A is built.
        # ─────────────────────────────────────────────────────────────────
        intent_vector = await self._commands.classify_intent(user_query, project_id)
        use_case_key, use_case_profile, use_case_label = (
            await self._inlet_orch.classify_intent_with_continuation(
                user_query, project_id, intent_vector
            )
        )
        self._ctx_builder.prepare_call_graph_mode(project_id, user_query, intent_vector)

        # ─────────────────────────────────────────────────────────────────
        # 🧠 ENRICHMENT (High value)
        #   6. Build system injections and assemble final messages
        #      (delegates Block A/B construction to ContextBuilder)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        state = self._conversation_state_manager.get(project_id)
        static_block, dynamic_injections, cached_response, prelim_system = (
            await self._inlet_build_system_injections(
                messages,
                project_id,
                user_query,
                user_question,
                is_code_session,
                last_user_msg,
                state,
                slot_free=slot_free,
                intent_vector=intent_vector,
            )
        )
        _inlet_timing("Step 6/7: Build system injections", step_start)

        # ── PREMATURE slot_restore REMOVED (moved to the end) ──

        if cached_response:
            messages.pop()
            messages.append(
                {"role": "assistant", "content": cached_response["response"]}
            )
            messages = self._inlet_orch.ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ─────────────────────────────────────────────────────────────────
        # 📦 COMPRESSION + ASSEMBLY (High value)
        #   7. Assemble final messages with CoT, multi-phase, trimming
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        messages = await self._inlet_assemble_final_messages(
            messages,
            project_id,
            static_block,
            dynamic_injections,
            prelim_system,
            last_user_msg,
            is_code_session,
            state,
            __user__,
            user_question,
            has_code_blocks,
            slot_free=slot_free,
        )
        _inlet_timing("Step 7/7: Assemble final messages", step_start)

        # ✅ active_blocks Validation (attribute, not dict)
        if not isinstance(state.active_blocks, dict):
            state.active_blocks = {}
            self._conversation_state_manager.set(project_id, state)

        body["messages"] = messages

        # ─────────────────────────────────────────────────────────────────
        # 🚀 KV CACHE FIX – Restore stable prefix AFTER all auxiliaries
        # ─────────────────────────────────────────────────────────────────
        # slot_restore_for_continuity is independent of slot_restore:
        #   - slot_restore: session start / first time Block A is built.
        #   - slot_restore_for_continuity: end of each inlet, after CoT,
        #     seed inference and other auxiliaries have dirtied the slot.
        # One restore at the end covers ALL auxiliary calls of this turn.
        # Gated by slot_free (in AutoContinue continuations the slot is already
        # configured for streaming and should not be touched).
        if slot_free and self.valves.enable_slot_persistence:
            await self._project_state_manager.slot_restore_for_continuity(project_id)

        _inlet_timing("total_inlet (end-to-end)", inlet_start)
        self._log_section(
            "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
        )
        return body

    # ═══════════════════════════════════════════════════════════════════════════
    # OUTLET – Post‑response processing
    # ═══════════════════════════════════════════════════════════════════════════
    # Value categories (same as inlet):
    #   🔥 STATE MANAGEMENT    – Update code state, persist LTM, response cache
    #   🚀 RESOURCE OPTIMISATION – Purge expired memories, DB checkpoints, free VRAM
    # ═══════════════════════════════════════════════════════════════════════════
    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Post‑process the response after the LLM has generated it.

        Runs maintenance and persistence tasks that do not block the user:
        * Updates active code blocks and stores the new message in LTM.
        * Intercepts ``/expand`` commands in the assistant's response and
            replaces them with real code from the SymbolIndex.
        * Stores the response in the semantic cache for future reuse.
        * Adjusts LOD thresholds adaptively based on which symbols the LLM
            actually used.
        * Runs speculative prefetch for the next likely query.
        * Purges expired memories, rebuilds RAPTOR clusters periodically,
            and runs SQLite + ChromaDB checkpoints.
        * Persists symbol edges, path views, and dirty conversation state.
        * Saves the KV slot (slot persistence) for future session restores.
        * Waits for background LLM tasks (docstrings, etc.) to finish before
            exiting, to prevent them from blocking the next request.

        Returns the (possibly modified) body unchanged.
        """
        self._log_debug("outlet called")
        start_time = time.monotonic()
        self._log_section("CONTEXT MANAGER - OUTLET START")

        if not (HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness):
            self._log_debug("outlet: prerequisites not met, returning body unchanged")
            return body

        try:
            messages = body.get("messages", [])
            project_id = self._inlet_orch.get_project_id()
            state = self._conversation_state_manager.get(project_id)
            is_code_session = await self._inlet_orch.classify_session(
                messages, project_id
            )
            last_msg = messages[-1] if messages else None

            pstate = self._project_state_manager.get_pstate(project_id)

            if last_msg:
                last_idx = len(messages) - 1

                if last_msg.get("role") == "assistant":
                    self._log_debug(
                        "outlet: processing assistant message for indexing and expansion"
                    )
                    if is_code_session and "/expand" in last_msg.get("content", ""):
                        self._log_debug(
                            "🔥 STATE MANAGEMENT – Intercepting /expand command to inject real code"
                        )
                        modified_content, did_expand = (
                            await self._commands.outlet_intercept_expand(
                                last_msg.get("content", ""), project_id
                            )
                        )
                        if did_expand:
                            messages[-1]["content"] = modified_content
                            body["messages"] = messages
                            self._log_debug(
                                "outlet: /expand intercepted — history rewritten with real code"
                            )

                    await self._llm_orchestrator.wait_for_llm_tasks()
                    if is_code_session:
                        self._log_debug(
                            "🔥 STATE MANAGEMENT – Updating active code blocks and storing in LTM (assistant code detected)"
                        )
                        await self._update_active_code(last_msg, project_id)

                        # ── C6: wrap store_messages with signal ──────────────────────────
                        self._ltm_store_complete.clear()

                        async def _store_and_signal():
                            try:
                                await self._ltm.store_messages(
                                    project_id, [last_msg], wait=False
                                )
                            except Exception as e:
                                self._log_debug(f"LTM: store_messages failed: {e}")
                            finally:
                                self._ltm_store_complete.set()

                        asyncio.create_task(_store_and_signal())

                        if self.valves.enable_auto_docstrings_background:
                            state = self._conversation_state_manager.get(project_id)
                            pending = []
                            for block in state.active_blocks.values():
                                if block.obsolete:
                                    continue
                                for sym in block.symbols:
                                    if (
                                        sym.kind in ("function", "method")
                                        and not sym.docstring
                                    ):
                                        pending.append((sym.name, sym.signature))
                            if pending:
                                self._enrichment.start_docstring_loop(project_id)
                    else:
                        if not self.valves.ltm_store_only_code_sessions:
                            self._log_debug(
                                "🔥 STATE MANAGEMENT – Storing non‑code session message in LTM"
                            )
                            # ── C6: also signal for non‑code stores ──────────────────
                            self._ltm_store_complete.clear()

                            async def _store_and_signal_noncode():
                                try:
                                    await self._ltm.store_messages(
                                        project_id, [last_msg], wait=False
                                    )
                                except Exception as e:
                                    self._log_debug(f"LTM: store_messages failed: {e}")
                                finally:
                                    self._ltm_store_complete.set()

                            asyncio.create_task(_store_and_signal_noncode())

                else:
                    if last_idx <= pstate.get("last_processed_message_idx", -1):
                        self._log_debug(
                            "outlet: last user message already processed in inlet, skipping"
                        )
                    else:
                        await self._llm_orchestrator.wait_for_llm_tasks()
                        if is_code_session:
                            # ── C6: signal for user message in code session ──────────
                            self._ltm_store_complete.clear()

                            async def _store_and_signal_user_code():
                                try:
                                    await self._ltm.store_messages(
                                        project_id, [last_msg], wait=False
                                    )
                                except Exception as e:
                                    self._log_debug(f"LTM: store_messages failed: {e}")
                                finally:
                                    self._ltm_store_complete.set()

                            asyncio.create_task(_store_and_signal_user_code())
                            await self._update_active_code(last_msg, project_id)
                        else:
                            if not self.valves.ltm_store_only_code_sessions:
                                self._ltm_store_complete.clear()

                                async def _store_and_signal_user_noncode():
                                    try:
                                        await self._ltm.store_messages(
                                            project_id, [last_msg], wait=False
                                        )
                                    except Exception as e:
                                        self._log_debug(
                                            f"LTM: store_messages failed: {e}"
                                        )
                                    finally:
                                        self._ltm_store_complete.set()

                                asyncio.create_task(_store_and_signal_user_noncode())

            # ─────────────────────────────────────────────────────────────────────
            # 🔥 Save slot NOW (stable state, before long tasks)
            # ─────────────────────────────────────────────────────────────────────
            if self.valves.enable_slot_persistence:
                try:
                    pstate["last_total_context_tokens"] = self._tokens.estimate_tokens(
                        messages
                    )
                except Exception as e:
                    self._log_debug(f"outlet: token estimation failed: {e}")
                await self._project_state_manager.slot_save(project_id)

            # ── Response cache ─────────────────────────────────────────
            self._log_debug("outlet: before cache store")
            if (
                self.valves.enable_response_cache
                and HAS_SENTENCE
                and len(messages) >= 2
            ):
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Storing response in cache (to avoid recomputation for similar future requests)"
                )
                last_user = next(
                    (m for m in reversed(messages) if m.get("role") == "user"), None
                )
                last_assistant = next(
                    (m for m in reversed(messages) if m.get("role") == "assistant"),
                    None,
                )
                if last_user and last_assistant:
                    _is_partial_mp = self.valves.enable_multi_phase_response and any(
                        marker in last_assistant.get("content", "")
                        for marker in self._MULTI_PHASE_MARKERS
                    )
                    if _is_partial_mp:
                        self._log_debug(
                            "Response cache: skipping storage for partial multi-phase response"
                        )
                    else:
                        context_hash = self._activation.compute_context_hash(
                            messages[:-1]
                        )
                        code_state_hash = self._activation.compute_code_state_hash(
                            project_id
                        )
                        await self._ltm.store_response_in_cache(
                            last_user.get("content", ""),
                            last_assistant.get("content", ""),
                            context_hash,
                            state,
                            code_state_hash,
                            wait=False,
                        )
            self._log_debug("outlet: after cache store")

            # ── LOD adaptive ───────────────────────────────────────────
            self._log_debug("outlet: before LOD adaptive")
            if self.valves.enable_lod_adaptive and is_code_session:
                last_assistant = next(
                    (m for m in reversed(messages) if m.get("role") == "assistant"),
                    None,
                )
                if last_assistant and last_assistant.get("content"):
                    _is_partial_mp_lod = self.valves.enable_multi_phase_response and (
                        any(
                            marker in last_assistant["content"]
                            for marker in self._MULTI_PHASE_MARKERS
                        )
                        or bool(
                            re.search(
                                r"##\s*📋\s*(?:PROTOCOLO|CONTINUACIÓN)\s+MULTI-FASE",
                                last_assistant["content"],
                                re.IGNORECASE,
                            )
                        )
                    )
                    if _is_partial_mp_lod:
                        self._log_debug(
                            "LOD adaptive: skipping feedback for partial multi-phase response"
                        )
                    else:
                        await self._enrichment.update_lod_thresholds_from_response(
                            project_id, last_assistant["content"]
                        )
            self._log_debug("outlet: after LOD adaptive")

            # ── Speculative prefetch ───────────────────────────────────
            self._log_debug("outlet: before speculative prefetch")
            if self.valves.enable_speculative_prefetch and is_code_session:
                last_activated = pstate.get("last_activation_scores", {})
                if last_activated:
                    self._log_debug(
                        f"outlet: speculative prefetch with {len(last_activated)} activated nodes"
                    )
                    await self._activation.speculative_prefetch(
                        project_id, last_activated
                    )
                else:
                    self._log_debug("outlet: no activation scores, skipping prefetch")
            self._log_debug("outlet: after speculative prefetch")

            # ── Purge expired memories ─────────────────────────────────
            self._log_debug("outlet: before purge expired memories")
            await self._ltm.purge_expired_memories()
            self._log_debug("outlet: after purge expired memories")

            if not hasattr(self, "_write_counter"):
                self._write_counter = 0
            self._write_counter += 1
            self._log_debug(f"outlet: write_counter={self._write_counter}")

            interval = self.valves.purge_orphaned_data_interval
            if interval > 0 and self._write_counter % interval == 0:
                await self._state_store.purge_orphaned_data(project_id)

            # ── RAPTOR rebuild ─────────────────────────────────────────
            self._log_debug("outlet: before RAPTOR rebuild")
            if (
                self.valves.enable_raptor
                and self._write_counter % self.valves.raptor_rebuild_interval == 0
                and self.memory_collection is not None
            ):
                self._log_debug("RAPTOR: triggering index rebuild")
                edges_out = self._symbol_index.get_all_edges_out(project_id)
                graph_weight = (
                    self.valves.raptor_graph_weight
                    if self.valves.raptor_use_call_graph_proximity
                    else 0.0
                )
                await self._raptor.rebuild(
                    project_id=project_id,
                    symbol_index=self._symbol_index,
                    edges_out=edges_out,
                    n_clusters=self.valves.raptor_clusters_per_level,
                    summary_model=self.valves.raptor_summary_model,
                    summary_max_tokens=self.valves.raptor_summary_max_tokens,
                    chroma_collection=self.memory_collection,
                    llm_caller=self._llm_orchestrator.call_llm,
                    embedder=self.embedder,
                    graph_weight=graph_weight,
                )
            self._log_debug("outlet: after RAPTOR rebuild")

            # ── DB checkpoints ─────────────────────────────────────────
            self._log_debug("outlet: before DB checkpoints")
            if self._write_counter % 100 == 0:
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Running DB checkpoints (to ensure data durability and prevent WAL buildup)"
                )
                await self._state_store.run_db_checkpoints()
            self._log_debug("outlet: after DB checkpoints")

            # ── Purge old versions ─────────────────────────────────────
            self._log_debug("outlet: before purge old versions")
            if (
                self.valves.purge_old_code_versions_enabled
                and self.valves.enable_block_paging
                and self._pager is not None
            ):
                await self._pager.purge_old_versions(
                    project_id=project_id,
                    state=state,
                    symbol_index=self._symbol_index,
                    chroma_collection=self.memory_collection,
                    embedder=self.embedder,
                    max_versions_per_file=self.valves.purge_old_code_versions_max_per_file,
                )
            self._log_debug("outlet: after purge old versions")

            # ── Save edges ─────────────────────────────────────────────
            self._log_debug("outlet: before save edges")
            if self.valves.enable_edge_persistence:
                await self._state_store.save_symbol_edges_to_db(project_id)
            self._log_debug("outlet: after save edges")

            # ── Save path views ───────────────────────────────────────
            self._log_debug("outlet: before save path views")
            if self.valves.enable_path_analysis:
                await self._state_store.save_path_views_to_db(
                    project_id, self._path_index.get_all(project_id)
                )
            self._log_debug("outlet: after save path views")

            # ── Save state if dirty ───────────────────────────────────
            self._log_debug(
                "🔥 STATE MANAGEMENT – Saving conversation state (to preserve context across restarts)"
            )
            await self._conversation_state_manager.save_if_dirty(project_id)

            self._log_debug(
                "🚀 RESOURCE OPTIMISATION – Skipping model unload to preserve KV cache"
            )

        except Exception as e:
            self._log_debug(f"❌ outlet error: {e}")
            import traceback

            self._log_debug(traceback.format_exc())

        finally:
            if getattr(self, "_is_silent_ingestion", False):
                self._is_silent_ingestion = False

            self._log_debug("outlet: waiting for background LLM tasks to complete")
            await self._llm_orchestrator.wait_for_llm_tasks()
            self._log_debug("outlet: background LLM tasks completed")

        self._log_section(
            "CONTEXT MANAGER - OUTLET END", duration=time.monotonic() - start_time
        )
        return body
