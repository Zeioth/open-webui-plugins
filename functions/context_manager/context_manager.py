"""
title: Code-Aware Context Manager with LTM & Summarization
description: Full-featured context manager for coding assistants — v8.0.0 (Context Scaling).
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 8.0.0
license: GPL3
requirements: loguru, tiktoken, sentence-transformers, chromadb, rapidfuzz, tree-sitter-language-pack>=1.5.0, llmlingua>=0.2.0
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

_llm_lock_path = os.path.join(tempfile.gettempdir(), "openwebui_llm.lock")


# ---------------------------------------------------------------------------
# Models & Enums
# ---------------------------------------------------------------------------
class ContentType(str, Enum):
    BASE_CODE = "base_code"
    PROPOSED_CHANGE = "proposed_change"
    COMMITTED_CHANGE = "committed_change"
    GENERAL = "general"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class CodeSymbol(BaseModel):
    name: str
    kind: str  # function, class, method
    signature: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    parent_block_hash: str = ""
    language: str = "unknown"
    calls: List[str] = Field(default_factory=list)
    summary: str = ""


class CodeBlock(BaseModel):
    content: str
    content_type: ContentType
    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    timestamp: float = Field(default_factory=time.time)
    is_active: bool = True
    hash: str = ""
    importance_score: float = 1.0
    mention_count: int = 1
    last_mentioned: float = Field(default_factory=time.time)
    generated_by_assistant: bool = False
    pinned: bool = False
    obsolete: bool = False
    is_raw: bool = False
    block_summary: str = ""
    symbols: List[CodeSymbol] = Field(default_factory=list)
    _cached_token_count: int = 0
    last_mentioned_msg_idx: Optional[int] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
        self._update_importance()

    def _update_importance(self):
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


# ---------------------------------------------------------------------------
# Edge types and base weights
# ---------------------------------------------------------------------------
EDGE_WEIGHTS: Dict[str, float] = {
    "calls": 1.0,
    "imports": 0.6,
    "reads": 0.7,
    "writes": 0.9,
    "inherits": 0.5,
    "references": 0.4,
    "data_flow": 0.8,
}


class Edge(BaseModel):
    src: str
    dst: str
    type: str
    weight: float = 1.0
    confidence: float = 1.0

    def effective_weight(self) -> float:
        return self.weight * self.confidence


# ---------------------------------------------------------------------------
# Activation Graph — query‑conditioned node activation
# ---------------------------------------------------------------------------
class ActivationState(BaseModel):
    node_id: str
    score: float
    depth: int
    source: str


class ActivationGraph:
    DECAY_BASE: float = 0.7

    def __init__(self):
        self._activations: Dict[str, ActivationState] = {}

    def seed(self, node_ids: List[str], initial_score: float = 1.0):
        for nid in node_ids:
            self._activations[nid] = ActivationState(
                node_id=nid,
                score=initial_score,
                depth=0,
                source="seed",
            )

    def propagate(
        self,
        edges_out: Dict[str, List[Edge]],
        max_steps: int = 20,
        min_score: float = 0.05,
        alpha: float = 0.85,
        tolerance: float = 1e-6,
    ):
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
        for node_id, score in r.items():
            if score < min_score:
                continue
            existing = self._activations.get(node_id)
            self._activations[node_id] = ActivationState(
                node_id=node_id,
                score=score,
                depth=existing.depth if existing else 99,
                source=existing.source if existing else "propagation",
            )

    def get_score(self, node_id: str) -> float:
        state = self._activations.get(node_id)
        return state.score if state else 0.0

    def get_activated_nodes(self, threshold: float = 0.1) -> Dict[str, float]:
        return {
            nid: s.score for nid, s in self._activations.items() if s.score >= threshold
        }

    def aggregate_path_score(self, symbol_list: List[str]) -> float:
        scores = [self.get_score(s) for s in symbol_list]
        active = [s for s in scores if s > 0]
        if not active:
            return 0.0
        return sum(active) / len(active)


# ---------------------------------------------------------------------------
# Query model and SubgraphExtractor skeleton
# ---------------------------------------------------------------------------
class SubgraphExtractor:
    def __init__(self, activation_threshold: float = 0.1, expand_hops: int = 1):
        self.activation_threshold = activation_threshold
        self.expand_hops = expand_hops

    def extract(
        self,
        activation: ActivationGraph,
        edges_out: Dict[str, List[Edge]],
        edges_in: Dict[str, List[Edge]],
    ) -> Tuple[Set[str], List[Edge]]:
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
    path_id: str
    entry_point: str
    seed_nodes: List[str]
    induced_nodes: Dict[str, float]
    induced_edges: List[Edge]
    activation_score: float
    business_label: str = ""
    summary: str = ""
    label_confidence: float = 0.0
    structural_hash: str = ""
    call_graph_hash: str = ""
    last_built: float = Field(default_factory=time.time)

    def is_stale(self, current_structural: str, current_call_graph: str) -> bool:
        return (
            self.structural_hash != current_structural
            or self.call_graph_hash != current_call_graph
        )

    def top_symbols(self, n: int = 10) -> List[str]:
        return sorted(
            self.induced_nodes.keys(),
            key=lambda s: self.induced_nodes[s],
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# StaticEvidence – deterministic proof from the SymbolGraph
# ---------------------------------------------------------------------------
class StaticEvidence(BaseModel):
    symbols_found: Dict[str, bool]
    call_relations_valid: Dict[str, bool]
    recent_changes: List[str]
    entry_points_mentioned: List[str]
    path_memberships: Dict[str, List[str]]
    data_flow_upstream: Dict[str, List[str]] = Field(default_factory=dict)
    objective_score: float


# ---------------------------------------------------------------------------
# PathIndex — index of CodePathViews
# ---------------------------------------------------------------------------
class PathIndex:
    def __init__(self):
        self._views: Dict[str, CodePathView] = {}
        self._symbol_to_views: Dict[str, Set[str]] = defaultdict(set)

    def add(self, view: CodePathView, project_id: str):
        key = f"{project_id}:{view.path_id}"
        self._views[key] = view
        for sym_name in view.induced_nodes:
            self._symbol_to_views[f"{project_id}:{sym_name}"].add(view.path_id)

    def remove(self, path_id: str, project_id: str):
        key = f"{project_id}:{path_id}"
        view = self._views.pop(key, None)
        if view:
            for sym_name in view.induced_nodes:
                sym_key = f"{project_id}:{sym_name}"
                self._symbol_to_views[sym_key].discard(path_id)

    def get(self, path_id: str, project_id: str) -> Optional[CodePathView]:
        return self._views.get(f"{project_id}:{path_id}")

    def get_all(self, project_id: str) -> List[CodePathView]:
        prefix = f"{project_id}:"
        return [v for k, v in self._views.items() if k.startswith(prefix)]

    def clear_project(self, project_id: str):
        prefix = f"{project_id}:"
        keys = [k for k in self._views if k.startswith(prefix)]
        for k in keys:
            del self._views[k]
        sym_keys = [k for k in self._symbol_to_views if k.startswith(prefix)]
        for k in sym_keys:
            del self._symbol_to_views[k]

    def mark_stale_for_symbol(self, symbol_name: str, project_id: str) -> List[str]:
        key = f"{project_id}:{symbol_name}"
        return list(self._symbol_to_views.get(key, set()))

    def find_entry_points(
        self, symbol_index: "SymbolIndex", project_id: str
    ) -> Set[str]:
        all_names = symbol_index.get_all_names(project_id)
        return {
            name for name in all_names if not symbol_index.get_callers(name, project_id)
        }


# ---------------------------------------------------------------------------
# AppliedChangeFeedback
# ---------------------------------------------------------------------------
class AppliedChangeFeedback(BaseModel):
    change_hash: str
    change_description: str
    file_path: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


# ---------------------------------------------------------------------------
# Tree‑sitter fallback queries (se mantienen igual que en v7)
# ---------------------------------------------------------------------------
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
# Reranker singleton factory (module level)
# ---------------------------------------------------------------------------
_CROSS_ENCODER = None
_CROSS_ENCODER_LOCK = threading.Lock()


def _get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
    Produce the hub-symbol text for Block A using top-N symbols by centrality.
    """

    # ── Public API ────────────────────────────────────────────────────────

    def get_hub_names(
        self,
        centrality: dict,
        top_n: int,
    ) -> list:
        """
        Return symbol names sorted by descending centrality, capped at top_n.

        centrality: {symbol_name: score in [0, 1]} from
                    SymbolIndex.precompute_centrality().
        Ties are broken alphabetically for deterministic, cache-stable output.
        """
        if not centrality or top_n <= 0:
            return []
        ranked = sorted(
            centrality.items(),
            key=lambda kv: (-kv[1], kv[0]),  # score desc, then name asc
        )
        return [name for name, _ in ranked[:top_n]]

    def is_hub(
        self,
        symbol_name: str,
        centrality: dict,
        top_n: int,
    ) -> bool:
        """True if symbol_name would appear in Block A for the given top_n."""
        return symbol_name in set(self.get_hub_names(centrality, top_n))

    def build(
        self,
        symbol_index: "SymbolIndex",
        centrality: dict,
        project_id: str,
        top_n: int = 30,
    ) -> str:
        """
        Build the Block A hub-symbol text (always <= top_n entries).

        Output is deterministic and stable between requests while the code
        state (and therefore centrality) is unchanged — this is what lets
        llama.cpp reuse the KV cache for Block A.

        Returns "" if there is no centrality data (no indexed code yet).
        """
        hub_names = self.get_hub_names(centrality, top_n)
        if not hub_names:
            return ""

        # Group by file (using index's file resolution, if available)
        by_file: dict = {}
        for name in hub_names:
            file_path = self._file_for(name, project_id, symbol_index)
            by_file.setdefault(file_path, []).append(name)

        lines = [
            f"## Code Symbol Index — Hub Symbols (top {len(hub_names)} by call-graph centrality)",
            "> Remaining symbols are available via LOD activation. "
            "Use /expand <name> for any symbol's full body.",
            "",
        ]

        # If file info is missing for all symbols, use flat list; otherwise group by file.
        if len(by_file) == 1 and None in by_file:
            for name in sorted(hub_names, key=lambda n: -centrality.get(n, 0)):
                lines.append(
                    self._format_symbol_line(name, centrality, symbol_index, project_id)
                )
        else:
            for file_path in sorted(by_file.keys(), key=lambda fp: (fp is None, fp)):
                if file_path is None:
                    continue
                lines.append(f"### {file_path}")
                for name in sorted(
                    by_file[file_path], key=lambda n: -centrality.get(n, 0)
                ):
                    lines.append(
                        self._format_symbol_line(
                            name, centrality, symbol_index, project_id
                        )
                    )
                lines.append("")

        lines.append(
            "To see any symbol's full body, mention it in your message "
            "or use /expand <name>."
        )
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────

    def _file_for(
        self, name: str, project_id: str, symbol_index: "SymbolIndex"
    ) -> Optional[str]:
        """
        Resolve a symbol's file path from the SymbolIndex, if possible.

        Tries public methods first (get_file_for_symbol, get_symbol_file,
        file_for), then internal attributes (_file_by_symbol, _symbol_files).
        Returns None when the index exposes no file info at all, which makes
        build() emit a flat centrality-ranked list instead of a useless
        '### unknown' section.
        """
        # Try public methods first
        for attr in ("get_file_for_symbol", "get_symbol_file", "file_for"):
            fn = getattr(symbol_index, attr, None)
            if callable(fn):
                try:
                    res = fn(name, project_id)
                    if res:
                        return res
                except Exception:
                    pass
        # Try internal attributes
        internal = getattr(symbol_index, "_file_by_symbol", None) or getattr(
            symbol_index, "_symbol_files", None
        )
        if isinstance(internal, dict):
            key = f"{project_id}:{name}"
            if key in internal:
                return internal[key]
            if name in internal:
                return internal[name]
        return None

    def _safe_callers(
        self, name: str, project_id: str, symbol_index: "SymbolIndex"
    ) -> set:
        """
        Return set of caller names for `name`, or empty set on any error.

        Tries `get_callers(name, project_id)`, then `get_callers(name)` as a
        fallback for index implementations that omit the project argument.
        """
        fn = getattr(symbol_index, "get_callers", None)
        if not callable(fn):
            return set()
        try:
            res = fn(name, project_id)
            if isinstance(res, (set, list)):
                return set(res)
        except Exception:
            pass
        try:
            res = fn(name)  # fallback without project_id
            if isinstance(res, (set, list)):
                return set(res)
        except Exception:
            pass
        return set()

    def _format_symbol_line(
        self,
        name: str,
        centrality: dict,
        symbol_index: "SymbolIndex",
        project_id: str,
    ) -> str:
        """
        Render one hub symbol line using data available from the SymbolIndex.

        Format:
          - `name` (centrality: 0.87)  ← used by: x, y
        Callers come from the index; callee data requires the symbol object
        which Block A does not carry, so the "→ calls" segment is omitted
        rather than fabricated.
        """
        score = centrality.get(name, 0.0)
        callers = self._safe_callers(name, project_id, symbol_index)

        parts = [f"- `{name}` (centrality: {score:.2f})"]

        if callers:
            shown = sorted(callers)[:5]
            extra = f", ... (+{len(callers) - 5} more)" if len(callers) > 5 else ""
            parts.append(f"  ← used by: {', '.join(shown)}{extra}")

        return "".join(parts)


class ContextPager:
    """
    Manages CodeBlock lifecycle between active_blocks (RAM) and ChromaDB (paged).
    """

    def __init__(self) -> None:
        # project_id → set of block hashes currently paged out.
        self._paged_hashes: dict = {}

    # ── Eviction candidate selection ──────────────────────────────────────

    def get_eviction_candidates(
        self,
        state: dict,
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
        """
        active = state.get("active_blocks", {})
        if len(active) <= paging_threshold:
            return []

        candidates = []
        for h, block in active.items():
            if block.pinned or block.obsolete:
                continue
            if block.symbols:
                block_activation = max(
                    (activation_scores.get(s.name, 0.0) for s in block.symbols),
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

    def is_paged(self, block_hash: str, project_id: str) -> bool:
        """True if block_hash has been paged out to ChromaDB for this project."""
        return block_hash in self._paged_hashes.get(project_id, set())

    # ── Page out (active_blocks → ChromaDB) ───────────────────────────────

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
        Soft-evict `block` to ChromaDB.

        The caller is responsible for removing the block from active_blocks
        AFTER this returns True. Symbols remain in the SymbolIndex (the block
        is still indexed and searchable, just not resident in RAM). The full
        body stays in the SQLite code_contents table; we do not duplicate it.

        Returns True if the ChromaDB write succeeded, False otherwise (in which
        case the caller must keep the block in active_blocks).
        """
        if chroma_collection is None or embedder is None:
            return False

        entry_id = f"{project_id}_paged_{block.hash}"
        excerpt = block.content[:500]
        symbol_names = ",".join(s.name for s in block.symbols)

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

        try:
            embedding = await anyio.to_thread.run_sync(
                lambda: embedder.encode(block.content[:1000]).tolist()
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
            return False

        self._paged_hashes.setdefault(project_id, set()).add(block.hash)
        return True

    # ── Page in (ChromaDB → temporary CodeBlock) ──────────────────────────

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

        # Recover the full body from code_contents (authoritative).
        content = ""
        if db_conn is not None:
            try:
                row = await anyio.to_thread.run_sync(
                    lambda: db_conn.execute(
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

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear_project(self, project_id: str) -> None:
        """Drop the in-memory paged registry for a project (on project switch)."""
        self._paged_hashes.pop(project_id, None)


class RaptorCodeIndex:
    """
    Hierarchical clustering of code symbols (RAPTOR adapted for code).
    """

    _N_LANDMARKS: int = 8

    # ── Public API ────────────────────────────────────────────────────────

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

        level == 1: cluster raw symbols.
        level >= 2: cluster the previous level's summaries (read back from the
                    store); graph features are not used at L2 (summaries have no
                    direct call edges), so the augmented vector degrades to the
                    plain semantic embedding.

        Returns the number of clusters actually created.
        """
        import numpy as np

        # ── Gather items + embeddings for this level ──────────────────────
        if level == 1:
            names = list(symbol_index.get_all_names(project_id))
            texts = []
            for n in names:
                sig = self._safe(
                    getattr(symbol_index, "get_signature", None),
                    n,
                    project_id,
                    default=n,
                )
                summ = self._safe(
                    getattr(symbol_index, "get_summary", None),
                    n,
                    project_id,
                    default="",
                )
                texts.append(f"{sig} — {summ}".strip(" —"))
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

    # ── Distance helpers ──────────────────────────────────────────────────

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

    # ── Graph feature construction ────────────────────────────────────────

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

    # ── Summary generation + storage ──────────────────────────────────────

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

    # ── Misc helpers ──────────────────────────────────────────────────────

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
    Builds Block A + Block B, owns KV slot lifecycle.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        """
        filter_ref: the parent Filter instance. Used to reach valves, state,
        symbol_index, path_index, tokenizer, embedder, memory_collection,
        the LLM caller, the hub index, the pager, and the RAPTOR index.
        This is a deliberate back-reference: ContextBuilder wraps Filter data
        rather than duplicating it.
        """
        self._f = filter_ref
        # Per-project Block A cache: project_id → (static_hash, rendered_text).
        # Invalidated by invalidate_block_a_cache() when symbols/feedback change.
        self._block_a_cache: dict = {}

    # ── Block construction ────────────────────────────────────────────────

    async def build_block_a(
        self,
        project_id: str,
        is_code_session: bool,
        is_continuation: bool,
    ) -> str:
        """
        Build Block A: stable, KV-cache-anchoring content.

        MIGRATION: body is copied verbatim from Filter._get_static_context_block()
        with exactly ONE change — the symbol-index section is produced by the
        HubSymbolIndex (top-N by centrality) instead of _build_lightweight_context()
        (all symbols). Everything else (confidence prompt, checklist, feedback
        context, the static_hash cache key, the cache read/write) is identical.

        Returns "" when not a code session.
        """
        if not is_code_session:
            return ""

        current_code_hash = self._f._activation.compute_code_state_hash(project_id)
        cached = self._block_a_cache.get(project_id)

        if cached:
            cached_hash, cached_text = cached
            if cached_hash == current_code_hash:
                return cached_text  # ✓ Hit: same code → same block
            # ── Continuation: freeze Block A to prevent KV cache misses ──
            if is_continuation:
                self._f._log_debug(
                    "🧱 Block A: frozen for AutoContinue (KV cache stability)"
                )
                return cached_text

        # ── Build the static block ──────────────────────────────────────
        parts: List[str] = []

        # 1. Base instructions (completely static)
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

        # 2. Symbol index (hub symbols only — stable while code unchanged)
        if is_code_session and self._f.valves.enable_code_awareness:
            state = self._f._state_store.get_state(project_id)
            if state and state.get("active_blocks"):
                centrality = self._f._node_centrality.get(project_id, {})
                symbol_section = self._f._hub_index.build(
                    symbol_index=self._f._symbol_index,
                    centrality=centrality,
                    project_id=project_id,
                    top_n=self._f.valves.symbol_index_max_in_block_a,
                )
                if symbol_section:
                    parts.append(symbol_section)

        # 3. Feedback context (stable between requests barring new feedback)
        if (
            is_code_session
            and self._f.valves.enable_feedback_tracking
            and self._f.valves.inject_feedback_context
        ):
            feedback_ctx = self._f._enrichment.get_feedback_context(project_id)
            if feedback_ctx:
                parts.append(feedback_ctx)

        static_block = "\n\n".join(p for p in parts if p.strip())

        # ── Cache and track ─────────────────────────────────────────────
        self._block_a_cache[project_id] = (current_code_hash, static_block)

        # Detect and log prefix changes (= cache miss in llama.cpp)
        new_prefix_hash = hashlib.md5(static_block.encode()).hexdigest()[:16]
        last_hash = self._f._last_static_prefix_hash.get(project_id)
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
        self._f._last_static_prefix_hash[project_id] = new_prefix_hash

        tokens = (
            len(self._f.tokenizer.encode(static_block))
            if self._f.tokenizer
            else len(static_block) // 4
        )
        self._f._log_debug(f"Static Context Block: ~{tokens} tokens")

        return static_block

    def invalidate_block_a_cache(self, project_id: str, reason: str = "") -> None:
        """
        Force Block A rebuild on the next request and refresh centrality scores.

        MIGRATION: from Filter._invalidate_static_context_block(), plus one
        addition — recompute centrality so HubSymbolIndex sees fresh scores.
        """
        self._block_a_cache.pop(project_id, None)
        # Refresh centrality so the next build_block_a() ranks hubs correctly.
        try:
            self._f._node_centrality[project_id] = (
                self._f._symbol_index.precompute_centrality(project_id)
            )
        except Exception:
            pass
        if reason:
            self._f._log_debug(f"Block A cache invalidated ({reason})")

    def get_effective_context_budget(self, project_id: str) -> int:
        """
        Tokens available for history + user message after Block A + Block B.

        Uses the token counts recorded for the last request
        (self._f._last_system_tokens[project_id]).
        """
        window = self._f.valves.context_window_tokens
        used = getattr(self._f, "_last_system_tokens", {}).get(project_id, 0)
        reserve = self._f.valves.response_reserve_tokens
        return max(0, window - used - reserve)

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

        MIGRATION: body is copied from Filter._get_path_context(), but the final
        assembly is reorganised into four LOD tiers and reordered so LOD-3 sits
        LAST (closest to the user message). Two NEW injections are added:
          - RAPTOR cluster summaries (into the LOD-2 tier)
          - Page‑in support for evicted LOD-3 blocks
        The persisted conversation-summary read path is handled by the Filter
        injection list, not here (see Step 5.5).

        Returns "" if there is nothing to inject.
        """
        if not self._f.valves.enable_path_analysis:
            active_ctx = self._f._activation.get_active_code_context(project_id, query)
            return active_ctx if active_ctx else ""

        state = self._f._state_store.get_state(project_id)
        if not state or not state.get("active_blocks"):
            return ""

        # ── Fix 2: Fast path for inventory / listing queries ─────────────
        if self._LIST_INTENTS.search(query):
            all_names = self._f._symbol_index.get_all_names(project_id)
            if all_names:
                return await self._format_full_symbol_inventory(all_names, project_id)

        # ── Step 1: ActivationGraph ──────────────────────────────────────
        ag = self._f._activation.build_activation_graph(
            query, project_id, messages=messages
        )
        activated = ag.get_activated_nodes(
            threshold=self._f.valves.path_activation_threshold
        )
        if not activated:
            self._f._log_debug(
                "build_block_b: no activated nodes, falling back to full context"
            )
            return self._f._activation.get_active_code_context(project_id, query)

        # ── Step 2: Adjust LOD thresholds by intent ───────────────────────
        debug_weight = intent_vector.get("debug", 0.2)
        modify_weight = intent_vector.get("modify", 0.3)
        refactor_weight = intent_vector.get("refactor", 0.1)

        lod3 = self._f.valves.lod3_threshold
        lod2 = self._f.valves.lod2_threshold
        lod1 = self._f.valves.lod1_threshold

        if debug_weight + modify_weight > 0.6:
            scale = 0.7
        elif refactor_weight > 0.4:
            scale = 0.0
        else:
            scale = 1.0

        lod3 *= scale
        lod2 *= scale
        lod1 *= scale

        # ── Step 3: Build LOD tiers as separate accumulators ──────────────
        total_tokens = 0
        budget = self._f.valves.active_context_max_tokens or 32000

        # Auto-budget for multi-phase
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

        injected_blocks: Set[str] = set()

        sorted_nodes = sorted(activated.items(), key=lambda x: x[1], reverse=True)

        # Centrality LOD bump
        if self._f.valves.enable_centrality_lod_bump:
            centrality = self._f._node_centrality.get(project_id, {})
            threshold = self._f.valves.centrality_lod_bump_threshold
            adjusted = []
            for node_id, score in sorted_nodes:
                cent = centrality.get(node_id, 0.0)
                if cent >= threshold:
                    effective = min(
                        1.0, score + cent * self._f.valves.centrality_lod_bump_weight
                    )
                else:
                    effective = score
                adjusted.append((node_id, effective))
            sorted_nodes = adjusted

        _lod0_parts: List[str] = []
        _lod1_parts: List[str] = []
        _lod2_parts: List[str] = []
        _lod3_parts: List[str] = []

        for node_id, score in sorted_nodes:
            if total_tokens >= budget:
                break

            if score < lod1:
                _lod0_parts.append(f"`{node_id}`")
                total_tokens += 2
                continue

            block_hashes = self._f._symbol_index.find_blocks(node_id, project_id)
            for bh in block_hashes:
                if bh in injected_blocks:
                    continue
                block = state["active_blocks"].get(bh)

                # ── Page-in support for evicted blocks (NEW) ──────────
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

                if score < lod2:
                    # LOD‑1: signature only
                    sig = next(
                        (sym.signature for sym in block.symbols if sym.name == node_id),
                        node_id,
                    )
                    tok = len(sig) // 4 + 2
                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    _lod1_parts.append(f"- `{sig}`{loc} _(score: {score:.2f})_")
                    total_tokens += tok
                    injected_blocks.add(bh)

                elif score < lod3:
                    # LOD‑2: signature + summary
                    sig = next(
                        (sym.signature for sym in block.symbols if sym.name == node_id),
                        node_id,
                    )
                    summary = next(
                        (
                            sym.summary
                            for sym in block.symbols
                            if sym.name == node_id and sym.summary
                        ),
                        "",
                    )
                    text = f"- `{sig}`: {summary}" if summary else f"- `{sig}`"
                    tok = len(text) // 4 + 2
                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    _lod2_parts.append(f"{text}{loc} _(score: {score:.2f})_")
                    total_tokens += tok
                    injected_blocks.add(bh)

                else:
                    # LOD‑3: full code (with optional compression)
                    content_to_inject = block.content
                    tok = block._cached_token_count or (len(block.content) // 4)

                    # Overflow action at injection time
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
                        and block.block_summary
                    ):
                        content_to_inject = (
                            f"[Summary of {tok}-token block]\n{block.block_summary}"
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
                                block.content,
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
                        f"### `{node_id}`{loc} [activation: {score:.2f}]\n"
                        f"```\n{content_to_inject}\n```\n"
                    )
                    total_tokens += tok
                    injected_blocks.add(bh)

                break  # only one block per symbol at the highest LOD level

        # ── RAPTOR cluster summaries (NEW) → LOD-2 tier ───────────────────
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
        ordered = []
        ordered.append("## Code Context (activation-based LOD)\n")
        if _lod0_parts:
            ordered.append(
                "**Known symbols** (minimal activation):\n" + ", ".join(_lod0_parts)
            )
        if _lod1_parts:
            ordered.append(
                "\n**Signatures** (low activation):\n" + "\n".join(_lod1_parts)
            )
        if _lod2_parts:
            ordered.append(
                "\n**Signatures + summaries** (medium activation):\n"
                + "\n".join(_lod2_parts)
            )
        if _lod3_parts:
            ordered.append(
                "\n### Directly relevant code (high activation)\n"
                + "\n".join(_lod3_parts)
            )

        if len(ordered) <= 1:
            self._f._log_debug(
                "build_block_b: empty output (un-filled ContextBuilder?) — FIX #11"
            )
            return ""

        summary_line = (
            f"\n_(Context: {len(injected_blocks)} symbols, "
            f"~{total_tokens} tokens, "
            f"{len(activated)} nodes activated)_\n"
        )
        ordered.append(summary_line)

        # ── LOD tracking for adaptive feedback ────────────────────────────
        if self._f.valves.enable_lod_adaptive:
            if not hasattr(self._f, "_last_lod_levels"):
                self._f._last_lod_levels: Dict[str, Dict[str, int]] = {}
            lod_map: Dict[str, int] = {}
            for node_id, score in activated.items():
                if score < lod1:
                    lod_map[node_id] = 0
                elif score < lod2:
                    lod_map[node_id] = 1
                elif score < lod3:
                    lod_map[node_id] = 2
                else:
                    lod_map[node_id] = 3
            self._f._last_lod_levels[project_id] = lod_map

        return "\n".join(ordered)

    # ── Fix 2: Inventory listing helper ──────────────────────────────────

    async def _format_full_symbol_inventory(
        self, all_names: set, project_id: str
    ) -> str:
        """Return a formatted inventory of all indexed symbols, grouped by file."""
        state = self._f._state_store.get_state(project_id)
        if not state or not state.get("active_blocks"):
            return ""

        by_file: dict = {}
        for name in sorted(all_names):
            block_hashes = self._f._symbol_index.find_blocks(name, project_id)
            for bh in block_hashes:
                block = state["active_blocks"].get(bh)

                # Page-in fallback: si el bloque fue eviccionado, recuperarlo
                if block is None and self._f._pager is not None:
                    if self._f._pager.is_paged(bh, project_id):
                        block = await self._f._pager.page_in_block(
                            block_hash=bh,
                            project_id=project_id,
                            chroma_collection=self._f.memory_collection,
                            db_conn=self._f._db_conn,
                        )

                if block and not block.obsolete:
                    file_key = block.file_path or "(unknown)"
                    by_file.setdefault(file_key, []).append((name, block))
                    break

        if not by_file:
            return ""

        lines = ["## Full Symbol Inventory\n"]
        total_tokens = self._f._tokens.estimate_code_tokens(lines[0])

        # Presupuesto dinámico igual que en Fix 4
        effective_budget = max(
            4000,
            self._f.valves.context_window_tokens
            - self._f._last_system_tokens.get(project_id, 0)
            - self._f.valves.response_reserve_tokens,
        )
        budget = min(effective_budget // 2, 16000)

        for file_path in sorted(by_file.keys()):
            lines.append(f"### {file_path}")
            for name, block in by_file[file_path]:
                sym = next((s for s in block.symbols if s.name == name), None)
                sig = sym.signature if sym else name
                summary = f" — {sym.summary}" if (sym and sym.summary) else ""
                line = f"- `{sig}`{summary}"
                tok = self._f._tokens.estimate_code_tokens(line)
                if total_tokens + tok > budget:
                    lines.append(
                        f"\n_(Truncated at {budget} tokens — {len(all_names)} symbols total)_"
                    )
                    return "\n".join(lines)
                lines.append(line)
                total_tokens += tok
            lines.append("")

        lines.append(
            f"\n_{len(all_names)} symbols indexed. Use `/expand <name>` for full body._"
        )
        return "\n".join(lines)

    def _build_swa_ordered_lod_parts(
        self, lod0: list, lod1: list, lod2: list, lod3: list
    ) -> str:
        """
        Helper: join LOD tier lists in SWA order (LOD-3 last).
        Each argument is a list of already-rendered section strings.
        """
        sections = []
        for tier in (lod0, lod1, lod2, lod3):
            if tier:
                sections.append("\n".join(tier))
        return "\n".join(s for s in sections if s.strip())

    # ── KV slot lifecycle ─────────────────────────────────────────────────

    async def slot_save(self, project_id: str) -> bool:
        """
        Save the KV slot after a turn.

        MIGRATE-VERBATIM: Filter._slot_save_if_needed().
        """
        if not self._f.valves.enable_slot_persistence:
            return False

        cached = self._block_a_cache.get(project_id)
        if not cached:
            return False
        _, static_text = cached
        static_hash = hashlib.md5(static_text.encode()).hexdigest()[:16]

        if self._f._last_saved_slot_hash.get(project_id) == static_hash:
            return False

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
                    self._f._last_saved_slot_hash[project_id] = static_hash
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

        MIGRATE-VERBATIM: Filter._slot_restore_if_available().
        """
        if not self._f.valves.enable_slot_persistence:
            return False
        if self._f._slot_restore_attempted.get(project_id):
            return self._f._slot_restored.get(project_id, False)

        self._f._slot_restore_attempted[project_id] = True

        cached = self._block_a_cache.get(project_id)
        if not cached:
            return False
        _, static_text = cached
        static_hash = hashlib.md5(static_text.encode()).hexdigest()[:16]
        filename = self._slot_filename(project_id, static_hash)

        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        if not os.path.exists(os.path.join(slot_dir, filename)):
            self._f._log_debug(f"Slot restore: no file found for {filename}")
            return False

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
                    self._f._slot_restored[project_id] = True
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

        MIGRATE-VERBATIM: Filter._slot_restore_for_continuity().
        """
        if not self._f.valves.enable_slot_persistence:
            return False

        cached = self._block_a_cache.get(project_id)
        if not cached:
            return False
        _, static_text = cached
        static_hash = hashlib.md5(static_text.encode()).hexdigest()[:16]
        filename = self._slot_filename(project_id, static_hash)

        slot_dir = self._f.valves.slot_save_path.rstrip("/")
        if not os.path.exists(os.path.join(slot_dir, filename)):
            return False

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
                    data = await resp.json()
                    self._f._log_debug(
                        f"✓ KV cache restored post-aux ← {filename} "
                        f"({data.get('n_restored', '?')} tokens)"
                    )
                    return True
                body = await resp.text()
                self._f._log_debug(
                    f"KV cache continuity restore failed: HTTP {resp.status} — {body}"
                )
                return False
        except Exception as e:
            self._f._log_debug(f"KV cache continuity restore error: {e}")
            return False

    def _slot_filename(self, project_id: str, static_hash: str) -> str:
        """
        Deterministic slot file name.
        Encodes: project + static block hash + model hash.
        If any of the three changes → different name → no stale restore.

        MIGRATE-VERBATIM: Filter._slot_filename().
        """
        model_hash = hashlib.md5(self._f.valves.llm_model.encode()).hexdigest()[:8]
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        return f"slot{self._f.valves.slot_id}_{project_slug}_{static_hash}_{model_hash}.bin"

    async def _cleanup_old_slot_files(self, project_id: str, keep: str) -> None:
        """
        Delete stale slot files, keeping only the current one.

        MIGRATE-VERBATIM: Filter._cleanup_old_slot_files().
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


# ---------------------------------------------------------------------------
# Utility – Reentrant async lock
# ---------------------------------------------------------------------------
class ReentrantAsyncLock:
    """Reentrant asyncio lock with optional timeout to prevent deadlocks."""

    def __init__(self, default_timeout: float = 60.0) -> None:
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count = 0
        self._default_timeout = default_timeout

    async def acquire(self, timeout: Optional[float] = None) -> None:
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
        task = asyncio.current_task()
        if self._owner is not task:
            raise RuntimeError("Lock not owned by current task")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        self.release()


# ---------------------------------------------------------------------------
# SymbolIndex – central name→block mapping and typed edges
# ---------------------------------------------------------------------------
class SymbolIndex:
    """Maps symbol names to block hashes, tracks call edges, and computes centrality."""

    MAX_ENTRIES = 10_000

    def __init__(self) -> None:
        self._name_to_blocks: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._callee_to_callers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._stats: Counter = Counter()
        # Typed edges (v7+)
        self._edges_out: Dict[str, List[Edge]] = defaultdict(list)
        self._edges_in: Dict[str, List[Edge]] = defaultdict(list)
        # Centrality cache (v8)
        self._centrality_cache: Dict[str, Dict[str, float]] = {}

    # ── Name ↔ block hash mapping ─────────────────────────────────────
    def add(self, symbol: "CodeSymbol", block_hash: str, project_id: str) -> None:
        key = (project_id, symbol.name)
        self._name_to_blocks[key].add(block_hash)
        self._stats[key] += 1
        for callee in symbol.calls:
            callee_key = (project_id, callee)
            self._callee_to_callers[callee_key].add(symbol.name)
        self._evict_if_needed()

    def remove(self, symbol: "CodeSymbol", block_hash: str, project_id: str) -> None:
        key = (project_id, symbol.name)
        s = self._name_to_blocks.get(key)
        if s:
            s.discard(block_hash)
            if not s:
                del self._name_to_blocks[key]
                del self._stats[key]
        for callee in symbol.calls:
            callee_key = (project_id, callee)
            if callee_key in self._callee_to_callers:
                self._callee_to_callers[callee_key].discard(symbol.name)
                if not self._callee_to_callers[callee_key]:
                    del self._callee_to_callers[callee_key]

    def remove_all_for_block(
        self, block_hash: str, symbols: List["CodeSymbol"], project_id: str
    ) -> None:
        for sym in symbols:
            self.remove(sym, block_hash, project_id)
            self.remove_edges_for_symbol(sym.name, project_id)

    def find_blocks(self, name: str, project_id: str) -> Set[str]:
        return self._name_to_blocks.get((project_id, name), set())

    def get_all_names(self, project_id: str) -> Set[str]:
        return {key[1] for key in self._name_to_blocks if key[0] == project_id}

    # ── Call relationships (legacy) ────────────────────────────────────
    def get_callers(self, callee_name: str, project_id: str) -> Set[str]:
        return self._callee_to_callers.get((project_id, callee_name), set())

    # ── Typed edge storage (v7+) ───────────────────────────────────────
    def add_edge(self, edge: "Edge", project_id: str) -> None:
        """Register a typed edge in the index. Deduplicates by (src, dst, type)."""
        src_key = f"{project_id}:{edge.src}"
        dst_key = f"{project_id}:{edge.dst}"
        existing = self._edges_out.get(src_key, [])
        for e in existing:
            if e.dst == edge.dst and e.type == edge.type:
                return  # already registered
        self._edges_out[src_key].append(edge)
        self._edges_in[dst_key].append(edge)

    def remove_edges_for_symbol(self, symbol_name: str, project_id: str) -> None:
        """Remove all edges where this symbol is source or destination."""
        src_key = f"{project_id}:{symbol_name}"
        # Remove outgoing edges
        for edge in self._edges_out.pop(src_key, []):
            dst_key = f"{project_id}:{edge.dst}"
            self._edges_in[dst_key] = [
                e for e in self._edges_in.get(dst_key, []) if e.src != symbol_name
            ]
        # Remove incoming edges
        dst_key = f"{project_id}:{symbol_name}"
        for edge in self._edges_in.pop(dst_key, []):
            src_key_in = f"{project_id}:{edge.src}"
            self._edges_out[src_key_in] = [
                e for e in self._edges_out.get(src_key_in, []) if e.dst != symbol_name
            ]

    def get_edges_out(self, symbol_name: str, project_id: str) -> List["Edge"]:
        """Outgoing edges for a given symbol."""
        return self._edges_out.get(f"{project_id}:{symbol_name}", [])

    def get_edges_in(self, symbol_name: str, project_id: str) -> List["Edge"]:
        """Incoming edges for a given symbol."""
        return self._edges_in.get(f"{project_id}:{symbol_name}", [])

    def get_all_edges_out(self, project_id: str) -> Dict[str, List["Edge"]]:
        """
        Full outgoing edge map for a project.
        Used by ActivationGraph.propagate() and RaptorCodeIndex.
        Returns {symbol_name: [Edge, ...]}.
        """
        prefix = f"{project_id}:"
        return {
            key[len(prefix) :]: edges
            for key, edges in self._edges_out.items()
            if key.startswith(prefix)
        }

    # ── Centrality (v8) ────────────────────────────────────────────────
    def precompute_centrality(
        self,
        project_id: str,
        alpha: float = 0.85,
        max_steps: int = 30,
        tolerance: float = 1e-7,
    ) -> Dict[str, float]:
        """
        Compute normalised PageRank centrality for all symbols in the project.
        Returns {symbol_name: score in [0, 1]} where 1.0 = most central.
        Cached in self._centrality_cache[project_id].
        """
        names = list(self._iter_names(project_id))
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            scores = {names[0]: 1.0}
            self._store_centrality(project_id, scores)
            return scores

        idx = {name: i for i, name in enumerate(names)}

        # Build out-adjacency (only edges whose dst is a known symbol)
        out_links: list = [[] for _ in range(n)]
        for name in names:
            i = idx[name]
            for edge in self._edges_out.get(f"{project_id}:{name}", []):
                j = idx.get(edge.dst)
                if j is not None and j != i:
                    out_links[i].append(j)

        # PageRank power iteration
        rank = [1.0 / n] * n
        base = (1.0 - alpha) / n
        dangling_nodes = [i for i in range(n) if not out_links[i]]

        for _ in range(max_steps):
            new_rank = [base] * n

            # Distribute dangling mass uniformly
            dangling_sum = sum(rank[i] for i in dangling_nodes)
            if dangling_sum:
                share = alpha * dangling_sum / n
                for k in range(n):
                    new_rank[k] += share

            # Distribute each node's rank to its out-neighbours
            for i in range(n):
                links = out_links[i]
                if not links:
                    continue
                contrib = alpha * rank[i] / len(links)
                for j in links:
                    new_rank[j] += contrib

            # Convergence check (L1 delta)
            delta = sum(abs(new_rank[k] - rank[k]) for k in range(n))
            rank = new_rank
            if delta < tolerance:
                break

        # Normalise to [0, 1] by the maximum score
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
        """
        Return [(symbol_name, score)] for the top_n symbols by centrality,
        sorted by descending score (ties broken alphabetically for stable
        output). Used by HubSymbolIndex.build().

        If `centrality` is empty, falls back to the cached scores from the last
        precompute_centrality() call for this project.
        """
        if not centrality:
            centrality = getattr(self, "_centrality_cache", {}).get(project_id, {})
        if not centrality or top_n <= 0:
            return []
        ranked = sorted(centrality.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_n]

    # ── Internal helpers ───────────────────────────────────────────────
    def _evict_if_needed(self) -> None:
        while len(self._name_to_blocks) > self.MAX_ENTRIES:
            least_common = self._stats.most_common()[-1][0]
            project_id, symbol_name = least_common
            self.remove_edges_for_symbol(symbol_name, project_id)
            del self._name_to_blocks[least_common]
            del self._callee_to_callers[least_common]
            del self._stats[least_common]

    def _store_centrality(self, project_id: str, scores: Dict[str, float]) -> None:
        """Cache centrality scores for cheap re-reads by get_hub_symbols()."""
        self._centrality_cache[project_id] = scores

    def _iter_names(self, project_id: str):
        """Yield every symbol name in the project."""
        return iter(self.get_all_names(project_id))

    def _iter_out_edges(self, project_id: str, name: str):
        """Yield callee names (edge targets) for `name` in the project."""
        key = f"{project_id}:{name}"
        for edge in self._edges_out.get(key, []):
            yield edge.dst

    # ── Project lifecycle ──────────────────────────────────────────────
    def clear_project(self, project_id: str) -> None:
        # Remove name-to-blocks mappings
        keys_to_remove = [key for key in self._name_to_blocks if key[0] == project_id]
        for key in keys_to_remove:
            del self._name_to_blocks[key]
            del self._stats[key]

        # Remove callee-to-callers mappings
        inv_keys = [key for key in self._callee_to_callers if key[0] == project_id]
        for key in inv_keys:
            del self._callee_to_callers[key]

        # Remove typed edges for this project
        prefix = f"{project_id}:"
        for k in list(self._edges_out.keys()):
            if k.startswith(prefix):
                del self._edges_out[k]
        for k in list(self._edges_in.keys()):
            if k.startswith(prefix):
                del self._edges_in[k]

        # Remove centrality cache for this project
        self._centrality_cache.pop(project_id, None)

    def clear(self) -> None:
        self._name_to_blocks.clear()
        self._callee_to_callers.clear()
        self._stats.clear()
        self._edges_out.clear()
        self._edges_in.clear()
        self._centrality_cache.clear()


# ---------------------------------------------------------------------------
# SignatureExtractor – tree‑sitter based symbol and call extraction
# ---------------------------------------------------------------------------
class SignatureExtractor:
    """Extracts CodeSymbols and call relationships from source code."""

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

    @staticmethod
    async def extract_async(
        code: str, file_path: Optional[str] = None
    ) -> List["CodeSymbol"]:
        if len(code.encode()) > SignatureExtractor.MAX_PARSE_SIZE_BYTES:
            return []
        if not HAS_TREE_SITTER:
            syms = SignatureExtractor._extract_generic(code, file_path)
            call_map = SignatureExtractor._extract_calls_generic(code)
            for sym in syms:
                sym.calls = call_map.get(sym.name, [])
            return syms

        lang = SignatureExtractor._guess_language(file_path, code)
        if lang == "unknown":
            syms = SignatureExtractor._extract_generic(code, file_path)
            call_map = SignatureExtractor._extract_calls_generic(code)
            for sym in syms:
                sym.calls = call_map.get(sym.name, [])
            return syms

        try:
            loop = asyncio.get_event_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(
                    None, SignatureExtractor._parse_sync, code.encode(), lang
                ),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception):
            syms = SignatureExtractor._extract_generic(code, file_path)
            call_map = SignatureExtractor._extract_calls_generic(code)
            for sym in syms:
                sym.calls = call_map.get(sym.name, [])
            return syms

        syms = SignatureExtractor._extract_symbols_from_tree(
            tree, lang, code, file_path
        )
        call_map = SignatureExtractor._extract_calls_from_tree(tree, lang, code)
        del tree
        for sym in syms:
            sym.calls = call_map.get(sym.name, [])
        if lang == "python" or (file_path and file_path.endswith(".py")):
            SignatureExtractor._extract_docstrings_python(code, syms)
        return syms

    @staticmethod
    def _guess_language(file_path: Optional[str], code: str) -> str:
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
        if re.search(r"\bdef\s+\w+\s*\(", code):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code):
            return "javascript"
        return "unknown"

    @staticmethod
    def _parse_sync(code_bytes: bytes, lang: str):
        from tree_sitter import Parser as TSParser

        with SignatureExtractor._parser_cache_lock:
            parser = SignatureExtractor._parser_cache.get(lang)
        if parser is None:
            lang_obj = get_language(lang)
            parser = TSParser()
            parser.set_language(lang_obj)
            with SignatureExtractor._parser_cache_lock:
                SignatureExtractor._parser_cache[lang] = parser
        return parser.parse(code_bytes)

    @staticmethod
    def _extract_symbols_from_tree(
        tree, lang: str, code: str, file_path: Optional[str]
    ) -> List["CodeSymbol"]:
        query_str = FALLBACK_LANGUAGE_QUERIES.get(lang)
        if not query_str:
            return SignatureExtractor._extract_generic(code, file_path)
        try:
            lang_obj = get_language(lang)
            query = lang_obj.query(query_str)
            cursor = query.cursor()
            try:
                captures = cursor.captures(tree.root_node)
            except TypeError:
                captures = query.captures(tree.root_node)
            symbols = []
            for cap_name, node in captures:
                if cap_name != "name":
                    continue
                parent = node.parent
                kind = "unknown"
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
                while parent:
                    if parent.type in func_types:
                        kind = "function"
                        break
                    elif parent.type in class_types:
                        kind = "class"
                        break
                    parent = parent.parent
                sig = (
                    parent.text.decode("utf-8").split("\n")[0].strip()[:200]
                    if parent
                    else node.text.decode("utf-8")
                )
                name = node.text.decode("utf-8")
                symbols.append(
                    CodeSymbol(
                        name=name,
                        kind=kind,
                        signature=sig,
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=lang,
                    )
                )
            return symbols
        except Exception:
            return SignatureExtractor._extract_generic(code, file_path)

    @staticmethod
    def _extract_calls_from_tree(tree, lang: str, code: str) -> Dict[str, List[str]]:
        query_str = FALLBACK_CALL_QUERIES.get(lang)
        if not query_str:
            if lang == "python":
                return SignatureExtractor._extract_calls_fallback_python(code)
            return SignatureExtractor._extract_calls_generic(code)
        try:
            lang_obj = get_language(lang)
            query = lang_obj.query(query_str)
            cursor = query.cursor()
            try:
                captures = cursor.captures(tree.root_node)
            except TypeError:
                captures = query.captures(tree.root_node)
            call_map: Dict[str, Set[str]] = defaultdict(set)
            current_arrow_caller = None
            for cap_name, node in captures:
                if cap_name == "caller_name":
                    current_arrow_caller = node.text.decode("utf-8")
                    continue
                if cap_name == "callee":
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
                            break
                        parent = parent.parent
                    if caller:
                        call_map[caller].add(callee_name)
            return {k: list(v) for k, v in call_map.items()}
        except Exception:
            return {}

    @staticmethod
    def _extract_docstrings_python(code: str, symbols: List["CodeSymbol"]) -> None:
        try:
            tree = ast.parse(code)
            doc_map = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        first_line = docstring.strip().split("\n")[0].strip()
                        if first_line:
                            doc_map[node.name] = first_line[:120]
            for sym in symbols:
                if sym.name in doc_map and not sym.summary:
                    sym.summary = doc_map[sym.name]
        except SyntaxError:
            pass

    @staticmethod
    def _extract_calls_fallback_python(code: str) -> Dict[str, List[str]]:
        call_map: Dict[str, Set[str]] = defaultdict(set)
        try:
            tree = ast.parse(code)
            scope_stack = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope_stack.append(node.name)
                elif isinstance(node, ast.Call) and scope_stack:
                    if isinstance(node.func, ast.Name):
                        callee = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        callee = node.func.attr
                    else:
                        callee = None
                    if callee:
                        for scope in reversed(scope_stack):
                            call_map[scope].add(callee)
            return {k: list(v) for k, v in call_map.items()}
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            return {}

    @staticmethod
    def _extract_calls_generic(code: str) -> Dict[str, List[str]]:
        call_map: Dict[str, Set[str]] = defaultdict(set)
        func_pattern = (
            r"^\s*(?:def|function|fn|func)\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\S+)?\s*:?"
        )
        for match in re.finditer(func_pattern, code, re.MULTILINE | re.I):
            func_name = match.group(1)
            rest = code[match.end() :]
            next_match = re.search(
                r"^\s*(?:def|function|class|fn|func|export)\s+", rest, re.MULTILINE
            )
            body = rest[: next_match.start()] if next_match else rest
            calls_simple = set(re.findall(r"\b(\w+)\s*\(", body))
            calls_dotted = set(re.findall(r"\.(\w+)\s*\(", body))
            calls = calls_simple | calls_dotted
            keywords = {
                "if",
                "for",
                "while",
                "switch",
                "return",
                "print",
                "assert",
                "throw",
                "new",
                "typeof",
                "instanceof",
                "delete",
                "void",
                "in",
                "of",
                "catch",
                "finally",
                "class",
                "import",
                "export",
                "from",
                "as",
                "try",
                "except",
                "raise",
                "yield",
                "await",
                "async",
                "break",
                "continue",
                "pass",
            }
            calls -= keywords
            call_map[func_name] = calls
        return {k: list(v) for k, v in call_map.items()}

    @staticmethod
    def _extract_generic(
        code: str, file_path: Optional[str] = None
    ) -> List["CodeSymbol"]:
        symbols = []
        for match in re.finditer(
            r"^\s*(def|function|class|fn|func)\s+(\w+)", code, re.MULTILINE | re.I
        ):
            kind = (
                "function"
                if match.group(1).lower() in ("def", "function", "fn", "func")
                else "class"
            )
            symbols.append(
                CodeSymbol(
                    name=match.group(2),
                    kind=kind,
                    signature=match.group(0).strip(),
                    file_path=file_path,
                    language="unknown",
                )
            )
        return symbols


class StateStore:
    """SQLite-backed conversation state, DB write queue, and project locks."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ── State access ──────────────────────────────────────────────────
    def get_state(self, project_id: str) -> dict:
        """Return the conversation state for the given project, loading from DB if needed."""
        if project_id in self._f._conversation_state:
            self._f._conversation_state.move_to_end(project_id)
            return self._f._conversation_state[project_id]

        state = self._load_state_from_db(project_id)
        if not state:
            state = self._f._state_factory()

        self._f._conversation_state[project_id] = state
        self._f._conversation_state.move_to_end(project_id)

        while len(self._f._conversation_state) > self._f.valves.max_cached_projects:
            oldest_pid = next(iter(self._f._conversation_state))
            oldest_state = self._f._conversation_state[oldest_pid]
            self._f._symbol_index.clear_project(oldest_pid)
            del self._f._conversation_state[oldest_pid]
            self._f._cached_lightweight_context.pop(oldest_pid, None)
            self._f._path_index.clear_project(oldest_pid)
            self._f._pager.clear_project(oldest_pid)
            self._f._node_centrality.pop(oldest_pid, None)
            self._f._last_static_prefix_hash.pop(oldest_pid, None)
            self._f._last_saved_slot_hash.pop(oldest_pid, None)
            self._f._slot_restored.pop(oldest_pid, None)
            self._f._slot_restore_attempted.pop(oldest_pid, None)
            self._f._last_processed_message_idx.pop(oldest_pid, None)
            self._f._response_cache_count.pop(oldest_pid, None)
            self._f._summarize_inactive_in_progress.pop(oldest_pid, None)
            getattr(self._f, "_last_activation_scores", {}).pop(oldest_pid, None)
            getattr(self._f, "_last_lod_levels", {}).pop(oldest_pid, None)

        if state.get("active_blocks"):
            self._rebuild_symbol_index(state, project_id)

        return state

    def set_state(self, project_id: str, state: dict) -> None:
        """Mark the conversation state as dirty without persisting immediately."""
        self._f._conversation_state[project_id] = state
        self._f._conversation_state.move_to_end(project_id)
        self._f._state_dirty = True

    async def save_state_if_dirty(self, project_id: str) -> None:
        """
        Persist the state if dirty and at least 2 seconds have passed since last save.
        Waits for the DB write to complete, logging any error.
        """
        if not self._f._state_dirty:
            return
        now = time.time()
        if now - self._f._state_last_saved < 2.0:
            return

        self._f._state_last_saved = now
        self._f._state_dirty = False

        try:
            await self._save_state_to_db_async(
                project_id, self._f._conversation_state[project_id]
            )
        except Exception as e:
            import traceback

            self._f._log_debug(f"Failed to save state: {e}\n{traceback.format_exc()}")

    async def _save_state_to_db_async(self, project_id: str, state: dict) -> None:
        """Acquire the project lock, then persist the state to DB."""
        lock = await self.get_project_lock(project_id)
        async with lock:
            await self._save_state_to_db(project_id, state)

    async def _save_state_to_db(self, project_id: str, state: dict) -> None:
        """Serialize active blocks metadata and persist to SQLite."""
        # Serialize active blocks metadata
        active_blocks_meta = {}
        for k, v in state["active_blocks"].items():
            d = v.dict()
            d["content_type"] = v.content_type.value
            content_hash = v.hash
            # Persist the raw content via the write queue
            await self._db_enqueue(
                lambda ch=content_hash, ct=v.content: self._f._db_conn.execute(
                    "INSERT OR IGNORE INTO code_contents (hash, content, created_at) VALUES (?, ?, ?)",
                    (ch, ct, time.time()),
                )
            )
            d["content"] = f"@@hash:{content_hash}"
            active_blocks_meta[k] = d

        serializable = {
            "active_blocks": active_blocks_meta,
            "recent_changes": [b.dict() for b in state["recent_changes"]],
            "committed_changes": [b.dict() for b in state["committed_changes"]],
            "feedback_history": [fb.dict() for fb in state["feedback_history"]],
            "message_count": state["message_count"],
            "last_compression_timestamp": state.get("last_compression_timestamp", 0),
            "response_cache": state.get("response_cache", []),
            "last_suggestion_timestamp": state.get("last_suggestion_timestamp", 0),
            "last_cleanup_suggestion_msg_idx": state.get(
                "last_cleanup_suggestion_msg_idx", 0
            ),
            "has_any_calls": state.get("has_any_calls", False),
            "last_cot_level": state.get("last_cot_level", 0),
            "conversation_summaries": state.get("conversation_summaries", []),
        }

        def _write():
            self._f._db_conn.execute(
                "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
                (project_id, json.dumps(serializable), time.time()),
            )
            self._f._db_conn.commit()

        await self._db_enqueue(_write)

    # ── DB lifecycle ──────────────────────────────────────────────────
    def init_db(self) -> None:
        """Create tables and indexes (idempotent)."""
        db_path = self._f.valves.state_db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._f._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._f._db_conn.execute(
            f"PRAGMA busy_timeout = {self._f.valves.llm_per_call_timeout * 1000}"
        )
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
        self._f._db_conn.commit()

    def _load_state_from_db(self, project_id: str) -> Optional[dict]:
        """Carga el estado de conversación desde SQLite. Devuelve None si no existe."""
        cur = self._f._db_conn.execute(
            "SELECT state_json FROM conversation_state WHERE project_id = ?",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row[0])

        # Asegurar claves esperadas con valores por defecto
        for key in [
            "feedback_history",
            "last_compression_timestamp",
            "last_suggestion_timestamp",
            "response_cache",
            "has_any_calls",
            "last_cot_level",
        ]:
            data.setdefault(
                key, [] if key in ("feedback_history", "response_cache") else 0
            )
        data.setdefault("last_cleanup_suggestion_msg_idx", 0)

        # Detección de corrupción en active_blocks
        raw_active = data.get("active_blocks")
        if raw_active is None:
            self._f._log_debug(
                "⚠️  CORRUPT STATE: 'active_blocks' is missing or null in DB. "
                "To fix, delete the database file: %s and restart."
                % self._f.valves.state_db_path
            )
            raw_active = {}
        elif not isinstance(raw_active, dict):
            self._f._log_debug(
                "⚠️  CORRUPT STATE: 'active_blocks' is not a dict (type=%s). "
                "Resetting to empty. If problems persist, delete the DB file: %s"
                % (type(raw_active).__name__, self._f.valves.state_db_path)
            )
            raw_active = {}

        active = {}
        for k, v in raw_active.items():
            try:
                content_field = v.get("content", "")
                if content_field.startswith("@@hash:"):
                    content_hash = content_field[7:]
                    cur2 = self._f._db_conn.execute(
                        "SELECT content FROM code_contents WHERE hash = ?",
                        (content_hash,),
                    )
                    row2 = cur2.fetchone()
                    if row2:
                        v["content"] = row2[0]
                    else:
                        v["content"] = f"[Content not found for hash {content_hash}]"
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
                self._f._log_debug("Skipping corrupted block %s in state DB" % k)

        # Restaurar otras colecciones
        recent = []
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

        committed = []
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

        feedback = [
            AppliedChangeFeedback(**fb) for fb in data.get("feedback_history", [])
        ]

        state = {
            "active_blocks": active,
            "recent_changes": recent,
            "committed_changes": committed,
            "feedback_history": feedback,
            "message_count": data.get("message_count", 0),
            "last_compression_timestamp": data.get("last_compression_timestamp", 0),
            "response_cache": data.get("response_cache", []),
            "last_suggestion_timestamp": data.get("last_suggestion_timestamp", 0),
            "has_any_calls": data.get("has_any_calls", False),
            "last_cleanup_suggestion_msg_idx": data.get(
                "last_cleanup_suggestion_msg_idx", 0
            ),
            "last_cot_level": data.get("last_cot_level", 0),
            "conversation_summaries": data.get("conversation_summaries", []),
        }

        # Recalculate cached token counts
        for blk in (
            list(state["active_blocks"].values())
            + state["recent_changes"]
            + state["committed_changes"]
        ):
            if self._f.tokenizer:
                blk._cached_token_count = len(self._f.tokenizer.encode(blk.content))
            else:
                blk._cached_token_count = len(blk.content) // 4

        return state

    def _rebuild_symbol_index(self, state: dict, project_id: str) -> None:
        """Reconstruye el SymbolIndex al cargar un estado en frío."""
        for block in state.get("active_blocks", {}).values():
            if block.obsolete:
                continue
            for sym in block.symbols:
                self._f._symbol_index.add(sym, block.hash, project_id)
                for callee in sym.calls:
                    edge = Edge(
                        src=sym.name,
                        dst=callee,
                        type="calls",
                        weight=EDGE_WEIGHTS["calls"],
                    )
                    self._f._symbol_index.add_edge(edge, project_id)

    async def _db_enqueue(self, fn, args=(), kwargs=None) -> None:
        """Encola una operación de escritura en la cola del worker de BD."""
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

    async def _db_worker_loop(self) -> None:
        """Single run of the DB write loop. Exits on CancelledError."""
        while True:
            try:
                job = await asyncio.wait_for(self._f._db_write_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            func, args, kwargs = job
            with _db_global_lock:
                for attempt in range(5):
                    try:
                        await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower() and attempt < 4:
                            await asyncio.sleep(0.5 * (attempt + 1))
                        else:
                            raise

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

    # ── Edge & path persistence ───────────────────────────────────────
    async def save_symbol_edges_to_db(self, project_id: str) -> int:
        """
        Persist the typed edges from the SymbolIndex to SQLite.
        Saves alongside the current code_state_hash for invalidation detection.
        Returns the number of edges saved.
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

        # Check if the saved hash matches
        import anyio

        meta_row = await anyio.to_thread.run_sync(
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

        # Hash matches → restore edges
        rows = await anyio.to_thread.run_sync(
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

    async def save_path_views_to_db(self, project_id: str, views: list) -> None:
        """Persist CodePathViews to SQLite, replacing any existing views for the project."""

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
        """Carga las CodePathViews almacenadas en SQLite para un proyecto."""
        import anyio

        rows = await anyio.to_thread.run_sync(
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

    # ── Project locks ─────────────────────────────────────────────────
    async def get_project_lock(self, project_id: str) -> "ReentrantAsyncLock":
        """Return (or create) the reentrant async lock for the given project."""
        async with self._f._lock_lock:
            if project_id not in self._f._project_locks:
                self._f._project_locks[project_id] = ReentrantAsyncLock()
            return self._f._project_locks[project_id]


class LongTermMemory:
    """ChromaDB embeddings, conversation memory, and response cache."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def init(self) -> None:
        """Initialise ChromaDB, embedder, and response cache collection."""
        os.makedirs(self._f.valves.long_term_memory_dir, exist_ok=True)
        self._f.embedder = _shared_get_embedder()
        self._f._log_debug("Embedder: using multilingual-e5-large")

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Response cache & duplicate detection
    # ═══════════════════════════════════════════════════════════════════════════

    async def find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        if not self._f.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        col = getattr(self._f, "_response_cache_collection", None)
        if col is None:
            return None

        query_vec = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
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

        dist = results["distances"][0][0]
        similarity = 1.0 - (dist / 2.0)
        if similarity < self._f.valves.response_cache_similarity_threshold:
            return None

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
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return None
        if not query or len(query.strip()) < 15:
            return None
        try:
            q_emb = await anyio.to_thread.run_sync(
                lambda: self._f.embedder.encode(query[:1000]).tolist()
            )
            now = time.time()
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

            best_candidate = None
            best_sim = 0.0
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i]
                sim = 1.0 - (dist / 2.0)
                if sim >= self._f.valves.duplicate_question_threshold and doc != query:
                    pairs = [(query[:500], doc[:500])]
                    ce_score = await self._f._commands._predict_cross_encoder(pairs)
                    if ce_score is None:
                        self._f._log_debug(
                            "_find_duplicate_question: CrossEncoder not loaded, "
                            "using cosine similarity only (higher false positive risk)."
                        )
                        best_candidate = (sim, doc, None)
                        break
                    if ce_score[0] > 0.85:
                        best_candidate = (sim, doc, ce_score[0])
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
    # Query helpers
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
        if self._f.valves.long_term_memory_expiration_days > 0:
            where["$and"].append({"expires_at": {"$gt": now}})
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

    async def _expand_query_for_retrieval(
        self, query: str, slot_free: bool = True
    ) -> List[str]:
        """Generate alternative phrasings of the query for LTM retrieval."""
        if not self._f.valves.enable_multi_query_retrieval:
            return [query]
        if not slot_free:
            return [query]
        if len(query.strip()) < 15:
            return [query]

        prompt = (
            f"Generate {self._f.valves.multi_query_variants} alternative phrasings "
            f"of this programming question for document search. "
            f"Focus on different vocabulary (errors, function names, behaviors).\n\n"
            f"Original: {query[:250]}\n\n"
            f"Output only the alternatives, one per line. No numbering."
        )
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt=(
                "Output only the alternative phrasings, one per line. "
                "Be concise and specific to the code context."
            ),
            model_override=self._f.valves.llm_model,
            max_tokens=80,
            temperature=0.6,
            label="multi_query_expand",
        )

        queries = [query]
        if response:
            alternatives = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            queries.extend(alternatives[: self._f.valves.multi_query_variants])
            self._f._log_debug(
                f"Multi-query expansion: {len(queries)} queries "
                f"({[q[:40] for q in queries]})"
            )
        return queries

    def _extract_query_symbols(self, query: str, project_id: str) -> Set[str]:
        """Return symbol names from the query that exist in the SymbolIndex."""
        if not query or not project_id:
            return set()
        words = set(re.findall(r"\b\w+\b", query))
        project_symbols = self._f._symbol_index.get_all_names(project_id)
        return words.intersection(project_symbols)

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

    async def _build_retrieval_context(
        self,
        content: str,
        project_id: str,
        role: str,
        code_symbols: List[str],
        file_paths: List[str],
        content_type: str,
    ) -> str:
        if not self._f.valves.enable_contextual_retrieval:
            return ""

        if self._f.valves.contextual_retrieval_mode == "llm":
            return await self._build_retrieval_context_llm(content, project_id)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Memory retrieval
    # ═══════════════════════════════════════════════════════════════════════════

    async def retrieve_memories_unified(
        self, query: str, project_id: str, slot_free: bool = True
    ) -> list:
        """Retrieve relevant LTM entries, with multi‑query expansion and reranking."""
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return []

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            return await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )

        try:
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self._f.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})

            query_variants = await self._expand_query_for_retrieval(
                query, slot_free=slot_free
            )

            all_raw_results: Dict[str, Tuple[str, float, Any, Any]] = {}

            for variant_query in query_variants:
                q_emb = await anyio.to_thread.run_sync(
                    lambda q=variant_query: self._f.embedder.encode(q[:1000]).tolist()
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
                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None

                    mem_id = meta.get(
                        "memory_id",
                        hashlib.md5(doc.encode()).hexdigest()[:16],
                    )
                    if (
                        mem_id not in all_raw_results
                        or raw_sim > all_raw_results[mem_id][1]
                    ):
                        all_raw_results[mem_id] = (doc, raw_sim, ts, meta)

            results_list = list(all_raw_results.values())

            docs_with_meta = []
            if results_list:
                for doc, raw_sim, ts, meta in results_list:
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

                    if (
                        effective_sim
                        < self._f.valves.long_term_memory_similarity_threshold
                    ):
                        continue

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
                doc_to_meta = {d[0]: (d[1], d[2]) for d in docs_with_meta}
                docs_with_meta = [
                    (doc, *doc_to_meta.get(doc, (0.0, None))) for doc in reranked
                ]

            docs_with_meta = docs_with_meta[: self._f.valves.long_term_memory_top_k]

            normalized = []
            for entry in docs_with_meta:
                if len(entry) == 5:
                    doc, score, ts, _, _ = entry
                elif len(entry) == 3:
                    doc, score, ts = entry
                else:
                    doc, score, ts = entry[0], entry[1], entry[2]
                normalized.append((doc, score, ts))

            return [{"doc": doc, "timestamp": ts} for doc, _, ts in normalized]
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
            q_emb = await anyio.to_thread.run_sync(
                lambda: self._f.embedder.encode(query[:1000]).tolist()
            )
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self._f.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})

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
    # Message storage
    # ═══════════════════════════════════════════════════════════════════════════

    async def store_messages(self, project_id: str, messages: list) -> None:
        """Store user/assistant messages in the LTM ChromaDB collection."""
        if not HAS_SENTENCE or not HAS_CHROMA or self._f.memory_collection is None:
            return
        valid = [
            m
            for m in messages
            if m.get("content", "").strip() and len(m["content"].strip()) >= 15
        ]
        if not valid:
            return

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
                        blk["code"], blk.get("language")
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
                            blk["code"], blk.get("language")
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

        embeddings = await anyio.to_thread.run_sync(
            lambda: self._f.embedder.encode(
                texts_for_embedding, convert_to_numpy=True
            ).tolist()
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

    async def store_response_in_cache(
        self,
        query: str,
        response: str,
        context_hash: str,
        state: dict,
        code_state_hash: str,
    ) -> None:
        """Store a response in the ChromaDB response cache for future reuse."""
        if not self._f.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
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
        current_size = self._f._response_cache_count.get(project, 0)
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
                    self._f._response_cache_count[project] -= len(old_entries["ids"])
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
        self._f._response_cache_count[project] = (
            self._f._response_cache_count.get(project, 0) + 1
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Maintenance
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


class LLMOrchestrator:
    """Shared LLM caller, cache, and task tracking."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    async def _maybe_unload_for_model(
        self, model_name: str, base_url: str, is_ollama: bool
    ) -> None:
        """
        Unload models only if switching to a *different* auxiliary model.
        The main model (self._f.valves.llm_model) is NEVER unloaded to preserve its KV cache.
        """
        if is_ollama:
            return

        main_model = self._f.valves.llm_model

        async with self._f._model_lock:
            current_model = self._f._last_used_model

        if model_name == main_model:
            if current_model is None:
                self._f._log_debug(
                    f"Loading main model '{model_name}' for the first time"
                )
            else:
                self._f._log_debug(
                    f"Keeping main model '{model_name}' loaded (no unload)"
                )
            return

        if current_model == main_model:
            self._f._log_debug(
                f"Keeping main model '{main_model}' loaded while loading auxiliary '{model_name}'"
            )
            return

        if current_model is not None and model_name != current_model:
            self._f._log_debug(
                f"Switching auxiliary model from '{current_model}' to '{model_name}'"
            )
            try:
                await _shared_unload_all_models(base_url)
                self._f._log_debug("Auxiliary model unloaded before switching")
                async with self._f._model_lock:
                    self._f._last_used_model = None
            except Exception as e:
                self._f._log_debug(f"Unload via shared_resources failed: {e}")
        elif current_model is None:
            self._f._log_debug(
                f"Loading auxiliary model '{model_name}' (no model was loaded)"
            )
        else:
            self._f._log_debug(
                f"Reusing auxiliary model '{model_name}' (already loaded)"
            )

    async def _acquire_llm_lock(self):
        """Acquire an inter‑process file lock for exclusive LLM access."""
        loop = asyncio.get_event_loop()
        fd = open(_llm_lock_path, "w")
        await loop.run_in_executor(self._f._db_executor, fcntl.flock, fd, fcntl.LOCK_EX)
        return fd

    @staticmethod
    def _release_llm_lock(fd):
        """Release the inter‑process file lock and close the file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

    def init_cache(self) -> None:
        """Return the shared AsyncLRUCache instance for LLM response caching."""
        self._f._llm_cache = _AsyncLRUCache(
            max_size=self._f.valves.LLM_CACHE_MAX_SIZE,
            ttl=self._f.valves.LLM_CACHE_TTL,
        )

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, Any]] = None,
        label: str = "",
        total_timeout: Optional[float] = None,
    ) -> Optional[str]:
        """
        Call the LLM with automatic retries for transient errors.
        Retries continue until *total_timeout* seconds have passed since the
        first attempt.  If *total_timeout* is None, the valve default
        ``llm_retry_total_timeout`` is used.
        Between retries, a fixed 1‑second pause is applied.
        """
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

        effective_total_timeout = (
            total_timeout
            if total_timeout is not None
            else self._f.valves.llm_retry_total_timeout
        )
        deadline = t_start + effective_total_timeout

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

            RETRY_DELAY = 1.0

            if self._f.tokenizer:
                prompt_tokens = len(self._f.tokenizer.encode(prompt))
                self._f._log_debug(
                    f"LLM call to {model}{label_str} – prompt size: ~{prompt_tokens} tokens"
                )

            task = asyncio.current_task()
            async with self._f._active_llm_tasks_lock:
                self._f._active_llm_tasks.add(task)
            try:
                llm_fd = await self._acquire_llm_lock()
                try:
                    # Unload/load management fuera del semáforo de concurrencia
                    await self._maybe_unload_for_model(model, base_url, is_ollama)

                    attempt = 0
                    while time.monotonic() < deadline:
                        attempt += 1
                        try:
                            async with self._f._llm_semaphore:
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
                                reason = "empty response"
                        except asyncio.CancelledError:
                            raise
                        except RuntimeError as exc:
                            reason = f"RuntimeError: {exc}"
                            if not any(
                                c in str(exc)
                                for c in ("429", "500", "502", "503", "504")
                            ):
                                self._f._log_debug(
                                    f"[LLM] {model}{label_str} attempt {attempt} failed "
                                    f"with non-retryable error: {exc}"
                                )
                                break
                        except Exception as exc:
                            reason = f"{type(exc).__name__}: {exc}"

                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        wait = min(RETRY_DELAY, remaining)
                        self._f._log_debug(
                            f"[LLM] {model}{label_str} attempt {attempt} failed ({reason}), "
                            f"retrying in {wait:.1f}s (deadline in {remaining:.0f}s)"
                        )
                        await asyncio.sleep(wait)
                finally:
                    self._release_llm_lock(llm_fd)
            finally:
                async with self._f._active_llm_tasks_lock:
                    self._f._active_llm_tasks.discard(task)

            logger.warning(f"[LLM] {model}{label_str} failed: {prompt[:100]}...")
            future.set_result(None)
            self._f._log_debug(
                f"[LLM] {model}{label_str} (failed) after {time.monotonic() - t_start:.3f}s"
            )
            return None

        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._f._pending_llm_lock:
                self._f._pending_llm.pop(dedup_key, None)

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

    async def wait_for_llm_tasks(self) -> None:
        """Block until all LLM-using tasks have completed."""
        while True:
            async with self._f._active_llm_tasks_lock:
                if not self._f._active_llm_tasks:
                    break
                tasks = list(self._f._active_llm_tasks)
            if tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)


class ReasoningEngine:
    """Chain‑of‑Thought detection and generation."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

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
            state["last_cot_level"] = level

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
            prev_level = state.get("last_cot_level", 0)
            if prev_level >= 2 and has_complex_kw:
                signals += 1

        if signals >= 5:
            return 2
        elif signals >= 3:
            return 1
        else:
            return 0

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


class MultiPhasePlanner:
    """Multi‑phase response instructions and wrap‑up hints."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

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
    """Explicit commands, natural‑language intents, and suggestions."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    async def _predict_cross_encoder(self, pairs: list) -> Optional[list]:
        if self._f._cross_encoder is None:
            if not self._f._cross_encoder_unavailable_logged:
                self._f._log_debug(
                    "CrossEncoder not loaded – predictions will return None."
                )
                self._f._cross_encoder_unavailable_logged = True
            return None
        async with self._f._cross_encoder_lock:
            return await anyio.to_thread.run_sync(self._f._cross_encoder.predict, pairs)

    async def _detect_contradictions(self, messages: list) -> Optional[str]:
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

    async def classify_intent(self, user_query: str, project_id: str) -> dict:
        pairs = [
            (user_query[:500], "The user wants to understand or explain code."),
            (user_query[:500], "The user wants to modify, fix, or create code."),
            (user_query[:500], "The user is debugging an error or exception."),
            (user_query[:500], "The user wants to refactor or restructure code."),
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
                state = self._f._state_store.get_state(project_id)
                for h in candidates:
                    blk = state["active_blocks"].get(h)
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

    async def outlet_intercept_expand(
        self,
        assistant_content: str,
        project_id: str,
    ) -> Tuple[str, bool]:
        """
        Intercept /expand commands in the assistant's response and replace them
        with the actual expanded symbol code from the SymbolIndex.
        Returns (modified_content, did_expand).
        """
        if not self._f.valves.outlet_expand_intercept_enabled:
            return assistant_content, False

        EXPAND_RE = re.compile(r"/expand\s+(?:(\d+)\s+)?(\w+)", re.IGNORECASE)
        matches = list(EXPAND_RE.finditer(assistant_content))
        if not matches:
            return assistant_content, False

        all_names = self._f._symbol_index.get_all_names(project_id)
        replaced_content = assistant_content
        did_any = False
        state = self._f._state_store.get_state(project_id)

        max_syms = self._f.valves.outlet_expand_intercept_max_symbols
        matches_to_process = matches if max_syms == 0 else matches[:max_syms]

        for match in matches_to_process:
            depth_str = match.group(1)
            func_name = match.group(2)
            depth = (
                int(depth_str)
                if depth_str
                else self._f.valves.outlet_expand_intercept_depth
            )
            if depth == 0:
                depth = 9999

            if func_name not in all_names:
                continue

            expanded = await self._expand_symbol_dependencies(
                func_name, depth, project_id
            )
            if not expanded:
                continue

            did_any = True
            replacement = f"[Retrieved `{func_name}`]\n{expanded}"
            replaced_content = replaced_content.replace(match.group(0), replacement, 1)

            lock = await self._f._state_store.get_project_lock(project_id)
            async with lock:
                block_hashes = self._f._symbol_index.find_blocks(func_name, project_id)
                for h in block_hashes:
                    block = state["active_blocks"].get(h)
                    if block and not block.obsolete:
                        block.is_raw = True
                        block.pinned = True
                        block.importance_score = 10.0
                        block.last_mentioned = time.time()
                        block.last_mentioned_msg_idx = state["message_count"]
                        break
                self._f._activation.invalidate_lightweight_cache(project_id)
                self._f._state_store.set_state(project_id, state)

        return replaced_content, did_any

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
        if not self._f.valves.enable_natural_language_forget:
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

        code_spans = await self._f._code_blocks.get_code_spans(user_message)
        cleaned = CodeBlockManager.remove_code_spans(user_message, code_spans).strip()

        model = (
            self._f.valves.natural_language_forget_model
            or self._f.valves.llm_model
            or self._f.valves.summarization_model
        )
        prompt = (
            "You are a command parser for a code assistant. Analyze the user message and detect "
            "ALL of these intents simultaneously.\n\n"
            "FORGET actions: forget_last, forget_n (n=int), forget_file (file=str), "
            "forget_block (hash=str), forget_all\n"
            "REMEMBER/PIN actions: pin_last, pin_n (n=int), pin_file (file=str), "
            "pin_block (description=str), pin_all, unpin_last, unpin_file, unpin_all\n"
            "OBSOLETE actions: obsolete_last, obsolete_n (n=int), obsolete_file (file=str), "
            "obsolete_block (hash=str), obsolete_all, revive_last, revive_file, revive_all\n\n"
            f'User message (code removed): "{cleaned[:400]}"\n\n'
            "Output JSON with exactly three keys. Use action='none' when not detected:\n"
            '{"forget": {"action": "..."}, "remember": {"action": "..."}, "obsolete": {"action": "..."}}\n'
            "Output only JSON."
        )
        response = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=200,
            temperature=0.0,
            label="parse_intents",
        )

        none = {"action": "none"}
        if not response:
            return {"forget": none, "remember": none, "obsolete": none}
        try:
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            return {
                "forget": data.get("forget", none),
                "remember": data.get("remember", none),
                "obsolete": data.get("obsolete", none),
            }
        except Exception:
            return {"forget": none, "remember": none, "obsolete": none}

    async def suggest_commands(self, project_id: str, state: dict) -> Optional[str]:
        """Suggest context management commands to the user after enough messages."""
        if not self._f.valves.enable_command_suggestions:
            return None
        now = time.time()
        last_sugg = state.get("last_suggestion_timestamp", 0)
        if now - last_sugg < self._f.valves.command_suggestion_cooldown_minutes * 60:
            return None
        if state["message_count"] > 15 and not state.get("has_any_calls"):
            state["last_suggestion_timestamp"] = now
            return (
                "[CodeAware] Tip: You can manage context with commands like "
                "`/forget`, `/remember`, `/status`, `/clean`. Use `/help` for more info."
            )
        return None

    async def is_code_only_message(self, content: str) -> bool:
        """
        Detect messages that contain only code without a question.
        Uses a fast path for large raw code pastes (no fences) based on
        Python structural line ratio and optional CrossEncoder intent check
        on the non‑code prose.
        """
        if not content or len(content.strip()) < 20:
            return False

        # ── Fast path: large raw code paste without fences ──────────────
        estimated_tokens = self._f._tokens.estimate_code_tokens(content)
        if estimated_tokens >= self._f.valves.lean_user_code_min_tokens:

            _PY_STRUCTURAL = re.compile(
                r"^\s*(?:def |async def |class |import |from |@\w|"
                r"if |elif |else:|for |while |try:|except|with |"
                r"return |yield |raise |pass\b|break\b|continue\b|#)"
            )
            non_blank = [l for l in content.splitlines() if l.strip()]
            prose_lines = [
                l
                for l in non_blank
                if not _PY_STRUCTURAL.match(l)
                and not re.match(r'^\s*[\w.]+\s*[=({"\']', l)
                and not re.match(r'^\s*"""', l)
                and not re.match(r"^\s*'''", l)
            ]
            prose_text = " ".join(prose_lines).strip()

            if not prose_text or len(prose_text) < 3:
                code_ratio = sum(1 for l in non_blank if _PY_STRUCTURAL.match(l)) / len(
                    non_blank
                )
                if code_ratio > 0.07:
                    return True
                return False

            has_question = any(l.strip().endswith("?") for l in prose_lines)
            if has_question:
                self._f._log_debug(
                    "_is_code_only_message: explicit question detected → not silent"
                )
                return False

            pairs = [
                (prose_text, "The user is asking a question or making a request."),
                (prose_text, "This text contains no user question or request."),
            ]
            scores = await self._predict_cross_encoder(pairs)
            if scores is None:
                self._f._log_debug(
                    "_is_code_only_message: CrossEncoder not available, "
                    "falling back to keyword intent detection"
                )
                _INTENT_RE = re.compile(
                    r"\b(?:explain|describe|analyze|review|fix|refactor|optimize|"
                    r"improve|rewrite|check|summarize|show|tell|what|how|why|"
                    r"explica|analiza|revisa|corrige|refactoriza|optimiza|mejora|"
                    r"reescribe|comprueba|resume|muestra|dime|qu[eé]|c[oó]mo|"
                    r"por qu[eé])\b",
                    re.IGNORECASE,
                )
                if _INTENT_RE.search(prose_text):
                    return False
            else:
                has_intent = scores[0] > scores[1]
                if has_intent:
                    self._f._log_debug(
                        f"_is_code_only_message: intent detected "
                        f"('{prose_text[:60]}') → not silent"
                    )
                    return False

            code_ratio = sum(1 for l in non_blank if _PY_STRUCTURAL.match(l)) / len(
                non_blank
            )
            return code_ratio > 0.07

        # ── Original logic for fenced blocks or smaller messages ──
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


class CodeBlockManager:
    """Extraction, classification, deduplication, and diff application."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        self._f = filter_ref

    # ── Block extraction ─────────────────────────────────────────────────

    async def get_code_spans(self, content: str) -> List[Tuple[int, int]]:
        """Return tree‑sitter code spans for the given content (cached)."""
        if not HAS_TREE_SITTER:
            return []
        cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
        if cache_key in self._code_spans_cache:
            return self._code_spans_cache[cache_key]
        try:
            config = ProcessConfig()
            blocks = process(content, config)
            spans = [(b.start_byte, b.end_byte) for b in blocks]
        except Exception:
            spans = []
        if len(self._code_spans_cache) >= 200:
            keys_to_evict = list(self._code_spans_cache.keys())[:50]
            for key in keys_to_evict:
                del self._code_spans_cache[key]
        self._code_spans_cache[cache_key] = spans
        return spans

    @staticmethod
    def remove_code_spans(content: str, spans: List[Tuple[int, int]]) -> str:
        """Replace code regions with spaces."""
        chars = list(content)
        for start, end in spans:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        return "".join(chars)

    async def extract_code_blocks(
        self, content: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        """Extract fenced and indented code blocks from message content."""
        blocks = []
        spans = []
        if not self._f.valves.auto_detect_code_blocks:
            return blocks, spans
        # tree-sitter attempt
        if HAS_TREE_SITTER:
            try:
                config = ProcessConfig()
                ts_blocks = await anyio.to_thread.run_sync(
                    lambda: process(content, config)
                )
                for tsb in ts_blocks:
                    start, end = tsb.start_byte, tsb.end_byte
                    raw = content[start:end].strip()
                    lang = tsb.language or "text"
                    if lang in ("text", ""):
                        guessed = SignatureExtractor._guess_language(None, raw)
                        if guessed != "unknown":
                            lang = guessed
                        else:
                            lang = await self._infer_code_language(raw)
                    lines = raw.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        code = "\n".join(lines).strip()
                        block_type = "fenced"
                    else:
                        code = raw
                        block_type = "indented"
                    blocks.append({"language": lang, "code": code, "type": block_type})
                    spans.append((start, end))
                if blocks:
                    return blocks, spans
            except Exception:
                pass

        # Regex fallback
        for match in self._f.code_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            blocks.append({"language": lang, "code": code, "type": "fenced"})
            spans.append((match.start(), match.end()))
        # indented blocks
        lines = content.split("\n")
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)
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
                    blocks.append(
                        {"language": "text", "code": code, "type": "indented"}
                    )
                    start_offset = line_offsets[i - len(indented)]
                    end_offset = line_offsets[i] - 1
                    spans.append((start_offset, end_offset))
                indented = []
                i += 1
        if len(indented) >= 3:
            code = "\n".join(indented)
            blocks.append({"language": "text", "code": code, "type": "indented"})
            start_offset = line_offsets[len(lines) - len(indented)]
            end_offset = line_offsets[-1] - 1 if line_offsets[-1] > 0 else len(content)
            spans.append((start_offset, end_offset))

        # Post-processing and file path extraction with auto‑symbol filter
        processed_blocks = []
        processed_spans = []
        for idx, block in enumerate(blocks):
            blk_file = None
            if self._f.valves.track_file_paths and spans:
                blk_file = self.extract_file_path_for_block(content, spans[idx][0])
            if not blk_file and len(blocks) == 1:
                extracted_paths = self.extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None

            # Exclude blocks that belong to the filter's own source code
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
        """Simple heuristic language detection for a code snippet."""
        if re.search(r"\bdef\s+\w+\s*\(", code_snippet):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code_snippet):
            return "javascript"
        return "unknown"

    # ── Content classification ───────────────────────────────────────────

    def classify_content(self, content: str, extracted_blocks: list) -> "ContentType":
        """Classify a user/assistant message into one of the ContentType categories."""
        cl = content.lower()
        if self._f.diff_pattern.search(content) or "diff --git" in content:
            return ContentType.PROPOSED_CHANGE
        if self._f.commit_pattern.search(content):
            return (
                ContentType.COMMITTED_CHANGE
                if ("applied" in cl or "committed" in cl or "merged" in cl)
                else ContentType.PROPOSED_CHANGE
            )
        if (
            "traceback" in cl
            or ('file "' in cl and "line " in cl)
            or ("exception" in cl and ("traceback" in cl or 'file "' in cl))
        ):
            return ContentType.ERROR
        if '"tool_calls"' in content or '"function"' in content:
            return ContentType.TOOL_CALL
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
        return ContentType.GENERAL

    # ── Similarity & deduplication ───────────────────────────────────────

    def calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Compute structural (AST) similarity for Python, fallback to token-sort ratio."""
        if (
            self._f.valves.enable_ast_deduplication
            and len(code1) > 30
            and len(code2) > 30
        ):
            ast_sim = self._ast_similarity(code1, code2)
            if ast_sim is not None:
                return ast_sim
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))
        return fuzz.token_sort_ratio(code1, code2) / 100.0

    def _ast_similarity(self, code1: str, code2: str) -> Optional[float]:
        """Compute Jaccard similarity on AST node type distributions for Python code."""
        if not (
            re.search(r"\bdef\s+\w+\s*\(", code1) or re.search(r"\bclass\s+\w+", code1)
        ):
            return None
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            return None

        def _strip_docstrings(tree: ast.AST) -> ast.AST:
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
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
            return tree

        clean1 = _strip_docstrings(tree1)
        clean2 = _strip_docstrings(tree2)

        if ast.dump(clean1) == ast.dump(clean2):
            return 1.0

        def _node_type_counts(tree: ast.AST) -> Counter:
            return Counter(type(node).__name__ for node in ast.walk(tree))

        c1 = _node_type_counts(clean1)
        c2 = _node_type_counts(clean2)
        all_types = set(c1.keys()) | set(c2.keys())
        if not all_types:
            return 0.0
        intersection = sum(min(c1.get(t, 0), c2.get(t, 0)) for t in all_types)
        union = sum(max(c1.get(t, 0), c2.get(t, 0)) for t in all_types)
        return intersection / union if union > 0 else 0.0

    def remove_duplicate_blocks(self, state: dict, project_id: str) -> None:
        """Remove duplicate or near‑duplicate code blocks from the active set."""
        if not self._f.valves.auto_remove_duplicate_blocks:
            return
        blocks = list(state["active_blocks"].values())
        to_remove = set()
        for i, block in enumerate(blocks):
            if block.hash in to_remove or block.pinned or block.obsolete:
                continue
            for j, other in enumerate(blocks[i + 1 :], start=i + 1):
                if other.hash in to_remove or other.pinned or other.obsolete:
                    continue
                sim = self.calculate_code_similarity(block.content, other.content)
                if sim >= self._f.valves.code_similarity_threshold:
                    age_diff = abs(block.timestamp - other.timestamp) / 3600
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
                    score_diff = abs(block.importance_score - other.importance_score)
                    if score_diff < 1.0:
                        if block.timestamp >= other.timestamp:
                            to_remove.add(other.hash)
                        else:
                            to_remove.add(block.hash)
                    elif block.importance_score >= other.importance_score:
                        to_remove.add(other.hash)
                    else:
                        to_remove.add(block.hash)
        blocks_by_file = defaultdict(list)
        for b in blocks:
            if b.file_path and not b.pinned:
                blocks_by_file[b.file_path].append(b)
        for file_path, blks in blocks_by_file.items():
            if len(blks) > 1:
                blks.sort(key=lambda b: b.timestamp, reverse=True)
                for b in blks[1:]:
                    to_remove.add(b.hash)
        for h in to_remove:
            if h in state["active_blocks"]:
                block = state["active_blocks"][h]
                self._f._symbol_index.remove_all_for_block(
                    block.hash, block.symbols, project_id
                )
                del state["active_blocks"][h]
        state["recent_changes"] = [
            b for b in state["recent_changes"] if b.hash not in to_remove
        ]
        state["committed_changes"] = [
            b for b in state["committed_changes"] if b.hash not in to_remove
        ]
        if to_remove:
            state["has_any_calls"] = any(
                any(s.calls for s in b.symbols) for b in state["active_blocks"].values()
            )
            self._f._activation.invalidate_lightweight_cache(project_id)

    # ── File path extraction ─────────────────────────────────────────────

    def extract_file_paths(self, content: str) -> list:
        """Extract all file paths matching the configured pattern from content."""
        if not self._f.valves.track_file_paths:
            return []
        matches = re.findall(self._f.valves.file_path_pattern, content)
        return [m[0] if isinstance(m, tuple) else m for m in matches]

    def extract_file_path_for_block(
        self, content: str, block_start: int
    ) -> Optional[str]:
        """Try to find a file path associated with a code block by scanning backwards."""
        if block_start <= 0:
            return None
        before = content[:block_start]
        lines = before.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            match = re.search(self._f.valves.file_path_pattern, line)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
            file_path, _, _ = self._extract_line_range(line)
            if file_path:
                return file_path
            break
        return None

    def _extract_line_range(
        self, content: str
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
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

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove non-printable characters and replace backticks."""
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        cleaned = cleaned.replace("`", "'")
        return cleaned

    # ── Proposed changes & diffs ─────────────────────────────────────────

    def has_conflicting_proposed_changes(
        self, state: dict, new_block: "CodeBlock"
    ) -> bool:
        """Check if a proposed change conflicts with an existing recent change."""
        if new_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        for existing in state["recent_changes"]:
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

    def apply_change_with_diff(
        self, base_block: "CodeBlock", proposed_block: "CodeBlock"
    ) -> bool:
        """Apply a unified diff from a proposed change block onto a base block."""
        if proposed_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        if not (
            "@@" in proposed_block.content
            and ("-" in proposed_block.content or "+" in proposed_block.content)
        ):
            return False
        new_code = self._apply_unified_diff(base_block.content, proposed_block.content)
        if new_code and new_code != base_block.content:
            project_id = self._f.valves.project_id
            self._f._symbol_index.remove_all_for_block(
                base_block.hash, base_block.symbols, project_id
            )
            base_block.content = new_code
            base_block.hash = hashlib.md5(new_code.encode()).hexdigest()[:16]
            base_block.symbols = SignatureExtractor._extract_generic(
                new_code, base_block.file_path
            )
            for sym in base_block.symbols:
                sym.parent_block_hash = base_block.hash
                self._f._symbol_index.add(sym, base_block.hash, project_id)
            if self._f.tokenizer:
                base_block._cached_token_count = len(self._f.tokenizer.encode(new_code))
            else:
                base_block._cached_token_count = len(new_code) // 4
            base_block.timestamp = time.time()
            base_block.is_active = True
            base_block.potentially_affected = False
            base_block.importance_score = min(base_block.importance_score + 2.0, 10.0)
            self._f._activation.invalidate_lightweight_cache(project_id)
            return True
        return False

    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        """Apply a unified diff patch to original text. Returns new text or None."""
        if not self._f.valves.enable_diff_application:
            return None
        lines = original.splitlines(keepends=False)
        result_lines = lines[:]
        hunks = []
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
        applied_any = False
        for old_start_idx, old_lines, new_lines in reversed(hunks):
            if old_start_idx < 0 or old_start_idx + len(old_lines) > len(result_lines):
                logger.warning(
                    f"Unified diff hunk out of bounds (start={old_start_idx}, "
                    f"lines={len(old_lines)}, total={len(result_lines)})"
                )
                continue
            if (
                result_lines[old_start_idx : old_start_idx + len(old_lines)]
                != old_lines
            ):
                logger.warning(f"Unified diff hunk mismatch at line {old_start_idx}")
                continue
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

    # ── Data flow edges ──────────────────────────────────────────────────

    def extract_data_flow_edges(
        self, code: str, file_path: Optional[str], project_id: str
    ) -> List["Edge"]:
        """Extract data flow edges from Python code using ast, with regex fallback."""
        if not file_path or not file_path.endswith(".py"):
            return self._extract_data_flow_edges_regex(code, project_id)

        all_names = self._f._symbol_index.get_all_names(project_id)
        if not all_names:
            return []

        edges: List[Edge] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller_name = func_node.name
            if caller_name not in all_names:
                continue
            assigned_vars: Set[str] = set()
            for child in ast.walk(func_node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
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
                args_are_local_vars = any(
                    isinstance(arg, ast.Name) and arg.id in assigned_vars
                    for arg in child.args
                )
                if args_are_local_vars or child.args:
                    edges.append(
                        Edge(
                            src=caller_name,
                            dst=callee_name,
                            type="data_flow",
                            weight=EDGE_WEIGHTS["data_flow"],
                            confidence=0.7,
                        )
                    )
        return edges

    def _extract_data_flow_edges_regex(
        self, code: str, project_id: str
    ) -> List["Edge"]:
        """Fallback data flow extraction for non‑Python languages via regex."""
        all_names = self._f._symbol_index.get_all_names(project_id)
        edges: List[Edge] = []
        pattern = re.compile(
            r"\b(\w+)\s*=\s*(" + "|".join(re.escape(n) for n in all_names) + r")\s*\("
        )
        for match in pattern.finditer(code):
            callee = match.group(2)
            var_name = match.group(1)
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


class ActivationEngine:
    """PPR activation, path index, centrality, and speculative prefetch."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ═══════════════════════════════════════════════════════════════════════════
    # Active code context
    # ═══════════════════════════════════════════════════════════════════════════

    def get_active_code_context(self, project_id: str, user_query: str = "") -> str:
        """Return a formatted string with the currently active code context for the LLM."""
        state = self._f._state_store.get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        now = time.time()
        active = []
        for block in state["active_blocks"].values():
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
                parts.append(self._f._format_block_context(b, is_latest) + tag)
        if proposed:
            parts.append("### Proposed Changes (pending review):")
            for b in proposed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._f._format_block_context(b, is_latest) + tag)
        if committed:
            parts.append("### Recently Committed Changes:")
            for b in committed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._f._format_block_context(b, is_latest) + tag)
        if errors:
            parts.append("### Recent Errors:")
            for b in errors:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._f._format_block_context(b, is_latest) + tag)

        # Presupuesto dinámico (Fix 4 con guard contra negativo)
        effective_budget = max(
            4000,
            self._f.valves.context_window_tokens
            - self._f._last_system_tokens.get(project_id, 0)
            - self._f.valves.response_reserve_tokens,
        )
        max_tokens = min(
            self._f.valves.active_context_max_tokens or effective_budget,
            effective_budget,
        )

        # ── v2.0: Tokenización O(n) con pre‑cálculo de tamaños ──────────
        if max_tokens > 0 and self._f.tokenizer:
            part_sizes = [len(self._f.tokenizer.encode(p)) for p in parts]
            current_tokens = sum(part_sizes)

            while current_tokens > max_tokens and len(parts) > 2:
                excess = current_tokens - max_tokens

                # Buscar la parte más grande usando los tamaños pre‑calculados
                largest_idx = max(range(len(part_sizes)), key=lambda i: part_sizes[i])
                largest_tok = part_sizes[largest_idx]

                if largest_tok >= excess + 100:
                    target = max(100, largest_tok - excess - 50)
                    truncated_text = self._f._tokens.truncate_text_to_tokens(
                        parts[largest_idx], target
                    )
                    if self._has_open_fence(truncated_text):
                        truncated_text += "\n```"
                    parts[largest_idx] = truncated_text + "\n[...truncado...]"

                    # Actualizar tamaño y total sin re‑tokenizar todo
                    new_size = len(self._f.tokenizer.encode(parts[largest_idx]))
                    current_tokens = current_tokens - largest_tok + new_size
                    part_sizes[largest_idx] = new_size
                else:
                    # Eliminar la parte más pequeña (pop)
                    current_tokens -= part_sizes.pop()
                    parts.pop()

            if current_tokens > max_tokens:
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
    # Activation graph building
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

    def _build_single_seed_graph(
        self,
        exact_seeds: List[str],
        partial_seeds: List[str],
        tb_seeds: List[Tuple[str, float]],
        history_boosts: Dict[str, float],
        edges_out: dict,
        project_id: str,
    ) -> "ActivationGraph":
        """Build activation graph when multi‑seed activation is disabled."""
        ag = ActivationGraph()

        if exact_seeds:
            for sym_name in exact_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(1.0, 0.5 + 0.5 * min(specificity, 1.0))
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=score, depth=0, source="seed"
                )
        if partial_seeds:
            for sym_name in partial_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.6, 0.3 + 0.3 * min(specificity, 1.0))
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=score, depth=0, source="seed"
                )
        for sym_name, tb_score in tb_seeds:
            existing = ag._activations.get(sym_name)
            if existing:
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name,
                    score=min(1.0, existing.score + tb_score * 0.4),
                    depth=0,
                    source="seed",
                )
            else:
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=tb_score, depth=0, source="seed"
                )
        for sym_name, boost in history_boosts.items():
            existing = ag._activations.get(sym_name)
            if existing:
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name,
                    score=min(1.0, existing.score + boost),
                    depth=0,
                    source=existing.source,
                )
            else:
                ag._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=boost, depth=0, source="seed"
                )

        if not ag._activations:
            entry_points = self._f._path_index.find_entry_points(
                self._f._symbol_index, project_id
            )
            if entry_points:
                centrality = self._f._node_centrality.get(project_id, {})
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
    ) -> "ActivationGraph":
        """Build activation graph combining lexical, structural and historical seed vectors."""
        w_lex = self._f.valves.multi_seed_weight_lexical
        w_str = self._f.valves.multi_seed_weight_structural
        w_his = self._f.valves.multi_seed_weight_historical

        # ── Vector 1: Lexical ──────────────────────────────────────────
        ag_lex = ActivationGraph()
        if exact_seeds:
            for sym_name in exact_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(1.0, 0.5 + 0.5 * min(specificity, 1.0))
                ag_lex._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=score, depth=0, source="seed"
                )
        if partial_seeds:
            for sym_name in partial_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.6, 0.3 + 0.3 * min(specificity, 1.0))
                ag_lex._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=score, depth=0, source="seed"
                )
        for sym_name, tb_score in tb_seeds:
            existing = ag_lex._activations.get(sym_name)
            if existing:
                ag_lex._activations[sym_name] = ActivationState(
                    node_id=sym_name,
                    score=min(1.0, existing.score + tb_score * 0.4),
                    depth=0,
                    source="seed",
                )
            else:
                ag_lex._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=tb_score, depth=0, source="seed"
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
        lexical_seed_names = set(exact_seeds) | {s for s, _ in tb_seeds}
        structural_seeds: Set[str] = set()
        for view in self._f._path_index.get_all(project_id):
            for lex_seed in lexical_seed_names:
                if lex_seed in view.induced_nodes:
                    structural_seeds.add(view.entry_point)
                    break
        if structural_seeds:
            for sym_name in structural_seeds:
                specificity = self._compute_node_specificity(sym_name, project_id)
                score = min(0.8, 0.5 * min(specificity, 1.4))
                ag_str._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=score, depth=0, source="seed"
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
                ag_his._activations[sym_name] = ActivationState(
                    node_id=sym_name, score=boost, depth=0, source="seed"
                )
            ag_his.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.03,
                alpha=self._f.valves.ppr_alpha,
            )

        # ── Combine the three vectors ──────────────────────────────────
        all_activated = (
            set(ag_lex.get_activated_nodes(0.01).keys())
            | set(ag_str.get_activated_nodes(0.01).keys())
            | set(ag_his.get_activated_nodes(0.01).keys())
        )

        ag_final = ActivationGraph()
        if not all_activated:
            entry_points = self._f._path_index.find_entry_points(
                self._f._symbol_index, project_id
            )
            if entry_points:
                centrality = self._f._node_centrality.get(project_id, {})
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
                    source = "seed" if node in lexical_seed_names else "propagation"
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

    def build_activation_graph(
        self,
        query: str,
        project_id: str,
        max_propagation_steps: int = 4,
        messages: Optional[List[dict]] = None,
    ) -> "ActivationGraph":
        """
        Build an ActivationGraph combining up to three independent seed vectors.

        Delegates seed extraction and the single‑seed / multi‑seed construction
        to private helpers, keeping the top‑level logic easy to read.
        """
        edges_out = self._f._symbol_index.get_all_edges_out(project_id)

        # 1. Extract all seed symbols from the query and history
        exact_seeds, partial_seeds, tb_seeds, history_boosts = (
            self._prepare_seed_symbols(query, project_id, messages)
        )

        # 2. Build the activation graph in the appropriate mode
        if not self._f.valves.enable_multi_seed_activation:
            ag = self._build_single_seed_graph(
                exact_seeds,
                partial_seeds,
                tb_seeds,
                history_boosts,
                edges_out,
                project_id,
            )
        else:
            ag = self._build_multi_seed_graph(
                exact_seeds,
                partial_seeds,
                tb_seeds,
                history_boosts,
                edges_out,
                project_id,
            )

        # 3. Store scores for downstream consumers (LOD, prefetch, pager)
        self._store_activation_scores(ag, project_id)
        return ag

    # ═══════════════════════════════════════════════════════════════════════════
    # Path index & cross‑chunk edges
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
        state = self._f._state_store.get_state(project_id)
        if not state or not state.get("active_blocks"):
            return
        entry_points = self._f._path_index.find_entry_points(
            self._f._symbol_index, project_id
        )
        for ep in entry_points:
            ag = self.build_activation_graph(ep, project_id)
            await self._build_view_from_activation(ep, ag, project_id)

        if self._f.valves.enable_centrality_prior:
            self._f._node_centrality[project_id] = (
                self._f._symbol_index.precompute_centrality(project_id)
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
    # Speculative prefetch
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
            ag = ActivationGraph()
            ag.seed([sym_name], initial_score=1.0)
            ag.propagate(edges_out, max_steps=2, min_score=0.1)
            await self._build_view_from_activation(sym_name, ag, project_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # Hash utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_structural_hash(
        self, symbol_names: Iterable[str], project_id: str
    ) -> str:
        """Hash of the symbols' content blocks (changes when code changes)."""
        state = self._f._state_store.get_state(project_id)
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
        if self._f._cached_code_state_hash is not None:
            return self._f._cached_code_state_hash
        state = self._f._state_store.get_state(project_id)
        h = self._compute_code_state_hash_from_state(state)
        self._f._cached_code_state_hash = h
        return h

    def _compute_code_state_hash_from_state(self, state: dict) -> str:
        if not state or not state.get("active_blocks"):
            return ""
        sorted_hashes = sorted(
            h for h, b in state["active_blocks"].items() if not b.obsolete
        )
        return hashlib.md5("|".join(sorted_hashes).encode()).hexdigest()[:16]

    def compute_context_hash(self, messages: list) -> str:
        """Hash of system message content, used for response cache keying."""
        if not self._f.valves.response_cache_include_context_hash:
            return ""
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        context_str = "\n".join([m.get("content", "") for m in sys_msgs])
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════════════════════════════
    # Inactive block candidates & cache invalidation
    # ═══════════════════════════════════════════════════════════════════════════

    def get_inactive_block_candidates(self, project_id: str) -> list:
        """
        Return block hashes that haven't been mentioned in the last
        `cleanup_inactive_threshold_messages` messages, excluding pinned,
        obsolete, and content types listed in `cleanup_excluded_content_types`.
        """
        state = self._f._state_store.get_state(project_id)
        if not state or not state.get("active_blocks"):
            return []
        threshold = self._f.valves.cleanup_inactive_threshold_messages
        excluded_types = set(self._f.valves.cleanup_excluded_content_types)
        current_msg_idx = state["message_count"]
        candidates = []
        for h, block in state["active_blocks"].items():
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
        self._f._cached_lightweight_context.pop(project_id, None)
        self._f._cached_code_state_hash = None
        self._f._node_centrality.pop(project_id, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Static evidence
    # ═══════════════════════════════════════════════════════════════════════════

    def _gather_static_evidence(
        self, hypothesis_text: str, project_id: str
    ) -> "StaticEvidence":
        """
        Gather deterministic evidence about the structural claims in a hypothesis.
        No LLM. No GPU. Instant.
        """
        all_names = self._f._symbol_index.get_all_names(project_id)
        state = self._f._state_store.get_state(project_id)

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
                state["active_blocks"].get(bh) is not None
                and (now - state["active_blocks"][bh].timestamp) < recent_window
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
    """Code history compression, message summarisation, and block summaries."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ── Public API ────────────────────────────────────────────────────────
    def compress_code_history(self, messages: list, project_id: str) -> list:
        """
        Replace old assistant code-part messages with compact commit summaries.

        Compression pipeline:
          1. Find all assistant messages with multi-phase code part headers.
          2. Keep the last `code_history_keep_last_n_parts` in full.
          3. For each older part, verify symbols are indexed in the SymbolGraph.
          4. If indexed (ratio >= threshold): replace with commit summary.
          5. If NOT indexed: keep full (defensive — never lose code that isn't
             safely retrievable from the graph).
        """
        if not self._f.valves.enable_code_history_compression:
            return messages

        _PART_HEADER = re.compile(r"##\s*Código\s*[—\-]\s*Parte\s*(\d+)/(\d+)")
        _ALREADY_COMPRESSED = re.compile(r"\[🗜️ PARTE \d+/\d+")
        keep = self._f.valves.code_history_keep_last_n_parts

        # Collect indices of uncompressed code-part messages
        code_part_indices: List[Tuple[int, int, int]] = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if _ALREADY_COMPRESSED.search(content):
                continue  # already compressed, skip
            m = _PART_HEADER.search(content)
            if m:
                code_part_indices.append((i, int(m.group(1)), int(m.group(2))))

        if len(code_part_indices) <= keep:
            return messages  # Nothing old enough to compress

        to_compress = code_part_indices[:-keep]
        new_messages = list(messages)
        compressed_n = 0

        for msg_idx, part_num, total_parts in to_compress:
            msg = new_messages[msg_idx]
            content = msg.get("content", "")

            safe, ratio = self._verify_code_symbols_indexed(content, project_id)
            if not safe:
                self._f._log_debug(
                    f"Code history: skipping compression of Part {part_num}/{total_parts} "
                    f"(symbol ratio {ratio:.0%} < threshold "
                    f"{self._f.valves.code_history_symbol_index_threshold:.0%})"
                )
                continue

            summary = self._build_code_commit_summary(
                content, project_id, part_num, total_parts
            )
            tokens_before = self._f._tokens.estimate_tokens(content)
            tokens_after = self._f._tokens.estimate_tokens(summary)
            new_messages[msg_idx] = {**msg, "content": summary}
            compressed_n += 1
            self._f._log_debug(
                f"Code history: compressed Part {part_num}/{total_parts} — "
                f"{tokens_before:,} → {tokens_after:,} tokens "
                f"(ratio {ratio:.0%})"
            )

        if compressed_n:
            self._f._log_debug(
                f"Code history: {compressed_n} part(s) compressed, "
                f"last {keep} kept in full."
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

    def _build_code_commit_summary(
        self,
        content: str,
        project_id: str,
        part_num: int,
        total_parts: int,
    ) -> str:
        """Generate a compact commit summary for a compressed code message."""
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

    def lean_user_code_messages(self, messages: list, project_id: str) -> list:
        """
        Replace code blocks in user messages with compressed stubs when the
        SymbolGraph already indexes the symbols, and the code exceeds the
        minimum token threshold.
        """
        if not self._f.valves.enable_lean_user_code:
            return messages

        try:
            symbol_count = len(self._f._symbol_index.get_all_names(project_id))
        except Exception:
            symbol_count = 0

        if symbol_count < 20:
            self._f._log_debug(
                f"Lean user code: SymbolGraph too sparse ({symbol_count} symbols) — skipping."
            )
            return messages

        _CODE_BLOCK = re.compile(r"```(?P<lang>\w*)\n(?P<body>.*?)```", re.DOTALL)
        _ALREADY_LEAN = "[CÓDIGO COMPRIMIDO"
        min_tokens = self._f.valves.lean_user_code_min_tokens

        new_messages = list(messages)
        lean_n = 0

        for i, msg in enumerate(new_messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if _ALREADY_LEAN in content:
                continue

            total_code_tokens = sum(
                self._f._tokens.estimate_tokens(m.group("body"))
                for m in _CODE_BLOCK.finditer(content)
            )
            if total_code_tokens < min_tokens:
                continue

            def _replace(match: re.Match) -> str:
                body = match.group("body")
                lang = match.group("lang") or "code"
                blk_tokens = self._f._tokens.estimate_tokens(body)
                if blk_tokens < min_tokens:
                    return match.group(0)
                return (
                    f"```{lang}\n"
                    f"[CÓDIGO COMPRIMIDO — {blk_tokens:,} tokens — "
                    f"{symbol_count} símbolos indexados en SymbolGraph. "
                    f"Recuperar implementaciones con /expand <nombre> o via LOD.]\n"
                    f"```"
                )

            new_content = _CODE_BLOCK.sub(_replace, content)
            if new_content != content:
                new_messages[i] = {**msg, "content": new_content}
                lean_n += 1
                self._f._log_debug(
                    f"Lean user code: message {i} — replaced ~{total_code_tokens:,} tokens "
                    f"({symbol_count} symbols available via LOD)"
                )

        if lean_n:
            self._f._log_debug(f"Lean user code: applied to {lean_n} message(s).")

        return new_messages

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
    """Token estimation and text truncation helpers."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

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
    """Post‑processing enrichment: summaries, feedback, LOD adaptation."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

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

            await self._f._db_enqueue(_write)

    async def run_missing_summaries_task(self, params: dict, model: str) -> bool:
        """Generate a missing summary for one symbol."""
        signature = params["signature"]
        code_snippet = params["code_snippet"]
        prompt = (
            f"Summarize in one short sentence what this code does:\n\n"
            f"```{signature}\n{code_snippet}```"
        )
        summary = await self._f._llm_orchestrator.call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant. Output only one concise sentence.",
            model_override=model,
            max_tokens=50,
            temperature=0.1,
            label="missing_summaries",
        )
        if summary and summary.strip():
            project_id = params["project_id"]
            lock = await self._f._state_store.get_project_lock(project_id)
            async with lock:
                state = self._f._state_store.get_state(project_id)
                for blk in state["active_blocks"].values():
                    for sym in blk.symbols:
                        if sym.signature == signature:
                            sym.summary = summary.strip()
                self._f._state_store.set_state(project_id, state)
            return True
        return False

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
            block = state["active_blocks"].get(block_hash)
            if block:
                block.mention_count += 1
                block.last_mentioned = time.time()
                block.last_mentioned_msg_idx = state["message_count"]
                block._update_importance()

    async def expire_blocks_by_time(self, project_id: str) -> None:
        """Remove blocks that have not been mentioned recently, based on configured timeouts."""
        lock = await self._f._state_store.get_project_lock(project_id)
        async with lock:
            state = self._f._state_store.get_state(project_id)
            if not state:
                return
            now = time.time()
            expiration_seconds = self._f.valves.block_expiration_hours * 3600
            to_remove = []
            for h, block in state["active_blocks"].items():
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
                if h in state["active_blocks"]:
                    block = state["active_blocks"][h]
                    self._f._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                del state["active_blocks"][h]
            if to_remove:
                state["has_any_calls"] = any(
                    any(s.calls for s in b.symbols)
                    for b in state["active_blocks"].values()
                )
                self._f._activation.invalidate_lightweight_cache(project_id)
                self._f._state_store.set_state(project_id, state)

    async def update_lod_thresholds_from_response(
        self,
        project_id: str,
        response_text: str,
    ) -> None:
        """
        Adjust lod3_threshold based on which symbols appear in the LLM's
        response compared to the LOD level they received.

        Logic:
        - If the LLM mentions symbols that only got LOD ≤ 2 (summary/signature):
          → Lower lod3_threshold: give more full code next time.
        - If the LLM does NOT mention symbols that got LOD 3 (full code):
          → Raise lod3_threshold slightly: those expansions were unnecessary.

        Adjustments are small (lod_adapt_rate) and bounded [min, max].
        Threshold state is NOT persisted across server restarts.
        """
        if not self._f.valves.enable_lod_adaptive:
            return

        last_lod_map = getattr(self._f, "_last_lod_levels", {}).get(project_id, {})
        if not last_lod_map:
            return

        all_names = self._f._symbol_index.get_all_names(project_id)
        response_words = set(re.findall(r"\b\w+\b", response_text))
        referenced = all_names.intersection(response_words)

        # Symbols mentioned in the response that only received summary/signature
        underserved = [sym for sym in referenced if last_lod_map.get(sym, 3) < 3]

        # Symbols that got full code but do not appear in the response
        overserved = [
            sym
            for sym in last_lod_map
            if last_lod_map[sym] == 3 and sym not in referenced
        ]

        old_threshold = self._f.valves.lod3_threshold
        changed = False

        if len(underserved) >= self._f.valves.lod_adapt_underserved_min:
            # Lower threshold → more full code next time
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
            # Raise threshold → fewer unnecessary expansions
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

    def get_feedback_context(self, project_id: str) -> str:
        """Return a formatted string of recent feedback for the given project."""
        state = self._f._state_store.get_state(project_id)
        feedback = state.get("feedback_history", [])
        if not feedback:
            return ""
        recent = feedback[-self._f.valves.feedback_history_limit :]
        lines = ["## Previous Feedback"]
        for fb in recent:
            success = "✅" if fb.success else "❌"
            lines.append(f"- {success} {fb.change_description[:100]}")
        return "\n".join(lines)

    async def parallel_context_checks(
        self,
        messages: list,
        query: str,
        context_hash: str,
        project_id: str,
        state: dict,
        skip_contradiction: bool = False,
        skip_cache: bool = False,
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
                if (self._f.valves.duplicate_question_threshold and HAS_SENTENCE)
                else asyncio.sleep(0, result=None)
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        contradiction = results[0] if not isinstance(results[0], Exception) else None
        cached = results[1] if not isinstance(results[1], Exception) else None
        duplicate = results[2] if not isinstance(results[2], Exception) else None
        return contradiction, cached, duplicate


class ActiveCodeUpdater:
    """
    Processes a new user/assistant message: extracts code, updates
    active_blocks, runs symbol extraction, handles duplicates, diffs,
    and triggers post‑update tasks (expiration, enrichment, soft‑eviction).
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ── Public API ────────────────────────────────────────────────────────

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
        state_before = self._f._state_store.get_state(project_id)

        # 3. Detect duplicates
        duplicate_info = self._detect_duplicates(new_blocks_pending, state_before)

        async with lock:
            state = self._f._state_store.get_state(project_id)

            # 4. Housekeeping
            self._f._enrichment.update_mentions_from_message(state, content, project_id)
            for block in state["active_blocks"].values():
                if (
                    block.content
                    and self._f._code_blocks.calculate_code_similarity(
                        block.content[:200], content[:200]
                    )
                    > 0.7
                ):
                    block.mention_count += 1
                    block.last_mentioned = time.time()
                    block.last_mentioned_msg_idx = state["message_count"]
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
                    state["active_blocks"].get(existing_hash) if existing_hash else None
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

            self._f._state_store.set_state(project_id, state)

    # ── Private helpers (called in order by `process`) ────────────────────

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
        for blk in new_blocks_pending:
            syms = await SignatureExtractor.extract_async(blk.content, blk.file_path)
            symbols_list.append(syms)

        content_to_syms: Dict[str, List[CodeSymbol]] = {
            blk.content: syms
            for blk, syms in zip(new_blocks_pending, symbols_list)
            if not isinstance(syms, Exception)
        }

        return new_blocks_pending, symbols_list, content_to_syms, extracted_blocks

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

        existing_contents = {
            h: b.content for h, b in state.get("active_blocks", {}).items()
        }

        for new_block in new_blocks:
            is_dup = False
            existing_dup = None
            for h, ex_content in existing_contents.items():
                ex_block = state["active_blocks"].get(h)
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
            existing.last_mentioned_msg_idx = state["message_count"]
            existing.pinned = True
            existing.is_raw = existing.is_raw or new_block.is_raw
            existing.importance_score = 10.0
            existing.symbols = syms
            await self._reindex_block_symbols(existing, project_id)
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
            existing.last_mentioned_msg_idx = state["message_count"]
            existing.symbols = syms
            await self._reindex_block_symbols(existing, project_id)
            if prev_content != new_block.content:
                await self._f._enrichment.generate_change_summary(
                    existing.hash, prev_content, new_block.content
                )

    async def _reindex_block_symbols(self, block: "CodeBlock", project_id: str) -> None:
        """Re‑extract symbols for a block and register them + edges in the index."""
        for s in block.symbols:
            s.parent_block_hash = block.hash
            self._f._symbol_index.add(s, block.hash, project_id)
            for callee_name in s.calls:
                edge = Edge(
                    src=s.name,
                    dst=callee_name,
                    type="calls",
                    weight=EDGE_WEIGHTS["calls"],
                    confidence=1.0,
                )
                self._f._symbol_index.add_edge(edge, project_id)
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
        """
        for sym in syms:
            sym.parent_block_hash = new_block.hash
        new_block.symbols = syms
        new_block.last_mentioned_msg_idx = state["message_count"]

        # Index symbols and edges
        for sym in syms:
            self._f._symbol_index.add(sym, new_block.hash, project_id)
            for callee_name in sym.calls:
                edge = Edge(
                    src=sym.name,
                    dst=callee_name,
                    type="calls",
                    weight=EDGE_WEIGHTS["calls"],
                    confidence=1.0,
                )
                self._f._symbol_index.add_edge(edge, project_id)
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
            state["has_any_calls"] = True

        # Check for conflicting proposed changes
        is_conflicting = False
        if new_block.content_type == ContentType.PROPOSED_CHANGE:
            is_conflicting = self._f._code_blocks.has_conflicting_proposed_changes(
                state, new_block
            )
            if is_conflicting:
                new_block.importance_score = max(new_block.importance_score, 7.0)

        state["active_blocks"][new_block.hash] = new_block

        # Mark older blocks for the same file as obsolete
        if new_block.file_path and self._f.valves.enable_obsolete_marking:
            for h, blk in list(state["active_blocks"].items()):
                if h == new_block.hash:
                    continue
                if blk.file_path == new_block.file_path and not blk.pinned:
                    blk.obsolete = True
                    blk._update_importance()
                    # bug #9 fix: remove symbols of the now‑obsolete block from the index
                    self._f._symbol_index.remove_all_for_block(
                        blk.hash, blk.symbols, project_id
                    )

        # Handle content-type specific actions
        if new_block.content_type == ContentType.PROPOSED_CHANGE:
            if new_block.file_path:
                state["recent_changes"] = [
                    c
                    for c in state["recent_changes"]
                    if not (
                        c.file_path
                        and c.file_path == new_block.file_path
                        and c.hash != new_block.hash
                    )
                ]
            state["recent_changes"].append(new_block)
            if self._f.valves.enable_diff_application and not is_conflicting:
                for base in list(state["active_blocks"].values()):
                    if (
                        base.content_type == ContentType.BASE_CODE
                        and base.file_path == new_block.file_path
                    ):
                        if self._f._code_blocks.apply_change_with_diff(base, new_block):
                            state["recent_changes"] = [
                                c
                                for c in state["recent_changes"]
                                if c.hash != new_block.hash
                            ]
                            state["committed_changes"].append(new_block)
                            break
        elif new_block.content_type == ContentType.COMMITTED_CHANGE:
            state["committed_changes"].append(new_block)
        elif (
            new_block.content_type == ContentType.ERROR
            and self._f.valves.preserve_error_context
        ):
            new_block.importance_score = min(new_block.importance_score + 3.0, 10.0)

        # Hard eviction if too many active blocks
        if (
            self._f.valves.max_active_blocks > 0
            and len(state["active_blocks"]) > self._f.valves.max_active_blocks
        ):
            sorted_blocks = sorted(
                state["active_blocks"].values(),
                key=lambda b: b.importance_score
                + (self._f.valves.raw_file_priority_boost if b.is_raw else 0),
                reverse=True,
            )
            keep_hashes = {
                b.hash for b in sorted_blocks[: self._f.valves.max_active_blocks]
            }
            to_remove = [h for h in state["active_blocks"] if h not in keep_hashes]
            for h in to_remove:
                block = state["active_blocks"].get(h)
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
                        del state["active_blocks"][h]
                        continue
                # Fallback: remove without paging
                if h in state["active_blocks"]:
                    del state["active_blocks"][h]
            if to_remove:
                self._f._log_debug(
                    f"Evicted {len(to_remove)} blocks due to max_active_blocks limit. "
                    f"Their symbols remain in the index for lightweight context."
                )

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
            for base in state["active_blocks"].values():
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
                    state["has_any_calls"] = True
                if prev_content != block_info["code"]:
                    await self._f._enrichment.generate_change_summary(
                        best_base.hash, prev_content, block_info["code"]
                    )

    async def _post_update_tasks(
        self,
        state: dict,
        project_id: str,
        new_blocks_pending: List["CodeBlock"],
        is_continuation: bool,
    ) -> None:
        """Expiration, enrichment, oversized‑block summaries, path index, soft eviction."""
        if not is_continuation:
            state["message_count"] += 1
        if self._f.valves.auto_remove_duplicate_blocks:
            self._f._code_blocks.remove_duplicate_blocks(state, project_id)

        # Inline block expiration
        await self._f._enrichment.expire_blocks_by_time(project_id)

        # Enrichment tasks – run sequentially
        tasks_to_run = []
        max_tasks_per_type = 5

        for block in list(state["active_blocks"].values()):
            if block.obsolete:
                continue
            if self._f.valves.enable_auto_summaries:
                syms_without_summary = [
                    s
                    for s in block.symbols
                    if not s.summary and s.kind in ("function", "method")
                ]
                for sym in syms_without_summary[:3]:
                    tasks_to_run.append(
                        (
                            "missing_summaries",
                            {
                                "signature": sym.signature,
                                "code_snippet": block.content[:500],
                                "project_id": project_id,
                            },
                        )
                    )
                    if len(tasks_to_run) >= max_tasks_per_type:
                        break

        for task_type, params in tasks_to_run:
            try:
                if task_type == "missing_summaries":
                    await self._f._enrichment.run_missing_summaries_task(
                        params, self._f.valves.llm_model
                    )
            except Exception as e:
                self._f._log_debug(f"Immediate enrichment task {task_type} failed: {e}")

        if self._f.valves.enable_session_summary and not is_continuation:
            interval = self._f.valves.session_summary_interval_messages
            if (
                interval > 0
                and state["message_count"] % interval == 0
                and state["message_count"] > 0
            ):
                await self._f._enrichment.run_session_summary_task(
                    {
                        "project_id": project_id,
                        "message_count": state["message_count"],
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
            for block in state["active_blocks"].values():
                if not block.obsolete:
                    await self._f._history_compressor.maybe_generate_block_summary(
                        block
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
                block = state["active_blocks"].get(hash_)
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
                        del state["active_blocks"][hash_]
            if candidates:
                self._f._log_debug(
                    f"Soft-evicted {len(candidates)} block(s) via ContextPager "
                    f"(active_blocks now {len(state['active_blocks'])})"
                )


class InletOrchestrator:
    """Inlet pre‑processing, user info extraction, session classification."""

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    def get_project_id(self) -> str:
        """Return the current project id from the valves configuration."""
        return self._f.valves.project_id

    async def inlet_preprocess(self, body: dict, project_id: str) -> list:
        """Handle project switching, symbol cache loading, and KV slot restore."""
        messages = body.get("messages", [])

        if self._f._last_project_id and self._f._last_project_id != project_id:
            self._f._log_debug(
                f"Project changed from {self._f._last_project_id} to {project_id}"
            )
            old_state = self._f._conversation_state.get(self._f._last_project_id)
            if old_state:
                self._f._symbol_index.clear_project(self._f._last_project_id)
            self._f._cached_lightweight_context.pop(self._f._last_project_id, None)
            self._f._block_change_summaries.clear()
        self._f._last_project_id = project_id

        # ── v7 (PASO-15): load persisted CodePathViews if index is empty ──
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

        # ── v7 Phase 5 (PASO-28): restore typed edges from DB ─────────
        if self._f.valves.enable_edge_persistence:
            restored = await self._f._state_store.load_symbol_edges_from_db(project_id)
            if restored > 0:
                self._f._log_debug(
                    f"Cross-session: {restored} symbol edges restored from DB. "
                    f"No need to re-paste code."
                )

        return messages

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

        return (
            last_user_msg,
            user_query,
            user_question,
            is_explicit_command,
            has_code_blocks,
        )

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
            self._f._last_processed_message_idx[project_id] = last_idx
        else:
            user_question = user_query

        state = self._f._state_store.get_state(project_id)
        if not isinstance(state.get("active_blocks"), dict):
            self._f._log_debug(
                "CRITICAL: active_blocks corrupted even after load; resetting to empty. "
                "Delete %s if this recurs." % self._f.valves.state_db_path
            )
            state["active_blocks"] = {}
            self._f._state_store.set_state(project_id, state)

        return is_code_session, user_question

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

        state = self._f._state_store.get_state(project_id)
        if state and state.get("active_blocks"):
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
            and not state.get("active_blocks")
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


class SystemPromptBuilder:
    """
    Builds the system prompt in two blocks:
      - Block A (static, via ContextBuilder)
      - Block B (dynamic: LTM, parallel checks, activated code, suggestions,
        persisted summaries)
    Returns (static_block, dynamic_injections, cached_response, prelim_system).
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref

    # ── Public API ────────────────────────────────────────────────────────

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
    ) -> Tuple[str, List[Tuple[str, str]], Optional[dict], str]:
        """
        Orchestrate the construction of the two-block system prompt.
        Returns (static_block, dynamic_injections, cached_response, prelim_system).
        """
        # ══════════════════════════════════════════════════════════════
        # BLOCK A — STATIC (via ContextBuilder)
        # ══════════════════════════════════════════════════════════════
        self._f._log_debug("🧱 Block A (static): building / retrieving from cache")
        static_block = await self._f._ctx_builder.build_block_a(
            project_id=project_id,
            is_code_session=is_code_session,
            is_continuation=not slot_free,
        )

        # ══════════════════════════════════════════════════════════════
        # BLOCK B — DYNAMIC (per-query)
        # ══════════════════════════════════════════════════════════════
        dynamic_injections: List[Tuple[str, str]] = []

        # B1: LTM retrieval
        self._f._log_debug("🔄 Block B – Step 1/5: LTM per-query retrieval")
        ltm_text = await self._build_ltm_injection(
            project_id, user_question, user_query, is_code_session, slot_free
        )
        if ltm_text:
            dynamic_injections.append(("high", ltm_text))

        # B2: Parallel checks (contradiction, cache, duplicate)
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

        # B3: Activated code (per-query, via ContextBuilder)
        self._f._log_debug("🔄 Block B – Step 3/5: Code activated by query")
        active_ctx = await self._build_activated_code(
            user_query, project_id, messages, is_code_session, slot_free
        )
        if active_ctx:
            dynamic_injections.append(("critical", active_ctx))

        # B4: Proactive suggestions + command suggestions
        self._f._log_debug("🔄 Block B – Step 4/5: Proactive suggestions")
        for prio, text in await self._build_suggestions(
            state, project_id, messages, is_code_session
        ):
            dynamic_injections.append((prio, text))

        # B5: Assemble prelim_system (budget-aware)
        self._f._log_debug("🔄 Block B – Step 5/5: Assemble prelim_system")
        prelim_system = self._assemble_prelim_system(
            static_block, dynamic_injections, messages
        )

        self._f._log_debug("🔄 Block B: complete")
        return static_block, dynamic_injections, None, prelim_system

    # ── Private helpers ───────────────────────────────────────────────────

    async def _build_ltm_injection(
        self,
        project_id: str,
        user_question: str,
        user_query: str,
        is_code_session: bool,
        slot_free: bool,
    ) -> Optional[str]:
        """Retrieve and format relevant LTM entries for the current query, RAPTOR‑first."""
        if not (
            self._f.valves.enable_code_awareness
            and is_code_session
            and not self._f.valves.smart_context_selection
            and HAS_SENTENCE
            and HAS_CHROMA
        ):
            return None

        _ltm_query = user_question if user_question else user_query

        # ── v2.0: RAPTOR‑first retrieval ────────────────────────────
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

        all_meta = await self._f._ltm.retrieve_memories_unified(
            refined_query, project_id, slot_free=slot_free
        )
        if not all_meta:
            return None

        all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
        unique_meta = []
        seen_docs: Set[str] = set()
        for m in all_meta:
            if m["doc"] not in seen_docs:
                seen_docs.add(m["doc"])
                unique_meta.append(m)

        max_ltm = self._f.valves.ltm_retrieval_max_tokens
        parts: List[str] = []
        current_tokens = 0
        header = "## Relevant Past Context\n\n"
        for mem in unique_meta:
            ts = mem.get("timestamp")
            if ts and ts > 1_000_000_000:
                time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%SZ"
                )
                text = f"[{time_str}] {mem['doc']}"
            else:
                text = f"[unknown date] {mem['doc']}"
            frag_tok = self._f._tokens.estimate_code_tokens(text)
            if max_ltm > 0 and current_tokens + frag_tok > max_ltm:
                continue
            parts.append(text)
            current_tokens += frag_tok

        if not parts:
            return None
        return header + "\n---\n".join(parts)

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
            return active_ctx or self._f._activation.get_active_code_context(
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
                last_sugg_idx = state.get("last_cleanup_suggestion_msg_idx", 0)
                if (
                    state["message_count"] - last_sugg_idx
                    >= self._f.valves.cleanup_suggestion_cooldown_messages
                ):
                    suggestions.append(
                        (
                            "low",
                            f"[CodeAware] {len(candidates)} inactive block(s). "
                            f"Use `/status` or `/clean`.",
                        )
                    )
                    state["last_cleanup_suggestion_msg_idx"] = state["message_count"]
                    self._f._state_store.set_state(project_id, state)

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
        summaries = state.get("conversation_summaries", [])
        if summaries:
            joined = "\n\n".join(
                f"[Summary of earlier conversation — "
                f"{datetime.fromtimestamp(s['created_at'], tz=timezone.utc):%Y-%m-%d %H:%M}]"
                f"\n{s['text']}"
                for s in summaries
            )
            suggestions.append(("medium", joined))

        return suggestions

    def _assemble_prelim_system(
        self,
        static_block: str,
        dynamic_injections: List[Tuple[str, str]],
        messages: List[dict],
    ) -> str:
        """Apply token budget constraints and build the preliminary system prompt."""
        sys_msgs = [m for m in messages if m.get("role") == "system"]

        budget = self._f.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self._f.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected: List[str] = []
            used = 0
            static_tokens = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            remaining_budget = max(0, budget - static_tokens)
            for prio, text in dynamic_injections:
                if not text:
                    continue
                tok = len(self._f.tokenizer.encode(text))
                if used + tok <= remaining_budget:
                    selected.append(text)
                    used += tok
                elif prio in ("critical", "high"):
                    avail = remaining_budget - used
                    if avail > 20:
                        selected.append(text[: avail * 4] + "\n[truncated]")
                        break
            dynamic_block = "\n\n".join(selected)
        else:
            dynamic_block = "\n\n".join(t for _, t in dynamic_injections if t)

        separator = "\n\n---\n\n" if static_block and dynamic_block else ""
        prelim_system = static_block + separator + dynamic_block

        base_content = sys_msgs[0].get("content", "") if sys_msgs else ""
        if base_content.strip():
            prelim_system = prelim_system + "\n\n" + base_content

        return prelim_system


class MessageAssembler:
    """
    Post‑processes the final message list: CoT detection & generation,
    history LLMLingua compression, multi‑phase instructions, adaptive
    trimming, and final system‑message injection.
    Returns the final list of messages ready for the LLM.
    """

    def __init__(self, filter_ref: "Filter") -> None:
        self._f = filter_ref
        self._last_cot_degraded: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    async def assemble(
        self,
        messages: List[dict],
        project_id: str,
        static_block: str,
        dynamic_injections: List[Tuple[str, str]],
        prelim_system: str,
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: dict,
        __user__: Optional[dict],
        user_question: str,
        has_code_blocks: bool,
        slot_free: bool = True,
    ) -> List[dict]:
        """Orchestrate CoT, multi‑phase, trimming, and final assembly."""
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
            messages,  # ← bug #6 fix
        )

        # 2. History LLMLingua compression (optional)
        messages = await self._apply_history_llmlingua(
            messages, project_id, user_question
        )

        # 3. Multi-phase instructions injection
        await self._inject_multi_phase_instructions(
            dynamic_injections,
            prelim_system,
            messages,
            user_question,
            slot_free,
        )

        # 4. Trim and summarize old messages
        messages, pending_summary = await self._trim_and_summarize(
            messages, state, project_id, __user__
        )

        # 5. Code history compression and lean user code
        messages = await self._compress_code_history_and_lean(
            messages, project_id, dynamic_injections
        )

        # 6. Assemble final system message and inject into message list
        messages = self._assemble_final_system_and_log(
            static_block, dynamic_injections, messages, project_id, pending_summary
        )

        return messages

    # ── Private helpers ───────────────────────────────────────────────────

    async def _detect_and_generate_cot(
        self,
        dynamic_injections: List[Tuple[str, str]],
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: dict,
        user_question: str,
        prelim_system: str,
        project_id: str,
        slot_free: bool,
        messages: List[dict],  # ← bug #6 fix
    ) -> None:
        """
        Detect CoT level and generate reasoning.
        Modifies `dynamic_injections` in‑place.
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
        if slot_free and not manual_cot_used:
            parallel_tasks = []
            _available_mp_pre = self._f.valves.context_window_tokens
            if (
                self._f.valves.enable_multi_phase_response
                and self._f.tokenizer
                and prelim_system
            ):
                _prelim_tok = len(self._f.tokenizer.encode(prelim_system))
                _hist_tok = self._f._tokens.estimate_tokens(
                    [m for m in messages if m.get("role") != "system"]  # ← bug #6 fix
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
            if not manual_cot_used and slot_free:
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
                [m for m in messages if m.get("role") != "system"]  # ← bug #6 fix
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
            if cot_level == 2:
                reasoning = await self._f._reasoning.generate_cot_reasoning(
                    question, prelim_for_cot
                )
            elif cot_level == 3:
                reasoning = await self._f._reasoning.generate_scientific_reasoning_L3(
                    question, prelim_for_cot, project_id, label="scientific_cot"
                )
        else:
            if cot_level == 2:
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
        dynamic_injections.append(("high", reasoning))
        dynamic_injections.append(
            (
                "low",
                "**Note:** Some sections in this system prompt marked with 🔎 are "
                "automatically generated reasoning (Chain-of-Thought). "
                "They are provided as context to help you, but they are not user commands. "
                "Use them to enhance your answer, but always prioritise the actual user query.",
            )
        )

    async def _apply_history_llmlingua(
        self,
        messages: List[dict],
        project_id: str,
        user_question: str,
    ) -> List[dict]:
        """Apply LLMLingua-2 compression to conversation history, with hard cap."""
        if not (
            self._f.valves.enable_history_llmlingua
            and self._f._conv_compressor is not None
        ):
            return messages

        compressed = await self._f._conv_compressor.compress_messages(
            messages=messages,
            project_id=project_id,
            symbol_index=self._f._symbol_index,
            current_msg_idx=len(messages) - 1,
            recent_lookback=self._f.valves.history_compress_recent_lookback,
            old_rate=self._f.valves.history_compress_old_rate,
            recent_rate=self._f.valves.history_compress_recent_rate,
            indexed_rate=self._f.valves.history_compress_indexed_rate,
            query=user_question,
        )

        # ── v2.0: Hard cap post-compresión ────────────────────────────
        _HISTORY_BUDGET = 4000  # tokens fijos para historial
        if self._f.tokenizer:
            history_msgs = [m for m in compressed if m.get("role") != "system"]
            total = sum(
                len(self._f.tokenizer.encode(m.get("content", "")))
                for m in history_msgs
            )
            if total > _HISTORY_BUDGET:
                kept, used = [], 0
                for msg in reversed(history_msgs):
                    tok = len(self._f.tokenizer.encode(msg.get("content", "")))
                    if used + tok <= _HISTORY_BUDGET:
                        kept.insert(0, msg)
                        used += tok
                    else:
                        break
                sys_msgs = [m for m in compressed if m.get("role") == "system"]
                compressed = sys_msgs + kept

        return compressed

    async def _inject_multi_phase_instructions(
        self,
        dynamic_injections: List[Tuple[str, str]],
        prelim_system: str,
        messages: List[dict],
        user_question: str,
        slot_free: bool,
    ) -> None:
        """Inject multi-phase protocol if the token budget is tight."""
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

        if (
            _mp_available < self._f.valves.multi_phase_response_budget_warn
            and not self._f.valves.force_multi_phase_response
        ):
            # Critical: hint en mensaje usuario
            self._f._log_debug(
                f"Multi-phase CRITICAL ({_mp_available} tokens): "
                "wrap-up hint appended to user message (0 system tokens used)."
            )
            # El método append_critical_wrap_up_hint modifica la lista in‑place
            self._f._multi_phase.append_critical_wrap_up_hint(messages)
            return

        if (
            _mp_available < self._f.valves.multi_phase_response_threshold
            or self._f.valves.force_multi_phase_response
        ):
            _INSTRUCTION_OVERHEAD = 450
            _mp_budget_reported = max(500, _mp_available - _INSTRUCTION_OVERHEAD)
            _mp_instructions = self._f._multi_phase.build_multi_phase_instructions(
                available_tokens=_mp_budget_reported,
                user_query=user_question,
                cot_degraded_to_l1=False,  # este valor se determinará fuera; lo mantengo simple
                is_continuation=not slot_free,
            )
            dynamic_injections.append(("critical", _mp_instructions))
            self._f._log_debug(
                f"Multi-phase injected (priority=critical): "
                f"{_mp_available} available, reporting {_mp_budget_reported} to model "
                f"(overhead={_INSTRUCTION_OVERHEAD})."
            )
        else:
            self._f._log_debug(
                f"Multi-phase: not needed ({_mp_available} tokens > threshold "
                f"{self._f.valves.multi_phase_response_threshold})."
            )

    async def _trim_and_summarize(
        self,
        messages: List[dict],
        state: dict,
        project_id: str,
        __user__: Optional[dict],
    ) -> Tuple[List[dict], str]:
        """
        Apply adaptive trimming to fit messages within the token window.
        Optionally summarize trimmed messages and persist the summary.
        Returns (updated_messages, pending_summary).
        """
        history_msgs = [m for m in messages if m.get("role") != "system"]
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        pending_summary = ""

        # ── v2.0: Proactive history budget enforcement ──────────────────
        if self._f.valves.history_max_tokens > 0 and self._f.tokenizer:
            budget = self._f.valves.history_max_tokens
            kept, used = [], 0
            for msg in reversed(history_msgs):
                tok = len(self._f.tokenizer.encode(msg.get("content", "")))
                if used + tok <= budget:
                    kept.insert(0, msg)
                    used += tok
                else:
                    # Summarize dropped messages if enabled
                    if self._f.valves.summarize_old_messages and not pending_summary:
                        old = [m for m in history_msgs if m not in kept]
                        if old:
                            has_code = any("```" in m.get("content", "") for m in old)
                            summary = (
                                await self._f._history_compressor.summarize_messages(
                                    old, is_code_context=has_code
                                )
                            )
                            if summary:
                                state["conversation_summaries"].append(
                                    {
                                        "text": summary,
                                        "created_at": time.time(),
                                        "covers_msgs": len(old),
                                    }
                                )
                                cap = self._f.valves.max_conversation_summaries
                                if (
                                    cap > 0
                                    and len(state["conversation_summaries"]) > cap
                                ):
                                    dropped = len(state["conversation_summaries"]) - cap
                                    state["conversation_summaries"] = state[
                                        "conversation_summaries"
                                    ][-cap:]
                                    self._f._log_debug(
                                        f"Summary cap: dropped {dropped} oldest summary block(s) "
                                        f"(max_conversation_summaries={cap})"
                                    )
                                self._f._state_store.set_state(project_id, state)
                                pending_summary = (
                                    f"[Summary of earlier conversation]\n{summary}"
                                )
                    break
            history_msgs = kept

        # ── Existing adaptive_trim logic ─────────────────────────────────
        if self._f.valves.adaptive_trim:
            total_tokens = self._f._tokens.estimate_tokens(history_msgs + sys_msgs)
            if total_tokens > self._f.valves.context_window_tokens:
                keep = self._f.valves.max_turns
                last_user_idx = -1
                for i in range(len(history_msgs) - 1, -1, -1):
                    if history_msgs[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx != -1:
                    start_idx = max(0, last_user_idx - keep + 1)
                    old_block = history_msgs[:start_idx] if start_idx > 0 else []
                    kept_block = history_msgs[start_idx:]
                else:
                    old_block = history_msgs[:-keep] if keep > 0 else []
                    kept_block = history_msgs[-keep:] if keep > 0 else []

                if self._f.valves.summarize_old_messages and old_block:
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._f._history_compressor.summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    if summary:
                        state["conversation_summaries"].append(
                            {
                                "text": summary,
                                "created_at": time.time(),
                                "covers_msgs": len(old_block),
                            }
                        )
                        cap = self._f.valves.max_conversation_summaries
                        if cap > 0 and len(state["conversation_summaries"]) > cap:
                            dropped = len(state["conversation_summaries"]) - cap
                            state["conversation_summaries"] = state[
                                "conversation_summaries"
                            ][-cap:]
                            self._f._log_debug(
                                f"Summary cap: dropped {dropped} oldest summary block(s) "
                                f"(max_conversation_summaries={cap})"
                            )
                        self._f._state_store.set_state(project_id, state)
                        pending_summary = (
                            f"[Summary of earlier conversation]\n{summary}"
                        )
                    history_msgs = kept_block
                else:
                    history_msgs = kept_block if old_block else history_msgs

                if self._f.valves.preserve_tool_calls:
                    while history_msgs and history_msgs[0].get("role") == "tool":
                        history_msgs.pop(0)
                    if (
                        history_msgs
                        and history_msgs[0].get("role") == "assistant"
                        and history_msgs[0].get("tool_calls")
                    ):
                        tool_call_ids = {
                            tc.get("id") for tc in history_msgs[0]["tool_calls"]
                        }
                        tool_response_ids = {
                            m.get("tool_call_id")
                            for m in history_msgs[1:]
                            if m.get("role") == "tool"
                        }
                        if not tool_call_ids.issubset(tool_response_ids):
                            history_msgs.pop(0)
        else:
            user_max = (
                __user__["valves"].max_turns
                if __user__ and hasattr(__user__, "valves")
                else None
            )
            eff_max = user_max if user_max is not None else self._f.valves.max_turns
            if len(history_msgs) > eff_max:
                keep = eff_max
                last_user_idx = -1
                for i in range(len(history_msgs) - 1, -1, -1):
                    if history_msgs[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx != -1:
                    start_idx = max(0, last_user_idx - keep + 1)
                    old_block = history_msgs[:start_idx] if start_idx > 0 else []
                    kept_block = history_msgs[start_idx:]
                else:
                    old_block = history_msgs[:-keep] if keep > 0 else []
                    kept_block = history_msgs[-keep:] if keep > 0 else []

                if self._f.valves.summarize_old_messages and old_block:
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._f._history_compressor.summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    if summary:
                        state["conversation_summaries"].append(
                            {
                                "text": summary,
                                "created_at": time.time(),
                                "covers_msgs": len(old_block),
                            }
                        )
                        cap = self._f.valves.max_conversation_summaries
                        if cap > 0 and len(state["conversation_summaries"]) > cap:
                            dropped = len(state["conversation_summaries"]) - cap
                            state["conversation_summaries"] = state[
                                "conversation_summaries"
                            ][-cap:]
                            self._f._log_debug(
                                f"Summary cap: dropped {dropped} oldest summary block(s) "
                                f"(max_conversation_summaries={cap})"
                            )
                        self._f._state_store.set_state(project_id, state)
                        pending_summary = (
                            f"[Summary of earlier conversation]\n{summary}"
                        )
                    history_msgs = kept_block
                else:
                    history_msgs = kept_block

        return sys_msgs + history_msgs, pending_summary

    async def _compress_code_history_and_lean(
        self,
        messages: List[dict],
        project_id: str,
        dynamic_injections: List[Tuple[str, str]],
    ) -> List[dict]:
        """Apply code history compression and lean user code if enabled."""
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
            messages = self._f._history_compressor.lean_user_code_messages(
                messages, project_id
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

    def _assemble_final_system_and_log(
        self,
        static_block: str,
        dynamic_injections: List[Tuple[str, str]],
        messages: List[dict],
        project_id: str,
        pending_summary: str,
    ) -> List[dict]:
        """Assemble final system prompt, inject it, and log token breakdown."""
        budget = self._f.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self._f.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_dynamic: List[str] = []
            used_dyn = 0
            static_tokens = (
                len(self._f.tokenizer.encode(static_block)) if static_block else 0
            )
            dyn_budget = max(0, budget - static_tokens)
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
        final_system = static_block + separator + dynamic_block

        # Append base system content (from original message)
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        base_content = sys_msgs[0].get("content", "") if sys_msgs else ""
        if base_content.strip():
            final_system = final_system + "\n\n" + base_content

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
            total_system_tok = len(self._f.tokenizer.encode(final_system))

            self._f._last_system_tokens[project_id] = total_system_tok

            prefix_hash = self._f._last_static_prefix_hash.get(project_id, "N/A")
            self._f._log_debug("─" * 60)
            self._f._log_debug("TOKEN BREAKDOWN — system prompt")
            self._f._log_debug(f"  BLOCK A (static, cacheable):  ~{static_tok} tokens")
            self._f._log_debug(f"  BLOCK B (dynamic, per-query): ~{dynamic_tok} tokens")
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
                if "_mp_available" not in dir():  # Not available here; log simple
                    self._f._log_debug(
                        "  Multi-phase:                  (see earlier log)"
                    )
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

        return messages


# ---------------------------------------------------------------------------
# Valves
# ---------------------------------------------------------------------------
class Filter:

    class Valves(BaseModel):
        # ═══════════════════════════════════════════════════════════════
        #  Core
        # ═══════════════════════════════════════════════════════════════
        llm_per_call_timeout: int = Field(
            default=900,
            ge=1,
            description="Timeout (seconds) for a single LLM call attempt.",
        )
        llm_retry_total_timeout: int = Field(
            default=950,
            ge=10,
            description="Total time budget for retrying failed LLM calls.",
        )
        priority: int = Field(default=0)
        max_turns: int = Field(default=8)  # ← v2.0: was 15
        debug: bool = Field(default=True)
        debug_context: bool = Field(
            default=False,
            description="Print the full system message content at the end of the inlet for debugging.",
        )
        state_db_path: str = Field(default="/app/backend/data/conversation_state.db")
        track_line_numbers: bool = Field(default=True)
        adaptive_trim: bool = Field(default=True)
        context_window_tokens: int = Field(
            default=262000,
            description="It's crucial this has the same value as the llama server to prevent uncontrolled errors.",
        )
        use_tiktoken: bool = Field(default=True)
        project_id: str = Field(default="default")
        max_cached_projects: int = Field(default=10)

        # ═══════════════════════════════════════════════════════════════
        #  Long‑Term Memory (ChromaDB)
        # ═══════════════════════════════════════════════════════════════
        long_term_memory_dir: str = Field(default="/app/backend/data/long_term_memory")
        long_term_memory_expiration_days: int = Field(default=30)
        long_term_memory_top_k: int = Field(default=10)
        long_term_memory_similarity_threshold: float = Field(default=0.65)
        ltm_time_decay_hours: float = Field(default=12.0)
        ltm_retrieval_max_tokens: int = Field(default=0)
        ltm_store_only_code_sessions: bool = Field(default=True)
        ltm_include_timestamps: bool = Field(default=True)
        ltm_compress_after_messages: int = Field(default=50)
        ltm_summarization_trigger_similarity: float = Field(default=0.85)
        ltm_index_symbols_enabled: bool = Field(default=True)
        ltm_symbol_index_max_per_message: int = Field(default=20)
        ltm_symbol_boost_enabled: bool = Field(default=True)
        ltm_symbol_boost_factor: float = Field(default=1.5)
        ltm_symbol_boost_min_similarity: float = Field(default=0.5)
        ltm_symbol_force_mode_enabled: bool = Field(default=False)
        ltm_symbol_force_fallback_to_semantic: bool = Field(default=True)
        enable_reranking: bool = Field(default=True)
        reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker_top_k: int = Field(default=5)

        # ═══════════════════════════════════════════════════════════════
        #  Code Awareness & Context
        # ═══════════════════════════════════════════════════════════════
        enable_code_awareness: bool = Field(default=True)
        code_similarity_threshold: float = Field(default=0.85)
        max_base_code_blocks: int = Field(default=3)
        max_proposed_changes: int = Field(default=5)
        max_committed_changes: int = Field(default=10)
        prioritize_recent_code: bool = Field(default=True)
        auto_detect_code_blocks: bool = Field(default=True)
        max_active_blocks: int = Field(default=0, ge=0)
        track_file_paths: bool = Field(default=True)
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b"
        )
        max_code_block_tokens: int = Field(default=0)
        code_block_overflow_action: str = Field(default="warn")
        code_block_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        code_block_truncate_keep_head: int = Field(default=50)
        code_block_truncate_keep_tail: int = Field(default=50)
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]"
        )
        enable_call_graph_extraction: bool = Field(default=True)
        enable_auto_summaries: bool = Field(default=True)
        summary_code_max_chars: int = Field(default=8000)
        oversized_summary_max_tokens: int = Field(default=500)
        active_context_max_tokens: int = Field(default=32000)
        global_injection_token_budget: int = Field(default=0)
        exclude_filter_internals: bool = Field(default=True)
        enable_ast_deduplication: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Smart Context Selection
        # ═══════════════════════════════════════════════════════════════
        smart_context_selection: bool = Field(default=False)
        smart_context_top_k: int = Field(default=15)
        smart_context_min_tokens: int = Field(default=1024)
        smart_context_include_last_user: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Graph Analysis (paths, LOD, centrality, seeds)
        # ═══════════════════════════════════════════════════════════════
        enable_path_analysis: bool = Field(default=True)
        path_activation_threshold: float = Field(default=0.1, ge=0.01, le=1.0)
        path_relevance_high_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        path_propagation_steps: int = Field(default=4, ge=1, le=8)
        path_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        path_summary_max_tokens: int = Field(default=80)
        lod3_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
        lod2_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
        lod1_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
        enable_centrality_prior: bool = Field(default=True)
        enable_centrality_lod_bump: bool = Field(default=True)
        centrality_lod_bump_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        centrality_lod_bump_weight: float = Field(default=0.15, ge=0.0, le=0.5)
        enable_traceback_activation: bool = Field(default=True)
        enable_history_seeds: bool = Field(default=True)
        history_seeds_lookback: int = Field(default=6, ge=2, le=20)
        history_seeds_max_boost: float = Field(default=0.6, ge=0.1, le=0.9)
        enable_multi_seed_activation: bool = Field(default=True)
        multi_seed_weight_lexical: float = Field(default=0.5, ge=0.0, le=1.0)
        multi_seed_weight_structural: float = Field(default=0.3, ge=0.0, le=1.0)
        multi_seed_weight_historical: float = Field(default=0.2, ge=0.0, le=1.0)
        ppr_alpha: float = Field(default=0.85, ge=0.5, le=0.99)

        # ═══════════════════════════════════════════════════════════════
        #  LLM Configuration
        # ═══════════════════════════════════════════════════════════════
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
        )
        openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", "dummy"))
        LLM_BASE_URL: str = Field(default="http://host.docker.internal:8080")
        LLM_API_TOKEN: str = Field(default="")
        llm_model: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        LLM_MAX_CONCURRENT_CALLS: int = Field(default=1, ge=1, le=10)
        llm_request_timeout: int = Field(default=900)
        LLM_CACHE_TTL: int = Field(default=300)
        LLM_CACHE_MAX_SIZE: int = Field(default=100)
        llamacpp_endpoint_type: str = Field(default="chat")
        intent_classifier_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        enable_intent_llm_fallback: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Multi-Phase Response
        # ═══════════════════════════════════════════════════════════════
        enable_multi_phase_response: bool = Field(default=True)
        force_multi_phase_response: bool = Field(default=True)
        multi_phase_effective_max_tokens: int = Field(default=4500, ge=1000, le=200000)
        multi_phase_response_threshold: int = Field(default=7000, ge=0, le=200000)
        multi_phase_response_budget_warn: int = Field(default=800, ge=500, le=40000)
        auto_budget_context_for_parts: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Code History Compression
        # ═══════════════════════════════════════════════════════════════
        enable_code_history_compression: bool = Field(default=True)
        code_history_keep_last_n_parts: int = Field(default=2, ge=1, le=5)
        code_history_symbol_index_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
        enable_lean_user_code: bool = Field(default=True)
        lean_user_code_min_tokens: int = Field(default=3000)  # ← v2.0: was 8000

        # ═══════════════════════════════════════════════════════════════
        #  Code Compression (LLMLingua-2)
        # ═══════════════════════════════════════════════════════════════
        enable_code_compression: bool = Field(default=True)
        code_compression_rate: float = Field(default=0.5, ge=0.3, le=0.8)
        code_compression_min_tokens: int = Field(default=150)
        enable_question_aware_compression: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Importance & Expiration
        # ═══════════════════════════════════════════════════════════════
        importance_mention_boost: float = Field(default=0.2)
        importance_recency_half_life_hours: float = Field(default=2.0)
        block_expiration_hours: float = Field(default=24.0)
        proposed_change_retention_turns: int = Field(default=20)
        preserve_tool_calls: bool = Field(default=True)
        error_retention_turns: int = Field(default=15)
        track_active_code_age: bool = Field(default=True)
        active_code_timeout_minutes: int = Field(default=45)
        recent_activity_window_minutes: int = Field(default=15)
        max_change_summaries: int = Field(default=1000)

        # ═══════════════════════════════════════════════════════════════
        #  Duplicate Blocks & Frequency
        # ═══════════════════════════════════════════════════════════════
        auto_remove_duplicate_blocks: bool = Field(default=True)
        max_duplicate_age_hours: float = Field(default=6.0)
        frequency_weight_factor: float = Field(default=0.3)
        min_mentions_for_boost: int = Field(default=3)
        frequency_decay_hours: float = Field(default=12.0)

        # ═══════════════════════════════════════════════════════════════
        #  Confidence Scoring & Chain‑of‑Thought
        # ═══════════════════════════════════════════════════════════════
        enable_confidence_scoring: bool = Field(default=True)
        confidence_prompt: str = Field(
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]'..."
        )
        enable_cot_on_demand: bool = Field(default=True)
        auto_cot_enabled: bool = Field(default=False)
        auto_cot_min_chars: int = Field(default=200)
        cot_max_tokens: int = Field(default=1500)  # ← v2.0: was 0
        cot_model: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level2: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level3: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        enable_cot_llm_detection: bool = Field(default=True)
        cot_detection_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        enforce_scientific_method: bool = Field(default=False)
        scientific_hypotheses_count: int = Field(default=3, ge=2, le=6)
        scientific_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
        scientific_max_iterations: int = Field(default=2, ge=1, le=4)
        enable_step_back_prompting: bool = Field(default=True)
        step_back_always: bool = Field(default=False)
        step_back_max_tokens: int = Field(default=150, ge=50, le=400)
        enable_contradiction_detection: bool = Field(default=True)
        contradiction_detection_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        contradiction_inject_warning: bool = Field(default=True)
        enable_assumption_extraction: bool = Field(default=True)
        assumption_extraction_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )

        # ═══════════════════════════════════════════════════════════════
        #  Proactive Suggestions & Commands
        # ═══════════════════════════════════════════════════════════════
        proactive_context_warning_threshold: float = Field(default=0.85)
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Context Warning**: ..."
        )
        proactive_summary_threshold: float = Field(default=0.75)
        proactive_summary_growth_window: int = Field(default=3)
        enable_command_suggestions: bool = Field(default=True)
        command_suggestion_cooldown_minutes: int = Field(default=10)
        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=True)
        natural_language_forget_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        cleanup_suggestions_enabled: bool = Field(default=True)
        cleanup_inactive_threshold_messages: int = Field(default=30)
        cleanup_excluded_content_types: list = Field(
            default_factory=lambda: ["BASE_CODE"]
        )
        cleanup_status_command_enabled: bool = Field(default=True)
        cleanup_proactive_suggestions: bool = Field(default=True)
        cleanup_suggestion_cooldown_messages: int = Field(default=20)
        cleanup_command_enabled: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Duplicate Question Detection
        # ═══════════════════════════════════════════════════════════════
        duplicate_question_threshold: float = Field(default=0.92)
        duplicate_question_lookback: int = Field(default=20)
        duplicate_question_lookback_hours: float = Field(default=24.0)

        # ═══════════════════════════════════════════════════════════════
        #  Similar Messages & Obsolete Marking
        # ═══════════════════════════════════════════════════════════════
        similar_message_handling: str = Field(default="replace")
        similar_message_threshold: float = Field(default=0.92)
        similar_message_check_code_only: bool = Field(default=True)
        enable_obsolete_marking: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Response Cache
        # ═══════════════════════════════════════════════════════════════
        enable_response_cache: bool = Field(default=True)
        response_cache_similarity_threshold: float = Field(default=0.92)
        response_cache_ttl_hours: float = Field(default=24.0)
        response_cache_max_entries: int = Field(default=100)
        response_cache_include_context_hash: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Selective Summarization
        # ═══════════════════════════════════════════════════════════════
        selective_summarization: bool = Field(default=True)
        error_preserve_verbatim: bool = Field(default=True)
        error_max_age_hours: float = Field(default=48.0)
        code_summary_level: str = Field(default="balanced")
        general_summary_max_tokens: int = Field(default=200)
        tool_call_preserve: bool = Field(default=True)
        code_always_keep_signature: bool = Field(default=True)
        summary_fallback_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        summary_include_metadata: bool = Field(default=True)
        summarize_old_messages: bool = Field(default=True)
        summarization_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )

        # ═══════════════════════════════════════════════════════════════
        #  Feedback Tracking
        # ═══════════════════════════════════════════════════════════════
        enable_feedback_tracking: bool = Field(default=True)
        feedback_history_limit: int = Field(default=10)
        inject_feedback_context: bool = Field(default=True)
        feedback_importance_penalty_for_failure: float = Field(default=2.0)
        enable_diff_application: bool = Field(default=True)
        preserve_error_context: bool = Field(default=True)
        code_block_pattern: str = Field(default="```(\\w*)\\n(.*?)```")
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@"
        )
        commit_pattern: str = Field(default="commit\\s+([a-f0-9]{7,40})")

        # ═══════════════════════════════════════════════════════════════
        #  Session Summaries & Secondary Tasks
        # ═══════════════════════════════════════════════════════════════
        enable_session_summary: bool = Field(default=True)
        session_summary_interval_messages: int = Field(default=8)
        session_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        session_summary_max_tokens: int = Field(default=200)

        # ═══════════════════════════════════════════════════════════════
        #  Raw File Priority Boost
        # ═══════════════════════════════════════════════════════════════
        raw_file_priority_boost: float = Field(default=2.0)

        # ═══════════════════════════════════════════════════════════════
        #  Expand Intercept & Default Depth
        # ═══════════════════════════════════════════════════════════════
        outlet_expand_intercept_enabled: bool = Field(default=True)
        outlet_expand_intercept_max_symbols: int = Field(default=0, ge=0)
        outlet_expand_intercept_depth: int = Field(default=5, ge=0)
        expand_default_depth: int = Field(default=2)

        # ═══════════════════════════════════════════════════════════════
        #  RAPTOR Hierarchical LTM
        # ═══════════════════════════════════════════════════════════════
        enable_raptor: bool = Field(default=True)  # ← v2.0: was False
        raptor_clusters_per_level: int = Field(default=5, ge=2, le=20)
        raptor_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact"
        )
        raptor_summary_max_tokens: int = Field(default=150)
        raptor_rebuild_interval: int = Field(default=20)

        # ═══════════════════════════════════════════════════════════════
        #  KV Cache Stability & Slot Persistence
        # ═══════════════════════════════════════════════════════════════
        enable_kv_cache_stability: bool = Field(default=True)
        enable_slot_persistence: bool = Field(default=True)
        slot_save_path: str = Field(default="/tmp/llama_slots")
        slot_id: int = Field(default=0, ge=0)

        # ═══════════════════════════════════════════════════════════════
        #  Retrieval Enhancements
        # ═══════════════════════════════════════════════════════════════
        enable_contextual_retrieval: bool = Field(default=True)
        contextual_retrieval_mode: str = Field(default="metadata")
        enable_multi_query_retrieval: bool = Field(default=True)
        multi_query_variants: int = Field(default=2, ge=1, le=4)

        # ═══════════════════════════════════════════════════════════════
        #  Edge Persistence (Cross-Session SymbolGraph)
        # ═══════════════════════════════════════════════════════════════
        enable_edge_persistence: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Adaptive LOD Thresholds
        # ═══════════════════════════════════════════════════════════════
        enable_lod_adaptive: bool = Field(default=True)
        lod_adapt_rate: float = Field(default=0.05, ge=0.01, le=0.2)
        lod_adapt_min: float = Field(default=0.25, ge=0.1, le=0.5)
        lod_adapt_max: float = Field(default=0.75, ge=0.5, le=0.95)
        lod_adapt_underserved_min: int = Field(default=2, ge=1, le=10)
        lod_adapt_overserved_min: int = Field(default=3, ge=1, le=10)

        # ═══════════════════════════════════════════════════════════════
        #  Speculative Pre‑fetching
        # ═══════════════════════════════════════════════════════════════
        enable_speculative_prefetch: bool = Field(default=True)
        speculative_prefetch_max: int = Field(default=5, ge=1, le=20)

        # ═══════════════════════════════════════════════════════════════
        #  Silent Ingestion
        # ═══════════════════════════════════════════════════════════════
        enable_silent_ingestion: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  Data Flow Analysis
        # ═══════════════════════════════════════════════════════════════
        enable_data_flow_analysis: bool = Field(default=True)

        # ═══════════════════════════════════════════════════════════════
        #  v8 — Hub symbol index (Block A reduction)
        # ═══════════════════════════════════════════════════════════════
        symbol_index_max_in_block_a: int = Field(
            default=30,
            ge=5,
            le=200,
            description=(
                "Maximum number of hub symbols (top-N by call-graph centrality) "
                "kept in Block A. Non-hub symbols remain available via LOD on "
                "demand. Lower = smaller, more cache-stable Block A; higher = "
                "more symbols visible up-front at the cost of prefill time."
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        #  v8 — Context paging (lossless soft-eviction)
        # ═══════════════════════════════════════════════════════════════
        enable_block_paging: bool = Field(
            default=True,
            description=(
                "Soft-evict low-activation code blocks to ChromaDB instead of "
                "dropping them. Paged blocks stay fully recoverable via LOD."
            ),
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
            description=(
                "PPR activation score below which a block becomes a paging "
                "candidate. Blocks with any hotter symbol are kept in RAM."
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        #  v8 — RAPTOR code clustering
        # ═══════════════════════════════════════════════════════════════
        raptor_use_call_graph_proximity: bool = Field(
            default=True,
            description=(
                "Weight call-graph distance alongside semantic similarity when "
                "clustering symbols into RAPTOR subsystems."
            ),
        )
        raptor_graph_weight: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description=(
                "Weight of call-graph proximity in the clustering metric. "
                "0.0 = semantic only (original RAPTOR), 1.0 = graph only."
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        #  v8 — History LLMLingua compression
        # ═══════════════════════════════════════════════════════════════
        enable_history_llmlingua: bool = Field(
            default=True,  # ← v2.0: was False
            description=(
                "Apply LLMLingua-2 compression to full conversation history. "
                "Reduces old turns 40-80% by tier. Experimental — disable if "
                "responses become incoherent on long sessions."
            ),
        )
        history_compress_recent_rate: float = Field(
            default=0.75,
            ge=0.3,
            le=1.0,
            description="Compression rate for the last `recent_lookback` turns.",
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
            description=(
                "Compression rate for old turns whose code is fully indexed in "
                "the SymbolGraph. Aggressive is safe — bodies recover via LOD."
            ),
        )
        history_compress_recent_lookback: int = Field(
            default=4,
            ge=1,
            le=20,
            description="Number of recent turns exempt from aggressive compression.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  v8 — Conversation summary cap (working-memory lookback)
        # ═══════════════════════════════════════════════════════════════
        max_conversation_summaries: int = Field(
            default=3,
            ge=0,
            description=(
                "Maximum number of conversation summary blocks kept and "
                "re-injected each request. Oldest dropped when exceeded. "
                "0 = disabled (keep all). Controls how far back the model can "
                "see: higher = longer memory, more tokens."
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        #  v8 — Response reserve tokens (for adaptive trimming)
        # ═══════════════════════════════════════════════════════════════
        response_reserve_tokens: int = Field(
            default=2048,
            ge=256,
            le=16384,
            description=(
                "Minimum tokens reserved for the LLM's response when computing "
                "the effective context budget for adaptive trimming."
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        #  v2.0 — Hard history budget
        # ═══════════════════════════════════════════════════════════════
        history_max_tokens: int = Field(
            default=4000,
            description=(
                "Maximum tokens for conversation history (non-system messages). "
                "Enforced after LLMLingua compression. 0 = disabled."
            ),
        )

    # --------------------------------------------------------------------------
    # Class-level constants
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(self):
        # Valves and basic objects
        self.valves = self.Valves()

        self.tokenizer = None
        self._db_conn = None
        self._cross_encoder = None
        self._cross_encoder_unavailable_logged = False
        self._cross_encoder_lock = asyncio.Lock()

        # ── v8: New manager instances ──────────────────────────────────────
        self._conv_compressor = _shared_get_conversation_compressor()
        self._llmlingua_compressor = (
            self._conv_compressor.raw if self._conv_compressor else None
        )

        self._state_store = StateStore(self)
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
        self._system_prompt_builder = SystemPromptBuilder(self)  # ← v8
        self._message_assembler = MessageAssembler(self)  # ← v8

        self._hub_index = HubSymbolIndex()
        self._ctx_builder = ContextBuilder(self)
        self._pager = ContextPager()
        self._raptor = RaptorCodeIndex()
        # ────────────────────────────────────────────────────────────────────

        # Conversation state (moved to StateStore)
        self._conversation_state: OrderedDict = OrderedDict()
        self._state_factory = lambda: {
            "active_blocks": {},
            "recent_changes": [],
            "committed_changes": [],
            "message_count": 0,
            "feedback_history": [],
            "last_compression_timestamp": 0,
            "last_suggestion_timestamp": 0,
            "response_cache": [],
            "has_any_calls": False,
            "last_cleanup_suggestion_msg_idx": 0,
            "last_cot_level": 0,
            "conversation_summaries": [],  # ← v8 (Step 5.3)
        }

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
        self._llm_semaphore = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENT_CALLS)
        self._pending_llm: Dict[str, asyncio.Future] = {}
        self._pending_llm_lock = asyncio.Lock()
        self._llm_orchestrator.init_cache()
        self._last_used_model: Optional[str] = None
        self._main_model_ready = False

        # ── Tracking of active LLM tasks (prevents model switching conflicts) ──
        self._active_llm_tasks: Set[asyncio.Task] = set()
        self._active_llm_tasks_lock = asyncio.Lock()

        # ── Database write queue (prevents "database is locked") ──
        self._db_write_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._db_worker_task = asyncio.create_task(self._state_store.db_worker())

        # Session classification cache
        self._session_classify_cache: Dict[str, Tuple[bool, float]] = {}
        self._session_classify_ttl: float = 1800.0

        # Symbol index and lightweight context
        self._symbol_index = SymbolIndex()
        self._path_index = PathIndex()
        self._node_centrality: Dict[str, Dict[str, float]] = {}
        self._cached_lightweight_context: Dict[str, str] = {}
        self._cached_code_state_hash: Optional[str] = None

        self._last_system_tokens: Dict[str, int] = {}

        # ── KV Cache Stability ──
        self._static_context_block_cache: Dict[str, Tuple[str, str]] = {}
        self._last_static_prefix_hash: Dict[str, str] = {}

        # ── KV Cache Slot Persistence ──
        self._last_saved_slot_hash: Dict[str, str] = {}
        self._slot_restored: Dict[str, bool] = {}
        self._slot_restore_attempted: Dict[str, bool] = {}

        # Project tracking
        self._last_processed_message_idx: Dict[str, int] = {}
        self._last_project_id: str = ""
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}

        # Response cache counter
        self._response_cache_count: Dict[str, int] = {}
        self._summarize_inactive_in_progress: Dict[str, bool] = {}
        self._write_counter = 0

        # Block change summaries LRU
        self._block_change_summaries: OrderedDict = OrderedDict()
        self._MAX_CHANGE_SUMMARIES = self.valves.max_change_summaries

        # Thread pools for blocking operations
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

        # State debounce
        self._state_dirty = False
        self._state_last_saved = 0.0

        print("[CodeAware] Filter loaded")

    # --------------------------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Inlet helpers
    # --------------------------------------------------------------------------
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
    ):
        return await self._system_prompt_builder.build(
            messages,
            project_id,
            user_query,
            user_question,
            is_code_session,
            last_user_msg,
            state,
            slot_free,
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

    async def _update_active_code(
        self, message: dict, project_id: str, is_continuation: bool = False
    ) -> None:
        await self._active_code_updater.process(message, project_id, is_continuation)

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
        self._log_debug("inlet called")
        inlet_start = time.monotonic()
        self._log_section("CONTEXT MANAGER - INLET START")

        def _inlet_timing(step_name: str, start: float, end: float = None):
            if end is None:
                end = time.monotonic()
            self._log_timing(step_name, start - inlet_start, end - start)

        project_id = self._inlet_orch.get_project_id()
        slot_free = True
        # Cold‑start guard: si no hay modelo cargado, no hay slot que liberar
        if slot_free and self._last_used_model is None:
            slot_free = False

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

                _msg_to_index = last_user_msg
                if "```" not in user_query and self._commands.has_code_indicators(
                    user_query
                ):
                    _guessed_lang = SignatureExtractor._guess_language(None, user_query)
                    _lang = _guessed_lang if _guessed_lang != "unknown" else "python"
                    _msg_to_index = {
                        **last_user_msg,
                        "content": f"```{_lang}\n{user_query}\n```",
                    }
                    self._log_debug(
                        f"Silent ingestion: wrapping raw code as {_lang} "
                        f"({self._tokens.estimate_code_tokens(user_query)} tokens)"
                    )

                # Process code into SymbolGraph without invoking main LLM
                await self._update_active_code(_msg_to_index, project_id)

                # Resolve cross‑references with previous chunks
                await self._activation.resolve_dangling_edges(project_id)

                # Rebuild PathIndex with new symbols
                if self.valves.enable_path_analysis:
                    await self._activation.rebuild_path_index(project_id)

                # Invalidate static block (new code → new Block A)
                self._ctx_builder.invalidate_block_a_cache(
                    project_id, "new chunk ingested"
                )

                # Forzar guardado del estado tras la ingesta (no hay outlet)
                self._state_dirty = True
                await self._state_store.save_state_if_dirty(project_id)

                # Statistics for the user
                state = self._state_store.get_state(project_id)
                num_blocks = len(state.get("active_blocks", {}))
                num_symbols = len(self._symbol_index.get_all_names(project_id))

                response = (
                    f"✅ {num_symbols} símbolos indexados ({num_blocks} bloques activos). "
                    "El código está disponible en el SymbolGraph para futuras consultas. "
                    "Usa `/expand <nombre>` para ver una función/clase completa."
                )

                # ── Fix 1: Replace code with compressed stub instead of deleting it ──
                all_names = sorted(self._symbol_index.get_all_names(project_id))[:100]
                compressed_stub = (
                    f"```python\n"
                    f"# [CÓDIGO COMPRIMIDO — {num_symbols} símbolos indexados]\n"
                    + "\n".join(f"# class/func: {n}" for n in sorted(all_names)[:50])
                    + "\n```\n\n"
                    f"(Usa `/expand <nombre>` para ver la implementación completa "
                    f"de cualquier símbolo de los {num_symbols} disponibles)"
                )
                messages[-1] = {**messages[-1], "content": compressed_stub}
                messages.append({"role": "assistant", "content": response})
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
        # 🧠 ENRICHMENT (High value)
        #   6. Build system injections and assemble final messages
        #      (delegates Block A/B construction to ContextBuilder)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        state = self._state_store.get_state(project_id)
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
            )
        )
        _inlet_timing("Step 6/7: Build system injections", step_start)

        # ── v8: Restore KV slot after Block A has been built (fix #20) ──
        if (
            self.valves.enable_slot_persistence
            and project_id not in self._slot_restore_attempted
        ):
            await self._ctx_builder.slot_restore(project_id)

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

        body["messages"] = messages

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
        self._log_debug("outlet called")
        start_time = time.monotonic()
        self._log_section("CONTEXT MANAGER - OUTLET START")

        if not (HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness):
            return body

        try:
            messages = body.get("messages", [])
            project_id = self._inlet_orch.get_project_id()
            state = self._state_store.get_state(project_id)
            is_code_session = await self._inlet_orch.classify_session(
                messages, project_id
            )
            last_msg = messages[-1] if messages else None
            if last_msg:
                last_idx = len(messages) - 1
                if last_idx <= self._last_processed_message_idx.get(project_id, -1):
                    self._log_debug(
                        "outlet: last message already processed in inlet, skipping"
                    )
                else:
                    # ── 🔥 STATE MANAGEMENT: intercept /expand commands ──
                    if (
                        last_msg.get("role") == "assistant"
                        and is_code_session
                        and "/expand" in last_msg.get("content", "")
                    ):
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

                    # ── 🔥 STATE MANAGEMENT: update active code blocks & store in LTM ──
                    if last_msg.get("role") in ("user", "assistant"):
                        await self._llm_orchestrator.wait_for_llm_tasks()
                        if is_code_session:
                            self._log_debug(
                                "🔥 STATE MANAGEMENT – Updating active code blocks and storing in LTM "
                                "(new code detected)"
                            )
                            await self._update_active_code(last_msg, project_id)
                            # Store immediately in LTM
                            await self._ltm.store_messages(project_id, [last_msg])
                        else:
                            if not self.valves.ltm_store_only_code_sessions:
                                self._log_debug(
                                    "🔥 STATE MANAGEMENT – Storing non‑code session message in LTM"
                                )
                                await self._ltm.store_messages(project_id, [last_msg])

            # 🚀 RESOURCE OPTIMISATION: response cache storage
            if (
                self.valves.enable_response_cache
                and HAS_SENTENCE
                and len(messages) >= 2
            ):
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Storing response in cache "
                    "(to avoid recomputation for similar future requests)"
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
                            "Response cache: skipping storage for partial "
                            "multi-phase response (continuation marker detected)."
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
                        )

            # 🔄 LOD ADAPTIVE FEEDBACK (v7 Phase 5)
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
                            "LOD adaptive: skipping feedback for partial "
                            "multi-phase response (continuation marker detected)."
                        )
                    else:
                        await self._enrichment.update_lod_thresholds_from_response(
                            project_id,
                            last_assistant["content"],
                        )

            # 🚀 RESOURCE OPTIMISATION – Speculative prefetch (inline)
            if self.valves.enable_speculative_prefetch and is_code_session:
                last_activated = getattr(self, "_last_activation_scores", {}).get(
                    project_id, {}
                )
                if last_activated:
                    await self._activation.speculative_prefetch(
                        project_id, last_activated
                    )

            # 🚀 RESOURCE OPTIMISATION: purge expired memories periodically
            await self._ltm.purge_expired_memories()

            # 🚀 RESOURCE OPTIMISATION: DB checkpoints every 100 writes
            self._write_counter += 1

            # ── v8: RAPTOR rebuild (delegated to RaptorCodeIndex) ──────────
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

            if self._write_counter % 100 == 0:
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Running DB checkpoints "
                    "(to ensure data durability and prevent WAL buildup)"
                )
                await self._state_store.run_db_checkpoints()

            # 🚀 RESOURCE OPTIMISATION – Save KV slot (delegated to ContextBuilder)
            if self.valves.enable_slot_persistence:
                await self._ctx_builder.slot_save(project_id)

            # 🔥 STATE MANAGEMENT – Persistir edges del SymbolGraph
            if self.valves.enable_edge_persistence:
                await self._state_store.save_symbol_edges_to_db(project_id)

            # ── v8: Persistir CodePathViews ──
            if self.valves.enable_path_analysis:
                await self._state_store.save_path_views_to_db(
                    project_id, self._path_index.get_all(project_id)
                )

            # 🔥 STATE MANAGEMENT: persist conversation state if dirty
            self._log_debug(
                "🔥 STATE MANAGEMENT – Saving conversation state "
                "(to preserve context across restarts)"
            )
            await self._state_store.save_state_if_dirty(project_id)

            # 🚀 RESOURCE OPTIMISATION: Skipping unload to keep main model loaded
            self._log_debug(
                "🚀 RESOURCE OPTIMISATION – Skipping model unload to preserve KV cache"
            )
        finally:
            pass

        self._log_section(
            "CONTEXT MANAGER - OUTLET END", duration=time.monotonic() - start_time
        )
        return body
