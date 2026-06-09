"""
title: Code-Aware Context Manager with LTM & Summarization
description: Full-featured context manager for coding assistants.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 7.0.0
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
)

_inlet_background_tasks: contextvars.ContextVar[list] = contextvars.ContextVar(
    "_inlet_background_tasks", default=None
)
_db_global_lock = threading.Lock()
_llm_semaphore = asyncio.Semaphore(1)
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
        # The penalty for potentially_affected has been removed because that feature is gone
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
    "calls": 1.0,  # direct function call
    "imports": 0.6,  # import / from import
    "reads": 0.7,  # variable/attribute read
    "writes": 0.9,  # variable/attribute write
    "inherits": 0.5,  # class inheritance
    "references": 0.4,  # reference without a direct call
    "data_flow": 0.8,  # ← v7 (PASO-19)
}


class Edge(BaseModel):
    src: str  # source symbol id
    dst: str  # destination symbol id
    type: str  # key from EDGE_WEIGHTS
    weight: float = 1.0  # base weight for the edge type
    confidence: float = 1.0  # detection confidence (1.0 = detected by tree-sitter)

    def effective_weight(self) -> float:
        """Effective weight: type weight × confidence."""
        return self.weight * self.confidence


# ---------------------------------------------------------------------------
# Activation Graph — query‑conditioned node activation
# ---------------------------------------------------------------------------


class ActivationState(BaseModel):
    node_id: str
    score: float  # activation score [0.0, 1.0]
    depth: int  # depth from the nearest seed node
    source: str  # "seed" | "propagation"


class ActivationGraph:
    """
    Query‑conditioned activation graph.
    Created per query — never persisted across requests.
    """

    DECAY_BASE: float = 0.7
    # child_score = parent_score × edge.effective_weight × DECAY_BASE^depth

    def __init__(self):
        self._activations: Dict[str, ActivationState] = {}

    def seed(self, node_ids: List[str], initial_score: float = 1.0):
        """Activate seed nodes with an initial score."""
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
        max_steps: int = 20,  # PPR uses iterations, not BFS steps
        min_score: float = 0.05,
        alpha: float = 0.85,  # standard PageRank damping factor
        tolerance: float = 1e-6,
    ):
        """
        Personalized PageRank on the call graph.

        Equation: r = α * M * r + (1-α) * e
        where:
          M = transition matrix normalized by out-degree
          e = personalization vector (normalized seeds)
          α = probability of following an edge (vs. teleporting back to the seed)

        Advantages over BFS-decay:
        - Out-degree normalization: a node with 10 callees does not activate them
          10× stronger than one with 1 callee.
        - Guaranteed mathematical convergence even with cycles.
        - Teleportation: nodes disconnected from the seed receive a minimal score
          but not zero, avoiding false negatives.
        """
        if not self._activations:
            return

        # ── Personalization vector (normalized seeds) ───────────────
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

        # ── Out-degree normalization by weight ──────────────────────
        out_weight_total: Dict[str, float] = {}
        for src, edges in edges_out.items():
            total_w = sum(e.effective_weight() for e in edges)
            out_weight_total[src] = total_w if total_w > 0 else 1.0

        # ── Initialize r from personalization ───────────────────────
        r: Dict[str, float] = dict(personalization)

        # ── Power iteration ─────────────────────────────────────────
        for iteration in range(max_steps):
            r_new: Dict[str, float] = {}

            # Teleportation step: (1-α) * e
            for node, score in personalization.items():
                r_new[node] = (1.0 - alpha) * score

            # Propagation step: α * M * r
            for src, edges in edges_out.items():
                src_score = r.get(src, 0.0)
                if src_score < min_score:
                    continue
                out_w = out_weight_total.get(src, 1.0)
                for edge in edges:
                    contribution = alpha * src_score * edge.effective_weight() / out_w
                    r_new[edge.dst] = r_new.get(edge.dst, 0.0) + contribution

            # ── Convergence check ───────────────────────────────────
            all_keys = set(r.keys()) | set(r_new.keys())
            delta = sum(abs(r_new.get(k, 0.0) - r.get(k, 0.0)) for k in all_keys)
            r = r_new
            if delta < tolerance:
                # self._log_ppr_converged(iteration + 1)  # optional log
                break

        # ── Update activations with PPR scores ──────────────────────
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
        """Activation score of a node. 0.0 if not activated."""
        state = self._activations.get(node_id)
        return state.score if state else 0.0

    def get_activated_nodes(self, threshold: float = 0.1) -> Dict[str, float]:
        """Return {node_id: score} for nodes with score >= threshold."""
        return {
            nid: s.score for nid, s in self._activations.items() if s.score >= threshold
        }

    def aggregate_path_score(self, symbol_list: List[str]) -> float:
        """
        Aggregated score of a path.
        Average of the scores of the path's activated symbols.
        """
        scores = [self.get_score(s) for s in symbol_list]
        active = [s for s in scores if s > 0]
        if not active:
            return 0.0
        return sum(active) / len(active)  # penalises paths with many inactive symbols


# ---------------------------------------------------------------------------
# Query model and SubgraphExtractor skeleton
# ---------------------------------------------------------------------------


class SubgraphExtractor:
    """
    Extracts a relevant subgraph from the SymbolGraph given an ActivationGraph.
    Full implementation in PASO-08.
    """

    def __init__(
        self,
        activation_threshold: float = 0.1,
        expand_hops: int = 1,
    ):
        self.activation_threshold = activation_threshold
        self.expand_hops = expand_hops

    def extract(
        self,
        activation: ActivationGraph,
        edges_out: Dict[str, List[Edge]],
        edges_in: Dict[str, List[Edge]],
    ) -> Tuple[Set[str], List[Edge]]:
        """
        Extract a subgraph from the activation graph.

        Process:
        1. Include all nodes with score >= threshold.
        2. Expand 1-hop high-confidence neighbours (only 'calls' edges with weight ≥ 0.8).
        3. Include edges whose src and dst are both in the subgraph.
        """
        # Step 1: nodes above the threshold
        activated = activation.get_activated_nodes(self.activation_threshold)
        included_nodes: Set[str] = set(activated.keys())

        # Step 2: expand to 1-hop neighbours (high-confidence calls only)
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

        # Step 3: internal edges (both endpoints inside the subgraph)
        included_edges: List[Edge] = []
        for node_id in included_nodes:
            for edge in edges_out.get(node_id, []):
                if edge.dst in included_nodes:
                    included_edges.append(edge)

        return included_nodes, included_edges


# ---------------------------------------------------------------------------
# CodePathView — a cached projection of an activated subgraph (v7)
# ---------------------------------------------------------------------------


class CodePathView(BaseModel):
    path_id: str
    # hash(entry_point + "|" + "|".join(sorted(induced_nodes)))

    entry_point: str
    # Seed symbol that originated this view.

    seed_nodes: List[str]
    # Seed nodes that activated it (usually 1–3).

    induced_nodes: Dict[str, float]
    # {symbol_name: activation_score}
    # KEY CHANGE: each symbol carries its relevance weight.

    induced_edges: List[Edge]
    # Internal edges of the subgraph (both src and dst are in induced_nodes).

    activation_score: float
    # Aggregated score of the subgraph (weighted average).

    # ── Lazy semantic cache (populated by LLM when first needed) ──
    business_label: str = ""
    summary: str = ""
    label_confidence: float = 0.0

    # ── Invalidation hashes ──
    structural_hash: str = ""
    # hash(sorted(block_hashes of all symbols))
    # Changes when any symbol's content changes.

    call_graph_hash: str = ""
    # hash(sorted(edges as "src:type:dst"))
    # Changes when the call graph structure changes.

    last_built: float = Field(default_factory=time.time)

    def is_stale(
        self,
        current_structural: str,
        current_call_graph: str,
    ) -> bool:
        """True if either hash has changed since this view was built."""
        return (
            self.structural_hash != current_structural
            or self.call_graph_hash != current_call_graph
        )

    def top_symbols(self, n: int = 10) -> List[str]:
        """The N symbols with the highest activation scores."""
        return sorted(
            self.induced_nodes.keys(),
            key=lambda s: self.induced_nodes[s],
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# StaticEvidence – deterministic proof from the SymbolGraph (v7)
# ---------------------------------------------------------------------------


class StaticEvidence(BaseModel):
    symbols_found: Dict[str, bool]
    call_relations_valid: Dict[str, bool]
    recent_changes: List[str]
    entry_points_mentioned: List[str]
    path_memberships: Dict[str, List[str]]
    data_flow_upstream: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # ← v7 (PASO-19)
    objective_score: float


# ---------------------------------------------------------------------------
# PathIndex — index of CodePathViews (v7)
# ---------------------------------------------------------------------------


class PathIndex:
    """
    Index of CodePathViews.
    Analogous to SymbolIndex but for activated subgraph views.
    """

    def __init__(self):
        self._views: Dict[str, CodePathView] = {}
        # key = f"{project_id}:{path_id}"

        self._symbol_to_views: Dict[str, Set[str]] = defaultdict(set)
        # key = f"{project_id}:{symbol_name}" → {path_ids}

    # ── Basic operations ───────────────────────────────────────────

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

    # ── Invalidation ─────────────────────────────────────────────────

    def mark_stale_for_symbol(self, symbol_name: str, project_id: str) -> List[str]:
        """
        Return path_ids that contain this symbol.
        The caller must recompute hashes and clear summary/label if changed.
        """
        key = f"{project_id}:{symbol_name}"
        return list(self._symbol_to_views.get(key, set()))

    # ── Entry points ─────────────────────────────────────────────────

    def find_entry_points(self, symbol_index: SymbolIndex, project_id: str) -> Set[str]:
        """
        Symbols present in the index without any callers = potential entry points.
        """
        all_names = symbol_index.get_all_names(project_id)
        return {
            name for name in all_names if not symbol_index.get_callers(name, project_id)
        }


# DEPRECATED?
# ================================


class AppliedChangeFeedback(BaseModel):
    change_hash: str
    change_description: str
    file_path: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


class SecondaryTask(BaseModel):
    task_type: str
    params: Dict[str, Any]
    retries: int = 0
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tree‑sitter fallback queries
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
# Reentrant async lock
# ---------------------------------------------------------------------------
class ReentrantAsyncLock:
    """Reentrant asyncio lock with optional timeout to prevent deadlocks."""

    def __init__(self, default_timeout: float = 60.0):
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count = 0
        self._default_timeout = default_timeout

    async def acquire(self, timeout: Optional[float] = None):
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

    def release(self):
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

    async def __aexit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# In‑memory symbol index
# ---------------------------------------------------------------------------
class SymbolIndex:
    MAX_ENTRIES = 10_000

    def __init__(self):
        self._name_to_blocks: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._callee_to_callers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._stats: Counter = Counter()
        # Outgoing / incoming typed edges (added in v7 – PASO-04)
        self._edges_out: Dict[str, List[Edge]] = defaultdict(list)
        # key = f"{project_id}:{symbol_name}"  →  list of outgoing Edge objects

        self._edges_in: Dict[str, List[Edge]] = defaultdict(list)
        # key = f"{project_id}:{symbol_name}"  →  list of incoming Edge objects

    def _evict_if_needed(self):
        while len(self._name_to_blocks) > self.MAX_ENTRIES:
            least_common = self._stats.most_common()[-1][0]
            del self._name_to_blocks[least_common]
            del self._callee_to_callers[least_common]
            del self._stats[least_common]

    def add(self, symbol: CodeSymbol, block_hash: str, project_id: str):
        key = (project_id, symbol.name)
        self._name_to_blocks[key].add(block_hash)
        self._stats[key] += 1
        for callee in symbol.calls:
            callee_key = (project_id, callee)
            self._callee_to_callers[callee_key].add(symbol.name)
        self._evict_if_needed()

    def remove(self, symbol: CodeSymbol, block_hash: str, project_id: str):
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
        self, block_hash: str, symbols: List[CodeSymbol], project_id: str
    ):
        for sym in symbols:
            self.remove(sym, block_hash, project_id)
            self.remove_edges_for_symbol(sym.name, project_id)  # ← v7 (PASO-04)

    def find_blocks(self, name: str, project_id: str) -> Set[str]:
        return self._name_to_blocks.get((project_id, name), set())

    def get_all_names(self, project_id: str) -> Set[str]:
        return {key[1] for key in self._name_to_blocks if key[0] == project_id}

    def get_callers(self, callee_name: str, project_id: str) -> Set[str]:
        return self._callee_to_callers.get((project_id, callee_name), set())

    def add_edge(self, edge: Edge, project_id: str):
        """Register a typed edge in the index. Deduplicates by (src, dst, type)."""
        src_key = f"{project_id}:{edge.src}"
        dst_key = f"{project_id}:{edge.dst}"
        existing = self._edges_out.get(src_key, [])
        for e in existing:
            if e.dst == edge.dst and e.type == edge.type:
                return  # already registered
        self._edges_out[src_key].append(edge)
        self._edges_in[dst_key].append(edge)

    def remove_edges_for_symbol(self, symbol_name: str, project_id: str):
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

    def get_edges_out(self, symbol_name: str, project_id: str) -> List[Edge]:
        """Outgoing edges for a given symbol."""
        return self._edges_out.get(f"{project_id}:{symbol_name}", [])

    def get_edges_in(self, symbol_name: str, project_id: str) -> List[Edge]:
        """Incoming edges for a given symbol."""
        return self._edges_in.get(f"{project_id}:{symbol_name}", [])

    def get_all_edges_out(self, project_id: str) -> Dict[str, List[Edge]]:
        """
        Full outgoing edge map for a project.
        Used by ActivationGraph.propagate().
        Returns {symbol_name: [Edge, ...]}.
        """
        prefix = f"{project_id}:"
        return {
            key[len(prefix) :]: edges
            for key, edges in self._edges_out.items()
            if key.startswith(prefix)
        }

    def clear_project(self, project_id: str):
        # Remove name-to-blocks mappings
        keys_to_remove = [key for key in self._name_to_blocks if key[0] == project_id]
        for key in keys_to_remove:
            del self._name_to_blocks[key]
            del self._stats[key]

        # Remove callee-to-callers mappings
        inv_keys = [key for key in self._callee_to_callers if key[0] == project_id]
        for key in inv_keys:
            del self._callee_to_callers[key]

        # ── v7 (PASO-04): clean typed edges for this project ──
        prefix = f"{project_id}:"
        for k in list(self._edges_out.keys()):
            if k.startswith(prefix):
                del self._edges_out[k]
        for k in list(self._edges_in.keys()):
            if k.startswith(prefix):
                del self._edges_in[k]

    def clear(self):
        self._name_to_blocks.clear()
        self._callee_to_callers.clear()
        self._stats.clear()


# ---------------------------------------------------------------------------
# Signature Extractor (tree‑sitter with regex fallback)
# ---------------------------------------------------------------------------
class SignatureExtractor:
    MAX_PARSE_SIZE_BYTES = 5_000_000
    _LANG_MAP = {
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
    _parser_cache: Dict[str, "tree_sitter.Parser"] = {}
    _parser_cache_lock = threading.Lock()

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
    async def extract_async(
        code: str, file_path: Optional[str] = None
    ) -> List[CodeSymbol]:
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
    def _extract_symbols_from_tree(
        tree, lang: str, code: str, file_path: Optional[str]
    ) -> List[CodeSymbol]:
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
    def _extract_docstrings_python(code: str, symbols: List[CodeSymbol]):
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
    ) -> List[CodeSymbol]:
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


# ---------------------------------------------------------------------------
# Main Filter class
# ---------------------------------------------------------------------------
class Filter:

    class Valves(BaseModel):
        # ═══════════════════════════════════════════════════════════════
        #  Core
        # ═══════════════════════════════════════════════════════════════
        priority: int = Field(default=0)
        max_turns: int = Field(default=15)
        debug: bool = Field(default=True)
        debug_context: bool = Field(
            default=False,
            description="Print the full system message content at the end of the inlet for debugging.",
        )
        state_db_path: str = Field(default="/app/backend/data/conversation_state.db")
        track_line_numbers: bool = Field(default=True)
        adaptive_trim: bool = Field(default=True)
        context_window_tokens: int = Field(default=1000000)
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
        max_active_blocks: int = Field(
            default=0,
            ge=0,
            description="Maximum number of active code blocks to keep (0 = unlimited).",
        )
        track_file_paths: bool = Field(default=True)
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b"
        )
        max_code_block_tokens: int = Field(default=0)
        code_block_overflow_action: str = Field(default="warn")
        code_block_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        code_block_truncate_keep_head: int = Field(default=50)
        code_block_truncate_keep_tail: int = Field(default=50)
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]"
        )
        enable_call_graph_extraction: bool = Field(
            default=True,
            description="Extract call relationships (who calls whom) for code symbols.",
        )
        enable_auto_summaries: bool = Field(
            default=True,
            description="Automatically generate one-line summaries for code symbols.",
        )
        summary_code_max_chars: int = Field(
            default=8000,
            description="Maximum characters of code to include when summarizing code blocks.",
        )
        oversized_summary_max_tokens: int = Field(
            default=500,
            description="Max tokens for summarizing oversized code blocks.",
        )
        active_context_max_tokens: int = Field(
            default=32000,
            description="Maximum tokens for the injected active code context. 0 = unlimited.",
        )
        global_injection_token_budget: int = Field(
            default=0,
            description="Maximum tokens allowed for all system injections combined (0 = unlimited).",
        )
        exclude_filter_internals: bool = Field(
            default=True,
            description="Exclude symbols from the filter's own source code to prevent self-analysis.",
        )
        enable_ast_deduplication: bool = Field(
            default=True,
            description="Use AST comparison for Python code deduplication instead of text similarity.",
        )

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
        enable_path_analysis: bool = Field(
            default=True,
            description="Enable graph activation-based context selection.",
        )
        path_activation_threshold: float = Field(
            default=0.1,
            ge=0.01,
            le=1.0,
            description="Minimum activation score for a node to be considered for context.",
        )
        path_relevance_high_threshold: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Nodes with activation >= this get full code expansion; below get summary.",
        )
        path_propagation_steps: int = Field(
            default=4,
            ge=1,
            le=8,
            description="Max BFS/Power iteration steps during activation propagation.",
        )
        path_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
            description="Model for lazy path summary generation.",
        )
        path_summary_max_tokens: int = Field(
            default=80,
            description="Max tokens for a path summary.",
        )
        lod3_threshold: float = Field(
            default=0.50,
            ge=0.0,
            le=1.0,
            description="Activation score above which full code body is injected (LOD-3).",
        )
        lod2_threshold: float = Field(
            default=0.25,
            ge=0.0,
            le=1.0,
            description="Activation score above which signature + summary is injected (LOD-2).",
        )
        lod1_threshold: float = Field(
            default=0.10,
            ge=0.0,
            le=1.0,
            description="Activation score above which signature only is injected (LOD-1). Symbols below are name only (LOD-0).",
        )
        enable_centrality_prior: bool = Field(
            default=True,
            description="Compute static PageRank centrality on the call graph after code changes.",
        )
        enable_centrality_lod_bump: bool = Field(
            default=True,
            description="Boost LOD level for high-centrality symbols even when query-specific activation is low.",
        )
        centrality_lod_bump_threshold: float = Field(
            default=0.7,
            ge=0.0,
            le=1.0,
            description="Centrality score above which a symbol receives a LOD bump.",
        )
        centrality_lod_bump_weight: float = Field(
            default=0.15,
            ge=0.0,
            le=0.5,
            description="Weight of centrality boost (effective_score = activation + centrality * weight).",
        )
        enable_traceback_activation: bool = Field(
            default=True,
            description="Seed the ActivationGraph with traceback frame function names when present.",
        )
        enable_history_seeds: bool = Field(
            default=True,
            description="Boost activation of symbols frequently mentioned in recent conversation messages.",
        )
        history_seeds_lookback: int = Field(
            default=6,
            ge=2,
            le=20,
            description="Number of recent messages to scan for history seeds.",
        )
        history_seeds_max_boost: float = Field(
            default=0.6,
            ge=0.1,
            le=0.9,
            description="Maximum activation score boost from conversation history.",
        )
        enable_multi_seed_activation: bool = Field(
            default=True,
            description="Run three independent activation passes (lexical, structural, historical) and combine results.",
        )
        multi_seed_weight_lexical: float = Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Weight for lexical seeds.",
        )
        multi_seed_weight_structural: float = Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Weight for structural seeds.",
        )
        multi_seed_weight_historical: float = Field(
            default=0.2,
            ge=0.0,
            le=1.0,
            description="Weight for historical seeds.",
        )
        ppr_alpha: float = Field(
            default=0.85,
            ge=0.5,
            le=0.99,
            description="Damping factor for Personalized PageRank propagation.",
        )

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
        LLM_MAX_CONCURRENT_CALLS: int = Field(default=2, ge=1, le=10)
        llm_request_timeout: int = Field(default=900)
        LLM_CACHE_TTL: int = Field(default=300)
        LLM_CACHE_MAX_SIZE: int = Field(default=100)
        llamacpp_endpoint_type: str = Field(
            default="chat",
            description="Endpoint type for llama.cpp: 'chat' (default) or 'completion'.",
        )
        intent_classifier_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
            description="Model for intent classification fallback.",
        )
        enable_intent_llm_fallback: bool = Field(
            default=True,
            description="Use LLM as fallback when heuristic intent classification produces a weak signal.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Code Compression (LLMLingua-2)
        # ═══════════════════════════════════════════════════════════════
        enable_code_compression: bool = Field(
            default=False,
            description="Enable LLMLingua-2 token compression within code blocks. Requires llmlingua>=0.2.0.",
        )
        code_compression_rate: float = Field(
            default=0.5,
            ge=0.3,
            le=0.8,
            description="Fraction of tokens to KEEP after compression.",
        )
        code_compression_min_tokens: int = Field(
            default=150,
            description="Minimum token count for a code block to be compressed.",
        )
        enable_question_aware_compression: bool = Field(
            default=True,
            description="Pass the current user query to LLMLingua-2 to preserve query-relevant tokens.",
        )

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
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]'...",
        )
        enable_cot_on_demand: bool = Field(default=True)
        auto_cot_enabled: bool = Field(default=False)
        auto_cot_min_chars: int = Field(default=200)
        cot_max_tokens: int = Field(default=0)
        cot_model: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level2: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        cot_model_level3: str = Field(default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact")
        enable_cot_llm_detection: bool = Field(default=True)
        cot_detection_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        enforce_scientific_method: bool = Field(
            default=False,
            description="Force Level 3 Scientific CoT regardless of detected complexity.",
        )
        scientific_hypotheses_count: int = Field(
            default=3,
            ge=2,
            le=6,
            description="Number of hypotheses per round in Scientific CoT.",
        )
        scientific_confidence_threshold: float = Field(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="Stop iterating when best hypothesis reaches this score.",
        )
        scientific_max_iterations: int = Field(
            default=2,
            ge=1,
            le=4,
            description="Max refinement iterations in Level 3 Scientific CoT.",
        )
        enable_step_back_prompting: bool = Field(
            default=False,
            description="Before CoT, ask an abstract architectural question for better context.",
        )
        step_back_always: bool = Field(
            default=False,
            description="If True, generate step-back for all CoT queries (not just debugging).",
        )
        step_back_max_tokens: int = Field(
            default=150,
            ge=50,
            le=400,
            description="Max tokens for the step-back architectural context.",
        )
        enable_contradiction_detection: bool = Field(default=True)
        contradiction_detection_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        contradiction_inject_warning: bool = Field(default=True)
        enable_assumption_extraction: bool = Field(default=True)
        assumption_extraction_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Proactive Suggestions & Commands
        # ═══════════════════════════════════════════════════════════════
        proactive_context_warning_threshold: float = Field(default=0.85)
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Context Warning**: The conversation is using more than {percent}% of the available context window..."
        )
        proactive_summary_threshold: float = Field(default=0.75)
        proactive_summary_growth_window: int = Field(default=3)
        enable_command_suggestions: bool = Field(default=True)
        command_suggestion_cooldown_minutes: int = Field(default=10)
        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=True)
        natural_language_forget_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
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
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        summary_include_metadata: bool = Field(default=True)
        summarize_old_messages: bool = Field(default=True)
        summarization_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
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
        #  Inactive Code Summarization
        # ═══════════════════════════════════════════════════════════════
        summarize_inactive_code: bool = Field(default=True)
        inactive_code_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Session Summaries & Secondary Tasks
        # ═══════════════════════════════════════════════════════════════
        enable_session_summary: bool = Field(
            default=True,
            description="Generate an autobiographical session summary every N turns and store it in LTM.",
        )
        session_summary_interval_messages: int = Field(default=8)
        session_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        session_summary_max_tokens: int = Field(default=200)
        defer_secondary_tasks: bool = Field(
            default=True,
            description="Defer secondary LLM tasks to the next inlet to avoid concurrency.",
        )
        secondary_task_max_retries: int = Field(default=5)
        secondary_task_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
        )
        secondary_llm_max_concurrent: int = Field(default=2)

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
        enable_raptor: bool = Field(
            default=False,
            description="Enable RAPTOR hierarchical summarization of LTM. Requires scikit-learn.",
        )
        raptor_clusters_per_level: int = Field(
            default=5,
            ge=2,
            le=20,
            description="Number of k-means clusters per RAPTOR level.",
        )
        raptor_summary_model: str = Field(
            default="Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact",
            description="Model for RAPTOR cluster summary generation.",
        )
        raptor_summary_max_tokens: int = Field(
            default=150, description="Max tokens per RAPTOR cluster summary."
        )
        raptor_rebuild_interval: int = Field(
            default=20,
            description="Rebuild RAPTOR index every N outlet calls.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  KV Cache Stability & Slot Persistence
        # ═══════════════════════════════════════════════════════════════
        enable_kv_cache_stability: bool = Field(
            default=True,
            description="Separate system prompt into static (Block A) and dynamic (Block B) for KV cache stability.",
        )
        enable_slot_persistence: bool = Field(
            default=True,
            description="Persist and restore llama.cpp KV cache slot between sessions.",
        )
        slot_save_path: str = Field(
            default="/tmp/llama_slots",
            description="Directory for slot cache files.",
        )
        slot_id: int = Field(
            default=0, ge=0, description="llama.cpp slot ID to save/restore."
        )

        # ═══════════════════════════════════════════════════════════════
        #  Retrieval Enhancements
        # ═══════════════════════════════════════════════════════════════
        enable_contextual_retrieval: bool = Field(
            default=True,
            description="Prepend a context summary to each LTM entry before embedding.",
        )
        contextual_retrieval_mode: str = Field(
            default="metadata",
            description="Context generation mode: 'metadata' (fast, no LLM) or 'llm' (better, slower).",
        )
        enable_multi_query_retrieval: bool = Field(
            default=False,
            description="Generate multiple query variants before LTM retrieval and merge results.",
        )
        multi_query_variants: int = Field(
            default=2,
            ge=1,
            le=4,
            description="Number of alternative query variants to generate.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Edge Persistence (Cross-Session SymbolGraph)
        # ═══════════════════════════════════════════════════════════════
        enable_edge_persistence: bool = Field(
            default=True,
            description="Persist typed SymbolGraph edges to SQLite for cross-session continuity.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Adaptive LOD Thresholds
        # ═══════════════════════════════════════════════════════════════
        enable_lod_adaptive: bool = Field(
            default=True,
            description="Automatically adjust lod3_threshold based on LLM response references.",
        )
        lod_adapt_rate: float = Field(
            default=0.05,
            ge=0.01,
            le=0.2,
            description="Step size for each LOD threshold adjustment.",
        )
        lod_adapt_min: float = Field(
            default=0.25,
            ge=0.1,
            le=0.5,
            description="Minimum value for lod3_threshold.",
        )
        lod_adapt_max: float = Field(
            default=0.75,
            ge=0.5,
            le=0.95,
            description="Maximum value for lod3_threshold.",
        )
        lod_adapt_underserved_min: int = Field(
            default=2,
            ge=1,
            le=10,
            description="Minimum underserved symbols to trigger threshold decrease.",
        )
        lod_adapt_overserved_min: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Minimum overserved symbols to trigger threshold increase.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Speculative Pre‑fetching
        # ═══════════════════════════════════════════════════════════════
        enable_speculative_prefetch: bool = Field(
            default=True,
            description="Pre-build CodePathViews for symbols likely needed in the next query.",
        )
        speculative_prefetch_max: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum number of symbols to pre-fetch per response.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Silent Ingestion
        # ═══════════════════════════════════════════════════════════════
        enable_silent_ingestion: bool = Field(
            default=True,
            description="Index code silently when the user pastes code without a question.",
        )

        # ═══════════════════════════════════════════════════════════════
        #  Data Flow Analysis
        # ═══════════════════════════════════════════════════════════════
        enable_data_flow_analysis: bool = Field(
            default=True,
            description="Extract data flow edges between functions using ast analysis.",
        )

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

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(self):
        # Valves and basic objects
        self.valves = self.Valves()
        # ── Purge task tracker MUST be before _init_long_term_memory ──
        self._purge_task: Optional[asyncio.Task] = None

        self.tokenizer = None
        self._db_conn = None
        self._cross_encoder = None

        # ── LLMLingua-2 compressor (optional) ──
        self._llmlingua_compressor = None
        if self.valves.enable_code_compression:
            self._init_llmlingua()

        # Conversation state
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
            "pending_secondary_tasks": [],
            "last_cot_level": 0,
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
        self._init_state_db()

        # Long‑term memory (ChromaDB + embeddings)
        self.embedder = None
        self.chroma_client = None
        self.memory_collection = None
        self._response_cache_collection = None
        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._init_long_term_memory()
        else:
            logger.warning("Long‑term memory or code awareness disabled")

        # Reranker
        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._load_reranker()

        # HTTP session and locks
        self._project_locks: Dict[str, ReentrantAsyncLock] = {}
        self._lock_lock = asyncio.Lock()

        # Semaphores
        self._llm_semaphore = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENT_CALLS)
        self._low_priority_llm_semaphore = asyncio.Semaphore(1)
        self._secondary_llm_semaphore = asyncio.Semaphore(
            self.valves.secondary_llm_max_concurrent
        )
        self._pending_llm: Dict[str, asyncio.Future] = {}
        self._pending_llm_lock = asyncio.Lock()
        self._db_write_lock = asyncio.Lock()  # kept for extra safety, optional
        self._llm_cache = self._init_llm_cache()
        self._last_used_model: Optional[str] = None

        # ── Tracking of active LLM tasks (prevents model switching conflicts) ──
        self._active_llm_tasks: Set[asyncio.Task] = set()
        self._active_llm_tasks_lock = asyncio.Lock()

        # ── Database write queue (prevents "database is locked") ──
        self._db_write_queue: asyncio.Queue = asyncio.Queue()
        self._db_worker_task = asyncio.create_task(self._db_worker())

        # Background tasks tracking
        self._summarize_inactive_in_progress: Dict[str, bool] = {}
        self._write_counter = 0
        self._response_cache_cleanup_task: Optional[asyncio.Task] = None

        # Session classification cache
        self._session_classify_cache: Dict[str, Tuple[bool, float]] = {}
        self._session_classify_ttl: float = 1800.0

        # Symbol index and lightweight context
        self._symbol_index = SymbolIndex()
        self._path_index = PathIndex()  # v7 (PASO-06)
        self._node_centrality: Dict[str, Dict[str, float]] = {}  # v7 Phase 6 (PASO-33)
        self._cached_lightweight_context: Dict[str, str] = {}
        self._cached_code_state_hash: Optional[str] = None

        # ── KV Cache Stability (v7 – PASO-21) ──
        self._static_context_block_cache: Dict[str, Tuple[str, str]] = (
            {}
        )  # project_id → (code_state_hash, static_block_text)
        self._last_static_prefix_hash: Dict[str, str] = (
            {}
        )  # project_id → md5 del bloque estático de la última request

        # ── KV Cache Slot Persistence (v7 – PASO-25) ──
        self._last_saved_slot_hash: Dict[str, str] = (
            {}
        )  # project_id → static_block_hash of the last save to disk

        self._slot_restored: Dict[str, bool] = (
            {}
        )  # project_id → True if restore succeeded in this server session

        self._slot_restore_attempted: Dict[str, bool] = (
            {}
        )  # project_id → True if restore was already attempted (avoid retries)

        # Project tracking
        self._last_processed_message_idx: Dict[str, int] = {}
        self._last_project_id: str = ""
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}

        # Response cache counter
        self._response_cache_count: Dict[str, int] = {}

        # Batch LTM
        self._pending_ltm_messages: List[dict] = []
        self._ltm_batch_lock = asyncio.Lock()
        self._ltm_batch_task: Optional[asyncio.Task] = None

        # ── New: Symbol blacklist (empty by default) ──
        self._SYMBOL_BLACKLIST: Set[str] = set()

        # ── New: LRU-ordered cache for block change summaries (max size from valves) ──
        self._block_change_summaries: OrderedDict = OrderedDict()
        self._MAX_CHANGE_SUMMARIES = self.valves.max_change_summaries

        # ── New: Dedicated thread pools for blocking DB and ChromaDB operations ──
        import concurrent.futures

        self._db_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="codeaware_db"
        )
        self._chroma_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="codeaware_chroma"
        )

        # ── CoT heuristic feature flags ──
        self.ENABLE_ACCENT_NORMALIZATION = True  # normalize Spanish accents in keywords
        self.ENABLE_KEYWORD_COUNT_WEIGHT = True  # multiple keywords increase signals
        self.ENABLE_COT_STICKY = False  # keep last CoT level in conversation state

        # ── State debounce (to reduce DB writes) ──
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

    def _background_task(self, coro, name: str = "task", is_llm_task: bool = False):
        """Create a background task that logs errors, optionally tracking for cancellation."""
        task = asyncio.create_task(coro)

        def _on_done_log(t):
            if t.cancelled():
                return
            if t.exception():
                self._log_debug(f"Background task '{name}' failed: {t.exception()}")

        task.add_done_callback(_on_done_log)

        if is_llm_task:
            tasks_list = _inlet_background_tasks.get(None)
            if tasks_list is not None:
                tasks_list.append(task)

                def _remove_from_list(t):
                    try:
                        tasks_list.remove(t)
                    except ValueError:
                        pass

                task.add_done_callback(_remove_from_list)

        return task

    # --------------------------------------------------------------------------
    # Code extraction and classification
    # --------------------------------------------------------------------------
    async def _extract_code_blocks(
        self, content: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        blocks = []
        spans = []
        if not self.valves.auto_detect_code_blocks:
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
                    code = await self._handle_oversized_code_block(code, lang)
                    blocks.append({"language": lang, "code": code, "type": block_type})
                    spans.append((start, end))
                if blocks:
                    return blocks, spans
            except Exception:
                pass

        # Regex fallback
        for match in self.code_pattern.finditer(content):
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
            if self.valves.track_file_paths and spans:
                blk_file = self._extract_file_path_for_block(content, spans[idx][0])
            if not blk_file and len(blocks) == 1:
                extracted_paths = self._extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None

            # === Exclude blocks that belong to the filter's own source code ===
            if self.valves.exclude_filter_internals and blk_file:
                if (
                    "/app/backend/data/functions/" in blk_file
                    or "open-webui/functions/" in blk_file
                ):
                    continue  # skip this block entirely
            # ================================================================

            block["code"] = await self._handle_oversized_code_block(
                block["code"], block["language"]
            )
            block["file_path"] = blk_file
            processed_blocks.append(block)
            processed_spans.append(spans[idx])

        return processed_blocks, processed_spans

    async def _infer_code_language(self, code_snippet: str) -> str:
        # Simple heuristic first
        if re.search(r"\bdef\s+\w+\s*\(", code_snippet):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(", code_snippet):
            return "javascript"
        return "unknown"

    def _extract_file_path_for_block(
        self, content: str, block_start: int
    ) -> Optional[str]:
        if block_start <= 0:
            return None
        before = content[:block_start]
        lines = before.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            match = re.search(self.valves.file_path_pattern, line)
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
        if not self.valves.track_line_numbers:
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

    def _extract_file_paths(self, content: str) -> List[str]:
        if not self.valves.track_file_paths:
            return []
        matches = re.findall(self.valves.file_path_pattern, content)
        return [m[0] if isinstance(m, tuple) else m for m in matches]

    # --------------------------------------------------------------------------
    # Code span utilities
    # --------------------------------------------------------------------------
    async def _get_code_spans(self, content: str) -> List[Tuple[int, int]]:
        if not HAS_TREE_SITTER:
            return []
        cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
        if cache_key in self._code_spans_cache:
            return self._code_spans_cache[cache_key]
        try:
            config = ProcessConfig()
            blocks = process(content, config)
            spans = [(b.start_byte, b.end_byte) for b in blocks]
        except Exception as e:
            if any(
                msg in str(e)
                for msg in (
                    "Language '' not available",
                    "Language '' not available for download",
                    "Download error: Language ''",
                )
            ):
                spans = []
            else:
                self._log_debug(f"Tree‑sitter process failed: {e}")
                spans = []
        if len(self._code_spans_cache) >= 200:
            keys_to_evict = list(self._code_spans_cache.keys())[:50]
            for key in keys_to_evict:
                del self._code_spans_cache[key]
        self._code_spans_cache[cache_key] = spans
        return spans

    def _remove_code_spans(self, content: str, spans: List[Tuple[int, int]]) -> str:
        chars = list(content)
        for start, end in spans:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        return "".join(chars)

    def _classify_content(
        self, content: str, extracted_blocks: List[Dict]
    ) -> ContentType:
        cl = content.lower()
        if self.diff_pattern.search(content) or "diff --git" in content:
            return ContentType.PROPOSED_CHANGE
        if self.commit_pattern.search(content):
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

    def _has_code_indicators(self, content: str) -> bool:
        if "```" in content:
            return True
        if re.search(
            r"\b(def |class |import |from |function |const |let |var |#include |package |fn |func )",
            content,
        ):
            return True
        if re.search(
            r"\b[\w\-/]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp)\b", content
        ):
            return True
        return False

    async def _is_code_only_message(self, content: str) -> bool:
        """
        Detect messages that contain only code without a question.
        Heuristic: if the text outside fenced code blocks is < 30 chars,
        treat as pure ingestion.

        Returns True for:
          ```python\ndef foo(): pass\n```
        Returns False for:
          What does this function do?\n```python\ndef foo(): pass\n```
        """
        if not content or len(content.strip()) < 20:
            return False

        # Must contain at least one code block
        code_blocks, _ = await self._extract_code_blocks(content)
        if not code_blocks:
            return False

        # Calculate text outside code blocks
        spans = await self._get_code_spans(content)
        if not spans:
            # No tree-sitter spans: use regex to remove fenced blocks
            text_outside = re.sub(r"```[\s\S]*?```", "", content).strip()
        else:
            text_outside = self._remove_code_spans(content, spans).strip()

        return len(text_outside) < 30

    async def _classify_session(self, messages: List[dict], project_id: str) -> bool:
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        cache_key = None
        if last_user:
            content_key = last_user.get("content", "")[:200]
            cache_key = (
                f"{project_id}:{hashlib.md5(content_key.encode()).hexdigest()[:12]}"
            )
            cached = self._session_classify_cache.get(cache_key)
            if cached is not None:
                result, ts = cached
                if time.time() - ts < self._session_classify_ttl:
                    return result
                del self._session_classify_cache[cache_key]
        state = self._get_state(project_id)
        if state and state.get("active_blocks"):
            if cache_key:
                self._session_classify_cache[cache_key] = (True, time.time())
            return True
        for msg in reversed(messages[-10:]):
            if msg.get("role") != "user":
                continue
            if self._has_code_indicators(msg.get("content", "")):
                if cache_key:
                    self._session_classify_cache[cache_key] = (True, time.time())
                return True
        if last_user and last_user.get("content", "").strip().startswith("/"):
            if cache_key:
                self._session_classify_cache[cache_key] = (True, time.time())
            return True
        if last_user and "```" in last_user.get("content", ""):
            if cache_key:
                self._session_classify_cache[cache_key] = (True, time.time())
            return True
        if (
            last_user
            and not state.get("active_blocks")
            and not self._has_code_indicators(last_user.get("content", ""))
        ):
            if len(last_user.get("content", "")) > 200:
                blocks, _ = await self._extract_code_blocks(
                    last_user.get("content", "")
                )
                if blocks:
                    if cache_key:
                        self._session_classify_cache[cache_key] = (True, time.time())
                    return True
        if not last_user or len(last_user.get("content", "")) < 20:
            result = False
        else:
            model = self.valves.natural_language_forget_model or self.valves.llm_model
            prompt = f"Is this message about programming or code? Answer only 'yes' or 'no'.\n\nMessage: {last_user.get('content','')[:300]}"
            response = await self._try_llm_quick(
                prompt=prompt,
                system_prompt="You are a classifier. Answer only 'yes' or 'no'.",
                model_override=model,
                max_tokens=3,
                temperature=0.0,
            )
            result = bool(response and response.strip().lower().startswith("yes"))
        if cache_key:
            self._session_classify_cache[cache_key] = (result, time.time())
            if len(self._session_classify_cache) >= 500:
                items = sorted(
                    self._session_classify_cache.items(), key=lambda x: x[1][1]
                )
                self._session_classify_cache = dict(items[-400:])
        return result

    # --------------------------------------------------------------------------
    # Context formatting and lightweight context
    # --------------------------------------------------------------------------
    def _format_block_context(
        self, block: CodeBlock, is_latest: bool = False, full_body: bool = False
    ) -> str:
        timestamp_str = datetime.fromtimestamp(
            block.timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        latest = " [LATEST]" if is_latest else ""
        loc = ""
        if block.file_path:
            loc = f" (file: {block.file_path}"
            if block.line_range:
                loc += f", lines {block.line_range[0]}-{block.line_range[1]}"
            loc += ")"
        pin = " [PINNED]" if block.pinned else ""
        raw = " [RAW]" if block.is_raw else ""
        show_full = full_body or block.is_raw or block.pinned
        content = block.content if show_full else block.content[:600]
        return (
            f"```\n{content}\n```{loc}{latest}  "
            f"(importance: {block.importance_score:.1f}, modified: {timestamp_str})"
            f"{pin}{raw}"
        )

    def _get_active_code_context(self, project_id: str, user_query: str = "") -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        now = time.time()
        active = []
        for block in state["active_blocks"].values():
            if block.obsolete:
                continue
            if not block.is_active and self.valves.track_active_code_age:
                if now - block.timestamp > self.valves.active_code_timeout_minutes * 60:
                    continue
            active.append(block)
        if not active:
            return ""

        recent_window = self.valves.recent_activity_window_minutes * 60
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
                f"## Recent Activity (last {self.valves.recent_activity_window_minutes} min)\n"
                + "\n".join(recent_lines)
                + "\n\n"
            )

        mentioned_files = set()
        mentioned_symbols = set()
        if user_query:
            mentioned_files = set(re.findall(self.valves.file_path_pattern, user_query))
            all_symbol_names = self._symbol_index.get_all_names(project_id)
            words = set(re.findall(r"\b[\w-]+\b", user_query))
            mentioned_symbols = all_symbol_names.intersection(words)

        BOOST = 5.0

        def relevance_boost(block: CodeBlock) -> float:
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
        boost_priority = self.valves.raw_file_priority_boost
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
        ][: self.valves.max_base_code_blocks]
        proposed = [
            b
            for b, _ in boosted_active
            if b.content_type == ContentType.PROPOSED_CHANGE
        ][: self.valves.max_proposed_changes]
        committed = [
            b
            for b, _ in boosted_active
            if b.content_type == ContentType.COMMITTED_CHANGE
        ][: self.valves.max_committed_changes]
        errors = (
            [b for b, _ in boosted_active if b.content_type == ContentType.ERROR][:3]
            if self.valves.preserve_error_context
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
                parts.append(self._format_block_context(b, is_latest) + tag)
        if proposed:
            parts.append("### Proposed Changes (pending review):")
            for b in proposed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._format_block_context(b, is_latest) + tag)
        if committed:
            parts.append("### Recently Committed Changes:")
            for b in committed:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._format_block_context(b, is_latest) + tag)
        if errors:
            parts.append("### Recent Errors:")
            for b in errors:
                is_latest = b.hash in latest_hashes
                tag = " [RELEVANT]" if relevance_boost(b) > 0 else ""
                parts.append(self._format_block_context(b, is_latest) + tag)

        max_tokens = self.valves.active_context_max_tokens
        if max_tokens > 0 and self.tokenizer:
            full_text = "\n".join(parts)
            while len(self.tokenizer.encode(full_text)) > max_tokens and len(parts) > 3:
                parts.pop()
                full_text = "\n".join(parts)
            if len(self.tokenizer.encode(full_text)) > max_tokens:
                parts.append(f"[Context truncated to fit token limit ({max_tokens})]")
        return "\n".join(parts)

    async def _build_lightweight_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        if project_id in self._cached_lightweight_context:
            return self._cached_lightweight_context[project_id]

        lines = ["## Code Symbol Index (full bodies available on request)\n"]
        grouped: Dict[str, List[CodeSymbol]] = defaultdict(list)
        for block in state["active_blocks"].values():
            if block.obsolete:
                continue
            for sym in block.symbols:
                key = sym.file_path or "unknown"
                grouped[key].append(sym)

        for file_path, syms in grouped.items():
            lines.append(f"### {file_path}")
            for s in sorted(syms, key=lambda x: x.name.lower()):
                loc = (
                    f" (lines {s.line_start}-{s.line_end})"
                    if s.line_start and s.line_end
                    else ""
                )
                calls_str = ""
                if s.calls:
                    calls_list = ", ".join(s.calls[:5])
                    if len(s.calls) > 5:
                        calls_list += f", ... (+{len(s.calls)-5} more)"
                    calls_str = f" → calls: {calls_list}"
                used_by = self._symbol_index.get_callers(s.name, project_id)
                if used_by:
                    ub_list = ", ".join(sorted(used_by)[:5])
                    if len(used_by) > 5:
                        ub_list += f", ... (+{len(used_by)-5} more)"
                    used_str = f"  ← used by: {ub_list}"
                else:
                    used_str = ""
                summary_str = f"  Summary: {s.summary}" if s.summary else ""
                num_versions = len(self._symbol_index.find_blocks(s.name, project_id))
                version_info = (
                    f" ({num_versions} versions available)" if num_versions > 1 else ""
                )
                lines.append(
                    f"- `{s.signature}` [{s.kind}]{loc}{calls_str}{used_str}{summary_str}{version_info}"
                )

        lines.append("\n## Code Previews (first 10 lines of each block)\n")
        max_preview_chars = 4000
        chars_added = 0
        for block in state["active_blocks"].values():
            if block.obsolete:
                continue
            preview = "\n".join(block.content.splitlines()[:10])
            if chars_added + len(preview) > max_preview_chars:
                lines.append("```\n... (previews truncated)\n```")
                break
            loc = f" (file: {block.file_path})" if block.file_path else ""
            lines.append(f"### Block {block.hash[:8]}{loc}\n```\n{preview}\n```")
            chars_added += len(preview)

        lines.append(
            "\nTo see the full body of any symbol, mention it in your message or use `/expand <name>`."
        )
        result = "\n".join(lines)
        self._cached_lightweight_context[project_id] = result
        return result

    # --------------------------------------------------------------------------
    # Message helpers
    # --------------------------------------------------------------------------
    def _ensure_last_message_is_user(self, messages: List[dict]) -> List[dict]:
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

    def _estimate_tokens(self, messages: List[dict]) -> int:
        if self.tokenizer:
            total = 0
            for m in messages:
                content = str(m.get("content", ""))
                total += len(self.tokenizer.encode(content))
                total += 4
            return total
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        total_chars += sum(30 for _ in messages)
        return total_chars // 4

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        if not self.tokenizer:
            return text[: max_tokens * 4]
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = self.tokenizer.decode(tokens[:max_tokens])
        for pattern in ("\n\n", "\n", ". ", " "):
            last = truncated.rfind(pattern)
            if last > max_tokens * 0.6:
                truncated = truncated[: last + len(pattern)]
                break
        return truncated.rstrip()

    # --------------------------------------------------------------------------
    # State database
    # --------------------------------------------------------------------------
    async def _db_worker(self):
        """Serialize all database writes through a single task with global lock and retry."""
        try:
            while True:
                try:
                    job = await asyncio.wait_for(
                        self._db_write_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

                func, args, kwargs = job
                # Acquire global lock to ensure only one write at a time across all workers
                with _db_global_lock:
                    for attempt in range(5):
                        try:
                            await anyio.to_thread.run_sync(
                                lambda: func(*args, **kwargs)
                            )
                            break
                        except sqlite3.OperationalError as e:
                            if "locked" in str(e).lower() and attempt < 4:
                                await asyncio.sleep(0.5 * (attempt + 1))
                            else:
                                raise
        except asyncio.CancelledError:
            pass
        finally:
            self._log_debug("DB worker exiting")

    def _init_state_db(self):
        db_path = self.valves.state_db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds wait on lock
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                project_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS code_contents (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS block_change_summaries (
                block_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # ── v7 (PASO-13): tables for CodePathView and typed edges ──
        self._db_conn.execute("""
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
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cpv_project "
            "ON code_path_views(project_id)"
        )

        self._db_conn.execute("""
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
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_project_src "
            "ON symbol_edges(project_id, src)"
        )
        self._db_conn.execute("""
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
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raptor_project_level "
            "ON raptor_clusters(project_id, level)"
        )
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_edges_meta (
                project_id      TEXT PRIMARY KEY,
                code_state_hash TEXT NOT NULL,
                edge_count      INTEGER NOT NULL DEFAULT 0,
                saved_at        REAL NOT NULL
            )
        """)
        self._db_conn.commit()

    def _get_project_id(self) -> str:
        return self.valves.project_id

    async def _get_project_lock(self, project_id: str) -> ReentrantAsyncLock:
        async with self._lock_lock:
            if project_id not in self._project_locks:
                self._project_locks[project_id] = ReentrantAsyncLock()
            return self._project_locks[project_id]

    def _get_state(self, project_id: str) -> Dict:
        if project_id in self._conversation_state:
            self._conversation_state.move_to_end(project_id)
            return self._conversation_state[project_id]
        state = self._load_state_from_db(project_id)
        if not state:
            state = self._state_factory()
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        while len(self._conversation_state) > self.valves.max_cached_projects:
            oldest_pid = next(iter(self._conversation_state))
            oldest_state = self._conversation_state[oldest_pid]
            self._remove_project_from_index_by_id(oldest_pid, oldest_state)
            del self._conversation_state[oldest_pid]
            self._cached_lightweight_context.pop(oldest_pid, None)
            self._project_locks.pop(oldest_pid, None)
        if state["active_blocks"]:
            self._rebuild_symbol_index(state, project_id)
        return state

    def _set_state(self, project_id: str, state: Dict):
        """Mark the conversation state as dirty without persisting immediately."""
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        self._state_dirty = True

    async def _save_state_if_dirty(self, project_id: str):
        """
        Persist the state if dirty and at least 2 seconds have passed since last save.
        Waits for the DB write to complete, logging any error.
        """
        if not self._state_dirty:
            return
        now = time.time()
        if now - self._state_last_saved < 2.0:
            return

        self._state_last_saved = now
        self._state_dirty = False

        try:
            await self._save_state_to_db_async(
                project_id, self._conversation_state[project_id]
            )
        except Exception as e:
            import traceback

            self._log_debug(f"Failed to save state: {e}\n{traceback.format_exc()}")

    async def _save_state_to_db(self, project_id: str, state: Dict):
        active_blocks_meta = {}
        for k, v in state["active_blocks"].items():
            d = v.dict()
            d["content_type"] = v.content_type.value
            content_hash = v.hash
            await anyio.to_thread.run_sync(
                lambda: self._db_conn.execute(
                    "INSERT OR IGNORE INTO code_contents (hash, content, created_at) VALUES (?, ?, ?)",
                    (content_hash, v.content, time.time()),
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
            "pending_secondary_tasks": state.get("pending_secondary_tasks", []),
            "last_cot_level": state.get("last_cot_level", 0),
        }

        def _write():
            self._db_conn.execute(
                "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
                (project_id, json.dumps(serializable), time.time()),
            )
            self._db_conn.commit()

        await self._db_write_queue.put((_write, (), {}))

    async def _save_state_to_db_async(self, project_id: str, state: Dict):
        """Acquire the project lock, then persist the state to DB."""
        lock = await self._get_project_lock(project_id)
        async with lock:
            await self._save_state_to_db(project_id, state)

    def _load_state_from_db(self, project_id: str) -> Optional[Dict]:
        cur = self._db_conn.execute(
            "SELECT state_json FROM conversation_state WHERE project_id = ?",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        for key in [
            "feedback_history",
            "last_compression_timestamp",
            "last_suggestion_timestamp",
            "response_cache",
            "has_any_calls",
            "pending_secondary_tasks",
            "last_cot_level",
        ]:
            data.setdefault(
                key,
                (
                    []
                    if key
                    in ("feedback_history", "response_cache", "pending_secondary_tasks")
                    else 0
                ),
            )
        data.setdefault("last_cleanup_suggestion_msg_idx", 0)
        active = {}
        for k, v in data.get("active_blocks", {}).items():
            try:
                content_field = v.get("content", "")
                if content_field.startswith("@@hash:"):
                    content_hash = content_field[7:]
                    cur2 = self._db_conn.execute(
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
                self._log_debug(f"Skipping corrupted block {k} in state DB")
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
            "pending_secondary_tasks": data.get("pending_secondary_tasks", []),
            "last_cot_level": data.get("last_cot_level", 0),
        }
        for blk in (
            list(state["active_blocks"].values())
            + state["recent_changes"]
            + state["committed_changes"]
        ):
            if self.tokenizer:
                blk._cached_token_count = len(self.tokenizer.encode(blk.content))
            else:
                blk._cached_token_count = len(blk.content) // 4
        return state

    # ── v7 (PASO-13): CodePathView persistence ─────────────────────

    async def _save_path_views_to_db(self, project_id: str, views: List[CodePathView]):
        def _write():
            self._db_conn.execute(
                "DELETE FROM code_path_views WHERE project_id = ?", (project_id,)
            )
            for v in views:
                self._db_conn.execute(
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
            self._db_conn.commit()

        await self._db_write_queue.put((_write, (), {}))

    async def _load_path_views_from_db(self, project_id: str) -> List[CodePathView]:
        rows = await anyio.to_thread.run_sync(
            lambda: self._db_conn.execute(
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
                self._log_debug(f"Skipping corrupt CodePathView: {e}")
        return views

    # --------------------------------------------------------------------------
    # SymbolGraph edge persistence (v7 – Phase 5, PASO-28)
    # --------------------------------------------------------------------------

    async def _save_symbol_edges_to_db(self, project_id: str) -> int:
        """
        Persist the typed edges from the SymbolIndex to SQLite.
        Saves alongside the current code_state_hash for invalidation detection.
        Returns the number of edges saved.
        """
        if not self.valves.enable_edge_persistence:
            return 0

        edges_out = self._symbol_index.get_all_edges_out(project_id)
        if not edges_out:
            return 0

        code_hash = self._compute_code_state_hash(project_id)
        if not code_hash:
            return 0  # no active code, nothing to persist

        total_edges = sum(len(edges) for edges in edges_out.values())

        def _write():
            # Clear previous edges for this project
            self._db_conn.execute(
                "DELETE FROM symbol_edges WHERE project_id = ?", (project_id,)
            )
            # Insert all current edges
            for src, edges in edges_out.items():
                for edge in edges:
                    self._db_conn.execute(
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
            # Update metadata
            self._db_conn.execute(
                "INSERT OR REPLACE INTO symbol_edges_meta "
                "(project_id, code_state_hash, edge_count, saved_at) "
                "VALUES (?,?,?,?)",
                (project_id, code_hash, total_edges, time.time()),
            )
            self._db_conn.commit()

        await self._db_write_queue.put((_write, (), {}))
        self._log_debug(
            f"Edge persistence: saved {total_edges} edges " f"(code_hash={code_hash})"
        )
        return total_edges

    async def _load_symbol_edges_from_db(self, project_id: str) -> int:
        """
        Restore typed edges from SQLite.
        Only restores if the saved code_state_hash matches the current state.
        Returns the number of edges restored (0 if stale or no data).
        """
        if not self.valves.enable_edge_persistence:
            return 0

        # Skip if edges are already loaded in memory
        existing = self._symbol_index.get_all_edges_out(project_id)
        if existing:
            return 0

        current_code_hash = self._compute_code_state_hash(project_id)
        if not current_code_hash:
            return 0  # no active code

        # Check if the saved hash matches
        meta_row = await anyio.to_thread.run_sync(
            lambda: self._db_conn.execute(
                "SELECT code_state_hash, edge_count FROM symbol_edges_meta "
                "WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        )

        if not meta_row:
            self._log_debug("Edge persistence: no saved edges for this project")
            return 0

        saved_hash, saved_count = meta_row
        if saved_hash != current_code_hash:
            self._log_debug(
                f"Edge persistence: stale edges detected "
                f"(saved={saved_hash}, current={current_code_hash}). "
                f"Edges will be rebuilt when code is processed."
            )
            return 0

        # Hash matches → restore edges
        rows = await anyio.to_thread.run_sync(
            lambda: self._db_conn.execute(
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
            self._symbol_index.add_edge(edge, project_id)
            count += 1

        self._log_debug(
            f"✓ Edge persistence: restored {count} edges "
            f"(code_hash={current_code_hash})"
        )
        return count

    # --------------------------------------------------------------------------
    # Symbol index helpers
    # --------------------------------------------------------------------------
    def _rebuild_symbol_index(self, state: Dict, project_id: str):
        self._symbol_index.clear_project(project_id)
        has_any = False
        for block in state["active_blocks"].values():
            for sym in block.symbols:
                self._symbol_index.add(sym, block.hash, project_id)
                if sym.calls:
                    has_any = True
        state["has_any_calls"] = has_any

    def _remove_project_from_index_by_id(self, project_id: str, state: Dict):
        for block in state["active_blocks"].values():
            self._symbol_index.remove_all_for_block(
                block.hash, block.symbols, project_id
            )

    def _invalidate_lightweight_cache(self, project_id: str):
        self._cached_lightweight_context.pop(project_id, None)
        self._cached_code_state_hash = None
        # ── Phase 6 (PASO-33): centrality stale when code changes ──
        self._node_centrality.pop(project_id, None)

    # --------------------------------------------------------------------------
    # Long‑term memory initialization
    # --------------------------------------------------------------------------
    def _init_long_term_memory(self):
        os.makedirs(self.valves.long_term_memory_dir, exist_ok=True)
        self.embedder = _shared_get_embedder()
        self._log_debug("Embedder: using shared singleton")

        self.chroma_client = _shared_get_chroma_client(self.valves.long_term_memory_dir)
        self._log_debug("ChromaDB: using shared singleton")

        if self.chroma_client is None:
            self._log_debug("ChromaDB not available")
            return

        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="conversation_memory", metadata={"hnsw:space": "cosine"}
        )
        self._response_cache_collection = self.chroma_client.get_or_create_collection(
            name=f"response_cache_{self.valves.project_id or 'default'}",
            metadata={"hnsw:space": "cosine"},
        )
        # ── Only launch purge task if not already active ──
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if self._purge_task is None or self._purge_task.done():
                self._purge_task = asyncio.create_task(self._purge_expired_memories())
        self._log_debug("LTM ready")

    async def _purge_expired_memories(self):
        await asyncio.sleep(0)  # yield to event loop to avoid blocking
        if not HAS_CHROMA or self.memory_collection is None:
            return
        if self.valves.long_term_memory_expiration_days <= 0:
            return
        try:
            await anyio.to_thread.run_sync(self._do_purge)
        except Exception as e:
            logger.warning(f"Purge failed: {e}")

    def _do_purge(self):
        now = time.time()
        expired = self.memory_collection.get(where={"expires_at": {"$lt": now}})
        if expired and expired["ids"]:
            self.memory_collection.delete(ids=expired["ids"])
            self._log_debug(f"Purged {len(expired['ids'])} expired memories")

    # --------------------------------------------------------------------------
    # LLM calling infrastructure
    # --------------------------------------------------------------------------
    def _init_llm_cache(self):
        """Return the shared AsyncLRUCache instance for LLM response caching."""
        return _AsyncLRUCache(
            max_size=self.valves.LLM_CACHE_MAX_SIZE,
            ttl=self.valves.LLM_CACHE_TTL,
        )

    async def _maybe_unload_for_model(
        self, model_name: str, base_url: str, is_ollama: bool
    ) -> None:
        """
        Unload models only if switching to a *different* auxiliary model.
        The main model (self.valves.llm_model) is NEVER unloaded to preserve its KV cache.
        """
        if is_ollama:
            return

        main_model = self.valves.llm_model

        # If the target is the main model, never unload anything – it must stay in VRAM.
        if model_name == main_model:
            if self._last_used_model is None:
                self._log_debug(f"Loading main model '{model_name}' for the first time")
            else:
                self._log_debug(f"Keeping main model '{model_name}' loaded (no unload)")
            return

        # Target is an auxiliary model.
        # If the currently loaded model is the main model, do NOT unload it.
        if self._last_used_model == main_model:
            self._log_debug(
                f"Keeping main model '{main_model}' loaded while loading auxiliary '{model_name}'"
            )
            return

        # If we are switching between two different auxiliary models, unload the old one.
        if self._last_used_model is not None and model_name != self._last_used_model:
            self._log_debug(
                f"Switching auxiliary model from '{self._last_used_model}' to '{model_name}'"
            )
            try:
                await _shared_unload_all_models(base_url)
                self._log_debug("Auxiliary model unloaded before switching")
                self._last_used_model = None
            except Exception as e:
                self._log_debug(f"Unload via shared_resources failed: {e}")
        elif self._last_used_model is None:
            self._log_debug(
                f"Loading auxiliary model '{model_name}' (no model was loaded)"
            )
        else:
            self._log_debug(f"Reusing auxiliary model '{model_name}' (already loaded)")

    async def _acquire_llm_lock(self):
        """Acquire an inter‑process file lock for exclusive LLM access."""
        loop = asyncio.get_event_loop()
        fd = open(_llm_lock_path, "w")
        await loop.run_in_executor(self._db_executor, fcntl.flock, fd, fcntl.LOCK_EX)
        return fd

    @staticmethod
    def _release_llm_lock(fd):
        """Release the inter‑process file lock and close the file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

    async def _wait_for_llm_tasks(self):
        """Block until all LLM-using tasks have completed."""
        while True:
            async with self._active_llm_tasks_lock:
                if not self._active_llm_tasks:
                    break
                tasks = list(self._active_llm_tasks)
            if tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def _wait_for_empty_slot(self, retries: int = 3, delay: float = 2.0) -> bool:
        """
        Check that the LLM server has no loaded models.
        Retries a few times with a delay between checks.
        Returns True if the slot is empty, False otherwise.
        """
        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        for attempt in range(retries):
            await asyncio.sleep(delay)
            try:
                session = await _shared_get_http_session(timeout_seconds=5)
                async with session.get(f"{base_url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        loaded = any(
                            m.get("status", {}).get("value") == "loaded"
                            for m in data.get("data", [])
                        )
                        if not loaded:
                            return True
                        else:
                            self._log_debug(
                                f"Models still loaded: {[m['id'] for m in data.get('data', []) if m.get('status', {}).get('value') == 'loaded']}"
                            )
                    else:
                        self._log_debug(f"Model list returned status {resp.status}")
            except Exception as e:
                self._log_debug(f"Error checking models (attempt {attempt+1}): {e}")

        self._log_debug(f"Slot still occupied after {retries} retries")
        return False

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: str = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        semaphore: asyncio.Semaphore = None,
        response_format: Optional[Dict[str, Any]] = None,
        label: str = "",
    ) -> Optional[str]:
        dedup_key = hashlib.md5(
            f"{prompt}|{system_prompt}|{temperature}|{max_tokens}|{model_override}".encode()
        ).hexdigest()
        async with self._pending_llm_lock:
            if dedup_key in self._pending_llm:
                future = self._pending_llm[dedup_key]
                is_producer = False
            else:
                future = asyncio.Future()
                self._pending_llm[dedup_key] = future
                is_producer = True
        if not is_producer:
            return await future

        t_start = time.monotonic()
        try:
            base_url = self.valves.LLM_BASE_URL.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3].rstrip("/")

            is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

            model = model_override or self.valves.llm_model
            if not model:
                logger.warning("No model available for LLM call")
                future.set_result(None)
                return None

            cache_key = hashlib.md5(
                f"{model}|{prompt}|{system_prompt}|{temperature}|{max_tokens}".encode()
            ).hexdigest()
            cached = await self._llm_cache.get(cache_key)
            if cached is not None:
                future.set_result(cached)
                self._log_debug(
                    f"[LLM] {model}"
                    + (f" ({label})" if label else "")
                    + f" (cached) took {time.monotonic() - t_start:.3f}s"
                )
                return cached

            ep_type = "chat"
            if model.startswith("llamacpp/"):
                ep_type = self.valves.llamacpp_endpoint_type

            effective_semaphore = semaphore or self._llm_semaphore
            max_retries = 2
            base_delay = 1.0

            # ── Log prompt size for diagnostics ──
            if self.tokenizer:
                prompt_tokens = len(self.tokenizer.encode(prompt))
                self._log_debug(
                    f"LLM call to {model}{f' ({label})' if label else ''} "
                    f"– prompt size: ~{prompt_tokens} tokens"
                )

            # ── Register this task as an active LLM user ──
            task = asyncio.current_task()
            async with self._active_llm_tasks_lock:
                self._active_llm_tasks.add(task)
            try:
                # ── Inter‑process lock: only one process can use the LLM at a time ──
                llm_fd = await self._acquire_llm_lock()
                try:
                    # ── Acquire global LLM semaphore to serialize all server calls ──
                    async with _llm_semaphore:
                        await self._maybe_unload_for_model(model, base_url, is_ollama)

                        for attempt in range(max_retries + 1):
                            try:
                                async with effective_semaphore:
                                    content = await _shared_call_llm(
                                        prompt=prompt,
                                        system=system_prompt,
                                        base_url=self.valves.LLM_BASE_URL,
                                        model=model,
                                        api_token=self.valves.LLM_API_TOKEN,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        timeout=self.valves.llm_request_timeout,
                                        endpoint_type=ep_type,
                                    )
                                if content:
                                    await self._llm_cache.set(cache_key, content)
                                    future.set_result(content)
                                    # ── Log input and output tokens ──
                                    in_tokens = (
                                        len(self.tokenizer.encode(prompt))
                                        if self.tokenizer
                                        else "?"
                                    )
                                    out_tokens = (
                                        len(self.tokenizer.encode(content))
                                        if self.tokenizer
                                        else "?"
                                    )
                                    self._log_debug(
                                        f"[LLM] {model}"
                                        + (f" ({label})" if label else "")
                                        + f" – in:{in_tokens} out:{out_tokens}"
                                        + f" took {time.monotonic() - t_start:.3f}s"
                                    )
                                    self._last_used_model = model
                                    return content
                            except asyncio.CancelledError:
                                raise
                            except RuntimeError as exc:
                                self._log_debug(
                                    f"[LLM] {model}{f' ({label})' if label else ''} "
                                    f"error: {exc}"
                                )
                                if any(
                                    c in str(exc)
                                    for c in ("429", "500", "502", "503", "504")
                                ):
                                    if attempt < max_retries:
                                        await asyncio.sleep(base_delay * (2**attempt))
                                        continue
                                break
                            except Exception:
                                if attempt < max_retries:
                                    await asyncio.sleep(base_delay * (2**attempt))
                                    continue
                                break
                finally:
                    self._release_llm_lock(llm_fd)
            finally:
                # Remove task from active set
                async with self._active_llm_tasks_lock:
                    self._active_llm_tasks.discard(task)

            logger.warning(
                f"LLM call failed for model {model}: prompt={prompt[:100]}..."
            )
            future.set_result(None)
            self._log_debug(
                f"[LLM] {model}"
                + (f" ({label})" if label else "")
                + f" (failed) after {time.monotonic() - t_start:.3f}s"
            )
            return None

        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._pending_llm_lock:
                self._pending_llm.pop(dedup_key, None)

    async def _try_llm_quick(
        self,
        prompt: str,
        system_prompt: str,
        timeout: float = 300,
        model_override: str = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        try:
            return await asyncio.wait_for(
                self._call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_override=model_override,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._log_debug(f"LLM call timed out after {timeout}s: {prompt[:80]}...")
            return None

    # --------------------------------------------------------------------------
    # Response cache (ChromaDB) – with thread pool
    # --------------------------------------------------------------------------
    def _ensure_cleanup_task(self) -> None:
        if (
            not self.valves.enable_response_cache
            or not HAS_CHROMA
            or self._response_cache_cleanup_task is not None
        ):
            return
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._response_cache_cleanup_task = asyncio.create_task(
                self._periodic_response_cache_cleanup()
            )

    async def _periodic_response_cache_cleanup(self):
        while True:
            await asyncio.sleep(3600)
            try:
                await self._purge_expired_response_cache()
            except Exception as e:
                self._log_debug(f"Response cache cleanup error: {e}")

    async def _purge_expired_response_cache(self):
        if (
            not hasattr(self, "_response_cache_collection")
            or self._response_cache_collection is None
        ):
            return
        ttl = self.valves.response_cache_ttl_hours * 3600
        if ttl <= 0:
            return
        try:
            results = await anyio.to_thread.run_sync(
                lambda: self._response_cache_collection.get(include=["metadatas"])
            )
            if not results or not results["ids"]:
                return
            now = time.time()
            to_delete = []
            for i, meta in enumerate(results["metadatas"]):
                if now - meta.get("timestamp", 0) > ttl:
                    to_delete.append(results["ids"][i])
            if to_delete:
                await anyio.to_thread.run_sync(
                    lambda: self._response_cache_collection.delete(ids=to_delete)
                )
                project = self.valves.project_id
                self._response_cache_count[project] = max(
                    0, self._response_cache_count.get(project, 0) - len(to_delete)
                )
        except Exception as e:
            self._log_debug(f"Error purging response cache: {e}")

    async def _store_response_in_cache(
        self,
        query: str,
        response: str,
        context_hash: str,
        state: dict,
        code_state_hash: str,
    ):
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return

        t_start = time.monotonic()
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
        entry_id = hashlib.md5(
            f"{self.valves.project_id}|{query}".encode()
        ).hexdigest()[:32]
        max_entries = self.valves.response_cache_max_entries
        project = self.valves.project_id
        current_size = self._response_cache_count.get(project, 0)
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
                    self._response_cache_count[project] -= len(old_entries["ids"])
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
        self._response_cache_count[project] = (
            self._response_cache_count.get(project, 0) + 1
        )
        self._log_timing(
            "resp_cache_total", time.monotonic() - t_start, time.monotonic() - t_start
        )

    async def _find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return None

        t_start = time.monotonic()
        query_vec = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
        results = await anyio.to_thread.run_sync(
            lambda: col.query(
                query_embeddings=[query_vec],
                n_results=1,
                where={"project_id": self.valves.project_id},
                include=["documents", "metadatas", "distances"],
            )
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return None

        dist = results["distances"][0][0]
        similarity = 1.0 - (dist / 2.0)
        if similarity < self.valves.response_cache_similarity_threshold:
            return None

        meta = results["metadatas"][0][0]
        stored_code_state = meta.get("code_state_hash", "")
        if stored_code_state and stored_code_state != self._compute_code_state_hash(
            self.valves.project_id
        ):
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        ttl = self.valves.response_cache_ttl_hours * 3600
        ts = meta.get("timestamp", 0)
        if ttl > 0 and time.time() - ts > ttl:
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        doc = results["documents"][0][0]
        return {"response": doc, "query": meta.get("query", ""), "timestamp": ts}

    # --------------------------------------------------------------------------
    # LTM storage and retrieval (symbol‑based & unified)
    # --------------------------------------------------------------------------
    def _is_symbol_indexable(self, symbol: CodeSymbol) -> bool:
        if symbol.kind not in ("function", "class", "method"):
            return False
        if len(symbol.name) < 3:
            return False
        if symbol.name in self._SYMBOL_BLACKLIST:
            return False
        return True

    def _extract_query_symbols(self, query: str, project_id: str) -> Set[str]:
        if not query or not project_id:
            return set()
        words = set(re.findall(r"\b\w+\b", query))
        project_symbols = self._symbol_index.get_all_names(project_id)
        return words.intersection(project_symbols)

    def _parse_forced_symbol_query(self, query: str) -> Tuple[Optional[str], str]:
        if not self.valves.ltm_symbol_force_mode_enabled:
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
        now = time.time()
        where = {
            "$and": [
                {"project_id": {"$eq": project_id}},
                {"code_symbols": {"$contains": f",{symbol},"}},
            ]
        }
        if self.valves.long_term_memory_expiration_days > 0:
            where["$and"].append({"expires_at": {"$gt": now}})
        q_emb = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(cleaned_query[:1000]).tolist()
        )
        results = await anyio.to_thread.run_sync(
            lambda: self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=self.valves.long_term_memory_top_k * 2,
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
                if self.valves.ltm_time_decay_hours > 0 and ts is not None:
                    age_hours = (now - ts) / 3600
                    sim *= 0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                if sim >= self.valves.long_term_memory_similarity_threshold:
                    docs_with_meta.append((doc, sim, ts, meta))
        docs_with_meta.sort(key=lambda x: x[1], reverse=True)
        docs_with_meta = docs_with_meta[: self.valves.long_term_memory_top_k]
        if not docs_with_meta and self.valves.ltm_symbol_force_fallback_to_semantic:
            return await self._retrieve_all_memories_unified(cleaned_query, project_id)
        return [{"doc": doc, "timestamp": ts} for doc, _, ts in docs_with_meta]

    async def _batch_store_messages(self, project_id: str, messages: List[dict]):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        valid = [
            m
            for m in messages
            if m.get("content", "").strip() and len(m["content"].strip()) >= 15
        ]
        if not valid:
            return

        t_start = time.monotonic()

        # Build contextualized texts for embedding and storage
        texts_for_embedding: List[str] = []
        documents_to_store: List[str] = []
        ids = []
        metadatas = []
        now = time.time()

        for i, msg in enumerate(valid):
            content = msg["content"]

            # Extract code metadata for context
            extracted, _ = await self._extract_code_blocks(content)
            content_type = self._classify_content(content, extracted)

            # Collect symbol names from extracted blocks
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

            # File paths
            ctx_file_paths: List[str] = []
            if self.valves.track_file_paths:
                ctx_file_paths = self._extract_file_paths(content)[:3]

            # Build retrieval context prefix
            context_prefix = await self._build_retrieval_context(
                content=content,
                project_id=project_id,
                role=msg.get("role", "user"),
                code_symbols=ctx_symbols[:6],
                file_paths=ctx_file_paths,
                content_type=content_type.value,
            )

            # The final text to embed and store includes the prefix.
            contextual_doc = context_prefix + content

            texts_for_embedding.append(contextual_doc)
            documents_to_store.append(contextual_doc)

            # Build ID and metadata (existing logic preserved)
            msg_id = f"{project_id}_{int(now)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            ids.append(msg_id)

            expires_at = (
                now + (self.valves.long_term_memory_expiration_days * 86400)
                if self.valves.long_term_memory_expiration_days > 0
                else None
            )

            code_symbols_str = ""
            if self.valves.ltm_index_symbols_enabled:
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
                                    >= self.valves.ltm_symbol_index_max_per_message
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

        # Embed all contextualized texts
        embeddings = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(
                texts_for_embedding, convert_to_numpy=True
            ).tolist()
        )

        if ids:
            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_to_store,
                )
            )
        self._log_timing(
            "batch_ltm_total", time.monotonic() - t_start, time.monotonic() - t_start
        )

    async def _retrieve_all_memories_unified(
        self, query: str, project_id: str
    ) -> List[Dict[str, Any]]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            return await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )

        try:
            t_start = time.monotonic()
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})

            # ── Phase 6 (PASO-34): Multi‑query expansion ─────────
            query_variants = await self._expand_query_for_retrieval(query)

            # Retrieve for each variant and merge by best score
            all_raw_results: Dict[str, Tuple[str, float, Any, Any]] = {}
            # key = memory_id → (doc, raw_sim, ts, meta)

            for variant_query in query_variants:
                q_emb = await anyio.to_thread.run_sync(
                    lambda q=variant_query: self.embedder.encode(q[:1000]).tolist()
                )
                try:
                    variant_results = await anyio.to_thread.run_sync(
                        lambda emb=q_emb: self.memory_collection.query(
                            query_embeddings=[emb],
                            n_results=self.valves.long_term_memory_top_k * 2,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"],
                        )
                    )
                except Exception as e:
                    self._log_debug(f"Multi-query retrieval failed for variant: {e}")
                    continue

                if not variant_results or not variant_results["documents"]:
                    continue

                for i, doc in enumerate(variant_results["documents"][0]):
                    meta = variant_results["metadatas"][0][i]
                    raw_sim = 1.0 - (variant_results["distances"][0][i] / 2.0)
                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None

                    # Dedup by memory_id, keeping max score
                    mem_id = meta.get(
                        "memory_id",
                        hashlib.md5(doc.encode()).hexdigest()[:16],
                    )
                    if (
                        mem_id not in all_raw_results
                        or raw_sim > all_raw_results[mem_id][1]
                    ):
                        all_raw_results[mem_id] = (doc, raw_sim, ts, meta)

            # Convert merged results to list
            results_list = list(all_raw_results.values())

            # ── Continue with existing scoring / decay / reranking ──
            docs_with_meta = []
            if results_list:
                for doc, raw_sim, ts, meta in results_list:
                    if self.valves.ltm_time_decay_hours > 0 and ts is not None:
                        age_hours = (now - ts) / 3600
                        effective_sim = raw_sim * (
                            0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                        )
                    else:
                        effective_sim = raw_sim

                    # ── v7 (PASO-20): boost raptor summaries ──
                    if meta.get("is_raptor_summary"):
                        raptor_level = meta.get("raptor_level", 1)
                        effective_sim *= 1.0 + 0.1 * raptor_level

                    if (
                        effective_sim
                        < self.valves.long_term_memory_similarity_threshold
                    ):
                        continue

                    docs_with_meta.append((doc, effective_sim, ts, meta, raw_sim))

            if self.valves.preserve_error_context:
                new_docs = []
                for doc, eff_sim, ts, meta, raw_sim in docs_with_meta:
                    if meta.get("content_type") == ContentType.ERROR.value:
                        eff_sim *= 1.1
                    new_docs.append((doc, eff_sim, ts, meta, raw_sim))
                docs_with_meta = new_docs

            docs_with_meta.sort(key=lambda x: x[1], reverse=True)

            if self.valves.ltm_symbol_boost_enabled and query:
                query_symbols = self._extract_query_symbols(query, project_id)
                if query_symbols:
                    new_docs = []
                    for doc, eff_sim, ts, meta, raw_sim in docs_with_meta:
                        meta_symbols_str = meta.get("code_symbols", "")
                        if (
                            meta_symbols_str
                            and eff_sim >= self.valves.ltm_symbol_boost_min_similarity
                        ):
                            meta_symbols = set(meta_symbols_str.split(","))
                            common = query_symbols.intersection(meta_symbols)
                            if common:
                                eff_sim *= self.valves.ltm_symbol_boost_factor
                        new_docs.append((doc, eff_sim, ts, meta, raw_sim))
                    new_docs.sort(key=lambda x: x[1], reverse=True)
                    docs_with_meta = new_docs

            if self.valves.enable_reranking and self._cross_encoder and docs_with_meta:
                rerank_k = min(
                    (
                        self.valves.reranker_top_k
                        if self.valves.reranker_top_k > 0
                        else self.valves.long_term_memory_top_k
                    ),
                    50,
                )
                docs_only = [d[0] for d in docs_with_meta[: rerank_k * 2]]
                reranked = await self._rerank_results(query, docs_only, rerank_k)
                doc_to_meta = {d[0]: (d[1], d[2]) for d in docs_with_meta}
                docs_with_meta = [
                    (doc, *doc_to_meta.get(doc, (0.0, None))) for doc in reranked
                ]

            docs_with_meta = docs_with_meta[: self.valves.long_term_memory_top_k]

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

    # --------------------------------------------------------------------------
    # Contextual Retrieval (v7 – Phase 6, PASO-32)
    # --------------------------------------------------------------------------

    async def _build_retrieval_context(
        self,
        content: str,
        project_id: str,
        role: str,
        code_symbols: List[str],
        file_paths: List[str],
        content_type: str,
    ) -> str:
        """
        Build a context prefix for a message before embedding it in LTM.
        Makes stored chunks more self-contained, improving retrieval when
        query phrasing differs from stored content.

        Mode 'metadata' (default): deterministic, no LLM call.
        Mode 'llm': generates a sentence via the secondary LLM.

        Returns a 1-3 line string to prepend to the document, or "" if disabled.
        """
        if not self.valves.enable_contextual_retrieval:
            return ""

        if self.valves.contextual_retrieval_mode == "llm":
            return await self._build_retrieval_context_llm(content, project_id)

        # ── Metadata mode (default) ─────────────────────────────────
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
        """
        Use a small LLM to generate a semantic retrieval context.
        Called only when contextual_retrieval_mode='llm'.
        """
        prompt = (
            "In one sentence (10-20 words), describe what the following "
            "code/conversation excerpt is about, for search retrieval:\n\n"
            f"{content[:400]}"
        )
        context = await self._try_llm_quick(
            prompt=prompt,
            system_prompt=(
                "Output only one descriptive sentence. "
                "Be specific about functions, files, or errors mentioned."
            ),
            model_override=self.valves.secondary_task_model,
            max_tokens=40,
            temperature=0.2,
            timeout=8.0,
        )
        if context and context.strip():
            return f"[Context: {context.strip()}]\n\n"
        return ""

    # --------------------------------------------------------------------------
    # RAPTOR Hierarchical LTM (v7 – PASO-20)
    # --------------------------------------------------------------------------

    async def _build_raptor_layer(
        self,
        project_id: str,
        level: int = 1,
        n_clusters: int = 5,
        min_cluster_size: int = 3,
    ) -> int:
        """
        Build one layer of the RAPTOR tree:
        1. Retrieve embeddings from ChromaDB for the previous level.
        2. Cluster with k-means.
        3. Generate a summary per cluster via LLM.
        4. Store the summary in ChromaDB and SQLite.
        Returns number of clusters created. Runs asynchronously in background.
        """
        if not HAS_CHROMA or self.memory_collection is None:
            return 0
        if not HAS_SENTENCE:
            return 0

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            import numpy as np
        except ImportError:
            self._log_debug("scikit-learn not installed — RAPTOR disabled")
            return 0

        # ── Get entries from previous level ─────────────────────────
        where_filter: Dict = {"project_id": {"$eq": project_id}}
        if level == 1:
            where_filter["is_raptor_summary"] = {"$ne": True}
        else:
            where_filter["raptor_level"] = {"$eq": level - 1}

        results = await anyio.to_thread.run_sync(
            lambda: self.memory_collection.get(
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "ids"],
                limit=500,
            )
        )

        if not results or not results["ids"] or len(results["ids"]) < min_cluster_size:
            self._log_debug(
                f"RAPTOR level {level}: not enough entries "
                f"({len(results['ids']) if results else 0} < {min_cluster_size})"
            )
            return 0

        ids = results["ids"]
        docs = results["documents"]
        embeddings = np.array(results["embeddings"])
        embeddings_normalized = normalize(embeddings)

        # Adjust n_clusters if fewer entries than desired clusters
        actual_clusters = min(n_clusters, len(ids) // min_cluster_size)
        if actual_clusters < 2:
            return 0

        # ── k-means clustering ──────────────────────────────────────
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = await anyio.to_thread.run_sync(
            lambda: kmeans.fit_predict(embeddings_normalized)
        )

        # ── Generate summary per cluster ────────────────────────────
        clusters_created = 0
        for cluster_idx in range(actual_clusters):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
            if len(member_indices) < min_cluster_size:
                continue

            member_ids = [ids[i] for i in member_indices]
            member_docs = [docs[i] for i in member_indices]

            # Combine documents for the LLM (respect token limits)
            combined = "\n\n---\n\n".join(doc[:500] for doc in member_docs[:10])

            prompt = (
                f"Summarize the following {len(member_docs)} related code/conversation "
                f"fragments into 2-3 sentences capturing their common theme, "
                f"key functions involved, and main purpose:\n\n{combined}"
            )

            summary = await self._try_llm_quick(
                prompt=prompt,
                system_prompt=(
                    "You are a technical summarizer. "
                    "Output 2-3 concise sentences. No bullet points."
                ),
                model_override=self.valves.raptor_summary_model,
                max_tokens=self.valves.raptor_summary_max_tokens,
                temperature=0.2,
            )

            if not summary:
                continue

            # ── Embed the summary and store in ChromaDB ──────────────
            summary_embedding = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(summary).tolist()
            )

            cluster_id = (
                f"{project_id}_raptor_L{level}_C{cluster_idx}_{int(time.time())}"
            )
            centroid = kmeans.cluster_centers_[cluster_idx].tolist()

            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.upsert(
                    ids=[cluster_id],
                    embeddings=[summary_embedding],
                    documents=[f"[RAPTOR L{level} Summary]\n{summary}"],
                    metadatas=[
                        {
                            "project_id": project_id,
                            "is_raptor_summary": True,
                            "raptor_level": level,
                            "cluster_id": cluster_id,
                            "member_count": len(member_ids),
                            "timestamp": time.time(),
                            "content_type": "raptor_summary",
                        }
                    ],
                )
            )

            # ── Persist to SQLite for traceability ─────────────────
            def _write_cluster():
                self._db_conn.execute(
                    "INSERT OR REPLACE INTO raptor_clusters VALUES (?,?,?,?,?,?,?)",
                    (
                        cluster_id,
                        project_id,
                        level,
                        json.dumps(member_ids),
                        summary,
                        json.dumps(centroid),
                        time.time(),
                    ),
                )
                self._db_conn.commit()

            await self._db_write_queue.put((_write_cluster, (), {}))

            clusters_created += 1
            self._log_debug(
                f"RAPTOR L{level} cluster {cluster_idx}: "
                f"{len(member_ids)} members → summary stored"
            )

        return clusters_created

    async def _rebuild_raptor_index(self, project_id: str):
        """
        Rebuild the full RAPTOR tree for a project. Background task.
        Process:
        1. Delete old RAPTOR summaries for the project.
        2. Build level 1 (clusters of raw entries).
        3. If enough level-1 clusters, build level 2 (clusters of clusters).
        """
        if not self.valves.enable_raptor:
            return

        self._log_debug(f"RAPTOR: rebuilding index for project {project_id}")

        # Clean old summaries
        try:
            old = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.get(
                    where={
                        "$and": [
                            {"project_id": {"$eq": project_id}},
                            {"is_raptor_summary": {"$eq": True}},
                        ]
                    },
                    include=["ids"],
                )
            )
            if old and old["ids"]:
                await anyio.to_thread.run_sync(
                    lambda: self.memory_collection.delete(ids=old["ids"])
                )
                self._log_debug(f"RAPTOR: deleted {len(old['ids'])} old summaries")
        except Exception as e:
            self._log_debug(f"RAPTOR: cleanup failed: {e}")

        # Build levels
        l1_clusters = await self._build_raptor_layer(
            project_id,
            level=1,
            n_clusters=self.valves.raptor_clusters_per_level,
        )
        self._log_debug(f"RAPTOR: level 1 = {l1_clusters} clusters")

        if l1_clusters >= 4:
            l2_clusters = await self._build_raptor_layer(
                project_id,
                level=2,
                n_clusters=max(2, l1_clusters // 2),
            )
            self._log_debug(f"RAPTOR: level 2 = {l2_clusters} clusters")

    async def _flush_ltm_batch(self, project_id: str):
        await asyncio.sleep(0.5)
        async with self._ltm_batch_lock:
            if not self._pending_ltm_messages:
                return
            messages_to_store = self._pending_ltm_messages.copy()
            self._pending_ltm_messages.clear()
            self._ltm_batch_task = None
        await self._batch_store_messages(project_id, messages_to_store)

    # --------------------------------------------------------------------------
    # Multi‑Query LTM Expansion (v7 – Phase 6, PASO-34)
    # --------------------------------------------------------------------------

    async def _expand_query_for_retrieval(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query for LTM retrieval.

        Returns a list starting with the original query, followed by up to
        `multi_query_variants` alternatives.

        If the LLM call fails or the feature is disabled, returns [query].
        """
        if not self.valves.enable_multi_query_retrieval:
            return [query]
        if len(query.strip()) < 15:
            return [query]

        prompt = (
            f"Generate {self.valves.multi_query_variants} alternative phrasings "
            f"of this programming question for document search. "
            f"Focus on different vocabulary (errors, function names, behaviors).\n\n"
            f"Original: {query[:250]}\n\n"
            f"Output only the alternatives, one per line. No numbering."
        )
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt=(
                "Output only the alternative phrasings, one per line. "
                "Be concise and specific to the code context."
            ),
            model_override=self.valves.secondary_task_model,
            max_tokens=80,
            temperature=0.6,
            timeout=10.0,
        )

        queries = [query]  # always include original
        if response:
            alternatives = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            queries.extend(alternatives[: self.valves.multi_query_variants])
            self._log_debug(
                f"Multi-query expansion: {len(queries)} queries "
                f"({[q[:40] for q in queries]})"
            )
        return queries

    async def _retrieve_historical_messages(
        self, query: str, project_id: str, limit: int
    ) -> List[Dict]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            memories = await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )
            return [{"role": "user", "content": m["doc"]} for m in memories]

        try:
            q_emb = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(query[:1000]).tolist()
            )
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})

            results = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.query(
                    query_embeddings=[q_emb],
                    n_results=limit * 3,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            )

            query_symbols = set()
            if self.valves.ltm_symbol_boost_enabled and query:
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
                        and sim >= self.valves.ltm_symbol_boost_min_similarity
                    ):
                        meta_symbols_str = meta.get("code_symbols", "")
                        if meta_symbols_str:
                            meta_symbols = set(meta_symbols_str.split(","))
                            common = query_symbols.intersection(meta_symbols)
                            if common:
                                sim *= self.valves.ltm_symbol_boost_factor
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
                self.valves.enable_reranking
                and self._cross_encoder
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

    def _load_reranker(self):
        if not self.valves.enable_reranking or not HAS_CROSS_ENCODER:
            return
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.valves.reranker_model)
                self._log_debug(f"Loaded reranker model {self.valves.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")
                self.valves.enable_reranking = False

    async def _rerank_results(
        self, query: str, documents: List[str], top_k: int
    ) -> List[str]:
        if not self.valves.enable_reranking or not self._cross_encoder or not documents:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = await anyio.to_thread.run_sync(self._cross_encoder.predict, pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    # --------------------------------------------------------------------------
    # Block expiration and summarization (for inactive code blocks)
    # --------------------------------------------------------------------------
    async def _expire_blocks_by_time(self, project_id: str):
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return
            now = time.time()
            expiration_seconds = self.valves.block_expiration_hours * 3600
            to_remove = []
            for h, block in state["active_blocks"].items():
                if block.pinned or block.obsolete:
                    continue
                age = now - block.last_mentioned
                if (
                    block.content_type == ContentType.ERROR
                    and self.valves.error_retention_turns > 0
                ):
                    if age > max(
                        self.valves.error_retention_turns * 300, expiration_seconds
                    ):
                        to_remove.append(h)
                elif (
                    block.content_type == ContentType.PROPOSED_CHANGE
                    and self.valves.proposed_change_retention_turns > 0
                ):
                    if age > max(
                        self.valves.proposed_change_retention_turns * 300,
                        expiration_seconds,
                    ):
                        to_remove.append(h)
            for h in to_remove:
                if h in state["active_blocks"]:
                    block = state["active_blocks"][h]
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                del state["active_blocks"][h]
            if to_remove:
                state["has_any_calls"] = any(
                    any(s.calls for s in b.symbols)
                    for b in state["active_blocks"].values()
                )
                self._invalidate_lightweight_cache(project_id)
                self._set_state(project_id, state)

    def _update_mentions_from_message(
        self, state: Dict, message_content: str, project_id: str
    ):
        if not message_content:
            return
        all_symbol_names = self._symbol_index.get_all_names(project_id)
        words = set(re.findall(r"\b[\w-]+\b", message_content))
        mentioned_names = all_symbol_names.intersection(words)
        if not mentioned_names:
            return
        affected_blocks: Set[str] = set()
        for name in mentioned_names:
            affected_blocks.update(self._symbol_index.find_blocks(name, project_id))
        for block_hash in affected_blocks:
            block = state["active_blocks"].get(block_hash)
            if block:
                block.mention_count += 1
                block.last_mentioned = time.time()
                block.last_mentioned_msg_idx = state["message_count"]
                block._update_importance()

    async def _update_active_code(self, message: dict, project_id: str):
        if not self.valves.enable_code_awareness:
            return

        content = message.get("content", "")
        role = message.get("role", "")

        extracted, block_spans = await self._extract_code_blocks(content)
        new_blocks_pending = []
        for idx, block_info in enumerate(extracted):
            blk_file = None
            if self.valves.track_file_paths and block_spans:
                blk_file = self._extract_file_path_for_block(
                    content, block_spans[idx][0]
                )
            if not blk_file and len(extracted) == 1:
                extracted_paths = self._extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None
            content_type = self._classify_content(content, extracted)
            new_block = CodeBlock(
                content=block_info["code"],
                content_type=content_type,
                generated_by_assistant=(role == "assistant"),
                file_path=blk_file,
                line_range=None,
                timestamp=time.time(),
                is_active=True,
                mention_count=1,
                dependencies=[],
                potentially_affected=False,
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

        _content_to_syms: Dict[str, List[CodeSymbol]] = {
            blk.content: syms
            for blk, syms in zip(new_blocks_pending, symbols_list)
            if not isinstance(syms, Exception)
        }

        lock = await self._get_project_lock(project_id)
        state_before = self._get_state(project_id)
        existing_contents = {}
        if state_before:
            for h, b in state_before["active_blocks"].items():
                existing_contents[h] = b.content

        duplicate_info = {}
        for new_block in new_blocks_pending:
            is_dup = False
            existing_dup = None
            for h, ex_content in existing_contents.items():
                ex_block = state_before["active_blocks"].get(h)
                if (
                    ex_block
                    and self._calculate_code_similarity(new_block.content, ex_content)
                    >= self.valves.code_similarity_threshold
                ):
                    is_dup = True
                    existing_dup = h
                    break
            duplicate_info[new_block.hash] = (is_dup, existing_dup)

        async with lock:
            state = self._get_state(project_id)
            self._background_task(
                self._summarize_inactive_blocks_safely(project_id),
                name="summarize_inactive",
                is_llm_task=True,
            )
            self._update_mentions_from_message(state, content, project_id)
            for block in state["active_blocks"].values():
                if (
                    block.content
                    and self._calculate_code_similarity(
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

            for new_block, syms in zip(new_blocks_pending, symbols_list):
                if isinstance(syms, Exception):
                    syms = []

                new_block.content = self._sanitize_text(new_block.content)

                if self.tokenizer:
                    new_block._cached_token_count = len(
                        self.tokenizer.encode(new_block.content)
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
                    if existing.pinned or new_block.is_raw:
                        self._symbol_index.remove_all_for_block(
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
                        reused = _content_to_syms.get(new_block.content)
                        if reused is not None:
                            existing.symbols = [
                                s.copy(update={"parent_block_hash": existing.hash})
                                for s in reused
                            ]
                        else:
                            existing.symbols = await SignatureExtractor.extract_async(
                                existing.content, existing.file_path
                            )
                        for s in existing.symbols:
                            s.parent_block_hash = existing.hash
                            self._symbol_index.add(s, existing.hash, project_id)
                            # ── v7 (PASO-09): register typed edges ──
                            for callee_name in s.calls:
                                edge = Edge(
                                    src=s.name,
                                    dst=callee_name,
                                    type="calls",
                                    weight=EDGE_WEIGHTS["calls"],
                                    confidence=1.0,
                                )
                                self._symbol_index.add_edge(edge, project_id)
                            # ── v7 (PASO-19): register data flow edges ──
                            if (
                                self.valves.enable_data_flow_analysis
                                and existing.file_path
                            ):
                                df_edges = self._extract_data_flow_edges(
                                    existing.content,
                                    existing.file_path,
                                    project_id,
                                )
                                for df_edge in df_edges:
                                    self._symbol_index.add_edge(df_edge, project_id)
                                if df_edges:
                                    self._log_debug(
                                        f"Data flow: {len(df_edges)} edge(s) extracted from {existing.file_path}"
                                    )
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        if prev_content != new_block.content:
                            self._background_task(
                                self._generate_change_summary(
                                    existing.hash, prev_content, new_block.content
                                ),
                                name="change_summary",
                                is_llm_task=True,
                            )
                        continue

                    if self.valves.prioritize_recent_code:
                        self._symbol_index.remove_all_for_block(
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
                        reused = _content_to_syms.get(new_block.content)
                        if reused is not None:
                            existing.symbols = [
                                s.copy(update={"parent_block_hash": existing.hash})
                                for s in reused
                            ]
                        else:
                            existing.symbols = await SignatureExtractor.extract_async(
                                existing.content, existing.file_path
                            )
                        for s in existing.symbols:
                            s.parent_block_hash = existing.hash
                            self._symbol_index.add(s, existing.hash, project_id)
                            # ── v7 (PASO-09): register typed edges ──
                            for callee_name in s.calls:
                                edge = Edge(
                                    src=s.name,
                                    dst=callee_name,
                                    type="calls",
                                    weight=EDGE_WEIGHTS["calls"],
                                    confidence=1.0,
                                )
                                self._symbol_index.add_edge(edge, project_id)
                            # ── v7 (PASO-19): register data flow edges ──
                            if (
                                self.valves.enable_data_flow_analysis
                                and existing.file_path
                            ):
                                df_edges = self._extract_data_flow_edges(
                                    existing.content,
                                    existing.file_path,
                                    project_id,
                                )
                                for df_edge in df_edges:
                                    self._symbol_index.add_edge(df_edge, project_id)
                                if df_edges:
                                    self._log_debug(
                                        f"Data flow: {len(df_edges)} edge(s) extracted from {existing.file_path}"
                                    )
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        if prev_content != new_block.content:
                            self._background_task(
                                self._generate_change_summary(
                                    existing.hash, prev_content, new_block.content
                                ),
                                name="change_summary",
                                is_llm_task=True,
                            )
                    continue

                # New non‑duplicate block
                for sym in syms:
                    sym.parent_block_hash = new_block.hash
                new_block.symbols = syms
                new_block.last_mentioned_msg_idx = state["message_count"]
                for sym in syms:
                    self._symbol_index.add(sym, new_block.hash, project_id)
                    # ── v7 (PASO-09): register typed edges from call relationships ──
                    for callee_name in sym.calls:
                        edge = Edge(
                            src=sym.name,
                            dst=callee_name,
                            type="calls",
                            weight=EDGE_WEIGHTS["calls"],
                            confidence=1.0,
                        )
                        self._symbol_index.add_edge(edge, project_id)
                    # ── v7 (PASO-19): register data flow edges ──
                    if self.valves.enable_data_flow_analysis and new_block.file_path:
                        df_edges = self._extract_data_flow_edges(
                            new_block.content,
                            new_block.file_path,
                            project_id,
                        )
                        for df_edge in df_edges:
                            self._symbol_index.add_edge(df_edge, project_id)
                        if df_edges:
                            self._log_debug(
                                f"Data flow: {len(df_edges)} edge(s) extracted from {new_block.file_path}"
                            )
                if any(s.calls for s in syms):
                    state["has_any_calls"] = True

                is_conflicting = False
                if new_block.content_type == ContentType.PROPOSED_CHANGE:
                    is_conflicting = self._has_conflicting_proposed_changes(
                        state, new_block
                    )
                    if is_conflicting:
                        new_block.importance_score = max(
                            new_block.importance_score, 7.0
                        )

                state["active_blocks"][new_block.hash] = new_block

                if new_block.file_path and self.valves.enable_obsolete_marking:
                    for h, blk in list(state["active_blocks"].items()):
                        if h == new_block.hash:
                            continue
                        if blk.file_path == new_block.file_path and not blk.pinned:
                            blk.obsolete = True
                            blk._update_importance()

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
                    if self.valves.enable_diff_application and not is_conflicting:
                        for base in list(state["active_blocks"].values()):
                            if (
                                base.content_type == ContentType.BASE_CODE
                                and base.file_path == new_block.file_path
                            ):
                                if self._apply_change_with_diff(base, new_block):
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
                    and self.valves.preserve_error_context
                ):
                    new_block.importance_score = min(
                        new_block.importance_score + 3.0, 10.0
                    )

                if len(state["active_blocks"]) > self.valves.max_active_blocks:
                    sorted_blocks = sorted(
                        state["active_blocks"].values(),
                        key=lambda b: b.importance_score
                        + (self.valves.raw_file_priority_boost if b.is_raw else 0),
                        reverse=True,
                    )
                    keep = sorted_blocks[: self.valves.max_active_blocks]
                    state["active_blocks"] = {b.hash: b for b in keep}

            # Assistant implicit modifications
            if role == "assistant" and len(extracted) > 0:
                for block_info in extracted:
                    best_base = None
                    best_sim = 0.0
                    for base in state["active_blocks"].values():
                        if base.content_type == ContentType.BASE_CODE:
                            sim = self._calculate_code_similarity(
                                base.content, block_info["code"]
                            )
                            if sim > best_sim and sim > 0.6:
                                best_sim = sim
                                best_base = base
                    if best_base and best_sim > 0.5 and best_sim < 0.98:
                        self._symbol_index.remove_all_for_block(
                            best_base.hash, best_base.symbols, project_id
                        )
                        prev_content = best_base.content
                        best_base.content = self._sanitize_text(block_info["code"])
                        best_base.hash = hashlib.md5(
                            block_info["code"].encode()
                        ).hexdigest()[:16]
                        best_base.timestamp = time.time()
                        best_base.is_active = True
                        best_base.potentially_affected = False
                        best_base.importance_score = min(
                            best_base.importance_score + 1.0, 10.0
                        )
                        reused = _content_to_syms.get(block_info["code"])
                        if reused is not None:
                            best_base.symbols = [
                                s.copy(update={"parent_block_hash": best_base.hash})
                                for s in reused
                            ]
                        else:
                            best_base.symbols = await SignatureExtractor.extract_async(
                                best_base.content, best_base.file_path
                            )
                        for s in best_base.symbols:
                            s.parent_block_hash = best_base.hash
                            self._symbol_index.add(s, best_base.hash, project_id)
                            # ── v7 (PASO-09): register typed edges ──
                            for callee_name in s.calls:
                                edge = Edge(
                                    src=s.name,
                                    dst=callee_name,
                                    type="calls",
                                    weight=EDGE_WEIGHTS["calls"],
                                    confidence=1.0,
                                )
                                self._symbol_index.add_edge(edge, project_id)
                            # ── v7 (PASO-19): register data flow edges ──
                            if (
                                self.valves.enable_data_flow_analysis
                                and best_base.file_path
                            ):
                                df_edges = self._extract_data_flow_edges(
                                    best_base.content,
                                    best_base.file_path,
                                    project_id,
                                )
                                for df_edge in df_edges:
                                    self._symbol_index.add_edge(df_edge, project_id)
                                if df_edges:
                                    self._log_debug(
                                        f"Data flow: {len(df_edges)} edge(s) extracted from {best_base.file_path}"
                                    )
                        if self.tokenizer:
                            best_base._cached_token_count = len(
                                self.tokenizer.encode(best_base.content)
                            )
                        else:
                            best_base._cached_token_count = len(best_base.content) // 4
                        if any(s.calls for s in best_base.symbols):
                            state["has_any_calls"] = True
                        if prev_content != block_info["code"]:
                            self._background_task(
                                self._generate_change_summary(
                                    best_base.hash, prev_content, block_info["code"]
                                ),
                                name="change_summary",
                                is_llm_task=True,
                            )

            state["message_count"] += 1
            if self.valves.auto_remove_duplicate_blocks:
                self._remove_duplicate_blocks(state, project_id)
            self._background_task(
                self._expire_blocks_by_time(project_id), name="expire_blocks"
            )

            # ── Enrichment tasks (run immediately with limited concurrency) ──
            tasks_to_run = []
            max_tasks_per_type = 5

            for block in list(state["active_blocks"].values()):
                if block.obsolete:
                    continue

                if self.valves.enable_auto_summaries:
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

            if tasks_to_run:
                sem = self._secondary_llm_semaphore

                async def _run_one(task_type, params):
                    try:
                        if task_type == "missing_summaries":
                            await self._run_missing_summaries_task(
                                params, self.valves.secondary_task_model, sem
                            )
                    except Exception as e:
                        self._log_debug(
                            f"Immediate enrichment task {task_type} failed: {e}"
                        )

                sem_enrich = asyncio.Semaphore(4)
                async with sem_enrich:
                    await asyncio.gather(*[_run_one(t, p) for t, p in tasks_to_run])

            # ── Eviction by max_active_blocks ──
            if (
                self.valves.max_active_blocks > 0
                and len(state["active_blocks"]) > self.valves.max_active_blocks
            ):
                sorted_blocks = sorted(
                    state["active_blocks"].values(),
                    key=lambda b: b.importance_score
                    + (self.valves.raw_file_priority_boost if b.is_raw else 0),
                    reverse=True,
                )
                keep_hashes = {
                    b.hash for b in sorted_blocks[: self.valves.max_active_blocks]
                }
                to_remove = [h for h in state["active_blocks"] if h not in keep_hashes]
                for h in to_remove:
                    del state["active_blocks"][h]
                if to_remove:
                    self._log_debug(
                        f"Evicted {len(to_remove)} blocks due to max_active_blocks limit. "
                        f"Their symbols remain in the index for lightweight context."
                    )

            # Session summary
            if self.valves.enable_session_summary:
                interval = self.valves.session_summary_interval_messages
                if (
                    interval > 0
                    and state["message_count"] % interval == 0
                    and state["message_count"] > 0
                ):
                    task = SecondaryTask(
                        task_type="session_summary",
                        params={
                            "project_id": project_id,
                            "message_count": state["message_count"],
                            "code_state_hash": self._compute_code_state_hash(
                                project_id
                            ),
                        },
                    )
                    state.setdefault("pending_secondary_tasks", []).append(task.dict())

            self._invalidate_lightweight_cache(project_id)

            # ── v7 (PASO-09): invalidate affected CodePathViews ──────────
            if self.valves.enable_path_analysis:
                changed_symbols: Set[str] = set()
                for blk in new_blocks_pending:
                    for sym in blk.symbols:
                        changed_symbols.add(sym.name)

                stale_path_ids: Set[str] = set()
                for sym_name in changed_symbols:
                    for pid in self._path_index.mark_stale_for_symbol(
                        sym_name, project_id
                    ):
                        stale_path_ids.add(pid)

                for pid in stale_path_ids:
                    view = self._path_index.get(pid, project_id)
                    if not view:
                        continue
                    new_structural = self._compute_structural_hash(
                        view.induced_nodes.keys(), project_id
                    )
                    new_call_graph = self._compute_call_graph_hash(
                        view.induced_nodes.keys(), project_id
                    )
                    if view.is_stale(new_structural, new_call_graph):
                        view.structural_hash = new_structural
                        view.call_graph_hash = new_call_graph
                        view.summary = ""
                        view.business_label = ""
                        view.label_confidence = 0.0

                if stale_path_ids:
                    self._log_debug(
                        f"Invalidated {len(stale_path_ids)} CodePathView(s) "
                        f"due to changes in {len(changed_symbols)} symbol(s)"
                    )

            self._set_state(project_id, state)

    async def _summarize_inactive_blocks_safely(self, project_id: str):
        if self._summarize_inactive_in_progress.get(project_id, False):
            return
        self._summarize_inactive_in_progress[project_id] = True
        try:
            if not self.valves.summarize_inactive_code:
                return
            state = self._get_state(project_id)
            if not state or not state["active_blocks"]:
                return
            now = time.time()
            timeout = self.valves.active_code_timeout_minutes * 60
            to_summarize = []
            for h, block in state["active_blocks"].items():
                if block.pinned or block.obsolete:
                    continue
                if (
                    not block.is_active
                    and (now - block.timestamp) > timeout
                    and block.importance_score < 5.0
                ):
                    to_summarize.append((h, block))
            if not to_summarize:
                return

            async def _summarize_with_semaphore(block):
                async with self._low_priority_llm_semaphore:
                    return await self._summarize_code_block(block)

            tasks = [_summarize_with_semaphore(block) for _, block in to_summarize]
            summaries = await asyncio.gather(*tasks)

            for (h, block), summary in zip(to_summarize, summaries):
                if summary:
                    sig = self._extract_signature(block.content)
                    if sig:
                        summary = f"{sig}\n\n{summary}"
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                    summary_content = f"[Summary of inactive code]\n{summary}"
                    summary_block = CodeBlock(
                        content=summary_content,
                        content_type=ContentType.GENERAL,
                        timestamp=time.time(),
                        is_active=False,
                        importance_score=block.importance_score * 0.5,
                    )
                    if self.tokenizer:
                        summary_block._cached_token_count = len(
                            self.tokenizer.encode(summary_content)
                        )
                    else:
                        summary_block._cached_token_count = len(summary_content) // 4
                    state["active_blocks"][h] = summary_block
            self._invalidate_lightweight_cache(project_id)
            self._set_state(project_id, state)
        except Exception as e:
            self._log_debug(f"Error in summarize_inactive_blocks_safely: {e}")
        finally:
            self._summarize_inactive_in_progress[project_id] = False

    async def _summarize_code_block(self, block: CodeBlock) -> Optional[str]:
        if not self.valves.summarize_inactive_code:
            return None
        if self.valves.defer_secondary_tasks:
            task = SecondaryTask(
                task_type="inactive_code_summary",
                params={
                    "signature": self._extract_signature(block.content),
                    "content": block.content,
                    "project_id": self.valves.project_id,
                    "block_hash": block.hash,
                },
            )
            state = self._get_state(self.valves.project_id)
            if state is not None:
                state.setdefault("pending_secondary_tasks", []).append(task.dict())
                self._set_state(self.valves.project_id, state)
            return None
        sig = self._extract_signature(block.content)
        if sig:
            prompt = f"The code block has signature: {sig}\nProvide a very brief description of what this code does.\nCode:\n```{block.content[:1000]}```"
        else:
            prompt = (
                f"Summarise the following code block.\n```{block.content[:1500]}```"
            )
        return await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant.",
            model_override=self.valves.inactive_code_summary_model,
            max_tokens=200,
            temperature=0.2,
        )

    def _extract_signature(self, code: str) -> str:
        func_match = re.search(
            r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*[^:]*)?\s*:",
            code,
            re.MULTILINE,
        )
        if func_match:
            name, params = func_match.group(1), func_match.group(2).strip()
            doc_match = re.search(
                r'^\s*"""(.*?)"""', code[func_match.end() :], re.DOTALL
            ) or re.search(r"^\s*'''(.*?)'''", code[func_match.end() :], re.DOTALL)
            docstring = doc_match.group(1).strip()[:100] if doc_match else ""
            return (
                f"Function `{name}({params})` - {docstring}"
                if docstring
                else f"Function `{name}({params})`"
            )
        class_match = re.search(
            r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:",
            code,
            re.MULTILINE,
        )
        if class_match:
            name = class_match.group(1)
            doc_match = re.search(
                r'^\s*"""(.*?)"""', code[class_match.end() :], re.DOTALL
            ) or re.search(r"^\s*'''(.*?)'''", code[class_match.end() :], re.DOTALL)
            docstring = doc_match.group(1).strip()[:100] if doc_match else ""
            return f"Class `{name}` - {docstring}" if docstring else f"Class `{name}`"
        return ""

    @staticmethod
    def _sanitize_signature(sig: str, max_len: int = 200) -> str:
        safe = sig.replace("`", "'")
        return safe[:max_len] + ("…" if len(safe) > max_len else "")

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        cleaned = cleaned.replace("`", "'")
        return cleaned

    # --------------------------------------------------------------------------
    # Forget / remember / obsolete commands
    # --------------------------------------------------------------------------
    async def _handle_forget_command(
        self, messages: List[dict], project_id: str, __user__: Optional[dict]
    ) -> Tuple[List[dict], bool]:
        if not (
            self.valves.enable_forget_command
            or self.valves.enable_natural_language_forget
        ):
            return messages, False
        if not messages:
            return messages, False
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return messages, False
        content = last_msg.get("content", "").strip()
        if self.valves.enable_forget_command and content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            state = self._get_state(project_id)
            if not state:
                return messages, False
            if target == "all":
                for block in state["active_blocks"].values():
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                state["active_blocks"].clear()
                state["recent_changes"].clear()
                state["committed_changes"].clear()
                state["has_any_calls"] = False
                self._invalidate_lightweight_cache(project_id)
                confirmation = "Forgotten all context."
            elif target == "last":
                if state["active_blocks"]:
                    last_hash = max(
                        state["active_blocks"].keys(),
                        key=lambda h: state["active_blocks"][h].timestamp,
                    )
                    block = state["active_blocks"].get(last_hash)
                    if block:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state["active_blocks"][last_hash]
                    self._invalidate_lightweight_cache(project_id)
                    confirmation = "Forgotten the last context block."
                else:
                    confirmation = "No blocks to forget."
            else:
                to_remove = [
                    h
                    for h, blk in state["active_blocks"].items()
                    if (blk.file_path and target in blk.file_path) or target in h
                ]
                for h in to_remove:
                    block = state["active_blocks"].get(h)
                    if block:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state["active_blocks"][h]
                self._invalidate_lightweight_cache(project_id)
                confirmation = (
                    f"Forgotten {len(to_remove)} block(s) matching '{target}'."
                )
            self._set_state(project_id, state)
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return messages, True
        return messages, False

    # --------------------------------------------------------------------------
    # Proactive cleanup commands
    # --------------------------------------------------------------------------
    def _get_inactive_block_candidates(self, project_id: str) -> List[str]:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return []
        threshold = self.valves.cleanup_inactive_threshold_messages
        excluded_types = set(self.valves.cleanup_excluded_content_types)
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

    async def _handle_clean_command(self, command_text: str, project_id: str) -> str:
        if (
            not self.valves.cleanup_suggestions_enabled
            or not self.valves.cleanup_command_enabled
        ):
            return "Cleanup is disabled."
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            candidates = self._get_inactive_block_candidates(project_id)
            parts = command_text.split(maxsplit=1)
            subcommand = parts[1].strip() if len(parts) > 1 else ""
            if not subcommand:
                if not candidates:
                    return "✅ No inactive blocks to clean."
                lines = [
                    f"⚠️ {len(candidates)} inactive block(s) (not mentioned in last {self.valves.cleanup_inactive_threshold_messages} messages):"
                ]
                for h in candidates:
                    blk = state["active_blocks"].get(h)
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
                    block = state["active_blocks"].pop(h, None)
                    if block:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                state["recent_changes"] = [
                    c for c in state["recent_changes"] if c.hash not in candidates
                ]
                state["committed_changes"] = [
                    c for c in state["committed_changes"] if c.hash not in candidates
                ]
                self._invalidate_lightweight_cache(project_id)
                self._set_state(project_id, state)
                return f"✅ Cleaned {len(candidates)} inactive block(s)."
            target_hash = subcommand.strip()
            if target_hash in candidates:
                block = state["active_blocks"].pop(target_hash, None)
                if block:
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                self._invalidate_lightweight_cache(project_id)
                self._set_state(project_id, state)
                return f"✅ Cleaned block `{target_hash[:8]}...`."
            else:
                matched = [h for h in state["active_blocks"] if target_hash in h]
                for h in matched:
                    if h in candidates:
                        block = state["active_blocks"].pop(h, None)
                        if block:
                            self._symbol_index.remove_all_for_block(
                                block.hash, block.symbols, project_id
                            )
                        self._invalidate_lightweight_cache(project_id)
                        self._set_state(project_id, state)
                        return f"✅ Cleaned block `{h[:8]}...` (matched partial hash)."
                return "❌ Block not found among inactive candidates. Use `/status` to see candidates."

    # --------------------------------------------------------------------------
    # Expand (symbol dependency expansion)
    # --------------------------------------------------------------------------
    async def _handle_expand_command(self, text: str, project_id: str) -> str:
        parts = text.strip().split()
        if len(parts) < 2:
            return "Usage: `/expand [depth] <function_name>`\nExample: `/expand 3 calcularImpuesto`"
        depth = self.valves.expand_default_depth
        if parts[1].isdigit():
            depth = int(parts[1])
            func_name = parts[2] if len(parts) > 2 else None
        else:
            func_name = parts[1]
        if not func_name:
            return "Missing function name."

        all_names = self._symbol_index.get_all_names(project_id)
        if not all_names:
            state = self._get_state(project_id)
            if state and state.get("active_blocks"):
                self._rebuild_symbol_index(state, project_id)
                all_names = self._symbol_index.get_all_names(project_id)

        if not all_names:
            return (
                "❌ The symbol index is empty.\n\n"
                "No code has been processed in this session yet. "
                "Paste the code you want to analyze first, then use `/expand` again."
            )

        if func_name not in all_names:
            lower_name = func_name.lower()
            suggestions = sorted(
                [n for n in all_names if lower_name in n.lower()],
                key=lambda n: (not n.startswith(func_name[0]), n),
            )[:8]
            hint = ""
            if suggestions:
                hint = "\n\nDid you mean one of these?\n" + "\n".join(
                    f"- `{s}`" for s in suggestions
                )
            else:
                sample = sorted(all_names)[:10]
                hint = (
                    f"\n\nThe index contains {len(all_names)} symbol(s). Sample:\n"
                    + "\n".join(f"- `{s}`" for s in sample)
                )
            return f"❌ Symbol `{func_name}` not found in the index." + hint

        expanded = await self._expand_symbol_dependencies(func_name, depth, project_id)
        if not expanded:
            return (
                f"❌ Symbol `{func_name}` is indexed but its code body could not be retrieved. "
                "Try pasting the code again."
            )

        if expanded.count("### ") <= 1:
            return f"## Expanded: `{func_name}` (depth {depth})\n\n{expanded}\n\n> No further call dependencies found."

        return f"## Expanded: `{func_name}` (depth {depth})\n\n{expanded}"

    async def _expand_symbol_dependencies(
        self, name: str, max_depth: int, project_id: str
    ) -> str:
        state = self._get_state(project_id)
        if not state:
            return ""
        visited = set()
        lines = []

        async def recurse(current_name, current_depth):
            if current_depth > max_depth or current_name in visited:
                return
            visited.add(current_name)
            blocks = self._symbol_index.find_blocks(current_name, project_id)
            for h in blocks:
                block = state["active_blocks"].get(h)
                if block and not block.obsolete:
                    loc = f" (file: {block.file_path})" if block.file_path else ""
                    lines.append(
                        f"### `{current_name}` (depth {current_depth}){loc}\n```\n{block.content[:2000]}\n```"
                    )
                    for sym in block.symbols:
                        if sym.name == current_name:
                            for callee in sym.calls:
                                await recurse(callee, current_depth + 1)
                            break
                    break

        await recurse(name, 1)
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Outlet expand intercept
    # --------------------------------------------------------------------------
    async def _outlet_intercept_expand(
        self,
        assistant_content: str,
        project_id: str,
    ) -> Tuple[str, bool]:
        if not self.valves.outlet_expand_intercept_enabled:
            return assistant_content, False

        EXPAND_RE = re.compile(r"/expand\s+(?:(\d+)\s+)?(\w+)", re.IGNORECASE)
        matches = list(EXPAND_RE.finditer(assistant_content))
        if not matches:
            return assistant_content, False

        all_names = self._symbol_index.get_all_names(project_id)
        replaced_content = assistant_content
        did_any = False
        state = self._get_state(project_id)

        max_syms = self.valves.outlet_expand_intercept_max_symbols
        matches_to_process = matches if max_syms == 0 else matches[:max_syms]

        for match in matches_to_process:
            depth_str = match.group(1)
            func_name = match.group(2)
            depth = (
                int(depth_str)
                if depth_str
                else self.valves.outlet_expand_intercept_depth
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

            lock = await self._get_project_lock(project_id)
            async with lock:
                block_hashes = self._symbol_index.find_blocks(func_name, project_id)
                for h in block_hashes:
                    block = state["active_blocks"].get(h)
                    if block and not block.obsolete:
                        block.is_raw = True
                        block.pinned = True
                        block.importance_score = 10.0
                        block.last_mentioned = time.time()
                        block.last_mentioned_msg_idx = state["message_count"]
                        break
                self._invalidate_lightweight_cache(project_id)
                self._set_state(project_id, state)

        return replaced_content, did_any

    # --------------------------------------------------------------------------
    # Path analysis (v7) – graph activation and seed extraction
    # --------------------------------------------------------------------------

    def _extract_query_seeds(
        self, query: str, project_id: str
    ) -> Tuple[List[str], List[str]]:
        """
        Extract seed symbols from the query.
        Returns (exact_matches, partial_matches).

        exact_matches: words that are exact symbol names.
        partial_matches: words that are substrings of symbol names (used when few exact matches).
        """
        all_names = self._symbol_index.get_all_names(project_id)
        query_words = set(re.findall(r"\b\w+\b", query))

        exact = list(all_names.intersection(query_words))

        partial = []
        if len(exact) < 3:  # only look for partials if few exact matches
            for word in query_words:
                if len(word) < 4:  # ignore very short words
                    continue
                for name in all_names:
                    if word.lower() in name.lower() and name not in exact:
                        partial.append(name)
                        break
            partial = partial[:5]  # cap partial matches

        return exact, partial

    def _compute_node_specificity(self, symbol_name: str, project_id: str) -> float:
        """
        IDF-like specificity of a symbol.
        Symbols appearing in many blocks are less specific (like stop-words).
        Returns a multiplier in [0.1, 3.0] to adjust its weight as a seed.

        Examples:
        - '__init__' appears in 20 blocks → specificity ~0.3
        - 'validate_credit_card' appears in 1 block → specificity ~2.5
        """
        import math

        all_names = self._symbol_index.get_all_names(project_id)
        total = max(len(all_names), 1)
        n_blocks = len(self._symbol_index.find_blocks(symbol_name, project_id))
        if n_blocks == 0:
            return 1.0
        # IDF: log(total / n_blocks) + 1, clipped to [0.1, 3.0]
        specificity = math.log(total / n_blocks) + 1.0
        return max(0.1, min(3.0, specificity))

    def _store_activation_scores(self, ag: ActivationGraph, project_id: str):
        """Save activation scores for speculative prefetch and LOD tracking."""
        activated = ag.get_activated_nodes(
            threshold=self.valves.path_activation_threshold
        )
        if not hasattr(self, "_last_activation_scores"):
            self._last_activation_scores: Dict[str, Dict[str, float]] = {}
        self._last_activation_scores[project_id] = activated

    def _build_activation_graph(
        self,
        query: str,
        project_id: str,
        max_propagation_steps: int = 4,
        messages: Optional[List[dict]] = None,
    ) -> ActivationGraph:
        """
        Build an ActivationGraph combining up to three independent seed vectors.

        Vector 1 — Lexical:
          Exact/partial symbol name matches in the query, enriched with
          traceback frame seeds if present.
          Weight: multi_seed_weight_lexical (default 0.5)

        Vector 2 — Structural:
          Entry points of CodePathViews that contain the lexically matched
          symbols. Captures the "where it comes from" context.
          Weight: multi_seed_weight_structural (default 0.3)

        Vector 3 — Historical:
          Symbols with high mention frequency in recent conversation messages.
          Weight: multi_seed_weight_historical (default 0.2)

        If enable_multi_seed_activation is False, falls back to the previous
        single‑graph behaviour (lexical + traceback + history combined in one pass).
        """
        edges_out = self._symbol_index.get_all_edges_out(project_id)
        all_names = self._symbol_index.get_all_names(project_id)

        exact_seeds, partial_seeds = self._extract_query_seeds(query, project_id)
        tb_seeds = (
            self._extract_traceback_seeds(query, project_id)
            if self.valves.enable_traceback_activation
            else []
        )
        history_boosts = (
            self._extract_history_seeds(
                messages, project_id, lookback=self.valves.history_seeds_lookback
            )
            if (self.valves.enable_history_seeds and messages)
            else {}
        )

        # ──────────────────────────────────────────────────────────────
        # SINGLE‑GRAPH MODE (backward compatible)
        # ──────────────────────────────────────────────────────────────
        if not self.valves.enable_multi_seed_activation:
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
                # ── Fallback: entry points ordered by centrality ──
                entry_points = self._path_index.find_entry_points(
                    self._symbol_index, project_id
                )
                if entry_points:
                    centrality = self._node_centrality.get(project_id, {})
                    sorted_eps = sorted(
                        entry_points,
                        key=lambda ep: centrality.get(ep, 0.0),
                        reverse=True,
                    )
                    for sym_name in sorted_eps[:3]:
                        cent_score = centrality.get(sym_name, 0.0)
                        seed_score = 0.2 + 0.2 * cent_score  # [0.2, 0.4]
                        ag._activations[sym_name] = ActivationState(
                            node_id=sym_name,
                            score=seed_score,
                            depth=0,
                            source="seed",
                        )

            ag.propagate(
                edges_out=edges_out,
                max_steps=20,
                min_score=0.05,
                alpha=self.valves.ppr_alpha,
            )
            self._store_activation_scores(ag, project_id)
            return ag

        # ──────────────────────────────────────────────────────────────
        # MULTI‑SEED MODE (three independent vectors)
        # ──────────────────────────────────────────────────────────────

        # ── Vector 1: Lexical + Traceback ──────────────────────────
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
                alpha=self.valves.ppr_alpha,
            )

        # ── Vector 2: Structural (entry points of views containing lexical seeds) ──
        ag_str = ActivationGraph()
        lexical_seed_names = set(exact_seeds) | {s for s, _ in tb_seeds}
        structural_seeds: Set[str] = set()
        for view in self._path_index.get_all(project_id):
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
                alpha=self.valves.ppr_alpha,
            )

        # ── Vector 3: Historical ───────────────────────────────────
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
                alpha=self.valves.ppr_alpha,
            )

        # ── Combine the three vectors ──────────────────────────────
        w_lex = self.valves.multi_seed_weight_lexical
        w_str = self.valves.multi_seed_weight_structural
        w_his = self.valves.multi_seed_weight_historical

        all_activated = (
            set(ag_lex.get_activated_nodes(0.01).keys())
            | set(ag_str.get_activated_nodes(0.01).keys())
            | set(ag_his.get_activated_nodes(0.01).keys())
        )

        ag_final = ActivationGraph()

        if not all_activated:
            # ── Fallback: entry points ordered by centrality ─────────
            entry_points = self._path_index.find_entry_points(
                self._symbol_index, project_id
            )
            if entry_points:
                centrality = self._node_centrality.get(project_id, {})
                sorted_eps = sorted(
                    entry_points,
                    key=lambda ep: centrality.get(ep, 0.0),
                    reverse=True,
                )
                for sym_name in sorted_eps[:3]:
                    cent_score = centrality.get(sym_name, 0.0)
                    seed_score = 0.2 + 0.2 * cent_score  # [0.2, 0.4]
                    ag_final._activations[sym_name] = ActivationState(
                        node_id=sym_name,
                        score=seed_score,
                        depth=0,
                        source="seed",
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
        self._log_debug(
            f"Multi-seed ActivationGraph: {activated_count} nodes activated "
            f"(lex={len(ag_lex.get_activated_nodes(0.01))}, "
            f"str={len(ag_str.get_activated_nodes(0.01))}, "
            f"his={len(ag_his.get_activated_nodes(0.01))})"
        )

        self._store_activation_scores(ag_final, project_id)
        return ag_final

    def _compute_static_centrality(self, project_id: str) -> Dict[str, float]:
        """
        Compute static PageRank centrality on the call graph.

        Standard PageRank = PPR with uniform personalization vector.
        High centrality symbols are hubs: called from many places or
        calling many others.

        Returns: {symbol_name: score ∈ [0.0, 1.0]} where 1.0 = most central.
        """
        if not self.valves.enable_centrality_prior:
            return {}

        all_names = self._symbol_index.get_all_names(project_id)
        edges_out = self._symbol_index.get_all_edges_out(project_id)

        if not all_names or not edges_out:
            return {}

        n = len(all_names)
        initial_score = 1.0 / n if n > 0 else 0.0

        ag = ActivationGraph()
        for name in all_names:
            ag._activations[name] = ActivationState(
                node_id=name,
                score=initial_score,
                depth=0,
                source="seed",
            )

        ag.propagate(
            edges_out=edges_out,
            max_steps=30,  # more iterations for convergence
            min_score=0.0001,
            alpha=0.85,
            tolerance=1e-7,
        )

        raw_scores = {nid: s.score for nid, s in ag._activations.items()}
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        if max_score == 0:
            return {}

        normalized = {name: score / max_score for name, score in raw_scores.items()}

        top3 = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
        self._log_debug(
            f"Centrality computed for {len(normalized)} symbols. "
            f"Top-3: {[(n, f'{s:.3f}') for n, s in top3]}"
        )
        return normalized

    async def _resolve_dangling_edges(self, project_id: str) -> int:
        """
        Resolve cross-chunk symbol references.

        A 'dangling edge' is an edge whose destination is referenced in the
        call graph but has no code block yet. When a new chunk defines that
        symbol, the edge confidence is raised from 0.3 (provisional) to 1.0.

        Conversely, edges pointing to symbols that are referenced but not
        defined are marked with confidence 0.3.

        Returns the number of edges resolved.
        """
        all_names = self._symbol_index.get_all_names(project_id)
        resolved = 0

        for sym_name in all_names:
            has_definition = bool(self._symbol_index.find_blocks(sym_name, project_id))
            edges_in = self._symbol_index.get_edges_in(sym_name, project_id)

            for edge in edges_in:
                if has_definition and edge.confidence < 1.0:
                    # Symbol now defined → restore confidence
                    edge.confidence = 1.0
                    resolved += 1
                elif not has_definition and edge.confidence == 1.0:
                    # Symbol referenced but not defined → mark provisional
                    edge.confidence = 0.3

        if resolved > 0:
            self._log_debug(
                f"Cross-chunk resolution: {resolved} edge(s) resolved "
                f"(references confirmed with definitions)"
            )
        return resolved

    # --------------------------------------------------------------------------
    # Traceback seed extraction (v7 – Phase 5, PASO-26)
    # --------------------------------------------------------------------------

    def _extract_traceback_seeds(
        self, content: str, project_id: str
    ) -> List[Tuple[str, float]]:
        """
        Extract function names from a traceback with scores proportional
        to their depth in the call stack.

        Supports:
        - Python: File "path", line N, in function_name
        - JavaScript/TypeScript: at function_name (file:line:col)
        - Java: at package.Class.method(File.java:line)

        Returns: [(symbol_name, seed_score), ...]
        The last frame (deepest) receives score 1.0.
        The first frame receives score 0.5.
        Only symbols present in the project's SymbolIndex are included.
        """
        all_names = self._symbol_index.get_all_names(project_id)
        if not all_names:
            return []

        frames: List[str] = []

        # ── Python traceback ─────────────────────────────────────────
        py_pattern = re.compile(
            r'File\s+"[^"]+",\s+line\s+\d+,\s+in\s+(\w+)',
            re.MULTILINE,
        )
        for match in py_pattern.finditer(content):
            func = match.group(1)
            if func in all_names and func != "<module>":
                frames.append(func)

        # ── JavaScript / TypeScript ──────────────────────────────────
        js_pattern = re.compile(
            r"\bat\s+(\w+)\s*\([^)]*:\d+:\d+\)",
            re.MULTILINE,
        )
        for match in js_pattern.finditer(content):
            func = match.group(1)
            if func in all_names:
                frames.append(func)

        # ── Java / Kotlin ────────────────────────────────────────────
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

        # Deduplicate while preserving order (last = deepest = most relevant)
        seen: Set[str] = set()
        unique_frames: List[str] = []
        for f in frames:
            if f not in seen:
                seen.add(f)
                unique_frames.append(f)

        n = len(unique_frames)
        results = []
        for i, func_name in enumerate(unique_frames):
            # Linear interpolation: 0.5 for the first frame, 1.0 for the last
            score = 0.5 + 0.5 * (i / max(n - 1, 1))
            # Adjust by symbol specificity
            specificity = self._compute_node_specificity(func_name, project_id)
            adjusted = min(1.0, score * min(specificity, 1.5))
            results.append((func_name, adjusted))

        self._log_debug(
            f"Traceback seeds: {len(results)} frame(s) detected "
            f"({[r[0] for r in results]})"
        )
        return results

    # --------------------------------------------------------------------------
    # History seed extraction (v7 – Phase 5, PASO-27)
    # --------------------------------------------------------------------------

    def _extract_history_seeds(
        self,
        messages: List[dict],
        project_id: str,
        lookback: int = 6,
    ) -> Dict[str, float]:
        """
        Extract symbols with high mention frequency in recent messages.

        Returns {symbol_name: boost_score} where boost_score ∈ (0.0, 0.6].
        The boost is proportional to the relative mention frequency.

        Only considers the last `lookback` messages to avoid stale context.
        """
        all_names = self._symbol_index.get_all_names(project_id)
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
                self.valves.history_seeds_max_boost,
                self.valves.history_seeds_max_boost * (count / max_count),
            )
            for sym, count in mention_counts.items()
            if count > 0
        }

    # --------------------------------------------------------------------------
    # LLMLingua-2 code compression (v7 – PASO-18)
    # --------------------------------------------------------------------------

    def _init_llmlingua(self):
        """Initialise the LLMLingua-2 compressor. Runs on CPU, no GPU needed."""
        try:
            from llmlingua import PromptCompressor

            self._llmlingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
            self._log_debug("LLMLingua-2 compressor initialized (CPU)")
        except ImportError:
            self._log_debug("llmlingua not installed — code compression disabled")
            self.valves.enable_code_compression = False
        except Exception as e:
            self._log_debug(f"LLMLingua-2 init failed: {e} — compression disabled")
            self.valves.enable_code_compression = False

    async def _compress_code_block(
        self,
        code: str,
        language: str = "python",
        rate: float = 0.5,
        query: str = "",  # ← Phase 6 (PASO-31)
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
        if not self._llmlingua_compressor:
            return code

        # Don't compress small blocks (overhead > benefit)
        estimated_tokens = self._estimate_code_tokens(code)
        if estimated_tokens < self.valves.code_compression_min_tokens:
            return code

        # Structural tokens that are NEVER removed
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
            # Activate question-aware compression if a query is provided
            if query and self.valves.enable_question_aware_compression:
                compress_kwargs["question"] = query[:300]  # cap to avoid overhead
                self._log_debug(
                    f"LLMLingua-2: question-aware mode active "
                    f"(query={query[:60]}...)"
                )

            result = await anyio.to_thread.run_sync(
                lambda: self._llmlingua_compressor.compress_prompt(
                    code,
                    **compress_kwargs,
                )
            )
            compressed = result.get("compressed_prompt", code)
            compressed_tokens = self._estimate_code_tokens(compressed)
            self._log_debug(
                f"LLMLingua-2: {estimated_tokens} → {compressed_tokens} tokens "
                f"({100*(1-compressed_tokens/max(estimated_tokens,1)):.0f}% reduction)"
            )
            return compressed
        except Exception as e:
            self._log_debug(f"LLMLingua-2 compression failed: {e} — using original")
            return code

    # --------------------------------------------------------------------------
    # Data flow edge extraction (v7 – PASO-19)
    # --------------------------------------------------------------------------

    def _extract_data_flow_edges(
        self,
        code: str,
        file_path: Optional[str],
        project_id: str,
    ) -> List[Edge]:
        """
        Extract data flow edges from Python code using ast.

        Detects calls to known project functions where arguments come from
        local variables → edge type 'data_flow' from caller to callee.

        Falls back to regex for non‑Python languages.
        """
        if not file_path or not file_path.endswith(".py"):
            return self._extract_data_flow_edges_regex(code, project_id)

        all_names = self._symbol_index.get_all_names(project_id)
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

            # Collect variables assigned within this function
            assigned_vars: Set[str] = set()
            for child in ast.walk(func_node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)

            # Detect calls to other project functions with local variables as arguments
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
                            confidence=0.7,  # lower confidence than tree-sitter
                        )
                    )

        return edges

    def _extract_data_flow_edges_regex(self, code: str, project_id: str) -> List[Edge]:
        """
        Fallback data flow extraction for non‑Python languages.
        Detects assignments whose right-hand side is a call to a known function.
        """
        all_names = self._symbol_index.get_all_names(project_id)
        edges: List[Edge] = []

        # Pattern: var = known_function(...)
        pattern = re.compile(
            r"\b(\w+)\s*=\s*(" + "|".join(re.escape(n) for n in all_names) + r")\s*\("
        )
        for match in pattern.finditer(code):
            callee = match.group(2)
            var_name = match.group(1)
            # Look for functions that use this variable as argument
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

    async def _build_view_from_activation(
        self,
        entry_point: str,
        activation: ActivationGraph,
        project_id: str,
    ) -> Optional[CodePathView]:
        """
        Build a CodePathView from an ActivationGraph and cache it in PathIndex.
        """
        edges_out = self._symbol_index.get_all_edges_out(project_id)
        edges_in_map: Dict[str, List[Edge]] = defaultdict(list)
        for sym, edge_list in edges_out.items():
            for e in edge_list:
                edges_in_map[e.dst].append(e)

        extractor = SubgraphExtractor(
            activation_threshold=self.valves.path_activation_threshold,
            expand_hops=1,
        )
        induced_nodes_set, induced_edges = extractor.extract(
            activation, edges_out, edges_in_map
        )

        if not induced_nodes_set:
            return None

        # Scores for the induced nodes
        induced_nodes_scored = {
            node: activation.get_score(node) for node in induced_nodes_set
        }

        path_id = hashlib.md5(
            f"{entry_point}|{'|'.join(sorted(induced_nodes_set))}".encode()
        ).hexdigest()[:16]

        # Try to reuse summary/label if the subgraph hasn't changed
        existing = self._path_index.get(path_id, project_id)
        structural_hash = self._compute_structural_hash(induced_nodes_set, project_id)
        call_graph_hash = self._compute_call_graph_hash(induced_nodes_set, project_id)

        view = CodePathView(
            path_id=path_id,
            entry_point=entry_point,
            seed_nodes=[entry_point],
            induced_nodes=induced_nodes_scored,
            induced_edges=induced_edges,
            activation_score=activation.aggregate_path_score(list(induced_nodes_set)),
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

        self._path_index.add(view, project_id)
        return view

    async def _rebuild_path_index(self, project_id: str):
        """Reconstruct PathIndex from SymbolIndex for all entry points."""
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return
        entry_points = self._path_index.find_entry_points(
            self._symbol_index, project_id
        )
        for ep in entry_points:
            ag = self._build_activation_graph(ep, project_id)
            await self._build_view_from_activation(ep, ag, project_id)

        # ── Phase 6 (PASO-33): compute static centrality ──────────
        if self.valves.enable_centrality_prior:
            self._node_centrality[project_id] = self._compute_static_centrality(
                project_id
            )

    async def _speculative_prefetch(
        self,
        project_id: str,
        last_activated: Dict[str, float],
    ):
        """
        Pre‑build CodePathViews for symbols likely to be relevant in the
        next query. Runs as a background task during LLM decode.

        Prediction: high‑confidence direct callees of the top‑N activated symbols.
        """
        if not self.valves.enable_speculative_prefetch:
            return
        if not last_activated:
            return

        # Top symbols from the current query
        top_syms = sorted(last_activated, key=last_activated.get, reverse=True)[:3]

        prefetch_candidates: Set[str] = set()
        for sym in top_syms:
            for edge in self._symbol_index.get_edges_out(sym, project_id):
                if (
                    edge.type == "calls"
                    and edge.effective_weight() >= 0.7
                    and edge.dst not in last_activated  # not already activated
                ):
                    prefetch_candidates.add(edge.dst)

        if not prefetch_candidates:
            return

        candidates = list(prefetch_candidates)[: self.valves.speculative_prefetch_max]
        self._log_debug(
            f"Speculative prefetch: pre-building {len(candidates)} CodePathView(s) "
            f"for next likely query"
        )

        edges_out = self._symbol_index.get_all_edges_out(project_id)
        for sym_name in candidates:
            if not self._symbol_index.find_blocks(sym_name, project_id):
                continue  # symbol referenced but not defined
            ag = ActivationGraph()
            ag.seed([sym_name], initial_score=1.0)
            ag.propagate(edges_out, max_steps=2, min_score=0.1)
            await self._build_view_from_activation(sym_name, ag, project_id)

    # ── Hash helpers for structural / call‑graph invalidation ─────────

    def _compute_structural_hash(
        self, symbol_names: Iterable[str], project_id: str
    ) -> str:
        """Hash of the symbols' content blocks (changes when code changes)."""
        state = self._get_state(project_id)
        hashes = []
        for name in sorted(symbol_names):
            for bh in sorted(self._symbol_index.find_blocks(name, project_id)):
                hashes.append(bh)
        return hashlib.md5("|".join(hashes).encode()).hexdigest()[:16] if hashes else ""

    def _compute_call_graph_hash(
        self, symbol_names: Iterable[str], project_id: str
    ) -> str:
        """Hash of the call relationships (changes when the graph changes)."""
        edge_strs = []
        for name in sorted(symbol_names):
            for edge in self._symbol_index.get_edges_out(name, project_id):
                edge_strs.append(f"{edge.src}:{edge.type}:{edge.dst}")
        return (
            hashlib.md5("|".join(sorted(edge_strs)).encode()).hexdigest()[:16]
            if edge_strs
            else ""
        )

    # ── Intent classification (v7) ──────────────────────────────────

    async def _classify_intent(
        self, user_query: str, project_id: str
    ) -> Dict[str, float]:
        """
        Classify the user's intent into a continuous weight vector.

        Returns a dict where values sum to ~1.0.
        Keys: "explain" | "modify" | "debug" | "refactor"

        Process:
        1. Fast deterministic heuristic (no LLM).
        2. LLM fallback only when the heuristic signal is weak.
        """
        query_lower = user_query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        # ── Heuristic: count signals per intent ──────────────────────

        EXPLAIN_KW = {
            "explain",
            "how",
            "what",
            "describe",
            "show",
            "diagram",
            "explica",
            "cómo",
            "qué",
            "describe",
            "muestra",
        }
        MODIFY_KW = {
            "fix",
            "add",
            "change",
            "implement",
            "update",
            "create",
            "make",
            "corrige",
            "añade",
            "cambia",
            "implementa",
            "actualiza",
            "crea",
        }
        DEBUG_KW = {
            "error",
            "bug",
            "fail",
            "crash",
            "wrong",
            "broken",
            "exception",
            "traceback",
            "not working",
            "falla",
            "error",
            "excepción",
        }
        REFACTOR_KW = {
            "refactor",
            "restructure",
            "reorganize",
            "redesign",
            "architecture",
            "refactoriza",
            "reestructura",
            "arquitectura",
            "reorganiza",
        }

        scores = {
            "explain": len(EXPLAIN_KW.intersection(query_words)) * 1.0,
            "modify": len(MODIFY_KW.intersection(query_words)) * 1.0,
            "debug": len(DEBUG_KW.intersection(query_words)) * 1.5,  # debug weighs more
            "refactor": len(REFACTOR_KW.intersection(query_words))
            * 2.0,  # refactor even more
        }

        # Additional signals
        if "traceback" in query_lower or "exception" in query_lower:
            scores["debug"] += 2.0
        if "```" in user_query:
            scores["modify"] += 0.5
        if len(user_query) > 500:
            scores["explain"] += 0.3

        total = sum(scores.values())

        # If the signal is clear, normalise and return without LLM
        if total > 1.5:
            normalized = {k: v / total for k, v in scores.items()}
            self._log_debug(
                f"Intent (heuristic): {max(normalized, key=normalized.get)}="
                f"{max(normalized.values()):.2f}"
            )
            return normalized

        # ── LLM fallback when heuristic signal is weak ────────────────
        if not self.valves.enable_intent_llm_fallback:
            return {"explain": 0.3, "modify": 0.4, "debug": 0.2, "refactor": 0.1}

        prompt = (
            f'User message: "{user_query[:300]}"\n\n'
            "Score the user intent from 0.0 to 1.0 for each category "
            "(total should sum to 1.0):\n"
            "explain: (wants to understand)\n"
            "modify: (wants to change/add/fix code)\n"
            "debug: (is debugging an error)\n"
            "refactor: (wants architectural changes)\n\n"
            "Output only: explain=X.X modify=X.X debug=X.X refactor=X.X"
        )
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="Output only scores in the format: explain=X.X modify=X.X debug=X.X refactor=X.X",
            model_override=self.valves.intent_classifier_model,
            max_tokens=20,
            temperature=0.0,
        )

        if response:
            result = {}
            for match in re.finditer(r"(\w+)=([\d.]+)", response):
                key, val = match.group(1), float(match.group(2))
                if key in ("explain", "modify", "debug", "refactor"):
                    result[key] = val
            if len(result) == 4:
                total_r = sum(result.values())
                if total_r > 0:
                    return {k: v / total_r for k, v in result.items()}

        # Default fallback
        return {"explain": 0.25, "modify": 0.45, "debug": 0.2, "refactor": 0.1}

    async def _get_static_context_block(
        self,
        project_id: str,
        is_code_session: bool,
    ) -> str:
        """
        Build or retrieve from cache the static block of the system prompt.

        This block is IDENTICAL across consecutive requests as long as the
        code has not changed. It serves as the KV cache anchor for llama.cpp.

        Contents:
        1. Base behavioural instructions (always the same)
        2. Symbol index / lightweight context (stable until code changes)
        3. Feedback context (stable until new feedback arrives)

        Invalidation: regenerated when _compute_code_state_hash() returns a
        different hash.
        """
        current_code_hash = self._compute_code_state_hash(project_id)
        cached = self._static_context_block_cache.get(project_id)

        if cached:
            cached_hash, cached_text = cached
            if cached_hash == current_code_hash:
                return cached_text  # ✓ Hit: same code → same block

        # ── Build the static block ──────────────────────────────────
        parts: List[str] = []

        # 1. Base instructions (completely static)
        if self.valves.enable_confidence_scoring and is_code_session:
            parts.append(self.valves.confidence_prompt.strip())

        if is_code_session and self.valves.enable_code_awareness:
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

        # 2. Symbol index (lightweight context — stable while code unchanged)
        if is_code_session and self.valves.enable_code_awareness:
            state = self._get_state(project_id)
            if state and state["active_blocks"]:
                lightweight = await self._build_lightweight_context(project_id)
                if lightweight:
                    parts.append(lightweight)

        # 3. Feedback context (stable between requests barring new feedback)
        if (
            is_code_session
            and self.valves.enable_feedback_tracking
            and self.valves.inject_feedback_context
        ):
            feedback_ctx = self._get_feedback_context(project_id)
            if feedback_ctx:
                parts.append(feedback_ctx)

        static_block = "\n\n".join(p for p in parts if p.strip())

        # ── Cache and track ─────────────────────────────────────────
        self._static_context_block_cache[project_id] = (current_code_hash, static_block)

        # Detect and log prefix changes (= cache miss in llama.cpp)
        new_prefix_hash = hashlib.md5(static_block.encode()).hexdigest()[:16]
        last_hash = self._last_static_prefix_hash.get(project_id)
        if last_hash and last_hash != new_prefix_hash:
            self._log_debug(
                f"⚠️  KV CACHE MISS detected: static block changed "
                f"({last_hash} → {new_prefix_hash}). "
                f"llama.cpp will do a full prefill on this request."
            )
        elif not last_hash:
            self._log_debug(
                f"KV Cache: first request for project, "
                f"static prefix established ({new_prefix_hash})."
            )
        else:
            self._log_debug(
                f"✓ KV Cache: static prefix stable ({new_prefix_hash}). "
                f"llama.cpp will reuse KV states for Block A."
            )
        self._last_static_prefix_hash[project_id] = new_prefix_hash

        tokens = (
            len(self.tokenizer.encode(static_block))
            if self.tokenizer
            else len(static_block) // 4
        )
        self._log_debug(f"Static Context Block: ~{tokens} tokens")

        return static_block

    def _slot_filename(self, project_id: str, static_hash: str) -> str:
        """
        Deterministic slot file name.
        Encodes: project + static block hash + model hash.
        If any of the three changes → different name → no stale restore.
        """
        model_hash = hashlib.md5(self.valves.llm_model.encode()).hexdigest()[:8]
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        return (
            f"slot{self.valves.slot_id}_{project_slug}_{static_hash}_{model_hash}.bin"
        )

    async def _slot_restore_if_available(self, project_id: str) -> bool:
        """
        Attempt to restore the llama.cpp slot at session start.
        Executed only ONCE per project per server startup.
        Returns True if the restore was successful.
        """
        if not self.valves.enable_slot_persistence:
            return False
        if self._slot_restore_attempted.get(project_id):
            return self._slot_restored.get(project_id, False)

        self._slot_restore_attempted[project_id] = True

        # Get the current static block hash
        cached = self._static_context_block_cache.get(project_id)
        if not cached:
            return False  # static block not built yet
        _, static_text = cached
        static_hash = hashlib.md5(static_text.encode()).hexdigest()[:16]

        filename = self._slot_filename(project_id, static_hash)
        slot_dir = self.valves.slot_save_path.rstrip("/")
        full_path = os.path.join(slot_dir, filename)

        if not os.path.exists(full_path):
            self._log_debug(f"Slot restore: no file found for {filename}")
            return False

        # Check if the slot is already warm
        try:
            session = await _shared_get_http_session(timeout_seconds=5)
            base = self.valves.LLM_BASE_URL.rstrip("/")
            async with session.get(f"{base}/slots") as resp:
                if resp.status == 200:
                    slots = await resp.json()
                    slot = next(
                        (s for s in slots if s.get("id") == self.valves.slot_id),
                        None,
                    )
                    if slot and slot.get("n_past", 0) > 100:
                        self._log_debug(
                            f"Slot {self.valves.slot_id} already warm "
                            f"(n_past={slot['n_past']}), skipping restore"
                        )
                        self._slot_restored[project_id] = True
                        return True
        except Exception as e:
            self._log_debug(f"Slot status check failed: {e}")

        # Perform the restore
        try:
            session = await _shared_get_http_session(timeout_seconds=30)
            base = self.valves.LLM_BASE_URL.rstrip("/")
            async with session.post(
                f"{base}/slots/{self.valves.slot_id}/restore",
                json={"filename": filename},
            ) as resp:
                if resp.status == 200:
                    self._slot_restored[project_id] = True
                    self._log_debug(
                        f"✓ Slot restored from {filename} — "
                        f"Block A pre-loaded, first query warm"
                    )
                    return True
                else:
                    body_txt = await resp.text()
                    self._log_debug(
                        f"Slot restore failed: HTTP {resp.status} — {body_txt}"
                    )
                    return False
        except Exception as e:
            self._log_debug(f"Slot restore error: {e}")
            return False

    async def _slot_save_if_needed(self, project_id: str) -> bool:
        """
        Save the slot state to disk if the static block hash changed
        since the last save. Called at the end of the outlet.
        """
        if not self.valves.enable_slot_persistence:
            return False

        cached = self._static_context_block_cache.get(project_id)
        if not cached:
            return False
        _, static_text = cached
        static_hash = hashlib.md5(static_text.encode()).hexdigest()[:16]

        # Only save if the hash changed
        if self._last_saved_slot_hash.get(project_id) == static_hash:
            return False

        filename = self._slot_filename(project_id, static_hash)

        try:
            session = await _shared_get_http_session(timeout_seconds=30)
            base = self.valves.LLM_BASE_URL.rstrip("/")
            async with session.post(
                f"{base}/slots/{self.valves.slot_id}/save",
                json={"filename": filename},
            ) as resp:
                if resp.status == 200:
                    self._last_saved_slot_hash[project_id] = static_hash
                    self._log_debug(f"✓ Slot saved → {filename}")
                    await self._cleanup_old_slot_files(project_id, filename)
                    return True
                else:
                    body_txt = await resp.text()
                    self._log_debug(
                        f"Slot save failed: HTTP {resp.status} — {body_txt}"
                    )
                    return False
        except Exception as e:
            self._log_debug(f"Slot save error: {e}")
            return False

    async def _cleanup_old_slot_files(self, project_id: str, keep_filename: str):
        """Remove obsolete slot files belonging to the same project."""
        slot_dir = self.valves.slot_save_path.rstrip("/")
        if not os.path.isdir(slot_dir):
            return
        project_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)[:20]
        prefix = f"slot{self.valves.slot_id}_{project_slug}_"
        try:
            for fname in os.listdir(slot_dir):
                if fname.startswith(prefix) and fname != keep_filename:
                    os.remove(os.path.join(slot_dir, fname))
                    self._log_debug(f"Removed obsolete slot file: {fname}")
        except Exception as e:
            self._log_debug(f"Slot cleanup error: {e}")

    def _invalidate_static_context_block(self, project_id: str, reason: str = ""):
        """
        Force regeneration of the Static Context Block on the next request.
        Call when content that belongs to Block A changes.
        """
        self._static_context_block_cache.pop(project_id, None)
        if reason:
            self._log_debug(f"SCB invalidated: {reason}")

    async def _get_path_context(
        self,
        project_id: str,
        user_query: str,
        intent_vector: Dict[str, float],
        messages: Optional[List[dict]] = None,  # ← PASO-27
    ) -> str:
        """
        Build code context using the graph‑activation system.

        Flow:
        1. Build an ActivationGraph from the query.
        2. Extract the activated subgraph.
        3. Assign each node a LOD level (0‑3) based on activation score
           and the intent vector.
        4. Inject:
           LOD‑3 (high): full code → placed last (Lost in the Middle)
           LOD‑2 (medium): signature + summary
           LOD‑1 (low): signature only
           LOD‑0 (minimal): name only → placed first (background)
        """
        if not self.valves.enable_path_analysis:
            return self._get_active_code_context(project_id, user_query)

        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""

        # Step 1: ActivationGraph
        ag = self._build_activation_graph(user_query, project_id, messages=messages)
        activated = ag.get_activated_nodes(
            threshold=self.valves.path_activation_threshold
        )

        if not activated:
            self._log_debug(
                "_get_path_context: no activated nodes, falling back to full context"
            )
            return self._get_active_code_context(project_id, user_query)

        # Step 2: Adjust LOD thresholds according to intent.
        debug_weight = intent_vector.get("debug", 0.2)
        modify_weight = intent_vector.get("modify", 0.3)
        refactor_weight = intent_vector.get("refactor", 0.1)

        lod3 = self.valves.lod3_threshold
        lod2 = self.valves.lod2_threshold
        lod1 = self.valves.lod1_threshold

        if debug_weight + modify_weight > 0.6:
            scale = 0.7
        elif refactor_weight > 0.4:
            scale = 0.0
        else:
            scale = 1.0

        lod3 *= scale
        lod2 *= scale
        lod1 *= scale

        # Step 3: Build context text with Lost in the Middle ordering
        total_tokens = 0
        budget = self.valves.active_context_max_tokens or 32000
        injected_blocks: Set[str] = set()

        sorted_nodes = sorted(activated.items(), key=lambda x: x[1], reverse=True)

        # ── Phase 6 (PASO-33): centrality LOD bump ─────────────────
        if self.valves.enable_centrality_lod_bump:
            centrality = self._node_centrality.get(project_id, {})
            threshold = self.valves.centrality_lod_bump_threshold
            adjusted = []
            for node_id, score in sorted_nodes:
                cent = centrality.get(node_id, 0.0)
                if cent >= threshold:
                    effective = min(
                        1.0, score + cent * self.valves.centrality_lod_bump_weight
                    )
                else:
                    effective = score
                adjusted.append((node_id, effective))
            sorted_nodes = adjusted  # use effective score for LOD decisions

        lod0_parts: List[str] = []  # name only
        lod1_parts: List[str] = []  # signature only
        lod2_parts: List[str] = []  # signature + summary
        lod3_parts: List[str] = []  # full code

        for node_id, score in sorted_nodes:
            if total_tokens >= budget:
                break

            # LOD‑0: score < lod1 → just the name
            if score < lod1:
                lod0_parts.append(f"`{node_id}`")
                total_tokens += 2
                continue

            block_hashes = self._symbol_index.find_blocks(node_id, project_id)
            for bh in block_hashes:
                if bh in injected_blocks:
                    continue
                block = state["active_blocks"].get(bh)
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
                    lod1_parts.append(f"- `{sig}`{loc} _(score: {score:.2f})_")
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
                    lod2_parts.append(f"{text}{loc} _(score: {score:.2f})_")
                    total_tokens += tok
                    injected_blocks.add(bh)

                else:
                    # LOD‑3: full code (with optional compression)
                    content_to_inject = block.content
                    tok = block._cached_token_count or (len(block.content) // 4)
                    if (
                        self.valves.enable_code_compression
                        and self._llmlingua_compressor
                        and tok > self.valves.code_compression_min_tokens
                    ):
                        content_to_inject = await self._compress_code_block(
                            block.content,
                            language=(
                                block.symbols[0].language
                                if block.symbols
                                else "unknown"
                            ),
                            rate=self.valves.code_compression_rate,
                            query=user_query,  # ← Phase 6 (PASO-31)
                        )
                        tok = self._estimate_code_tokens(content_to_inject)
                    if total_tokens + tok > budget:
                        break
                    loc = f" ({block.file_path})" if block.file_path else ""
                    lod3_parts.append(
                        f"### `{node_id}`{loc} [activation: {score:.2f}]\n"
                        f"```\n{content_to_inject}\n```\n"
                    )
                    total_tokens += tok
                    injected_blocks.add(bh)

                break  # use the first non‑obsolete block per symbol

        # Assemble respecting Lost in the Middle:
        # LOD‑0 + LOD‑1 (background), LOD‑2 (medium), LOD‑3 (most relevant, last)
        parts = ["## Code Context (activation-based LOD)\n"]
        if lod0_parts:
            parts.append(
                "**Known symbols** (minimal activation):\n" + ", ".join(lod0_parts)
            )
        if lod1_parts:
            parts.append("\n**Signatures** (low activation):\n" + "\n".join(lod1_parts))
        if lod2_parts:
            parts.append(
                "\n**Signatures + summaries** (medium activation):\n"
                + "\n".join(lod2_parts)
            )
        if lod3_parts:
            parts.append("\n### Directly relevant code (high activation)\n")
            parts.extend(lod3_parts)

        if len(parts) == 1:  # only header, no content
            return ""

        summary_line = (
            f"\n_(Context: {len(injected_blocks)} symbols, "
            f"~{total_tokens} tokens, "
            f"{len(activated)} nodes activated)_\n"
        )
        parts.append(summary_line)

        # ── v7 Phase 5 (PASO-29): LOD tracking for adaptive feedback ──
        if self.valves.enable_lod_adaptive:
            if not hasattr(self, "_last_lod_levels"):
                self._last_lod_levels: Dict[str, Dict[str, int]] = {}
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
            self._last_lod_levels[project_id] = lod_map
            self._log_debug(
                f"LOD tracking: {sum(1 for v in lod_map.values() if v == 3)} LOD-3, "
                f"{sum(1 for v in lod_map.values() if v == 2)} LOD-2, "
                f"{sum(1 for v in lod_map.values() if v <= 1)} LOD-0/1"
            )

        return "\n".join(parts)

    # --------------------------------------------------------------------------
    # Adaptive LOD feedback (v7 – Phase 5, PASO-29)
    # --------------------------------------------------------------------------

    async def _update_lod_thresholds_from_response(
        self, project_id: str, response_text: str
    ):
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
        if not self.valves.enable_lod_adaptive:
            return

        last_lod_map = getattr(self, "_last_lod_levels", {}).get(project_id, {})
        if not last_lod_map:
            return

        all_names = self._symbol_index.get_all_names(project_id)
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

        old_threshold = self.valves.lod3_threshold
        changed = False

        if len(underserved) >= self.valves.lod_adapt_underserved_min:
            # Lower threshold → more full code next time
            self.valves.lod3_threshold = max(
                self.valves.lod_adapt_min,
                self.valves.lod3_threshold - self.valves.lod_adapt_rate,
            )
            changed = True
            self._log_debug(
                f"LOD adaptive ↓: threshold {old_threshold:.2f} → "
                f"{self.valves.lod3_threshold:.2f} "
                f"({len(underserved)} underserved: {underserved[:3]})"
            )
        elif len(overserved) >= self.valves.lod_adapt_overserved_min:
            # Raise threshold → fewer unnecessary expansions
            self.valves.lod3_threshold = min(
                self.valves.lod_adapt_max,
                self.valves.lod3_threshold + self.valves.lod_adapt_rate * 0.5,
            )
            changed = True
            self._log_debug(
                f"LOD adaptive ↑: threshold {old_threshold:.2f} → "
                f"{self.valves.lod3_threshold:.2f} "
                f"({len(overserved)} overserved symbols)"
            )

        if not changed:
            self._log_debug(
                f"LOD adaptive: no adjustment needed "
                f"(threshold={self.valves.lod3_threshold:.2f})"
            )

    # ── Scientific CoT (v7) ──
    def _gather_static_evidence(
        self, hypothesis_text: str, project_id: str
    ) -> StaticEvidence:
        """
        Gather deterministic evidence about the structural claims in a hypothesis.
        No LLM. No GPU. Instant.

        v2: uses SymbolIndex with typed edges for more precise validation.
        v7 (PASO-19): includes data flow upstream information.
        """
        all_names = self._symbol_index.get_all_names(project_id)
        state = self._get_state(project_id)

        # ── 1. Symbols mentioned in the hypothesis ──────────────────────
        words = set(re.findall(r"\b\w+\b", hypothesis_text))
        mentioned = all_names.intersection(words)

        symbols_found = {
            name: bool(self._symbol_index.find_blocks(name, project_id))
            for name in mentioned
        }

        # ── 2. Claimed call relationships ───────────────────────────────
        # Detects patterns: "A calls B", "A uses B", "A invokes B"
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
            # Verify using typed edges (more precise than symbol.calls)
            caller_edges = self._symbol_index.get_edges_out(caller, project_id)
            verified = any(
                e.dst == callee and e.type in ("calls", "reads", "writes")
                for e in caller_edges
            )
            call_relations_valid[key] = verified

        # ── 3. Recent changes (last hour) ───────────────────────────────
        now = time.time()
        recent_window = 3600
        recent_changes = [
            name
            for name in mentioned
            if any(
                state["active_blocks"].get(bh) is not None
                and (now - state["active_blocks"][bh].timestamp) < recent_window
                for bh in self._symbol_index.find_blocks(name, project_id)
            )
        ]

        # ── 4. Entry points mentioned ───────────────────────────────────
        all_views = self._path_index.get_all(project_id)
        entry_points_mentioned = [
            v.entry_point for v in all_views if v.entry_point in mentioned
        ]

        # ── 5. Path memberships (using PathIndex) ───────────────────────
        path_memberships: Dict[str, List[str]] = {}
        for name in mentioned:
            path_memberships[name] = self._path_index.mark_stale_for_symbol(
                name, project_id
            )
            # Note: mark_stale_for_symbol only reads; it's safe here.

        # ── 6. Data flow upstream (backward slicing) ────────────────────
        data_flow_upstream: Dict[str, List[str]] = {}
        if mentioned:
            for sym_name in mentioned:
                incoming_edges = self._symbol_index.get_edges_in(sym_name, project_id)
                data_flow_sources = [
                    e.src for e in incoming_edges if e.type == "data_flow"
                ]
                if data_flow_sources:
                    data_flow_upstream[sym_name] = data_flow_sources

        # ── 7. Objective score ─────────────────────────────────────────
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
            data_flow_upstream=data_flow_upstream,  # ← v7 (PASO-19)
            objective_score=objective_score,
        )

    # --------------------------------------------------------------------------
    # Intent detection (natural language)
    # --------------------------------------------------------------------------
    async def _parse_all_intents(self, user_message: str) -> Dict[str, Any]:
        if not self.valves.enable_natural_language_forget:
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

        code_spans = await self._get_code_spans(user_message)
        # (ya no filtramos con _should_parse_intents; la detección LLM se encarga)

        cleaned = self._remove_code_spans(user_message, code_spans).strip()

        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
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
        t0 = time.monotonic()
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=200,
            temperature=0.0,
            timeout=8.0,
        )
        dur = time.monotonic() - t0
        self._log_timing("parse_intents_llm", dur, dur)

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

    # --------------------------------------------------------------------------
    # Parallel context checks
    # --------------------------------------------------------------------------
    @staticmethod
    async def _noop():
        return None

    async def _parallel_context_checks(
        self,
        messages: List[dict],
        query: str,
        context_hash: str,
        project_id: str,
        state: dict,
        skip_contradiction: bool = False,
    ) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        tasks = [
            (
                self._detect_contradictions(messages)
                if (
                    self.valves.enable_contradiction_detection
                    and not skip_contradiction
                )
                else self._noop()
            ),
            (
                self._find_cached_response(query, context_hash, state)
                if (self.valves.enable_response_cache and HAS_SENTENCE)
                else self._noop()
            ),
            (
                self._find_duplicate_question(query, project_id)
                if (self.valves.duplicate_question_threshold and HAS_SENTENCE)
                else self._noop()
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        contradiction = results[0] if not isinstance(results[0], Exception) else None
        cached = results[1] if not isinstance(results[1], Exception) else None
        duplicate = results[2] if not isinstance(results[2], Exception) else None
        return contradiction, cached, duplicate

    # --------------------------------------------------------------------------
    # Contradiction detection
    # --------------------------------------------------------------------------
    async def _detect_contradictions(self, messages: List[dict]) -> Optional[str]:
        if not self.valves.enable_contradiction_detection or len(messages) < 3:
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

        prompt = (
            f"Conversation history:\n{history_text[-2000:]}\n\n"
            f"New user message:\n{last_user['content'][:500]}\n\n"
            "Does the new message contradict any previously established fact or decision in the history? "
            "Answer only 'yes' or 'no'."
        )
        model = self.valves.contradiction_detection_model or self.valves.llm_model
        t0 = time.monotonic()
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="You are a contradiction detector. Answer only 'yes' or 'no'.",
            model_override=model,
            max_tokens=3,
            temperature=0.0,
        )
        dur = time.monotonic() - t0
        self._log_timing("detect_contradictions_llm", dur, dur)
        if response and response.strip().lower().startswith("yes"):
            return (
                "⚠️ **Contradiction detected**: The last message appears to contradict something established earlier. "
                "Please review and clarify if needed."
            )
        return None

    # --------------------------------------------------------------------------
    # Duplicate question detection
    # --------------------------------------------------------------------------
    async def _find_duplicate_question(
        self, query: str, project_id: str
    ) -> Optional[dict]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return None
        if not query or len(query.strip()) < 15:
            return None
        try:
            q_emb = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(query[:1000]).tolist()
            )
            now = time.time()
            where = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"role": {"$eq": "user"}},
                    {
                        "timestamp": {
                            "$gt": time.time()
                            - self.valves.duplicate_question_lookback_hours * 3600
                        }
                    },
                ]
            }
            results = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.query(
                    query_embeddings=[q_emb],
                    n_results=self.valves.duplicate_question_lookback,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            )
            if not results or not results["ids"] or not results["ids"][0]:
                return None
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i]
                sim = 1.0 - (dist / 2.0)
                if sim >= self.valves.duplicate_question_threshold and doc != query:
                    self._log_debug(f"Duplicate question found (sim={sim:.3f})")
                    return {"sim": sim, "doc": doc}
        except Exception as e:
            self._log_debug(f"Error in duplicate question detection: {e}")
        return None

    # --------------------------------------------------------------------------
    # Chain‑of‑Thought helpers
    # --------------------------------------------------------------------------
    async def _parse_cot_intent(self, user_content: str) -> Tuple[Optional[str], int]:
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

    async def _detect_cot_level(self, user_content, is_code_session, state):
        """Determine CoT depth, optionally storing it in conversation state."""
        if not user_content:
            return 0

        # ── v7 (PASO-15): force Level 3 Scientific CoT if valve is enabled ──
        if self.valves.enforce_scientific_method:
            self._log_debug("CoT: enforce_scientific_method=True → forcing Level 3")
            return 3

        if self.valves.enable_cot_llm_detection:
            level = await self._detect_cot_level_via_llm(
                user_content, is_code_session, state
            )
        else:
            level = self._detect_cot_level_heuristic(
                user_content, is_code_session, state
            )

        # Persist level for conversational continuity if feature is enabled
        if self.ENABLE_COT_STICKY:
            state["last_cot_level"] = level

        return level

    def _detect_cot_level_heuristic(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        """
        Determine the depth of Chain-of-Thought reasoning needed.

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
        if self.ENABLE_ACCENT_NORMALIZATION:
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
        length_ok = len(user_content) >= self.valves.auto_cot_min_chars
        too_short = word_count < 5

        # ── Context: expand active set when in a code session ──────────────
        active_complex = set(complex_keywords_generic)
        if is_code_session:
            active_complex |= complex_keywords_code_only

        # ── Negation guard (uses class‑level prefixes) ─────────────────────
        def _is_negated(text: str, kw: str) -> bool:
            """Check if *every* occurrence of `kw` is negated (at least one non‑negated → False)."""
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx == -1:
                    break
                before = text[:idx].strip().split()[-3:]
                if not any(neg in before for neg in self._COT_NEGATION_PREFIXES):
                    return False
                start = idx + 1
            return True

        # ── Pre‑sort keyword lists once for efficient compound matching ────
        _sorted_deep = sorted(deep_keywords, key=len, reverse=True)
        _sorted_complex = sorted(active_complex, key=len, reverse=True)

        def _contains_any(text: str, sorted_kw: list) -> bool:
            """Check if text contains any keyword (longest first), respecting negation."""
            for kw in sorted_kw:
                if kw in text and not _is_negated(text, kw):
                    return True
            return False

        # ── Guard: very short messages → inconclusive ──────────────────────
        if too_short and not has_code:
            return 0

        # ── Level 3: deep (highest priority) ───────────────────────────────
        if _contains_any(content_lower, _sorted_deep):
            return 3

        # ── Signals for level 1 / 2 ────────────────────────────────────────
        has_complex_kw = _contains_any(content_lower, _sorted_complex)

        # Combine redundant length signals into a single elaborated-message flag
        is_elaborate = length_ok or word_count > 30

        signals = 0
        if self.ENABLE_KEYWORD_COUNT_WEIGHT:
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
        # Bonus: code session + any complex keyword → higher confidence
        if is_code_session and has_complex_kw:
            signals += 1

        # ── "in detail" / "en detalle" as a weak independent signal ───────
        for phrase in ("in detail", "en detalle"):
            if phrase in content_lower and not _is_negated(content_lower, phrase):
                signals += 1
                break

        # ── Stack trace detection (code sessions only) ────────────────────
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

        # ── Multiple code blocks → comparison / review → level 2 ──────────
        code_block_count = user_content.count("```") // 2
        if code_block_count >= 2:
            signals += 1

        # ── Optional: conversation history ─────────────────────────────────
        if self.ENABLE_COT_STICKY:
            prev_level = state.get("last_cot_level", 0)
            if prev_level >= 2 and has_complex_kw:
                signals += 1

        # ── Thresholds ─────────────────────────────────────────────────────
        if signals >= 5:
            return 2
        elif signals >= 3:
            return 1
        else:
            return 0  # inconclusive → LLM decides

    async def _detect_cot_level_via_llm(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        t0 = time.monotonic()
        prompt = (
            f"The user is working on a {'code' if is_code_session else 'general'} task.\n"
            f"User message:\n{user_content[:500]}\n\n"
            "Decide the depth of Chain-of-Thought reasoning needed:\n"
            "0 = none (simple fact, greeting, trivial)\n"
            "1 = basic (ask to think step by step internally)\n"
            "2 = moderate (generate reasoning automatically)\n"
            "3 = deep (generate reasoning + self-critique)\n\n"
        )
        # Include user intent regarding code completeness
        if (
            hasattr(self, "_user_intent_full_code")
            and self._user_intent_full_code is not None
        ):
            intent_note = (
                "The user likely needs the full code."
                if self._user_intent_full_code
                else "The user likely needs only a summary of the code."
            )
            prompt += f"{intent_note}\n"
        prompt += "Respond with only the digit 0, 1, 2, or 3."

        try:
            response = await self._try_llm_quick(
                prompt=prompt,
                system_prompt="You are a classifier. Output only a single digit.",
                model_override=self.valves.cot_detection_model,
                max_tokens=2,
                temperature=0.0,
            )
            if response and response.strip().isdigit():
                level = int(response.strip())
                if 0 <= level <= 3:
                    dur = time.monotonic() - t0
                    self._log_timing("cot_detection_llm", dur, dur)
                    return level
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log_debug(f"LLM CoT detection failed, falling back to heuristic: {e}")

        dur = time.monotonic() - t0
        self._log_timing("cot_detection_llm_fallback", dur, dur)
        return self._detect_cot_level_heuristic(user_content, is_code_session, state)

    async def _generate_cot_reasoning(
        self, question: str, context: str, label: str = ""
    ) -> str:
        effective_max_tokens = (
            self.valves.cot_max_tokens if self.valves.cot_max_tokens > 0 else None
        )

        # ── Phase 6 (PASO-35): Step-Back context ──────────────────
        step_back = await self._generate_step_back_context(question, context)

        # Prepend step-back to context if available
        enriched_context = step_back + context if step_back else context

        prompt = (
            f"Context:\n{enriched_context}\n\n"
            f"Question:\n{question}\n\n"
            "Think step by step and provide your reasoning:"
        )
        response = await self._call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a helpful assistant that thinks step by step before answering."
            ),
            model_override=self.valves.cot_model_level2,
            max_tokens=effective_max_tokens,
            temperature=0.4,
            label=label,
        )
        if response:
            prefix = (
                "## 🔎 Automated Chain-of-Thought Reasoning (Level 2)\n"
                f"*Generated by {self.valves.cot_model_level2}.*"
            )
            if step_back:
                prefix += " *Includes step-back architectural context.*"
            return f"{prefix}\n\n{response}"
        return "Unable to generate reasoning."

    async def _generate_scientific_reasoning_L3(
        self,
        question: str,
        context: str,
        project_id: str,
        label: str = "",
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

        This is the Level 3 CoT that replaces the old self-reflection only approach.
        """
        max_hypotheses = self.valves.scientific_hypotheses_count
        threshold = self.valves.scientific_confidence_threshold
        max_iters = self.valves.scientific_max_iterations

        # ── Helpers ─────────────────────────────────────────────────
        def _parse_hypotheses_from_response(text: str) -> List[Tuple[str, float]]:
            """Extract a list of (hypothesis_text, confidence) from LLM output."""
            results = []
            # Expected format: Hypothesis: ... Confidence: 0.8
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
        response = await self._call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a scientific reasoning engine. Output exactly the requested "
                "hypotheses with confidence scores. No extra commentary."
            ),
            model_override=self.valves.cot_model_level3,
            max_tokens=600,
            temperature=0.4,
            label=label + "_gen_hypotheses" if label else "sci_gen_hypotheses",
        )

        if not response:
            return "Unable to generate hypotheses for scientific reasoning."

        hypotheses = _parse_hypotheses_from_response(response)
        if len(hypotheses) < 2:
            # Not enough hypotheses → fallback to plain CoT
            return await self._generate_cot_reasoning(question, context, label)

        best_hypothesis = ""
        best_combined_score = 0.0
        iteration = 0

        # ── Iterative refinement loop ──────────────────────────────
        while iteration < max_iters:
            iteration += 1
            scored = []
            for hyp_text, llm_conf in hypotheses:
                # Gather deterministic evidence
                evidence = self._gather_static_evidence(hyp_text, project_id)
                obj_score = evidence.objective_score

                # Combined score: 50% structural evidence, 50% LLM confidence
                combined = 0.5 * obj_score + 0.5 * llm_conf
                scored.append((hyp_text, combined, obj_score, llm_conf, evidence))

            # Sort by combined score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[0]
            best_hypothesis, best_combined, best_obj, best_llm_conf, best_evidence = top

            self._log_debug(
                f"Scientific CoT iter {iteration}: best hypothesis "
                f"'{best_hypothesis[:80]}...' "
                f"score={best_combined:.3f} "
                f"(obj={best_obj:.3f}, llm_conf={best_llm_conf:.3f})"
            )

            # Stopping condition
            if best_combined >= threshold or iteration >= max_iters:
                break

            # ── Refine: ask LLM to improve hypotheses using evidence ──
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
            refine_response = await self._call_llm(
                prompt=refine_prompt,
                system_prompt="You are a scientific reasoning engine refining hypotheses based on evidence.",
                model_override=self.valves.cot_model_level3,
                max_tokens=600,
                temperature=0.4,
                label=label + "_refine" if label else "sci_refine",
            )
            if refine_response:
                new_hypotheses = _parse_hypotheses_from_response(refine_response)
                if len(new_hypotheses) >= 2:
                    hypotheses = new_hypotheses
                else:
                    # If parsing fails, keep old hypotheses and break
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
        reasoning = await self._call_llm(
            prompt=final_prompt,
            system_prompt="You are a helpful assistant that reasons step by step based on verified evidence.",
            model_override=self.valves.cot_model_level3,
            max_tokens=(
                self.valves.cot_max_tokens if self.valves.cot_max_tokens > 0 else None
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

    # --------------------------------------------------------------------------
    # Step-Back Prompting (v7 – Phase 6, PASO-35)
    # --------------------------------------------------------------------------

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
        if not self.valves.enable_step_back_prompting:
            return ""
        if len(question.strip()) < 15:
            return ""

        # Step-back is most useful for debugging queries
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
            if not self.valves.step_back_always:
                return ""

        # Generate the abstract question
        step_back_prompt = (
            f"A programmer is debugging this specific issue:\n{question[:300]}\n\n"
            "What is the underlying architectural principle, design invariant, or "
            "general concept that governs correct behavior here? "
            "State it as an abstract question and answer it in 2-3 sentences. "
            "Focus on system-level understanding, not the specific bug."
        )

        step_back_response = await self._try_llm_quick(
            prompt=step_back_prompt,
            system_prompt=(
                "You are a senior software architect. "
                "Answer the abstract question concisely (2-3 sentences). "
                "Focus on principles, not the specific implementation."
            ),
            model_override=self.valves.cot_model_level2,
            max_tokens=self.valves.step_back_max_tokens,
            temperature=0.3,
            timeout=30.0,
        )

        if step_back_response and step_back_response.strip():
            self._log_debug(
                "Step-back context generated "
                f"({len(step_back_response.split())} words)"
            )
            return (
                "## Architectural Context (Step-Back)\n"
                f"{step_back_response.strip()}\n\n"
                "---\n\n"
            )
        return ""

    # --------------------------------------------------------------------------
    # Feedback context
    # --------------------------------------------------------------------------
    def _get_feedback_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        feedback = state.get("feedback_history", [])
        if not feedback:
            return ""
        recent = feedback[-self.valves.feedback_history_limit :]
        lines = ["## Previous Feedback"]
        for fb in recent:
            success = "✅" if fb.success else "❌"
            lines.append(f"- {success} {fb.change_description[:100]}")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Proactive summary suggestion
    # --------------------------------------------------------------------------
    async def _check_and_suggest_summarization(
        self, project_id: str, total_tokens: int, max_tokens: int
    ) -> Optional[str]:
        if max_tokens <= 0:
            return None
        ratio = total_tokens / max_tokens
        if ratio > self.valves.proactive_summary_threshold:
            return (
                f"[CodeAware] The conversation is using {total_tokens}/{max_tokens} tokens (≈{int(ratio*100)}%). "
                "Consider using `/forget` or asking me to summarize older parts."
            )
        return None

    # --------------------------------------------------------------------------
    # Summarization of old messages
    # --------------------------------------------------------------------------
    async def _summarize_messages(
        self, old_messages: List[dict], is_code_context: bool = False
    ) -> Optional[str]:
        if not old_messages:
            return None
        combined = "\n".join(
            [m.get("content", "") for m in old_messages if m.get("content")]
        )
        if not combined.strip():
            return None
        prompt = f"Summarize the following conversation segment, preserving key decisions and code changes:\n\n{combined[:4000]}"
        system_prompt = (
            "You produce concise summaries of technical conversations."
            if is_code_context
            else "You produce concise summaries."
        )
        summary = await self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model_override=self.valves.summarization_model,
            max_tokens=500,
            temperature=0.3,
        )
        return summary.strip() if summary else None

    # --------------------------------------------------------------------------
    # Command suggestion
    # --------------------------------------------------------------------------
    async def _suggest_commands(self, project_id: str, state: dict) -> Optional[str]:
        if not self.valves.enable_command_suggestions:
            return None
        now = time.time()
        last_sugg = state.get("last_suggestion_timestamp", 0)
        if now - last_sugg < self.valves.command_suggestion_cooldown_minutes * 60:
            return None
        if state["message_count"] > 15 and not state.get("has_any_calls"):
            state["last_suggestion_timestamp"] = now
            return (
                "[CodeAware] Tip: You can manage context with commands like `/forget`, `/remember`, `/status`, `/clean`. "
                "Use `/help` for more info."
            )
        return None

    # --------------------------------------------------------------------------
    # Intent execution (forget, remember, obsolete)
    # --------------------------------------------------------------------------
    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return "No active context to forget."

            action = intent.get("action")
            if action == "forget_all":
                return (
                    "⚠️ For safety, the natural language 'forget all' is disabled. "
                    "Please type `/forget all` explicitly to confirm."
                )

            if action == "forget_last":
                if state["active_blocks"]:
                    last_hash = max(
                        state["active_blocks"].keys(),
                        key=lambda h: state["active_blocks"][h].timestamp,
                    )
                    block = state["active_blocks"].get(last_hash)
                    if block:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state["active_blocks"][last_hash]
                    self._invalidate_lightweight_cache(project_id)
                return "Forgotten the last context block."

            elif action == "forget_n":
                n = intent.get("n", 1)
                blocks_by_time = sorted(
                    state["active_blocks"].items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True,
                )
                removed = 0
                for h, block in blocks_by_time[:n]:
                    if h in state["active_blocks"]:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                        del state["active_blocks"][h]
                        removed += 1
                if removed:
                    self._invalidate_lightweight_cache(project_id)
                return f"Forgotten the last {removed} context block(s)."

            elif action == "forget_file":
                file_path = intent.get("file", "")
                if not file_path:
                    return "No file specified."
                to_remove = [
                    h
                    for h, blk in state["active_blocks"].items()
                    if blk.file_path and file_path in blk.file_path
                ]
                for h in to_remove:
                    block = state["active_blocks"].get(h)
                    if block:
                        self._symbol_index.remove_all_for_block(
                            block.hash, block.symbols, project_id
                        )
                    del state["active_blocks"][h]
                if to_remove:
                    self._invalidate_lightweight_cache(project_id)
                return f"Forgotten {len(to_remove)} block(s) related to {file_path}."

            elif action == "forget_block":
                block_id = intent.get("hash") or intent.get("id") or ""
                if not block_id:
                    return "No block specified."
                if block_id in state["active_blocks"]:
                    block = state["active_blocks"][block_id]
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                    del state["active_blocks"][block_id]
                    self._invalidate_lightweight_cache(project_id)
                    return f"Forgotten block {block_id}."
                matches = [h for h in state["active_blocks"] if block_id in h]
                if matches:
                    for h in matches:
                        block = state["active_blocks"].get(h)
                        if block:
                            self._symbol_index.remove_all_for_block(
                                block.hash, block.symbols, project_id
                            )
                        del state["active_blocks"][h]
                    self._invalidate_lightweight_cache(project_id)
                    return f"Forgotten {len(matches)} block(s) matching {block_id}."
                return f"No block found for {block_id}."

            else:
                return "Unrecognized forget action."

    async def _execute_remember_intent(self, project_id: str, intent: Dict) -> str:
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
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
            blocks = list(state["active_blocks"].values())
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
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return "No active context to mark as obsolete."

            action = intent.get("action", "")
            if action == "obsolete_all":
                return (
                    "⚠️ For safety, the natural language 'obsolete all' is disabled. "
                    "Please type `/obsolete all` explicitly to confirm."
                )

            blocks = list(state["active_blocks"].values())
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

    async def _run_change_summary_task(
        self, params: dict, model: str, sem: asyncio.Semaphore
    ) -> bool:
        """Execute a deferred change summary generation."""
        block_hash = params["block_hash"]
        prev_content = params["prev_content"]
        new_content = params["new_content"]
        prompt = (
            f"Summarise the code change in ONE short sentence (max 15 words).\n\n"
            f"Previous:\n```\n{prev_content[:500]}\n```\n\n"
            f"New:\n```\n{new_content[:500]}\n```\n\n"
            f"Change summary:"
        )
        async with sem:
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a code change summariser. Output only one short sentence.",
                model_override=model,
                max_tokens=40,
                temperature=0.1,
                label="change_summary",
            )
        if summary:
            now = time.time()
            self._block_change_summaries[block_hash] = (summary.strip(), now)
            if len(self._block_change_summaries) > self._MAX_CHANGE_SUMMARIES:
                self._block_change_summaries.popitem(last=False)
            return True
        return False

    async def _run_missing_summaries_task(
        self, params: dict, model: str, sem: asyncio.Semaphore
    ) -> bool:
        """Execute a deferred missing summary generation for one symbol."""
        signature = params["signature"]
        code_snippet = params["code_snippet"]
        prompt = f"Summarize in one short sentence what this code does:\n\n```{signature}\n{code_snippet}```"
        async with sem:
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a code summarization assistant. Output only one concise sentence.",
                model_override=model,
                max_tokens=50,
                temperature=0.1,
                label="missing_summaries",
            )
        if summary and summary.strip():
            project_id = params["project_id"]
            lock = await self._get_project_lock(project_id)
            async with lock:
                state = self._get_state(project_id)
                for blk in state["active_blocks"].values():
                    for sym in blk.symbols:
                        if sym.signature == signature:
                            sym.summary = summary.strip()
                self._set_state(project_id, state)
            return True
        return False

    async def _run_inactive_code_summary_task(
        self, params: dict, model: str, sem: asyncio.Semaphore
    ) -> bool:
        """Execute a deferred summarisation of an inactive code block."""
        sig = params.get("signature", "")
        content = params["content"]
        if sig:
            prompt = f"The code block has signature: {sig}\nProvide a very brief description of what this code does.\nCode:\n```{content[:1000]}```"
        else:
            prompt = f"Summarise the following code block.\n```{content[:1500]}```"
        async with sem:
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a code summarization assistant.",
                model_override=model,
                max_tokens=200,
                temperature=0.2,
                label="inactive_code_summary",
            )
        if summary and summary.strip():
            project_id = params["project_id"]
            block_hash = params["block_hash"]
            lock = await self._get_project_lock(project_id)
            async with lock:
                state = self._get_state(project_id)
                if block_hash in state["active_blocks"]:
                    blk = state["active_blocks"][block_hash]
                    summary_content = f"[Summary of inactive code]\n{summary.strip()}"
                    blk.content = summary_content
                    blk.importance_score *= 0.5
                    self._invalidate_lightweight_cache(project_id)
                    self._set_state(project_id, state)
            return True
        return False

    async def _run_session_summary_task(
        self, params: dict, model: str, sem: asyncio.Semaphore
    ) -> bool:
        """Generate an autobiographical session summary and store it in LTM."""
        project_id = params["project_id"]
        code_state_hash = params.get("code_state_hash", "")

        recent = await self._retrieve_historical_messages(
            query="recent conversation summary",
            project_id=project_id,
            limit=self.valves.session_summary_interval_messages,
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
        async with sem:
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a helpful assistant that produces concise autobiographical session summaries.",
                model_override=model,
                max_tokens=self.valves.session_summary_max_tokens,
                temperature=0.2,
                label="session_summary",
            )
        if not summary:
            return False

        msg_id = f"{project_id}_session_summary_{int(time.time())}"
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(summary).tolist()
        )
        await anyio.to_thread.run_sync(
            lambda: self.memory_collection.upsert(
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
        self._log_debug(f"Session summary stored in LTM (msg_id={msg_id})")
        return True

    # --------------------------------------------------------------------------
    # Inlet helper methods
    # --------------------------------------------------------------------------
    async def _inlet_preprocess(self, body: dict, project_id: str) -> dict:
        """Handle project switching, symbol cache loading. No secondary tasks here."""
        messages = body.get("messages", [])

        if self._last_project_id and self._last_project_id != project_id:
            self._log_debug(
                f"Project changed from {self._last_project_id} to {project_id}"
            )
            old_state = self._conversation_state.get(self._last_project_id)
            if old_state:
                self._remove_project_from_index_by_id(self._last_project_id, old_state)
            self._cached_lightweight_context.pop(self._last_project_id, None)
            self._block_change_summaries.clear()
        self._last_project_id = project_id

        # ── v7 (PASO-15): load persisted CodePathViews if index is empty ──
        if self.valves.enable_path_analysis and HAS_TREE_SITTER:
            existing_views = self._path_index.get_all(project_id)
            all_names = self._symbol_index.get_all_names(project_id)
            if all_names and not existing_views:
                self._log_debug("PathIndex empty but symbols exist — loading from DB")
                db_views = await self._load_path_views_from_db(project_id)
                for view in db_views:
                    self._path_index.add(view, project_id)

        # ── v7 Phase 5 (PASO-28): restore typed edges from DB ─────────
        if self.valves.enable_edge_persistence:
            restored = await self._load_symbol_edges_from_db(project_id)
            if restored > 0:
                self._log_debug(
                    f"Cross-session: {restored} symbol edges restored from DB. "
                    f"No need to re-paste code."
                )

        # ── v7 (PASO-25): restore KV slot at session start (once per project) ──
        if (
            self.valves.enable_slot_persistence
            and project_id not in self._slot_restore_attempted
            and project_id in self._static_context_block_cache
        ):
            self._background_task(
                self._slot_restore_if_available(project_id),
                name="slot_restore",
            )

        return messages

    async def _inlet_extract_user_info(self, messages: List[dict]):
        """Extract last user message, question, and detect explicit commands."""
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        user_query = last_user_msg.get("content", "") if last_user_msg else ""

        # Extract real question and determine if there are code blocks
        has_code_blocks = False
        user_question = user_query
        if last_user_msg and user_query:
            try:
                spans = await self._get_code_spans(user_query)
                if spans:
                    user_question = self._remove_code_spans(user_query, spans).strip()
                # Check for fenced code blocks independently of tree‑sitter
                if "```" in user_query:
                    has_code_blocks = True
                # If spans were found, code blocks exist
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

    async def _inlet_handle_explicit_commands(
        self,
        messages: List[dict],
        project_id: str,
        is_explicit_command: bool,
        last_user_msg: Optional[dict],
        __user__: Optional[dict],
    ) -> Tuple[bool, Optional[List[dict]]]:
        """Handle /forget, /status, /clean, /expand.
        Returns (handled, messages) if a command was processed, else (False, None).
        """
        if not last_user_msg:
            return False, None

        content = last_user_msg.get("content", "").strip()

        # /forget
        if self.valves.enable_forget_command and is_explicit_command:
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                return True, self._ensure_last_message_is_user(new_messages)

        # /status
        if (
            content == "/status"
            and self.valves.cleanup_status_command_enabled
            and self.valves.cleanup_suggestions_enabled
        ):
            candidates = self._get_inactive_block_candidates(project_id)
            if not candidates:
                response = "✅ No inactive blocks detected."
            else:
                lines = [
                    f"⚠️ {len(candidates)} inactive block(s) (not mentioned in last {self.valves.cleanup_inactive_threshold_messages} messages):"
                ]
                state = self._get_state(project_id)
                for h in candidates:
                    blk = state["active_blocks"].get(h)
                    if blk:
                        snippet = blk.content[:80].replace("\n", " ")
                        file_info = f" ({blk.file_path})" if blk.file_path else ""
                        lines.append(f"- `{h[:8]}...`{file_info}: {snippet}...")
                response = "\n".join(lines)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._ensure_last_message_is_user(messages)

        # /clean
        if (
            content.startswith("/clean")
            and self.valves.cleanup_command_enabled
            and self.valves.cleanup_suggestions_enabled
        ):
            response = await self._handle_clean_command(content, project_id)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._ensure_last_message_is_user(messages)

        # /expand
        if content.startswith("/expand"):
            response = await self._handle_expand_command(content, project_id)
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            return True, self._ensure_last_message_is_user(messages)

        return False, None

    async def _inlet_handle_natural_intents(
        self,
        messages: List[dict],
        project_id: str,
        is_explicit_command: bool,
        last_user_msg: Optional[dict],
        slot_free: bool = True,
    ) -> Tuple[bool, Optional[List[dict]]]:
        """Handle natural language intents (forget, remember, obsolete).
        Returns (handled, messages) if an intent was processed, else (False, None).
        """
        if (
            not self.valves.enable_natural_language_forget
            or not last_user_msg
            or is_explicit_command
            or self._has_code_indicators(last_user_msg.get("content", ""))
        ):
            return False, None

        # Skip LLM-based intent parsing if the main model is loaded
        if not slot_free:
            self._log_debug(
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
            elif intent_type == "obsolete" and self.valves.enable_obsolete_marking:
                confirmation = await self._execute_obsolete_intent(project_id, fi)
            else:
                continue

            status_msg = f"[CodeAware] {confirmation}"
            messages.insert(0, {"role": "system", "content": status_msg})
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return True, self._ensure_last_message_is_user(messages)

        return False, None

    async def _inlet_prepare_code_session(
        self,
        messages: List[dict],
        project_id: str,
        user_query: str,
    ) -> Tuple[bool, str]:
        """Classify session, update active code, clean user question.
        Returns (is_code_session, user_question).
        """
        is_code_session = await self._classify_session(messages, project_id)

        if self.valves.enable_code_awareness and is_code_session:
            last_idx = len(messages) - 1
            await self._update_active_code(messages[last_idx], project_id)
            extracted_blocks, block_spans = await self._extract_code_blocks(user_query)
            if block_spans:
                user_question = self._remove_code_spans(user_query, block_spans).strip()
                if not user_question or len(user_question) < 10:
                    user_question = user_query
            else:
                user_question = user_query
            self._last_processed_message_idx[project_id] = last_idx
        else:
            user_question = user_query

        return is_code_session, user_question

    async def _inlet_build_system_injections(
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
        Build the system prompt in two separate blocks:

        - static_block (Block A): stable content, placed first.
        - dynamic_injections (Block B): per-query content, placed after.

        Returns: (static_block, dynamic_injections, cached_response, prelim_system)
        """
        # ══════════════════════════════════════════════════════════════
        # BLOCK A — STATIC
        # ══════════════════════════════════════════════════════════════
        self._log_debug("🧱 Block A (static): building / retrieving from cache")
        static_block = await self._get_static_context_block(project_id, is_code_session)

        # ══════════════════════════════════════════════════════════════
        # BLOCK B — DYNAMIC (per-query)
        # ══════════════════════════════════════════════════════════════
        dynamic_injections: List[Tuple[str, str]] = []

        # ── Step B1: LTM per-query retrieval ─────────────────────────
        self._log_debug("🔄 Block B – Step 1/5: LTM per-query retrieval")
        if (
            self.valves.enable_code_awareness
            and is_code_session
            and not self.valves.smart_context_selection
            and HAS_SENTENCE
            and HAS_CHROMA
            and user_query
        ):
            all_meta = await self._retrieve_all_memories_unified(user_query, project_id)
            all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
            unique_meta = []
            seen_docs: Set[str] = set()
            for m in all_meta:
                if m["doc"] not in seen_docs:
                    seen_docs.add(m["doc"])
                    unique_meta.append(m)

            max_ltm = self.valves.ltm_retrieval_max_tokens
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
                frag_tok = (
                    len(self.tokenizer.encode(text))
                    if self.tokenizer
                    else len(text) // 4
                )
                if max_ltm > 0 and current_tokens + frag_tok > max_ltm:
                    continue
                parts.append(text)
                current_tokens += frag_tok
            if parts:
                ltm_text = header + "\n---\n".join(parts)
                dynamic_injections.append(("high", ltm_text))
                self._log_debug("🔄 Block B – Step 1/5: LTM injected")

        # ── Step B2: Parallel checks (contradiction, cache, duplicate) ─
        self._log_debug("🔄 Block B – Step 2/5: Parallel checks")
        context_hash = self._compute_context_hash(messages)
        contradiction_warning = None
        cached_response = None
        duplicate_match = None

        if last_user_msg:
            contradiction_warning, cached_response, duplicate_match = (
                await self._parallel_context_checks(
                    messages, user_query, context_hash, project_id, state
                )
            )

        if cached_response:
            return static_block, [], cached_response, ""

        if contradiction_warning and self.valves.contradiction_inject_warning:
            dynamic_injections.append(("medium", contradiction_warning))
        if duplicate_match:
            dynamic_injections.append(
                (
                    "medium",
                    f"⚠️ **Note**: Similar question asked before "
                    f"(similarity {duplicate_match['sim']:.2f}).",
                )
            )

        # ── Step B3: Activated code (per-query, varies each request) ──
        self._log_debug("🔄 Block B – Step 3/5: Code activated by query")
        if is_code_session and self.valves.enable_code_awareness:
            # ── v7: graph‑based path context ────────────────
            if self.valves.enable_path_analysis:
                intent_vector = await self._classify_intent(user_query, project_id)
                active_ctx = await self._get_path_context(
                    project_id, user_query, intent_vector, messages=messages
                )
                if not active_ctx:
                    active_ctx = self._get_active_code_context(project_id, user_query)
                if active_ctx:
                    dynamic_injections.append(("critical", active_ctx))
            else:
                active_ctx = self._get_active_code_context(project_id, user_query)
                if active_ctx:
                    dynamic_injections.append(("critical", active_ctx))

        # ── Step B4: Proactive suggestions ───────────────────────────
        self._log_debug("🔄 Block B – Step 4/5: Proactive suggestions")
        if (
            self.valves.cleanup_suggestions_enabled
            and self.valves.cleanup_proactive_suggestions
            and is_code_session
        ):
            candidates = self._get_inactive_block_candidates(project_id)
            if candidates:
                last_sugg_idx = state.get("last_cleanup_suggestion_msg_idx", 0)
                if (
                    state["message_count"] - last_sugg_idx
                    >= self.valves.cleanup_suggestion_cooldown_messages
                ):
                    dynamic_injections.append(
                        (
                            "low",
                            f"[CodeAware] {len(candidates)} inactive block(s). "
                            f"Use `/status` or `/clean`.",
                        )
                    )
                    state["last_cleanup_suggestion_msg_idx"] = state["message_count"]
                    self._set_state(project_id, state)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._estimate_tokens(sys_msgs + history_msgs)
        if self.valves.context_window_tokens > 0:
            suggestion = await self._check_and_suggest_summarization(
                project_id, total_tokens, self.valves.context_window_tokens
            )
            if suggestion:
                dynamic_injections.append(("low", suggestion))

        cmd_suggestion = await self._suggest_commands(project_id, state)
        if cmd_suggestion:
            dynamic_injections.append(("low", cmd_suggestion))

        # ── Step B5: Assemble prelim_system ──────────────────────────
        self._log_debug("🔄 Block B – Step 5/5: Assemble prelim_system")
        budget = self.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected: List[str] = []
            used = 0
            static_tokens = (
                len(self.tokenizer.encode(static_block)) if static_block else 0
            )
            remaining_budget = max(0, budget - static_tokens)
            for prio, text in dynamic_injections:
                if not text:
                    continue
                tok = len(self.tokenizer.encode(text))
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

        self._log_debug("🔄 Block B: complete")
        return static_block, dynamic_injections, None, prelim_system

    async def _should_keep_full_code(self, user_question: str) -> bool:
        """
        Ask a lightweight LLM whether the full code should be kept in the user message.
        Returns True if the model responds 'full', False otherwise.
        """
        if not user_question.strip():
            return False

        prompt = (
            f"The user has provided a large block of code. Their question/message is:\n"
            f'"{user_question[:500]}"\n\n'
            "Should the full code be included in the final context, or is it sufficient "
            "to provide only an analysis summary and relevant code fragments?\n"
            'Answer with only one word: "full" or "summary".'
        )

        # Use the secondary task model (or main model as fallback)
        model = self.valves.secondary_task_model or self.valves.llm_model
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a concise classifier. Answer with only one word.",
            model_override=model,
            max_tokens=5,
            temperature=0.0,
            label="lean_context_check",
        )
        return response and response.strip().lower() == "full"

    async def _inlet_assemble_final_messages(
        self,
        messages: List[dict],
        project_id: str,
        static_block: str,  # ← v7 (PASO-21)
        dynamic_injections: List[Tuple[str, str]],  # ← v7 (PASO-21)
        prelim_system: str,
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: dict,
        __user__: Optional[dict],
        background_tasks: List[asyncio.Task],
        user_question: str,
        has_code_blocks: bool,
        slot_free: bool = True,
    ) -> List[dict]:
        """Apply CoT, final token budget, trimming, and insert system prompt."""
        self._log_debug(
            "Assembling final messages (CoT, trimming, system prompt injection)"
        )

        # Determine user intent for context reduction (only if slot is free, otherwise keep full code)
        if slot_free:
            self._user_intent_full_code = await self._should_keep_full_code(
                user_question
            )
        else:
            self._user_intent_full_code = (
                True  # keep full code to avoid degrading response
            )

        # ── 🧠 ENRICHMENT – CoT Step 1/3: Detect CoT level ──
        self._log_debug("🧠 ENRICHMENT – CoT Step 1/3: Detect CoT level")
        manual_cot_used = False
        cot_any_used = False
        cot_level = 2
        reasoning = None
        cot_question = ""

        if self.valves.enable_cot_on_demand or self.valves.auto_cot_enabled:
            if last_user_msg:
                user_content = last_user_msg.get("content", "")
                if self.valves.enable_cot_on_demand and user_content.strip().startswith(
                    "/think"
                ):
                    cot_question, level = await self._parse_cot_intent(user_content)
                    if cot_question:
                        manual_cot_used = True
                        cot_any_used = True
                        cot_level = level
                        if level == 1:
                            cot_prompt = "Please think step by step before answering. Show your reasoning, then provide the final answer."
                            dynamic_injections.append(("high", cot_prompt))
                elif not manual_cot_used and slot_free:
                    cot_level = await self._detect_cot_level(
                        user_content, is_code_session, state
                    )
                    self._log_debug(
                        f"🧠 ENRICHMENT – CoT Step 1/3: Detected level {cot_level}"
                    )
                    if cot_level > 0:
                        cot_any_used = True
        if not cot_any_used:
            self._log_debug("🧠 ENRICHMENT – CoT Step 1/3: No CoT needed")
        elif not slot_free:
            self._log_debug(
                "🧠 ENRICHMENT – CoT Step 1/3: CoT detection skipped (no free slot)"
            )

        # Wait for background tasks before heavy LLM calls
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
            background_tasks.clear()

        # If the main model is loaded, disable CoT entirely
        if cot_any_used and not slot_free:
            self._log_debug(
                "🧠 ENRICHMENT – CoT skipped because main model is still loaded (no free slot)"
            )
            cot_any_used = False

        # ── 🧠 ENRICHMENT – CoT Step 2/3: Generate reasoning ──
        if cot_any_used:
            self._log_debug("🧠 ENRICHMENT – CoT Step 2/3: Generate reasoning")
            if manual_cot_used:
                self._log_debug(
                    f"Generating manual CoT level {cot_level} with model "
                    f"{self.valves.cot_model_level2 if cot_level == 2 else self.valves.cot_model_level3}"
                )
            else:
                self._log_debug(
                    f"Generating auto CoT level {cot_level} with model "
                    f"{self.valves.cot_model_level2 if cot_level == 2 else self.valves.cot_model_level3}"
                )

            # TODO: Bear in mind here we pass the entire active context
            #       which can be big enough to overflow 3-9b models.
            #       This should be auto fixed once we implement
            #       more advanced techniques of context compression.
            #       This applies to CoT level 2 and 3.
            _model_ctx = self.valves.active_context_max_tokens or 28000
            _cot_context_limit = _model_ctx // 3
            if self.tokenizer:
                _prelim_tokens = len(self.tokenizer.encode(prelim_system))
                if _prelim_tokens > _cot_context_limit:
                    prelim_for_cot = self._truncate_text_to_tokens(
                        prelim_system, _cot_context_limit
                    )
                else:
                    prelim_for_cot = prelim_system
            else:
                prelim_for_cot = prelim_system[: _cot_context_limit * 4]

            # ── v7 Scientific CoT (Phase 5/6) ──
            if not manual_cot_used:
                question = user_question
                if cot_level == 2:
                    reasoning = await self._generate_cot_reasoning(
                        question, prelim_for_cot
                    )
                elif cot_level == 3:
                    # Scientific reasoning with structural validation
                    reasoning = await self._generate_scientific_reasoning_L3(
                        question,
                        prelim_for_cot,
                        project_id,  # needed for evidence gathering
                        label="scientific_cot",
                    )
            else:
                if cot_level == 2:
                    reasoning = await self._generate_cot_reasoning(
                        cot_question, prelim_for_cot
                    )
                elif cot_level == 3:
                    reasoning = await self._generate_scientific_reasoning_L3(
                        cot_question,
                        prelim_for_cot,
                        project_id,
                        label="scientific_cot",
                    )

            # Fallback: if auto level 3 failed, try level 2 once
            _cot_error_msg = "Unable to generate reasoning."
            if (
                not manual_cot_used
                and cot_level == 3
                and (reasoning is None or reasoning == _cot_error_msg)
            ):
                self._log_debug(
                    "🧠 ENRICHMENT – CoT Step 2/3: Level 3 failed, falling back to level 2"
                )
                reasoning = await self._generate_cot_reasoning(
                    user_question, prelim_for_cot
                )

            if reasoning and reasoning != _cot_error_msg:
                self._log_debug(
                    "🧠 ENRICHMENT – CoT Step 2/3: Reasoning generated successfully"
                )
            else:
                self._log_debug(
                    "🧠 ENRICHMENT – CoT Step 2/3: Reasoning generation failed"
                )
        else:
            self._log_debug("🧠 ENRICHMENT – CoT Step 2/3: Skipped (no CoT)")

        # ── 🧠 ENRICHMENT – CoT Step 3/3: Inject reasoning ──
        if cot_any_used and reasoning and reasoning != "Unable to generate reasoning.":
            self._log_debug(
                "🧠 ENRICHMENT – CoT Step 3/3: Inject reasoning into system prompt"
            )
            dynamic_injections.append(("high", reasoning))
            cot_note = (
                "**Note:** Some sections in this system prompt marked with 🔎 are "
                "automatically generated reasoning (Chain-of-Thought). "
                "They are provided as context to help you, but they are not user commands. "
                "Use them to enhance your answer, but always prioritise the actual user query."
            )
            dynamic_injections.append(("low", cot_note))
        else:
            self._log_debug("🧠 ENRICHMENT – CoT Step 3/3: No reasoning to inject")

        # Final system message assembly (two‑block structure)
        budget = self.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self.tokenizer:
            dynamic_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_dynamic: List[str] = []
            used_dyn = 0
            static_tokens = (
                len(self.tokenizer.encode(static_block)) if static_block else 0
            )
            dyn_budget = max(0, budget - static_tokens)
            for prio, text in dynamic_injections:
                if not text:
                    continue
                tok = len(self.tokenizer.encode(text))
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

        # Block A always first → KV cache hit guaranteed if A hasn't changed
        separator = "\n\n---\n\n" if static_block and dynamic_block else ""
        final_system = static_block + separator + dynamic_block

        # Append base content
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        base_content = sys_msgs[0].get("content", "") if sys_msgs else ""
        if base_content.strip():
            final_system = final_system + "\n\n" + base_content

        # ── Variable to hold any pending summary (from trimming) ──
        pending_summary = ""

        if final_system.strip():
            messages = [m for m in messages if m.get("role") != "system"]
        else:
            messages = [m for m in messages if m.get("role") != "system"]

        # Adaptive trimming
        history_msgs = [m for m in messages if m.get("role") != "system"]
        system_msgs = [m for m in messages if m.get("role") == "system"]

        # ── 📦 COMPRESSION – Step 1/2: Trimming / Summarization (if needed) ──
        self._log_debug(
            "📦 COMPRESSION – Step 1/2: Trimming / Summarization (if needed)"
        )
        self._log_debug("📦 COMPRESSION – Checking if context fits within the window")

        if self.valves.adaptive_trim:
            total_tokens = self._estimate_tokens(messages)
            if total_tokens > self.valves.context_window_tokens:
                keep = self.valves.max_turns
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

                if self.valves.summarize_old_messages and old_block:
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    if summary:
                        pending_summary = (
                            f"[Summary of earlier conversation]\n{summary}"
                        )
                        self._log_debug(
                            "📦 COMPRESSION – Context exceeds token budget, "
                            "older messages trimmed and summarized"
                        )
                    else:
                        self._log_debug(
                            "📦 COMPRESSION – Context exceeds token budget, "
                            "older messages trimmed (summarization failed or empty)"
                        )
                    history_msgs = kept_block
                else:
                    self._log_debug(
                        "📦 COMPRESSION – Context exceeds token budget, "
                        "older messages trimmed (summarization disabled)"
                    )
                    history_msgs = kept_block if old_block else history_msgs

                if self.valves.preserve_tool_calls:
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
                self._log_debug(
                    "📦 COMPRESSION – Context fits within token budget, "
                    "no trimming needed"
                )
        else:
            user_max = (
                __user__["valves"].max_turns
                if __user__ and hasattr(__user__, "valves")
                else None
            )
            eff_max = user_max if user_max is not None else self.valves.max_turns
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

                if self.valves.summarize_old_messages and old_block:
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    if summary:
                        pending_summary = (
                            f"[Summary of earlier conversation]\n{summary}"
                        )
                        self._log_debug(
                            "📦 COMPRESSION – Context exceeds max turns, "
                            "older messages trimmed and summarized"
                        )
                    history_msgs = kept_block
                else:
                    self._log_debug(
                        "📦 COMPRESSION – Context exceeds max turns, "
                        "older messages trimmed (summarization disabled)"
                    )
                    history_msgs = kept_block
            else:
                self._log_debug(
                    "📦 COMPRESSION – Context fits within max turns, "
                    "no trimming needed"
                )

        # ── 📦 COMPRESSION – Step 2/2: Lean user message ──
        self._log_debug(
            "📦 COMPRESSION – Step 2/2: Lean context (replace full code with relevant fragments)"
        )
        if has_code_blocks and last_user_msg:
            self._log_debug(
                "📦 COMPRESSION – Lean user message: full code kept (path-aware context active)"
            )
        else:
            self._log_debug(
                "📦 COMPRESSION – Lean user message not applied (no code blocks or no user message)"
            )

        # ── Concatenate pending summary to the final system message ──
        if pending_summary:
            final_system = (
                final_system + "\n\n" + pending_summary
                if final_system
                else pending_summary
            )
            pending_summary = ""

        if final_system.strip():
            messages.insert(0, {"role": "system", "content": final_system})

        messages = system_msgs + history_msgs

        # Ensure last message is user
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

        # ═══════════════════════════════════════════════════════════════
        # Token breakdown log (updated for two‑block structure)
        # ═══════════════════════════════════════════════════════════════
        if self.valves.debug and self.tokenizer and final_system.strip():
            static_tok = len(self.tokenizer.encode(static_block)) if static_block else 0
            dynamic_tok = (
                len(self.tokenizer.encode(dynamic_block)) if dynamic_block else 0
            )
            total_system_tok = len(self.tokenizer.encode(final_system))

            prefix_hash = self._last_static_prefix_hash.get(project_id, "N/A")
            self._log_debug("─" * 60)
            self._log_debug("TOKEN BREAKDOWN — system prompt")
            self._log_debug(f"  BLOCK A (static, cacheable):  ~{static_tok} tokens")
            self._log_debug(f"  BLOCK B (dynamic, per-query): ~{dynamic_tok} tokens")
            self._log_debug(
                f"  TOTAL system tokens:          ~{total_system_tok} tokens"
            )
            self._log_debug(f"  Prefix hash (Block A):        {prefix_hash}")
            self._log_debug(
                f"  → If hash matches previous:   KV cache HIT in llama.cpp"
            )
            self._log_debug(
                f"  → If hash changed:            KV cache MISS, full prefill"
            )
            self._log_debug("─" * 60)
        elif self.valves.debug:
            self._log_debug("No system prompt injected (token breakdown skipped).")

        return messages

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

        self._ensure_cleanup_task()
        project_id = self._get_project_id()

        # ─────────────────────────────────────────────────────────────────
        # 🚀 RESOURCE OPTIMISATION – Check if the main model is loaded
        # We will avoid loading any auxiliary model while the slot is occupied.
        # ─────────────────────────────────────────────────────────────────
        slot_free = await self._wait_for_empty_slot(retries=1, delay=0.5)
        if not slot_free:
            self._log_debug(
                "Main model slot occupied – auxiliary model calls will be skipped"
            )

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   1. Preprocess (project switch, cache load)
        #   2. Process pending secondary tasks (session summaries, etc.)
        #   4. Extract user info (last message, question, code blocks)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        messages = await self._inlet_preprocess(body, project_id)
        _inlet_timing("Step 1/9: Preprocess (project switch, cache load)", step_start)
        if not messages:
            return body

        step_start = time.monotonic()
        await self._process_pending_secondary_tasks(project_id)
        _inlet_timing("Step 2/9: Process pending secondary tasks", step_start)

        # NOTE: We no longer unload models at the start; the outlet handles cleanup.
        # Step 3 is now a no‑op for resource optimisation.
        step_start = time.monotonic()
        _inlet_timing(
            "Step 3/9: Unload models safely (free VRAM) – SKIPPED", step_start
        )

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   (continued) 4. Extract user info
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        (
            last_user_msg,
            user_query,
            user_question,
            is_explicit_command,
            has_code_blocks,
        ) = await self._inlet_extract_user_info(messages)
        _inlet_timing("Step 4/9: Extract user info", step_start)

        # ─────────────────────────────────────────────────────────────────
        # ⚡ COMMAND HANDLING (High value)
        #   5. Explicit commands (/forget, /status, /clean, /expand)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        handled, handled_messages = await self._inlet_handle_explicit_commands(
            messages, project_id, is_explicit_command, last_user_msg, __user__
        )
        _inlet_timing("Step 5/9: Handle explicit commands", step_start)
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ⚡ COMMAND HANDLING (High value)
        #   6. Natural language intents (forget, remember, obsolete)
        # ─────────────────────────────────────────────────────────────────
        step_start = time.monotonic()
        handled, handled_messages = await self._inlet_handle_natural_intents(
            messages,
            project_id,
            is_explicit_command,
            last_user_msg,
            slot_free=slot_free,
        )
        _inlet_timing("Step 6/9: Handle natural language intents", step_start)
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ── Silent Ingestion (Modo B: chunked paste) ────────────────────
        # v7 (PASO-22)
        if (
            self.valves.enable_silent_ingestion
            and last_user_msg is not None
            and not is_explicit_command
        ):
            if await self._is_code_only_message(user_query):
                self._log_section("SILENT INGESTION MODE")

                # Process code into SymbolGraph without invoking main LLM
                await self._update_active_code(last_user_msg, project_id)

                # Resolve cross‑references with previous chunks
                resolved = await self._resolve_dangling_edges(project_id)

                # Rebuild PathIndex with new symbols
                if self.valves.enable_path_analysis:
                    await self._rebuild_path_index(project_id)

                # Invalidate static block (new code → new Block A)
                self._invalidate_static_context_block(project_id, "new chunk ingested")

                # Statistics for the user
                state = self._get_state(project_id)
                n_blocks = len(state.get("active_blocks", {}))
                n_symbols = len(self._symbol_index.get_all_names(project_id))
                n_paths = len(self._path_index.get_all(project_id))
                cross_note = (
                    f", {resolved} cross-references resolved" if resolved > 0 else ""
                )

                confirmation = (
                    f"✓ **Code indexed**: {n_blocks} blocks · "
                    f"{n_symbols} symbols · {n_paths} paths{cross_note}\n"
                    f"Ready for queries."
                )

                # Replace the user message with the confirmation
                # (it never reaches the main LLM, no conversation context consumed)
                messages[-1]["content"] = confirmation
                body["messages"] = messages
                body["messages"].append({"role": "assistant", "content": confirmation})

                await self._save_state_if_dirty(project_id)
                self._log_section(
                    "SILENT INGESTION END", duration=time.monotonic() - inlet_start
                )
                return body

        # Per‑request tracking for graceful STOP cancellation
        background_tasks: list[asyncio.Task] = []
        token = _inlet_background_tasks.set(background_tasks)
        _inlet_aborted = True

        try:
            state = self._get_state(project_id)

            # ─────────────────────────────────────────────────────────────
            # 🔥 STATE MANAGEMENT + 🧠 ENRICHMENT (Critical)
            #   7. Prepare code session:
            #      - classify if session is about code
            #      - update active blocks (extract code, detect duplicates)
            #      - run immediate enrichment tasks (auto‑summaries)
            #      - evict blocks if max_active_blocks > 0
            # ─────────────────────────────────────────────────────────────
            step_start = time.monotonic()
            is_code_session, user_question = await self._inlet_prepare_code_session(
                messages, project_id, user_query
            )
            _inlet_timing(
                "Step 7/9: Prepare code session (classify, update blocks, enrich)",
                step_start,
            )

            # Wait for any background tasks launched by _update_active_code
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()

            # Ensure all LLM-related work from this step is complete
            await self._wait_for_llm_tasks()

            # ─────────────────────────────────────────────────────────────
            # 🧠 ENRICHMENT (Critical)
            #   8. Build system injections:
            #      - LTM retrieval
            #      - Active code context (full or lightweight)
            #      - Symbol analysis summary
            #      - Chain‑of‑Thought detection (level computed here)
            #      - Feedback context, cleanup suggestions, confidence
            #      - Parallel checks: contradictions, duplicate questions,
            #        response cache lookup
            #      - Now separates into static_block (Block A) and dynamic_injections (Block B)
            # ─────────────────────────────────────────────────────────────
            step_start = time.monotonic()
            static_block, dynamic_injections, cached_response, prelim_system = (
                await self._inlet_build_system_injections(
                    messages,
                    project_id,
                    user_query,
                    user_question,
                    is_code_session,
                    last_user_msg,
                    state,
                    slot_free=slot_free,  # <-- pass the flag
                )
            )
            _inlet_timing("Step 8/9: Build system injections", step_start)

            # 🚀 RESOURCE OPTIMISATION (High value)
            #    Return cached response immediately if found
            if isinstance(cached_response, dict):
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Returning cached response (no further processing)"
                )
                messages.append(
                    {"role": "assistant", "content": cached_response["response"]}
                )
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                _inlet_timing("total_inlet (end-to-end)", inlet_start)
                self._log_section(
                    "CONTEXT MANAGER - INLET END",
                    duration=time.monotonic() - inlet_start,
                )
                _inlet_aborted = False
                return body

            # ─────────────────────────────────────────────────────────────
            # 🧠 ENRICHMENT + 📦 COMPRESSION (Critical)
            #   9. Assemble final messages:
            #      - Apply Chain‑of‑Thought reasoning
            #      - Trim old history (adaptive or max_turns)
            #      - Summarise trimmed messages if enabled
            #      - Inject final system prompt respecting token budget
            #      - Display token breakdown (debug)
            #      - Now uses static_block + dynamic_injections for KV cache stability
            # ─────────────────────────────────────────────────────────────
            step_start = time.monotonic()
            messages = await self._inlet_assemble_final_messages(
                messages,
                project_id,
                static_block,  # ← v7 (PASO-21)
                dynamic_injections,  # ← v7 (PASO-21)
                prelim_system,
                last_user_msg,
                is_code_session,
                state,
                __user__,
                background_tasks,
                user_question,
                has_code_blocks,
                slot_free=slot_free,  # <-- pass the flag
            )
            _inlet_timing(
                "Step 9/9: Assemble final messages (CoT, trim, system prompt)",
                step_start,
            )

            body["messages"] = messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            _inlet_aborted = False

        finally:
            # 🔥 STATE MANAGEMENT (Medium)
            #   - Save state if dirty (debounced write)
            #   - Process remaining secondary tasks (session summaries, etc.)
            await self._save_state_if_dirty(project_id)
            lock = await self._get_project_lock(project_id)
            async with lock:
                await self._process_pending_secondary_tasks(project_id)

            # 🚀 RESOURCE OPTIMISATION (Critical)
            #   - Wait for any unfinished background tasks
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()

            if _inlet_aborted:
                for task in background_tasks:
                    if not task.done():
                        task.cancel()
            _inlet_background_tasks.reset(token)

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
            project_id = self._get_project_id()
            state = self._get_state(project_id)
            is_code_session = await self._classify_session(messages, project_id)
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
                            await self._outlet_intercept_expand(
                                last_msg.get("content", ""), project_id
                            )
                        )
                        if did_expand:
                            messages[-1]["content"] = modified_content
                            body["messages"] = messages
                            self._log_debug(
                                "outlet: /expand intercepted — history rewritten with real code"
                            )

                    # ── 🔥 STATE MANAGEMENT: update active code blocks & LTM ──
                    if last_msg.get("role") in ("user", "assistant"):
                        await self._wait_for_llm_tasks()
                        if is_code_session:
                            self._log_debug(
                                "🔥 STATE MANAGEMENT – Updating active code blocks and storing in LTM "
                                "(new code detected)"
                            )
                            await self._update_active_code(last_msg, project_id)
                            async with self._ltm_batch_lock:
                                self._pending_ltm_messages.append(last_msg)
                                if (
                                    self._ltm_batch_task is None
                                    or self._ltm_batch_task.done()
                                ):
                                    self._ltm_batch_task = asyncio.create_task(
                                        self._flush_ltm_batch(project_id)
                                    )
                        else:
                            if not self.valves.ltm_store_only_code_sessions:
                                self._log_debug(
                                    "🔥 STATE MANAGEMENT – Storing non‑code session message in LTM"
                                )
                                async with self._ltm_batch_lock:
                                    self._pending_ltm_messages.append(last_msg)
                                    if (
                                        self._ltm_batch_task is None
                                        or self._ltm_batch_task.done()
                                    ):
                                        self._ltm_batch_task = asyncio.create_task(
                                            self._flush_ltm_batch(project_id)
                                        )

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
                    context_hash = self._compute_context_hash(messages[:-1])
                    code_state_hash = self._compute_code_state_hash(project_id)
                    await self._store_response_in_cache(
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
                    self._background_task(
                        self._update_lod_thresholds_from_response(
                            project_id,
                            last_assistant["content"],
                        ),
                        name="lod_adaptive_feedback",
                    )

            # 🚀 RESOURCE OPTIMISATION – Speculative prefetch
            if self.valves.enable_speculative_prefetch and is_code_session:
                last_activated = getattr(self, "_last_activation_scores", {}).get(
                    project_id, {}
                )
                if last_activated:
                    self._background_task(
                        self._speculative_prefetch(project_id, last_activated),
                        name="speculative_prefetch",
                    )

            # 🚀 RESOURCE OPTIMISATION: purge expired memories periodically
            if self._purge_task is None or self._purge_task.done():
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Purging expired memories "
                    "(reclaiming disk space and keeping LTM fresh)"
                )
                self._purge_task = asyncio.create_task(self._purge_expired_memories())

            # 🚀 RESOURCE OPTIMISATION: DB checkpoints every 100 writes
            self._write_counter += 1

            # ── v7 (PASO-20): RAPTOR periodic rebuild ──
            if (
                self.valves.enable_raptor
                and self._write_counter % self.valves.raptor_rebuild_interval == 0
            ):
                self._log_debug("RAPTOR: triggering background index rebuild")
                self._background_task(
                    self._rebuild_raptor_index(project_id),
                    name="raptor_rebuild",
                )

            if self._write_counter % 100 == 0:
                self._log_debug(
                    "🚀 RESOURCE OPTIMISATION – Running DB checkpoints "
                    "(to ensure data durability and prevent WAL buildup)"
                )
                self._purge_task = asyncio.create_task(self._run_db_checkpoints())

            # 🚀 RESOURCE OPTIMISATION – Save KV slot if static block changed
            if self.valves.enable_slot_persistence:
                self._background_task(
                    self._slot_save_if_needed(project_id),
                    name="slot_save",
                )

            # 🔥 STATE MANAGEMENT – Persistir edges del SymbolGraph
            if self.valves.enable_edge_persistence:
                self._background_task(
                    self._save_symbol_edges_to_db(project_id),
                    name="save_symbol_edges",
                )

            # 🔥 STATE MANAGEMENT: persist conversation state if dirty
            self._log_debug(
                "🔥 STATE MANAGEMENT – Saving conversation state "
                "(to preserve context across restarts)"
            )
            await self._save_state_if_dirty(project_id)

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

    # --------------------------------------------------------------------------
    # DB checkpoints
    # --------------------------------------------------------------------------
    async def _run_db_checkpoints(self):
        try:
            await anyio.to_thread.run_sync(
                lambda: self._db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            )
            self._log_debug("SQLite WAL checkpoint completed")
        except Exception as e:
            self._log_debug(f"SQLite checkpoint error: {e}")
        try:
            if self.chroma_client:
                await anyio.to_thread.run_sync(lambda: self.chroma_client.persist())
            self._log_debug("ChromaDB persist/checkpoint completed")
        except Exception as e:
            self._log_debug(f"ChromaDB checkpoint error: {e}")

    def _compute_context_hash(self, messages: List[dict]) -> str:
        if not self.valves.response_cache_include_context_hash:
            return ""
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        context_str = "\n".join([m.get("content", "") for m in sys_msgs])
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

    def _compute_code_state_hash(self, project_id: str) -> str:
        if self._cached_code_state_hash is not None:
            return self._cached_code_state_hash
        state = self._get_state(project_id)
        h = self._compute_code_state_hash_from_state(state)
        self._cached_code_state_hash = h
        return h

    def _compute_code_state_hash_from_state(self, state: dict) -> str:
        if not state or not state["active_blocks"]:
            return ""
        sorted_hashes = sorted(
            h for h, b in state["active_blocks"].items() if not b.obsolete
        )
        return hashlib.md5("|".join(sorted_hashes).encode()).hexdigest()[:16]

    # --------------------------------------------------------------------------
    # Shutdown and cleanup
    # --------------------------------------------------------------------------
    def shutdown(self):
        # Flush pending LTM batch synchronously with a short timeout
        if self._pending_ltm_messages:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.create_task(
                        self._flush_ltm_batch(self._get_project_id())
                    )
                    # Give it up to 2 seconds to complete
                    loop.run_until_complete(asyncio.wait_for(task, timeout=2.0))
            except Exception:
                pass

        # Cancel response cache cleanup task if it exists
        if (
            hasattr(self, "_response_cache_cleanup_task")
            and self._response_cache_cleanup_task is not None
        ):
            self._response_cache_cleanup_task.cancel()

        # Cancel the database worker cleanly
        self._db_worker_task.cancel()

        # Clear in‑memory structures
        self._symbol_index.clear()
        self._cached_lightweight_context.clear()
        self._project_locks.clear()

        # Shut down thread pools
        self._db_executor.shutdown(wait=True)
        self._chroma_executor.shutdown(wait=True)

    # --------------------------------------------------------------------------
    # Miscellaneous helpers
    # --------------------------------------------------------------------------
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        # ── Phase 6 (PASO-36): try AST-based comparison for Python ──
        if self.valves.enable_ast_deduplication and len(code1) > 30 and len(code2) > 30:
            ast_sim = self._ast_similarity(code1, code2)
            if ast_sim is not None:
                return ast_sim  # reliable structural comparison

        # ── Fallback: text similarity (non-Python or parse error) ──
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))
        return fuzz.token_sort_ratio(code1, code2) / 100.0

    def _ast_similarity(self, code1: str, code2: str) -> Optional[float]:
        """
        Compute structural similarity between two Python code blocks via AST.

        Process:
        1. Parse both codes as AST.
        2. Strip docstrings from function/class bodies.
        3. Compare AST dumps (exact structural match).
        4. If not exact, compute Jaccard similarity on AST node type distributions.

        Returns:
            1.0 if structurally identical (same logic, different docstrings/comments).
            0.0-1.0 for partial structural similarity.
            None if either code is not valid Python (caller should use text similarity).
        """
        # Quick heuristic: only attempt AST if code looks like Python
        if not (
            re.search(r"\bdef\s+\w+\s*\(", code1) or re.search(r"\bclass\s+\w+", code1)
        ):
            return None  # not Python-like → use text similarity

        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            return None  # parse error → fall back to text similarity

        # ── Step 1: strip docstrings ──────────────────────────────
        def _strip_docstrings(tree: ast.AST) -> ast.AST:
            """Remove docstrings from function/class/module bodies."""
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Module,
                    ),
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

        # ── Step 2: exact AST comparison ──────────────────────────
        dump1 = ast.dump(clean1)
        dump2 = ast.dump(clean2)

        if dump1 == dump2:
            return 1.0  # structurally identical

        # ── Step 3: Jaccard on node type distribution ─────────────
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

    def _has_conflicting_proposed_changes(
        self, state: Dict, new_block: CodeBlock
    ) -> bool:
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
                or self._calculate_code_similarity(existing.content, new_block.content)
                > 0.8
            ):
                return True
        return False

    def _remove_duplicate_blocks(self, state: Dict, project_id: str):
        if not self.valves.auto_remove_duplicate_blocks:
            return
        blocks = list(state["active_blocks"].values())
        to_remove = set()
        for i, block in enumerate(blocks):
            if block.hash in to_remove or block.pinned or block.obsolete:
                continue
            for j, other in enumerate(blocks[i + 1 :], start=i + 1):
                if other.hash in to_remove or other.pinned or other.obsolete:
                    continue
                sim = self._calculate_code_similarity(block.content, other.content)
                if sim >= self.valves.code_similarity_threshold:
                    age_diff = abs(block.timestamp - other.timestamp) / 3600
                    if age_diff > self.valves.max_duplicate_age_hours:
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
                self._symbol_index.remove_all_for_block(
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
            self._invalidate_lightweight_cache(project_id)

    # --------------------------------------------------------------------------
    # Oversized code block handling
    # --------------------------------------------------------------------------
    def _estimate_code_tokens(self, code: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(code))
        return len(code) // 4

    async def _handle_oversized_code_block(self, code: str, language: str) -> str:
        max_tokens = self.valves.max_code_block_tokens
        if max_tokens <= 0:
            return code
        estimated = self._estimate_code_tokens(code)
        if estimated <= max_tokens:
            return code
        action = self.valves.code_block_overflow_action.lower()
        if action == "truncate":
            lines = code.splitlines()
            head = self.valves.code_block_truncate_keep_head
            tail = self.valves.code_block_truncate_keep_tail
            if len(lines) <= head + tail:
                return code
            return "\n".join(
                lines[:head]
                + [f"... [{len(lines) - head - tail} lines truncated] ..."]
                + lines[-tail:]
            )
        elif action == "summarize":
            model = (
                self.valves.code_block_summary_model
                or self.valves.llm_model
                or self.valves.summarization_model
            )
            signatures = []
            for match in re.finditer(
                r"^\s*(def|class|function|fn|func|async def)\s+(\w+)[^(]*\([^)]*\)",
                code,
                re.MULTILINE | re.IGNORECASE,
            ):
                signatures.append(match.group(0).strip())
            header = ""
            if signatures:
                header = (
                    f"Signatures found ({len(signatures)}):\n"
                    + "\n".join(signatures[:50])
                    + "\n\n"
                )
            t0 = time.monotonic()
            summary = await self._call_llm(
                prompt=f"Summarize the following {language} code block.\n{header}First part of code:\n```{language}\n{code[:8000]}\n```",
                system_prompt="You are a code summarization assistant.",
                model_override=model,
                max_tokens=self.valves.oversized_summary_max_tokens,
                temperature=0.2,
            )
            dur = time.monotonic() - t0
            self._log_timing("oversized_block_summarize", dur, dur)
            return (
                f"[Automatic summary of a {estimated} token code block]\n{summary}"
                if summary
                else f"[Code block too large, could not summarize] Original size: {estimated} tokens."
            )
        elif action == "warn":
            return self.valves.code_block_warn_message
        return code

    # --------------------------------------------------------------------------
    # Diff application
    # --------------------------------------------------------------------------
    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        if not self.valves.enable_diff_application:
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

    def _apply_change_with_diff(
        self, base_block: CodeBlock, proposed_block: CodeBlock
    ) -> bool:
        if proposed_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        if not (
            "@@" in proposed_block.content
            and ("-" in proposed_block.content or "+" in proposed_block.content)
        ):
            return False
        new_code = self._apply_unified_diff(base_block.content, proposed_block.content)
        if new_code and new_code != base_block.content:
            self._symbol_index.remove_all_for_block(
                base_block.hash, base_block.symbols, self.valves.project_id
            )
            base_block.content = new_code
            base_block.hash = hashlib.md5(new_code.encode()).hexdigest()[:16]
            base_block.symbols = SignatureExtractor._extract_generic(
                new_code, base_block.file_path
            )
            for sym in base_block.symbols:
                sym.parent_block_hash = base_block.hash
                self._symbol_index.add(sym, base_block.hash, self.valves.project_id)
            if self.tokenizer:
                base_block._cached_token_count = len(self.tokenizer.encode(new_code))
            else:
                base_block._cached_token_count = len(new_code) // 4
            base_block.timestamp = time.time()
            base_block.is_active = True
            base_block.potentially_affected = False
            base_block.importance_score = min(base_block.importance_score + 2.0, 10.0)
            self._invalidate_lightweight_cache(self.valves.project_id)
            return True
        return False

    # --------------------------------------------------------------------------
    # Change summary generation
    # --------------------------------------------------------------------------
    async def _generate_change_summary(
        self, block_hash: str, prev_content: str, new_content: str
    ):
        """Enqueue a change summary task (or run immediately if deferral disabled)."""
        if self.valves.defer_secondary_tasks:
            task = SecondaryTask(
                task_type="change_summary",
                params={
                    "block_hash": block_hash,
                    "prev_content": prev_content,
                    "new_content": new_content,
                },
            )
            state = self._get_state(self.valves.project_id)
            if state is not None:
                state.setdefault("pending_secondary_tasks", []).append(task.dict())
                self._set_state(self.valves.project_id, state)
            return

        model = self.valves.secondary_task_model
        prompt = (
            f"Summarise the code change in ONE short sentence (max 15 words).\n\n"
            f"Previous:\n```\n{prev_content[:1000]}\n```\n\n"
            f"New:\n```\n{new_content[:1000]}\n```\n\n"
            f"Change summary:"
        )
        summary = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code change summariser. Output only one short sentence.",
            model_override=model,
            max_tokens=40,
            temperature=0.1,
        )
        if summary:
            now = time.time()
            # LRU in memory
            self._block_change_summaries[block_hash] = (summary.strip(), now)
            if len(self._block_change_summaries) > self._MAX_CHANGE_SUMMARIES:
                self._block_change_summaries.popitem(last=False)

            # Persist to SQLite
            def _write():
                self._db_conn.execute(
                    "INSERT OR REPLACE INTO block_change_summaries (block_hash, summary, created_at) VALUES (?, ?, ?)",
                    (block_hash, summary.strip(), now),
                )
                # Enforce max entries
                self._db_conn.execute(
                    "DELETE FROM block_change_summaries WHERE block_hash NOT IN (SELECT block_hash FROM block_change_summaries ORDER BY created_at DESC LIMIT ?)",
                    (self._MAX_CHANGE_SUMMARIES,),
                )
                self._db_conn.commit()

            await self._db_write_queue.put((_write, (), {}))

    # --------------------------------------------------------------------------
    # Deferred secondary task execution
    # --------------------------------------------------------------------------

    async def _process_pending_secondary_tasks(self, project_id: str):
        """Execute all pending secondary tasks at the start of an inlet (MUST be called under project lock)."""
        state = self._get_state(project_id)
        if not state or not state.get("pending_secondary_tasks"):
            return

        tasks = state["pending_secondary_tasks"]
        self._log_debug(f"Processing {len(tasks)} pending secondary task(s)...")

        remaining = []
        for task_dict in tasks:
            task = SecondaryTask(**task_dict)
            success = await self._execute_secondary_task(task, project_id)
            if not success:
                task.retries += 1
                if task.retries < self.valves.secondary_task_max_retries:
                    remaining.append(task.dict())
                else:
                    self._log_debug(
                        f"Dropping secondary task {task.task_type} after {task.retries} retries"
                    )

        state["pending_secondary_tasks"] = remaining
        self._set_state(project_id, state)

    async def _execute_secondary_task(
        self, task: SecondaryTask, project_id: str
    ) -> bool:
        """Dispatch a secondary task to the appropriate handler (dependency_refresh removed)."""
        model = self.valves.secondary_task_model
        sem = self._secondary_llm_semaphore
        try:
            if task.task_type == "change_summary":
                return await self._run_change_summary_task(task.params, model, sem)
            elif task.task_type == "missing_summaries":
                return await self._run_missing_summaries_task(task.params, model, sem)
            elif task.task_type == "inactive_code_summary":
                return await self._run_inactive_code_summary_task(
                    task.params, model, sem
                )
            elif task.task_type == "session_summary":
                return await self._run_session_summary_task(task.params, model, sem)
            else:
                return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log_debug(f"Secondary task {task.task_type} failed: {e}")
            return False
