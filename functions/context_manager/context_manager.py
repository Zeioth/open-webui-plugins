"""
title: Code-Aware Context Manager with LTM & Summarization (v5.29.0)
description: Full-featured context manager for coding assistants. Persists state per project, tracks line ranges, applies diffs, compresses LTM, scores importance, learns from responses, summarizes inactive code, supports manual markers, natural language forget/remember commands, feedback tracking, hierarchical memory, LRU cache, optional reranking, dependency detection (AST for Python + regex for other languages), handling of oversized blocks, smart context selection, hierarchical compression, duplicate removal, frequency prioritization, selective summarization, iterative commands, consecutive message deduplication, contradiction detection, chain-of-thought reasoning, assumption extraction, obsolete marking, proactive suggestions, duplicate question detection, command suggestions, semantic response caching, raw file priority boost, LTM retrieval token limit, and lightweight signature-based context with call graphs and summaries for massive code injections.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.3.2
license: GPL3
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb, rapidfuzz, tree-sitter-language-pack>=1.5.0
"""

import os
import time
import re
import anyio
import hashlib
import sqlite3
import ast
from collections import OrderedDict, defaultdict, Counter
import json
import asyncio
import difflib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from enum import Enum
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional dependencies
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
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from rapidfuzz import fuzz

    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

from loguru import logger

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# ---------------------------------------------------------------------------
# Tree-sitter for fast signature extraction (fallback if not installed)
# ---------------------------------------------------------------------------
try:
    from tree_sitter_language_pack import (
        get_language,
        get_parser,
        detect_language_from_extension,
        process,
        ProcessConfig,
    )

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

# ---------------------------------------------------------------------------
# Path for shared resources
# ---------------------------------------------------------------------------
import sys

if "/app/backend/data/custom_lib" not in sys.path:
    sys.path.append("/app/backend/data/custom_lib")

try:
    from shared_resources import (
        get_embedder as _shared_get_embedder,
        get_chroma_client as _shared_get_chroma_client,
        AsyncLRUCache as _AsyncLRUCache,
        get_http_session as _shared_get_http_session,
    )

    _SHARED_RESOURCES_AVAILABLE = True
except ImportError:
    _SHARED_RESOURCES_AVAILABLE = False


class ContentType(str, Enum):
    BASE_CODE = "base_code"
    PROPOSED_CHANGE = "proposed_change"
    COMMITTED_CHANGE = "committed_change"
    GENERAL = "general"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class CodeSymbol(BaseModel):
    """Extracted signature of a code entity (function, class, method)."""

    name: str
    kind: str  # "function", "class", "method"
    signature: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    parent_block_hash: str = ""
    language: str = "unknown"
    calls: List[str] = Field(
        default_factory=list
    )  # names of functions called inside this entity
    summary: str = ""  # one-line description of what this entity does


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
    dependencies: List[str] = Field(default_factory=list)
    potentially_affected: bool = False
    pinned: bool = False
    affected_timestamp: float = 0.0
    obsolete: bool = False
    ast_imports: List[str] = Field(default_factory=list)
    ast_calls: List[str] = Field(default_factory=list)
    is_raw: bool = False

    # Lightweight signature index and cached token count
    symbols: List[CodeSymbol] = Field(default_factory=list)
    _cached_token_count: int = 0  # not serialized

    def __init__(self, **data):
        super().__init__(**data)
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
        self._update_importance()
        # _cached_token_count will be set externally after creation

    def _update_importance(self):
        base_score = {
            ContentType.BASE_CODE: 8.0,
            ContentType.ERROR: 7.0,
            ContentType.COMMITTED_CHANGE: 6.0,
            ContentType.PROPOSED_CHANGE: 5.0,
            ContentType.TOOL_CALL: 3.0,
            ContentType.GENERAL: 2.0,
        }.get(self.content_type, 2.0)

        keyword_boost = 0.0
        if re.search(
            r"\b(fix|bug|security|critical|important|todo)\b",
            self.content,
            re.IGNORECASE,
        ):
            keyword_boost = 2.0

        if self.generated_by_assistant:
            base_score *= 0.8

        mention_boost = min(self.mention_count / 5, 3.0)
        age_hours = (time.time() - self.last_mentioned) / 3600
        recency_factor = 0.5**age_hours
        penalty = 0.7 if self.potentially_affected else 1.0
        if self.obsolete:
            penalty = 0.1
            self.is_active = False
        self.importance_score = (
            (base_score + keyword_boost) * mention_boost * recency_factor * penalty
        )


# ---------------------------------------------------------------------------
# Fallback tree-sitter queries for signature extraction
# Used only when tree_sitter_language_pack.process() is unavailable (older versions)
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

# ---------------------------------------------------------------------------
# Fallback call-graph queries (who calls whom)
# ---------------------------------------------------------------------------
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
    """Reentrant asyncio lock that allows the same task to acquire it multiple times."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count = 0

    async def acquire(self):
        task = asyncio.current_task()
        if self._owner is task:
            self._count += 1
            return
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
# In-memory inverted index: symbol name -> set of block hashes
# ---------------------------------------------------------------------------
class SymbolIndex:
    """Fast O(1) lookup of which blocks contain a given symbol name.
    Evicts the least frequent entries to prevent unbounded growth."""

    MAX_ENTRIES = 10_000

    def __init__(self):
        self._name_to_blocks: Dict[Tuple[str, str], Set[str]] = defaultdict(
            set
        )  # (project_id, name) -> blocks
        self._stats: Counter = Counter()

    def _evict_if_needed(self):
        while len(self._name_to_blocks) > self.MAX_ENTRIES:
            least_common = self._stats.most_common()[-1][0]
            del self._name_to_blocks[least_common]
            del self._stats[least_common]

    def add(self, symbol: CodeSymbol, block_hash: str, project_id: str):
        key = (project_id, symbol.name)
        self._name_to_blocks[key].add(block_hash)
        self._stats[key] += 1
        self._evict_if_needed()

    def remove(self, symbol: CodeSymbol, block_hash: str, project_id: str):
        key = (project_id, symbol.name)
        s = self._name_to_blocks.get(key)
        if s:
            s.discard(block_hash)
            if not s:
                del self._name_to_blocks[key]
                del self._stats[key]

    def remove_all_for_block(
        self, block_hash: str, symbols: List[CodeSymbol], project_id: str
    ):
        for sym in symbols:
            self.remove(sym, block_hash, project_id)

    def find_blocks(self, name: str, project_id: str) -> Set[str]:
        return self._name_to_blocks.get((project_id, name), set())

    def get_all_names(self, project_id: str) -> Set[str]:
        return {key[1] for key in self._name_to_blocks if key[0] == project_id}

    def clear_project(self, project_id: str):
        keys_to_remove = [key for key in self._name_to_blocks if key[0] == project_id]
        for key in keys_to_remove:
            del self._name_to_blocks[key]
            del self._stats[key]

    def clear(self):
        self._name_to_blocks.clear()
        self._stats.clear()


# ---------------------------------------------------------------------------
# Signature Extractor: multi-level fallback for extracting code symbols
# Level 1: tree_sitter_language_pack.process()   (305 languages, fastest)
# Level 2: tree-sitter queries (manual, ~7 langs)
# Level 3: regex (generic, any text)
# Also extracts call graphs (who-calls-who) and docstrings.
# ---------------------------------------------------------------------------
class SignatureExtractor:
    MAX_PARSE_SIZE_BYTES = 5_000_000  # 5 MB, covers most source files

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
    async def extract_async(
        code: str, file_path: Optional[str] = None
    ) -> List[CodeSymbol]:
        """Unified extraction: signatures + calls in one tree-sitter pass."""
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

        # ── Parse ONCE ──────────────────────────────────────────────────────
        try:
            parser = get_parser(lang)
            loop = asyncio.get_event_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(None, parser.parse, code.encode()), timeout=5.0
            )
        except (asyncio.TimeoutError, Exception):
            syms = SignatureExtractor._extract_generic(code, file_path)
            call_map = SignatureExtractor._extract_calls_generic(code)
            for sym in syms:
                sym.calls = call_map.get(sym.name, [])
            return syms

        # ── Ambas consultas sobre el mismo árbol ────────────────────────────
        syms = SignatureExtractor._extract_symbols_from_tree(
            tree, lang, code, file_path
        )
        call_map = SignatureExtractor._extract_calls_from_tree(tree, lang, code)
        del tree  # liberar RAM inmediatamente

        for sym in syms:
            sym.calls = call_map.get(sym.name, [])

        # Auto-extract docstrings for Python
        if lang == "python" or (file_path and file_path.endswith(".py")):
            SignatureExtractor._extract_docstrings_python(code, syms)

        return syms

    @staticmethod
    def _extract_symbols_from_tree(
        tree, lang: str, code: str, file_path: Optional[str]
    ) -> List[CodeSymbol]:
        """Level-2 extraction: uses an already built tree (no re-parse)."""
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
        """Extract calls (caller->callee) from an already built tree."""
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
        """Extract docstrings from Python functions/classes and assign to symbols."""
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
        """Use Python AST to extract caller->callee relationships."""
        call_map: Dict[str, Set[str]] = defaultdict(set)
        try:
            tree = ast.parse(code)
            # Track current function scope via a simple stack
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
        """Generic regex-based call extraction for unsupported languages."""
        call_map: Dict[str, Set[str]] = defaultdict(set)
        func_pattern = (
            r"^\s*(?:def|function|fn|func)\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\S+)?\s*:?"
        )
        for match in re.finditer(func_pattern, code, re.MULTILINE | re.IGNORECASE):
            func_name = match.group(1)
            start = match.end()
            rest = code[start:]
            next_match = re.search(
                r"^\s*(?:def|function|class|fn|func|export)\s+", rest, re.MULTILINE
            )
            if next_match:
                body = rest[: next_match.start()]
            else:
                body = rest
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
    async def _extract_calls_async(code: str, lang: str) -> Dict[str, List[str]]:
        """Return a dict mapping function name -> list of called function names."""
        if not HAS_TREE_SITTER:
            if lang == "python":
                return SignatureExtractor._extract_calls_fallback_python(code)
            return {}
        query_str = FALLBACK_CALL_QUERIES.get(lang)
        if not query_str:
            return SignatureExtractor._extract_calls_generic(code)
        try:
            parser = get_parser(lang)
            loop = asyncio.get_event_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(None, parser.parse, code.encode()), timeout=5.0
            )
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
                    # Extract callee name, handling attribute/member expressions
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

                    # Find the enclosing function/method
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
            del tree
            return {k: list(v) for k, v in call_map.items()}
        except Exception:
            return {}

    @staticmethod
    async def _extract_with_queries_async(
        code: str, lang: str, file_path: Optional[str]
    ) -> List[CodeSymbol]:
        """Extract symbols using manual tree-sitter queries (Level 2)."""
        query_str = FALLBACK_LANGUAGE_QUERIES.get(lang)
        if not query_str:
            return SignatureExtractor._extract_generic(code, file_path)
        try:
            parser = get_parser(lang)
            loop = asyncio.get_event_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(None, parser.parse, code.encode()), timeout=5.0
            )
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
                sig = ""
                if parent:
                    sig = parent.text.decode("utf-8").split("\n")[0].strip()[:200]
                else:
                    sig = node.text.decode("utf-8")
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
            del tree
            return symbols
        except asyncio.TimeoutError:
            return SignatureExtractor._extract_generic(code, file_path)
        except Exception:
            return SignatureExtractor._extract_generic(code, file_path)

    @staticmethod
    def _extract_generic(
        code: str, file_path: Optional[str] = None
    ) -> List[CodeSymbol]:
        """Generic regex fallback for unsupported languages (Level 3)."""
        symbols = []
        for match in re.finditer(
            r"^\s*(def|function|class|fn|func)\s+(\w+)",
            code,
            re.MULTILINE | re.IGNORECASE,
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


class AppliedChangeFeedback(BaseModel):
    change_hash: str
    change_description: str
    file_path: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


class Filter:

    class Valves(BaseModel):
        priority: int = Field(default=0)
        max_turns: int = Field(default=20)
        debug: bool = Field(default=True)
        state_db_path: str = Field(default="/app/backend/data/conversation_state.db")
        track_line_numbers: bool = Field(default=True)
        adaptive_trim: bool = Field(default=True)
        context_window_tokens: int = Field(default=1000000)
        use_tiktoken: bool = Field(default=True)

        long_term_memory_dir: str = Field(default="/app/backend/data/long_term_memory")
        long_term_memory_expiration_days: int = Field(default=30)
        long_term_memory_top_k: int = Field(default=10)
        long_term_memory_similarity_threshold: float = Field(default=0.65)
        ltm_time_decay_hours: float = Field(default=24.0)
        enable_reranking: bool = Field(default=False)
        reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker_top_k: int = Field(default=5)

        raw_file_priority_boost: float = Field(default=2.0)
        ltm_retrieval_max_tokens: int = Field(default=0)

        smart_context_selection: bool = Field(default=False)
        smart_context_top_k: int = Field(default=15)
        smart_context_min_tokens: int = Field(default=1024)
        smart_context_include_last_user: bool = Field(default=True)

        hierarchical_compression_enabled: bool = Field(default=False)
        hierarchical_compression_interval_messages: int = Field(default=100)
        hierarchical_summary_model: str = Field(default="ollama/llama3.2:3b")
        hierarchical_summary_max_tokens: int = Field(default=800)

        auto_remove_duplicate_blocks: bool = Field(default=True)
        max_duplicate_age_hours: float = Field(default=6.0)
        frequency_weight_factor: float = Field(default=0.3)
        min_mentions_for_boost: int = Field(default=3)
        frequency_decay_hours: float = Field(default=12.0)

        enable_confidence_scoring: bool = Field(default=True)
        confidence_prompt: str = Field(
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]' where XX is your estimated confidence (0-100) in the correctness and completeness of your answer, based on the available context. If you lack information, give lower confidence and suggest what context would help."
        )
        enable_cot_on_demand: bool = Field(default=True)
        auto_cot_enabled: bool = Field(default=False)  # Changed
        auto_cot_min_chars: int = Field(default=200)
        enable_code_review_mode: bool = Field(default=True)
        cot_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        cot_max_tokens: int = Field(default=1000)
        enable_assumption_extraction: bool = Field(default=False)  # Changed
        assumption_extraction_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        enable_contradiction_detection: bool = Field(default=False)  # Changed
        contradiction_detection_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        contradiction_inject_warning: bool = Field(default=True)
        proactive_context_warning_threshold: float = Field(default=0.85)
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Context Warning**: The conversation is using more than {percent}% of the available context window ({used_tokens}/{max_tokens} tokens). Consider using `/forget` to remove irrelevant parts, `/remember` to pin important context, or ask me to summarize older parts."
        )
        enable_facts: bool = Field(default=True)
        fact_max_age_days: int = Field(default=90)
        inject_facts_in_context: bool = Field(default=True)
        fact_importance_boost: float = Field(default=1.5)
        fact_command_prefix: str = Field(default="/fact")
        enable_auto_fact_detection: bool = Field(default=False)

        enable_iterative_mode: bool = Field(default=True)
        iterative_auto_continue: bool = Field(default=False)
        iterative_max_steps: int = Field(default=10)
        iterative_diff_format: str = Field(default="unified")
        iterative_planning_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        iterative_execution_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        iterative_resume_command: str = Field(default="/iterate resume")
        natural_language_iterate: bool = Field(default=True)

        iterative_state_ttl_hours: float = Field(
            default=2.0,
            description="Maximum hours an iterative state persists before being automatically cleared.",
        )
        similar_message_handling: str = Field(default="replace")
        similar_message_threshold: float = Field(default=0.92)  # Raised
        similar_message_check_code_only: bool = Field(default=True)

        enable_obsolete_marking: bool = Field(default=True)

        proactive_summary_threshold: float = Field(default=0.75)
        proactive_summary_growth_window: int = Field(default=3)

        duplicate_question_threshold: float = Field(default=0.92)
        duplicate_question_lookback: int = Field(default=20)

        enable_command_suggestions: bool = Field(default=True)
        command_suggestion_cooldown_minutes: int = Field(default=10)

        enable_response_cache: bool = Field(default=True)
        response_cache_similarity_threshold: float = Field(default=0.92)
        response_cache_ttl_hours: float = Field(default=24.0)
        response_cache_max_entries: int = Field(default=100)
        response_cache_include_context_hash: bool = Field(default=True)

        selective_summarization: bool = Field(default=True)
        error_preserve_verbatim: bool = Field(default=True)
        error_max_age_hours: float = Field(default=48.0)
        code_summary_level: str = Field(default="balanced")
        general_summary_max_tokens: int = Field(default=200)
        tool_call_preserve: bool = Field(default=True)
        code_always_keep_signature: bool = Field(default=True)
        summary_fallback_model: str = Field(default="ollama/llama3.2:3b")
        summary_include_metadata: bool = Field(default=True)

        summarize_old_messages: bool = Field(default=True)
        summarization_model: str = Field(default="ollama/llama3.2:3b")
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
        )
        openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", "dummy"))
        LLM_BASE_URL: str = Field(default="http://host.docker.internal:11434")
        LLM_API_TOKEN: str = Field(default="")

        enable_code_awareness: bool = Field(default=True)
        code_similarity_threshold: float = Field(default=0.85)
        max_base_code_blocks: int = Field(default=3)

        project_id: str = Field(default="default")

        max_proposed_changes: int = Field(default=5)
        max_committed_changes: int = Field(default=10)
        prioritize_recent_code: bool = Field(default=True)
        auto_detect_code_blocks: bool = Field(default=True)
        max_cached_projects: int = Field(default=10)
        track_file_paths: bool = Field(default=True)
        max_active_blocks: int = Field(default=50)
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b"
        )

        max_code_block_tokens: int = Field(default=20000)
        code_block_overflow_action: str = Field(default="summarize")
        code_block_summary_model: str = Field(default="ollama/llama3.2:3b")
        code_block_truncate_keep_head: int = Field(default=50)
        code_block_truncate_keep_tail: int = Field(default=50)
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]"
        )

        importance_mention_boost: float = Field(default=0.2)
        importance_recency_half_life_hours: float = Field(default=2.0)

        ltm_compress_after_messages: int = Field(default=50)
        ltm_summarization_trigger_similarity: float = Field(default=0.85)

        enable_diff_application: bool = Field(default=True)
        preserve_error_context: bool = Field(default=True)
        error_retention_turns: int = Field(default=15)
        block_expiration_hours: float = Field(default=24.0)
        proposed_change_retention_turns: int = Field(default=20)
        preserve_tool_calls: bool = Field(default=True)

        enable_feedback_tracking: bool = Field(default=True)
        feedback_history_limit: int = Field(default=10)
        inject_feedback_context: bool = Field(default=True)
        feedback_importance_penalty_for_failure: float = Field(default=2.0)

        code_block_pattern: str = Field(default="```(\\w*)\\n(.*?)```")
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@"
        )
        commit_pattern: str = Field(default="commit\\s+([a-f0-9]{7,40})")

        enable_dependency_tracking: bool = Field(default=False)
        dependency_extraction_model: str = Field(default="ollama/llama3.2:3b")
        dependency_refresh_on_update: bool = Field(default=True)
        affected_importance_penalty: float = Field(default=0.7)
        affected_decay_hours: float = Field(default=4.0)

        llm_request_timeout: int = Field(default=300)
        track_active_code_age: bool = Field(default=True)
        active_code_timeout_minutes: int = Field(default=30)

        summarize_inactive_code: bool = Field(default=True)
        inactive_code_summary_model: str = Field(default="ollama/llama3.2:3b")

        llm_model: str = Field(default="ollama/llama3.2:3b")

        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=False)  # Changed
        natural_language_forget_model: str = Field(
            default="ollama/Inference/Schematron:3B"
        )
        LLM_MAX_CONCURRENT_CALLS: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Max simultaneous LLM calls for summarization, filtering, etc.",
        )
        ltm_store_only_code_sessions: bool = Field(default=True)
        ltm_include_timestamps: bool = Field(default=True)

        LLM_CACHE_TTL: int = Field(default=300)
        LLM_CACHE_MAX_SIZE: int = Field(
            default=100,
            description="Maximum number of cached LLM responses in RAM.",
        )

        huge_injection_threshold_tokens: int = Field(
            default=100000,
            description="Threshold of active code tokens above which lightweight context (signatures only) is used. 0 = never.",
        )
        enable_call_graph_extraction: bool = Field(
            default=True,
            description="Extract call relationships (who calls whom) for code symbols.",
        )
        enable_auto_summaries: bool = Field(
            default=False,
            description="Automatically generate one-line summaries for code symbols using a small LLM.",
        )
        summary_code_max_chars: int = Field(
            default=8000,
            description="Maximum characters of code to include when summarizing code blocks.",
        )
        oversized_summary_max_tokens: int = Field(
            default=500, description="Max tokens for summarizing oversized code blocks."
        )

    class UserValves(BaseModel):
        max_turns: Optional[int] = Field(default=None)
        enable_code_awareness: Optional[bool] = Field(default=None)

    def __init__(self):
        self.valves = self.Valves()
        self.embedder = None
        self.chroma_client = None
        self.memory_collection = None
        self._response_cache_collection = None
        self.tokenizer = None
        self._db_conn = None
        self._cross_encoder = None
        self._init_state_db()
        self._conversation_state: OrderedDict = OrderedDict()
        self._state_factory = lambda: {
            "active_blocks": {},
            "recent_changes": [],
            "committed_changes": [],
            "message_count": 0,
            "feedback_history": [],
            "iterative_state": None,
            "facts": [],
            "last_compression_timestamp": 0,
            "last_suggestion_timestamp": 0,
            "response_cache": [],
            "has_any_calls": False,
        }
        self.code_pattern = re.compile(self.valves.code_block_pattern, re.DOTALL)
        self.diff_pattern = re.compile(self.valves.diff_pattern)
        self.commit_pattern = re.compile(self.valves.commit_pattern, re.IGNORECASE)

        if HAS_TIKTOKEN and self.valves.use_tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._log_debug("Tiktoken initialized")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")

        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._init_long_term_memory()
        else:
            logger.warning("Long‑term memory or code awareness disabled")

        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._load_reranker()

        if self.valves.enable_facts:
            self._log_debug("Fact storage enabled.")

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._project_locks: dict[str, ReentrantAsyncLock] = {}
        self._lock_lock = asyncio.Lock()
        self._llm_semaphore = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENT_CALLS)
        self._pending_llm: Dict[str, asyncio.Future] = {}
        self._pending_llm_lock = asyncio.Lock()
        self._llm_cache = self._init_llm_cache()
        self._llm_cache_ttl = self.valves.LLM_CACHE_TTL

        self._hierarchical_compress_in_progress: Dict[str, bool] = {}
        self._summarize_inactive_in_progress: Dict[str, bool] = {}
        self._hierarchical_compress_tasks: list[asyncio.Task] = []
        self._summarize_tasks: list[asyncio.Task] = []
        self._dependency_tasks: list[asyncio.Task] = []
        self._write_counter = 0
        self._response_cache_cleanup_task: Optional[asyncio.Task] = None
        self._session_classify_cache: Dict[str, Tuple[bool, float]] = {}
        self._session_classify_ttl: float = 1800.0  # Extended TTL

        # Lightweight context caches and project tracking
        self._symbol_index = SymbolIndex()
        self._cached_lightweight_context: Dict[str, str] = {}
        self._last_processed_message_idx: Dict[str, int] = {}
        self._last_project_id: str = ""
        self._response_cache_size: int = 0
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}

        print("[CodeAware] Filter loaded")

    # --------------------------------------------------------------------------
    #  Helper methods (logging, state, locks, cache, etc.)
    # --------------------------------------------------------------------------
    def _log_debug(self, msg: str):
        if self.valves.debug:
            print(f"[CodeAware] {msg}")
            logger.info(msg)

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
            self._log_debug(f"Tree‑sitter found {len(spans)} code spans")
        except Exception as e:
            self._log_debug(f"Tree‑sitter process failed: {e}")
            spans = []

        if len(self._code_spans_cache) >= 200:
            keys_to_evict = list(self._code_spans_cache.keys())[:50]
            for key in keys_to_evict:
                del self._code_spans_cache[key]
            self._log_debug(
                f"Evicted {len(keys_to_evict)} oldest code span cache entries"
            )

        self._code_spans_cache[cache_key] = spans
        return spans

    def _remove_code_spans(self, content: str, spans: List[Tuple[int, int]]) -> str:
        chars = list(content)
        for start, end in spans:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        return "".join(chars)

    def _is_span_in_code(
        self, code_spans: List[Tuple[int, int]], span: Tuple[int, int]
    ) -> bool:
        s, e = span
        return any(cs <= s and e <= ce for cs, ce in code_spans)

    def _ensure_cleanup_task(self) -> None:
        if (
            not self.valves.enable_response_cache
            or not HAS_CHROMA
            or self._response_cache_cleanup_task is not None
        ):
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._response_cache_cleanup_task = asyncio.create_task(
                    self._periodic_response_cache_cleanup()
                )
        except RuntimeError:
            pass

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
            self._log_debug(f"Evicting project {oldest_pid} from cache")
        if state["active_blocks"]:
            self._rebuild_symbol_index(state, project_id)
        return state

    def _set_state(self, project_id: str, state: Dict):
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        task = asyncio.create_task(self._save_state_to_db_async(project_id, state))
        task.add_done_callback(
            lambda t: (
                self._log_debug(f"Failed to save state: {t.exception()}")
                if t.exception()
                else None
            )
        )

    async def _save_state_to_db_async(self, project_id: str, state: Dict):
        lock = await self._get_project_lock(project_id)
        async with lock:
            await self._save_state_to_db(project_id, state)

    async def _save_state_to_db(self, project_id: str, state: Dict):
        serializable = {
            "active_blocks": {k: v.dict() for k, v in state["active_blocks"].items()},
            "recent_changes": [b.dict() for b in state["recent_changes"]],
            "committed_changes": [b.dict() for b in state["committed_changes"]],
            "feedback_history": [fb.dict() for fb in state["feedback_history"]],
            "facts": state.get("facts", []),
            "iterative_state": state.get("iterative_state"),
            "message_count": state["message_count"],
            "last_compression_timestamp": state.get("last_compression_timestamp", 0),
            "response_cache": state.get("response_cache", []),
            "last_suggestion_timestamp": state.get("last_suggestion_timestamp", 0),
        }
        await anyio.to_thread.run_sync(
            lambda: self._db_conn.execute(
                "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
                (project_id, json.dumps(serializable), time.time()),
            )
        )
        await anyio.to_thread.run_sync(lambda: self._db_conn.commit())

    def _init_state_db(self):
        db_path = self.valves.state_db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                project_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._log_debug(f"State DB initialized at {db_path}")

    def _get_project_id(self) -> str:
        return self.valves.project_id

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
            "facts",
            "last_suggestion_timestamp",
            "response_cache",
            "has_any_calls",
        ]:
            data.setdefault(
                key, [] if key in ("feedback_history", "facts", "response_cache") else 0
            )
        active = {}
        for k, v in data.get("active_blocks", {}).items():
            try:
                active[k] = CodeBlock(**v)
            except Exception:
                self._log_debug(f"Skipping corrupted block {k} in state DB")
        recent = [CodeBlock(**b) for b in data.get("recent_changes", [])]
        committed = [CodeBlock(**b) for b in data.get("committed_changes", [])]
        feedback = [
            AppliedChangeFeedback(**fb) for fb in data.get("feedback_history", [])
        ]
        state = {
            "active_blocks": active,
            "recent_changes": recent,
            "committed_changes": committed,
            "feedback_history": feedback,
            "facts": data.get("facts", []),
            "iterative_state": data.get("iterative_state"),
            "message_count": data.get("message_count", 0),
            "last_compression_timestamp": data.get("last_compression_timestamp", 0),
            "response_cache": data.get("response_cache", []),
            "last_suggestion_timestamp": data.get("last_suggestion_timestamp", 0),
            "has_any_calls": data.get("has_any_calls", False),
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

    def _init_long_term_memory(self):
        os.makedirs(self.valves.long_term_memory_dir, exist_ok=True)
        if _SHARED_RESOURCES_AVAILABLE:
            try:
                self.embedder = _shared_get_embedder()
                self._log_debug("Embedder: using shared singleton")
            except Exception as e:
                self._log_debug(f"shared embedder failed ({e}), loading local")
                self.embedder = (
                    SentenceTransformer("all-MiniLM-L6-v2") if HAS_SENTENCE else None
                )
        else:
            self.embedder = (
                SentenceTransformer("all-MiniLM-L6-v2") if HAS_SENTENCE else None
            )

        if _SHARED_RESOURCES_AVAILABLE:
            try:
                self.chroma_client = _shared_get_chroma_client(
                    self.valves.long_term_memory_dir
                )
                self._log_debug("ChromaDB: using shared singleton")
            except Exception as e:
                self._log_debug(f"shared chroma failed ({e}), opening local")
                self.chroma_client = (
                    chromadb.PersistentClient(
                        path=self.valves.long_term_memory_dir,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    if HAS_CHROMA
                    else None
                )
        else:
            self.chroma_client = (
                chromadb.PersistentClient(
                    path=self.valves.long_term_memory_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                if HAS_CHROMA
                else None
            )

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
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._purge_expired_memories())
        except RuntimeError:
            pass
        self._log_debug("LTM ready")

    async def _purge_expired_memories(self):
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

    def _init_llm_cache(self):
        ttl = self.valves.LLM_CACHE_TTL
        max_size = self.valves.LLM_CACHE_MAX_SIZE
        if _SHARED_RESOURCES_AVAILABLE:
            return _AsyncLRUCache(max_size=max_size, ttl=ttl)
        import asyncio as _asyncio, time as _t

        class _MinimalLRU:
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
                    if self.ttl > 0 and _t.time() - ts > self.ttl:
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
                    self._store[key] = (val, _t.time())
                    self._store[key] = self._store.pop(key)

        return _MinimalLRU(max_size, ttl)

    def _clean_llm_cache(self):
        pass

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: str = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        if not HAS_AIOHTTP:
            return None

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
            self._log_debug(f"Awaiting existing LLM call for dedup key {dedup_key[:8]}")
            return await future

        try:
            base_url = self.valves.LLM_BASE_URL.rstrip("/")
            api_token = self.valves.LLM_API_TOKEN.strip() or None
            is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

            models_to_try = []
            seen = set()
            for m in [
                model_override,
                self.valves.llm_model,
                self.valves.summarization_model,
            ]:
                if m and m not in seen:
                    models_to_try.append(m)
                    seen.add(m)

            max_retries = 2
            base_delay = 1.0

            for model in models_to_try:
                cache_key = hashlib.md5(
                    f"{model}|{prompt}|{system_prompt}|{temperature}|{max_tokens}".encode()
                ).hexdigest()
                cached = await self._llm_cache.get(cache_key)
                if cached is not None:
                    self._log_debug(f"LLM cache hit for model {model}")
                    future.set_result(cached)
                    return cached

                if _SHARED_RESOURCES_AVAILABLE:
                    from shared_resources import call_llm as _shared_call_llm

                    for attempt in range(max_retries + 1):
                        try:
                            async with self._llm_semaphore:
                                content = await _shared_call_llm(
                                    prompt=prompt,
                                    system=system_prompt,
                                    base_url=self.valves.LLM_BASE_URL,
                                    model=model,
                                    api_token=self.valves.LLM_API_TOKEN,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    timeout=self.valves.llm_request_timeout,
                                )
                            if content:
                                await self._llm_cache.set(cache_key, content)
                                future.set_result(content)
                                return content
                        except RuntimeError as exc:
                            status_hint = str(exc)
                            if any(
                                c in status_hint
                                for c in ("429", "500", "502", "503", "504")
                            ):
                                if attempt < max_retries:
                                    await asyncio.sleep(base_delay * (2**attempt))
                                    continue
                            break
                        except Exception as exc:
                            if attempt < max_retries:
                                await asyncio.sleep(base_delay * (2**attempt))
                                continue
                            logger.warning(
                                f"shared call_llm failed after retries: {exc}"
                            )
                            break
                    continue

                try:
                    http_session = await _shared_get_http_session(
                        self.valves.llm_request_timeout
                    )
                except Exception:
                    http_session = self._http_session
                if http_session is None:
                    continue

                for attempt in range(max_retries + 1):
                    try:
                        async with self._llm_semaphore:
                            model_name = (
                                model.split("/", 1)[1]
                                if is_ollama and model.startswith("ollama/")
                                else model
                            )
                            if is_ollama:
                                url = f"{base_url}/api/generate"
                                payload = {
                                    "model": model_name,
                                    "prompt": prompt,
                                    "system": system_prompt,
                                    "stream": False,
                                    "options": {
                                        "temperature": temperature,
                                        "num_predict": max_tokens,
                                    },
                                }
                                headers = {"Content-Type": "application/json"}
                            else:
                                url = f"{base_url}/v1/chat/completions"
                                headers = {"Content-Type": "application/json"}
                                if api_token:
                                    headers["Authorization"] = f"Bearer {api_token}"
                                elif self.valves.openai_api_key:
                                    headers["Authorization"] = (
                                        f"Bearer {self.valves.openai_api_key}"
                                    )
                                payload = {
                                    "model": model_name,
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": prompt},
                                    ],
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                }

                            async with http_session.post(
                                url, json=payload, headers=headers
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    if is_ollama:
                                        content = data.get("response", "")
                                        if not content.strip():
                                            continue
                                        content = content.strip()
                                    else:
                                        choices = data.get("choices", [])
                                        if not choices:
                                            continue
                                        content = (
                                            choices[0]
                                            .get("message", {})
                                            .get("content", "")
                                        )
                                        if not content:
                                            continue
                                        content = content.strip()
                                    await self._llm_cache.set(cache_key, content)
                                    future.set_result(content)
                                    return content
                                elif resp.status in (429, 500, 502, 503, 504):
                                    if attempt < max_retries:
                                        delay = base_delay * (2**attempt)
                                        self._log_debug(
                                            f"LLM call failed with {resp.status}, retrying in {delay}s..."
                                        )
                                        await asyncio.sleep(delay)
                                        continue
                                    else:
                                        self._log_debug(
                                            f"LLM call failed after {max_retries} retries"
                                        )
                                        break
                                else:
                                    break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries:
                            delay = base_delay * (2**attempt)
                            self._log_debug(
                                f"LLM connection error: {e}, retrying in {delay}s..."
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                f"LLM call failed after {max_retries} retries: {e}"
                            )
                            break

            logger.warning(f"All LLM models failed for prompt: {prompt[:100]}...")
            future.set_result(None)
            return None

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
        timeout: float = 8.0,
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

    async def _verify_command_intent(
        self, user_message_cleaned: str, action_description: str
    ) -> bool:
        prompt = (
            f"A command parser interpreted the following user message (code already removed) as:\n"
            f"{action_description}\n\n"
            f"Original message: {user_message_cleaned[:500]}\n\n"
            f"Is this interpretation correct? Answer only 'yes' or 'no'."
        )
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="You are a strict yes/no verifier. Answer only 'yes' or 'no'.",
            model_override=None,
            max_tokens=3,
            temperature=0.0,
            timeout=4.0,
        )
        return bool(response and response.strip().lower().startswith("yes"))

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
                self._log_debug(
                    f"Purged {len(to_delete)} expired response cache entries"
                )
        except Exception as e:
            self._log_debug(f"Error purging response cache: {e}")

    async def _store_response_in_cache(
        self, query: str, response: str, context_hash: str, state: dict
    ):
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return
        try:
            embedding = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
            )
            entry_id = hashlib.md5(
                f"{self.valves.project_id}|{query}".encode()
            ).hexdigest()[:32]

            max_entries = self.valves.response_cache_max_entries
            if self._response_cache_size >= max_entries:
                existing = await anyio.to_thread.run_sync(
                    lambda: col.get(
                        where={"project_id": self.valves.project_id},
                        include=["metadatas"],
                    )
                )
                if existing and len(existing["ids"]) >= max_entries:
                    items = sorted(
                        zip(existing["ids"], existing["metadatas"]),
                        key=lambda x: x[1].get("timestamp", 0),
                    )
                    to_delete = [iid for iid, _ in items[: max(1, len(items) // 10)]]
                    await anyio.to_thread.run_sync(lambda: col.delete(ids=to_delete))
                    self._response_cache_size = len(existing["ids"]) - len(to_delete)
                else:
                    self._response_cache_size = len(existing["ids"]) if existing else 0

            await anyio.to_thread.run_sync(
                lambda: col.upsert(
                    ids=[entry_id],
                    embeddings=[embedding],
                    documents=[response],
                    metadatas=[
                        {
                            "query": query[:500],
                            "project_id": self.valves.project_id,
                            "context_hash": context_hash,
                            "timestamp": time.time(),
                        }
                    ],
                )
            )
            self._response_cache_size += 1
            self._log_debug(f"Stored response in ChromaDB cache for: {query[:50]}")
        except Exception as e:
            self._log_debug(f"Error storing response in cache: {e}")

    async def _find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return None
        try:
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
            if (
                self.valves.response_cache_include_context_hash
                and meta.get("context_hash", "") != context_hash
            ):
                return None
            ttl = self.valves.response_cache_ttl_hours * 3600
            ts = meta.get("timestamp", 0)
            if ttl > 0 and time.time() - ts > ttl:
                await anyio.to_thread.run_sync(
                    lambda: col.delete(ids=[results["ids"][0][0]])
                )
                return None
            doc = results["documents"][0][0]
            self._log_debug(
                f"Response cache HIT (sim={similarity:.3f}) for: {query[:50]}"
            )
            return {"response": doc, "query": meta.get("query", ""), "timestamp": ts}
        except Exception as e:
            self._log_debug(f"Error finding cached response: {e}")
            return None

    def _ensure_last_message_is_user(self, messages: List[dict]) -> List[dict]:
        if not messages:
            messages.append({"role": "user", "content": "continue"})
            self._log_debug("Inserted dummy user message to satisfy API (empty list)")
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
            self._log_debug(
                "Inserted dummy user message to satisfy API (no user in list)"
            )
        else:
            if last_user_idx + 1 < len(messages):
                removed = len(messages) - (last_user_idx + 1)
                messages = messages[: last_user_idx + 1]
                self._log_debug(f"Trimmed {removed} trailing assistant messages")

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

    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))
        return fuzz.token_sort_ratio(code1, code2) / 100.0

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    async def _extract_code_blocks(
        self, content: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        blocks = []
        spans = []
        if not self.valves.auto_detect_code_blocks:
            return blocks, spans

        if HAS_TREE_SITTER:
            try:
                config = ProcessConfig()
                ts_blocks = process(content, config)
                for tsb in ts_blocks:
                    start, end = tsb.start_byte, tsb.end_byte
                    raw = content[start:end].strip()
                    lang = tsb.language or "text"

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

                self._log_debug(f"Tree‑sitter extracted {len(blocks)} code blocks")
                return blocks, spans

            except Exception as e:
                self._log_debug(
                    f"Tree‑sitter extraction failed, falling back to regex: {e}"
                )

        for match in self.code_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            code = await self._handle_oversized_code_block(code, lang)
            blocks.append({"language": lang, "code": code, "type": "fenced"})
            spans.append((match.start(), match.end()))

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
                    code = await self._handle_oversized_code_block(code, "text")
                    start_offset = line_offsets[i - len(indented)]
                    end_offset = line_offsets[i] - 1
                    blocks.append(
                        {"language": "text", "code": code, "type": "indented"}
                    )
                    spans.append((start_offset, end_offset))
                indented = []
                i += 1
        if len(indented) >= 3:
            code = "\n".join(indented)
            code = await self._handle_oversized_code_block(code, "text")
            start_offset = line_offsets[len(lines) - len(indented)]
            end_offset = line_offsets[-1] - 1 if line_offsets[-1] > 0 else len(content)
            blocks.append({"language": "text", "code": code, "type": "indented"})
            spans.append((start_offset, end_offset))

        return blocks, spans

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
            file_path = match.group(1)
            line_start = int(match.group(2))
            line_end = int(match.group(3)) if match.group(3) else line_start
            return file_path, line_start, line_end
        return None, None, None

    def _extract_file_paths(self, content: str) -> List[str]:
        if not self.valves.track_file_paths:
            return []
        matches = re.findall(self.valves.file_path_pattern, content)
        return [m[0] if isinstance(m, tuple) else m for m in matches]

    def _classify_content(
        self, content: str, extracted_blocks: List[Dict]
    ) -> ContentType:
        cl = content.lower()
        if self.diff_pattern.search(content) or "diff --git" in content:
            return ContentType.PROPOSED_CHANGE
        if self.commit_pattern.search(content):
            if "applied" in cl or "committed" in cl or "merged" in cl:
                return ContentType.COMMITTED_CHANGE
            return ContentType.PROPOSED_CHANGE
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

    async def _classify_session(self, messages: List[dict], project_id: str) -> bool:
        """Determine if this is a coding session.
        Uses a 1800s TTL cache, state heuristics, code indicators in the last
        10 user messages, and falls back to a fast LLM call only when uncertain.
        """
        # 1. Cache check (TTL 1800s set in __init__)
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

        # 2. Existing state → definitely a coding session
        state = self._get_state(project_id)
        if state and state.get("active_blocks"):
            if cache_key:
                self._session_classify_cache[cache_key] = (True, time.time())
            return True

        # 3. Heuristics on the last 10 user messages (no LLM)
        for msg in reversed(messages[-10:]):
            if msg.get("role") != "user":
                continue
            if self._has_code_indicators(msg.get("content", "")):
                if cache_key:
                    self._session_classify_cache[cache_key] = (True, time.time())
                return True

        # 4. Explicit command (starts with "/") → treat as coding
        if last_user and last_user.get("content", "").strip().startswith("/"):
            if cache_key:
                self._session_classify_cache[cache_key] = (True, time.time())
            return True

        # 5. Fallback LLM only when no heuristic matched and there is enough text
        if not last_user or len(last_user.get("content", "")) < 20:
            result = False
        else:
            model = self.valves.natural_language_forget_model or self.valves.llm_model
            prompt = (
                f"Is this message about programming or code? Answer only 'yes' or 'no'.\n\n"
                f"Message: {last_user.get('content','')[:300]}"
            )
            response = await self._try_llm_quick(
                prompt=prompt,
                system_prompt="You are a classifier. Answer only 'yes' or 'no'.",
                model_override=model,
                max_tokens=3,
                temperature=0.0,
                timeout=5.0,
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

    def _get_active_code_context(self, project_id: str) -> str:
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
        boost = self.valves.raw_file_priority_boost
        active.sort(
            key=lambda b: b.importance_score + (boost if b.is_raw else 0), reverse=True
        )
        base_codes = sorted(
            [b for b in active if b.content_type == ContentType.BASE_CODE],
            key=lambda b: b.importance_score,
            reverse=True,
        )[: self.valves.max_base_code_blocks]
        proposed = sorted(
            [b for b in active if b.content_type == ContentType.PROPOSED_CHANGE],
            key=lambda b: b.importance_score,
            reverse=True,
        )[: self.valves.max_proposed_changes]
        committed = [
            b for b in active if b.content_type == ContentType.COMMITTED_CHANGE
        ][: self.valves.max_committed_changes]
        errors = (
            [b for b in active if b.content_type == ContentType.ERROR][:3]
            if self.valves.preserve_error_context
            else []
        )
        parts = ["## Currently Active Code Context (by importance)\n"]
        if base_codes:
            parts.append("### Base Code (current work):")
            for b in base_codes:
                loc = (
                    f" (file: {b.file_path}{(' lines ' + str(b.line_range[0]) + '-' + str(b.line_range[1]) if b.line_range else '')})"
                    if b.file_path
                    else ""
                )
                pin = " [PINNED]" if b.pinned else ""
                raw = " [RAW]" if b.is_raw else ""
                aff = (
                    " [AFFECTED BY DEPENDENCY CHANGE]" if b.potentially_affected else ""
                )
                parts.append(
                    f"```\n{b.content[:600]}\n```{loc}  (importance: {b.importance_score:.1f}){aff}{pin}{raw}"
                )
        if proposed:
            parts.append("### Proposed Changes (pending review):")
            for b in proposed:
                parts.append(f"```diff\n{b.content[:500]}\n```")
        if committed:
            parts.append("### Recently Committed Changes:")
            for b in committed:
                parts.append(f"```\n{b.content[:300]}\n```")
        if errors:
            parts.append("### Recent Errors:")
            for b in errors:
                parts.append(f"```\n{b.content[:500]}\n```")
        return "\n".join(parts)

    # --------------------------------------------------------------------------
    #  Lightweight context support methods
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
                summary_str = f"  Summary: {s.summary}" if s.summary else ""
                lines.append(
                    f"- `{s.signature}` [{s.kind}]{loc}{calls_str}{summary_str}"
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

    def _expand_referenced_symbols(self, project_id: str, user_query: str) -> str:
        state = self._get_state(project_id)
        has_calls = state.get("has_any_calls", False) if state else False
        MAX_EXPANDED_BODIES = (
            2 if (self.valves.enable_call_graph_extraction and has_calls) else 5
        )
        if not state:
            return ""

        all_symbol_names = self._symbol_index.get_all_names(project_id)
        words = set(re.findall(r"\b[\w-]+\b", user_query))
        mentioned_names = all_symbol_names.intersection(words)

        if not mentioned_names:
            return ""

        mentioned_hashes: Set[str] = set()
        for name in mentioned_names:
            mentioned_hashes.update(self._symbol_index.find_blocks(name, project_id))

        sorted_hashes = sorted(
            mentioned_hashes,
            key=lambda h: state["active_blocks"]
            .get(h, CodeBlock(content=""))
            .importance_score,
            reverse=True,
        )

        parts = ["\n## Expanded Code Bodies (referenced symbols)\n"]
        for block_hash in sorted_hashes[:MAX_EXPANDED_BODIES]:
            block = state["active_blocks"].get(block_hash)
            if not block:
                continue
            loc = f" (file: {block.file_path})" if block.file_path else ""
            parts.append(
                f"### Block {block.hash[:8]}{loc}\n```\n{block.content[:3000]}\n```"
            )
        return "\n".join(parts)

    def _invalidate_lightweight_cache(self, project_id: str):
        self._cached_lightweight_context.pop(project_id, None)

    async def _is_structural_task(self, user_message: str) -> bool:
        indicators = [
            r"\bdiagrama\b",
            r"\bdiagram\b",
            r"\bflowchart\b",
            r"\bflujo\b",
            r"\bgrafo\b",
            r"\bgraph\b",
            r"\bestructura\b",
            r"\bstructure\b",
            r"\bdependencias\b",
            r"\bdependencies\b",
            r"\barquitectura\b",
            r"\barchitecture\b",
            r"\bcall graph\b",
            r"\bllamadas\b",
            r"\bresumen\s+de\s+arquitectura\b",
            r"\bclass diagram\b",
            r"\bdependency graph\b",
            r"\bcall hierarchy\b",
            r"\bhow\s+do\s+these\b",
            r"\bhow\s+are\s+these\b",
            r"\brelationship\b",
            r"\brelación\b",
            r"\bconnected\b",
            r"\bconectados\b",
            r"\bcomponent diagram\b",
            r"\bmodule\s+dependencies\b",
            r"\bsequence diagram\b",
        ]
        content = user_message.strip().lower()
        if any(re.search(pat, content) for pat in indicators):
            return True
        if len(content) > 50 and (
            "?" in content or "cómo" in content or "how" in content
        ):
            try:
                model = (
                    self.valves.natural_language_forget_model or self.valves.llm_model
                )
                prompt = f'Is this question asking for a structural representation (diagram, flowchart, call graph, dependency) of code? Answer only "yes" or "no".\n\nQuestion: {content[:300]}'
                response = await asyncio.wait_for(
                    self._call_llm(
                        prompt=prompt,
                        system_prompt="You are a classifier. Answer only 'yes' or 'no'.",
                        model_override=model,
                        max_tokens=3,
                        temperature=0.0,
                    ),
                    timeout=1.0,
                )
                return response and response.strip().lower().startswith("yes")
            except (asyncio.TimeoutError, Exception):
                pass
        return False

    async def _generate_missing_summaries(self, project_id: str):
        if not self.valves.enable_auto_summaries or not HAS_AIOHTTP:
            return
        state = self._get_state(project_id)
        symbols_to_summarize = []
        for block in state["active_blocks"].values():
            for sym in block.symbols:
                if sym.summary:
                    continue
                if sym.kind not in ("function", "method"):
                    continue
                symbols_to_summarize.append((sym, block.content[:500]))
        if not symbols_to_summarize:
            return
        batch_size = 10
        for i in range(0, len(symbols_to_summarize), batch_size):
            batch = symbols_to_summarize[i : i + batch_size]
            tasks = []
            for sym, code_snippet in batch:
                prompt = f"Summarize in one short sentence what this code does:\n\n```{sym.signature}\n{code_snippet}```"
                tasks.append(
                    self._call_llm(
                        prompt=prompt,
                        system_prompt="You are a code summarization assistant. Output only one concise sentence.",
                        model_override=self.valves.summarization_model,
                        max_tokens=50,
                        temperature=0.1,
                    )
                )
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for (sym, _), resp in zip(batch, responses):
                if isinstance(resp, str) and resp.strip():
                    sym.summary = resp.strip()
            await asyncio.sleep(1.0)
        self._set_state(project_id, state)

    async def _update_active_code(self, message: dict, project_id: str):
        if not self.valves.enable_code_awareness:
            return
        lock = await self._get_project_lock(project_id)

        content = message.get("content", "")
        role = message.get("role", "")
        extracted, block_spans = await self._extract_code_blocks(content)

        # Extract symbols for all new blocks OUTSIDE the lock to avoid blocking
        new_blocks_pending = []
        for idx, block_info in enumerate(extracted):
            # Per‑block file path: use the text immediately preceding this block
            blk_file = None
            if self.valves.track_file_paths and block_spans:
                blk_file = self._extract_file_path_for_block(
                    content, block_spans[idx][0]
                )

            # If no per‑block path found, fallback to global detection only for single‑block messages
            if not blk_file and len(extracted) == 1:
                extracted_paths = self._extract_file_paths(content)
                blk_file = extracted_paths[0] if extracted_paths else None

            # line_start/line_end are still extracted from the whole message (rarely used)
            content_type = self._classify_content(content, extracted)

            new_block = CodeBlock(
                content=block_info["code"],
                content_type=content_type,
                generated_by_assistant=(role == "assistant"),
                file_path=blk_file,
                line_range=None,  # We no longer use a global line range; per‑block ranges could be added later
                timestamp=time.time(),
                is_active=True,
                mention_count=1,
                dependencies=[],
                potentially_affected=False,
                pinned=False,
                obsolete=False,
            )
            if "[KEEP]" in content:
                new_block.is_raw = True
            if "[KEEP]" in content or "#important" in content.lower():
                new_block.importance_score = 10.0
                new_block.pinned = True
                self._log_debug(
                    f"Manual importance marker detected for block {new_block.hash}, pinned automatically"
                )
            new_blocks_pending.append(new_block)

        # Extract signatures and call graphs sequentially in the same thread
        symbols_list = []
        for blk in new_blocks_pending:
            syms = await SignatureExtractor.extract_async(blk.content, blk.file_path)
            symbols_list.append(syms)

        _content_to_syms: Dict[str, List[CodeSymbol]] = {
            blk.content: syms
            for blk, syms in zip(new_blocks_pending, symbols_list)
            if not isinstance(syms, Exception)
        }

        async with lock:
            state = self._get_state(project_id)
            # Kick off background tasks
            task = asyncio.create_task(
                self._summarize_inactive_blocks_safely(project_id)
            )
            task.add_done_callback(
                lambda t: (
                    self._log_debug(
                        f"Summarize inactive blocks failed: {t.exception()}"
                    )
                    if t.exception()
                    else None
                )
            )
            self._summarize_tasks.append(task)
            if len(self._summarize_tasks) > 10:
                self._summarize_tasks = self._summarize_tasks[-10:]

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
                    block._update_importance()

            if not content and not new_blocks_pending:
                return

            # Process each new block with its extracted symbols
            for new_block, syms in zip(new_blocks_pending, symbols_list):
                if isinstance(syms, Exception):
                    syms = []
                # Compute token count
                if self.tokenizer:
                    new_block._cached_token_count = len(
                        self.tokenizer.encode(new_block.content)
                    )
                else:
                    new_block._cached_token_count = len(new_block.content) // 4

                is_dup, existing = self._is_duplicate_code(
                    new_block, list(state["active_blocks"].values())
                )
                if is_dup and existing:
                    if existing.pinned or new_block.is_raw:
                        # Update pinned block
                        self._symbol_index.remove_all_for_block(
                            existing.hash, existing.symbols, project_id
                        )
                        existing.content = new_block.content
                        existing.hash = new_block.hash
                        if new_block.file_path:
                            existing.file_path = new_block.file_path
                        existing.line_range = new_block.line_range
                        existing.timestamp = time.time()
                        existing.mention_count += 1
                        existing.last_mentioned = time.time()
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
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        self._log_debug(
                            f"Updated existing pinned block {existing.hash} (raw extraction or similar code)"
                        )
                        if (
                            self.valves.enable_dependency_tracking
                            and self.valves.dependency_refresh_on_update
                        ):
                            task = asyncio.create_task(
                                self._refresh_dependencies_for_block(
                                    existing.hash, project_id
                                )
                            )
                            task.add_done_callback(
                                lambda t: (
                                    self._log_debug(
                                        f"Dependency refresh failed: {t.exception()}"
                                    )
                                    if t.exception()
                                    else None
                                )
                            )
                            self._dependency_tasks.append(task)
                            if len(self._dependency_tasks) > 10:
                                self._dependency_tasks = self._dependency_tasks[-10:]
                        continue
                    if self.valves.prioritize_recent_code:
                        # Update existing block
                        self._symbol_index.remove_all_for_block(
                            existing.hash, existing.symbols, project_id
                        )
                        existing.content = new_block.content
                        existing.hash = new_block.hash
                        if new_block.file_path:
                            existing.file_path = new_block.file_path
                        existing.line_range = new_block.line_range
                        existing.timestamp = time.time()
                        existing.mention_count += 1
                        existing.last_mentioned = time.time()
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
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        self._log_debug(f"Updated existing block {existing.hash}")
                        if (
                            self.valves.enable_dependency_tracking
                            and self.valves.dependency_refresh_on_update
                        ):
                            task = asyncio.create_task(
                                self._refresh_dependencies_for_block(
                                    existing.hash, project_id
                                )
                            )
                            task.add_done_callback(
                                lambda t: (
                                    self._log_debug(
                                        f"Dependency refresh failed: {t.exception()}"
                                    )
                                    if t.exception()
                                    else None
                                )
                            )
                            self._dependency_tasks.append(task)
                            if len(self._dependency_tasks) > 10:
                                self._dependency_tasks = self._dependency_tasks[-10:]
                    continue

                # --- Only for truly new blocks (not duplicates) ---
                # Assign symbols and update index
                for sym in syms:
                    sym.parent_block_hash = new_block.hash
                new_block.symbols = syms
                for sym in syms:
                    self._symbol_index.add(sym, new_block.hash, project_id)
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
                        self._log_debug(
                            f"Proposed change {new_block.hash} marked as conflicting"
                        )

                state["active_blocks"][new_block.hash] = new_block
                self._log_debug(
                    f"New {new_block.content_type.value} block: {new_block.hash}"
                )

                if new_block.content_type == ContentType.PROPOSED_CHANGE:
                    # Resolve any previous proposed changes on the same file
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

                if (
                    self.valves.enable_dependency_tracking
                    and new_block.content_type
                    in (
                        ContentType.BASE_CODE,
                        ContentType.PROPOSED_CHANGE,
                        ContentType.COMMITTED_CHANGE,
                    )
                ):
                    task = asyncio.create_task(
                        self._refresh_dependencies_for_block(new_block.hash, project_id)
                    )
                    task.add_done_callback(
                        lambda t: (
                            self._log_debug(
                                f"Dependency refresh failed: {t.exception()}"
                            )
                            if t.exception()
                            else None
                        )
                    )
                    self._dependency_tasks.append(task)
                    if len(self._dependency_tasks) > 10:
                        self._dependency_tasks = self._dependency_tasks[-10:]

            # Assistant update: detect if base code blocks were modified implicitly
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
                        best_base.content = block_info["code"]
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
                        if self.tokenizer:
                            best_base._cached_token_count = len(
                                self.tokenizer.encode(best_base.content)
                            )
                        else:
                            best_base._cached_token_count = len(best_base.content) // 4
                        if any(s.calls for s in best_base.symbols):
                            state["has_any_calls"] = True
                        self._log_debug(
                            f"Assistant updated base code block {best_base.hash} (sim={best_sim:.2f})"
                        )
                        if (
                            self.valves.enable_dependency_tracking
                            and self.valves.dependency_refresh_on_update
                        ):
                            task = asyncio.create_task(
                                self._refresh_dependencies_for_block(
                                    best_base.hash, project_id
                                )
                            )
                            task.add_done_callback(
                                lambda t: (
                                    self._log_debug(
                                        f"Dependency refresh failed: {t.exception()}"
                                    )
                                    if t.exception()
                                    else None
                                )
                            )
                            self._dependency_tasks.append(task)
                            if len(self._dependency_tasks) > 10:
                                self._dependency_tasks = self._dependency_tasks[-10:]

            state["message_count"] += 1
            if self.valves.auto_remove_duplicate_blocks:
                self._remove_duplicate_blocks(state, project_id)
            if self.valves.hierarchical_compression_enabled:
                if (
                    state["message_count"]
                    % self.valves.hierarchical_compression_interval_messages
                    == 0
                ):
                    task = asyncio.create_task(
                        self._hierarchical_compress(project_id, state)
                    )
                    task.add_done_callback(
                        lambda t: (
                            self._log_debug(
                                f"Hierarchical compress failed: {t.exception()}"
                            )
                            if t.exception()
                            else None
                        )
                    )
                    self._hierarchical_compress_tasks.append(task)
                    if len(self._hierarchical_compress_tasks) > 10:
                        self._hierarchical_compress_tasks = (
                            self._hierarchical_compress_tasks[-10:]
                        )
            asyncio.create_task(
                self._expire_blocks_by_time(project_id)
            ).add_done_callback(
                lambda t: (
                    self._log_debug(f"Expire blocks failed: {t.exception()}")
                    if t.exception()
                    else None
                )
            )
            asyncio.create_task(
                self._clean_affected_flags(project_id)
            ).add_done_callback(
                lambda t: (
                    self._log_debug(f"Clean affected flags failed: {t.exception()}")
                    if t.exception()
                    else None
                )
            )
            if self.valves.enable_auto_summaries:
                asyncio.create_task(self._generate_missing_summaries(project_id))
            self._invalidate_lightweight_cache(project_id)
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    #  Optimized mention tracking using symbol index
    # --------------------------------------------------------------------------
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
                block._update_importance()
                self._log_debug(
                    f"Boosted importance of {block.hash} due to symbol mention in message"
                )

    # --------------------------------------------------------------------------
    #  Unchanged helpers
    # --------------------------------------------------------------------------
    def _is_duplicate_code(
        self, new_block: CodeBlock, existing_blocks: List[CodeBlock]
    ) -> Tuple[bool, Optional[CodeBlock]]:
        new_len = len(new_block.content)
        for ex in existing_blocks:
            ex_len = len(ex.content)
            if ex_len > 0 and abs(new_len - ex_len) > 0.2 * max(new_len, ex_len):
                continue
            if (
                self._calculate_code_similarity(new_block.content, ex.content)
                >= self.valves.code_similarity_threshold
            ):
                return True, ex
        return False, None

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

    # --------------------------------------------------------------------------
    #  Cleanup
    # --------------------------------------------------------------------------
    def shutdown(self):
        if (
            hasattr(self, "_response_cache_cleanup_task")
            and self._response_cache_cleanup_task is not None
        ):
            self._response_cache_cleanup_task.cancel()
            self._log_debug("Response cache cleanup task cancelled")

        if hasattr(self, "_hierarchical_compress_tasks"):
            for task in self._hierarchical_compress_tasks:
                task.cancel()
            self._log_debug(
                f"Cancelled {len(self._hierarchical_compress_tasks)} hierarchical compress tasks"
            )

        if hasattr(self, "_summarize_tasks"):
            for task in self._summarize_tasks:
                task.cancel()
            self._log_debug(
                f"Cancelled {len(self._summarize_tasks)} summarization tasks"
            )

        if hasattr(self, "_dependency_tasks"):
            for task in self._dependency_tasks:
                task.cancel()
            self._log_debug(f"Cancelled {len(self._dependency_tasks)} dependency tasks")

        self._symbol_index.clear()
        self._cached_lightweight_context.clear()
        self._project_locks.clear()

    def _cleanup_completed_tasks(self):
        self._hierarchical_compress_tasks = [
            t for t in self._hierarchical_compress_tasks if not t.done()
        ]
        self._summarize_tasks = [t for t in self._summarize_tasks if not t.done()]
        self._dependency_tasks = [t for t in self._dependency_tasks if not t.done()]

    # --------------------------------------------------------------------------
    #  MODIFIED: inlet
    # --------------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("inlet called")
        inlet_start = time.time()

        self._ensure_cleanup_task()
        messages = body.get("messages", [])
        project_id = self._get_project_id()

        if self._last_project_id and self._last_project_id != project_id:
            self._log_debug(
                f"Project changed from {self._last_project_id} to {project_id}"
            )
            old_state = self._conversation_state.get(self._last_project_id)
            if old_state:
                self._remove_project_from_index_by_id(self._last_project_id, old_state)
            self._cached_lightweight_context.pop(self._last_project_id, None)
        self._last_project_id = project_id

        if not messages:
            return body

        state = self._get_state(project_id)
        is_code_session = await self._classify_session(messages, project_id)
        self._log_debug(f"Session: {'code' if is_code_session else 'non-code'}")

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        is_explicit_command = last_user_msg and last_user_msg.get(
            "content", ""
        ).startswith("/")

        # ── Unified Intent Handling ──────────────────────────────────
        if self.valves.enable_forget_command and is_explicit_command:
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                return body

        if (
            self.valves.enable_natural_language_forget
            and last_user_msg
            and not is_explicit_command
            and (is_code_session or is_explicit_command)
        ):
            intents = await self._parse_all_intents(last_user_msg.get("content", ""))

            # Forget
            fi = intents.get("forget", {})
            if fi.get("action") not in (None, "none"):
                confirmation = await self._execute_forget_intent(project_id, fi)
                status_msg = f"[CodeAware] {confirmation}"
                messages.insert(0, {"role": "system", "content": status_msg})
                messages.pop()
                messages.append({"role": "assistant", "content": confirmation})
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                return body

            # Remember
            ri = intents.get("remember", {})
            if ri.get("action") not in (None, "none"):
                confirmation = await self._execute_remember_intent(project_id, ri)
                status_msg = f"[CodeAware] {confirmation}"
                messages.insert(0, {"role": "system", "content": status_msg})
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                return body

            # Obsolete
            oi = intents.get("obsolete", {})
            if (
                oi.get("action") not in (None, "none")
                and self.valves.enable_obsolete_marking
            ):
                confirmation = await self._execute_obsolete_intent(project_id, oi)
                status_msg = f"[CodeAware] {confirmation}"
                messages.insert(0, {"role": "system", "content": status_msg})
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                return body

        # Facts – no LLM, kept as before
        if self.valves.enable_facts and last_user_msg:
            facts = await self._extract_facts_from_message(
                last_user_msg.get("content", "")
            )
            for fact_text in facts:
                await self._add_fact(project_id, fact_text)
            if (
                last_user_msg.get("content", "")
                .strip()
                .startswith(self.valves.fact_command_prefix)
            ):
                response_msg = await self._handle_fact_command(
                    last_user_msg.get("content", ""), project_id
                )
                if response_msg:
                    messages.pop()
                    messages.append({"role": "assistant", "content": response_msg})
                    messages = self._ensure_last_message_is_user(messages)
                    body["messages"] = messages
                    return body

        # ── Code interpretation note ──────────────────────────────────
        if is_code_session:
            note = (
                "When reading user messages, treat code inside triple backticks "
                "as literal source code without interpreting Markdown. "
                "You may still use Markdown in your own responses."
            )
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            if sys_msgs:
                if note not in sys_msgs[0].get("content", ""):
                    sys_msgs[0]["content"] = note + "\n" + sys_msgs[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": note})

        # ── /think or auto Chain-of-Thought ───────────────────────────
        if self.valves.enable_cot_on_demand or self.valves.auto_cot_enabled:
            if last_user_msg:
                user_content = last_user_msg.get("content", "")
                if self.valves.enable_cot_on_demand and user_content.strip().startswith(
                    "/think"
                ):
                    cot_question = await self._parse_cot_intent(user_content)
                    if cot_question:
                        active_ctx = self._get_active_code_context(project_id)
                        facts_ctx = self._get_facts_context(project_id)
                        context = f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
                        reasoning = await self._generate_cot(cot_question, context)
                        messages.pop()
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"**Chain-of-Thought Reasoning**\n{reasoning}",
                            }
                        )
                        messages = self._ensure_last_message_is_user(messages)
                        body["messages"] = messages
                        return body
                elif (
                    self.valves.auto_cot_enabled
                    and self._should_auto_cot(user_content)
                    and not user_content.strip().startswith("/")
                ):
                    self._log_debug("Auto-injecting Chain-of-Thought prompt")
                    sys_msgs = [m for m in messages if m.get("role") == "system"]
                    cot_prompt = (
                        "Please think step by step before answering. "
                        "Show your reasoning, then provide the final answer."
                    )
                    if sys_msgs:
                        sys_msgs[0]["content"] = (
                            cot_prompt + "\n" + sys_msgs[0]["content"]
                        )
                    else:
                        messages.insert(0, {"role": "system", "content": cot_prompt})
                    body["messages"] = messages

        # ── /assume ────────────────────────────────────────────────────
        if self.valves.enable_assumption_extraction:
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                assumption_target = await self._parse_assumption_intent(
                    last_user_msg.get("content", "")
                )
                if assumption_target:
                    analysis = await self._extract_assumptions(assumption_target)
                    messages.pop()
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Assumption Analysis**\n{analysis}",
                        }
                    )
                    messages = self._ensure_last_message_is_user(messages)
                    body["messages"] = messages
                    return body

        # ── /iterate ───────────────────────────────────────────────────
        if self.valves.enable_iterative_mode:
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                result, consumed = await self._run_iteration(
                    project_id, last_user_msg.get("content", "")
                )
                if consumed:
                    messages.pop()
                    messages.append({"role": "assistant", "content": result})
                    messages = self._ensure_last_message_is_user(messages)
                    body["messages"] = messages
                    return body

        # ── Smart context selection ────────────────────────────────────
        if (
            self.valves.smart_context_selection
            and len(messages) > 0
            and is_code_session
        ):
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                query = messages[last_user_idx].get("content", "")
                if query:
                    historical = await self._retrieve_historical_messages(
                        query, project_id, self.valves.smart_context_top_k
                    )
                    new_history = []
                    for msg in historical:
                        if msg["content"] != query:
                            new_history.append(msg)
                    new_history.append(messages[last_user_idx])
                    if (
                        self.valves.smart_context_include_last_user
                        and last_user_idx + 1 < len(messages)
                        and messages[last_user_idx + 1].get("role") == "assistant"
                    ):
                        new_history.append(messages[last_user_idx + 1])
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    messages = system_msgs + new_history
                    body["messages"] = messages

        # ── Parallel Checks (contradiction, cache, duplicate) ──────────
        last_user_query = last_user_msg.get("content", "") if last_user_msg else ""
        context_hash = self._compute_context_hash(messages)

        contradiction_warning, cached_response, duplicate_match = (
            await self._parallel_context_checks(
                messages, last_user_query, context_hash, project_id, state
            )
        )

        if contradiction_warning and self.valves.contradiction_inject_warning:
            messages.insert(0, {"role": "system", "content": contradiction_warning})
            body["messages"] = messages

        if cached_response:
            messages.append(
                {"role": "assistant", "content": cached_response["response"]}
            )
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            return body

        if duplicate_match:
            warn_msg = (
                f"⚠️ **Note**: This question is very similar to one you asked before "
                f"(similarity {duplicate_match['sim']:.2f})."
            )
            messages.insert(0, {"role": "system", "content": warn_msg})
            body["messages"] = messages

        # ── Update active code (historical in background, last message sync) ─
        if self.valves.enable_code_awareness and is_code_session:
            last_idx = len(messages) - 1
            start_idx = max(0, self._last_processed_message_idx.get(project_id, -1) + 1)

            if start_idx < last_idx:

                async def _process_historical(msgs, start, end, pid):
                    for i in range(start, end):
                        await self._update_active_code(msgs[i], pid)

                task = asyncio.create_task(
                    _process_historical(messages, start_idx, last_idx, project_id)
                )
                task.add_done_callback(
                    lambda t: (
                        self._log_debug(f"Historical update error: {t.exception()}")
                        if t.exception()
                        else None
                    )
                )

            await self._update_active_code(messages[last_idx], project_id)
            self._last_processed_message_idx[project_id] = last_idx

        # ── LTM retrieval ──────────────────────────────────────────────
        unique_meta = []
        if not self.valves.smart_context_selection and is_code_session:
            if (
                last_user_msg
                and HAS_SENTENCE
                and HAS_CHROMA
                and self.valves.enable_code_awareness
            ):
                query = last_user_msg.get("content", "")
                all_meta = await self._retrieve_all_memories_unified(query, project_id)
                all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
                seen = set()
                unique_meta = []
                for m in all_meta:
                    if m["doc"] not in seen:
                        seen.add(m["doc"])
                        unique_meta.append(m)

        max_ltm_tokens = self.valves.ltm_retrieval_max_tokens
        parts = []
        current_tokens = 0
        header = "## Relevant Past Context (with timestamps)\n\n"
        if max_ltm_tokens > 0:
            current_tokens += (
                len(self.tokenizer.encode(header))
                if self.tokenizer
                else (len(header) // 4)
            )
        for mem in unique_meta:
            ts = mem.get("timestamp")
            if ts and ts > 1000000000:
                time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%SZ"
                )
                text = f"[{time_str}] {mem['doc']}"
            else:
                text = f"[unknown date] {mem['doc']}"
            frag_tokens = (
                len(self.tokenizer.encode(text)) if self.tokenizer else (len(text) // 4)
            )
            if max_ltm_tokens > 0 and current_tokens + frag_tokens > max_ltm_tokens:
                continue
            parts.append(text)
            current_tokens += frag_tokens
        if parts:
            ctx = header + "\n---\n".join(parts)
            if max_ltm_tokens > 0 and len(parts) < len(unique_meta):
                ctx += "\n[Some older fragments omitted to fit token budget]"
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            if sys_msgs:
                sys_msgs[0]["content"] = ctx + "\n\n" + sys_msgs[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": ctx})
            body["messages"] = messages

        # ── Inject active code context (lightweight or full) ───────────
        if is_code_session and self.valves.enable_code_awareness:
            is_structural = last_user_msg and await self._is_structural_task(
                last_user_msg.get("content", "")
            )

            code_blocks = [
                b
                for b in state["active_blocks"].values()
                if b.content_type
                in (
                    ContentType.BASE_CODE,
                    ContentType.PROPOSED_CHANGE,
                    ContentType.COMMITTED_CHANGE,
                )
                and not b.obsolete
            ]
            total_code_tokens = sum(b._cached_token_count for b in code_blocks)

            if total_code_tokens > self.valves.huge_injection_threshold_tokens > 0:
                self._log_debug(
                    f"Massive injection detected ({total_code_tokens} tokens). Using lightweight context."
                )
                active_ctx = await self._build_lightweight_context(project_id)

                if last_user_msg and not is_structural:
                    expanded = self._expand_referenced_symbols(
                        project_id, last_user_msg.get("content", "")
                    )
                    if expanded:
                        active_ctx += "\n" + expanded
                elif is_structural:
                    self._log_debug(
                        "Structural task detected, not expanding code bodies."
                    )
                    active_ctx += (
                        "\n\n[Note: Structural analysis requested. Use the symbol index "
                        "with call graphs and summaries to generate the diagram. Do not "
                        "request code bodies.]"
                    )
            else:
                active_ctx = self._get_active_code_context(project_id)

            if active_ctx:
                checklist = (
                    "## If you are reviewing, fixing, or improving code, follow this checklist:\n"
                    "1. Execute the code mentally with 3 different inputs, including edge cases.\n"
                    "2. Identify every assumption the code makes and verify each one.\n"
                    "3. For every regex or string match, test it against 5 counter-examples.\n"
                    "4. If the code processes a list/collection, test with empty, single-element, and large inputs.\n"
                    "5. Ask yourself: what is the worst-case scenario for this code?\n"
                    "6. Output your reasoning step by step, then provide the corrected code.\n"
                )
                active_ctx = checklist + "\n\n" + active_ctx
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = (
                        active_ctx + "\n\n" + sys_msgs[0]["content"]
                    )
                else:
                    messages.insert(0, {"role": "system", "content": active_ctx})
                body["messages"] = messages

        # ── Code review checklist injection ────────────────────────────
        if (
            is_code_session
            and self.valves.enable_code_review_mode
            and self._is_code_review_request(
                last_user_msg.get("content", "") if last_user_msg else ""
            )
        ):
            self._log_debug("Injecting code review checklist")
            review_prompt = (
                "## Code Review Checklist\n"
                "You are reviewing code. Follow these steps:\n"
                "1. Execute the code mentally with 3 different inputs, including edge cases.\n"
                "2. Identify every assumption the code makes and verify each one.\n"
                "3. For every regex or string match, test it against 5 counter-examples.\n"
                "4. If the code processes a list/collection, test with empty, single-element, and large inputs.\n"
                "5. Ask yourself: what is the worst-case scenario for this code?\n"
                "6. Output your reasoning step by step, then provide the corrected code.\n"
            )
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            if sys_msgs:
                sys_msgs[0]["content"] = review_prompt + "\n" + sys_msgs[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": review_prompt})
            body["messages"] = messages

        # ── Inject facts ───────────────────────────────────────────────
        if (
            is_code_session
            and self.valves.enable_facts
            and self.valves.inject_facts_in_context
        ):
            facts_ctx = self._get_facts_context(project_id)
            if facts_ctx:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = facts_ctx + "\n\n" + sys_msgs[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": facts_ctx})
                body["messages"] = messages

        # ── Confidence scoring ─────────────────────────────────────────
        if self.valves.enable_confidence_scoring and is_code_session:
            total_tokens = self._estimate_tokens(messages)
            if total_tokens > self.valves.context_window_tokens * 0.8:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] += self.valves.confidence_prompt
                else:
                    messages.insert(
                        0, {"role": "system", "content": self.valves.confidence_prompt}
                    )
                body["messages"] = messages

        # ── Inject feedback context ────────────────────────────────────
        if (
            is_code_session
            and self.valves.enable_feedback_tracking
            and self.valves.inject_feedback_context
        ):
            feedback_ctx = self._get_feedback_context(project_id)
            if feedback_ctx:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = (
                        feedback_ctx + "\n\n" + sys_msgs[0]["content"]
                    )
                else:
                    messages.insert(0, {"role": "system", "content": feedback_ctx})
                body["messages"] = messages

        # ── Proactive suggestions ─────────────────────────────────────
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._estimate_tokens(system_msgs + history_msgs)
        if self.valves.context_window_tokens > 0:
            suggestion = await self._check_and_suggest_summarization(
                project_id, total_tokens, self.valves.context_window_tokens
            )
            if suggestion:
                messages.insert(0, {"role": "system", "content": suggestion})
                body["messages"] = messages

        cmd_suggestion = await self._suggest_commands(project_id, state)
        if cmd_suggestion:
            messages.insert(0, {"role": "system", "content": cmd_suggestion})
            body["messages"] = messages

        # ── Adaptive context trim ─────────────────────────────────────
        trim_needed = False
        if self.valves.adaptive_trim:
            total_tokens = self._estimate_tokens(system_msgs + history_msgs)
            if total_tokens > self.valves.context_window_tokens:
                trim_needed = True
        else:
            user_max = (
                __user__["valves"].max_turns
                if __user__ and hasattr(__user__, "valves")
                else None
            )
            eff_max = user_max if user_max is not None else self.valves.max_turns
            if len(history_msgs) > eff_max:
                trim_needed = True

        if trim_needed or len(history_msgs) > self.valves.max_turns:
            self._log_debug("Trimming old messages")
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
                    history_msgs = [
                        {
                            "role": "assistant",
                            "content": f"[Summary of earlier conversation]\n{summary}",
                        }
                    ] + kept_block
                else:
                    history_msgs = kept_block
            else:
                history_msgs = kept_block

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

        messages = system_msgs + history_msgs

        if messages and messages[-1].get("role") != "user":
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                messages = messages[: last_user_idx + 1]
                self._log_debug("Trimmed trailing assistant messages")
            else:
                messages.append({"role": "user", "content": "continue"})
                self._log_debug("Inserted dummy user message to satisfy API")

        # debug: benchmark results
        outlet_elapsed = time.time() - outlet_start
        self._log_debug(f"outlet processing time: {outlet_elapsed:.3f}s")

        # return of inlet
        body["messages"] = messages
        return body

    # --------------------------------------------------------------------------
    #  MODIFIED: outlet
    # --------------------------------------------------------------------------
    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("outlet called")
        if not (HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness):
            return body
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
                if last_msg.get("role") in ("user", "assistant"):
                    if is_code_session:
                        await self._update_active_code(last_msg, project_id)
                        await self._store_message_in_memory(last_msg, project_id)
                    else:
                        if not self.valves.ltm_store_only_code_sessions:
                            await self._store_message_in_memory(last_msg, project_id)
        if self.valves.enable_response_cache and HAS_SENTENCE and len(messages) >= 2:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"), None
            )
            if last_user and last_assistant:
                context_hash = self._compute_context_hash(messages[:-1])
                await self._store_response_in_cache(
                    last_user.get("content", ""),
                    last_assistant.get("content", ""),
                    context_hash,
                    state,
                )
        asyncio.create_task(self._purge_expired_memories())
        self._clean_llm_cache()

        self._write_counter += 1
        if self._write_counter % 100 == 0:
            asyncio.create_task(self._run_db_checkpoints())
            self._cleanup_completed_tasks()

        return body

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

    async def _store_message_in_memory(self, message: dict, project_id: str):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        content = message.get("content", "")
        if not content or len(content.strip()) < 15:
            return
        extracted = await self._extract_code_blocks(content)
        content_type = self._classify_content(content, extracted)
        msg_id = f"{project_id}_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        expires_at = None
        if self.valves.long_term_memory_expiration_days > 0:
            expires_at = time.time() + (
                self.valves.long_term_memory_expiration_days * 86400
            )
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(content).tolist()
        )
        await anyio.to_thread.run_sync(
            lambda: self.memory_collection.upsert(
                ids=[msg_id],
                embeddings=[embedding],
                metadatas=[
                    {
                        "role": message.get("role"),
                        "project_id": project_id,
                        "timestamp": time.time(),
                        "expires_at": expires_at,
                        "content_type": content_type.value,
                        "has_code": len(extracted) > 0,
                    }
                ],
                documents=[content],
            )
        )
        self._log_debug(f"Stored message {msg_id} in LTM")
        state = self._get_state(project_id)
        msg_count = state.get("message_count", 0)
        if msg_count > 0 and msg_count % self.valves.ltm_compress_after_messages == 0:
            asyncio.create_task(self._compress_ltm_for_conversation(project_id))

    async def _compress_ltm_for_conversation(self, project_id: str):
        if not HAS_AIOHTTP or not self.memory_collection:
            return
        try:
            results = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.get(
                    where={"$and": [{"project_id": {"$eq": project_id}}]}
                )
            )
            if (
                not results
                or len(results["ids"]) < self.valves.ltm_compress_after_messages
            ):
                return
            ids = results["ids"]
            docs = results["documents"]
            metadatas = results["metadatas"]
            pairs = sorted(
                zip(ids, docs, metadatas), key=lambda x: x[2].get("timestamp", 0)
            )
            to_compress = pairs[: max(len(pairs) // 3, 5)]
            if len(to_compress) < 2:
                return

            max_chars = 3000
            prompt_template = (
                "Summarise the following conversation segment, keeping key technical "
                "decisions and code changes:\n\n{text}"
            )
            overhead = len(prompt_template.format(text=""))

            batches = []
            current_batch = []
            current_len = 0
            for entry in to_compress:
                entry_len = len(entry[1])
                if current_len + entry_len > max_chars - overhead and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_len = 0
                current_batch.append(entry)
                current_len += entry_len
            if current_batch:
                batches.append(current_batch)

            all_summaries = []
            ids_to_delete = []
            for batch in batches:
                texts = "\n---\n".join([doc for _, doc, _ in batch])
                prompt = prompt_template.format(text=texts[: max_chars - overhead])
                summary = await self._call_llm(
                    prompt=prompt,
                    system_prompt="You produce concise, information-dense summaries.",
                    max_tokens=500,
                    temperature=0.3,
                )
                if summary:
                    all_summaries.append(summary)
                    ids_to_delete.extend([id for id, _, _ in batch])

            if not ids_to_delete:
                return

            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.delete(ids=ids_to_delete)
            )

            combined_summary = "\n---\n".join(all_summaries)
            if len(combined_summary) > 2000:
                combined_summary = combined_summary[:2000] + "\n[summary truncated]"

            summary_id = f"{project_id}_summary_{int(time.time())}"
            summary_embedding = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(combined_summary).tolist()
            )
            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.upsert(
                    ids=[summary_id],
                    embeddings=[summary_embedding],
                    metadatas=[
                        {
                            "project_id": project_id,
                            "is_summary": True,
                            "timestamp": time.time(),
                        }
                    ],
                    documents=[combined_summary],
                )
            )
            self._log_debug(
                f"Compressed {len(ids_to_delete)} messages into summary for {project_id}"
            )
        except Exception as e:
            logger.warning(f"LTM compression failed: {e}")

    async def _retrieve_all_memories_unified(
        self, query: str, project_id: str
    ) -> List[Dict[str, Any]]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []
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
                    n_results=self.valves.long_term_memory_top_k * 3,
                    where=where_filter,
                )
            )
            docs_with_meta = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    sim = 1 - results["distances"][0][i]
                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None
                    if self.valves.ltm_time_decay_hours > 0 and ts is not None:
                        age_hours = (now - ts) / 3600
                        sim *= 0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                    if sim >= self.valves.long_term_memory_similarity_threshold:
                        docs_with_meta.append((doc, sim, ts, meta))

            if self.valves.preserve_error_context:
                for i, (doc, sim, ts, meta) in enumerate(docs_with_meta):
                    if meta.get("content_type") == ContentType.ERROR.value:
                        docs_with_meta[i] = (doc, sim * 1.1, ts, meta)

            docs_with_meta.sort(key=lambda x: x[1], reverse=True)

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
            return [{"doc": doc, "timestamp": ts} for doc, _, ts in docs_with_meta]
        except Exception as e:
            logger.warning(f"Unified memory retrieval failed: {e}")
            return []

    async def _retrieve_historical_messages(
        self, query: str, project_id: str, limit: int
    ) -> List[Dict]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []
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

            regular_messages = []
            summary_messages = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    is_summary = meta.get("is_hierarchical_summary", False)
                    role = meta.get("role", "user")
                    if is_summary:
                        summary_messages.append({"role": "assistant", "content": doc})
                    else:
                        regular_messages.append({"role": role, "content": doc})

            messages = summary_messages + regular_messages

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
        scores = self._cross_encoder.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

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

    async def _summarize_inactive_blocks_safely(self, project_id: str):
        if self._summarize_inactive_in_progress.get(project_id, False):
            self._log_debug(
                f"Summarize inactive already in progress for {project_id}, skipping"
            )
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
            tasks = [self._summarize_code_block(block) for _, block in to_summarize]
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
        if not self.valves.summarize_inactive_code or not HAS_AIOHTTP:
            return None
        sig = self._extract_signature(block.content)
        if sig:
            prompt = f"""The code block has signature: {sig}
Provide a very brief description of what this code does.
Code:
```{block.content[:1000]}```"""
        else:
            prompt = f"""Summarise the following code block.
```{block.content[:1500]}```"""
        return await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant.",
            model_override=self.valves.inactive_code_summary_model,
            max_tokens=200,
            temperature=0.2,
        )

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

    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return "No active context to forget."
            action = intent.get("action")
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
            elif action == "forget_all":
                for block in state["active_blocks"].values():
                    self._symbol_index.remove_all_for_block(
                        block.hash, block.symbols, project_id
                    )
                state["active_blocks"].clear()
                state["recent_changes"].clear()
                state["committed_changes"].clear()
                state["has_any_calls"] = False
                self._invalidate_lightweight_cache(project_id)
                return "Forgotten all context."
            else:
                return "Unrecognized forget action."

    # --------------------------------------------------------------------------
    #  Remember & Obsolete execution
    # --------------------------------------------------------------------------
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
            elif action == "obsolete_all":
                count = set_obsolete(blocks, True)
                return f"Marked all {count} block(s) as obsolete."
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

    # --------------------------------------------------------------------------
    #  Fact management
    # --------------------------------------------------------------------------
    async def _extract_facts_from_message(self, content: str) -> List[str]:
        pattern = r"\[FACT:\s*(.*?)\]"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches]

    async def _add_fact(self, project_id: str, fact_text: str, source: str = "user"):
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return
            expires_at = None
            if self.valves.fact_max_age_days > 0:
                expires_at = time.time() + (self.valves.fact_max_age_days * 86400)
            new_fact = {
                "fact": fact_text,
                "timestamp": time.time(),
                "source": source,
                "expires_at": expires_at,
            }
            for existing in state["facts"]:
                if existing["fact"] == fact_text:
                    return
            state["facts"].append(new_fact)
            if len(state["facts"]) > 100:
                state["facts"] = state["facts"][-100:]
            self._set_state(project_id, state)
            self._log_debug(f"Added fact: {fact_text[:50]}...")

    async def _remove_fact(self, project_id: str, fact_text_or_index: str):
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return
            original_len = len(state["facts"])
            if fact_text_or_index.isdigit():
                idx = int(fact_text_or_index)
                if 0 <= idx < len(state["facts"]):
                    state["facts"].pop(idx)
            else:
                state["facts"] = [
                    f for f in state["facts"] if f["fact"] != fact_text_or_index
                ]
            if len(state["facts"]) != original_len:
                self._set_state(project_id, state)
                self._log_debug(f"Removed fact: {fact_text_or_index}")

    async def _handle_fact_command(
        self, command_text: str, project_id: str
    ) -> Optional[str]:
        if not command_text.startswith(self.valves.fact_command_prefix):
            return None
        parts = command_text.split(maxsplit=2)
        if len(parts) < 2:
            return (
                "**Fact commands:**\n"
                "- `/fact add <text>` – store a fact\n"
                "- `/fact list` – list all facts\n"
                "- `/fact remove <index>` – remove a fact by its number\n"
                "- `/fact clear` – remove all facts"
            )
        subcommand = parts[1].lower()
        if subcommand == "add":
            if len(parts) < 3:
                return "Usage: `/fact add <fact text>`"
            fact_text = parts[2].strip()
            await self._add_fact(project_id, fact_text, source="user")
            return f"Fact added: {fact_text}"
        elif subcommand == "list":
            state = self._get_state(project_id)
            facts = state.get("facts", [])
            if not facts:
                return "No facts stored."
            now = time.time()
            active_facts = [
                f for f in facts if not f.get("expires_at") or f["expires_at"] > now
            ]
            if not active_facts:
                return "All facts have expired."
            lines = ["**Stored facts:**"]
            for i, f in enumerate(active_facts):
                lines.append(f"{i}. {f['fact']}")
            return "\n".join(lines)
        elif subcommand == "remove":
            if len(parts) < 3:
                return "Usage: `/fact remove <index>` (use `/fact list` to see indices)"
            try:
                idx = int(parts[2].strip())
                state = self._get_state(project_id)
                if 0 <= idx < len(state.get("facts", [])):
                    removed = state["facts"].pop(idx)
                    self._set_state(project_id, state)
                    return f"Fact removed: {removed['fact']}"
                else:
                    return (
                        f"Invalid index. There are {len(state.get('facts', []))} facts."
                    )
            except ValueError:
                return "Index must be a number."
        elif subcommand == "clear":
            state = self._get_state(project_id)
            count = len(state.get("facts", []))
            state["facts"] = []
            self._set_state(project_id, state)
            return f"Cleared {count} fact(s)."
        else:
            return f"Unknown subcommand: {subcommand}"

    def _get_facts_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["facts"]:
            return ""
        now = time.time()
        active_facts = []
        for f in state["facts"]:
            if f.get("expires_at") and f["expires_at"] < now:
                continue
            active_facts.append(f["fact"])
        if not active_facts:
            return ""
        return "## Explicitly Agreed Facts\n" + "\n".join(
            [f"- {fact}" for fact in active_facts]
        )

    # --------------------------------------------------------------------------
    #  Intent detection (natural language)
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

    QUESTION_PATTERNS = re.compile(
        r"^\s*(how|what|why|when|where|can you|could you|please|fix|help|"
        r"cómo|qué|por qué|arregla|ayuda|explica)\b",
        re.IGNORECASE,
    )

    def _has_intent_keywords(self, text: str) -> bool:
        """Return True if the text contains any intent-related keyword as a whole word."""
        return bool(
            re.search(
                r"\b(?:"
                + "|".join(re.escape(kw) for kw in self.INTENT_KEYWORDS)
                + r")\b",
                text,
                re.IGNORECASE,
            )
        )

    async def _should_parse_intents(self, user_message: str, code_spans) -> bool:
        cleaned = self._remove_code_spans(user_message, code_spans).strip()
        if len(cleaned) < 15:
            return False
        if not self._has_intent_keywords(cleaned):
            return False
        code_ratio = sum(e - s for s, e in code_spans) / max(len(user_message), 1)
        if code_ratio > 0.6:
            return False
        if self.QUESTION_PATTERNS.match(cleaned):
            return False
        return True

    async def _parse_all_intents(self, user_message: str) -> Dict[str, Any]:
        if not self.valves.enable_natural_language_forget:
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

        code_spans = await self._get_code_spans(user_message)
        if not await self._should_parse_intents(user_message, code_spans):
            none = {"action": "none"}
            return {"forget": none, "remember": none, "obsolete": none}

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
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=200,
            temperature=0.0,
            timeout=8.0,
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
    ) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        tasks = [
            (
                self._detect_contradictions(messages)
                if self.valves.enable_contradiction_detection
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
    #  Diff application and dependency tracking (unchanged from original)
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
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
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
            hunks.append((old_start - 1, old_lines, new_lines))
        applied_any = False
        for old_start, old_lines, new_lines in reversed(hunks):
            if old_start < 0 or old_start + len(old_lines) > len(result_lines):
                logger.warning(
                    f"Unified diff hunk out of bounds (start={old_start}, lines={len(old_lines)}, total={len(result_lines)})"
                )
                continue
            if result_lines[old_start : old_start + len(old_lines)] != old_lines:
                logger.warning(
                    f"Unified diff hunk mismatch at line {old_start}: expected {old_lines[:3]}..., got {result_lines[old_start:old_start+3]}..."
                )
                continue
            result_lines = (
                result_lines[:old_start]
                + new_lines
                + result_lines[old_start + len(old_lines) :]
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

    def _extract_dependencies_ast(self, code: str) -> Tuple[List[str], List[str]]:
        imports = []
        calls = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(
                            f"{module}.{alias.name}" if module else alias.name
                        )
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        except (SyntaxError, MemoryError, RecursionError, ValueError):
            pass
        return list(set(imports)), list(set(calls))

    def _extract_dependencies_regex(self, code: str, language: str) -> List[str]:
        deps = set()
        if language in ("javascript", "typescript", "js", "ts", "jsx", "tsx"):
            for m in re.finditer(r"""from\s+['"]([^'"]+)['"]""", code):
                deps.add(m.group(1))
            for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", code):
                deps.add(m.group(1))
            for m in re.finditer(r"""import\s+['"]([^'"]+)['"]""", code):
                deps.add(m.group(1))
        elif language == "go":
            for m in re.finditer(r'import\s+"([^"]+)"', code):
                deps.add(m.group(1))
        elif language == "rust":
            for m in re.finditer(r"""use\s+([\w:]+)""", code):
                deps.add(m.group(1))
        elif language == "java":
            for m in re.finditer(r"""import\s+([\w.]+)""", code):
                deps.add(m.group(1))
        elif language in ("c", "cpp", "c++"):
            for m in re.finditer(r"""#include\s+[<"]([^>"]+)[>"]""", code):
                deps.add(m.group(1))
        return list(deps)

    async def _extract_dependencies_hybrid(
        self, code: str, file_path: Optional[str] = None
    ) -> List[str]:
        if not self.valves.enable_dependency_tracking:
            return []
        lang = "unknown"
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".jsx": "javascript",
                ".tsx": "typescript",
                ".go": "go",
                ".rs": "rust",
                ".java": "java",
                ".c": "c",
                ".cpp": "cpp",
                ".h": "c",
                ".hpp": "cpp",
            }
            lang = lang_map.get(ext, "unknown")
        else:
            if re.search(r"\bdef\s+\w+\s*\(", code) and re.search(
                r"\bimport\s+\w+", code
            ):
                lang = "python"
            elif re.search(r"\b(function|const|let|var|=>)\b", code):
                lang = "javascript"
        if lang == "python":
            imports, calls = self._extract_dependencies_ast(code)
            return list(set(imports + calls))
        if lang != "unknown":
            deps = self._extract_dependencies_regex(code, lang)
            if deps:
                return deps
        model = (
            self.valves.dependency_extraction_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analyze the following code and extract dependencies...
```\n{code[:1500]}\n```"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output only JSON arrays.",
            model_override=model,
            max_tokens=300,
            temperature=0.1,
        )
        if not response:
            return []
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            deps = json.loads(response)
            if isinstance(deps, list):
                return list(set(deps))
        except:
            pass
        return []

    async def _update_dependencies(self, block_hash: str, state: Dict):
        block = state["active_blocks"].get(block_hash)
        if not block:
            return
        deps = await self._extract_dependencies_hybrid(block.content, block.file_path)
        if (
            block.file_path
            and block.file_path.endswith(".py")
            or "def " in block.content
        ):
            imports, calls = self._extract_dependencies_ast(block.content)
            block.ast_imports = imports
            block.ast_calls = calls
        block.dependencies = deps

    async def _mark_affected_blocks(self, changed_hash: str, state: Dict):
        changed_block = state["active_blocks"].get(changed_hash)
        if not changed_block:
            return
        affected_identifiers = set()
        if changed_block.file_path:
            base = os.path.splitext(os.path.basename(changed_block.file_path))[0]
            affected_identifiers.add(base)
            affected_identifiers.add(changed_block.file_path)
        sig = self._extract_signature(changed_block.content)
        if sig:
            name_match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)", sig)
            if name_match:
                affected_identifiers.add(name_match.group(1))
        for h, block in state["active_blocks"].items():
            if h == changed_hash:
                continue
            block_deps = (
                block.dependencies
                + getattr(block, "ast_imports", [])
                + getattr(block, "ast_calls", [])
            )
            if any(dep in affected_identifiers for dep in block_deps):
                block.potentially_affected = True
                block.affected_timestamp = time.time()
                block._update_importance()

    async def _refresh_dependencies_for_block(self, block_hash: str, project_id: str):
        if not self.valves.enable_dependency_tracking:
            return
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state or block_hash not in state["active_blocks"]:
                return
            await self._update_dependencies(block_hash, state)
            await self._mark_affected_blocks(block_hash, state)
            self._set_state(project_id, state)

    async def _clean_affected_flags(self, project_id: str):
        if not self.valves.enable_dependency_tracking:
            return
        lock = await self._get_project_lock(project_id)
        async with lock:
            state = self._get_state(project_id)
            if not state:
                return
            now = time.time()
            decay = self.valves.affected_decay_hours * 3600
            if decay <= 0:
                return
            changed = False
            for block in state["active_blocks"].values():
                if (
                    block.potentially_affected
                    and (now - block.affected_timestamp) > decay
                ):
                    block.potentially_affected = False
                    block._update_importance()
                    changed = True
            if changed:
                self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    #  Oversized code block handling
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
            summary = await self._call_llm(
                prompt=f"Summarize the following {language} code block.\n{header}First part of code:\n```{language}\n{code[:8000]}\n```",
                system_prompt="You are a code summarization assistant.",
                model_override=model,
                max_tokens=self.valves.oversized_summary_max_tokens,
                temperature=0.2,
            )
            return (
                f"[Automatic summary of a {estimated} token code block]\n{summary}"
                if summary
                else f"[Code block too large, could not summarize] Original size: {estimated} tokens."
            )
        elif action == "warn":
            return self.valves.code_block_warn_message
        return code

    # --------------------------------------------------------------------------
    #  Duplicate removal and hierarchical compression
    # --------------------------------------------------------------------------
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

    async def _hierarchical_compress(self, project_id: str, state: Dict):
        if self._hierarchical_compress_in_progress.get(project_id, False):
            self._log_debug(
                f"Hierarchical compress already in progress for {project_id}, skipping"
            )
            return
        self._hierarchical_compress_in_progress[project_id] = True

        try:
            if not self.valves.hierarchical_compression_enabled:
                return
            if not self.memory_collection:
                return
            last_ts = state.get("last_compression_timestamp", 0)
            if time.time() - last_ts < 3600:
                return

            now = time.time()
            where_filter = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"is_hierarchical_summary": {"$ne": True}},
                    {"timestamp": {"$lt": now}},
                ]
            }
            results = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.get(
                    where=where_filter,
                    include=["documents", "metadatas", "ids"],
                    limit=self.valves.hierarchical_compression_interval_messages * 2,
                )
            )

            if (
                not results
                or not results["ids"]
                or len(results["ids"])
                < self.valves.hierarchical_compression_interval_messages
            ):
                return

            pairs = sorted(
                zip(results["ids"], results["documents"], results["metadatas"]),
                key=lambda x: x[2].get("timestamp", 0),
            )
            to_compress = pairs[
                : self.valves.hierarchical_compression_interval_messages
            ]

            max_chars = 4000
            prompt_template = (
                "Summarise the following conversation segment, keeping key technical "
                "decisions and code changes:\n\n{text}"
            )
            overhead = len(prompt_template.format(text=""))

            batches = []
            current_batch = []
            current_len = 0
            for entry in to_compress:
                entry_len = len(entry[1])
                if current_len + entry_len > max_chars - overhead and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_len = 0
                current_batch.append(entry)
                current_len += entry_len
            if current_batch:
                batches.append(current_batch)

            all_summaries = []
            ids_to_delete = []
            model = (
                self.valves.hierarchical_summary_model
                or self.valves.llm_model
                or self.valves.summarization_model
            )
            for batch in batches:
                texts = "\n---\n".join([doc for _, doc, _ in batch])
                prompt = prompt_template.format(text=texts[: max_chars - overhead])
                summary = await self._call_llm(
                    prompt=prompt,
                    system_prompt="You are a code-aware assistant that produces concise, information-dense summaries.",
                    model_override=model,
                    max_tokens=self.valves.hierarchical_summary_max_tokens,
                    temperature=0.2,
                )
                if summary:
                    all_summaries.append(summary)
                    ids_to_delete.extend([id for id, _, _ in batch])

            if not ids_to_delete:
                return

            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.delete(ids=ids_to_delete)
            )

            combined_summary = "\n---\n".join(all_summaries)
            if len(combined_summary) > 4000:
                combined_summary = combined_summary[:4000] + "\n[summary truncated]"

            summary_id = f"{project_id}_hierarchical_{int(time.time())}"
            summary_embedding = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(combined_summary).tolist()
            )
            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.upsert(
                    ids=[summary_id],
                    embeddings=[summary_embedding],
                    metadatas=[
                        {
                            "role": "assistant",
                            "project_id": project_id,
                            "timestamp": time.time(),
                            "is_hierarchical_summary": True,
                            "summary_level": 1,
                        }
                    ],
                    documents=[f"[Hierarchical summary]\n{combined_summary}"],
                )
            )

            state["last_compression_timestamp"] = time.time()
            self._set_state(project_id, state)

            self._log_debug(f"Hierarchical compression completed for {project_id}")

        except Exception as e:
            self._log_debug(f"Error in hierarchical_compress: {e}")
        finally:
            self._hierarchical_compress_in_progress[project_id] = False


# endregion (end of Filter class)
