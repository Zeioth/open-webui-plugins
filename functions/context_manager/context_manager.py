"""
title: Code-Aware Context Manager with LTM & Summarization
description: Full-featured context manager for coding assistants. Persists state per project, tracks line ranges, applies diffs, compresses LTM, scores importance, learns from responses, summarizes inactive code, supports manual markers, natural language forget/remember commands, feedback tracking, hierarchical memory, LRU cache, optional reranking, dependency detection (AST for Python + regex for other languages), handling of oversized blocks, smart context selection, hierarchical compression, duplicate removal, frequency prioritization, selective summarization, iterative commands, consecutive message deduplication, contradiction detection, chain-of-thought reasoning, assumption extraction, obsolete marking, proactive suggestions, duplicate question detection, command suggestions, semantic response caching, raw file priority boost, LTM retrieval token limit, and lightweight signature-based context with call graphs and summaries for massive code injections.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.5.4
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

try:
    import tree_sitter

    HAS_TREE_SITTER_CORE = True
except ImportError:
    HAS_TREE_SITTER_CORE = False

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

    symbols: List[CodeSymbol] = Field(default_factory=list)
    _cached_token_count: int = 0  # not serialized

    last_mentioned_msg_idx: Optional[int] = None  # for proactive cleanup tracking

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
# ---------------------------------------------------------------------------
class SignatureExtractor:
    MAX_PARSE_SIZE_BYTES = 5_000_000  # 5 MB

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
    def _parse_sync(code_bytes: bytes, lang: str):
        """Creates a new parser in the current thread and parses the code."""
        from tree_sitter import (
            Parser as TSParser,
        )  # local import to avoid global dependencies.

        try:
            lang_obj = get_language(lang)
            parser = TSParser()
            parser.set_language(lang_obj)
            return parser.parse(code_bytes)
        except Exception as e:
            raise RuntimeError(f"Tree‑sitter parse error: {e}")

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
            # Llama a _parse_sync, que crea un parser NUEVO dentro del hilo,
            # evitando el error de "unsendable".
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
    def _extract_generic(
        code: str, file_path: Optional[str] = None
    ) -> List[CodeSymbol]:
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
        auto_cot_enabled: bool = Field(default=False)
        auto_cot_min_chars: int = Field(default=200)
        enable_code_review_mode: bool = Field(default=True)
        cot_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        cot_max_tokens: int = Field(default=1000)
        cot_model_level2: str = Field(
            default="ollama/llama3.2:3b",
            description="Model used for CoT level 2 (auto-reasoning).",
        )
        cot_model_level3: str = Field(
            default="ollama/llama3.2:3b",
            description="Model used for CoT level 3 (self-reflection).",
        )
        enable_cot_llm_detection: bool = Field(
            default=True,
            description="If true, uses a lightweight LLM to decide the CoT level instead of static keywords.",
        )
        cot_detection_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Model used to detect the appropriate CoT level when enable_cot_llm_detection is active.",
        )

        enable_assumption_extraction: bool = Field(default=False)
        assumption_extraction_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        enable_contradiction_detection: bool = Field(default=False)
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
        similar_message_threshold: float = Field(default=0.92)
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
        recent_activity_window_minutes: int = Field(
            default=15,
            description="How many minutes back to consider a file 'recently modified' in the context header.",
        )

        summarize_inactive_code: bool = Field(default=True)
        inactive_code_summary_model: str = Field(default="ollama/llama3.2:3b")

        llm_model: str = Field(default="ollama/llama3.2:3b")

        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=False)
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
            default=100, description="Maximum number of cached LLM responses in RAM."
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

        # Symbolic LTM indexing and retrieval
        ltm_index_symbols_enabled: bool = Field(default=False)
        ltm_symbol_index_max_per_message: int = Field(default=20)
        ltm_symbol_boost_enabled: bool = Field(default=False)
        ltm_symbol_boost_factor: float = Field(default=1.5)
        ltm_symbol_boost_min_similarity: float = Field(default=0.5)
        ltm_symbol_force_mode_enabled: bool = Field(default=False)
        ltm_symbol_force_fallback_to_semantic: bool = Field(default=True)

        # Proactive cleanup
        cleanup_suggestions_enabled: bool = Field(default=False)
        cleanup_inactive_threshold_messages: int = Field(default=30)
        cleanup_excluded_content_types: list = Field(
            default_factory=lambda: ["BASE_CODE"]
        )
        cleanup_status_command_enabled: bool = Field(default=True)
        cleanup_proactive_suggestions: bool = Field(default=False)
        cleanup_suggestion_cooldown_messages: int = Field(default=20)
        cleanup_command_enabled: bool = Field(default=True)

        # Speculative preload
        speculative_log_missed_opportunities: bool = Field(default=False)
        speculative_preload_enabled: bool = Field(default=False)
        speculative_preload_max_tokens_percent: float = Field(default=0.10)
        speculative_preload_max_dependencies: int = Field(default=2)
        speculative_preload_min_callers: int = Field(default=1)
        speculative_adaptive: bool = Field(default=False)
        speculative_max_limit: int = Field(default=5)
        speculative_boost_on_miss: float = Field(default=1.0)
        speculative_decay_after_ignored_turns: int = Field(default=3)
        speculative_stats_command_enabled: bool = Field(default=True)
        active_context_max_tokens: int = Field(
            default=0,
            description="Maximum tokens for the injected active code context. 0 = unlimited.",
        )
        duplicate_question_lookback_hours: float = Field(
            default=24.0
        )  # <-- Only last 24h
        expand_default_depth: int = Field(
            default=2, description="Default depth for /expand command"
        )
        max_change_summaries: int = Field(
            default=1000,
            description="Maximum number of block change summaries kept in memory per project.",
        )
        global_injection_token_budget: int = Field(
            default=0,
            description="Maximum tokens allowed for all system injections combined (0 = unlimited). Set to e.g. 30% of context window.",
        )

    class UserValves(BaseModel):
        max_turns: Optional[int] = Field(default=None)
        enable_code_awareness: Optional[bool] = Field(default=None)

    _SYMBOL_BLACKLIST = {
        "self",
        "cls",
        "args",
        "kwargs",
        "init",
        "main",
        "len",
        "print",
        "range",
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "object",
        "type",
        "super",
        "i",
        "j",
        "k",
        "x",
        "y",
        "z",
        "e",
        "ex",
        "err",
        "error",
        "data",
        "result",
        "value",
        "key",
        "item",
        "items",
        "func",
        "method",
        "function",
        "class",
        "return",
        "pass",
        "break",
        "continue",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "import",
        "from",
        "def",
        "lambda",
        "yield",
        "raise",
        "assert",
        "and",
        "or",
        "not",
        "in",
        "is",
        "None",
        "True",
        "False",
    }

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
            "last_cleanup_suggestion_msg_idx": 0,
            "speculative_missed_stats": {"total": 0, "details": {}},
            "speculative_preload_limit": None,
            "speculative_miss_count_since_last_boost": 0,
            "speculative_ignored_turns": 0,
            "speculative_last_preloaded_symbols": set(),
            "speculative_last_preload_turn": 0,
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
        self._session_classify_ttl: float = 1800.0

        self._symbol_index = SymbolIndex()
        self._cached_lightweight_context: Dict[str, str] = {}
        self._last_processed_message_idx: Dict[str, int] = {}
        self._last_project_id: str = ""
        self._response_cache_size: int = 0
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        self._block_change_summaries: Dict[str, Tuple[str, float]] = (
            {}
        )  # hash -> (summary, timestamp)

        print("[CodeAware] Filter loaded")

    # --------------------------------------------------------------------------
    #  Helper methods
    # --------------------------------------------------------------------------
    def _log_debug(self, msg: str):
        if self.valves.debug:
            print(f"[CodeAware] {msg}")
            logger.info(msg)

    def _log_timing(self, step_name: str, elapsed_since_start: float, duration: float):
        """Log timing information for a step when debug is enabled."""
        if self.valves.debug:
            self._log_debug(
                f"[Timing] {step_name}: +{elapsed_since_start:.3f}s (dur={duration:.3f}s)"
            )

    # --------------------------------------------------------------------------
    #  Code span utilities
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

    # --------------------------------------------------------------------------
    #  Project locks and state management
    # --------------------------------------------------------------------------
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
        # Serialize only the necessary parts, handling CodeBlock -> dict and other objects
        active_blocks = {}
        for k, v in state["active_blocks"].items():
            d = v.dict()
            # Convert Enum to value for JSON
            d["content_type"] = v.content_type.value
            active_blocks[k] = d
        serializable = {
            "active_blocks": active_blocks,
            "recent_changes": [b.dict() for b in state["recent_changes"]],
            "committed_changes": [b.dict() for b in state["committed_changes"]],
            "feedback_history": [fb.dict() for fb in state["feedback_history"]],
            "facts": state.get("facts", []),
            "iterative_state": state.get("iterative_state"),
            "message_count": state["message_count"],
            "last_compression_timestamp": state.get("last_compression_timestamp", 0),
            "response_cache": state.get("response_cache", []),
            "last_suggestion_timestamp": state.get("last_suggestion_timestamp", 0),
            "last_cleanup_suggestion_msg_idx": state.get(
                "last_cleanup_suggestion_msg_idx", 0
            ),
            "speculative_missed_stats": state.get(
                "speculative_missed_stats", {"total": 0, "details": {}}
            ),
            "speculative_preload_limit": state.get("speculative_preload_limit"),
            "speculative_miss_count_since_last_boost": state.get(
                "speculative_miss_count_since_last_boost", 0
            ),
            "speculative_ignored_turns": state.get("speculative_ignored_turns", 0),
            "speculative_last_preloaded_symbols": list(
                state.get("speculative_last_preloaded_symbols", set())
            ),
            "speculative_last_preload_turn": state.get(
                "speculative_last_preload_turn", 0
            ),
            "has_any_calls": state.get("has_any_calls", False),
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
        # Fill in missing keys with defaults
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
        data.setdefault("last_cleanup_suggestion_msg_idx", 0)
        data.setdefault("speculative_missed_stats", {"total": 0, "details": {}})
        data.setdefault("speculative_preload_limit", None)
        data.setdefault("speculative_miss_count_since_last_boost", 0)
        data.setdefault("speculative_ignored_turns", 0)
        data.setdefault("speculative_last_preloaded_symbols", set())
        data.setdefault("speculative_last_preload_turn", 0)
        active = {}
        for k, v in data.get("active_blocks", {}).items():
            try:
                # Convert content_type back from string to enum
                v["content_type"] = (
                    ContentType(v["content_type"])
                    if "content_type" in v
                    else ContentType.GENERAL
                )
                blk = CodeBlock(**v)
                # If loaded from old state, set message index to current count to avoid immediate cleanup
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
            "facts": data.get("facts", []),
            "iterative_state": data.get("iterative_state"),
            "message_count": data.get("message_count", 0),
            "last_compression_timestamp": data.get("last_compression_timestamp", 0),
            "response_cache": data.get("response_cache", []),
            "last_suggestion_timestamp": data.get("last_suggestion_timestamp", 0),
            "has_any_calls": data.get("has_any_calls", False),
            "last_cleanup_suggestion_msg_idx": data.get(
                "last_cleanup_suggestion_msg_idx", 0
            ),
            "speculative_missed_stats": data.get(
                "speculative_missed_stats", {"total": 0, "details": {}}
            ),
            "speculative_preload_limit": data.get("speculative_preload_limit"),
            "speculative_miss_count_since_last_boost": data.get(
                "speculative_miss_count_since_last_boost", 0
            ),
            "speculative_ignored_turns": data.get("speculative_ignored_turns", 0),
            "speculative_last_preloaded_symbols": set(
                data.get("speculative_last_preloaded_symbols", [])
            ),
            "speculative_last_preload_turn": data.get(
                "speculative_last_preload_turn", 0
            ),
        }
        # Recalculate cached token counts if tokenizer available
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

    # --------------------------------------------------------------------------
    #  Long-term memory initialization
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    #  LLM call utilities
    # --------------------------------------------------------------------------
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
                        except asyncio.CancelledError:
                            raise  # dejar que se propague
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
                        except Exception:
                            if attempt < max_retries:
                                await asyncio.sleep(base_delay * (2**attempt))
                                continue
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
                                        await asyncio.sleep(delay)
                                        continue
                                else:
                                    break
                    except asyncio.CancelledError:
                        raise  # propagar cancelación
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt < max_retries:
                            await asyncio.sleep(base_delay * (2**attempt))
                            continue
            logger.warning(f"All LLM models failed for prompt: {prompt[:100]}...")
            future.set_result(None)
            return None
        except asyncio.CancelledError:
            future.cancel()
            raise  # importante: re‑lanzar después de cancelar el futuro
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

    # --------------------------------------------------------------------------
    #  Response cache
    # --------------------------------------------------------------------------
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
            to_delete = [
                results["ids"][i]
                for i, meta in enumerate(results["metadatas"])
                if now - meta.get("timestamp", 0) > ttl
            ]
            if to_delete:
                await anyio.to_thread.run_sync(
                    lambda: self._response_cache_collection.delete(ids=to_delete)
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

        t_start = time.monotonic()

        # ── 1. Generate embedding ──
        t_emb = time.monotonic()
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
        emb_dur = time.monotonic() - t_emb
        self._log_timing("resp_cache_embedding", emb_dur, emb_dur)

        # ── 2. Prepare entry ──
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

        # ── 3. Upsert into ChromaDB ──
        t_db = time.monotonic()
        await anyio.to_thread.run_sync(
            lambda: col.upsert(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[response],
                metadatas=[
                    {
                        "query": query[:500],
                        "project_id": self.valves.project_id,
                        "context_hash": "",
                        "code_state_hash": self._compute_code_state_hash(
                            self.valves.project_id
                        ),
                        "timestamp": time.time(),
                    }
                ],
            )
        )
        db_dur = time.monotonic() - t_db
        self._log_timing("resp_cache_upsert", db_dur, db_dur)

        self._response_cache_size += 1

        total_dur = time.monotonic() - t_start
        self._log_timing("resp_cache_total", total_dur, total_dur)

    async def _find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return None

        t_start = time.monotonic()

        # ── 1. Generate embedding ──
        t_emb = time.monotonic()
        query_vec = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        )
        emb_dur = time.monotonic() - t_emb
        self._log_timing("resp_cache_query_embedding", emb_dur, emb_dur)

        # ── 2. Query ChromaDB ──
        t_db = time.monotonic()
        results = await anyio.to_thread.run_sync(
            lambda: col.query(
                query_embeddings=[query_vec],
                n_results=1,
                where={"project_id": self.valves.project_id},
                include=["documents", "metadatas", "distances"],
            )
        )
        db_dur = time.monotonic() - t_db
        self._log_timing("resp_cache_query_chromadb", db_dur, db_dur)

        if not results or not results["ids"] or not results["ids"][0]:
            return None

        dist = results["distances"][0][0]
        similarity = 1.0 - (dist / 2.0)
        if similarity < self.valves.response_cache_similarity_threshold:
            return None

        meta = results["metadatas"][0][0]

        # ── 3. Validate code state hash ──
        stored_code_state = meta.get("code_state_hash", "")
        if stored_code_state and stored_code_state != self._compute_code_state_hash(
            self.valves.project_id
        ):
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        # ── 4. Check TTL ──
        ttl = self.valves.response_cache_ttl_hours * 3600
        ts = meta.get("timestamp", 0)
        if ttl > 0 and time.time() - ts > ttl:
            await anyio.to_thread.run_sync(
                lambda: col.delete(ids=[results["ids"][0][0]])
            )
            return None

        doc = results["documents"][0][0]

        total_dur = time.monotonic() - t_start
        self._log_timing("resp_cache_find_total", total_dur, total_dur)

        return {"response": doc, "query": meta.get("query", ""), "timestamp": ts}

    def _compute_code_state_hash(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        sorted_hashes = sorted(
            h for h, b in state["active_blocks"].items() if not b.obsolete
        )
        return hashlib.md5("|".join(sorted_hashes).encode()).hexdigest()[:16]

    # --------------------------------------------------------------------------
    #  Message ordering / trimming helpers
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

    # --------------------------------------------------------------------------
    #  Code extraction and classification
    # --------------------------------------------------------------------------
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
                ts_blocks = await anyio.to_thread.run_sync(
                    lambda: process(content, config)
                )
                for tsb in ts_blocks:
                    start, end = tsb.start_byte, tsb.end_byte
                    raw = content[start:end].strip()
                    lang = tsb.language or "text"

                    if lang == "text" or lang == "":
                        guessed = SignatureExtractor._guess_language(None, raw)
                        if guessed != "unknown":
                            lang = guessed

                    if lang == "text" or lang == "":
                        llm_lang = await self._detect_language_via_llm(raw)
                        if llm_lang != "unknown":
                            lang = llm_lang
                            self._log_debug(f"LLM detected language: {lang}")
                        else:
                            self._log_debug(
                                "LLM language detection also failed; block will be treated as generic text"
                            )

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

                return blocks, spans

            except Exception as e:
                if "Language '' not available" not in str(e):
                    self._log_debug(
                        f"Tree‑sitter extraction unexpectedly failed, using regex fallback: {e}"
                    )

        # ====== Fallback por regex, movido a ejecutor ======
        def _regex_fallback(content):
            blocks, spans = [], []
            for match in self.code_pattern.finditer(content):
                lang = match.group(1) or "text"
                code = match.group(2).strip()
                # No podemos await aquí, lo haremos después
                blocks.append({"language": lang, "code": code, "type": "fenced"})
                spans.append((match.start(), match.end()))
            # Indented blocks
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
                end_offset = (
                    line_offsets[-1] - 1 if line_offsets[-1] > 0 else len(content)
                )
                spans.append((start_offset, end_offset))
            return blocks, spans

        # Ejecutar en hilo para no bloquear
        blocks, spans = await anyio.to_thread.run_sync(_regex_fallback, content)
        # Ahora procesar oversized blocks para cada uno (necesitan await)
        for i in range(len(blocks)):
            blocks[i]["code"] = await self._handle_oversized_code_block(
                blocks[i]["code"], blocks[i]["language"]
            )
        return blocks, spans

    async def _detect_language_via_llm(self, code_snippet: str) -> str:
        """Use a small, quick LLM call to identify the programming language."""
        if not HAS_AIOHTTP:
            return "unknown"
        prompt = (
            "Identify the programming language of this code. "
            "Answer with a single word (e.g., 'python', 'javascript', 'go', 'rust') or 'unknown':\n\n"
            f"```\n{code_snippet[:500]}\n```"
        )
        try:
            response = await asyncio.wait_for(
                self._call_llm(
                    prompt=prompt,
                    system_prompt="You are a programming language detector. Answer with only the language name or 'unknown'.",
                    model_override=self.valves.natural_language_forget_model
                    or self.valves.llm_model,
                    max_tokens=10,
                    temperature=0.0,
                ),
                timeout=3.0,
            )
            if response:
                lang = response.strip().lower()
                # Normalize common responses
                lang_map = {
                    "py": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "tsx": "tsx",
                    "jsx": "jsx",
                    "c++": "cpp",
                    "c#": "csharp",
                    "bash": "bash",
                    "sh": "bash",
                    "zsh": "bash",
                    "rb": "ruby",
                    "rs": "rust",
                    "golang": "go",
                }
                return lang_map.get(lang, lang)
        except (asyncio.TimeoutError, Exception):
            pass
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

    # --------------------------------------------------------------------------
    #  Context formatting
    # --------------------------------------------------------------------------
    def _format_block_context(self, block: CodeBlock, is_latest: bool = False) -> str:
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
        aff = " [AFFECTED BY DEPENDENCY CHANGE]" if block.potentially_affected else ""
        return f"```\n{block.content[:600]}\n```{loc}{latest}  (importance: {block.importance_score:.1f}, modified: {timestamp_str}){aff}{pin}{raw}"

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

        # ---- Calculate recent files ---
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

        # ---- Relevance by mention of files/symbols ----
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

        # ---- File's last version ----
        latest_per_file = {}
        for b in active:
            if b.file_path:
                if (
                    b.file_path not in latest_per_file
                    or b.timestamp > latest_per_file[b.file_path].timestamp
                ):
                    latest_per_file[b.file_path] = b
        latest_hashes = {b.hash for b in latest_per_file.values()}

        # ---- Construction of sections (ordered importance) ----
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
            "> **Note**: If multiple versions of a file appear, the one marked [LATEST] is the most recent and should be used. Older versions are retained for reference only.\n",
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

        # Token limit
        max_tokens = self.valves.active_context_max_tokens
        if max_tokens > 0 and self.tokenizer:
            # Removal priority: errors (lowest), proposed changes, committed changes, base code (highest).
            # Build the final text from parts; if it exceeds the limit, trim from the end (least important parts added last).
            # We've added sections in order: base, proposed, committed, errors (errors last), so we can truncate from the back.
            full_text = "\n".join(parts)
            while len(self.tokenizer.encode(full_text)) > max_tokens and len(parts) > 3:
                # Remove the last section (the least important in order of addition).
                # Strategy: remove errors first, then committed, etc.
                # Simply pop the last element of `parts`, which corresponds to the most recently added section.
                parts.pop()
                full_text = "\n".join(parts)
            if len(self.tokenizer.encode(full_text)) > max_tokens:
                # If it still exceeds the limit, truncate the remaining block (likely base code).
                # Leave a warning message.
                parts.append(f"[Context truncated to fit token limit ({max_tokens})]")
                full_text = "\n".join(parts)
        return "\n".join(parts)

    # --------------------------------------------------------------------------
    #  Lightweight context and symbol index
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

    async def _build_lightweight_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        if project_id in self._cached_lightweight_context:
            return self._cached_lightweight_context[project_id]

        lines = ["## Code Symbol Index (full bodies available on request)\n"]
        called_by: Dict[str, Set[str]] = defaultdict(set)
        for block in state["active_blocks"].values():
            if block.obsolete:
                continue
            for sym in block.symbols:
                for callee in sym.calls:
                    called_by[callee].add(sym.name)

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
                used_by = called_by.get(s.name, set())
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
    #  Active code update and mention tracking
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
                block.last_mentioned_msg_idx = state["message_count"]
                block._update_importance()

    async def _update_active_code(self, message: dict, project_id: str):
        if not self.valves.enable_code_awareness:
            return
        lock = await self._get_project_lock(project_id)
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

        async with lock:
            state = self._get_state(project_id)
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
                    block.last_mentioned_msg_idx = state["message_count"]
                    block._update_importance()

            if not content and not new_blocks_pending:
                return

            for new_block, syms in zip(new_blocks_pending, symbols_list):
                if isinstance(syms, Exception):
                    syms = []
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
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        if prev_content != new_block.content:
                            asyncio.create_task(
                                self._generate_change_summary(
                                    existing.hash, prev_content, new_block.content
                                )
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
                        if self.tokenizer:
                            existing._cached_token_count = len(
                                self.tokenizer.encode(existing.content)
                            )
                        else:
                            existing._cached_token_count = len(existing.content) // 4
                        existing._update_importance()
                        if prev_content != new_block.content:
                            asyncio.create_task(
                                self._generate_change_summary(
                                    existing.hash, prev_content, new_block.content
                                )
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

                for sym in syms:
                    sym.parent_block_hash = new_block.hash
                new_block.symbols = syms
                new_block.last_mentioned_msg_idx = state["message_count"]
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

                state["active_blocks"][new_block.hash] = new_block

                # Auto-obsolete older blocks for the same file unless pinned
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
                        prev_content = best_base.content
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
                        if prev_content != block_info["code"]:
                            asyncio.create_task(
                                self._generate_change_summary(
                                    best_base.hash, prev_content, block_info["code"]
                                )
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
    #  LTM storage and retrieval (with symbolic boost)
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
            self._log_debug(
                f"No memories found for symbol {symbol}, falling back to semantic search."
            )
            return await self._retrieve_all_memories_unified(cleaned_query, project_id)
        return [{"doc": doc, "timestamp": ts} for doc, _, ts in docs_with_meta]

    async def _store_message_in_memory(self, message: dict, project_id: str):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        content = message.get("content", "")
        if not content or len(content.strip()) < 15:
            return

        t_start = time.monotonic()

        # ── 1. Generate embedding ──
        t_emb = time.monotonic()
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(content).tolist()
        )
        emb_dur = time.monotonic() - t_emb
        self._log_timing("store_memory_embedding", emb_dur, emb_dur)

        # ── 2. Extract blocks / classify for metadata ──
        extracted, _ = await self._extract_code_blocks(content)
        content_type = self._classify_content(content, extracted)
        msg_id = f"{project_id}_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        expires_at = None
        if self.valves.long_term_memory_expiration_days > 0:
            expires_at = time.time() + (
                self.valves.long_term_memory_expiration_days * 86400
            )

        code_symbols_str = ""
        if self.valves.ltm_index_symbols_enabled:
            blocks_for_symbols = extracted if extracted else []
            if not blocks_for_symbols:
                blocks_for_symbols, _ = await self._extract_code_blocks(content)
            if blocks_for_symbols:
                all_symbols = set()
                for blk in blocks_for_symbols:
                    try:
                        syms = await SignatureExtractor.extract_async(
                            blk["code"], blk.get("language")
                        )
                        for sym in syms:
                            if self._is_symbol_indexable(sym):
                                all_symbols.add(sym.name)
                                if (
                                    len(all_symbols)
                                    >= self.valves.ltm_symbol_index_max_per_message
                                ):
                                    break
                    except Exception:
                        pass
                if all_symbols:
                    code_symbols_str = "," + ",".join(sorted(all_symbols)) + ","
                    self._log_debug(f"Indexed symbols for LTM: {code_symbols_str}")

        # ── 3. Upsert into ChromaDB ──
        t_db = time.monotonic()
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
                        "code_symbols": code_symbols_str,
                        "memory_id": msg_id,
                    }
                ],
                documents=[content],
            )
        )
        db_dur = time.monotonic() - t_db
        self._log_timing("store_memory_upsert", db_dur, db_dur)

        total_dur = time.monotonic() - t_start
        self._log_timing("store_memory_total", total_dur, total_dur)
        self._log_debug(f"Stored message {msg_id} in LTM")

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
            prompt_template = "Summarise the following conversation segment, keeping key technical decisions and code changes:\n\n{text}"
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

        forced_symbol, cleaned_query = self._parse_forced_symbol_query(query)
        if forced_symbol:
            return await self._retrieve_by_symbol(
                forced_symbol, cleaned_query, project_id
            )

        try:
            t_start = time.monotonic()

            # ── 1. Generate query embedding ──
            t_emb = time.monotonic()
            q_emb = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(query[:1000]).tolist()
            )
            emb_dur = time.monotonic() - t_emb
            self._log_timing("ltm_query_embedding", emb_dur, emb_dur)

            # ── 2. Query ChromaDB ──
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})

            t_db = time.monotonic()
            results = await anyio.to_thread.run_sync(
                lambda: self.memory_collection.query(
                    query_embeddings=[q_emb],
                    n_results=self.valves.long_term_memory_top_k * 3,
                    where=where_filter,
                )
            )
            db_dur = time.monotonic() - t_db
            self._log_timing("ltm_query_chromadb", db_dur, db_dur)

            # ── 3. Process results (scoring, decay, boost, rerank) ──
            docs_with_meta = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    # cosine space normalization
                    sim = 1.0 - (results["distances"][0][i] / 2.0)
                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None

                    # First, filter only by raw similitude
                    if sim < self.valves.long_term_memory_similarity_threshold:
                        continue

                    # Calculate order score with optional decay
                    order_score = sim
                    if self.valves.ltm_time_decay_hours > 0 and ts is not None:
                        age_hours = (now - ts) / 3600
                        order_score = sim * (
                            0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                        )

                    docs_with_meta.append((doc, order_score, ts, meta))

            # Boost for errors (only modifies order score)
            if self.valves.preserve_error_context:
                for i, (doc, order_score, ts, meta) in enumerate(docs_with_meta):
                    if meta.get("content_type") == ContentType.ERROR.value:
                        docs_with_meta[i] = (doc, order_score * 1.1, ts, meta)

            # Order by descending order_score (already includes soft decay)
            docs_with_meta.sort(key=lambda x: x[1], reverse=True)

            # Symbolic boost (applies over order_score)
            if self.valves.ltm_symbol_boost_enabled and query:
                query_symbols = self._extract_query_symbols(query, project_id)
                if query_symbols:
                    new_docs = []
                    for doc, order_score, ts, meta in docs_with_meta:
                        meta_symbols_str = meta.get("code_symbols", "")
                        if (
                            meta_symbols_str
                            and order_score
                            >= self.valves.ltm_symbol_boost_min_similarity
                        ):
                            meta_symbols = set(meta_symbols_str.split(","))
                            common = query_symbols.intersection(meta_symbols)
                            if common:
                                order_score *= self.valves.ltm_symbol_boost_factor
                                self._log_debug(
                                    f"Boosted memory {meta.get('memory_id','?')} with symbols {common}, new sim={order_score:.3f}"
                                )
                        new_docs.append((doc, order_score, ts, meta))
                    new_docs.sort(key=lambda x: x[1], reverse=True)
                    docs_with_meta = new_docs

            # Optional reranking (applies over ordered documents by order_score)
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

            total_dur = time.monotonic() - t_start
            self._log_timing("ltm_retrieval_total_internal", total_dur, total_dur)

            return [{"doc": doc, "timestamp": ts} for doc, _, ts in docs_with_meta]
        except Exception as e:
            logger.warning(f"Unified memory retrieval failed: {e}")
            return []

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
                                self._log_debug(
                                    f"Boosted history memory {meta.get('memory_id','?')} with symbols {common}, sim={sim:.3f}"
                                )

                    role = meta.get("role", "user")
                    is_summary = meta.get("is_hierarchical_summary", False)
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
    #  Block expiration and summarization
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

    # --------------------------------------------------------------------------
    #  Forget / remember / obsolete commands
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
    #  Proactive cleanup commands
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
    #  Speculative preload
    # --------------------------------------------------------------------------
    def _extract_requested_symbols(self, text: str) -> Set[str]:
        if not text:
            return set()
        patterns = [
            r"(?:show|see|view|provide|need|want)\s+(?:me\s+)?(?:the\s+)?(?:code|implementation|definition|source)\s+(?:of|for)\s+`?(\w+)`?",
            r"(?:show|see|view|provide|need|want)\s+(?:me\s+)?`?(\w+)`?\s*(?:function|class|method)?",
            r"(?:could|can)\s+you\s+(?:show|provide)\s+(?:me\s+)?(?:the\s+)?(?:code|implementation)\s+(?:of|for)\s+`?(\w+)`?",
        ]
        symbols = set()
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                symbols.add(m.group(1))
        return symbols

    def _calculate_dependency_interest(
        self, symbol_name: str, project_id: str
    ) -> List[Tuple[str, float]]:
        state = self._get_state(project_id)
        if not state:
            return []
        block_hashes = self._symbol_index.find_blocks(symbol_name, project_id)
        if not block_hashes:
            return []
        block = None
        for h in block_hashes:
            b = state["active_blocks"].get(h)
            if b and not b.obsolete:
                block = b
                break
        if not block:
            return []
        sym = None
        for s in block.symbols:
            if s.name == symbol_name:
                sym = s
                break
        if not sym or not sym.calls:
            return []
        called_by = defaultdict(set)
        for b in state["active_blocks"].values():
            if b.obsolete:
                continue
            for s in b.symbols:
                for callee in s.calls:
                    called_by[callee].add(s.name)
        interest = []
        for callee_name in sym.calls:
            if not self._symbol_index.find_blocks(callee_name, project_id):
                continue
            num_callers = len(called_by.get(callee_name, set()))
            if num_callers < self.valves.speculative_preload_min_callers:
                continue
            callee_hashes = self._symbol_index.find_blocks(callee_name, project_id)
            callee_block = None
            for h in callee_hashes:
                b = state["active_blocks"].get(h)
                if b and not b.obsolete:
                    callee_block = b
                    break
            tokens = callee_block._cached_token_count if callee_block else 100
            score = num_callers / (tokens + 1)
            interest.append((callee_name, score))
        interest.sort(key=lambda x: x[1], reverse=True)
        return interest

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
        non_obsolete_hashes = {
            h
            for h in mentioned_hashes
            if h in state["active_blocks"] and not state["active_blocks"][h].obsolete
        }
        if not non_obsolete_hashes:
            return ""

        sorted_hashes = sorted(
            non_obsolete_hashes,
            key=lambda h: state["active_blocks"]
            .get(h, CodeBlock(content=""))
            .importance_score,
            reverse=True,
        )
        parts = ["\n## Expanded Code Bodies (referenced symbols)\n"]
        expanded_hashes = set()

        for block_hash in sorted_hashes[:MAX_EXPANDED_BODIES]:
            block = state["active_blocks"].get(block_hash)
            if not block:
                continue
            expanded_hashes.add(block_hash)
            loc = f" (file: {block.file_path})" if block.file_path else ""
            parts.append(
                f"### Block {block.hash[:8]}{loc}\n```\n{block.content[:3000]}\n```"
            )
            if block.file_path:
                versions = sorted(
                    [
                        b
                        for b in state["active_blocks"].values()
                        if b.file_path == block.file_path and not b.obsolete
                    ],
                    key=lambda b: b.timestamp,
                    reverse=True,
                )
                if len(versions) > 1:
                    parts.append(f"\n**Version history ({len(versions)} versions):**")
                    for v in versions:
                        ts = datetime.fromtimestamp(
                            v.timestamp, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        entry = self._block_change_summaries.get(v.hash)
                        summary = entry[0] if entry else ""
                        if not summary:
                            first_line = v.content.strip().split("\n")[0][:80]
                            summary = f"(first line: {first_line}...)"
                        latest = " ← current" if v.hash == block_hash else ""
                        parts.append(f"- {ts}: {summary}{latest}")
                    parts.append("")

        # Speculative preload of dependencies
        if self.valves.speculative_preload_enabled and mentioned_names:
            if self.valves.speculative_adaptive:
                limit = state.get("speculative_preload_limit")
                if limit is None:
                    limit = self.valves.speculative_preload_max_dependencies
                    state["speculative_preload_limit"] = limit
            else:
                limit = self.valves.speculative_preload_max_dependencies

            max_preload_tokens = int(
                min(
                    self.valves.context_window_tokens
                    * self.valves.speculative_preload_max_tokens_percent,
                    2000,
                )
            )
            preload_tokens_used = 0
            preload_blocks = []
            preloaded_symbols = set()

            for name in mentioned_names:
                if (
                    preload_tokens_used >= max_preload_tokens
                    or len(preload_blocks) >= limit
                ):
                    break
                interest = self._calculate_dependency_interest(name, project_id)
                for callee_name, score in interest:
                    if len(preload_blocks) >= limit:
                        break
                    callee_hashes = self._symbol_index.find_blocks(
                        callee_name, project_id
                    )
                    for h in callee_hashes:
                        dep_block = state["active_blocks"].get(h)
                        if not dep_block or dep_block.obsolete:
                            continue
                        tokens = dep_block._cached_token_count
                        if preload_tokens_used + tokens > max_preload_tokens:
                            continue
                        if h in expanded_hashes or any(
                            b.hash == h for b in preload_blocks
                        ):
                            continue
                        preload_blocks.append(dep_block)
                        preloaded_symbols.add(callee_name)
                        preload_tokens_used += tokens
                        break

            if self.valves.speculative_adaptive:
                state["speculative_last_preloaded_symbols"] = preloaded_symbols
                state["speculative_last_preload_turn"] = state["message_count"]

            if preload_blocks:
                parts.append("\n### [Preloaded dependencies]\n")
                for dep_block in preload_blocks:
                    loc = (
                        f" (file: {dep_block.file_path})" if dep_block.file_path else ""
                    )
                    parts.append(
                        f"**Preloaded dependency (used by {', '.join(mentioned_names)})** {loc}\n```\n{dep_block.content[:2000]}\n```"
                    )

        return "\n".join(parts)

    # --------------------------------------------------------------------------
    #  Outlet and adaptive speculative feedback
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

        # Speculative miss logging (phase 1)
        if self.valves.speculative_log_missed_opportunities:
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"), None
            )
            if last_assistant:
                requested = self._extract_requested_symbols(
                    last_assistant.get("content", "")
                )
                if requested:
                    existing = set()
                    for sym in requested:
                        if self._symbol_index.find_blocks(sym, project_id):
                            existing.add(sym)
                    if existing:
                        stats = state.setdefault(
                            "speculative_missed_stats", {"total": 0, "details": {}}
                        )
                        stats["total"] += len(existing)
                        for sym in existing:
                            sym_stats = stats["details"].setdefault(sym, {"count": 0})
                            sym_stats["count"] += 1
                        self._set_state(project_id, state)
                        self._log_debug(
                            f"Missed opportunities: assistant asked for {existing}"
                        )

        # Adaptive speculative feedback (phase 3)
        if self.valves.speculative_adaptive and self.valves.speculative_preload_enabled:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user:
                last_preload_turn = state.get("speculative_last_preload_turn", 0)
                last_preloaded = state.get("speculative_last_preloaded_symbols", set())
                if state["message_count"] == last_preload_turn + 1:
                    assistant_response = next(
                        (m for m in reversed(messages) if m.get("role") == "assistant"),
                        None,
                    )
                    if assistant_response:
                        requested = self._extract_requested_symbols(
                            assistant_response.get("content", "")
                        )
                        if requested:
                            hit = last_preloaded.intersection(requested)
                            if hit:
                                new_limit = min(
                                    state.get(
                                        "speculative_preload_limit",
                                        self.valves.speculative_preload_max_dependencies,
                                    )
                                    + self.valves.speculative_boost_on_miss,
                                    self.valves.speculative_max_limit,
                                )
                                state["speculative_preload_limit"] = new_limit
                                state["speculative_ignored_turns"] = 0
                                state["speculative_last_preloaded_symbols"] = set()
                                self._log_debug(
                                    f"Speculative preload hit: {hit}. Boosted limit to {new_limit}."
                                )
                            else:
                                state["speculative_ignored_turns"] += 1
                        else:
                            state["speculative_ignored_turns"] += 1
                    else:
                        state["speculative_ignored_turns"] += 1
                else:
                    pass

                ignore_threshold = self.valves.speculative_decay_after_ignored_turns
                if state.get("speculative_ignored_turns", 0) >= ignore_threshold:
                    current_limit = state.get(
                        "speculative_preload_limit",
                        self.valves.speculative_preload_max_dependencies,
                    )
                    new_limit = max(current_limit - 1, 0)
                    if new_limit != current_limit:
                        state["speculative_preload_limit"] = new_limit
                        state["speculative_ignored_turns"] = 0
                        self._log_debug(
                            f"Speculative limit decayed to {new_limit} after {ignore_threshold} ignored turns."
                        )
                self._set_state(project_id, state)

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

    # --------------------------------------------------------------------------
    #  Shutdown and cleanup
    # --------------------------------------------------------------------------
    def shutdown(self):
        if (
            hasattr(self, "_response_cache_cleanup_task")
            and self._response_cache_cleanup_task is not None
        ):
            self._response_cache_cleanup_task.cancel()
        for task_list in [
            self._hierarchical_compress_tasks,
            self._summarize_tasks,
            self._dependency_tasks,
        ]:
            for task in task_list:
                task.cancel()
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
    #  Miscellaneous helpers (duplicates, diffs, dependencies)
    # --------------------------------------------------------------------------
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))
        return fuzz.token_sort_ratio(code1, code2) / 100.0

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

    async def _hierarchical_compress(self, project_id: str, state: Dict):
        if self._hierarchical_compress_in_progress.get(project_id, False):
            return
        self._hierarchical_compress_in_progress[project_id] = True
        try:
            if (
                not self.valves.hierarchical_compression_enabled
                or not self.memory_collection
            ):
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
            prompt_template = "Summarise the following conversation segment, keeping key technical decisions and code changes:\n\n{text}"
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

    # --------------------------------------------------------------------------
    #  Dependency tracking
    # --------------------------------------------------------------------------
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
        prompt = f"Analyze the following code and extract dependencies...\n```\n{code[:1500]}\n```"
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
    #  Diff application
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
                logger.warning(f"Unified diff hunk mismatch at line {old_start}")
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

    async def _handle_expand_command(self, text: str, project_id: str) -> Optional[str]:
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
        expanded = await self._expand_symbol_dependencies(func_name, depth, project_id)
        if not expanded:
            return f"No dependencies found for '{func_name}'."
        # If only the initial symbol is present (no further dependencies)
        if expanded.count("### ") <= 1:
            return (
                f"## Expanded dependencies for `{func_name}` (depth {depth})\n{expanded}\n\n"
                "[Note: No further dependencies were found for this symbol. "
                "The language may not be fully supported or the function has no calls.]"
            )
        return f"## Expanded dependencies for `{func_name}` (depth {depth})\n{expanded}"

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
    #  Placeholder methods for contradiction detection and duplicate questions
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #  Contradiction detection
    # --------------------------------------------------------------------------
    async def _detect_contradictions(self, messages: List[dict]) -> Optional[str]:
        """
        Detects contradictions between the user's latest message and the conversation history.
        Returns a warning string if a contradiction is found, otherwise None.
        """
        if not self.valves.enable_contradiction_detection or len(messages) < 3:
            return None
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user:
            return None
        history = messages[:-1]  # all messages before the last user
        if not history:
            return None
        # Combine history into a single text for analysis
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
        response = await self._try_llm_quick(
            prompt=prompt,
            system_prompt="You are a contradiction detector. Answer only 'yes' or 'no'.",
            model_override=model,
            max_tokens=3,
            temperature=0.0,
            timeout=6.0,
        )
        if response and response.strip().lower().startswith("yes"):
            return (
                "⚠️ **Contradiction detected**: The last message appears to contradict something established earlier. "
                "Please review and clarify if needed."
            )
        return None

    # --------------------------------------------------------------------------
    #  Duplicate question detection
    # --------------------------------------------------------------------------
    async def _find_duplicate_question(
        self, query: str, project_id: str
    ) -> Optional[dict]:
        """
        Checks if a very similar question has been asked recently in the LTM.
        Returns a dict with 'sim' and 'doc' if a duplicate is found, otherwise None.
        """
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return None
        if not query or len(query.strip()) < 15:
            return None
        try:
            q_emb = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode(query[:1000]).tolist()
            )
            # Look for recent user messages only
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
    #  Chain-of-Thought helpers
    # --------------------------------------------------------------------------
    async def _parse_cot_intent(self, user_content: str) -> Tuple[Optional[str], int]:
        """
        Extract the question and the requested CoT level from a /think command.
        Returns (question, level). Level defaults to 2 if not specified.
        """
        content = user_content.strip()
        if not content.startswith("/think"):
            return None, 2
        rest = content[6:].strip()  # remove "/think"
        if not rest:
            return None, 2
        # Check if the first word is a digit (level)
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

    async def _generate_cot(self, question: str, context: str) -> str:
        """Generate chain-of-thought reasoning using the configured COT model."""
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nThink step by step and provide your reasoning:"
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a helpful assistant that thinks step by step before answering.",
            model_override=self.valves.cot_model,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.4,
        )
        return response if response else "Unable to generate reasoning."

    # --------------------------------------------------------------------------
    #  Chain-of-Thought auto-detection (multi-level)
    # --------------------------------------------------------------------------
    async def _detect_cot_level(self, user_content, is_code_session, state):
        if not user_content:
            return 0
        if self.valves.enable_cot_llm_detection:
            return await self._detect_cot_level_via_llm(
                user_content, is_code_session, state
            )
        return self._detect_cot_level_heuristic(user_content, is_code_session, state)

    def _detect_cot_level_heuristic(
        self, user_content: str, is_code_session: bool, state: dict
    ) -> int:
        """Original static keyword-based detection (used as fallback)."""
        complex_keywords = {
            "analyze",
            "how",
            "why",
            "implement",
            "design",
            "architecture",
            "fix",
            "debug",
            "optimize",
            "refactor",
            "review",
            "compare",
            "analiza",
            "cómo",
            "por qué",
            "implementa",
            "diseña",
            "arquitectura",
            "corrige",
            "depura",
            "optimiza",
            "refactoriza",
            "revisa",
            "compara",
            "explain",
            "explica",
            "describe",
            "describir",
        }
        deep_keywords = {
            "deep review",
            "revisión profunda",
            "auto-evalúa",
            "comprueba cada paso",
            "itera varias veces",
            "razonamiento exhaustivo",
            "reflection",
            "deep",
            "reflexión",
        }

        content_lower = user_content.lower()
        has_code = "```" in user_content
        length_ok = len(user_content) >= self.valves.auto_cot_min_chars

        signals = 0
        if any(kw in content_lower for kw in complex_keywords):
            signals += 1
        if has_code:
            signals += 1
        if length_ok:
            signals += 1
        if user_content.count("?") >= 2:
            signals += 1

        if any(kw in content_lower for kw in deep_keywords):
            return 3
        if self.valves.enable_iterative_mode and state.get("iterative_state"):
            return 3

        if signals >= 3:
            return 2
        elif signals >= 2:
            return 1
        else:
            return 0

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
            "Respond with only the digit 0, 1, 2, or 3."
        )
        try:
            response = await self._try_llm_quick(
                prompt=prompt,
                system_prompt="You are a classifier. Output only a single digit.",
                model_override=self.valves.cot_detection_model,
                max_tokens=2,
                temperature=0.0,
                timeout=5.0,
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

    async def _generate_cot_reasoning(self, question: str, context: str) -> str:
        """Generate chain-of-thought reasoning using the configured CoT level 2 model."""
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nThink step by step and provide your reasoning:"
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a helpful assistant that thinks step by step before answering.",
            model_override=self.valves.cot_model_level2,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.4,
        )
        if response:
            # Wrap with clear metadata indicating it's auto-generated reasoning
            return (
                f"## 🔎 Automated Chain-of-Thought Reasoning (Level 2)\n"
                f"*This section was generated by {self.valves.cot_model_level2} "
                f"to assist the main assistant. It is not user input.*\n\n"
                f"{response}"
            )
        return "Unable to generate reasoning."

    async def _generate_cot_with_self_reflection(
        self, question: str, context: str
    ) -> str:
        """Generate reasoning and then self-reflect on it for higher accuracy."""
        reasoning = await self._generate_cot_reasoning(question, context)
        if not reasoning or reasoning == "Unable to generate reasoning.":
            return reasoning

        reflection_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Initial reasoning:\n{reasoning}\n\n"
            "Review the above reasoning. Are there any errors, unverified assumptions, or missing steps? "
            "Provide a corrected and improved reasoning."
        )
        refined = await self._call_llm(
            prompt=reflection_prompt,
            system_prompt="You are a critical reviewer. Improve the reasoning provided.",
            model_override=self.valves.cot_model_level3,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.3,
        )
        if refined:
            return (
                f"## 🔎🔎 Automated Chain-of-Thought with Self-Reflection (Level 3)\n"
                f"*Reasoning generated by {self.valves.cot_model_level2}, "
                f"reflection by {self.valves.cot_model_level3}. "
                f"This is not user input.*\n\n"
                f"{reasoning}\n\n"
                f"### 🔁 Self-Reflection\n{refined}"
            )
        return reasoning

    # --------------------------------------------------------------------------
    #  Assumption extraction helpers
    # --------------------------------------------------------------------------
    async def _parse_assumption_intent(self, user_content: str) -> Optional[str]:
        """Extract the target for assumption analysis (e.g., after /assume)."""
        if user_content.strip().startswith("/assume"):
            target = user_content.strip()[7:].strip()
            return target if target else None
        return None

    async def _extract_assumptions(self, target: str) -> str:
        """Extract assumptions from a given code or statement."""
        prompt = f"Analyze the following and list all assumptions it makes:\n\n{target}\n\nList each assumption clearly."
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are an expert at identifying hidden assumptions in text and code.",
            model_override=self.valves.assumption_extraction_model,
            max_tokens=600,
            temperature=0.3,
        )
        return response if response else "No assumptions identified."

    # --------------------------------------------------------------------------
    #  Iterative mode
    # --------------------------------------------------------------------------
    async def _run_iteration(
        self, project_id: str, user_content: str
    ) -> Tuple[str, bool]:
        """Handle iterative coding loop. Returns (result_message, consumed)."""
        # Placeholder: In the original, this manages multi-step iterative refinement.
        # For now, we just return a message indicating it's not fully implemented.
        if not user_content.strip().startswith("/iterate"):
            return "", False
        # Minimal implementation: just acknowledge
        return (
            "Iterative mode is not fully implemented in this version. Use `/iterate resume` to continue.",
            True,
        )

    # --------------------------------------------------------------------------
    #  Context helpers for structural / code review tasks
    # --------------------------------------------------------------------------
    async def _is_structural_task(self, user_query: str) -> bool:
        """Check if the user is asking for structural analysis (diagrams, call graphs, etc.)."""
        structural_keywords = {
            "diagram",
            "architecture",
            "call graph",
            "uml",
            "flowchart",
            "dependency graph",
            "structure",
        }
        query_lower = user_query.lower()
        return any(kw in query_lower for kw in structural_keywords)

    def _is_code_review_request(self, user_content: str) -> bool:
        """Check if the user message is requesting a code review."""
        review_phrases = {"review", "check my code", "code review", "audit", "inspect"}
        content_lower = user_content.lower()
        return any(phrase in content_lower for phrase in review_phrases)

    # --------------------------------------------------------------------------
    #  Feedback context
    # --------------------------------------------------------------------------
    def _get_feedback_context(self, project_id: str) -> str:
        """Return a summary of past feedback to inject into the system prompt."""
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
    #  Proactive summary suggestion
    # --------------------------------------------------------------------------
    async def _check_and_suggest_summarization(
        self, project_id: str, total_tokens: int, max_tokens: int
    ) -> Optional[str]:
        """Suggest summarizing old messages if the context is too large."""
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
    #  Summarization of old messages
    # --------------------------------------------------------------------------
    async def _summarize_messages(
        self, old_messages: List[dict], is_code_context: bool = False
    ) -> Optional[str]:
        """Summarize a list of old messages into a single summary."""
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
    #  Command suggestion (heuristic)
    # --------------------------------------------------------------------------
    async def _suggest_commands(self, project_id: str, state: dict) -> Optional[str]:
        """Suggest helpful commands after certain conditions."""
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
    #  Intent execution (forget, remember, obsolete)
    # --------------------------------------------------------------------------
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
    #  Inlet method (contains the main command routing)
    # --------------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Main entry point called before each LLM request.
        Orchestrates context retrieval, chain-of-thought, response cache,
        and all system injections. Now parallelizes CoT and cache checks.
        """
        self._log_debug("inlet called")
        inlet_start = time.monotonic()

        # Helper to log timing relative to inlet start
        def _inlet_timing(step_name: str, start: float, end: float = None):
            if end is None:
                end = time.monotonic()
            self._log_timing(step_name, start - inlet_start, end - start)

        self._ensure_cleanup_task()
        messages = body.get("messages", [])
        project_id = self._get_project_id()

        # Handle project switch: clean up old project's index and caches
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

        if not messages:
            return body

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        is_explicit_command = last_user_msg and last_user_msg.get(
            "content", ""
        ).startswith("/")

        # ------------------------------------------------------------------
        # Command routing: /forget (explicit)
        # ------------------------------------------------------------------
        if self.valves.enable_forget_command and is_explicit_command:
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                _inlet_timing("total_inlet", inlet_start)
                return body

        # ------------------------------------------------------------------
        # Command routing: natural language forget/remember/obsolete
        # ------------------------------------------------------------------
        if (
            self.valves.enable_natural_language_forget
            and last_user_msg
            and not is_explicit_command
        ):
            t0 = time.monotonic()
            intents = await self._parse_all_intents(last_user_msg.get("content", ""))
            _inlet_timing("parse_nl_intents", t0)
            for intent_type in ("forget", "remember", "obsolete"):
                fi = intents.get(intent_type, {})
                if fi.get("action") not in (None, "none"):
                    if intent_type == "forget":
                        confirmation = await self._execute_forget_intent(project_id, fi)
                    elif intent_type == "remember":
                        confirmation = await self._execute_remember_intent(
                            project_id, fi
                        )
                    elif (
                        intent_type == "obsolete"
                        and self.valves.enable_obsolete_marking
                    ):
                        confirmation = await self._execute_obsolete_intent(
                            project_id, fi
                        )
                    else:
                        continue
                    status_msg = f"[CodeAware] {confirmation}"
                    messages.insert(0, {"role": "system", "content": status_msg})
                    messages.pop()
                    messages.append({"role": "assistant", "content": confirmation})
                    messages = self._ensure_last_message_is_user(messages)
                    body["messages"] = messages
                    _inlet_timing("total_inlet", inlet_start)
                    return body

        # ------------------------------------------------------------------
        # Command routing: /status
        # ------------------------------------------------------------------
        if (
            last_user_msg
            and last_user_msg.get("content", "").strip() == "/status"
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
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet", inlet_start)
            return body

        # ------------------------------------------------------------------
        # Command routing: /speculative_stats
        # ------------------------------------------------------------------
        if (
            last_user_msg
            and last_user_msg.get("content", "").strip() == "/speculative_stats"
            and self.valves.speculative_stats_command_enabled
            and self.valves.speculative_log_missed_opportunities
        ):
            state = self._get_state(project_id)
            stats = state.get("speculative_missed_stats", {})
            lines = []
            if self.valves.speculative_adaptive:
                limit = state.get(
                    "speculative_preload_limit",
                    self.valves.speculative_preload_max_dependencies,
                )
                lines.append(
                    f"Current speculative preload limit: {limit} (max {self.valves.speculative_max_limit})"
                )
            if not stats or stats.get("total", 0) == 0:
                lines.append("No speculative miss data yet.")
            else:
                lines.append(f"Total missed opportunities: {stats['total']}")
                details = stats.get("details", {})
                if details:
                    sorted_syms = sorted(
                        details.items(), key=lambda x: x[1]["count"], reverse=True
                    )
                    lines.append("Most requested symbols:")
                    for sym, data in sorted_syms[:10]:
                        lines.append(f"- {sym}: {data['count']} times")
            response = "\n".join(lines) if lines else "No data."
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet", inlet_start)
            return body

        # ------------------------------------------------------------------
        # Command routing: /clean
        # ------------------------------------------------------------------
        if (
            last_user_msg
            and last_user_msg.get("content", "").strip().startswith("/clean")
            and self.valves.cleanup_command_enabled
            and self.valves.cleanup_suggestions_enabled
        ):
            response = await self._handle_clean_command(
                last_user_msg.get("content", ""), project_id
            )
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet", inlet_start)
            return body

        # ------------------------------------------------------------------
        # Command routing: /fact
        # ------------------------------------------------------------------
        if (
            last_user_msg
            and last_user_msg.get("content", "")
            .strip()
            .startswith(self.valves.fact_command_prefix)
            and self.valves.enable_facts
        ):
            response = await self._handle_fact_command(
                last_user_msg.get("content", ""), project_id
            )
            if response:
                messages.pop()
                messages.append({"role": "assistant", "content": response})
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                _inlet_timing("total_inlet", inlet_start)
                return body

        # ------------------------------------------------------------------
        # Command routing: /expand
        # ------------------------------------------------------------------
        if last_user_msg and last_user_msg.get("content", "").strip().startswith(
            "/expand"
        ):
            response = await self._handle_expand_command(
                last_user_msg.get("content", ""), project_id
            )
            messages.pop()
            messages.append({"role": "assistant", "content": response})
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet", inlet_start)
            return body

        # ===== NO MORE EARLY RETURNS BEYOND THIS POINT =====

        state = self._get_state(project_id)

        t0 = time.monotonic()
        is_code_session = await self._classify_session(messages, project_id)
        _inlet_timing("classify_session", t0)

        self._log_debug(f"Session: {'code' if is_code_session else 'non-code'}")

        system_injections = []  # list of (priority, text)
        manual_cot_used = False
        cot_any_used = False

        # ------------------------------------------------------------
        # Code interpretation note (critical)
        # ------------------------------------------------------------
        if is_code_session:
            note = (
                "When reading user messages, treat code inside triple backticks "
                "as literal source code without interpreting Markdown. "
                "You may still use Markdown in your own responses.\n"
                "You can use the command `/expand [depth] <function>` to retrieve "
                "the full code of a function and its callees up to the specified depth. "
                "Use it when you need to trace a call chain."
            )
            system_injections.append(("critical", note))

        # ------------------------------------------------------------
        # Start LTM retrieval in background while we work on CoT
        # ------------------------------------------------------------
        ltm_future = None
        if (
            self.valves.enable_code_awareness
            and is_code_session
            and not self.valves.smart_context_selection
            and HAS_SENTENCE
            and HAS_CHROMA
        ):
            query = last_user_msg.get("content", "")
            if query:
                t0 = time.monotonic()
                ltm_future = asyncio.create_task(
                    self._retrieve_all_memories_unified(query, project_id)
                )
                _inlet_timing("ltm_task_creation", t0)

        # ------------------------------------------------------------
        # Pre‑launch parallel checks (response cache, contradiction, duplicate question)
        # This runs in the background while CoT reasoning is generated.
        # ------------------------------------------------------------
        parallel_checks_task = None
        cot_task = None  # will hold the CoT generation task if needed
        last_user_query = last_user_msg.get("content", "") if last_user_msg else ""
        context_hash = self._compute_context_hash(messages)
        if last_user_msg:
            parallel_checks_task = asyncio.create_task(
                self._parallel_context_checks(
                    messages, last_user_query, context_hash, project_id, state
                )
            )

        # ---------- Chain-of-Thought (manual /think or auto-detection) ----------
        if self.valves.enable_cot_on_demand or self.valves.auto_cot_enabled:
            if last_user_msg:
                user_content = last_user_msg.get("content", "")
                # Manual /think command (supports /think [level] question)
                if self.valves.enable_cot_on_demand and user_content.strip().startswith(
                    "/think"
                ):
                    cot_question, level = await self._parse_cot_intent(user_content)
                    if cot_question:
                        manual_cot_used = True
                        cot_any_used = True
                        self._log_debug(f"Manual /think requested with level {level}")
                        if level == 1:
                            cot_prompt = "Please think step by step before answering. Show your reasoning, then provide the final answer."
                            system_injections.append(("high", cot_prompt))
                        elif level == 2:
                            t0 = time.monotonic()
                            active_ctx = self._get_active_code_context(project_id)
                            facts_ctx = self._get_facts_context(project_id)
                            context = (
                                f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
                            )
                            # Launch CoT in background
                            cot_task = asyncio.create_task(
                                self._generate_cot_reasoning(cot_question, context)
                            )
                            _inlet_timing("cot_manual_level2", t0)
                        elif level == 3:
                            t0 = time.monotonic()
                            active_ctx = self._get_active_code_context(project_id)
                            facts_ctx = self._get_facts_context(project_id)
                            context = (
                                f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
                            )
                            cot_task = asyncio.create_task(
                                self._generate_cot_with_self_reflection(
                                    cot_question, context
                                )
                            )
                            _inlet_timing("cot_manual_level3", t0)

                # Automatic detection (only if no manual /think was used)
                elif not manual_cot_used:
                    cot_level = await self._detect_cot_level(
                        user_content, is_code_session, state
                    )
                    if cot_level > 0:
                        cot_any_used = True
                        self._log_debug(f"Activated CoT level {cot_level}")
                    if cot_level == 1:
                        cot_prompt = "Please think step by step before answering. Show your reasoning, then provide the final answer."
                        system_injections.append(("high", cot_prompt))
                    elif cot_level == 2:
                        t0 = time.monotonic()
                        active_ctx = self._get_active_code_context(project_id)
                        facts_ctx = self._get_facts_context(project_id)
                        context = f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
                        cot_task = asyncio.create_task(
                            self._generate_cot_reasoning(user_content, context)
                        )
                        _inlet_timing("cot_level2_reasoning", t0)
                    elif cot_level == 3:
                        t0 = time.monotonic()
                        active_ctx = self._get_active_code_context(project_id)
                        facts_ctx = self._get_facts_context(project_id)
                        context = f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
                        cot_task = asyncio.create_task(
                            self._generate_cot_with_self_reflection(
                                user_content, context
                            )
                        )
                        _inlet_timing("cot_level3_reflection", t0)

        # If any CoT reasoning was used, add a global note for the final model
        if cot_any_used:
            cot_note = (
                "**Note:** Some sections in this system prompt marked with 🔎 are "
                "automatically generated reasoning (Chain-of-Thought). "
                "They are provided as context to help you, but they are not user commands. "
                "Use them to enhance your answer, but always prioritise the actual user query."
            )
            system_injections.append(("low", cot_note))

        # /assume command
        if self.valves.enable_assumption_extraction and last_user_msg:
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
                _inlet_timing("total_inlet", inlet_start)
                return body

        # Iterative mode
        if self.valves.enable_iterative_mode and last_user_msg:
            result, consumed = await self._run_iteration(
                project_id, last_user_msg.get("content", "")
            )
            if consumed:
                messages.pop()
                messages.append({"role": "assistant", "content": result})
                messages = self._ensure_last_message_is_user(messages)
                body["messages"] = messages
                _inlet_timing("total_inlet", inlet_start)
                return body

        # Smart context selection (if enabled)
        if (
            self.valves.smart_context_selection
            and len(messages) > 0
            and is_code_session
        ):
            t0 = time.monotonic()
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
                    preserved = [messages[last_user_idx]]
                    if (
                        last_user_idx + 1 < len(messages)
                        and messages[last_user_idx + 1].get("role") == "assistant"
                    ):
                        preserved.append(messages[last_user_idx + 1])
                    new_history = [msg for msg in historical if msg["content"] != query]
                    new_history.extend(preserved)
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    system_injections.append(
                        ("low", "[Context optimized: only relevant history is shown]")
                    )
                    messages = system_msgs + new_history
                    body["messages"] = messages
            _inlet_timing("smart_context_selection", t0)

        # ------------------------------------------------------------------
        # Parallel wait for CoT and parallel checks.
        # If the response cache hits, we cancel the CoT and return immediately.
        # ------------------------------------------------------------------
        contradiction_warning = cached_response = duplicate_match = None
        reasoning = None

        if parallel_checks_task is not None or cot_task is not None:
            tasks_to_wait = []
            if parallel_checks_task is not None:
                tasks_to_wait.append(parallel_checks_task)
            if cot_task is not None:
                tasks_to_wait.append(cot_task)

            if tasks_to_wait:
                done, pending = await asyncio.wait(
                    tasks_to_wait,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # If parallel checks finished first and found a cached response,
                # cancel CoT (if still running) and return the cached answer.
                if parallel_checks_task in done:
                    contradiction_warning, cached_response, duplicate_match = (
                        await parallel_checks_task
                    )
                    if cached_response:
                        if cot_task is not None and not cot_task.done():
                            cot_task.cancel()
                        messages.append(
                            {
                                "role": "assistant",
                                "content": cached_response["response"],
                            }
                        )
                        messages = self._ensure_last_message_is_user(messages)
                        body["messages"] = messages
                        _inlet_timing("total_inlet", inlet_start)
                        return body

                # Collect CoT result (either already done or still pending)
                if cot_task is not None:
                    if cot_task in done:
                        reasoning = cot_task.result()
                    else:
                        reasoning = await cot_task

                # Ensure parallel checks are fully completed
                if parallel_checks_task is not None and not parallel_checks_task.done():
                    contradiction_warning, cached_response, duplicate_match = (
                        await parallel_checks_task
                    )
                elif parallel_checks_task is not None and parallel_checks_task in done:
                    # already stored from above
                    pass

                # Double‑check for a late cache hit (shouldn't happen, but just in case)
                if cached_response:
                    messages.append(
                        {"role": "assistant", "content": cached_response["response"]}
                    )
                    messages = self._ensure_last_message_is_user(messages)
                    body["messages"] = messages
                    _inlet_timing("total_inlet", inlet_start)
                    return body

        # Inject the CoT reasoning (if any)
        if reasoning:
            system_injections.append(
                ("high", f"**Chain-of-Thought Reasoning**\n{reasoning}")
            )

        if contradiction_warning and self.valves.contradiction_inject_warning:
            system_injections.append(("high", contradiction_warning))
        if cached_response:
            # This case was already handled above; left for safety.
            messages.append(
                {"role": "assistant", "content": cached_response["response"]}
            )
            messages = self._ensure_last_message_is_user(messages)
            body["messages"] = messages
            _inlet_timing("total_inlet", inlet_start)
            return body
        if duplicate_match:
            warn_msg = f"⚠️ **Note**: This question is very similar to one you asked before (similarity {duplicate_match['sim']:.2f})."
            system_injections.append(("medium", warn_msg))

        # Update active code (historical in background, last message synchronously)
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

            t0 = time.monotonic()
            await self._update_active_code(messages[last_idx], project_id)
            _inlet_timing("update_active_code_last", t0)

            self._last_processed_message_idx[project_id] = last_idx

        # ------------------------------------------------------------
        # Get the result of LTM recovery that started in parallel
        # ------------------------------------------------------------
        unique_meta = []
        if ltm_future is not None:
            t0 = time.monotonic()
            all_meta = await ltm_future
            all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
            seen = set()
            for m in all_meta:
                if m["doc"] not in seen:
                    seen.add(m["doc"])
                    unique_meta.append(m)
            _inlet_timing("ltm_retrieval", t0)

        # Format and inject LTM (high priority)
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
            system_injections.append(("high", ctx))

        # ---- Proactive cleanup suggestion ----
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
                    suggestion = (
                        f"[CodeAware SUGGESTION] You have {len(candidates)} inactive code blocks. "
                        f"Type `/status` to review or `/clean` to forget them. "
                        f"(This note is not part of the conversation with the model.)"
                    )
                    system_injections.append(("medium", suggestion))
                    state["last_cleanup_suggestion_msg_idx"] = state["message_count"]
                    self._set_state(project_id, state)

        # Inject active code context (lightweight or full)
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
            user_query = last_user_msg.get("content", "") if last_user_msg else ""

            if total_code_tokens > self.valves.huge_injection_threshold_tokens > 0:
                self._log_debug(
                    f"Massive injection detected ({total_code_tokens} tokens). Using lightweight context."
                )
                active_ctx = await self._build_lightweight_context(project_id)
                if last_user_msg and not is_structural:
                    expanded = self._expand_referenced_symbols(project_id, user_query)
                    if expanded:
                        active_ctx += "\n" + expanded
                elif is_structural:
                    active_ctx += (
                        "\n\n[Note: Structural analysis requested. Use the symbol index "
                        "with call graphs and summaries to generate the diagram. Do not "
                        "request code bodies.]"
                    )
            else:
                active_ctx = self._get_active_code_context(
                    project_id, user_query=user_query
                )
                if last_user_msg and not is_structural and user_query:
                    expanded = self._expand_referenced_symbols(project_id, user_query)
                    if expanded:
                        active_ctx += "\n" + expanded

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
                system_injections.append(("critical", active_ctx))

        # Code review checklist injection
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
            system_injections.append(("high", review_prompt))

        # Inject facts context
        if (
            is_code_session
            and self.valves.enable_facts
            and self.valves.inject_facts_in_context
        ):
            facts_ctx = self._get_facts_context(project_id)
            if facts_ctx:
                system_injections.append(("high", facts_ctx))

        # Confidence scoring
        if self.valves.enable_confidence_scoring and is_code_session:
            total_tokens = self._estimate_tokens(messages)
            if total_tokens > self.valves.context_window_tokens * 0.8:
                system_injections.append(("high", self.valves.confidence_prompt))

        # Inject feedback context
        if (
            is_code_session
            and self.valves.enable_feedback_tracking
            and self.valves.inject_feedback_context
        ):
            feedback_ctx = self._get_feedback_context(project_id)
            if feedback_ctx:
                system_injections.append(("high", feedback_ctx))

        # Proactive summary suggestion (if context grows too fast)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._estimate_tokens(system_msgs + history_msgs)
        if self.valves.context_window_tokens > 0:
            t0 = time.monotonic()
            suggestion = await self._check_and_suggest_summarization(
                project_id, total_tokens, self.valves.context_window_tokens
            )
            _inlet_timing("check_summarization_suggestion", t0)
            if suggestion:
                system_injections.append(("medium", suggestion))

        # Command suggestion (after a certain number of messages without commands)
        t0 = time.monotonic()
        cmd_suggestion = await self._suggest_commands(project_id, state)
        _inlet_timing("suggest_commands", t0)
        if cmd_suggestion:
            system_injections.append(("medium", cmd_suggestion))

        # Adaptive context trim (based on token count or max_turns)
        if self.valves.adaptive_trim:
            total_tokens = self._estimate_tokens(system_msgs + history_msgs)
            if total_tokens > self.valves.context_window_tokens:
                self._log_debug("Trimming old messages due to token budget")
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
                    t0 = time.monotonic()
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    _inlet_timing("summarize_old_messages", t0)
                    if summary:
                        system_injections.append(
                            ("high", f"[Summary of earlier conversation]\n{summary}")
                        )
                    history_msgs = kept_block
                else:
                    history_msgs = kept_block

                # Preserve tool calls if present
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
            # Fallback: trim by max_turns
            user_max = (
                __user__["valves"].max_turns
                if __user__ and hasattr(__user__, "valves")
                else None
            )
            eff_max = user_max if user_max is not None else self.valves.max_turns
            if len(history_msgs) > eff_max:
                self._log_debug("Trimming old messages based on max_turns")
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
                    t0 = time.monotonic()
                    has_code = any("```" in m.get("content", "") for m in old_block)
                    summary = await self._summarize_messages(
                        old_block, is_code_context=has_code
                    )
                    _inlet_timing("summarize_old_messages", t0)
                    if summary:
                        system_injections.append(
                            ("high", f"[Summary of earlier conversation]\n{summary}")
                        )
                    history_msgs = kept_block
                else:
                    history_msgs = kept_block

        messages = system_msgs + history_msgs

        # Final safety: ensure last message is from user
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

        # Assemble system message with token budget
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        base_content = ""
        if sys_msgs:
            base_content = sys_msgs[0].get("content", "")
            messages = [m for m in messages if m.get("role") != "system"]

        t0 = time.monotonic()
        budget = self.valves.global_injection_token_budget
        if budget > 0 and self.tokenizer:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            system_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_texts = []
            total_tokens = 0
            for prio, text in system_injections:
                if not text:
                    continue
                tokens = len(self.tokenizer.encode(text))
                if total_tokens + tokens <= budget:
                    selected_texts.append(text)
                    total_tokens += tokens
                else:
                    if prio in ("critical", "high"):
                        available = budget - total_tokens
                        if available > 20:
                            truncated = text[: available * 4] + "\n[truncated]"
                            selected_texts.append(truncated)
                            total_tokens += len(self.tokenizer.encode(truncated))
                            break
            final_system = "\n\n".join(selected_texts)
        else:
            final_system = "\n\n".join(text for _, text in system_injections if text)
        _inlet_timing("assemble_system_message", t0)

        if base_content.strip():
            final_system = final_system + "\n\n" + base_content

        if final_system.strip():
            messages.insert(0, {"role": "system", "content": final_system})

        # Debug log for injected token count
        if self.valves.debug:
            total_system_tokens = 0
            for m in messages:
                if m.get("role") == "system":
                    content = m.get("content", "")
                    total_system_tokens += (
                        len(self.tokenizer.encode(content))
                        if self.tokenizer
                        else len(content) // 4
                    )
            self._log_debug(f"Injected system tokens: {total_system_tokens}")

        body["messages"] = messages
        _inlet_timing("total_inlet", inlet_start)
        return body

    # --------------------------------------------------------------------------
    #  Change summary generation
    # --------------------------------------------------------------------------
    async def _generate_change_summary(
        self, block_hash: str, prev_content: str, new_content: str
    ):
        """Generate a one‑line summary of what changed between two versions, using a tiny LLM."""
        if not HAS_AIOHTTP:
            return
        model = (
            self.valves.natural_language_forget_model or self.valves.summarization_model
        )
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
            self._block_change_summaries[block_hash] = (summary.strip(), now)
            # Limit dictionary
            max_entries = self.valves.max_change_summaries
            if len(self._block_change_summaries) > max_entries:
                # Eliminate oldest entries (order by timestamp)
                sorted_entries = sorted(
                    self._block_change_summaries.items(), key=lambda x: x[1][1]
                )
                # Delete the first (oldest) to be back to the limit
                to_remove = sorted_entries[
                    : len(self._block_change_summaries) - max_entries
                ]
                for key, _ in to_remove:
                    del self._block_change_summaries[key]

    # --------------------------------------------------------------------------
    #  Auto summaries for missing symbol docstrings
    # --------------------------------------------------------------------------
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
