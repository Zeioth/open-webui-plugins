"""
title: Code-Aware Context Manager with LTM & Summarization
description: Full-featured context manager for coding assistants.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 6.0.0
license: GPL3
requirements: loguru, tiktoken, sentence-transformers, chromadb, rapidfuzz, tree-sitter-language-pack>=1.5.0
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
from typing import Optional, List, Dict, Any, Tuple, Union, Set
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
    dependencies: List[str] = Field(default_factory=list)
    potentially_affected: bool = False
    pinned: bool = False
    affected_timestamp: float = 0.0
    obsolete: bool = False
    ast_imports: List[str] = Field(default_factory=list)
    ast_calls: List[str] = Field(default_factory=list)
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
        penalty = 0.7 if self.potentially_affected else 1.0
        if self.obsolete:
            penalty = 0.1
            self.is_active = False
        self.importance_score = (
            (base_score + keyword_boost) * mention_boost * recency_factor * penalty
        )


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

    def find_blocks(self, name: str, project_id: str) -> Set[str]:
        return self._name_to_blocks.get((project_id, name), set())

    def get_all_names(self, project_id: str) -> Set[str]:
        return {key[1] for key in self._name_to_blocks if key[0] == project_id}

    def get_callers(self, callee_name: str, project_id: str) -> Set[str]:
        return self._callee_to_callers.get((project_id, callee_name), set())

    def clear_project(self, project_id: str):
        keys_to_remove = [key for key in self._name_to_blocks if key[0] == project_id]
        for key in keys_to_remove:
            del self._name_to_blocks[key]
            del self._stats[key]
        inv_keys = [key for key in self._callee_to_callers if key[0] == project_id]
        for key in inv_keys:
            del self._callee_to_callers[key]

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
        #  MAIN VALVES – most commonly adjusted
        # ═══════════════════════════════════════════════════════════════
        # use_symbol_level_analysis – enable per‑symbol analysis (recommended)
        # symbol_analysis_model      – fast model for per‑symbol analysis
        # active_context_max_tokens  – max tokens for injected code context
        # global_injection_token_budget – overall limit for all injections
        # ltm_retrieval_max_tokens   – max LTM tokens to inject
        # defer_secondary_tasks      – delay summaries / change logs
        # secondary_task_model       – model for those secondary tasks
        # synthesis_max_tokens       – max tokens for the final summary
        # ═══════════════════════════════════════════════════════════════

        # ─── Core ───
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

        # ─── Long‑Term Memory (ChromaDB) ───
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

        # ─── Code Awareness & Context ───
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
            description="Maximum number of active code blocks to keep (0 = unlimited). If you limit this value, the remaining symbols will be just inclued without beign enriched.",
        )
        track_file_paths: bool = Field(default=True)
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b"
        )
        max_code_block_tokens: int = Field(default=0)
        code_block_overflow_action: str = Field(default="warn")
        code_block_summary_model: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M"
        )
        code_block_truncate_keep_head: int = Field(default=50)
        code_block_truncate_keep_tail: int = Field(default=50)
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]"
        )
        huge_injection_threshold_tokens: int = Field(
            default=25000,
            description="Threshold of active code tokens above which lightweight context (signatures only) is used. 0 = never.",
        )
        enable_call_graph_extraction: bool = Field(
            default=True,
            description="Extract call relationships (who calls whom) for code symbols.",
        )
        enable_auto_summaries: bool = Field(
            default=True,
            description="Automatically generate one-line summaries for code symbols using a small LLM.",
        )
        summary_code_max_chars: int = Field(
            default=8000,
            description="Maximum characters of code to include when summarizing code blocks.",
        )
        oversized_summary_max_tokens: int = Field(
            default=500, description="Max tokens for summarizing oversized code blocks."
        )
        full_code_injection_budget_percent: float = Field(
            default=0.7,
            ge=0.0,
            le=1.0,
            description="Percentage of the global injection token budget to use for full-code injection when a code review is requested.",
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
            description="Exclude symbols from the filter's own source code to prevent self-analysis. This prevent openwebui from analyzing its own code.",
        )

        # ─── Smart Pre‑Expand ───
        smart_pre_expand_enabled: bool = Field(default=True)
        smart_pre_expand_min_tokens: int = Field(default=2000)
        smart_pre_expand_max_tokens: int = Field(default=0)
        smart_pre_expand_use_llm: bool = Field(default=True)
        smart_pre_expand_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")
        smart_pre_expand_full_if_no_match: bool = Field(default=True)
        smart_pre_expand_embedding_threshold: float = Field(
            default=0.72, ge=0.0, le=1.0
        )
        smart_pre_expand_min_symbols: int = Field(default=3)
        enable_raw_code_detection: bool = Field(default=True)

        # ─── Outlet Expand Intercept ───
        outlet_expand_intercept_enabled: bool = Field(default=True)
        outlet_expand_intercept_max_symbols: int = Field(default=0, ge=0)
        outlet_expand_intercept_depth: int = Field(default=5, ge=0)
        expand_default_depth: int = Field(default=2)

        # ─── Smart Context Selection ───
        smart_context_selection: bool = Field(default=False)
        smart_context_top_k: int = Field(default=15)
        smart_context_min_tokens: int = Field(default=1024)
        smart_context_include_last_user: bool = Field(default=True)

        # ─── Duplicate Blocks & Frequency ───
        auto_remove_duplicate_blocks: bool = Field(default=True)
        max_duplicate_age_hours: float = Field(default=6.0)
        frequency_weight_factor: float = Field(default=0.3)
        min_mentions_for_boost: int = Field(default=3)
        frequency_decay_hours: float = Field(default=12.0)

        # ─── Confidence Scoring & Chain‑of‑Thought ───
        enable_confidence_scoring: bool = Field(default=True)
        confidence_prompt: str = Field(
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]' where XX is your estimated confidence (0-100) in the correctness and completeness of your answer, based on the available context. If you lack information, give lower confidence and suggest what context would help."
        )
        enable_cot_on_demand: bool = Field(default=True)
        auto_cot_enabled: bool = Field(default=False)
        auto_cot_min_chars: int = Field(default=200)
        enable_code_review_mode: bool = Field(default=True)
        cot_max_tokens: int = Field(default=0)
        cot_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")
        cot_model_level2: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M",
            description="Model used for CoT level 2 (auto‑reasoning). Must support large contexts.",
        )
        cot_model_level3: str = Field(
            default="Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Nano",
            description="Model used for CoT level 3 (self‑reflection). Must support large contexts.",
        )
        enable_cot_llm_detection: bool = Field(default=True)
        cot_detection_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")

        # ─── Assumptions & Contradictions ───
        enable_assumption_extraction: bool = Field(default=True)
        assumption_extraction_model: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M"
        )
        enable_contradiction_detection: bool = Field(default=True)
        contradiction_detection_model: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M"
        )
        contradiction_inject_warning: bool = Field(default=True)

        # ─── Proactive Context Warning ───
        proactive_context_warning_threshold: float = Field(default=0.85)
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Context Warning**: The conversation is using more than {percent}% of the available context window ({used_tokens}/{max_tokens} tokens). Consider using `/forget` to remove irrelevant parts, `/remember` to pin important context, or ask me to summarize older parts."
        )

        # ─── Similar Messages & Obsolete Marking ───
        similar_message_handling: str = Field(default="replace")
        similar_message_threshold: float = Field(default=0.92)
        similar_message_check_code_only: bool = Field(default=True)
        enable_obsolete_marking: bool = Field(default=True)

        # ─── Proactive Summary & Command Suggestions ───
        proactive_summary_threshold: float = Field(default=0.75)
        proactive_summary_growth_window: int = Field(default=3)
        enable_command_suggestions: bool = Field(default=True)
        command_suggestion_cooldown_minutes: int = Field(default=10)

        # ─── Duplicate Question Detection ───
        duplicate_question_threshold: float = Field(default=0.92)
        duplicate_question_lookback: int = Field(default=20)
        duplicate_question_lookback_hours: float = Field(default=24.0)

        # ─── Response Cache ───
        enable_response_cache: bool = Field(default=True)
        response_cache_similarity_threshold: float = Field(default=0.92)
        response_cache_ttl_hours: float = Field(default=24.0)
        response_cache_max_entries: int = Field(default=100)
        response_cache_include_context_hash: bool = Field(default=True)

        # ─── Selective Summarization ───
        selective_summarization: bool = Field(default=True)
        error_preserve_verbatim: bool = Field(default=True)
        error_max_age_hours: float = Field(default=48.0)
        code_summary_level: str = Field(default="balanced")
        general_summary_max_tokens: int = Field(default=200)
        tool_call_preserve: bool = Field(default=True)
        code_always_keep_signature: bool = Field(default=True)
        summary_fallback_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")
        summary_include_metadata: bool = Field(default=True)

        # ─── Summarize Old Messages ───
        summarize_old_messages: bool = Field(default=True)
        summarization_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")

        # ─── LLM Configuration ───
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
        )
        openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", "dummy"))
        LLM_BASE_URL: str = Field(default="http://host.docker.internal:8080")
        LLM_API_TOKEN: str = Field(default="")
        llm_model: str = Field(default="Qwen2.5-Coder-7B-Instruct-Q4_K_M")
        LLM_MAX_CONCURRENT_CALLS: int = Field(default=2, ge=1, le=10)
        llm_request_timeout: int = Field(default=900)
        LLM_CACHE_TTL: int = Field(default=300)
        LLM_CACHE_MAX_SIZE: int = Field(default=100)

        # ─── llama.cpp endpoint type ───
        llamacpp_endpoint_type: str = Field(
            default="chat",
            description="Endpoint type for llama.cpp: 'chat' (default) or 'completion'.",
        )

        # ─── Importance & Expiration ───
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

        # ─── Diff & Patterns ───
        enable_diff_application: bool = Field(default=True)
        preserve_error_context: bool = Field(default=True)
        code_block_pattern: str = Field(default="```(\\w*)\\n(.*?)```")
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@"
        )
        commit_pattern: str = Field(default="commit\\s+([a-f0-9]{7,40})")

        # ─── Feedback Tracking ───
        enable_feedback_tracking: bool = Field(default=True)
        feedback_history_limit: int = Field(default=10)
        inject_feedback_context: bool = Field(default=True)
        feedback_importance_penalty_for_failure: float = Field(default=2.0)

        # ─── Summarize Inactive Code ───
        summarize_inactive_code: bool = Field(default=True)
        inactive_code_summary_model: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M"
        )

        # ─── Forget Commands ───
        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=True)
        natural_language_forget_model: str = Field(
            default="Qwen2.5-Coder-7B-Instruct-Q4_K_M"
        )

        # ─── Proactive Cleanup ───
        cleanup_suggestions_enabled: bool = Field(default=True)
        cleanup_inactive_threshold_messages: int = Field(default=30)
        cleanup_excluded_content_types: list = Field(
            default_factory=lambda: ["BASE_CODE"]
        )
        cleanup_status_command_enabled: bool = Field(default=True)
        cleanup_proactive_suggestions: bool = Field(default=True)
        cleanup_suggestion_cooldown_messages: int = Field(default=20)
        cleanup_command_enabled: bool = Field(default=True)

        # ─── Raw File Priority Boost ───
        raw_file_priority_boost: float = Field(default=2.0)

        # ─── Symbol‑level analysis ───
        use_symbol_level_analysis: bool = Field(
            default=True,
            description="Analyze code symbol by symbol instead of raw chunks. Works reliably even with 7B models.",
        )
        symbol_analysis_model: str = Field(
            default="llama-3.2-3b-instruct-q4_k_m",
            description="Fast model used for per‑symbol analysis.",
        )
        symbol_analysis_max_retries: int = Field(
            default=5,
            description="Max attempts per symbol before giving up.",
        )
        synthesis_max_tokens: int = Field(
            default=1500,
            description="Max tokens for the synthesized summary of symbol analysis.",
        )
        symbol_batch_size: int = Field(
            default=20,
            description="Number of symbols to analyze in parallel per batch (lower = less RAM).",
        )

        # ─── Session summaries (autobiographical mini‑memory) ───
        enable_session_summary: bool = Field(
            default=True,
            description="Generate an autobiographical session summary every N turns and store it in LTM.",
        )
        session_summary_interval_messages: int = Field(
            default=8,
            description="How many messages between session summaries.",
        )
        session_summary_model: str = Field(
            default="llama-3.2-3b-instruct-q4_k_m",
            description="Model used to generate session summaries.",
        )
        session_summary_max_tokens: int = Field(
            default=200,
            description="Max tokens for the session summary.",
        )

        # ─── Secondary task deferral ───
        defer_secondary_tasks: bool = Field(
            default=True,
            description="Defer secondary LLM tasks to the next inlet to avoid concurrency.",
        )
        secondary_task_max_retries: int = Field(
            default=5,
            description="Max retries for deferred secondary tasks before giving up.",
        )
        secondary_task_model: str = Field(
            default="llama-3.2-3b-instruct-q4_k_m",
            description="Model to use for secondary tasks (summaries, change logs).",
        )
        secondary_llm_max_concurrent: int = Field(
            default=2,
            description="Max concurrent LLM calls for deferred secondary tasks.",
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

        # ── Caches ──
        self._symbol_analysis_cache: OrderedDict = OrderedDict()
        self._MAX_SYMBOL_ANALYSIS_CACHE = 1000
        self._symbol_analysis_generic_cache: OrderedDict = OrderedDict()

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

        # ── Database write queue (prevents "database is locked") ──
        self._db_write_queue: asyncio.Queue = asyncio.Queue()
        self._db_worker_task = asyncio.create_task(self._db_worker())

        # Background tasks tracking
        self._summarize_inactive_in_progress: Dict[str, bool] = {}
        self._dependency_tasks: List[asyncio.Task] = []
        self._write_counter = 0
        self._response_cache_cleanup_task: Optional[asyncio.Task] = None

        # Session classification cache
        self._session_classify_cache: Dict[str, Tuple[bool, float]] = {}
        self._session_classify_ttl: float = 1800.0

        # Symbol index and lightweight context
        self._symbol_index = SymbolIndex()
        self._cached_lightweight_context: Dict[str, str] = {}
        self._cached_code_state_hash: Optional[str] = None

        # Project tracking
        self._last_processed_message_idx: Dict[str, int] = {}
        self._last_project_id: str = ""
        self._code_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        self._symbol_cache_loaded_projects: Set[str] = set()

        # Smart pre‑expand prototype phrases
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._query_embedding_cache_max_size = 100

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

    def _is_span_in_code(
        self, code_spans: List[Tuple[int, int]], span: Tuple[int, int]
    ) -> bool:
        s, e = span
        return any(cs <= s and e <= ce for cs, ce in code_spans)

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
            CREATE TABLE IF NOT EXISTS symbol_analysis_cache (
                project_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                question_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (project_id, symbol_name, question_hash)
            )
        """)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS block_change_summaries (
                block_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at REAL NOT NULL
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
        Unload all models if switching to a *different* model.
        Skips unloading if the target model is the same as the last used one.
        """
        if is_ollama:
            return
        if self._last_used_model is not None and model_name != self._last_used_model:
            self._log_debug(
                f"Switching model from '{self._last_used_model}' to '{model_name}'"
            )
            try:
                await _shared_unload_all_models(base_url)
                self._log_debug("All loaded models unloaded before switching")
                self._last_used_model = None
            except Exception as e:
                self._log_debug(f"Unload via shared_resources failed: {e}")
        elif self._last_used_model is None:
            self._log_debug(f"Loading first model '{model_name}'")
        else:
            self._log_debug(f"Reusing model '{model_name}' (already loaded)")

    @staticmethod
    def _build_llm_request(
        model_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        is_ollama: bool,
        ep_type: str,
        base_url: str,
        api_token: Optional[str],
        openai_api_key: Optional[str],
    ) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        headers = {"Content-Type": "application/json"}
        if not is_ollama:
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            elif openai_api_key:
                headers["Authorization"] = f"Bearer {openai_api_key}"

        if is_ollama:
            url = f"{base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
        else:
            if ep_type == "completion":
                url = f"{base_url}/v1/completions"
                payload = {
                    "model": model_name,
                    "prompt": (
                        prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
                    ),
                    "temperature": temperature,
                }
            else:  # chat
                url = f"{base_url}/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

        return url, payload, headers

    @staticmethod
    def _parse_llm_response(data: Dict[str, Any], is_ollama: bool, ep_type: str) -> str:
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
            if ep_type == "completion":
                content = choices[0].get("text", "")
            else:
                content = choices[0].get("message", {}).get("content", "")
        return content.strip()

    async def _acquire_llm_lock(self):
        """Acquire an inter‑process file lock for exclusive LLM access."""
        loop = asyncio.get_event_loop()
        fd = open(_llm_lock_path, "w")
        await loop.run_in_executor(self._db_executor, fcntl.flock, fd, fcntl.LOCK_EX)
        return fd

    async def _wait_for_empty_slot(self, retries: int = 3, delay: float = 2.0) -> bool:
        """
        Check that the LLM server has no loaded models.
        Retries a few times with a delay between checks.
        Returns True if the slot is empty, False otherwise.
        """
        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")

        for attempt in range(retries):
            await asyncio.sleep(delay)
            try:
                session = await get_http_session(timeout=5)
                async with session.get(f"{base_url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        loaded_models = [
                            m["id"]
                            for m in data.get("data", [])
                            if m.get("status", {}).get("value") == "loaded"
                        ]
                        self._log_debug(
                            f"Models currently loaded: {loaded_models if loaded_models else 'none'}"
                        )
                        if not loaded_models:
                            return True
                    else:
                        self._log_debug(f"Model list returned status {resp.status}")
            except Exception as e:
                self._log_debug(f"Error checking models: {e}")

        self._log_debug(f"Slot still occupied after {retries} retries")
        return False

    @staticmethod
    def _release_llm_lock(fd):
        """Release the inter‑process file lock and close the file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

    async def _wait_for_empty_slot(self, retries: int = 3, delay: float = 2.0) -> bool:
        """
        Check that the LLM server has no loaded models.
        Retries a few times with a delay between checks.
        Returns True if the slot is empty, False otherwise.
        """
        from shared_resources import get_http_session

        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        for _ in range(retries):
            await asyncio.sleep(delay)
            try:
                session = await get_http_session(timeout=5)
                async with session.get(f"{base_url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        loaded = any(
                            m.get("status", {}).get("value") == "loaded"
                            for m in data.get("data", [])
                        )
                        if not loaded:
                            return True
            except Exception:
                pass
        return False

    async def _unload_models_under_lock(self):
        """
        Unload all models from the LLM server while holding the global file lock.
        This prevents other processes/tasks from interfering during the unload/check cycle.
        """
        llm_fd = await self._acquire_llm_lock()
        try:
            await _shared_unload_all_models(self.valves.LLM_BASE_URL)
            self._last_used_model = None
            slot_empty = await self._wait_for_empty_slot(retries=3, delay=2.0)
            if not slot_empty:
                self._log_debug("Slot not empty after unload – forcing extra unload")
                await _shared_unload_all_models(self.valves.LLM_BASE_URL)
                await self._wait_for_empty_slot(retries=2, delay=3.0)
        finally:
            self._release_llm_lock(llm_fd)

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
        contents = [m["content"] for m in valid]
        embeddings = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(contents, convert_to_numpy=True).tolist()
        )
        ids = []
        emb_list = []
        metadatas = []
        documents = []
        now = time.time()
        for i, msg in enumerate(valid):
            msg_id = f"{project_id}_{int(now)}_{hashlib.md5(msg['content'].encode()).hexdigest()[:8]}"
            ids.append(msg_id)
            emb_list.append(embeddings[i])
            extracted, _ = await self._extract_code_blocks(msg["content"])
            content_type = self._classify_content(msg["content"], extracted)
            expires_at = (
                now + (self.valves.long_term_memory_expiration_days * 86400)
                if self.valves.long_term_memory_expiration_days > 0
                else None
            )
            code_symbols_str = ""
            if self.valves.ltm_index_symbols_enabled:
                blocks_for_symbols = extracted if extracted else []
                if not blocks_for_symbols:
                    blocks_for_symbols, _ = await self._extract_code_blocks(
                        msg["content"]
                    )
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
            documents.append(msg["content"])
        if ids:
            await anyio.to_thread.run_sync(
                lambda: self.memory_collection.upsert(
                    ids=ids,
                    embeddings=emb_list,
                    metadatas=metadatas,
                    documents=documents,
                )
            )
        self._log_timing(
            "batch_ltm_total", time.monotonic() - t_start, time.monotonic() - t_start
        )

    async def _flush_ltm_batch(self, project_id: str):
        await asyncio.sleep(0.5)
        async with self._ltm_batch_lock:
            if not self._pending_ltm_messages:
                return
            messages_to_store = self._pending_ltm_messages.copy()
            self._pending_ltm_messages.clear()
            self._ltm_batch_task = None
        await self._batch_store_messages(project_id, messages_to_store)

    async def _store_message_in_memory(self, message: dict, project_id: str):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        content = message.get("content", "")
        if not content or len(content.strip()) < 15:
            return
        embedding = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(content).tolist()
        )
        extracted, _ = await self._extract_code_blocks(content)
        content_type = self._classify_content(content, extracted)
        msg_id = f"{project_id}_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        expires_at = (
            time.time() + (self.valves.long_term_memory_expiration_days * 86400)
            if self.valves.long_term_memory_expiration_days > 0
            else None
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
        self._log_debug(f"Stored message {msg_id} in LTM")

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
                    include=["documents", "metadatas", "distances"],
                )
            )

            docs_with_meta = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    raw_sim = 1.0 - (results["distances"][0][i] / 2.0)
                    ts = meta.get("timestamp")
                    if ts is not None and ts < 1000000000:
                        ts = None

                    if self.valves.ltm_time_decay_hours > 0 and ts is not None:
                        age_hours = (now - ts) / 3600
                        effective_sim = raw_sim * (
                            0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                        )
                    else:
                        effective_sim = raw_sim

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

                # Auto‑summaries for symbols missing summaries
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

            # ── Eviction by max_active_blocks (only if limit > 0) ──
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
                    # Do NOT remove symbols from the index – they may still be needed for context
                    del state["active_blocks"][h]
                if to_remove:
                    self._log_debug(
                        f"Evicted {len(to_remove)} blocks due to max_active_blocks limit. "
                        f"Their symbols remain in the index for lightweight context."
                    )

            # ── Session summary (still deferred because not needed for current prompt) ──
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
                            summary_block._cached_token_count = (
                                len(summary_content) // 4
                            )
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
    # Smart pre‑expand
    # --------------------------------------------------------------------------
    async def _ensure_full_code_intent_embeddings(self):
        if self._full_code_intent_embeddings is not None:
            return
        if self.embedder is None:
            self._log_debug(
                "Embedder not available, skipping full code intent embeddings"
            )
            return
        self._full_code_intent_embeddings = await anyio.to_thread.run_sync(
            lambda: self.embedder.encode(
                self._full_code_intent_phrases, convert_to_numpy=True
            )
        )

    async def _smart_pre_expand(
        self,
        user_query: str,
        project_id: str,
        token_budget: int = 0,
        seen_hashes: Optional[Set[str]] = None,
    ) -> str:
        if not self.valves.smart_pre_expand_enabled:
            return ""

        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""

        all_names = self._symbol_index.get_all_names(project_id)
        if not all_names:
            return ""

        effective_budget = token_budget or self.valves.smart_pre_expand_max_tokens
        needed_symbols: Set[str] = set()
        used_minimum_expansion = False

        # A) Direct mention
        words = set(re.findall(r"\b\w+\b", user_query))
        directly_mentioned = all_names.intersection(words)
        needed_symbols.update(directly_mentioned)

        # B) Optional LLM detection (embedding‑based detection has been removed)
        if (
            self.valves.smart_pre_expand_use_llm
            and not needed_symbols
            and len(user_query) > 20
        ):
            lightweight_ctx = await self._build_lightweight_context(project_id)
            available_list = ", ".join(sorted(all_names)[:60])
            prompt = (
                f"Symbol index:\n{lightweight_ctx[:2000]}\n\n"
                f'User query: "{user_query[:300]}"\n\n'
                f"Available symbols: {available_list}\n\n"
                f"Which symbols need their full source code to answer this query? "
                f"Output only a comma-separated list of names, or 'none'."
            )
            response = await self._try_llm_quick(
                prompt=prompt,
                system_prompt="You are a code context manager. Output only a comma-separated list of symbol names or 'none'.",
                model_override=self.valves.smart_pre_expand_model,
                max_tokens=100,
                temperature=0.0,
            )
            if response and response.strip().lower() != "none":
                detected = {
                    name.strip()
                    for name in response.split(",")
                    if name.strip() in all_names
                }
                needed_symbols.update(detected)

        # C) Minimum expansion fallback
        if not needed_symbols and self.valves.smart_pre_expand_min_symbols > 0:
            used_minimum_expansion = True
            min_token_budget = self.valves.smart_pre_expand_min_tokens
            top_blocks = sorted(
                state["active_blocks"].values(),
                key=lambda b: b.importance_score,
                reverse=True,
            )
            tokens_added = 0
            for block in top_blocks:
                if min_token_budget > 0 and tokens_added >= min_token_budget:
                    break
                block_tokens = block._cached_token_count or (len(block.content) // 4)
                for sym in block.symbols:
                    if sym.name in all_names:
                        needed_symbols.add(sym.name)
                tokens_added += block_tokens

        if not needed_symbols:
            return ""

        if used_minimum_expansion and self.valves.smart_pre_expand_min_tokens > 0:
            if effective_budget == 0:
                effective_budget = self.valves.smart_pre_expand_min_tokens
            else:
                effective_budget = min(
                    effective_budget, self.valves.smart_pre_expand_min_tokens
                )

        symbol_priority: List[Tuple[str, "CodeBlock", float]] = []
        for sym_name in needed_symbols:
            block_hashes = self._symbol_index.find_blocks(sym_name, project_id)
            for h in block_hashes:
                block = state["active_blocks"].get(h)
                if block and not block.obsolete:
                    symbol_priority.append((sym_name, block, block.importance_score))
                    break
        symbol_priority.sort(key=lambda x: x[2], reverse=True)

        parts = ["\n## Auto-Expanded Code (retrieved for your query)\n"]
        tokens_used = 0
        expanded_count = 0
        local_seen: Set[str] = set()

        for sym_name, block, _ in symbol_priority:
            if seen_hashes is not None and block.hash in seen_hashes:
                continue
            if block.hash in local_seen:
                continue
            local_seen.add(block.hash)
            if seen_hashes is not None:
                seen_hashes.add(block.hash)

            tok_count = (
                len(self.tokenizer.encode(block.content))
                if self.tokenizer
                else len(block.content) // 4
            )
            if effective_budget > 0 and tokens_used + tok_count > effective_budget:
                remaining = len(symbol_priority) - expanded_count
                if remaining > 0:
                    parts.append(
                        f"[{remaining} more symbol(s) omitted — token budget ({effective_budget}) reached]"
                    )
                break

            loc = f" (file: {block.file_path})" if block.file_path else ""
            parts.append(f"### `{sym_name}`{loc}\n```\n{block.content}\n```")
            tokens_used += tok_count
            expanded_count += 1

            if self.valves.enable_call_graph_extraction:
                for sym in block.symbols:
                    if sym.name != sym_name:
                        continue
                    for callee_name in sym.calls[:3]:
                        if callee_name in self._SYMBOL_BLACKLIST:
                            continue
                        callee_hashes = self._symbol_index.find_blocks(
                            callee_name, project_id
                        )
                        for ch in callee_hashes:
                            callee_block = state["active_blocks"].get(ch)
                            if not callee_block or callee_block.obsolete:
                                continue
                            if (
                                seen_hashes is not None
                                and callee_block.hash in seen_hashes
                            ):
                                break
                            if callee_block.hash in local_seen:
                                break
                            ctok = (
                                len(self.tokenizer.encode(callee_block.content))
                                if self.tokenizer
                                else len(callee_block.content) // 4
                            )
                            if (
                                effective_budget > 0
                                and tokens_used + ctok > effective_budget
                            ):
                                break
                            local_seen.add(callee_block.hash)
                            if seen_hashes is not None:
                                seen_hashes.add(callee_block.hash)
                            callee_loc = (
                                f" (file: {callee_block.file_path})"
                                if callee_block.file_path
                                else ""
                            )
                            parts.append(
                                f"### `{callee_name}` (callee of `{sym_name}`)"
                                f"{callee_loc}\n```\n{callee_block.content}\n```"
                            )
                            tokens_used += ctok
                            break
                    break

        if expanded_count == 0:
            return ""

        return "\n".join(parts)

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

    def _expand_referenced_symbols(
        self, project_id: str, user_query: str, seen_hashes: Optional[Set[str]] = None
    ) -> str:
        """Expand symbols mentioned directly or via file paths in the query."""
        state = self._get_state(project_id)
        if not state:
            return ""
        all_names = self._symbol_index.get_all_names(project_id)
        words = set(re.findall(r"\b\w+\b", user_query))
        mentioned = all_names.intersection(words)

        parts = []
        for name in sorted(mentioned):
            blocks = self._symbol_index.find_blocks(name, project_id)
            for h in blocks:
                block = state["active_blocks"].get(h)
                if block and not block.obsolete:
                    if seen_hashes is not None:
                        if block.hash in seen_hashes:
                            continue
                        seen_hashes.add(block.hash)
                    loc = f" (file: {block.file_path})" if block.file_path else ""
                    parts.append(f"### `{name}`{loc}\n```\n{block.content[:2000]}\n```")
                    break
        return "\n".join(parts) if parts else ""

    # --------------------------------------------------------------------------
    # Intent detection (natural language)
    # --------------------------------------------------------------------------

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

    async def _generate_cot(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nThink step by step and provide your reasoning:"
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a helpful assistant that thinks step by step before answering.",
            model_override=self.valves.cot_model,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.4,
        )
        return response if response else "Unable to generate reasoning."

    async def _detect_cot_level(self, user_content, is_code_session, state):
        """Determine CoT depth, optionally storing it in conversation state."""
        if not user_content:
            return 0

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
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nThink step by step and provide your reasoning:"
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a helpful assistant that thinks step by step before answering.",
            model_override=self.valves.cot_model_level2,
            max_tokens=effective_max_tokens,
            temperature=0.4,
            label=label,
        )
        if response:
            return (
                f"## 🔎 Automated Chain-of-Thought Reasoning (Level 2)\n"
                f"*This section was generated by {self.valves.cot_model_level2} "
                f"to assist the main assistant. It is not user input.*\n\n"
                f"{response}"
            )
        return "Unable to generate reasoning."

    async def _generate_cot_with_self_reflection(
        self, question: str, context: str, label: str = ""
    ) -> str:
        """Generate CoT reasoning with self-reflection, using safe model switching."""
        # Generate initial reasoning (level 2)
        reasoning = await self._generate_cot_reasoning(question, context, label=label)
        if not reasoning or reasoning == "Unable to generate reasoning.":
            return reasoning

        effective_max_tokens = (
            self.valves.cot_max_tokens if self.valves.cot_max_tokens > 0 else None
        )
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
            max_tokens=effective_max_tokens,
            temperature=0.3,
            label=label + "_reflection" if label else "cot_reflection",
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
    # Assumption extraction helpers
    # --------------------------------------------------------------------------
    async def _parse_assumption_intent(self, user_content: str) -> Optional[str]:
        if user_content.strip().startswith("/assume"):
            target = user_content.strip()[7:].strip()
            return target if target else None
        return None

    async def _extract_assumptions(self, target: str) -> str:
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
    # Structural / code review helpers
    # --------------------------------------------------------------------------
    async def _is_structural_task(self, user_query: str) -> bool:
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
    # Auto summaries for missing symbol docstrings
    # --------------------------------------------------------------------------
    async def _generate_missing_summaries(self, project_id: str):
        if not self.valves.enable_auto_summaries:
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

        if self.valves.defer_secondary_tasks:
            for sym, code_snippet in symbols_to_summarize:
                task = SecondaryTask(
                    task_type="missing_summaries",
                    params={
                        "signature": sym.signature,
                        "code_snippet": code_snippet,
                        "project_id": project_id,
                    },
                )
                state.setdefault("pending_secondary_tasks", []).append(task.dict())
            self._set_state(project_id, state)
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
                        semaphore=self._low_priority_llm_semaphore,
                    )
                )
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for (sym, _), resp in zip(batch, responses):
                if isinstance(resp, str) and resp.strip():
                    sym.summary = resp.strip()
            await asyncio.sleep(1.0)
        self._set_state(project_id, state)

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
            self._symbol_analysis_cache.clear()
        self._last_project_id = project_id

        if project_id not in self._symbol_cache_loaded_projects:
            await self._load_symbol_cache_from_db(project_id)
            self._symbol_cache_loaded_projects.add(project_id)

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
    ) -> Tuple[List[Tuple[str, str]], Optional[dict], str]:
        """Build all system injections: LTM, code context, confidence, etc.
        Returns (system_injections, cached_response, prelim_system).
        """
        system_injections: List[Tuple[str, str]] = []

        # LTM retrieval
        ltm_future = None
        if (
            self.valves.enable_code_awareness
            and is_code_session
            and not self.valves.smart_context_selection
            and HAS_SENTENCE
            and HAS_CHROMA
        ):
            if user_query:
                ltm_future = asyncio.create_task(
                    self._retrieve_all_memories_unified(user_query, project_id)
                )

        # Parallel checks (contradictions, cached response, duplicate question)
        context_hash = self._compute_context_hash(messages)
        contradiction_warning = None
        cached_response = None
        duplicate_match = None

        if last_user_msg:
            parallel_checks_task = asyncio.create_task(
                self._parallel_context_checks(
                    messages, user_query, context_hash, project_id, state
                )
            )
            contradiction_warning, cached_response, duplicate_match = (
                await parallel_checks_task
            )

        if cached_response:
            # Will be handled by caller; we just return early
            return [], cached_response, ""

        if contradiction_warning and self.valves.contradiction_inject_warning:
            system_injections.append(("high", contradiction_warning))
        if duplicate_match:
            warn_msg = f"⚠️ **Note**: This question is very similar to one you asked before (similarity {duplicate_match['sim']:.2f})."
            system_injections.append(("medium", warn_msg))

        # Wait for LTM and format
        if ltm_future is not None:
            all_meta = await ltm_future
            all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
            unique_meta = []
            seen = set()
            for m in all_meta:
                if m["doc"] not in seen:
                    seen.add(m["doc"])
                    unique_meta.append(m)

            max_ltm_tokens = self.valves.ltm_retrieval_max_tokens
            parts = []
            current_tokens = 0
            header = "## Relevant Past Context (with timestamps)\n\n"
            if max_ltm_tokens > 0 and self.tokenizer:
                current_tokens += len(self.tokenizer.encode(header))
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
                    len(self.tokenizer.encode(text))
                    if self.tokenizer
                    else (len(text) // 4)
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

        # Proactive cleanup suggestion
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

        # ============== CODE INJECTION ==============
        if is_code_session and self.valves.enable_code_awareness:
            code_blocks_for_injection = [
                b
                for b in state["active_blocks"].values()
                if b.content_type
                in (
                    ContentType.BASE_CODE,
                    ContentType.COMMITTED_CHANGE,
                    ContentType.PROPOSED_CHANGE,
                )
                and not b.obsolete
            ]
            total_code_tokens = sum(
                b._cached_token_count for b in code_blocks_for_injection
            )

            if self.valves.use_symbol_level_analysis:
                summary, suggested = await self._analyze_code_via_symbols(
                    user_question, project_id
                )
                if summary:
                    system_injections.append(("critical", summary))
                if suggested:
                    suggested_blocks = self._get_blocks_for_symbols(
                        list(suggested), project_id
                    )
                    if suggested_blocks:
                        extra_lines = []
                        tokens_used = 0
                        max_sugg_tokens = min(
                            self.valves.active_context_max_tokens or 3000,
                            3000,
                        )
                        for blk in suggested_blocks[:5]:
                            bt = blk._cached_token_count
                            if (
                                max_sugg_tokens > 0
                                and tokens_used + bt > max_sugg_tokens
                            ):
                                break
                            loc = f" (file: {blk.file_path})" if blk.file_path else ""
                            extra_lines.append(
                                f"**{blk.hash[:8]}**{loc}\n```\n{blk.content[:3000]}\n```"
                            )
                            tokens_used += bt
                        if extra_lines:
                            system_injections.append(
                                (
                                    "high",
                                    "## Additional suggested code\n\n"
                                    + "\n".join(extra_lines),
                                )
                            )
            else:
                is_structural = (
                    await self._is_structural_task(user_question)  # ← Bug #2 corregido
                    if user_question
                    else False
                )
                if total_code_tokens > self.valves.huge_injection_threshold_tokens > 0:
                    active_ctx = await self._build_lightweight_context(project_id)
                    injected_hashes: Set[str] = set()
                    if user_question:
                        pre_expanded = await self._smart_pre_expand(
                            user_query=user_question,  # ← Bug #2 corregido
                            project_id=project_id,
                            token_budget=self.valves.smart_pre_expand_max_tokens,
                            seen_hashes=injected_hashes,
                        )
                        if pre_expanded:
                            active_ctx += "\n" + pre_expanded
                        else:
                            expanded = self._expand_referenced_symbols(
                                project_id, user_question, seen_hashes=injected_hashes
                            )
                            if expanded:
                                active_ctx += "\n" + expanded
                    if is_structural:
                        active_ctx += (
                            "\n\n[Note: Structural analysis requested. "
                            "Full code bodies have been pre-expanded above where available.]"
                        )
                else:
                    active_ctx = self._get_active_code_context(
                        project_id, user_query=user_query
                    )
                    if user_question and not is_structural:
                        expanded = self._expand_referenced_symbols(
                            project_id, user_question  # ← Bug #2 corregido
                        )
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

        # Confidence scoring
        if self.valves.enable_confidence_scoring and is_code_session:
            system_injections.append(("high", self.valves.confidence_prompt))

        # Feedback context
        if (
            is_code_session
            and self.valves.enable_feedback_tracking
            and self.valves.inject_feedback_context
        ):
            feedback_ctx = self._get_feedback_context(project_id)
            if feedback_ctx:
                system_injections.append(("high", feedback_ctx))

        # Proactive summary suggestion
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._estimate_tokens(system_msgs + history_msgs)
        if self.valves.context_window_tokens > 0:
            suggestion = await self._check_and_suggest_summarization(
                project_id, total_tokens, self.valves.context_window_tokens
            )
            if suggestion:
                system_injections.append(("medium", suggestion))

        # Command suggestion
        cmd_suggestion = await self._suggest_commands(project_id, state)
        if cmd_suggestion:
            system_injections.append(("medium", cmd_suggestion))

        # Build preliminary system text respecting token budget
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        base_content = ""
        if sys_msgs:
            base_content = sys_msgs[0].get("content", "")

        budget = self.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self.tokenizer:
            system_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_texts = []
            total_inj_tokens = 0
            for prio, text in system_injections:
                if not text:
                    continue
                tokens = len(self.tokenizer.encode(text))
                if total_inj_tokens + tokens <= budget:
                    selected_texts.append(text)
                    total_inj_tokens += tokens
                else:
                    if prio in ("critical", "high"):
                        available = budget - total_inj_tokens
                        if available > 20:
                            truncated = text[: available * 4] + "\n[truncated]"
                            selected_texts.append(truncated)
                            total_inj_tokens += len(self.tokenizer.encode(truncated))
                            break
            prelim_system = "\n\n".join(selected_texts)
        else:
            prelim_system = "\n\n".join(text for _, text in system_injections if text)

        if base_content.strip():
            prelim_system = prelim_system + "\n\n" + base_content

        return system_injections, None, prelim_system

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

        model = self.valves.symbol_analysis_model or self.valves.llm_model
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
        system_injections: List[Tuple[str, str]],
        prelim_system: str,
        last_user_msg: Optional[dict],
        is_code_session: bool,
        state: dict,
        __user__: Optional[dict],
        background_tasks: List[asyncio.Task],
        user_question: str,
        has_code_blocks: bool,
    ) -> List[dict]:
        """Apply CoT, final token budget, trimming, and insert system prompt."""

        # Determine user intent for context reduction (also used by CoT detection)
        self._user_intent_full_code = await self._should_keep_full_code(user_question)

        # CoT detection
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
                            system_injections.append(("high", cot_prompt))
                elif not manual_cot_used:
                    cot_level = await self._detect_cot_level(
                        user_content, is_code_session, state
                    )
                    self._log_debug(f"CoT level detected: {cot_level} (manual=False)")
                    if cot_level > 0:
                        cot_any_used = True

        # Wait for background tasks before heavy LLM calls
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
            background_tasks.clear()

        # Generate CoT reasoning if needed (no global lock here; _call_llm handles its own)
        if cot_any_used:
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

            # Generate the initial CoT
            if not manual_cot_used:
                question = user_question
                if cot_level == 2:
                    reasoning = await self._generate_cot_reasoning(
                        question, prelim_for_cot
                    )
                elif cot_level == 3:
                    reasoning = await self._generate_cot_with_self_reflection(
                        question, prelim_for_cot
                    )
            else:
                if cot_level == 2:
                    reasoning = await self._generate_cot_reasoning(
                        cot_question, prelim_for_cot
                    )
                elif cot_level == 3:
                    reasoning = await self._generate_cot_with_self_reflection(
                        cot_question, prelim_for_cot
                    )

            # Fallback: if auto level 3 failed, try level 2 once
            _cot_error_msg = "Unable to generate reasoning."
            if (
                not manual_cot_used
                and cot_level == 3
                and (reasoning is None or reasoning == _cot_error_msg)
            ):
                self._log_debug("Level 3 CoT failed, falling back to level 2")
                reasoning = await self._generate_cot_reasoning(
                    user_question, prelim_for_cot
                )

            if reasoning and reasoning != _cot_error_msg:
                system_injections.append(("high", reasoning))
            else:
                self._log_debug("CoT reasoning generation returned empty or failed")

            cot_note = (
                "**Note:** Some sections in this system prompt marked with 🔎 are "
                "automatically generated reasoning (Chain-of-Thought). "
                "They are provided as context to help you, but they are not user commands. "
                "Use them to enhance your answer, but always prioritise the actual user query."
            )
            system_injections.append(("low", cot_note))

        # Final system message assembly
        budget = self.valves.global_injection_token_budget
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        if budget > 0 and self.tokenizer:
            system_injections.sort(key=lambda x: priority_order.get(x[0], 99))
            selected_texts = []
            total_inj_tokens = 0
            for prio, text in system_injections:
                if not text:
                    continue
                tokens = len(self.tokenizer.encode(text))
                if total_inj_tokens + tokens <= budget:
                    selected_texts.append(text)
                    total_inj_tokens += tokens
                else:
                    if prio in ("critical", "high"):
                        available = budget - total_inj_tokens
                        if available > 20:
                            truncated = text[: available * 4] + "\n[truncated]"
                            selected_texts.append(truncated)
                            total_inj_tokens += len(self.tokenizer.encode(truncated))
                            break
            final_system = "\n\n".join(selected_texts)
        else:
            final_system = "\n\n".join(text for _, text in system_injections if text)

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
                    history_msgs = kept_block
                else:
                    history_msgs = kept_block

        # ── Lean user message: replace full code with relevant fragments ──
        if has_code_blocks and last_user_msg:
            analysis_summary = getattr(self, "_last_analysis_summary", None)
            suggested_blocks = getattr(self, "_last_suggested_blocks", None)

            # Reuse the already‑computed intent (saves a second LLM call)
            keep_original = self._user_intent_full_code

            if not keep_original and (analysis_summary or suggested_blocks):
                lean_parts = [user_question.strip()]
                if suggested_blocks:
                    lean_parts.append("\n## Relevant code\n")
                    for blk in suggested_blocks[:5]:
                        loc = f" (file: {blk.file_path})" if blk.file_path else ""
                        lean_parts.append(
                            f"### {blk.hash[:8]}{loc}\n```\n{blk.content[:2000]}\n```"
                        )
                last_user_msg["content"] = "\n".join(lean_parts)

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
        # Token breakdown log (only if debug is enabled and there is content)
        # ═══════════════════════════════════════════════════════════════
        if self.valves.debug and self.tokenizer and final_system.strip():
            total_system_tokens = len(self.tokenizer.encode(final_system))
            ltm_tokens = 0
            summary_tokens = 0
            suggested_tokens = 0
            cot_tokens = 0
            other_tokens = total_system_tokens

            for _, text in system_injections:
                if not text:
                    continue
                t = len(self.tokenizer.encode(text))
                if "Relevant Past Context" in text:
                    ltm_tokens += t
                elif "synthesized" in text.lower() or "Synthesize" in text:
                    summary_tokens += t
                elif "Additional suggested code" in text:
                    suggested_tokens += t
                elif reasoning and text == reasoning:
                    cot_tokens += t

            other_tokens = total_system_tokens - (
                ltm_tokens + summary_tokens + suggested_tokens + cot_tokens
            )

            self._log_debug("─" * 50)
            self._log_debug("TOKEN BREAKDOWN – injected into system prompt")
            self._log_debug(f"  LTM (past messages, no LLM call):     ~{ltm_tokens}")
            self._log_debug(
                f"  Summary (symbol analysis synthesis):   ~{summary_tokens}"
            )
            self._log_debug(
                f"  Suggested code (verbatim source):      ~{suggested_tokens}"
            )
            self._log_debug(f"  CoT reasoning (LLM generated):         ~{cot_tokens}")
            self._log_debug(f"  Other (confidence, cleanup, etc.):     ~{other_tokens}")
            self._log_debug(
                f"  TOTAL injected system tokens:          ~{total_system_tokens}"
            )
            self._log_debug("─" * 50)
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
        # 🔥 STATE MANAGEMENT (Critical)
        #   1. Preprocess (project switch, cache load)
        #   2. Process pending secondary tasks (session summaries, etc.)
        #   4. Extract user info (last message, question, code blocks)
        # ─────────────────────────────────────────────────────────────────
        messages = await self._inlet_preprocess(body, project_id)
        if not messages:
            return body

        await self._process_pending_secondary_tasks(project_id)

        # ─────────────────────────────────────────────────────────────────
        # 🚀 RESOURCE OPTIMISATION (Critical)
        #   3. Free VRAM safely (global lock, avoids slot conflicts)
        # ─────────────────────────────────────────────────────────────────
        await self._unload_models_under_lock()

        # ─────────────────────────────────────────────────────────────────
        # 🔥 STATE MANAGEMENT (Critical)
        #   (continued) 4. Extract user info
        # ─────────────────────────────────────────────────────────────────
        (
            last_user_msg,
            user_query,
            user_question,
            is_explicit_command,
            has_code_blocks,
        ) = await self._inlet_extract_user_info(messages)

        # ─────────────────────────────────────────────────────────────────
        # ⚡ COMMAND HANDLING (High value)
        #   5. Explicit commands (/forget, /status, /clean, /expand)
        # ─────────────────────────────────────────────────────────────────
        handled, handled_messages = await self._inlet_handle_explicit_commands(
            messages, project_id, is_explicit_command, last_user_msg, __user__
        )
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
            )
            return body

        # ─────────────────────────────────────────────────────────────────
        # ⚡ COMMAND HANDLING (High value)
        #   6. Natural language intents (forget, remember, obsolete)
        # ─────────────────────────────────────────────────────────────────
        handled, handled_messages = await self._inlet_handle_natural_intents(
            messages, project_id, is_explicit_command, last_user_msg
        )
        if handled:
            body["messages"] = handled_messages
            _inlet_timing("total_inlet (end-to-end)", inlet_start)
            self._log_section(
                "CONTEXT MANAGER - INLET END", duration=time.monotonic() - inlet_start
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
            is_code_session, user_question = await self._inlet_prepare_code_session(
                messages, project_id, user_query
            )

            # Wait for any background tasks launched by _update_active_code
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()

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
            # ─────────────────────────────────────────────────────────────
            system_injections, cached_response, prelim_system = (
                await self._inlet_build_system_injections(
                    messages,
                    project_id,
                    user_query,
                    user_question,
                    is_code_session,
                    last_user_msg,
                    state,
                )
            )

            # 🚀 RESOURCE OPTIMISATION (High value)
            #    Return cached response immediately if found
            if isinstance(cached_response, dict):
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
            # ─────────────────────────────────────────────────────────────
            messages = await self._inlet_assemble_final_messages(
                messages,
                system_injections,
                prelim_system,
                last_user_msg,
                is_code_session,
                state,
                __user__,
                background_tasks,
                user_question,
                has_code_blocks,
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
            #   - Free VRAM for the main model
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()
            await self._unload_models_under_lock()

            if _inlet_aborted:
                for task in background_tasks:
                    if not task.done():
                        task.cancel()
            _inlet_background_tasks.reset(token)

        return body

    # ═══════════════════════════════════════════════════════════════════════════
    # OUTLET
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
                    if (
                        last_msg.get("role") == "assistant"
                        and is_code_session
                        and "/expand" in last_msg.get("content", "")
                    ):
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

                    if last_msg.get("role") in ("user", "assistant"):
                        if is_code_session:
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
                                async with self._ltm_batch_lock:
                                    self._pending_ltm_messages.append(last_msg)
                                    if (
                                        self._ltm_batch_task is None
                                        or self._ltm_batch_task.done()
                                    ):
                                        self._ltm_batch_task = asyncio.create_task(
                                            self._flush_ltm_batch(project_id)
                                        )

            # Response cache storage (with code_state_hash precomputed to avoid extra lock)
            if (
                self.valves.enable_response_cache
                and HAS_SENTENCE
                and len(messages) >= 2
            ):
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

            # Purge expired memories only if no purge is already running
            if self._purge_task is None or self._purge_task.done():
                self._purge_task = asyncio.create_task(self._purge_expired_memories())

            self._write_counter += 1
            if self._write_counter % 100 == 0:
                self._purge_task = asyncio.create_task(self._run_db_checkpoints())
                self._cleanup_completed_tasks()

            # Save state if dirty (debounced)
            await self._save_state_if_dirty(project_id)

            # Free VRAM for the main model
            await self._unload_models_under_lock()
        finally:
            # No secondary worker to resume – nothing to do
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

        # Cancel the database worker cleanly – do NOT try to create a new task
        self._db_worker_task.cancel()

        # Cancel the secondary task worker
        if hasattr(self, "_secondary_task_worker_task"):
            self._secondary_task_worker_task.cancel()

        # Cancel dependency tasks and wait for them to finish
        for task in self._dependency_tasks:
            task.cancel()
        if self._dependency_tasks:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.run_until_complete(
                        asyncio.gather(*self._dependency_tasks, return_exceptions=True)
                    )
            except Exception:
                pass

        # Clear in‑memory structures
        self._symbol_index.clear()
        self._cached_lightweight_context.clear()
        self._project_locks.clear()

        # Shut down thread pools
        self._db_executor.shutdown(wait=True)
        self._chroma_executor.shutdown(wait=True)

    def _cleanup_completed_tasks(self):
        self._dependency_tasks = [t for t in self._dependency_tasks if not t.done()]

    # --------------------------------------------------------------------------
    # Miscellaneous helpers
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

        model = self.valves.secondary_task_model or self.valves.smart_pre_expand_model
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
    # Symbol-level analysis
    # --------------------------------------------------------------------------
    def _build_symbol_context(
        self,
        sym: CodeSymbol,
        block: CodeBlock,
        project_id: str,
        max_tokens: int = 800,
        max_body_lines: int = 20,
    ) -> Optional[str]:
        sig = self._sanitize_signature(sym.signature or sym.name)
        if not sig.strip():
            return None
        ctx = f"Symbol: `{sig}` [{sym.kind}]"
        if sym.file_path:
            ctx += f" in {sym.file_path}"
        if sym.summary:
            ctx += f"\nSummary: {sym.summary}"

        clean_body = self._sanitize_text(block.content)
        lines = clean_body.splitlines()
        preview_lines = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if len(line) > 200:
                line = line[:200] + "…"
            preview_lines.append(line)
            if len(preview_lines) >= max_body_lines:
                break
        if preview_lines:
            body = "\n".join(preview_lines)
            body = textwrap.shorten(body, width=2000, placeholder="...")
            ctx += f"\nBody preview:\n```\n{body}\n```"

        callers = self._symbol_index.get_callers(sym.name, project_id)
        if callers:
            ctx += f"\nCalled by: {', '.join(sorted(callers)[:5])}"
        if sym.calls:
            ctx += f"\nCalls: {', '.join(sym.calls[:5])}"

        if self.tokenizer:
            tokens = len(self.tokenizer.encode(ctx))
            if tokens > max_tokens:
                if "\nBody preview:\n" in ctx:
                    header, body = ctx.split("\nBody preview:\n", 1)
                    ctx_no_body = header + "\nBody preview omitted (too large)."
                    if (
                        self.tokenizer
                        and len(self.tokenizer.encode(ctx_no_body)) > max_tokens
                    ):
                        ctx = header
                    else:
                        ctx = ctx_no_body
                if self.tokenizer and len(self.tokenizer.encode(ctx)) > max_tokens:
                    ctx = self._truncate_text_to_tokens(ctx, max_tokens)
        return ctx

    async def _analyze_code_via_symbols(
        self, question: str, project_id: str
    ) -> Tuple[str, List[str]]:
        """
        Analyze code symbols in batches to keep RAM usage low.
        Each batch produces an intermediate summary; a final summary
        is synthesized from those intermediates.
        Returns (final_summary, list_of_suggested_symbols).
        """
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return "", []

        question_hash = hashlib.md5(question.encode()).hexdigest()[:12]

        # ── Collect symbol contexts (lazy, no LLM calls yet) ────────────
        symbol_entries = []  # (name, importance_score, block_hash, ctx_or_cached)
        seen_symbols = set()
        for block in state["active_blocks"].values():
            if block.obsolete or block.content_type not in (
                ContentType.BASE_CODE,
                ContentType.COMMITTED_CHANGE,
                ContentType.PROPOSED_CHANGE,
            ):
                continue
            for sym in block.symbols:
                if sym.name in seen_symbols:
                    continue
                seen_symbols.add(sym.name)

                cached = self._get_cached_symbol_analysis(
                    sym.name, question_hash, content_hash=block.hash
                )
                if cached is not None:
                    cached["symbol_name"] = sym.name
                    symbol_entries.append(
                        (sym.name, block.importance_score, block.hash, cached)
                    )
                    continue

                ctx = self._build_symbol_context(sym, block, project_id)
                if ctx is None:
                    continue
                symbol_entries.append(
                    (sym.name, block.importance_score, block.hash, ctx)
                )

        if not symbol_entries:
            return "", []

        # Sort by importance descending so that the most relevant blocks are processed first
        symbol_entries.sort(key=lambda x: x[1], reverse=True)

        total_symbols = len(symbol_entries)
        batch_size = self.valves.symbol_batch_size
        intermediate_summaries = []
        all_suggested = set()

        # ── Process in batches ─────────────────────────────────────────
        for batch_start in range(0, total_symbols, batch_size):
            batch_num = batch_start // batch_size + 1
            total_batches = (total_symbols + batch_size - 1) // batch_size
            batch = symbol_entries[batch_start : batch_start + batch_size]
            self._log_debug(
                f"Symbol analysis batch {batch_num}/{total_batches} "
                f"({len(batch)} symbols)"
            )

            # Separate fresh vs cached
            fresh_entries = []
            cached_results = []
            for name, _, block_hash, ctx_or_cached in batch:
                if isinstance(ctx_or_cached, dict):
                    cached_results.append(ctx_or_cached)
                else:
                    fresh_entries.append((name, block_hash, ctx_or_cached))

            # Analyze fresh symbols in parallel
            if fresh_entries:
                prompt_template_full = (
                    "You are a code analysis assistant. "
                    "For the given symbol, provide:\n"
                    "FUNCTIONS: <comma separated list of relevant function/class names>\n"
                    "FINDINGS: <one key finding>\n"
                    "ISSUES: <any potential issue, or 'none'>\n"
                    "SUGGEST: <suggested next symbol to explore, or 'none'>\n"
                    "CONFIDENCE: <float 0.0-1.0>\n\n"
                    "EXAMPLE:\n"
                    "Symbol: `def calculate(x, y)` in math_utils.py\n"
                    "Body preview:\n```\nreturn x / y\n```\n"
                    "FUNCTIONS: calculate\n"
                    "FINDINGS: Performs division\n"
                    "ISSUES: Division by zero\n"
                    "SUGGEST: validate_input\n"
                    "CONFIDENCE: 0.9\n\n"
                    "Now analyze:\n{context}\n"
                    "OUTPUT:"
                )
                prompt_template_no_body = (
                    "You are a code analysis assistant. "
                    "For the given symbol, provide:\n"
                    "FUNCTIONS: <comma separated list of relevant function/class names>\n"
                    "FINDINGS: <one key finding>\n"
                    "ISSUES: <any potential issue, or 'none'>\n"
                    "SUGGEST: <suggested next symbol to explore, or 'none'>\n"
                    "CONFIDENCE: <float 0.0-1.0>\n\n"
                    "Now analyze (only signature and call info available):\n{context}\n"
                    "OUTPUT:"
                )

                model = self.valves.symbol_analysis_model or self.valves.llm_model
                semaphore = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENT_CALLS)
                max_retries = self.valves.symbol_analysis_max_retries

                parsed_fresh = []
                for name, block_hash, ctx in fresh_entries:
                    success = False
                    for attempt in range(max_retries):
                        if attempt == 0:
                            prompt = prompt_template_full.format(context=ctx)
                        elif attempt == 1:
                            ctx_clean = ctx.split("\nBody preview:\n")[0]
                            prompt = prompt_template_no_body.format(context=ctx_clean)
                        else:
                            ctx_sig = ctx.split("\n")[0]
                            prompt = prompt_template_no_body.format(context=ctx_sig)

                        try:
                            res = await self._analyze_single_symbol(
                                prompt, model, semaphore, label=f"symbol:{name}"
                            )
                        except Exception:
                            continue

                        if not res:
                            continue

                        parsed = self._parse_symbol_output(res)
                        if parsed:
                            parsed["symbol_name"] = name
                            parsed_fresh.append(parsed)
                            self._set_cached_symbol_analysis(
                                name, question_hash, parsed, content_hash=block_hash
                            )
                            success = True
                            break

                    if not success:
                        self._log_debug(f"Symbol analysis failed for {name}")

                # Combine with cached for this batch
                batch_analyses = cached_results + parsed_fresh
            else:
                batch_analyses = cached_results

            if batch_analyses:
                # Collect suggested symbols
                for r in batch_analyses:
                    if r.get("suggested_next") and r["suggested_next"] != "none":
                        all_suggested.add(r["suggested_next"])

                # Generate an intermediate summary for this batch
                batch_summary = await self._synthesize_from_symbol_results(
                    batch_analyses, question
                )
                if batch_summary:
                    intermediate_summaries.append(batch_summary)

            self._log_debug(
                f"Symbol analysis batch {batch_num}/{total_batches} complete"
            )

        if not intermediate_summaries:
            return "Symbol analysis produced no results.", []

        # ── Synthesize final summary from intermediate summaries ──────
        if len(intermediate_summaries) == 1:
            final_summary = intermediate_summaries[0]
        else:
            combined = "\n\n".join(
                f"Batch {i+1}:\n{s}" for i, s in enumerate(intermediate_summaries)
            )
            prompt = (
                f"Question: {question}\n\n"
                f"Intermediate batch summaries:\n{combined}\n\n"
                "Combine these into a single concise summary of the codebase. "
                "Include relevant functions, key findings, issues, and suggestions. "
                "Keep the response under {max_tokens} tokens. No code snippets."
            )
            model = self.valves.symbol_analysis_model or self.valves.llm_model
            final_summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a senior software architect summarizing code analysis.",
                model_override=model,
                max_tokens=self.valves.synthesis_max_tokens,
                temperature=0.2,
            )
            final_summary = final_summary or "\n".join(intermediate_summaries)

        return final_summary, list(all_suggested)

    async def _analyze_single_symbol(
        self, prompt: str, model: str, semaphore: asyncio.Semaphore, label: str = ""
    ) -> Optional[str]:
        async with semaphore:
            return await self._call_llm(
                prompt=prompt,
                system_prompt="You are a code analysis engine. Output only the structured text as requested.",
                model_override=model,
                max_tokens=200,
                temperature=0.0,
                label=label,
            )

    def _parse_symbol_output(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        try:
            result = {}
            lines = text.strip().splitlines()
            for line in lines:
                line = line.strip()
                if line.startswith("FUNCTIONS:"):
                    result["relevant_functions"] = [
                        f.strip() for f in line[10:].split(",") if f.strip()
                    ]
                elif line.startswith("FINDINGS:"):
                    result["key_findings"] = [line[9:].strip()]
                elif line.startswith("ISSUES:"):
                    issues = line[7:].strip()
                    if issues.lower() != "none":
                        result["potential_issues"] = [issues]
                    else:
                        result["potential_issues"] = []
                elif line.startswith("SUGGEST:"):
                    suggest = line[8:].strip()
                    if suggest.lower() != "none":
                        result["suggested_next"] = suggest
                elif line.startswith("CONFIDENCE:"):
                    try:
                        result["confidence"] = float(line[11:].strip())
                    except ValueError:
                        result["confidence"] = 0.5
            result.setdefault("relevant_functions", [])
            result.setdefault("key_findings", [])
            result.setdefault("potential_issues", [])
            result.setdefault("confidence", 0.5)
            return result
        except Exception:
            return None

    async def _synthesize_from_symbol_results(
        self, symbol_analyses: List[Dict], question: str
    ) -> str:
        if not symbol_analyses:
            return "No symbol analyses to synthesize."

        lines = []
        for sa in symbol_analyses:
            lines.append(
                f"`{sa.get('symbol_name', 'unknown')}`: "
                f"functions={sa.get('relevant_functions', [])}, "
                f"findings={sa.get('key_findings', [])}, "
                f"issues={sa.get('potential_issues', [])}"
            )

        combined = "\n".join(lines)
        if self.tokenizer:
            tokens = len(self.tokenizer.encode(combined))
            if tokens > 8000:
                combined = self._truncate_text_to_tokens(combined, 8000)

        prompt = (
            f"Question: {question}\n\n"
            f"Individual symbol analyses:\n{combined}\n\n"
            "Synthesize the above into a concise summary of the codebase. "
            "Include relevant functions, key findings, issues, and suggestions. "
            "Keep the response under 1500 tokens. No code snippets."
        )

        model = self.valves.symbol_analysis_model or self.valves.llm_model
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a senior software architect summarizing code analysis.",
            model_override=model,
            max_tokens=self.valves.synthesis_max_tokens,
            temperature=0.2,
        )
        return response if response else "Failed to synthesize summary."

    def _get_blocks_for_symbols(
        self, symbol_names: List[str], project_id: str
    ) -> List[CodeBlock]:
        state = self._get_state(project_id)
        blocks = []
        seen = set()
        for name in symbol_names:
            for h in self._symbol_index.find_blocks(name, project_id):
                if h not in seen:
                    blk = state["active_blocks"].get(h)
                    if blk and not blk.obsolete:
                        blocks.append(blk)
                        seen.add(h)
        return sorted(blocks, key=lambda b: b.importance_score, reverse=True)

    # --------------------------------------------------------------------------
    # Symbol-level analysis
    # --------------------------------------------------------------------------
    def _get_cached_symbol_analysis(
        self, symbol_name: str, question_hash: str, content_hash: str = ""
    ) -> Optional[Dict]:
        key = (symbol_name, question_hash)
        if key in self._symbol_analysis_cache:
            if content_hash:
                row = self._db_conn.execute(
                    "SELECT content_hash FROM symbol_analysis_cache "
                    "WHERE project_id = ? AND symbol_name = ? AND question_hash = ?",
                    (self.valves.project_id, symbol_name, question_hash),
                ).fetchone()
                if row and row[0] != content_hash:
                    del self._symbol_analysis_cache[key]
                    self._db_conn.execute(
                        "DELETE FROM symbol_analysis_cache "
                        "WHERE project_id = ? AND symbol_name = ? AND question_hash = ?",
                        (self.valves.project_id, symbol_name, question_hash),
                    )
                    self._db_conn.commit()
                    return None
            # Refresh LRU order
            self._symbol_analysis_cache.move_to_end(key)
            return self._symbol_analysis_cache[key]

        generic_key = (symbol_name, content_hash) if content_hash else None
        if generic_key and generic_key in self._symbol_analysis_generic_cache:
            return self._symbol_analysis_generic_cache[generic_key]

        row = self._db_conn.execute(
            "SELECT result_json, content_hash FROM symbol_analysis_cache "
            "WHERE project_id = ? AND symbol_name = ? AND question_hash = ?",
            (self.valves.project_id, symbol_name, question_hash),
        ).fetchone()
        if row:
            result_json, stored_content_hash = row
            if content_hash and stored_content_hash != content_hash:
                self._db_conn.execute(
                    "DELETE FROM symbol_analysis_cache "
                    "WHERE project_id = ? AND symbol_name = ? AND question_hash = ?",
                    (self.valves.project_id, symbol_name, question_hash),
                )
                self._db_conn.commit()
                return None
            try:
                result = json.loads(result_json)
                if len(self._symbol_analysis_cache) < self._MAX_SYMBOL_ANALYSIS_CACHE:
                    self._symbol_analysis_cache[key] = result
                    self._symbol_analysis_cache.move_to_end(key)
                return result
            except Exception:
                pass
        return None

    def _set_cached_symbol_analysis(
        self, symbol_name: str, question_hash: str, result: Dict, content_hash: str = ""
    ):
        key = (symbol_name, question_hash)
        if len(self._symbol_analysis_cache) >= self._MAX_SYMBOL_ANALYSIS_CACHE:
            self._symbol_analysis_cache.popitem(last=False)
        self._symbol_analysis_cache[key] = result
        self._symbol_analysis_cache.move_to_end(key)

        if content_hash:
            generic_key = (symbol_name, content_hash)
            if len(self._symbol_analysis_generic_cache) >= 2000:
                self._symbol_analysis_generic_cache.popitem(last=False)
            self._symbol_analysis_generic_cache[generic_key] = result
            self._symbol_analysis_generic_cache.move_to_end(generic_key)

        project_id = self.valves.project_id
        result_json = json.dumps(result)

        def _write():
            self._db_conn.execute(
                "REPLACE INTO symbol_analysis_cache "
                "(project_id, symbol_name, question_hash, content_hash, result_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    symbol_name,
                    question_hash,
                    content_hash,
                    result_json,
                    time.time(),
                ),
            )
            self._db_conn.commit()
            self._db_conn.execute(
                "DELETE FROM symbol_analysis_cache "
                "WHERE project_id = ? AND (symbol_name, question_hash) NOT IN ("
                "  SELECT symbol_name, question_hash FROM symbol_analysis_cache "
                "  WHERE project_id = ? "
                "  ORDER BY created_at DESC "
                "  LIMIT 1000"
                ")",
                (project_id, project_id),
            )

        asyncio.create_task(self._db_write_queue.put((_write, (), {})))

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

    async def _load_symbol_cache_from_db(self, project_id: str, limit: int = 500):
        """Pre‑load the most recent symbol analyses from DB into memory."""
        rows = await anyio.to_thread.run_sync(
            lambda: self._db_conn.execute(
                "SELECT symbol_name, question_hash, content_hash, result_json "
                "FROM symbol_analysis_cache "
                "WHERE project_id = ? "
                "ORDER BY created_at DESC "
                "LIMIT ?",
                (project_id, limit),
            ).fetchall()
        )
        loaded = 0
        for symbol_name, question_hash, content_hash, result_json in rows:
            try:
                result = json.loads(result_json)
                key = (symbol_name, question_hash)
                self._symbol_analysis_cache[key] = result
                loaded += 1
            except Exception:
                pass
        if loaded:
            self._log_debug(
                f"Loaded {loaded} symbol analyses from DB cache for project {project_id}"
            )
