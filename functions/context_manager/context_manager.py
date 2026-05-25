
"""
title: Code-Aware Context Manager with LTM & Summarization (v5.24.0)
description: Full-featured context manager for coding assistants. Persists state per project, tracks line ranges, applies diffs, compresses LTM, scores importance, learns from responses, summarizes inactive code, supports manual markers, natural language forget/remember commands, feedback tracking, hierarchical memory, LRU cache, optional reranking, dependency detection (AST for Python + regex for other languages), handling of oversized blocks, smart context selection, hierarchical compression, duplicate removal, frequency prioritization, selective summarization, iterative commands, consecutive message deduplication, contradiction detection, chain-of-thought reasoning, assumption extraction, obsolete marking, proactive suggestions, duplicate question detection, command suggestions, semantic response caching, raw file priority boost, and LTM retrieval token limit.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.24.0
license: GPL3
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb, rapidfuzz
"""

import os
import time
import re
import hashlib
import sqlite3
import ast
from collections import OrderedDict, defaultdict
import json
import asyncio
import difflib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
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
# Path for shared resources
# ---------------------------------------------------------------------------
import sys

if "/app/backend/data/custom_lib" not in sys.path:
    sys.path.append("/app/backend/data/custom_lib")

# Shared resources imports (singletons, cache, HTTP session)
try:
    from shared_resources import (  # type: ignore
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
    is_raw: bool = False  # <-- Nuevo: indica que el bloque proviene de un archivo raw

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


class AppliedChangeFeedback(BaseModel):
    change_hash: str
    change_description: str
    file_path: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


class Filter:

    # region ── Valves ─────────────────────────────────────────────────────────

    class Valves(BaseModel):
        priority: int = Field(default=0)
        max_turns: int = Field(default=20)
        debug: bool = Field(
            default=True,
            description="Enable detailed logging.",
        )
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

        # ── Nuevas válvulas para raw files ─────────────────────────────────
        raw_file_priority_boost: float = Field(
            default=2.0,
            description="Extra importance boost for raw file blocks (added during ordering).",
        )
        ltm_retrieval_max_tokens: int = Field(
            default=0,
            description="Maximum total tokens to inject from long‑term memory retrieval. 0 = unlimited.",
        )

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
        auto_cot_enabled: bool = Field(
            default=True,
            description="Automatically inject chain-of-thought prompt for complex questions.",
        )
        auto_cot_min_chars: int = Field(
            default=200,
            description="Minimum message length to consider auto-CoT.",
        )
        enable_code_review_mode: bool = Field(
            default=True,
            description="Injects a rigorous code-review checklist when the user asks for code review.",
        )
        cot_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        cot_max_tokens: int = Field(default=1000)
        enable_assumption_extraction: bool = Field(default=True)
        assumption_extraction_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest"
        )
        enable_contradiction_detection: bool = Field(default=True)
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

        similar_message_handling: str = Field(default="replace")
        similar_message_threshold: float = Field(default=0.85)
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

        llm_request_timeout: int = Field(
            default=300, description="Timeout in seconds for LLM API calls."
        )
        track_active_code_age: bool = Field(default=True)
        active_code_timeout_minutes: int = Field(default=30)

        summarize_inactive_code: bool = Field(default=True)
        inactive_code_summary_model: str = Field(default="ollama/llama3.2:3b")

        llm_model: str = Field(default="ollama/llama3.2:3b")

        enable_forget_command: bool = Field(default=True)
        enable_natural_language_forget: bool = Field(default=True)
        natural_language_forget_model: str = Field(
            default="ollama/Inference/Schematron:3B"
        )

        ltm_store_only_code_sessions: bool = Field(default=True)
        ltm_include_timestamps: bool = Field(
            default=True,
            title="LTM Include Timestamps",
            description="Include the date/time of each retrieved memory fragment in the context.",
        )

        # New valve for LLM cache TTL (C4)
        LLM_CACHE_TTL: int = Field(
            default=300,
            description="TTL in seconds for the in-process LLM response cache (AsyncLRU).",
        )

    # endregion

    # region ── User Valves ────────────────────────────────────────────────────

    class UserValves(BaseModel):
        max_turns: Optional[int] = Field(default=None)
        enable_code_awareness: Optional[bool] = Field(default=None)

    # endregion

    # region ── Init & Configuration ───────────────────────────────────────────

    def __init__(self):
        self.valves = self.Valves()
        self.embedder = None
        self.chroma_client = None
        self.memory_collection = None
        self._response_cache_collection = (
            None  # dedicated ChromaDB collection for response cache (C3)
        )
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

        # HTTP session will be obtained lazily via shared resources (C6)
        self._http_session: Optional[aiohttp.ClientSession] = None

        # LLM cache: use AsyncLRUCache instead of dict (C4)
        self._llm_cache = self._init_llm_cache()
        self._llm_cache_ttl = self.valves.LLM_CACHE_TTL  # kept for compatibility

        print("[CodeAware] Filter loaded (v5.24.0)")

    # endregion

    # region ── State Persistence Helpers ──────────────────────────────────────

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
            oldest = next(iter(self._conversation_state))
            self._log_debug(f"Evicting project {oldest} from cache")
            del self._conversation_state[oldest]
        return state

    def _set_state(self, project_id: str, state: Dict):
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        self._save_state_to_db(project_id, state)

    # endregion

    # region ── Database ───────────────────────────────────────────────────────

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
        if "feedback_history" not in data:
            data["feedback_history"] = []
        if "last_compression_timestamp" not in data:
            data["last_compression_timestamp"] = 0
        if "facts" not in data:
            data["facts"] = []
        if "last_suggestion_timestamp" not in data:
            data["last_suggestion_timestamp"] = 0
        if "response_cache" not in data:
            data["response_cache"] = []
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
        return {
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
        }

    def _save_state_to_db(self, project_id: str, state: Dict):
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
        self._db_conn.execute(
            "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(serializable), time.time()),
        )
        self._db_conn.commit()

    # endregion

    # region ── Debug logging ─────────────────────────────────────────────────

    def _log_debug(self, msg: str):
        if self.valves.debug:
            print(f"[CodeAware] {msg}")
            logger.info(msg)

    # endregion

    # region ── LTM initialization ─────────────────────────────────────────────

    def _init_long_term_memory(self):
        """
        Initialize embedder and ChromaDB using shared singletons when available,
        and create a dedicated ChromaDB collection for the response cache.
        """
        os.makedirs(self.valves.long_term_memory_dir, exist_ok=True)

        # C1: embedder singleton from shared_resources, or load locally
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

        # C2: ChromaDB singleton from shared_resources, or open a new one
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

        # Main LTM collection
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="conversation_memory", metadata={"hnsw:space": "cosine"}
        )

        # C3: dedicated collection for response cache (outside SQLite JSON)
        self._response_cache_collection = self.chroma_client.get_or_create_collection(
            name=f"response_cache_{self.valves.project_id or 'default'}",
            metadata={"hnsw:space": "cosine"},
        )

        self._purge_expired_memories()
        self._log_debug("LTM ready")

    # endregion

    # region ── Memory purging ─────────────────────────────────────────────────

    def _purge_expired_memories(self):
        if not HAS_CHROMA or self.memory_collection is None:
            return
        if self.valves.long_term_memory_expiration_days <= 0:
            return
        try:
            now = time.time()
            expired = self.memory_collection.get(where={"expires_at": {"$lt": now}})
            if expired and expired["ids"]:
                self.memory_collection.delete(ids=expired["ids"])
                self._log_debug(f"Purged {len(expired['ids'])} expired memories")
        except Exception as e:
            logger.warning(f"Purge failed: {e}")

    # endregion

    # region ── LLM Cache & HTTP Session (C4 + C6) ───────────────────────────

    def _init_llm_cache(self):
        """
        Initialize the internal LLM response cache.
        Uses AsyncLRUCache from shared_resources (TTL + LRU with async lock)
        instead of the old plain dict.
        """
        ttl = self.valves.LLM_CACHE_TTL
        max_size = 100  # double the original limit of 50

        if _SHARED_RESOURCES_AVAILABLE:
            return _AsyncLRUCache(max_size=max_size, ttl=ttl)

        # Fallback: minimal async-safe LRU dict
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
        """
        Compatibility: with AsyncLRUCache the TTL is applied automatically in .get().
        This method no longer needs to iterate the dict; kept so existing call sites don't break.
        """
        pass  # AsyncLRUCache manages TTL internally

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: str = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        Call the LLM with AsyncLRU cache and shared HTTP session.
        Falls back to local session if shared_resources is unavailable.
        """
        if not HAS_AIOHTTP:
            return None

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

        # C6: shared HTTP session from shared_resources, or fallback to local
        if _SHARED_RESOURCES_AVAILABLE:
            try:
                http_session = await _shared_get_http_session(
                    self.valves.llm_request_timeout
                )
            except Exception:
                http_session = self._http_session
        else:
            http_session = self._http_session

        if http_session is None:
            return None

        for model in models_to_try:
            model_name = (
                model.split("/", 1)[1]
                if is_ollama and model.startswith("ollama/")
                else model
            )

            # C4: async cache check
            cache_key = hashlib.md5(
                f"{model_name}|{prompt}|{system_prompt}|{temperature}|{max_tokens}".encode()
            ).hexdigest()
            cached = await self._llm_cache.get(cache_key)
            if cached is not None:
                self._log_debug(f"LLM cache hit for model {model_name}")
                return cached

            try:
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
                            content = choices[0].get("message", {}).get("content", "")
                            if not content:
                                continue
                            content = content.strip()

                        # C4: async cache store
                        await self._llm_cache.set(cache_key, content)
                        return content
                    else:
                        self._log_debug(
                            f"LLM call failed with model {model_name}, status {resp.status}"
                        )
            except Exception as e:
                self._log_debug(f"LLM model {model_name} error: {e}")
                continue

        logger.warning(f"All LLM models failed for prompt: {prompt[:100]}...")
        return None

    # endregion

    # region ── Response Cache (C3 + C5) ─────────────────────────────────────

    async def _store_response_in_cache(
        self, query: str, response: str, context_hash: str, state: dict
    ):
        """
        Store a response in the ChromaDB response cache.
        Falls back to the legacy state dict if ChromaDB is unavailable.
        """
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            # Fallback: original behavior (store in state dict)
            await self._store_response_in_cache_legacy(
                query, response, context_hash, state
            )
            return

        import anyio  # type: ignore

        try:
            embedding = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
            )
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache storage: {e}")
            return

        entry_id = hashlib.md5(
            f"{self.valves.project_id}|{query}".encode()
        ).hexdigest()[:32]

        try:
            max_entries = self.valves.response_cache_max_entries
            existing = col.get(
                where={"project_id": self.valves.project_id},
                include=["metadatas"],
            )
            if existing and len(existing["ids"]) >= max_entries:
                items = sorted(
                    zip(existing["ids"], existing["metadatas"]),
                    key=lambda x: x[1].get("timestamp", 0),
                )
                to_delete = [iid for iid, _ in items[: max(1, len(items) // 10)]]
                col.delete(ids=to_delete)

            col.upsert(
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
            self._log_debug(f"Stored response in ChromaDB cache for: {query[:50]}")
        except Exception as e:
            self._log_debug(f"_store_response_in_cache ChromaDB error: {e}")

    async def _store_response_in_cache_legacy(
        self, query: str, response: str, context_hash: str, state: dict
    ):
        """Fallback: original behavior (embedding in state dict)."""
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
        try:
            embedding = self.embedder.encode(query).tolist()
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache storage: {e}")
            return
        cache = state.get("response_cache", [])
        new_cache = [e for e in cache if e.get("query") != query]
        new_cache.append(
            {
                "query": query,
                "response": response,
                "embedding": embedding,
                "timestamp": time.time(),
                "context_hash": context_hash,
            }
        )
        if len(new_cache) > self.valves.response_cache_max_entries:
            new_cache.sort(key=lambda x: x.get("timestamp", 0))
            new_cache = new_cache[-self.valves.response_cache_max_entries :]
        state["response_cache"] = new_cache

    async def _find_cached_response(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        """
        Look up a semantically similar cached response using ChromaDB HNSW.
        Falls back to linear search in the state dict if ChromaDB is unavailable.
        """
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return None

        col = getattr(self, "_response_cache_collection", None)
        if col is None:
            return await self._find_cached_response_legacy(query, context_hash, state)

        import anyio  # type: ignore

        try:
            query_vec = await anyio.to_thread.run_sync(
                lambda: self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
            )
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache lookup: {e}")
            return None

        try:
            results = col.query(
                query_embeddings=[query_vec],
                n_results=1,
                where={"project_id": self.valves.project_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self._log_debug(f"_find_cached_response ChromaDB error: {e}")
            return None

        if not results or not results["ids"] or not results["ids"][0]:
            return None

        # ChromaDB cosine distance: 0 = identical → similarity = 1 - dist/2
        dist = results["distances"][0][0]
        similarity = 1.0 - (dist / 2.0)

        if similarity < self.valves.response_cache_similarity_threshold:
            return None

        meta = results["metadatas"][0][0]

        # Verify context_hash if required
        if (
            self.valves.response_cache_include_context_hash
            and meta.get("context_hash", "") != context_hash
        ):
            return None

        # Check TTL
        ttl = self.valves.response_cache_ttl_hours * 3600
        ts = meta.get("timestamp", 0)
        if ttl > 0 and time.time() - ts > ttl:
            col.delete(ids=[results["ids"][0][0]])
            return None

        doc = results["documents"][0][0]
        self._log_debug(f"Response cache HIT (sim={similarity:.3f}) for: {query[:50]}")
        return {"response": doc, "query": meta.get("query", ""), "timestamp": ts}

    async def _find_cached_response_legacy(
        self, query: str, context_hash: str, state: dict
    ) -> Optional[dict]:
        """Fallback: linear search in state dict (original behavior)."""
        cache = state.get("response_cache", [])
        if not cache:
            return None
        try:
            q_emb = self.embedder.encode(query).tolist()
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache: {e}")
            return None
        best_sim = 0.0
        best_entry = None
        now = time.time()
        ttl = self.valves.response_cache_ttl_hours * 3600
        for entry in cache:
            if ttl > 0 and (now - entry.get("timestamp", 0)) > ttl:
                continue
            if (
                self.valves.response_cache_include_context_hash
                and entry.get("context_hash", "") != context_hash
            ):
                continue
            entry_emb = entry.get("embedding")
            if not entry_emb:
                continue
            sim = self._cosine_similarity(q_emb, entry_emb)
            if (
                sim > best_sim
                and sim >= self.valves.response_cache_similarity_threshold
            ):
                best_sim = sim
                best_entry = entry
        if best_entry:
            self._log_debug(f"Cache hit (legacy, sim={best_sim:.3f}) for: {query[:50]}")
        return best_entry

    # endregion

    # region ── Helpers ────────────────────────────────────────────────────────

    def _ensure_last_message_is_user(self, messages: List[dict]) -> List[dict]:
        """Trim trailing assistant messages so the list ends with a user."""
        if not messages:
            return messages
        while messages and messages[-1].get("role") != "user":
            messages.pop()
        if not messages:
            messages.append({"role": "user", "content": "continue"})
            self._log_debug("Inserted dummy user message to satisfy API")
        return messages

    def _estimate_tokens(self, messages: List[dict]) -> int:
        if self.tokenizer:
            total = 0
            for m in messages:
                content = str(m.get("content", ""))
                total += len(self.tokenizer.encode(content))
                total += 4  # approximate overhead per message
            return total
        # Rough estimate: 1 token ≈ 4 chars
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        total_chars += sum(30 for _ in messages)  # role and other fields
        return total_chars // 4

    def _extract_function_names(self, code: str) -> List[str]:
        names = []
        names.extend(
            re.findall(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code, re.MULTILINE)
        )
        names.extend(
            re.findall(r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        names.extend(
            re.findall(
                r"^\s*(?:function|async function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                code,
                re.MULTILINE,
            )
        )
        names.extend(
            re.findall(
                r"^\s*(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(?",
                code,
                re.MULTILINE,
            )
        )
        names.extend(
            re.findall(r"^\s*fn\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        names.extend(
            re.findall(r"^\s*func\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        return list(set(names))

    def _extract_signature(self, code: str) -> str:
        """Extract a brief signature (function/class name + docstring) from code."""
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

    # endregion

    # region ── Code extraction & classification ──────────────────────────────

    async def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        blocks = []
        if not self.valves.auto_detect_code_blocks:
            return blocks
        # Fenced code blocks
        for match in self.code_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            code = await self._handle_oversized_code_block(code, lang)
            blocks.append({"language": lang, "code": code, "type": "fenced"})
        # Indented blocks (4 spaces or tab)
        lines = content.split("\n")
        indented = []
        for line in lines:
            if line.startswith(("    ", "\t")):
                indented.append(line.lstrip(" \t"))
            else:
                if len(indented) >= 3:
                    code = "\n".join(indented)
                    code = await self._handle_oversized_code_block(code, "text")
                    blocks.append(
                        {"language": "text", "code": code, "type": "indented"}
                    )
                indented = []
        if len(indented) >= 3:
            code = "\n".join(indented)
            code = await self._handle_oversized_code_block(code, "text")
            blocks.append({"language": "text", "code": code, "type": "indented"})
        return blocks

    def _extract_line_range(
        self, content: str
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        if not self.valves.track_line_numbers:
            return None, None, None
        pattern = r"(?:^|\s)([a-zA-Z0-9_\-\./]+\.\w+):(\d+)(?:-(\d+))?"
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
        if "error" in cl or "exception" in cl or "traceback" in cl:
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

    # endregion

    # region ── Session classification & Active code context ──────────────────

    async def _classify_session(self, messages: List[dict], project_id: str) -> bool:
        """Determine whether the current conversation is a code session."""
        state = self._get_state(project_id)
        if state and state.get("active_blocks"):
            self._log_debug("Session classified as code (existing blocks)")
            return True
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            if self._has_code_indicators(msg.get("content", "")):
                self._log_debug("Session classified as code (heuristic)")
                return True
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            self._log_debug("Session classified as non-code (no user message)")
            return False
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"Is the following user message about programming/code? Answer only 'yes' or 'no'.\n\nMessage: {last_user_msg.get('content','')[:500]}"
        self._log_debug("Classifying session with LLM...")
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a classifier. Answer only 'yes' or 'no'.",
            model_override=model,
            max_tokens=5,
            temperature=0.0,
        )
        result = response and response.strip().lower().startswith("yes")
        self._log_debug(
            f"LLM classification result: {'code' if result else 'non-code'}"
        )
        return result

    def _has_code_indicators(self, content: str) -> bool:
        """Return True if the content contains typical code patterns."""
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
        """Build the active code context string for injection into the system prompt."""
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
        # Sort by importance, giving an extra boost to raw file blocks
        boost = self.valves.raw_file_priority_boost
        active.sort(
            key=lambda b: b.importance_score + (boost if b.is_raw else 0),
            reverse=True,
        )
        base_codes = [b for b in active if b.content_type == ContentType.BASE_CODE][
            : self.valves.max_base_code_blocks
        ]
        proposed = [b for b in active if b.content_type == ContentType.PROPOSED_CHANGE][
            : self.valves.max_proposed_changes
        ]
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
                    f" (file: {b.file_path}"
                    + (
                        f", lines {b.line_range[0]}-{b.line_range[1]}"
                        if b.line_range
                        else ""
                    )
                    + ")"
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

    # endregion

    # region ── Update Active Code ─────────────────────────────────────────────

    async def _update_active_code(self, message: dict, project_id: str):
        """Extract code blocks from a message and update the active blocks state."""
        if not self.valves.enable_code_awareness:
            return
        state = self._get_state(project_id)
        if self.valves.auto_remove_duplicate_blocks:
            self._remove_duplicate_blocks(state)

        asyncio.create_task(self._summarize_inactive_blocks_safely(project_id))
        content = message.get("content", "")
        role = message.get("role", "")

        self._update_mentions_from_message(state, content)

        for block in state["active_blocks"].values():
            if (
                block.content
                and self._calculate_code_similarity(block.content[:200], content[:200])
                > 0.7
            ):
                block.mention_count += 1
                block.last_mentioned = time.time()
                block._update_importance()

        extracted = await self._extract_code_blocks(content)
        if not content and not extracted:
            return

        content_type = self._classify_content(content, extracted)
        file_path, line_start, line_end = self._extract_line_range(content)

        for block_info in extracted:
            extracted_paths = (
                self._extract_file_paths(content)
                if self.valves.track_file_paths
                else []
            )
            blk_file = file_path or (extracted_paths[0] if extracted_paths else None)
            new_block = CodeBlock(
                content=block_info["code"],
                content_type=content_type,
                generated_by_assistant=(role == "assistant"),
                file_path=blk_file,
                line_range=(line_start, line_end) if line_start and line_end else None,
                timestamp=time.time(),
                is_active=True,
                mention_count=1,
                dependencies=[],
                potentially_affected=False,
                pinned=False,
                obsolete=False,
            )

            # Mark as raw if the content came with the [KEEP] tag
            is_raw_extraction = "[KEEP]" in content
            if is_raw_extraction:
                new_block.is_raw = True

            # Honour manual importance markers
            if "[KEEP]" in content or "#important" in content.lower():
                new_block.importance_score = 10.0
                new_block.pinned = True
                self._log_debug(
                    f"Manual importance marker detected for block {new_block.hash}, pinned automatically"
                )

            is_dup, existing = self._is_duplicate_code(
                new_block, list(state["active_blocks"].values())
            )
            if is_dup and existing:
                # Always update pinned or raw blocks
                if existing.pinned or is_raw_extraction:
                    existing.content = new_block.content
                    existing.hash = new_block.hash
                    if new_block.file_path:
                        existing.file_path = new_block.file_path
                    existing.line_range = new_block.line_range
                    existing.timestamp = time.time()
                    existing.mention_count += 1
                    existing.last_mentioned = time.time()
                    existing.pinned = True
                    existing.is_raw = existing.is_raw or is_raw_extraction
                    existing.importance_score = 10.0
                    existing._update_importance()
                    self._log_debug(
                        f"Updated existing pinned block {existing.hash} (raw extraction or similar code)"
                    )
                    if (
                        self.valves.enable_dependency_tracking
                        and self.valves.dependency_refresh_on_update
                    ):
                        asyncio.create_task(
                            self._refresh_dependencies_for_block(
                                existing.hash, project_id
                            )
                        )
                    continue

                # Prioritise recent code: update the existing block
                if self.valves.prioritize_recent_code:
                    existing.content = new_block.content
                    existing.hash = new_block.hash
                    if new_block.file_path:
                        existing.file_path = new_block.file_path
                    existing.line_range = new_block.line_range
                    existing.timestamp = time.time()
                    existing.mention_count += 1
                    existing.last_mentioned = time.time()
                    existing._update_importance()
                    self._log_debug(f"Updated existing block {existing.hash}")
                    if (
                        self.valves.enable_dependency_tracking
                        and self.valves.dependency_refresh_on_update
                    ):
                        asyncio.create_task(
                            self._refresh_dependencies_for_block(
                                existing.hash, project_id
                            )
                        )
                continue

            # Mark conflicting proposed changes
            if (
                content_type == ContentType.PROPOSED_CHANGE
                and self._has_conflicting_proposed_changes(state, new_block)
            ):
                new_block.importance_score = max(new_block.importance_score, 7.0)
                self._log_debug(
                    f"Proposed change {new_block.hash} marked as conflicting"
                )

            state["active_blocks"][new_block.hash] = new_block
            self._log_debug(f"New {content_type.value} block: {new_block.hash}")

            if content_type == ContentType.PROPOSED_CHANGE:
                state["recent_changes"].append(new_block)
                conflict = self._has_conflicting_proposed_changes(state, new_block)
                if self.valves.enable_diff_application and not conflict:
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
                else:
                    self._log_debug(
                        f"Skipped auto-apply for conflicting proposed change {new_block.hash}"
                    )
            elif content_type == ContentType.COMMITTED_CHANGE:
                state["committed_changes"].append(new_block)
            elif (
                content_type == ContentType.ERROR and self.valves.preserve_error_context
            ):
                new_block.importance_score = min(new_block.importance_score + 3.0, 10.0)

            # Enforce max active blocks, keeping the most important ones
            if len(state["active_blocks"]) > self.valves.max_active_blocks:
                sorted_blocks = sorted(
                    state["active_blocks"].values(),
                    key=lambda b: b.importance_score
                    + (self.valves.raw_file_priority_boost if b.is_raw else 0),
                    reverse=True,
                )
                keep = sorted_blocks[: self.valves.max_active_blocks]
                state["active_blocks"] = {b.hash: b for b in keep}

            if self.valves.enable_dependency_tracking and content_type in (
                ContentType.BASE_CODE,
                ContentType.PROPOSED_CHANGE,
                ContentType.COMMITTED_CHANGE,
            ):
                asyncio.create_task(
                    self._refresh_dependencies_for_block(new_block.hash, project_id)
                )

        # Assistant learning: update the best matching base code block
        if role == "assistant" and len(extracted) > 0:
            for block_info in extracted:
                best_base = None
                best_sim = 0.0
                for base in state["active_blocks"].values():
                    if base.content_type == ContentType.BASE_CODE:
                        if base.file_path and file_path and base.file_path == file_path:
                            sim = self._calculate_code_similarity(
                                base.content, block_info["code"]
                            )
                            if sim > best_sim:
                                best_sim = sim
                                best_base = base
                        else:
                            sim = self._calculate_code_similarity(
                                base.content, block_info["code"]
                            )
                            if sim > best_sim and sim > 0.6:
                                best_sim = sim
                                best_base = base
                if best_base and best_sim > 0.6 and best_sim < 0.95:
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
                    self._log_debug(
                        f"Assistant updated base code block {best_base.hash} (sim={best_sim:.2f})"
                    )
                    if (
                        self.valves.enable_dependency_tracking
                        and self.valves.dependency_refresh_on_update
                    ):
                        asyncio.create_task(
                            self._refresh_dependencies_for_block(
                                best_base.hash, project_id
                            )
                        )

        state["message_count"] += 1

        if self.valves.auto_remove_duplicate_blocks:
            self._remove_duplicate_blocks(state)

        if self.valves.hierarchical_compression_enabled:
            if (
                state["message_count"]
                % self.valves.hierarchical_compression_interval_messages
                == 0
            ):
                asyncio.create_task(self._hierarchical_compress(project_id, state))

        asyncio.create_task(self._expire_blocks_by_time(project_id))
        asyncio.create_task(self._clean_affected_flags(project_id))
        self._set_state(project_id, state)

    def _update_mentions_from_message(self, state: Dict, message_content: str):
        """Boost importance of blocks whose function/class names are mentioned."""
        for block in state["active_blocks"].values():
            names = self._extract_function_names(block.content)
            if not names:
                continue
            for name in names:
                if re.search(r"\b" + re.escape(name) + r"\b", message_content):
                    block.mention_count += 1
                    block.last_mentioned = time.time()
                    block._update_importance()
                    self._log_debug(
                        f"Boosted importance of {block.hash} due to mention of '{name}'"
                    )
                    break

    def _is_duplicate_code(
        self, new_block: CodeBlock, existing_blocks: List[CodeBlock]
    ) -> Tuple[bool, Optional[CodeBlock]]:
        """Check if the new code block is a duplicate of an existing one."""
        for ex in existing_blocks:
            if (
                self._calculate_code_similarity(new_block.content, ex.content)
                >= self.valves.code_similarity_threshold
            ):
                return True, ex
        return False, None

    def _has_conflicting_proposed_changes(
        self, state: Dict, new_block: CodeBlock
    ) -> bool:
        """Return True if there is already a conflicting proposed change."""
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

    # endregion

    # region ── Inlet ─────────────────────────────────────────────────────────

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("inlet called")
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        if not messages:
            return body
        state = self._get_state(project_id)
        is_code_session = await self._classify_session(messages, project_id)
        self._log_debug(f"Session: {'code' if is_code_session else 'non-code'}")

        # 1. Forget
        if (
            self.valves.enable_forget_command
            or self.valves.enable_natural_language_forget
        ) and (
            is_code_session
            or (messages and messages[-1].get("content", "").startswith("/"))
        ):
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                body["messages"] = messages
                return body

        # 2. Remember
        if self.valves.enable_natural_language_forget:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                remember_intent = await self._parse_remember_intent(
                    last_user_msg.get("content", "")
                )
                if (
                    remember_intent
                    and remember_intent.get("action")
                    and remember_intent["action"] != "none"
                ):
                    confirmation = await self._execute_remember_intent(
                        project_id, remember_intent
                    )
                    messages.append({"role": "system", "content": confirmation})
                    body["messages"] = messages
                    return body

        # 3. Obsolete
        if self.valves.enable_obsolete_marking:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                intent = await self._parse_obsolete_intent(last_user_msg["content"])
                if intent and intent.get("action") != "none":
                    confirmation = await self._execute_obsolete_intent(
                        project_id, intent
                    )
                    messages.append({"role": "system", "content": confirmation})
                    body["messages"] = messages
                    return body

        # 4. /think (explicit) or auto Chain-of-Thought
        if self.valves.enable_cot_on_demand or self.valves.auto_cot_enabled:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
                user_content = last_user_msg.get("content", "")
                # Explicit /think command
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
                        body["messages"] = self._ensure_last_message_is_user(messages)
                        return body

                # Auto-CoT: inject prompt if message looks complex
                elif (
                    self.valves.auto_cot_enabled
                    and self._should_auto_cot(user_content)
                    and not user_content.strip().startswith("/")
                ):
                    self._log_debug("Auto-injecting Chain-of-Thought prompt")
                    sys_msgs = [m for m in messages if m.get("role") == "system"]
                    cot_prompt = "Please think step by step before answering. Show your reasoning, then provide the final answer."
                    if sys_msgs:
                        sys_msgs[0]["content"] = (
                            cot_prompt + "\n" + sys_msgs[0]["content"]
                        )
                    else:
                        messages.insert(0, {"role": "system", "content": cot_prompt})
                    body["messages"] = messages

        # 5. /assume
        if self.valves.enable_assumption_extraction:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
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
                    body["messages"] = self._ensure_last_message_is_user(messages)
                    return body

        # 6. /iterate
        if self.valves.enable_iterative_mode:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                result, consumed = await self._run_iteration(
                    project_id, last_user_msg.get("content", "")
                )
                if consumed:
                    messages.pop()
                    messages.append({"role": "assistant", "content": result})
                    body["messages"] = self._ensure_last_message_is_user(messages)
                    return body

        # 7. Smart context selection
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

        # 8. Contradiction detection
        if (
            self.valves.enable_contradiction_detection
            and self.valves.contradiction_inject_warning
            and is_code_session
        ):
            contradiction_warning = await self._detect_contradictions(messages)
            if contradiction_warning:
                messages.insert(0, {"role": "system", "content": contradiction_warning})
                body["messages"] = messages

        # 9. Update active code (process recent messages)
        if self.valves.enable_code_awareness and is_code_session:
            for msg in messages[-5:]:
                await self._update_active_code(msg, project_id)

        # 10. LTM retrieval
        # Ensure unique_meta is always defined to avoid UnboundLocalError
        unique_meta = []
        if not self.valves.smart_context_selection and is_code_session:
            last_user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if (
                last_user_msg
                and HAS_SENTENCE
                and HAS_CHROMA
                and self.valves.enable_code_awareness
            ):
                query = last_user_msg.get("content", "")
                base_mem = await self._retrieve_relevant_memories(
                    query, project_id, ContentType.BASE_CODE, include_meta=True
                )
                error_mem = (
                    await self._retrieve_relevant_memories(
                        query, project_id, ContentType.ERROR, include_meta=True
                    )
                    if self.valves.preserve_error_context
                    else []
                )
                general_mem = await self._retrieve_relevant_memories(
                    query, project_id, None, include_meta=True
                )
                # Combine, deduplicate by content, keep most recent timestamp
                all_meta = base_mem + error_mem + general_mem
                all_meta.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
                seen = set()
                unique_meta = []
                for m in all_meta:
                    if m["doc"] not in seen:
                        seen.add(m["doc"])
                        unique_meta.append(m)

        # Token‑budget driven fragment selection
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

        # 11. Response cache
        if self.valves.enable_response_cache and HAS_SENTENCE and is_code_session:
            last_user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user_msg:
                context_hash = self._compute_context_hash(messages)
                cached = await self._find_cached_response(
                    last_user_msg.get("content", ""), context_hash, state
                )
                if cached:
                    messages.append(
                        {"role": "assistant", "content": cached["response"]}
                    )
                    body["messages"] = messages
                    return body

        # 12. Inject active code context
        if is_code_session:
            active_ctx = self._get_active_code_context(project_id)
            if active_ctx:
                # Lightweight coding checklist
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

        # 12b. Code review checklist injection
        if (
            is_code_session
            and self.valves.enable_code_review_mode
            and self._is_code_review_request(
                next(
                    (
                        m.get("content", "")
                        for m in reversed(messages)
                        if m.get("role") == "user"
                    ),
                    "",
                )
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

        # 13. Inject facts
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

        # 14. Confidence scoring instruction
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

        # 15. Inject feedback context
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

        # 16. Proactive suggestions (summarisation and commands)
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

        # 17. Duplicate question detection
        if self.valves.duplicate_question_threshold and HAS_SENTENCE:
            last_user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user_msg:
                duplicate = await self._find_duplicate_question(
                    last_user_msg.get("content", ""), project_id
                )
                if duplicate:
                    warn_msg = f"⚠️ **Note**: This question is very similar to one you asked before (similarity {duplicate['sim']:.2f})."
                    messages.insert(0, {"role": "system", "content": warn_msg})
                    body["messages"] = messages

        # 18. Adaptive context trim
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

        if trim_needed and len(history_msgs) > self.valves.max_turns:
            self._log_debug("Trimming old messages")
            keep = self.valves.max_turns
            # Ensure the last message is from the user
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

        # 19. Final safety: ensure the last message is from the user
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

        body["messages"] = messages
        return body

    # endregion

    # region ── Outlet ─────────────────────────────────────────────────────────

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("outlet called")
        if not (HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness):
            return body
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        state = self._get_state(project_id)
        is_code_session = await self._classify_session(messages, project_id)
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                if is_code_session:
                    await self._update_active_code(msg, project_id)
                    if not self.valves.ltm_store_only_code_sessions or is_code_session:
                        await self._store_message_in_memory(msg, project_id)
                else:
                    if not self.valves.ltm_store_only_code_sessions:
                        await self._store_message_in_memory(msg, project_id)
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
        self._purge_expired_memories()
        self._clean_llm_cache()
        return body

    # endregion

    # region ── Remaining helper methods ──────────────────────────────────────

    def _compute_context_hash(self, messages: List[dict]) -> str:
        """Compute a hash of system messages for response cache invalidation."""
        if not self.valves.response_cache_include_context_hash:
            return ""
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        context_str = "\n".join([m.get("content", "") for m in sys_msgs])
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

    async def _store_message_in_memory(self, message: dict, project_id: str):
        """Store a conversation message in long‑term memory (ChromaDB)."""
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
        embedding = self.embedder.encode(content).tolist()
        self.memory_collection.upsert(
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
        self._log_debug(f"Stored message {msg_id} in LTM")
        state = self._get_state(project_id)
        msg_count = state.get("message_count", 0)
        if msg_count > 0 and msg_count % self.valves.ltm_compress_after_messages == 0:
            asyncio.create_task(self._compress_ltm_for_conversation(project_id))

    async def _compress_ltm_for_conversation(self, project_id: str):
        """Summarise older messages in the conversation to save LTM space."""
        if not HAS_AIOHTTP or not self.memory_collection:
            return
        try:
            results = self.memory_collection.get(
                where={"$and": [{"project_id": {"$eq": project_id}}]}
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
            texts = "\n---\n".join([doc for _, doc, _ in to_compress])
            summary = await self._call_llm(
                prompt=f"Summarise the following conversation segment, keeping key technical decisions and code changes:\n\n{texts[:3000]}",
                system_prompt="You produce concise, information-dense summaries.",
                max_tokens=500,
                temperature=0.3,
            )
            if summary:
                self.memory_collection.delete(ids=[id for id, _, _ in to_compress])
                summary_id = f"{project_id}_summary_{int(time.time())}"
                summary_embedding = self.embedder.encode(summary).tolist()
                self.memory_collection.upsert(
                    ids=[summary_id],
                    embeddings=[summary_embedding],
                    metadatas=[
                        {
                            "project_id": project_id,
                            "is_summary": True,
                            "timestamp": time.time(),
                        }
                    ],
                    documents=[summary],
                )
                self._log_debug(
                    f"Compressed {len(to_compress)} messages into summary for {project_id}"
                )
        except Exception as e:
            logger.warning(f"LTM compression failed: {e}")

    async def _retrieve_relevant_memories(
        self,
        query: str,
        project_id: str,
        content_type_filter: Optional[ContentType] = None,
        include_meta: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Retrieve semantically relevant memories from LTM."""
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return [] if not include_meta else []
        try:
            q_emb = self.embedder.encode(query[:1000]).tolist()
            now = time.time()
            where_filter = {"$and": [{"project_id": {"$eq": project_id}}]}
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})
            if content_type_filter:
                where_filter["$and"].append(
                    {"content_type": {"$eq": content_type_filter.value}}
                )
            results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=(
                    self.valves.long_term_memory_top_k * 2
                    if self.valves.enable_reranking
                    else self.valves.long_term_memory_top_k
                ),
                where=where_filter,
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
                        docs_with_meta.append((doc, sim, ts))
            docs_with_meta.sort(key=lambda x: x[1], reverse=True)
            if self.valves.enable_reranking and self._cross_encoder and docs_with_meta:
                rerank_k = (
                    self.valves.reranker_top_k
                    if self.valves.reranker_top_k > 0
                    else self.valves.long_term_memory_top_k
                )
                rerank_k = min(rerank_k, 50)
                docs_only = [d[0] for d in docs_with_meta[: rerank_k * 2]]
                reranked = await self._rerank_results(query, docs_only, rerank_k)
                doc_to_meta = {d[0]: (d[1], d[2]) for d in docs_with_meta}
                docs_with_meta = [
                    (doc, *doc_to_meta.get(doc, (0.0, None))) for doc in reranked
                ]
            docs_with_meta = docs_with_meta[: self.valves.long_term_memory_top_k]
            if include_meta:
                return [{"doc": doc, "timestamp": ts} for doc, _, ts in docs_with_meta]
            else:
                return [doc for doc, _, _ in docs_with_meta]
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return [] if not include_meta else []

    async def _retrieve_historical_messages(
        self, query: str, project_id: str, limit: int
    ) -> List[Dict]:
        """Retrieve full historical messages for smart context selection."""
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []
        try:
            q_emb = self.embedder.encode(query[:1000]).tolist()
            now = time.time()
            where_filter = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"is_hierarchical_summary": {"$ne": True}},
                ]
            }
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["$and"].append({"expires_at": {"$gt": now}})
            results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            messages = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    role = meta.get("role", "user")
                    messages.append({"role": role, "content": doc})
            summary_filter = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"is_hierarchical_summary": {"$eq": True}},
                ]
            }
            summary_results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=limit // 2,
                where=summary_filter,
                include=["documents", "metadatas"],
            )
            if summary_results and summary_results["documents"]:
                for i, doc in enumerate(summary_results["documents"][0]):
                    meta = summary_results["metadatas"][0][i]
                    role = meta.get("role", "assistant")
                    messages.append({"role": role, "content": doc})
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
        """Load the cross‑encoder reranker model if enabled."""
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
        """Rerank documents using the cross‑encoder for better relevance."""
        if not self.valves.enable_reranking or not self._cross_encoder or not documents:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = self._cross_encoder.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    async def _expire_blocks_by_time(self, project_id: str):
        """Remove blocks that haven't been mentioned for a long time."""
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
            del state["active_blocks"][h]
        if to_remove:
            self._set_state(project_id, state)

    async def _summarize_inactive_blocks_safely(self, project_id: str):
        """Summarise code blocks that haven't been active for a while."""
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
                summary_block = CodeBlock(
                    content=f"[Summary of inactive code]\n{summary}",
                    content_type=ContentType.GENERAL,
                    timestamp=time.time(),
                    is_active=False,
                    importance_score=block.importance_score * 0.5,
                )
                state["active_blocks"][h] = summary_block
        self._set_state(project_id, state)

    async def _summarize_code_block(self, block: CodeBlock) -> Optional[str]:
        """Generate a short summary of a code block using an LLM."""
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

    # --------------------------------------------------------------------------
    # Forget / Remember / Obsolete / Iterative / CoT / Assumptions / Feedback
    # --------------------------------------------------------------------------

    async def _handle_forget_command(
        self, messages: List[dict], project_id: str, __user__: Optional[dict]
    ) -> Tuple[List[dict], bool]:
        """Check for /forget or natural language forget requests and execute them."""
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

        if self.valves.enable_natural_language_forget:
            intent = await self._parse_forget_intent(content)
            if intent and intent.get("action") != "none":
                confirmation = await self._execute_forget_intent(project_id, intent)
                self._set_state(project_id, self._get_state(project_id))
                messages.pop()
                messages.append({"role": "assistant", "content": confirmation})
                return messages, True

        if self.valves.enable_forget_command and content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            state = self._get_state(project_id)
            if not state:
                return messages, False
            if target == "all":
                state["active_blocks"].clear()
                state["recent_changes"].clear()
                state["committed_changes"].clear()
                confirmation = "Forgotten all context."
            elif target == "last":
                if state["active_blocks"]:
                    last_hash = max(
                        state["active_blocks"].keys(),
                        key=lambda h: state["active_blocks"][h].timestamp,
                    )
                    del state["active_blocks"][last_hash]
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
                    del state["active_blocks"][h]
                confirmation = (
                    f"Forgotten {len(to_remove)} block(s) matching '{target}'."
                )
            self._set_state(project_id, state)
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return messages, True

        return messages, False

    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
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
                del state["active_blocks"][last_hash]
            return "Forgotten the last context block."
        elif action == "forget_n":
            n = intent.get("n", 1)
            blocks_by_time = sorted(
                state["active_blocks"].items(),
                key=lambda x: x[1].timestamp,
                reverse=True,
            )
            removed = 0
            for h, _ in blocks_by_time[:n]:
                if h in state["active_blocks"]:
                    del state["active_blocks"][h]
                    removed += 1
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
                del state["active_blocks"][h]
            return f"Forgotten {len(to_remove)} block(s) related to {file_path}."
        elif action == "forget_block":
            block_id = intent.get("hash") or intent.get("id") or ""
            if not block_id:
                return "No block specified."
            if block_id in state["active_blocks"]:
                del state["active_blocks"][block_id]
                return f"Forgotten block {block_id}."
            matches = [h for h in state["active_blocks"] if block_id in h]
            if matches:
                for h in matches:
                    del state["active_blocks"][h]
                return f"Forgotten {len(matches)} block(s) matching {block_id}."
            return f"No block found for {block_id}."
        elif action == "forget_all":
            state["active_blocks"].clear()
            state["recent_changes"].clear()
            state["committed_changes"].clear()
            return "Forgotten all context."
        else:
            return "Unrecognized forget action."

    async def _parse_forget_intent(self, user_message: str) -> Optional[Dict]:
        """Parse a user message for forget requests using heuristics and LLM fallback."""
        if not self.valves.enable_natural_language_forget:
            return None
        content = user_message.strip().lower()

        # Heuristics
        if content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "forget_all"}
            elif target in ("last", "último"):
                return {"action": "forget_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "forget_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "forget_block", "hash": target}
            else:
                return {"action": "forget_last"}

        if re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\s+(todo|all|todo el contexto|el contexto)\b",
            content,
        ):
            return {"action": "forget_all"}
        if re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\s+(el|la|lo)\s+(último|last|ultimo|último bloque|last block)\b",
            content,
        ):
            return {"action": "forget_last"}

        if not re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\b",
            content,
        ):
            return None

        # LLM fallback
        self._log_debug("Calling LLM for forget intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpret forget intent. Possible actions: forget_last, forget_n (with n), forget_file (with file), forget_block (with hash), forget_all.
User: "{user_message}"
Output JSON: {{"action": "...", "n": N, "file": "...", "hash": "..."}} or {{"action": "none"}}
Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") != "none":
                return data
        except:
            pass
        return None

    async def _parse_remember_intent(self, user_message: str) -> Optional[Dict]:
        """Parse a user message for pin/remember requests using heuristics and LLM fallback."""
        if not self.valves.enable_natural_language_forget:
            return None
        content = user_message.strip().lower()
        if content.startswith("/remember"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "pin_all"}
            elif target in ("last", "último"):
                return {"action": "pin_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "pin_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "pin_block", "hash": target}
            else:
                return {"action": "pin_last"}
        if re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\s+(todo|all|todo el contexto|el contexto)\b",
            content,
        ):
            return {"action": "pin_all"}
        if re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\s+(el|la|lo)\s+(último|last|ultimo|último bloque|last block)\b",
            content,
        ):
            return {"action": "pin_last"}
        if not re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for remember intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpret pin/remember intent. Possible actions: pin_last, pin_n (with n), pin_file (with file), pin_block (with description), pin_all, unpin_last, unpin_file, unpin_all, etc.
User: "{user_message}"
Output JSON with action and parameters. If no pinning intent: {{"action": "none"}}
Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") != "none":
                return data
        except:
            pass
        return None

    async def _execute_remember_intent(self, project_id: str, intent: Dict) -> str:
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
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
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
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = set_pinned(to_unpin, False)
            return f"Unpinned {count} block(s) related to {file_path}."
        elif action == "unpin_all":
            count = set_pinned(blocks, False)
            return f"Unpinned all {count} blocks."
        else:
            return "Unrecognized pin action."

    async def _parse_obsolete_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_obsolete_marking:
            return None
        content = user_message.strip().lower()
        if content.startswith("/obsolete"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "obsolete_all"}
            elif target in ("last", "último"):
                return {"action": "obsolete_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "obsolete_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "obsolete_block", "hash": target}
            else:
                return {"action": "obsolete_last"}
        if re.search(
            r"\b(obsoleto|obsoleta|descarta|marca\s+como\s+obsoleto|obsolete|discard|esto ya no sirve|ya no es relevante)\b",
            content,
        ):
            return {"action": "obsolete_last"}
        if re.search(
            r"\b(revive|revivir|recupera|restaura)\s+(el|la|lo)\s+(último|last|ultimo)\b",
            content,
        ):
            return {"action": "revive_last"}
        if not re.search(
            r"\b(obsoleto|obsoleta|descarta|obsolete|discard|revive|revivir|restaura|recupera)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for obsolete intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpret obsolete intent. Possible actions: obsolete_last, obsolete_n (with n), obsolete_file (with file), obsolete_block (with hash/description), obsolete_all, revive_last, revive_file, revive_all.
User: "{user_message}"
Output JSON with action and parameters. If no obsolete intent: {{"action": "none"}}
Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") != "none":
                return data
        except:
            pass
        return None

    async def _execute_obsolete_intent(self, project_id: str, intent: Dict) -> str:
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
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
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
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = set_obsolete(to_revive, False)
            return (
                f"Removed obsolete mark from {count} block(s) related to {file_path}."
            )
        elif action == "revive_all":
            count = set_obsolete(blocks, False)
            return f"Removed obsolete mark from all {count} block(s)."
        else:
            return "Unrecognized obsolete action."

    async def _parse_feedback_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_feedback_tracking:
            return None
        content = user_message.strip().lower()
        if re.search(r"\b(feedback|retroalimentación)\b", content):
            if "success" in content or "worked" in content:
                return {"action": "feedback", "outcome": "success", "comment": content}
            if "fail" in content or "error" in content:
                return {"action": "feedback", "outcome": "failure", "comment": content}
        if re.search(
            r"\b(funcionó perfecto|solucionado|resuelto|solved|fixed|correcto|excelente)\b",
            content,
        ):
            return {"action": "feedback", "outcome": "success", "comment": content}
        if re.search(
            r"\b(no funcionó|no funciono|sigue roto|still broken|error|falló|fallo|no solucionado|incorrecto)\b",
            content,
        ):
            return {"action": "feedback", "outcome": "failure", "comment": content}
        if not re.search(
            r"\b(worked|did not work|failed|success|correcto|incorrecto|error|fallo|funcionó|funciono)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for feedback intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""You are a feedback interpreter. The user is giving feedback about a code change that was applied.
Possible outcomes:
- success: the change worked, problem solved
- failure: the change did not work, error remains
- neutral: unclear or not feedback.

User message: "{user_message}"

If feedback, output JSON: {{"action": "feedback", "outcome": "success/failure", "comment": "..."}}
If not feedback: {{"action": "none"}}

Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") == "feedback":
                return data
        except:
            pass
        return None

    async def _record_feedback(self, project_id: str, outcome: str, comment: str):
        state = self._get_state(project_id)
        if not state or not state["committed_changes"]:
            return
        last_commit = max(state["committed_changes"], key=lambda c: c.timestamp)
        success = outcome == "success"
        feedback = AppliedChangeFeedback(
            change_hash=last_commit.hash,
            change_description=last_commit.content[:200],
            file_path=last_commit.file_path,
            timestamp=time.time(),
            success=success,
            user_comment=comment,
        )
        state["feedback_history"].append(feedback)
        if len(state["feedback_history"]) > self.valves.feedback_history_limit:
            state["feedback_history"] = state["feedback_history"][
                -self.valves.feedback_history_limit :
            ]
        if not success:
            last_commit.importance_score /= (
                self.valves.feedback_importance_penalty_for_failure
            )
            last_commit.importance_score = max(0.5, last_commit.importance_score)
        else:
            last_commit.importance_score = min(10.0, last_commit.importance_score + 1.0)
        self._set_state(project_id, state)

    def _get_feedback_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["feedback_history"]:
            return ""
        lines = ["## Recent Change Feedback\n"]
        for fb in state["feedback_history"][-5:]:
            status = "✅ SUCCESS" if fb.success else "❌ FAILED"
            desc = fb.change_description.replace("\n", " ")[:100]
            lines.append(f'- {status}: `{desc}` - User: "{fb.user_comment}"')
        return "\n".join(lines)

    async def _extract_facts_from_message(self, content: str) -> List[str]:
        pattern = r"\[FACT:\s*(.*?)\]"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches]

    async def _add_fact(self, project_id: str, fact_text: str, source: str = "user"):
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

    async def _parse_cot_intent(self, user_message: str) -> Optional[str]:
        if not self.valves.enable_cot_on_demand:
            return None
        if user_message.strip().startswith("/think"):
            parts = user_message.split(maxsplit=1)
            if len(parts) > 1:
                return parts[1].strip()
            return "What would you like me to think step by step about?"
        if re.search(
            r"\b(think step by step|reason step by step)\b", user_message, re.IGNORECASE
        ):
            return user_message
        return None

    async def _generate_cot(self, question: str, context: str) -> str:
        model = (
            self.valves.cot_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""You are a reasoning assistant. Please think step by step to answer the following question.
Show your reasoning clearly, then provide a final answer.

Context:
{context[:2000]}

Question: {question}

Proceed step by step."""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a meticulous reasoner. Output step-by-step reasoning, then a final answer.",
            model_override=model,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.3,
        )
        return response or "Could not generate reasoning."

    async def _parse_assumption_intent(self, user_message: str) -> Optional[str]:
        if not self.valves.enable_assumption_extraction:
            return None
        if user_message.strip().startswith("/assume"):
            parts = user_message.split(maxsplit=1)
            if len(parts) > 1:
                return parts[1].strip()
            return None
        if re.search(
            r"\b(what assumptions|underlying assumptions)\b",
            user_message,
            re.IGNORECASE,
        ):
            return user_message
        return None

    async def _extract_assumptions(self, text: str) -> str:
        model = (
            self.valves.assumption_extraction_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analyze the following statement or code and extract the underlying assumptions.
List each assumption clearly. Also note any unstated premises or biases.

Statement/Code:
{text[:2000]}

Output a structured list of assumptions and a brief comment on their validity or impact."""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are an analytical assistant that extracts hidden assumptions.",
            model_override=model,
            max_tokens=800,
            temperature=0.2,
        )
        return response or "Could not extract assumptions."

    # --------------------------------------------------------------------------
    # Iterative task execution (/iterate)
    # --------------------------------------------------------------------------
    async def _run_iteration(self, project_id: str, command: str) -> Tuple[str, bool]:
        if not self.valves.enable_iterative_mode:
            return "", False
        state = self._get_state(project_id)
        current_iter = state.get("iterative_state")
        if command.startswith("/iterate"):
            parts = command.split(maxsplit=1)
            if len(parts) == 1:
                return (
                    "**Iterative commands:**\n"
                    "- `/iterate <goal>` – start a multi-step plan.\n"
                    "- `/iterate --auto <goal>` – run all steps automatically.\n"
                    "- `/iterate resume` – resume an interrupted iteration.\n"
                    "- You can also use natural language: 'implement all features step by step'.",
                    True,
                )
            action = parts[1].lower()
            if action == "resume":
                if not current_iter:
                    return (
                        "No iteration in progress. Start one with `/iterate <goal>`.",
                        True,
                    )
                return await self._execute_next_step(project_id), True
            elif action.startswith("--auto"):
                goal = " ".join(parts[1:]).replace("--auto", "").strip()
                if not goal:
                    return "You must specify a goal for the iteration.", True
                if await self._start_new_iteration(
                    project_id, goal, auto_continue=True
                ):
                    return await self._execute_next_step(project_id), True
                return "Failed to start iteration.", True
            else:
                goal = parts[1]
                if await self._start_new_iteration(
                    project_id, goal, auto_continue=False
                ):
                    return await self._execute_next_step(project_id), True
                return "Failed to start iteration.", True
        # Natural language continuation
        if current_iter and command.lower() in [
            "siguiente",
            "continue",
            "next",
            "yes",
            "si",
            "ok",
            "apply",
        ]:
            return await self._execute_next_step(project_id), True
        # Natural language start
        if re.search(
            r"\b(implement|do|execute) (all|step by step|iteratively)\b",
            command,
            re.IGNORECASE,
        ):
            goal = command
            if await self._start_new_iteration(project_id, goal, auto_continue=False):
                return await self._execute_next_step(project_id), True
        return "", False

    async def _generate_plan(self, goal: str, context: str) -> List[Dict]:
        model = (
            self.valves.iterative_planning_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""You are an expert Python developer. The user wants to perform the following task:

{goal}

Current context (relevant files, functions, etc.):
{context[:3000]}

Break down the task into a sequence of concrete steps. Each step should be a small, actionable change that can be implemented as a unified diff (patch) to the codebase.
Output a JSON array of steps, each with:
- "description": a short description of what the step does
- "file": the file path to modify (if known, otherwise "unknown")
- "changes": a brief summary of the changes (will be used to generate the diff later)

Example:
[
  {{
    "description": "Add function calculate_average",
    "file": "utils/math.py",
    "changes": "Add function that takes a list and returns average"
  }}
]

The plan must have at most {self.valves.iterative_max_steps} steps. Output only JSON, no extra text.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a precise task planner. Output only JSON.",
            model_override=model,
            max_tokens=1500,
            temperature=0.2,
        )
        if not response:
            return []
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            plan = json.loads(response)
            if isinstance(plan, list):
                return plan[: self.valves.iterative_max_steps]
        except:
            pass
        return []

    async def _generate_diff_for_step(self, step: Dict, project_id: str) -> str:
        model = (
            self.valves.iterative_execution_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        state = self._get_state(project_id)
        active_code = "\n".join(
            [
                b.content
                for b in state["active_blocks"].values()
                if b.content_type == ContentType.BASE_CODE
            ]
        )
        prompt = f"""You are generating a unified diff (patch) to implement a specific change.

File to modify: {step.get('file', 'unknown')}
Change description: {step.get('changes', '')}
Goal: {step.get('description', '')}

Current relevant code (may include multiple files):
{active_code[:3000]}

Generate a unified diff that applies this change. Use the format:

```diff
--- a/file.py
+++ b/file.py
@@ -line,old +line,new @@
 -old line
 +new line
```
If the file does not exist in the provided context, assume it is a new file and create a diff from empty (e.g., use /dev/null as original).
Output only the diff, enclosed in diff ... .
"""

        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code assistant that generates correct unified diffs. Output only the diff with proper formatting.",
            model_override=model,
            max_tokens=2000,
            temperature=0.2,
        )
        if not response:
            return f"# Failed to generate diff for step: {step.get('description')}"
        diff_match = re.search(r"```diff\n(.*?)\n```", response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        if response.strip().startswith("--- "):
            return response.strip()
        return response

    async def _start_new_iteration(
        self, project_id: str, goal: str, auto_continue: bool
    ) -> bool:
        """Start a new iterative task."""
        state = self._get_state(project_id)
        active_ctx = self._get_active_code_context(project_id)
        facts_ctx = self._get_facts_context(project_id)
        context = f"Active code:\n{active_ctx}\n\nFacts:\n{facts_ctx}"
        plan = await self._generate_plan(goal, context)
        if not plan:
            return False
        state["iterative_state"] = {
            "goal": goal,
            "plan": plan,
            "current_step": 0,
            "results": [],
            "auto_continue": auto_continue,
            "created_at": time.time(),
        }
        self._set_state(project_id, state)
        self._log_debug(f"Started iteration for goal: {goal} with {len(plan)} steps")
        return True

    async def _execute_next_step(self, project_id: str) -> str:
        """Execute the next step of an ongoing iteration."""
        state = self._get_state(project_id)
        iter_state = state.get("iterative_state")
        if not iter_state:
            return "No iteration in progress. Use `/iterate <goal>` to start a task."
        plan = iter_state["plan"]
        step_idx = iter_state["current_step"]
        if step_idx >= len(plan):
            state["iterative_state"] = None
            self._set_state(project_id, state)
            return f"✅ **All steps completed!**\n\nGoal: {iter_state['goal']}\nYou can now apply the diffs manually. Use `/iterate` to start a new task."

        step = plan[step_idx]
        pre = f"**Step {step_idx+1}/{len(plan)}: {step.get('description')}**\n- File: {step.get('file', 'unknown')}\n- Changes: {step.get('changes', '')}"
        diff = await self._generate_diff_for_step(step, project_id)
        post = f"\n**Generated diff**\n```diff\n{diff[:1000]}\n```"
        full_output = pre + post

        iter_state["results"].append((step_idx, diff, full_output))
        iter_state["current_step"] = step_idx + 1
        self._set_state(project_id, state)

        if iter_state.get("auto_continue"):
            next_output = (
                await self._execute_next_step(project_id)
                if iter_state["current_step"] < len(plan)
                else ""
            )
            return full_output + "\n\n" + next_output
        else:
            return (
                full_output
                + "\n\n---\nRespond with **next** / **siguiente** to continue."
            )

    # --------------------------------------------------------------------------
    # Message summarization
    # --------------------------------------------------------------------------
    def _get_summary_prompt_for_content(
        self, content_type: ContentType, text: str, max_tokens: int
    ) -> str:
        """Build a prompt for summarization that depends on the content type."""
        if not self.valves.selective_summarization:
            return f"Summarise the following conversation. Keep key decisions, actions, and important context. Be concise.\n\n{text}"
        if content_type == ContentType.ERROR:
            if self.valves.error_preserve_verbatim:
                return text
            else:
                return f"Summarise the following error message, keeping the error type, location, and root cause. Do not omit technical details.\n\n{text}"
        elif content_type in (
            ContentType.BASE_CODE,
            ContentType.PROPOSED_CHANGE,
            ContentType.COMMITTED_CHANGE,
        ):
            level = self.valves.code_summary_level
            if level == "minimal":
                instruction = "Extract only the function/class signatures and the overall purpose. Do not include implementation details."
            elif level == "detailed":
                instruction = "Summarise the code, keeping key functions, classes, important logic, and any comments. Aim for a medium level of detail."
            else:
                instruction = "Summarise the code, focusing on what it does, its main functions/classes, and any non-trivial logic."
            return f"{instruction}\n\n```\n{text[:3000]}\n```"
        elif content_type == ContentType.TOOL_CALL:
            if self.valves.tool_call_preserve:
                return text
            else:
                return f"Summarise the following tool call sequence, keeping the tool names and main parameters.\n\n{text}"
        else:
            return f"Summarise the following conversation, keeping key decisions, action items, and unresolved issues. Be concise (target {max_tokens} tokens).\n\n{text}"

    async def _summarize_messages(
        self, messages: List[dict], is_code_context: bool = False
    ) -> Optional[str]:
        """Summarize a list of messages using an LLM."""
        if not HAS_AIOHTTP or not messages:
            return None
        content_type_counts = defaultdict(int)
        for m in messages:
            content = m.get("content", "")
            extracted = await self._extract_code_blocks(content)
            ctype = self._classify_content(content, extracted)
            content_type_counts[ctype] += 1
        if (
            self.valves.selective_summarization
            and self.valves.error_preserve_verbatim
            and content_type_counts.get(ContentType.ERROR, 0) > 0
        ):
            return "\n".join(
                [f"{m.get('role')}: {m.get('content', '')}" for m in messages]
            )
        conv_text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in messages
        )
        dominant_type = (
            max(content_type_counts.items(), key=lambda x: x[1])[0]
            if content_type_counts
            else ContentType.GENERAL
        )
        prompt = self._get_summary_prompt_for_content(
            dominant_type, conv_text, self.valves.general_summary_max_tokens
        )
        model_override = (
            self.valves.summary_fallback_model
            if self.valves.summary_fallback_model
            else None
        )
        max_tokens = (
            self.valves.general_summary_max_tokens
            if dominant_type == ContentType.GENERAL
            else 500
        )
        summary = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code-aware assistant that produces concise, information-dense summaries.",
            model_override=model_override,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        if not summary:
            return None
        if self.valves.selective_summarization and self.valves.summary_include_metadata:
            summary = f"[Summary of {len(messages)} messages, type: {dominant_type.value}]\n{summary}"
        return summary

    # --------------------------------------------------------------------------
    # Consecutive message deduplication
    # --------------------------------------------------------------------------
    async def _deduplicate_consecutive_messages(
        self, history_msgs: List[dict]
    ) -> List[dict]:
        if self.valves.similar_message_handling == "none" or len(history_msgs) < 2:
            return history_msgs
        new_history = []
        i = 0
        while i < len(history_msgs):
            current = history_msgs[i]
            j = i + 1
            while j < len(history_msgs) and history_msgs[j].get("role") == current.get(
                "role"
            ):
                j += 1
            if j - i > 1:
                first, last = history_msgs[i], history_msgs[j - 1]
                if self.valves.similar_message_check_code_only:
                    contains_code = any(
                        "```" in history_msgs[k].get("content", "") for k in range(i, j)
                    )
                else:
                    contains_code = True
                if (
                    contains_code
                    and self._calculate_code_similarity(
                        first.get("content", ""), last.get("content", "")
                    )
                    >= self.valves.similar_message_threshold
                ):
                    action = self.valves.similar_message_handling
                    if action == "replace":
                        new_history.append(last)
                        i = j
                        continue
                    elif action == "summarize_diff" and j - i == 2:
                        new_history.append(
                            {
                                "role": first.get("role"),
                                "content": await self._summarize_diff_between_messages(
                                    first, last
                                ),
                            }
                        )
                        i = j
                        continue
            new_history.append(current)
            i += 1
        return new_history

    async def _summarize_diff_between_messages(self, msg1: dict, msg2: dict) -> str:
        content1 = msg1.get("content", "")
        content2 = msg2.get("content", "")
        prompt = f"""The user sent two consecutive very similar messages...
FIRST: {content1[:1500]}
SECOND: {content2[:1500]}
Provide a concise summary of what changed."""
        diff_summary = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code-aware assistant.",
            max_tokens=300,
            temperature=0.2,
        )
        return (
            f"[Summary of changes]\n{diff_summary}\n\n[Final version]\n{content2}"
            if diff_summary
            else f"[Updated version]\n{content2}"
        )

    # --------------------------------------------------------------------------
    # Contradiction detection
    # --------------------------------------------------------------------------
    async def _detect_contradictions(
        self, conversation_messages: List[dict]
    ) -> Optional[str]:
        if (
            not self.valves.enable_contradiction_detection
            or len(conversation_messages) < 4
        ):
            return None

        recent = conversation_messages[-10:]
        contradictory_pairs = []
        for i, msg1 in enumerate(recent):
            for msg2 in recent[i + 1 :]:
                if msg1.get("role") != "user" or msg2.get("role") != "user":
                    continue
                sim = self._calculate_code_similarity(
                    msg1.get("content", ""), msg2.get("content", "")
                )
                if sim > 0.6:
                    c1 = msg1.get("content", "").lower()
                    c2 = msg2.get("content", "").lower()
                    if " no " in c1 or "error" in c1 or " no " in c2 or "error" in c2:
                        contradictory_pairs.append((msg1, msg2))
        contradictory_pairs = contradictory_pairs[:3]
        if not contradictory_pairs:
            return None

        pair_texts = []
        for idx, (msg1, msg2) in enumerate(contradictory_pairs, 1):
            pair_texts.append(f"Pair {idx}:")
            pair_texts.append(f"Message A: {msg1.get('content','')[:500]}")
            pair_texts.append(f"Message B: {msg2.get('content','')[:500]}")
            pair_texts.append("---")
        conv_text = "\n".join(pair_texts)

        model = (
            self.valves.contradiction_detection_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analyze the following pairs of user messages for contradictions. A contradiction occurs when the user says something in one message and later says the opposite, or makes statements that conflict with each other.

If you find a contradiction, output JSON: {{"contradiction": true, "explanation": "Brief explanation of the conflict", "suggestion": "What the user might really need"}}
If no contradiction, output {{"contradiction": false}}.

{conv_text}

Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a contradiction detection assistant. Output only JSON.",
            model_override=model,
            max_tokens=300,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("contradiction"):
                return f"⚠️ **Contradiction detected**: {data.get('explanation','')}\n\nSuggestion: {data.get('suggestion','')}"
        except:
            pass
        return None

    # --------------------------------------------------------------------------
    # Code review and auto-CoT detection
    # --------------------------------------------------------------------------
    def _is_code_review_request(self, user_message: str) -> bool:
        """Return True if the user is asking for a code review or bug fix."""
        if not user_message:
            return False
        content = user_message.strip().lower()
        review_keywords = [
            "revisa",
            "revisar",
            "review",
            "check",
            "bug",
            "error",
            "fix",
            "arregla",
            "corrige",
            "mejora",
            "improve",
            "refactor",
            "refactoriza",
            "optimize",
            "optimiza",
            "qué errores tiene",
            "what errors",
            "code review",
        ]
        return any(kw in content for kw in review_keywords)

    def _should_auto_cot(self, user_message: str) -> bool:
        """Return True if the user message looks complex enough for chain-of-thought."""
        if len(user_message) < self.valves.auto_cot_min_chars:
            return False
        indicators = [
            r"\b(if|else|for|while|function|def|class|return)\b",
            r"\b(error|bug|fix|issue|problem|exception|crash)\b",
            r"\b(how|why|explain|what|when|where)\b",
            r"\b(step|implement|create|design|architecture)\b",
            r"[{};]",
        ]
        for pat in indicators:
            if re.search(pat, user_message, re.IGNORECASE):
                return True
        return False

    # --------------------------------------------------------------------------
    # Proactive suggestions
    # --------------------------------------------------------------------------
    async def _check_and_suggest_summarization(
        self, project_id: str, current_tokens: int, max_tokens: int
    ) -> Optional[str]:
        if not self.valves.proactive_summary_threshold or max_tokens <= 0:
            return None
        usage_ratio = current_tokens / max_tokens
        if usage_ratio < self.valves.proactive_summary_threshold:
            return None
        state = self._get_state(project_id)
        if not state:
            return None
        obsolete_count = sum(1 for b in state["active_blocks"].values() if b.obsolete)
        inactive_count = sum(
            1
            for b in state["active_blocks"].values()
            if not b.is_active and not b.obsolete
        )
        suggestion_parts = []
        if obsolete_count > 2:
            suggestion_parts.append(
                f"- You have {obsolete_count} obsolete block(s). Use `/obsolete revive` if needed, or they will be ignored."
            )
        if inactive_count > 5:
            suggestion_parts.append(
                f"- There are {inactive_count} inactive code blocks. Use `/forget` to remove them or `/remember` to pin important ones."
            )
        if not suggestion_parts:
            suggestion_parts.append(
                f"- Context is at {int(usage_ratio*100)}% capacity. Consider summarizing old conversations with `/summarize` or using `/forget` to free space."
            )
        return "⚠️ **Context nearly full**\n" + "\n".join(suggestion_parts)

    async def _suggest_commands(self, project_id: str, state: Dict) -> Optional[str]:
        if not self.valves.enable_command_suggestions:
            return None
        last_suggestion = state.get("last_suggestion_timestamp", 0)
        if (
            time.time() - last_suggestion
            < self.valves.command_suggestion_cooldown_minutes * 60
        ):
            return None
        suggestions = []
        obsolete_count = sum(1 for b in state["active_blocks"].values() if b.obsolete)
        if obsolete_count > 3:
            suggestions.append(
                "`/forget all` or `/obsolete revive` to manage obsolete blocks."
            )
        high_importance_unpinned = [
            b
            for b in state["active_blocks"].values()
            if b.importance_score > 6.0 and not b.pinned
        ]
        if len(high_importance_unpinned) > 2:
            suggestions.append(
                "`/remember` to pin important blocks and prevent them from being summarized or expired."
            )
        if state.get("recent_changes"):
            suggestions.append("`/iterate` to apply pending changes automatically.")
        if len(state["active_blocks"]) > 30:
            suggestions.append(
                "Consider using `/summarize` to compress old conversations (if you have that feature)."
            )
        if suggestions:
            state["last_suggestion_timestamp"] = time.time()
            self._set_state(project_id, state)
            return "💡 **Tip**: " + " ".join(suggestions)
        return None

    # --------------------------------------------------------------------------
    # Duplicate question detection
    # --------------------------------------------------------------------------
    async def _find_duplicate_question(
        self, query: str, project_id: str
    ) -> Optional[Dict]:
        if not self.valves.duplicate_question_threshold or not HAS_SENTENCE:
            return None
        where_filter = {
            "$and": [
                {"project_id": {"$eq": project_id}},
                {"role": {"$eq": "user"}},
            ]
        }
        results = self.memory_collection.get(
            where=where_filter,
            limit=self.valves.duplicate_question_lookback,
            include=["documents", "metadatas"],
        )
        if not results or not results["documents"]:
            return None
        query_embedding = self.embedder.encode(query).tolist()
        best_sim = 0.0
        best_entry = None
        for i, doc in enumerate(results["documents"]):
            doc_embedding = self.embedder.encode(doc).tolist()
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            if sim > best_sim and sim >= self.valves.duplicate_question_threshold:
                best_sim = sim
                best_entry = {
                    "content": doc,
                    "sim": sim,
                    "timestamp": results["metadatas"][0][i].get("timestamp"),
                }
        return best_entry

    # --------------------------------------------------------------------------
    # Diff application
    # --------------------------------------------------------------------------
    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        """Apply a unified diff to an original string and return the result."""
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
        for old_start, old_lines, new_lines in reversed(hunks):
            if result_lines[old_start : old_start + len(old_lines)] != old_lines:
                continue
            result_lines = (
                result_lines[:old_start]
                + new_lines
                + result_lines[old_start + len(old_lines) :]
            )
        return "\n".join(result_lines)

    def _apply_change_with_diff(
        self, base_block: CodeBlock, proposed_block: CodeBlock
    ) -> bool:
        """If the proposed change looks like a diff, apply it to the base block."""
        if proposed_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        if not (
            "@@" in proposed_block.content
            and ("-" in proposed_block.content or "+" in proposed_block.content)
        ):
            return False
        new_code = self._apply_unified_diff(base_block.content, proposed_block.content)
        if new_code and new_code != base_block.content:
            base_block.content = new_code
            base_block.hash = hashlib.md5(new_code.encode()).hexdigest()[:16]
            base_block.timestamp = time.time()
            base_block.is_active = True
            base_block.potentially_affected = False
            base_block.importance_score = min(base_block.importance_score + 2.0, 10.0)
            return True
        return False

    # --------------------------------------------------------------------------
    # Dependency tracking
    # --------------------------------------------------------------------------
    def _extract_dependencies_ast(self, code: str) -> Tuple[List[str], List[str]]:
        """Extract imports and function calls from Python code using AST."""
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
        except SyntaxError:
            pass
        return list(set(imports)), list(set(calls))

    def _extract_dependencies_regex(self, code: str, language: str) -> List[str]:
        """Extract dependencies for non‑Python languages using regular expressions."""
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
        """Extract dependencies using AST (Python) or regex (other languages)."""
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
        # Fallback: LLM extraction
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
        """Refresh the dependency list for a given code block."""
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
        """Mark other blocks as potentially affected when a dependency changes."""
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
        """Refresh dependencies and mark affected blocks asynchronously."""
        if not self.valves.enable_dependency_tracking:
            return
        state = self._get_state(project_id)
        if not state or block_hash not in state["active_blocks"]:
            return
        await self._update_dependencies(block_hash, state)
        await self._mark_affected_blocks(block_hash, state)
        self._set_state(project_id, state)

    async def _clean_affected_flags(self, project_id: str):
        """Remove 'potentially affected' flags after a decay period."""
        if not self.valves.enable_dependency_tracking:
            return
        state = self._get_state(project_id)
        if not state:
            return
        now = time.time()
        decay = self.valves.affected_decay_hours * 3600
        if decay <= 0:
            return
        changed = False
        for block in state["active_blocks"].values():
            if block.potentially_affected and (now - block.affected_timestamp) > decay:
                block.potentially_affected = False
                block._update_importance()
                changed = True
        if changed:
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Oversized code block handling
    # --------------------------------------------------------------------------
    def _estimate_code_tokens(self, code: str) -> int:
        """Roughly estimate the number of tokens in a code snippet."""
        if self.tokenizer:
            return len(self.tokenizer.encode(code))
        return len(code) // 4

    async def _handle_oversized_code_block(self, code: str, language: str) -> str:
        """Truncate, summarise or warn when a code block exceeds the token limit."""
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
            summary = await self._call_llm(
                prompt=f"Summarize the following {language} code block.\n```{language}\n{code[:8000]}\n```",
                system_prompt="You are a code summarization assistant.",
                model_override=model,
                max_tokens=500,
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
    # Duplicate block removal
    # --------------------------------------------------------------------------
    def _remove_duplicate_blocks(self, state: Dict):
        """Remove code blocks that are very similar to a more important block."""
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
                    if block.importance_score >= other.importance_score:
                        to_remove.add(other.hash)
                    else:
                        to_remove.add(block.hash)
        for h in to_remove:
            if h in state["active_blocks"]:
                del state["active_blocks"][h]
        state["recent_changes"] = [
            b for b in state["recent_changes"] if b.hash not in to_remove
        ]
        state["committed_changes"] = [
            b for b in state["committed_changes"] if b.hash not in to_remove
        ]

    # --------------------------------------------------------------------------
    # Hierarchical compression
    # --------------------------------------------------------------------------
    async def _hierarchical_compress(self, project_id: str, state: Dict):
        """Compress older messages into a hierarchical summary."""
        if not self.valves.hierarchical_compression_enabled:
            return
        last_ts = state.get("last_compression_timestamp", 0)
        if time.time() - last_ts < 3600:
            return
        try:
            now = time.time()
            where_filter = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"is_hierarchical_summary": {"$ne": True}},
                    {"timestamp": {"$lt": now}},
                ]
            }
            results = self.memory_collection.get(
                where=where_filter,
                include=["documents", "metadatas", "ids"],
                limit=self.valves.hierarchical_compression_interval_messages * 2,
            )
        except Exception as e:
            self._log_debug(
                f"Failed to fetch messages for hierarchical compression: {e}"
            )
            return
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
        to_compress = pairs[: self.valves.hierarchical_compression_interval_messages]
        texts = "\n---\n".join([doc for _, doc, _ in to_compress])
        model = (
            self.valves.hierarchical_summary_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        summary = await self._call_llm(
            prompt=f"Summarise the following conversation segment...\n{texts[:4000]}",
            system_prompt="You are a code-aware assistant.",
            model_override=model,
            max_tokens=self.valves.hierarchical_summary_max_tokens,
            temperature=0.2,
        )
        if not summary:
            return
        summary_id = f"{project_id}_hierarchical_{int(time.time())}"
        summary_embedding = self.embedder.encode(summary).tolist()
        self.memory_collection.upsert(
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
            documents=[f"[Hierarchical summary]\n{summary}"],
        )
        state["last_compression_timestamp"] = time.time()
        self._set_state(project_id, state)

    # endregion
