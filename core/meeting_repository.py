"""SQLite-backed persistence for completed meeting analyses.

This module is the only place in the codebase that talks to the meetings
database. Both the Streamlit app and the CLI call save_meeting() after a
successful pipeline run; neither talks to sqlite3 directly.

Runtime-only pipeline objects — the RAG chain and the Chroma collection
name returned by core/pipeline.py — are intentionally never read or
persisted here. Reconstructing RAG for a historical meeting is deferred to
a later milestone; see get_meeting()'s docstring.
"""

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "meetings.db")

# Structured fields that are stored as JSON-encoded TEXT and must be decoded
# back into Python lists on read. Kept as a shared constant so save_meeting
# and get_meeting can't drift out of sync on which fields need (de)coding.
_JSON_FIELDS = ("summary", "action_items", "key_decisions", "open_questions")


def _get_connection() -> sqlite3.Connection:
    """Open a short-lived connection, creating data/ on first use.

    Every public function in this module opens one of these, does its
    work, and lets it close — there is no long-held shared connection.
    This keeps concurrent access (e.g. Streamlit's rerun model) simple and
    avoids "database is locked" errors from a connection held open across
    unrelated work.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database() -> None:
    """Create the meetings table if it doesn't already exist.

    Idempotent and safe to call every time an entry point starts up —
    mirrors the pattern already used by core/logging_config.py's
    configure_logging(). No migration framework is used; at this schema
    size, future column additions can use a guarded ALTER TABLE.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                title          TEXT NOT NULL,
                language       TEXT NOT NULL,
                transcript     TEXT NOT NULL,
                summary        TEXT NOT NULL,
                action_items   TEXT NOT NULL,
                key_decisions  TEXT NOT NULL,
                open_questions TEXT NOT NULL,
                word_count     INTEGER NOT NULL,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """)


def save_meeting(result: dict, language: str) -> int:
    """Persist one completed pipeline result and return its new row id.

    `result` is the dict returned by run_meeting_assistant_pipeline().
    Only its plain, JSON-serializable fields (title, transcript, summary,
    action_items, key_decisions, open_questions) are read; runtime-only
    keys such as "rag_chain" and "collection_name" are never accessed, so
    it's safe to pass the full pipeline result straight through.

    `language` is the internal pipeline value ("english" / "hinglish"),
    not a UI display label — mapping to a label is a presentation concern
    for the caller.
    """
    word_count = len(result["transcript"].split())

    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO meetings (
                title, language, transcript, summary,
                action_items, key_decisions, open_questions, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["title"],
                language,
                result["transcript"],
                json.dumps(result["summary"]),
                json.dumps(result["action_items"]),
                json.dumps(result["key_decisions"]),
                json.dumps(result["open_questions"]),
                word_count,
            ),
        )
        meeting_id = cursor.lastrowid

    logger.info("Saved meeting %d (%r) to %s", meeting_id, result["title"], DB_PATH)

    return meeting_id


def _decode_json_field(raw_value: str) -> list:
    """Best-effort JSON decode for one stored structured field.

    Defensive by design, matching the fallback philosophy already used in
    core/transcript_insights.py and core/transcript_summary.py: a single
    corrupted or unexpected value degrades to an empty list rather than
    raising and breaking the whole history view.
    """
    try:
        decoded = json.loads(raw_value)
    except Exception:
        return []

    return decoded if isinstance(decoded, list) else []


def list_meetings() -> list[dict]:
    """Return lightweight metadata for every saved meeting, newest first.

    Deliberately excludes transcript and the structured insight fields —
    those are only needed once a specific meeting is opened via
    get_meeting(), so listing history stays cheap regardless of how many
    or how long past meetings are.

    Each dict has keys: id, title, language, word_count, created_at.
    """
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, title, language, word_count, created_at
            FROM meetings
            ORDER BY created_at DESC, id DESC
            """).fetchall()

    return [dict(row) for row in rows]


def get_meeting(meeting_id: int) -> dict | None:
    """Return one full saved meeting, or None if no row has this id.

    The returned dict has keys: id, title, language, transcript, summary,
    action_items, key_decisions, open_questions, word_count, created_at.
    JSON-encoded fields are decoded back into their original list/dict
    shapes before being returned.

    There is no "rag_chain" key in this dict — it was never persisted (see
    module docstring). Callers loading a historical meeting into the UI
    are expected to treat it as unavailable / None; rebuilding RAG for a
    saved meeting is deferred to a later milestone.
    """
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()

    if row is None:
        return None

    meeting = dict(row)
    for field in _JSON_FIELDS:
        meeting[field] = _decode_json_field(meeting[field])

    return meeting
