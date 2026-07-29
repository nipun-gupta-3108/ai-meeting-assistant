"""SQLite-backed storage for user accounts.

This module is the only place in the codebase that talks to the users
table. It owns SQL only — no password hashing, validation, or business
rules live here (see core/auth.py for those). This mirrors the separation
of concerns already established by core/meeting_repository.py, which owns
the meetings table the same way.
"""

import os
import sqlite3
import uuid

# Same database file as core/meeting_repository.py — a single sqlite file
# for this portfolio project rather than splitting storage across
# multiple databases. meeting_repository.py imports this constant rather
# than redefining it, so the two modules can never drift onto different
# files.
DB_PATH = os.path.join("data", "meetings.db")


def _get_connection() -> sqlite3.Connection:
    """Open a short-lived connection, creating data/ on first use.

    Foreign key enforcement is turned on explicitly — SQLite has it off
    by default per connection, and core/meeting_repository.py's
    meetings.user_id column relies on it being enforced.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_users_table() -> None:
    """Create the users table if it doesn't already exist.

    Idempotent and safe to call on every app startup, matching the
    pattern used by core/meeting_repository.py's initialize_database().
    Must be called before core.meeting_repository.initialize_database(),
    since meetings.user_id references this table.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """)


def create_user(name: str, email: str, password_hash: str) -> dict:
    """Insert one new user row and return it as a dict.

    Pure storage: `email` must already be normalized (lowercased/
    stripped) and `password_hash` must already be a bcrypt hash — both
    are the caller's (core/auth.py's) responsibility, not this module's.

    Raises sqlite3.IntegrityError if `email` is already taken (UNIQUE
    constraint). core/auth.py is responsible for checking for that
    ahead of time and raising a typed, user-facing error instead.
    """
    user_id = uuid.uuid4().hex

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (id, name, email, password_hash)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, name, email, password_hash),
        )

    return get_user_by_id(user_id)


def get_user_by_email(email: str) -> dict | None:
    """Look up one user by exact email match, or None if there isn't one.

    Caller must pass an already-normalized email (see
    core.auth.normalize_email) — no normalization happens at this layer.
    """
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    """Look up one user by id, or None if there isn't one."""
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return dict(row) if row else None
