"""
Provide SQLite-based storage for chatbot conversations and messages.

This module manages persistent chat history for the application, including
conversation records and associated messages. It initializes the local
database, handles basic conversation and message operations, and supports the
data needed for conversation history and sidebar views.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chat_history.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    language    TEXT NOT NULL DEFAULT 'en',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources         TEXT NOT NULL DEFAULT '[]',
    provider        TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv
    ON messages(conversation_id, created_at);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = _connect()
    conn.executescript(_CREATE_SQL)
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Conversations ────────────────────────────────────────────────────

def create_conversation(title: str = "", language: str = "en") -> dict:
    conn = _connect()
    conv_id = uuid.uuid4().hex[:12]
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, title, language, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, language, now, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def list_conversations(limit: int = 50) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_conversation_title(conv_id: str, title: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, _now(), conv_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id: str) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


# ── Messages ─────────────────────────────────────────────────────────

def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: str = "[]",
    provider: str = "",
) -> dict:
    conn = _connect()
    msg_id = uuid.uuid4().hex[:12]
    now = _now()
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, sources, provider, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, sources, provider, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
    conn.close()
    return dict(row)


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict]:
    conn = _connect()
    if limit:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        rows = list(reversed(rows))
    else:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
