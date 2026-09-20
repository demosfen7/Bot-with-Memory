"""Реестр документов и настроек пользователей в SQLite (стандартный sqlite3)."""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                 TEXT PRIMARY KEY,
    user_id            INTEGER NOT NULL,
    original_filename  TEXT NOT NULL,
    saved_path         TEXT NOT NULL,
    uploaded_at        TEXT NOT NULL,
    chunks_count       INTEGER NOT NULL DEFAULT 0,
    status             TEXT NOT NULL,
    file_size_bytes    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id        INTEGER PRIMARY KEY,
    active_doc_id  TEXT,
    updated_at     TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    user_id: int
    original_filename: str
    saved_path: str
    uploaded_at: str
    chunks_count: int
    status: str
    file_size_bytes: int


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        user_id=row["user_id"],
        original_filename=row["original_filename"],
        saved_path=row["saved_path"],
        uploaded_at=row["uploaded_at"],
        chunks_count=row["chunks_count"],
        status=row["status"],
        file_size_bytes=row["file_size_bytes"],
    )


class DocumentRegistry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Соединение на каждую операцию: просто и безопасно для потоков.
        # `with conn` делает commit при успехе и rollback при ошибке.
        with closing(sqlite3.connect(self._db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    # ---- документы ----

    def add_document(
        self,
        doc_id: str,
        user_id: int,
        original_filename: str,
        saved_path: str,
        uploaded_at: str,
        file_size_bytes: int,
    ) -> None:
        """Регистрирует документ со статусом processing (индексация ещё идёт)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, user_id, original_filename, saved_path, uploaded_at,"
                " chunks_count, status, file_size_bytes) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (doc_id, user_id, original_filename, saved_path, uploaded_at, STATUS_PROCESSING, file_size_bytes),
            )

    def mark_ready(self, doc_id: str, chunks_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, chunks_count = ? WHERE id = ?",
                (STATUS_READY, chunks_count, doc_id),
            )

    def get_document(self, user_id: int, doc_id: str) -> DocumentRecord | None:
        """Возвращает документ, только если он принадлежит пользователю."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
            ).fetchone()
        return _to_record(row) if row else None

    def list_documents(self, user_id: int) -> list[DocumentRecord]:
        """Готовые документы пользователя в порядке загрузки (по ним строится нумерация)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE user_id = ? AND status = ? ORDER BY uploaded_at, rowid",
                (user_id, STATUS_READY),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def list_unfinished(self) -> list[DocumentRecord]:
        """Документы, чья индексация оборвалась (например, из-за перезапуска бота)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE status != ?", (STATUS_READY,)).fetchall()
        return [_to_record(row) for row in rows]

    def delete_document(self, user_id: int, doc_id: str) -> bool:
        """Удаляет запись и сбрасывает активный документ, если он был этим. True — если запись была."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
            conn.execute(
                "UPDATE user_settings SET active_doc_id = NULL, updated_at = ?"
                " WHERE user_id = ? AND active_doc_id = ?",
                (utc_now_iso(), user_id, doc_id),
            )
        return cursor.rowcount > 0

    # ---- настройки пользователя ----

    def get_active_doc_id(self, user_id: int) -> str | None:
        """Активный документ пользователя (только если он существует и принадлежит ему)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT s.active_doc_id FROM user_settings s"
                " JOIN documents d ON d.id = s.active_doc_id AND d.user_id = s.user_id"
                " WHERE s.user_id = ? AND d.status = ?",
                (user_id, STATUS_READY),
            ).fetchone()
        return row["active_doc_id"] if row else None

    def set_active_doc_id(self, user_id: int, doc_id: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_settings (user_id, active_doc_id, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET"
                " active_doc_id = excluded.active_doc_id, updated_at = excluded.updated_at",
                (user_id, doc_id, utc_now_iso()),
            )
