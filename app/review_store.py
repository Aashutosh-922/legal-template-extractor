import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ReviewStoreError(Exception):
    """Raised when persisted review operations fail."""


class ReviewStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_sessions (
                    review_session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    extraction_json TEXT NOT NULL,
                    template_text TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    reviewer_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create_pending(self, extraction_payload: Dict[str, Any], template_text: str, warnings: list) -> str:
        now = datetime.now(timezone.utc).isoformat()
        review_session_id = str(uuid.uuid4())
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO review_sessions (
                        review_session_id, status, extraction_json, template_text, warnings_json, reviewer_notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_session_id,
                        "pending",
                        json.dumps(extraction_payload),
                        template_text,
                        json.dumps(warnings),
                        None,
                        now,
                        now,
                    ),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not create review session: {exc}") from exc
        return review_session_id

    def get(self, review_session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT review_session_id, status, extraction_json, template_text, warnings_json, reviewer_notes, created_at, updated_at
                    FROM review_sessions
                    WHERE review_session_id = ?
                    """,
                    (review_session_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not read review session: {exc}") from exc

        if row is None:
            return None

        return {
            "review_session_id": row[0],
            "status": row[1],
            "extraction": json.loads(row[2]),
            "template_text": row[3],
            "warnings": json.loads(row[4]),
            "reviewer_notes": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def set_decision(self, review_session_id: str, status: str, notes: Optional[str]) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE review_sessions
                    SET status = ?, reviewer_notes = ?, updated_at = ?
                    WHERE review_session_id = ?
                    """,
                    (status, notes, now, review_session_id),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not update review session: {exc}") from exc
        return self.get(review_session_id)
