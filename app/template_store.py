import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TemplateStoreError(Exception):
    """Raised when persisted template operations fail."""


class TemplateStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    template_id TEXT PRIMARY KEY,
                    source_review_session_id TEXT NOT NULL UNIQUE,
                    template_name TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    extraction_json TEXT NOT NULL,
                    template_text TEXT NOT NULL,
                    placeholders_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS template_field_mappings (
                    template_id TEXT NOT NULL,
                    placeholder_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'payload_path',
                    static_value TEXT,
                    entity_type TEXT,
                    required INTEGER NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (template_id, placeholder_name)
                )
                """
            )
            self._ensure_mapping_column(
                connection=connection,
                column_name="source_type",
                definition="TEXT NOT NULL DEFAULT 'payload_path'",
            )
            self._ensure_mapping_column(
                connection=connection,
                column_name="static_value",
                definition="TEXT",
            )
            connection.commit()

    @staticmethod
    def _ensure_mapping_column(connection: sqlite3.Connection, column_name: str, definition: str) -> None:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(template_field_mappings)").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE template_field_mappings ADD COLUMN {column_name} {definition}")

    @staticmethod
    def _serialize_template(row: sqlite3.Row, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        placeholders = sorted(json.loads(row["placeholders_json"]))
        mapped_placeholders = sorted(mapping["placeholder_name"] for mapping in mappings)
        unmapped_placeholders = [placeholder for placeholder in placeholders if placeholder not in mapped_placeholders]

        return {
            "template_id": row["template_id"],
            "template_name": row["template_name"],
            "source_review_session_id": row["source_review_session_id"],
            "document_type": row["document_type"],
            "status": "ready" if not unmapped_placeholders else "draft",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "template_text": row["template_text"],
            "placeholders": placeholders,
            "mapped_placeholders": mapped_placeholders,
            "unmapped_placeholders": unmapped_placeholders,
            "mappings": mappings,
            "warnings": json.loads(row["warnings_json"]),
            "extraction": json.loads(row["extraction_json"]),
        }

    def _fetch_mappings(self, connection: sqlite3.Connection, template_id: str) -> List[Dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT placeholder_name, source_type, source_path, static_value, entity_type, required, notes
            FROM template_field_mappings
            WHERE template_id = ?
            ORDER BY placeholder_name
            """,
            (template_id,),
        ).fetchall()

        return [
            {
                "placeholder_name": row["placeholder_name"],
                "source_type": row["source_type"] or "payload_path",
                "source_path": row["source_path"],
                "static_value": row["static_value"],
                "entity_type": row["entity_type"],
                "required": bool(row["required"]),
                "notes": row["notes"],
            }
            for row in rows
        ]

    def create_from_review(
        self,
        review_session_id: str,
        template_name: str,
        document_type: str,
        extraction_payload: Dict[str, Any],
        template_text: str,
        placeholders: List[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        existing = self.get_by_review_session(review_session_id)
        if existing is not None:
            return existing

        now = datetime.now(timezone.utc).isoformat()
        template_id = str(uuid.uuid4())
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO templates (
                        template_id,
                        source_review_session_id,
                        template_name,
                        document_type,
                        extraction_json,
                        template_text,
                        placeholders_json,
                        warnings_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        review_session_id,
                        template_name,
                        document_type,
                        json.dumps(extraction_payload),
                        template_text,
                        json.dumps(sorted(set(placeholders))),
                        json.dumps(warnings),
                        now,
                        now,
                    ),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise TemplateStoreError(f"Could not create template: {exc}") from exc

        created = self.get(template_id)
        if created is None:
            raise TemplateStoreError("Template was created but could not be reloaded.")
        return created

    def get_by_review_session(self, review_session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM templates
                    WHERE source_review_session_id = ?
                    """,
                    (review_session_id,),
                ).fetchone()
                if row is None:
                    return None
                mappings = self._fetch_mappings(connection, row["template_id"])
                return self._serialize_template(row, mappings)
        except sqlite3.Error as exc:
            raise TemplateStoreError(f"Could not read template by review session: {exc}") from exc

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM templates
                    WHERE template_id = ?
                    """,
                    (template_id,),
                ).fetchone()
                if row is None:
                    return None
                mappings = self._fetch_mappings(connection, template_id)
                return self._serialize_template(row, mappings)
        except sqlite3.Error as exc:
            raise TemplateStoreError(f"Could not read template: {exc}") from exc

    def replace_mappings(self, template_id: str, mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT template_id FROM templates WHERE template_id = ?",
                    (template_id,),
                ).fetchone()
                if row is None:
                    return None

                connection.execute(
                    "DELETE FROM template_field_mappings WHERE template_id = ?",
                    (template_id,),
                )
                for mapping in mappings:
                    connection.execute(
                        """
                        INSERT INTO template_field_mappings (
                            template_id,
                            placeholder_name,
                            source_path,
                            source_type,
                            static_value,
                            entity_type,
                            required,
                            notes,
                            created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            template_id,
                            mapping["placeholder_name"],
                            mapping.get("source_path", ""),
                            mapping.get("source_type", "payload_path"),
                            mapping.get("static_value"),
                            mapping.get("entity_type"),
                            1 if mapping.get("required", True) else 0,
                            mapping.get("notes"),
                            now,
                            now,
                        ),
                    )

                connection.execute(
                    """
                    UPDATE templates
                    SET updated_at = ?
                    WHERE template_id = ?
                    """,
                    (now, template_id),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise TemplateStoreError(f"Could not update template mappings: {exc}") from exc

        return self.get(template_id)
