import json
import os
import tempfile
import unittest

from fastapi import HTTPException

from app import main as app_main
from app.metrics import MetricsStore
from app.schemas import CreateTemplateFromReviewRequest, GenerateDocumentRequest, UpdateTemplateMappingsRequest


DEFAULT_TEMPLATE_TEXT = (
    "Agreement between {{buyer.name}} for property at {{property.address}} "
    "on {{agreement.date}}."
)


def sample_extraction_payload() -> dict:
    return {
        "document_type": "sale_agreement",
        "agreement_date": "2026-05-17",
        "buyer": {"name": "Jane Doe"},
        "seller": {"name": "Acme Realty Pvt Ltd"},
        "normalized_data": {
            "buyer": {"name": "Jane Doe"},
            "property": {"address": "221B Baker Street, London"},
            "agreement": {"date": "2026-05-17"},
        },
        "dynamic_fields": [
            {
                "canonical_name": "buyer.name",
                "value": "Jane Doe",
                "confidence": 0.98,
                "evidence": "Jane Doe",
            },
            {
                "canonical_name": "property.address",
                "value": "221B Baker Street, London",
                "confidence": 0.95,
                "evidence": "221B Baker Street, London",
            },
            {
                "canonical_name": "agreement.date",
                "value": "2026-05-17",
                "confidence": 0.97,
                "evidence": "2026-05-17",
            },
        ],
        "missing_fields": [],
        "notes": [],
    }


class TemplateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "review_sessions.db")
        self.previous_env = {
            "REVIEW_DB_PATH": os.environ.get("REVIEW_DB_PATH"),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        }

        os.environ["REVIEW_DB_PATH"] = self.db_path
        os.environ.pop("GEMINI_API_KEY", None)

        app_main.get_settings.cache_clear()
        app_main.get_review_store.cache_clear()
        app_main.get_template_store.cache_clear()
        app_main.metrics_store = MetricsStore()

    def tearDown(self) -> None:
        app_main.get_settings.cache_clear()
        app_main.get_review_store.cache_clear()
        app_main.get_template_store.cache_clear()

        if self.previous_env["REVIEW_DB_PATH"] is None:
            os.environ.pop("REVIEW_DB_PATH", None)
        else:
            os.environ["REVIEW_DB_PATH"] = self.previous_env["REVIEW_DB_PATH"]

        if self.previous_env["GEMINI_API_KEY"] is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.previous_env["GEMINI_API_KEY"]

        self.temp_dir.cleanup()

    def create_review_session(
        self,
        *,
        status: str = "approved",
        template_text: str = DEFAULT_TEMPLATE_TEXT,
    ) -> str:
        review_store = app_main.get_review_store()
        review_session_id = review_store.create_pending(
            extraction_payload=sample_extraction_payload(),
            template_text=template_text,
            warnings=[],
        )
        if status != "pending":
            review_store.set_decision(review_session_id, status, "approved in test")
        return review_session_id

    def create_template(self, *, template_text: str = DEFAULT_TEMPLATE_TEXT) -> str:
        review_session_id = self.create_review_session(status="approved", template_text=template_text)
        response = app_main.create_template_from_review(
            review_session_id=review_session_id,
            payload=CreateTemplateFromReviewRequest(template_name="sale agreement test"),
        )
        self.assertEqual(response.status, "draft")
        return response.template_id

    def put_default_mappings(self, template_id: str) -> None:
        response = app_main.update_template_mappings(
            template_id=template_id,
            payload=UpdateTemplateMappingsRequest(
                mappings=[
                    {
                        "placeholder_name": "buyer.name",
                        "source_path": "buyer.legal_name",
                        "entity_type": "buyer",
                    },
                    {
                        "placeholder_name": "property.address",
                        "source_path": "property.registered_address",
                        "entity_type": "property",
                    },
                    {
                        "placeholder_name": "agreement.date",
                        "source_path": "deal.execution_date",
                        "entity_type": "deal",
                    },
                ]
            ),
        )
        self.assertEqual(response.status, "ready")

    def test_create_template_requires_approved_review(self) -> None:
        review_session_id = self.create_review_session(status="pending")

        with self.assertRaises(HTTPException) as context:
            app_main.create_template_from_review(
                review_session_id=review_session_id,
                payload=CreateTemplateFromReviewRequest(template_name="pending template"),
            )

        self.assertEqual(context.exception.status_code, 409)
        self.assertIn("must be approved", context.exception.detail)

    def test_update_template_mappings_rejects_invalid_placeholder(self) -> None:
        template_id = self.create_template()

        response = app_main.update_template_mappings(
            template_id=template_id,
            payload=UpdateTemplateMappingsRequest(
                mappings=[
                    {
                        "placeholder_name": "buyer.name",
                        "source_path": "buyer.legal_name",
                        "entity_type": "buyer",
                    },
                    {
                        "placeholder_name": "unknown.field",
                        "source_path": "deal.unknown",
                        "entity_type": "deal",
                    },
                ]
            ),
        )

        self.assertEqual(response.status_code, 422)
        body = json.loads(response.body)
        self.assertEqual(body["invalid_placeholders"], ["unknown.field"])
        self.assertEqual(body["duplicate_placeholders"], [])

    def test_generate_text_blocks_when_placeholders_are_unmapped(self) -> None:
        template_id = self.create_template()

        response = app_main.generate_template_text(
            template_id=template_id,
            payload=GenerateDocumentRequest(transaction_payload={"buyer": {"legal_name": "Jane Doe"}}),
        )

        self.assertEqual(response.status_code, 422)
        body = json.loads(response.body)
        self.assertEqual(
            body["unmapped_placeholders"],
            ["agreement.date", "buyer.name", "property.address"],
        )
        self.assertEqual(body["missing_payload_fields"], [])

    def test_generate_text_blocks_when_mapped_source_data_is_missing(self) -> None:
        template_id = self.create_template()
        self.put_default_mappings(template_id)

        response = app_main.generate_template_text(
            template_id=template_id,
            payload=GenerateDocumentRequest(
                transaction_payload={
                    "buyer": {"legal_name": "Jane Doe"},
                    "deal": {"execution_date": "2026-05-17"},
                }
            ),
        )

        self.assertEqual(response.status_code, 422)
        body = json.loads(response.body)
        self.assertEqual(body["unmapped_placeholders"], [])
        self.assertEqual(
            body["missing_payload_fields"],
            ["property.address -> property.registered_address"],
        )

    def test_generate_text_renders_document_when_payload_is_complete(self) -> None:
        template_id = self.create_template()
        self.put_default_mappings(template_id)

        response = app_main.generate_template_text(
            template_id=template_id,
            payload=GenerateDocumentRequest(
                transaction_payload={
                    "buyer": {"legal_name": "Jane Doe"},
                    "property": {"registered_address": "221B Baker Street, London"},
                    "deal": {"execution_date": "2026-05-17"},
                }
            ),
        )

        body = response.model_dump()
        self.assertEqual(
            body["rendered_text"],
            "Agreement between Jane Doe for property at 221B Baker Street, London on 2026-05-17.",
        )
        self.assertEqual(
            body["mapped_placeholders"],
            ["agreement.date", "buyer.name", "property.address"],
        )

    def test_generate_pdf_returns_pdf_bytes(self) -> None:
        template_id = self.create_template()
        self.put_default_mappings(template_id)

        response = app_main.generate_template_pdf(
            template_id=template_id,
            payload=GenerateDocumentRequest(
                transaction_payload={
                    "buyer": {"legal_name": "Jane Doe"},
                    "property": {"registered_address": "221B Baker Street, London"},
                    "deal": {"execution_date": "2026-05-17"},
                }
            ),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertTrue(response.body.startswith(b"%PDF"))

    def test_generate_pdf_handles_unicode_punctuation(self) -> None:
        template_text = 'Offer Letter - "OneIoT" - {{buyer.name}} - Hyderabad - 500081'
        template_text = template_text.replace("-", "\u2013", 1).replace('"', "\u201c", 1).replace('"', "\u201d", 1)
        template_id = self.create_template(template_text=template_text)

        response = app_main.update_template_mappings(
            template_id=template_id,
            payload=UpdateTemplateMappingsRequest(
                mappings=[
                    {
                        "placeholder_name": "buyer.name",
                        "source_path": "candidate.legal_name",
                        "entity_type": "candidate",
                    }
                ]
            ),
        )
        self.assertEqual(response.status, "ready")

        generated = app_main.generate_template_pdf(
            template_id=template_id,
            payload=GenerateDocumentRequest(
                transaction_payload={"candidate": {"legal_name": "Aashutosh Karale"}}
            ),
        )

        self.assertEqual(generated.status_code, 200)
        self.assertEqual(generated.headers["content-type"], "application/pdf")
        self.assertTrue(generated.body.startswith(b"%PDF"))

    def test_generate_text_supports_static_value_mappings(self) -> None:
        template_id = self.create_template()

        response = app_main.update_template_mappings(
            template_id=template_id,
            payload=UpdateTemplateMappingsRequest(
                mappings=[
                    {
                        "placeholder_name": "buyer.name",
                        "source_path": "candidate.legal_name",
                        "entity_type": "candidate",
                    },
                    {
                        "placeholder_name": "property.address",
                        "source_type": "static_value",
                        "static_value": "221B Baker Street, London",
                        "entity_type": "property",
                    },
                    {
                        "placeholder_name": "agreement.date",
                        "source_path": "document.execution_date",
                        "entity_type": "document",
                    },
                ]
            ),
        )
        self.assertEqual(response.status, "ready")

        rendered = app_main.generate_template_text(
            template_id=template_id,
            payload=GenerateDocumentRequest(
                transaction_payload={
                    "candidate": {"legal_name": "Aashutosh Karale"},
                    "document": {"execution_date": "2026-05-18"},
                }
            ),
        )

        self.assertEqual(
            rendered.rendered_text,
            "Agreement between Aashutosh Karale for property at 221B Baker Street, London on 2026-05-18.",
        )


if __name__ == "__main__":
    unittest.main()
