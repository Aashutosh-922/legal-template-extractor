import unittest

from app.schemas import TemplateFieldMapping
from app.template_engine import extract_placeholders, prepare_generation_context


class TemplateEngineTests(unittest.TestCase):
    def test_extract_placeholders_deduplicates_and_sorts(self) -> None:
        template_text = (
            "Agreement between {{ buyer.name }} and {{property.address}} "
            "dated {{agreement.date}}. {{buyer.name}} confirms receipt."
        )

        placeholders = extract_placeholders(template_text)

        self.assertEqual(placeholders, ["agreement.date", "buyer.name", "property.address"])

    def test_prepare_generation_context_reports_unmapped_and_missing_fields(self) -> None:
        mappings = [
            TemplateFieldMapping(
                placeholder_name="buyer.name",
                source_path="buyer.legal_name",
                entity_type="buyer",
            ),
            TemplateFieldMapping(
                placeholder_name="agreement.date",
                source_path="deal.execution_date",
                entity_type="deal",
            ),
        ]

        result = prepare_generation_context(
            placeholders=["buyer.name", "agreement.date", "property.address"],
            mappings=mappings,
            transaction_payload={
                "buyer": {"legal_name": "Jane Doe"},
                "deal": {},
            },
        )

        self.assertEqual(result.render_context, {"buyer": {"name": "Jane Doe"}})
        self.assertEqual(result.mapped_placeholders, ["buyer.name"])
        self.assertEqual(result.unmapped_placeholders, ["property.address"])
        self.assertEqual(
            result.missing_payload_fields,
            ["agreement.date -> deal.execution_date"],
        )

    def test_prepare_generation_context_supports_static_value_mappings(self) -> None:
        mappings = [
            TemplateFieldMapping(
                placeholder_name="buyer.name",
                source_path="buyer.legal_name",
                entity_type="buyer",
            ),
            TemplateFieldMapping(
                placeholder_name="property.address",
                source_type="static_value",
                static_value="221B Baker Street, London",
                entity_type="property",
            ),
        ]

        result = prepare_generation_context(
            placeholders=["buyer.name", "property.address"],
            mappings=mappings,
            transaction_payload={"buyer": {"legal_name": "Jane Doe"}},
        )

        self.assertEqual(
            result.render_context,
            {
                "buyer": {"name": "Jane Doe"},
                "property": {"address": "221B Baker Street, London"},
            },
        )
        self.assertEqual(result.missing_payload_fields, [])
        self.assertEqual(result.unmapped_placeholders, [])


if __name__ == "__main__":
    unittest.main()
