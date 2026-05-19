import json
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from app.schemas import (
    AgreementExtraction,
    ClassificationStage,
    ClassifiedField,
    CompletenessStage,
    DynamicField,
    PipelineResult,
    RawExtractionStage,
    StageRetries,
    TemplateStage,
)


class ExtractionError(Exception):
    """Raised when extraction stages fail."""


@dataclass
class ExtractionResult:
    extraction: AgreementExtraction
    pipeline: PipelineResult
    retries: StageRetries
    warnings: List[str]


RAW_STAGE_PROMPT = """You extract raw variable fields from legal agreements.
Return every dynamic field candidate in the schema:
- field_name
- field_type
- example_value
- confidence
- evidence

Rules:
1. Use only fields explicitly present in input.
2. Evidence must be exact copied text.
3. Do not infer values not grounded in the source text.
"""

CLASSIFICATION_STAGE_PROMPT = """You normalize extracted fields into canonical names.
Map document labels like Buyer Name / Purchaser / Client / Applicant to canonical names such as buyer.name.
Canonical names can span entities such as buyer, client, seller, vendor, company, property, product, deal, order, agent, representative, and agreement.
Examples: buyer.name, vendor.address, property.identifier, deal.close_date, order.total_amount.

Rules:
1. Keep values grounded in provided evidence.
2. Produce canonical names using lower-case dotted paths such as buyer.name, seller.name, agreement.date, agreement.total_amount.
3. Keep confidence conservative.
"""

TEMPLATE_STAGE_PROMPT = """Generate a Jinja2-ready template by replacing dynamic spans with placeholders.
Example: John Doe -> {{buyer.name}}
Example: 12 March 2026 -> {{agreement.date}}

Rules:
1. Keep static legal language intact.
2. Use placeholders from classified canonical names.
3. Return valid Jinja2-compatible template text.
"""

COMPLETENESS_STAGE_PROMPT = """Validate extraction completeness.
Given required canonical fields and classified fields:
1. List missing required fields.
2. List low confidence fields.
3. Keep notes concise and factual.
"""


class StructuredAgreementExtractor:
    def __init__(self, api_key: str, model: str, max_retries: int, required_fields: List[str]) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._required_fields = required_fields
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise ExtractionError(
                "Gemini SDK is not installed. Install `google-genai` and retry."
            ) from exc

        os.environ.setdefault("GEMINI_API_KEY", self._api_key)
        self._client = genai.Client()
        return self._client

    def _parse_stage(
        self,
        response_model: Type[BaseModel],
        stage_name: str,
        system_prompt: str,
        user_content: str,
    ) -> Tuple[BaseModel, int]:
        retry_note = ""
        last_error: Optional[str] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._get_client().models.generate_content(
                    model=self._model,
                    contents="{retry}\n{payload}".format(retry=retry_note, payload=user_content),
                    config={
                        "system_instruction": system_prompt,
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        "response_json_schema": response_model.model_json_schema(),
                    },
                )
                response_text = getattr(response, "text", None)
                if not response_text:
                    raise ExtractionError("{stage} returned no text payload.".format(stage=stage_name))
                return response_model.model_validate_json(response_text), attempt - 1
            except (ValidationError, TypeError, ValueError, ExtractionError, Exception) as exc:
                last_error = str(exc)
                if attempt == self._max_retries:
                    break
                retry_note = (
                    "Previous {stage} attempt failed.\n"
                    "Failure reason: {error}\n"
                    "Fix the output and strictly follow the schema.\n"
                ).format(stage=stage_name, error=last_error)

        raise ExtractionError(
            "{stage} failed after {attempts} attempts: {error}".format(
                stage=stage_name, attempts=self._max_retries, error=last_error
            )
        )

    @staticmethod
    def _deduplicate_raw_fields(raw_fields: List) -> List:
        best_by_key = {}
        for field in raw_fields:
            key = "{name}::{value}".format(
                name=field.field_name.strip().lower(), value=field.example_value.strip().lower()
            )
            current = best_by_key.get(key)
            if current is None or field.confidence > current.confidence:
                best_by_key[key] = field
        return list(best_by_key.values())

    @staticmethod
    def _local_template_fallback(source_text: str, classified_fields: List[ClassifiedField]) -> TemplateStage:
        template_text = source_text
        placeholders: List[str] = []
        ordered_fields = sorted(
            classified_fields,
            key=lambda field: len(field.evidence if field.evidence else field.value),
            reverse=True,
        )

        for field in ordered_fields:
            placeholder = "{{" + field.canonical_name + "}}"
            for candidate in [field.evidence, field.value]:
                if candidate and candidate in template_text:
                    template_text = template_text.replace(candidate, placeholder)
            placeholders.append(field.canonical_name)

        unique_placeholders = sorted(set(placeholders))
        return TemplateStage(
            template_text=template_text,
            placeholders=unique_placeholders,
            notes=["Template stage fallback applied after generation failure."],
        )

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _contains_loosely(cls, corpus: str, value: str) -> bool:
        if not value.strip():
            return False
        return cls._normalize_space(value) in cls._normalize_space(corpus)

    def _grounding_flags(self, fields: List[ClassifiedField], source_text: str) -> List[str]:
        flags: List[str] = []
        for field in fields:
            evidence_ok = self._contains_loosely(source_text, field.evidence)
            value_ok = self._contains_loosely(source_text, field.value) or self._contains_loosely(
                field.evidence, field.value
            )
            if not evidence_ok:
                flags.append(
                    "evidence_not_grounded:{name}".format(name=field.canonical_name)
                )
            if not value_ok:
                flags.append(
                    "value_not_grounded:{name}".format(name=field.canonical_name)
                )
        return sorted(set(flags))

    def _required_missing(self, fields: List[ClassifiedField]) -> List[str]:
        present = {field.canonical_name for field in fields}
        return [field for field in self._required_fields if field not in present]

    @staticmethod
    def _set_normalized_value(target: dict, path: str, value: str) -> None:
        segments = [segment for segment in path.split(".") if segment]
        if not segments:
            return

        current = target
        for segment in segments[:-1]:
            existing = current.get(segment)
            if not isinstance(existing, dict):
                existing = {}
                current[segment] = existing
            current = existing

        current[segments[-1]] = value

    @staticmethod
    def _to_agreement_extraction(
        document_type: str,
        classified_fields: List[ClassifiedField],
        missing_fields: List[str],
        notes: List[str],
    ) -> AgreementExtraction:
        extraction = AgreementExtraction(document_type=document_type, missing_fields=missing_fields, notes=notes)
        dynamic_fields: List[DynamicField] = []

        for field in classified_fields:
            dynamic = DynamicField(
                canonical_name=field.canonical_name,
                value=field.value,
                confidence=field.confidence,
                evidence=field.evidence,
            )
            dynamic_fields.append(dynamic)
            StructuredAgreementExtractor._set_normalized_value(
                extraction.normalized_data,
                field.canonical_name,
                field.value,
            )

            if field.canonical_name == "buyer.name":
                extraction.buyer.name = field.value
            elif field.canonical_name == "buyer.address":
                extraction.buyer.address = field.value
            elif field.canonical_name == "buyer.tax_id":
                extraction.buyer.tax_id = field.value
            elif field.canonical_name == "seller.name":
                extraction.seller.name = field.value
            elif field.canonical_name == "seller.address":
                extraction.seller.address = field.value
            elif field.canonical_name == "seller.tax_id":
                extraction.seller.tax_id = field.value
            elif field.canonical_name == "agreement.date":
                extraction.agreement_date = field.value
            elif field.canonical_name == "agreement.total_amount":
                extraction.total_amount = field.value
            elif field.canonical_name == "agreement.currency":
                extraction.currency = field.value

        extraction.dynamic_fields = dynamic_fields
        return extraction

    def extract(self, text: str, chunks: List[str]) -> ExtractionResult:
        if not chunks:
            raise ExtractionError("Chunked input is empty.")

        warnings: List[str] = []
        raw_retries_used = 0
        aggregated_raw = []

        for index, chunk in enumerate(chunks, start=1):
            user_content = "Chunk {index}/{total}\n\n{chunk}".format(index=index, total=len(chunks), chunk=chunk)
            parsed_raw, retries = self._parse_stage(
                response_model=RawExtractionStage,
                stage_name="raw_extraction",
                system_prompt=RAW_STAGE_PROMPT,
                user_content=user_content,
            )
            raw_retries_used += retries
            aggregated_raw.extend(parsed_raw.raw_fields)

        deduped_raw_fields = self._deduplicate_raw_fields(aggregated_raw)
        if not deduped_raw_fields:
            raise ExtractionError("Raw extraction produced no usable fields.")

        classification_payload = {
            "required_fields": self._required_fields,
            "raw_fields": [field.model_dump() for field in deduped_raw_fields],
        }
        parsed_classification, classification_retries = self._parse_stage(
            response_model=ClassificationStage,
            stage_name="classification",
            system_prompt=CLASSIFICATION_STAGE_PROMPT,
            user_content=json.dumps(classification_payload),
        )

        if not parsed_classification.classified_fields:
            raise ExtractionError("Classification stage returned no normalized fields.")

        template_payload = {
            "source_text": text[:15000],
            "classified_fields": [field.model_dump() for field in parsed_classification.classified_fields],
        }
        template_retries = 0
        try:
            parsed_template, template_retries = self._parse_stage(
                response_model=TemplateStage,
                stage_name="template_generation",
                system_prompt=TEMPLATE_STAGE_PROMPT,
                user_content=json.dumps(template_payload),
            )
        except ExtractionError:
            parsed_template = self._local_template_fallback(text, parsed_classification.classified_fields)
            warnings.append("Template generation stage failed; deterministic fallback template was used.")

        completeness_payload = {
            "required_fields": self._required_fields,
            "classified_fields": [field.model_dump() for field in parsed_classification.classified_fields],
        }
        parsed_completeness, completeness_retries = self._parse_stage(
            response_model=CompletenessStage,
            stage_name="completeness_validation",
            system_prompt=COMPLETENESS_STAGE_PROMPT,
            user_content=json.dumps(completeness_payload),
        )

        local_missing = self._required_missing(parsed_classification.classified_fields)
        grounding_flags = self._grounding_flags(parsed_classification.classified_fields, text)

        missing_fields = sorted(set(parsed_completeness.missing_fields + local_missing))
        low_confidence_fields = sorted(
            set(
                parsed_completeness.low_confidence_fields
                + [
                    field.canonical_name
                    for field in parsed_classification.classified_fields
                    if field.confidence < 0.65
                ]
            )
        )
        required_total = float(len(self._required_fields)) if self._required_fields else 1.0
        required_coverage = max(0.0, 1.0 - (float(len(missing_fields)) / required_total))

        completeness = CompletenessStage(
            missing_fields=missing_fields,
            low_confidence_fields=low_confidence_fields,
            hallucination_flags=sorted(set(parsed_completeness.hallucination_flags + grounding_flags)),
            required_coverage=required_coverage,
            notes=parsed_completeness.notes,
        )

        if completeness.missing_fields:
            warnings.append("Missing required fields detected; review is required before template finalization.")
        if completeness.hallucination_flags:
            warnings.append("Potential hallucination risk detected via grounding checks.")

        extraction_notes = parsed_classification.notes + parsed_template.notes + completeness.notes
        agreement_extraction = self._to_agreement_extraction(
            document_type=parsed_classification.document_type,
            classified_fields=parsed_classification.classified_fields,
            missing_fields=completeness.missing_fields,
            notes=sorted(set(extraction_notes)),
        )

        pipeline = PipelineResult(
            raw_fields=deduped_raw_fields,
            classified_fields=parsed_classification.classified_fields,
            template_text=parsed_template.template_text,
            placeholders=sorted(set(parsed_template.placeholders)),
            completeness=completeness,
            chunks_processed=len(chunks),
        )

        retries = StageRetries(
            raw_extraction=raw_retries_used,
            classification=classification_retries,
            template_generation=template_retries,
            completeness_validation=completeness_retries,
        )

        return ExtractionResult(
            extraction=agreement_extraction,
            pipeline=pipeline,
            retries=retries,
            warnings=warnings,
        )
