from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.chunking import chunk_text
from app.config import Settings
from app.extractor import ExtractionError, ExtractionResult, StructuredAgreementExtractor
from app.metrics import MetricsStore
from app.pdf import PDFExtractionError, extract_text_from_pdf_bytes
from app.review_store import ReviewStore, ReviewStoreError
from app.schemas import (
    CreateTemplateFromReviewRequest,
    ErrorResponse,
    ExtractionResponse,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    MappingSuggestion,
    MetricsSummaryResponse,
    RenderTemplateRequest,
    RenderTemplateResponse,
    ReviewDecisionRequest,
    ReviewSessionResponse,
    SuggestTemplateMappingsRequest,
    SuggestTemplateMappingsResponse,
    TemplateDefinitionResponse,
    TemplateFieldMapping,
    TemplateIssueResponse,
    TextExtractionRequest,
    UpdateTemplateMappingsRequest,
)
from app.template_engine import extract_placeholders, prepare_generation_context
from app.template_suggester import TemplateMappingSuggester
from app.template_store import TemplateStore, TemplateStoreError
from app.templating import TemplateRenderingError, render_template_text, render_text_to_pdf_bytes

app = FastAPI(
    title="Legal Template Extractor",
    version="0.3.0",
    description=(
        "Reliability-first legal extraction pipeline with staged structured outputs, "
        "validation/retries, template review, field mapping, and document generation."
    ),
)

metrics_store = MetricsStore()


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_review_store() -> ReviewStore:
    settings = get_settings()
    return ReviewStore(settings.review_db_path)


@lru_cache
def get_template_store() -> TemplateStore:
    settings = get_settings()
    return TemplateStore(settings.review_db_path)


def build_extractor(settings: Settings) -> StructuredAgreementExtractor:
    if not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    return StructuredAgreementExtractor(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_retries=settings.extraction_max_retries,
        required_fields=settings.required_field_paths,
    )


def build_template_suggester(settings: Settings) -> TemplateMappingSuggester:
    return TemplateMappingSuggester(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )


def _record_success(result: ExtractionResult) -> None:
    metrics_store.record_success(
        retries_used=(
            result.retries.raw_extraction
            + result.retries.classification
            + result.retries.template_generation
            + result.retries.completeness_validation
        ),
        hallucination_flags=len(result.pipeline.completeness.hallucination_flags),
        missing_fields=len(result.pipeline.completeness.missing_fields),
        required_coverage=result.pipeline.completeness.required_coverage,
    )


def _template_issue_response(
    detail: str,
    *,
    unmapped_placeholders: Optional[List[str]] = None,
    missing_payload_fields: Optional[List[str]] = None,
    invalid_placeholders: Optional[List[str]] = None,
    duplicate_placeholders: Optional[List[str]] = None,
    status_code: int = 422,
) -> JSONResponse:
    issue = TemplateIssueResponse(
        detail=detail,
        unmapped_placeholders=unmapped_placeholders or [],
        missing_payload_fields=missing_payload_fields or [],
        invalid_placeholders=invalid_placeholders or [],
        duplicate_placeholders=duplicate_placeholders or [],
    )
    return JSONResponse(status_code=status_code, content=issue.model_dump())


def _get_review_record(review_session_id: str) -> Dict[str, object]:
    review_store = get_review_store()
    try:
        record = review_store.get(review_session_id)
    except ReviewStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return record


def _get_template_record(template_id: str) -> Dict[str, object]:
    template_store = get_template_store()
    try:
        record = template_store.get(template_id)
    except TemplateStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(status_code=404, detail="Template not found.")
    return record


def _validate_mapping_payload(
    template_placeholders: List[str],
    mappings: List[TemplateFieldMapping],
) -> Optional[JSONResponse]:
    placeholder_set = set(template_placeholders)
    invalid_placeholders = sorted(
        {mapping.placeholder_name for mapping in mappings if mapping.placeholder_name not in placeholder_set}
    )

    seen = set()
    duplicate_placeholders = set()
    for mapping in mappings:
        if mapping.placeholder_name in seen:
            duplicate_placeholders.add(mapping.placeholder_name)
        seen.add(mapping.placeholder_name)

    if invalid_placeholders or duplicate_placeholders:
        return _template_issue_response(
            detail="Template mappings contain invalid or duplicate placeholders.",
            invalid_placeholders=invalid_placeholders,
            duplicate_placeholders=sorted(duplicate_placeholders),
        )

    return None


def _prepare_rendered_template(
    template_record: Dict[str, object],
    transaction_payload: Dict[str, object],
) -> Tuple[str, List[str], Optional[JSONResponse]]:
    try:
        preparation = prepare_generation_context(
            placeholders=template_record["placeholders"],
            mappings=[TemplateFieldMapping(**mapping) for mapping in template_record["mappings"]],
            transaction_payload=transaction_payload,
        )
    except ValueError as exc:
        return "", [], _template_issue_response(detail=f"Template mapping structure is invalid: {exc}")

    if preparation.unmapped_placeholders or preparation.missing_payload_fields:
        return (
            "",
            preparation.mapped_placeholders,
            _template_issue_response(
                detail="Generation blocked because some placeholders are unmapped or source data is missing.",
                unmapped_placeholders=preparation.unmapped_placeholders,
                missing_payload_fields=preparation.missing_payload_fields,
            ),
        )

    try:
        rendered = render_template_text(template_record["template_text"], preparation.render_context)
    except TemplateRenderingError as exc:
        return (
            "",
            preparation.mapped_placeholders,
            _template_issue_response(detail=f"Template rendering failed: {exc}"),
        )

    return rendered, preparation.mapped_placeholders, None


def _best_extracted_values(template_record: Dict[str, object]) -> Dict[str, str]:
    extraction = template_record.get("extraction") or {}
    dynamic_fields = extraction.get("dynamic_fields", []) if isinstance(extraction, dict) else []
    best: Dict[str, Tuple[float, str]] = {}

    for field in dynamic_fields:
        canonical_name = field.get("canonical_name")
        value = field.get("value")
        confidence = float(field.get("confidence", 0.0))
        if not canonical_name or not value:
            continue
        current = best.get(canonical_name)
        if current is None or confidence > current[0]:
            best[canonical_name] = (confidence, value)

    return {key: value for key, (_, value) in best.items()}


def _merge_suggested_mappings(
    existing_mappings: List[Dict[str, object]],
    suggestions: List[MappingSuggestion],
    confidence_threshold: float,
) -> Tuple[List[TemplateFieldMapping], List[TemplateFieldMapping]]:
    merged: Dict[str, TemplateFieldMapping] = {
        mapping["placeholder_name"]: TemplateFieldMapping(**mapping)
        for mapping in existing_mappings
    }
    applied: List[TemplateFieldMapping] = []

    for suggestion in suggestions:
        if suggestion.placeholder_name in merged:
            continue
        if suggestion.source_type == "unresolved" or suggestion.confidence < confidence_threshold:
            continue

        mapping = TemplateFieldMapping(
            placeholder_name=suggestion.placeholder_name,
            source_type="static_value" if suggestion.source_type == "static_value" else "payload_path",
            source_path=suggestion.source_path,
            static_value=suggestion.static_value,
            entity_type=suggestion.entity_type,
            notes=suggestion.notes,
        )
        merged[mapping.placeholder_name] = mapping
        applied.append(mapping)

    return list(merged.values()), applied


def extract_from_text(text: str, seed_warnings: Optional[List[str]] = None) -> ExtractionResponse:
    settings = get_settings()
    review_store = get_review_store()

    if len(text) > settings.max_input_characters:
        raise HTTPException(
            status_code=422,
            detail="Input too large ({size} chars). Limit is {limit}.".format(
                size=len(text), limit=settings.max_input_characters
            ),
        )

    chunks = chunk_text(
        text=text,
        chunk_size=settings.chunk_size_characters,
        overlap=settings.chunk_overlap_characters,
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="Input text is empty after normalization.")
    if len(chunks) > settings.max_chunks:
        raise HTTPException(
            status_code=422,
            detail="Input requires {count} chunks, exceeding limit {limit}.".format(
                count=len(chunks), limit=settings.max_chunks
            ),
        )

    extractor = build_extractor(settings)
    try:
        result = extractor.extract(text=text, chunks=chunks)
    except ExtractionError as exc:
        metrics_store.record_failure()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    warnings = (seed_warnings or []) + result.warnings
    try:
        review_session_id = review_store.create_pending(
            extraction_payload=result.extraction.model_dump(),
            template_text=result.pipeline.template_text,
            warnings=warnings,
        )
    except ReviewStoreError as exc:
        metrics_store.record_failure()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _record_success(result)
    return ExtractionResponse(
        extraction=result.extraction,
        pipeline=result.pipeline,
        retries=result.retries,
        warnings=warnings,
        review_session_id=review_session_id,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    settings = get_settings()
    status = "ok" if settings.gemini_api_key else "degraded"
    return {"status": status, "model": settings.gemini_model}


@app.post("/extract/text", response_model=ExtractionResponse, responses={500: {"model": ErrorResponse}})
def extract_text(payload: TextExtractionRequest) -> ExtractionResponse:
    return extract_from_text(payload.text)


@app.post(
    "/extract/pdf",
    response_model=ExtractionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def extract_pdf(file: UploadFile = File(...)) -> ExtractionResponse:
    settings = get_settings()
    filename = file.filename or ""
    content_type = file.content_type or ""
    is_pdf = filename.lower().endswith(".pdf") or content_type in {"application/pdf", "application/octet-stream"}
    if not is_pdf:
        raise HTTPException(status_code=422, detail="Only PDF uploads are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Uploaded PDF is empty.")

    try:
        extraction = extract_text_from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            enable_ocr_fallback=settings.enable_ocr_fallback,
        )
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    seed_warnings: List[str] = []
    if extraction.ocr_used:
        seed_warnings.append("OCR fallback was used for one or more pages.")

    return extract_from_text(extraction.text, seed_warnings=seed_warnings)


@app.post("/render/text", response_model=RenderTemplateResponse, responses={422: {"model": ErrorResponse}})
def render_text(payload: RenderTemplateRequest) -> RenderTemplateResponse:
    try:
        rendered = render_template_text(payload.template_text, payload.payload)
    except TemplateRenderingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RenderTemplateResponse(rendered_text=rendered)


@app.post("/render/pdf", responses={422: {"model": ErrorResponse}})
def render_pdf(payload: RenderTemplateRequest) -> Response:
    try:
        rendered = render_template_text(payload.template_text, payload.payload)
        pdf_bytes = render_text_to_pdf_bytes(rendered)
    except TemplateRenderingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=pdf_bytes, media_type="application/pdf")


@app.get(
    "/review/sessions/{review_session_id}",
    response_model=ReviewSessionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_review_session(review_session_id: str) -> ReviewSessionResponse:
    record = _get_review_record(review_session_id)
    return ReviewSessionResponse(**record)


@app.post(
    "/review/sessions/{review_session_id}/decision",
    response_model=ReviewSessionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def decide_review_session(review_session_id: str, payload: ReviewDecisionRequest) -> ReviewSessionResponse:
    review_store = get_review_store()
    try:
        updated = review_store.set_decision(
            review_session_id=review_session_id,
            status=payload.decision,
            notes=payload.notes,
        )
    except ReviewStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return ReviewSessionResponse(**updated)


@app.post(
    "/templates/from-review/{review_session_id}",
    response_model=TemplateDefinitionResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": TemplateIssueResponse},
        500: {"model": ErrorResponse},
    },
)
def create_template_from_review(
    review_session_id: str,
    payload: Optional[CreateTemplateFromReviewRequest] = None,
):
    review_record = _get_review_record(review_session_id)
    if review_record["status"] != "approved":
        raise HTTPException(
            status_code=409,
            detail="Review session must be approved before creating a reusable template.",
        )

    template_text = str(review_record["template_text"])
    placeholders = extract_placeholders(template_text)
    if not placeholders:
        return _template_issue_response(
            detail="No Jinja placeholders were found in the confirmed template. Re-run extraction or reject the review session.",
        )

    extraction = dict(review_record["extraction"])
    document_type = str(extraction.get("document_type", "document"))
    template_name = (
        payload.template_name.strip()
        if payload and payload.template_name and payload.template_name.strip()
        else f"{document_type} template"
    )

    template_store = get_template_store()
    try:
        template_record = template_store.create_from_review(
            review_session_id=review_session_id,
            template_name=template_name,
            document_type=document_type,
            extraction_payload=extraction,
            template_text=template_text,
            placeholders=placeholders,
            warnings=list(review_record["warnings"]),
        )
    except TemplateStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TemplateDefinitionResponse(**template_record)


@app.get(
    "/templates/{template_id}",
    response_model=TemplateDefinitionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_template(template_id: str) -> TemplateDefinitionResponse:
    return TemplateDefinitionResponse(**_get_template_record(template_id))


@app.put(
    "/templates/{template_id}/mappings",
    response_model=TemplateDefinitionResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": TemplateIssueResponse},
        500: {"model": ErrorResponse},
    },
)
def update_template_mappings(
    template_id: str,
    payload: UpdateTemplateMappingsRequest,
):
    template_record = _get_template_record(template_id)
    validation_error = _validate_mapping_payload(
        template_placeholders=list(template_record["placeholders"]),
        mappings=payload.mappings,
    )
    if validation_error is not None:
        return validation_error

    template_store = get_template_store()
    try:
        updated = template_store.replace_mappings(
            template_id=template_id,
            mappings=[mapping.model_dump() for mapping in payload.mappings],
        )
    except TemplateStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Template not found.")
    return TemplateDefinitionResponse(**updated)


@app.post(
    "/templates/{template_id}/suggest-mappings",
    response_model=SuggestTemplateMappingsResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": TemplateIssueResponse},
        500: {"model": ErrorResponse},
    },
)
def suggest_template_mappings(
    template_id: str,
    payload: SuggestTemplateMappingsRequest,
) -> SuggestTemplateMappingsResponse:
    template_record = _get_template_record(template_id)
    settings = get_settings()
    suggester = build_template_suggester(settings)
    extracted_values = _best_extracted_values(template_record)

    suggestions, warnings = suggester.suggest(
        document_type=str(template_record["document_type"]),
        placeholders=list(template_record["placeholders"]),
        extracted_values=extracted_values,
        sample_transaction_payload=payload.sample_transaction_payload,
        allow_static_value_suggestions=payload.allow_static_value_suggestions,
    )

    applied_mappings: List[TemplateFieldMapping] = []
    mapped_placeholders = list(template_record["mapped_placeholders"])
    unmapped_placeholders = list(template_record["unmapped_placeholders"])

    if payload.apply_suggestions:
        merged, applied_mappings = _merge_suggested_mappings(
            existing_mappings=list(template_record["mappings"]),
            suggestions=suggestions,
            confidence_threshold=payload.confidence_threshold,
        )
        validation_error = _validate_mapping_payload(
            template_placeholders=list(template_record["placeholders"]),
            mappings=merged,
        )
        if validation_error is not None:
            return validation_error

        template_store = get_template_store()
        try:
            updated = template_store.replace_mappings(
                template_id=template_id,
                mappings=[mapping.model_dump() for mapping in merged],
            )
        except TemplateStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if updated is None:
            raise HTTPException(status_code=404, detail="Template not found.")

        mapped_placeholders = list(updated["mapped_placeholders"])
        unmapped_placeholders = list(updated["unmapped_placeholders"])

    return SuggestTemplateMappingsResponse(
        template_id=template_id,
        suggestions=suggestions,
        applied_mappings=applied_mappings,
        mapped_placeholders=mapped_placeholders,
        unmapped_placeholders=unmapped_placeholders,
        warnings=warnings,
    )


@app.post(
    "/templates/{template_id}/generate/text",
    response_model=GenerateDocumentResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": TemplateIssueResponse},
        500: {"model": ErrorResponse},
    },
)
def generate_template_text(template_id: str, payload: GenerateDocumentRequest):
    template_record = _get_template_record(template_id)
    rendered, mapped_placeholders, error_response = _prepare_rendered_template(
        template_record=template_record,
        transaction_payload=payload.transaction_payload,
    )
    if error_response is not None:
        return error_response

    return GenerateDocumentResponse(
        template_id=template_id,
        rendered_text=rendered,
        mapped_placeholders=mapped_placeholders,
    )


@app.post(
    "/templates/{template_id}/generate/pdf",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": TemplateIssueResponse},
        500: {"model": ErrorResponse},
    },
)
def generate_template_pdf(template_id: str, payload: GenerateDocumentRequest) -> Response:
    template_record = _get_template_record(template_id)
    rendered, _, error_response = _prepare_rendered_template(
        template_record=template_record,
        transaction_payload=payload.transaction_payload,
    )
    if error_response is not None:
        return error_response

    try:
        pdf_bytes = render_text_to_pdf_bytes(rendered)
    except TemplateRenderingError as exc:
        return _template_issue_response(detail=f"PDF rendering failed: {exc}")

    return Response(content=pdf_bytes, media_type="application/pdf")


@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary() -> MetricsSummaryResponse:
    return MetricsSummaryResponse(**metrics_store.summary())
