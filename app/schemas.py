from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Party(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None


class RawField(BaseModel):
    field_name: str = Field(min_length=1)
    field_type: str = Field(min_length=1, description="name, date, currency, id, address, etc.")
    example_value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, description="Exact supporting text from source")


class RawExtractionStage(BaseModel):
    raw_fields: List[RawField] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ClassifiedField(BaseModel):
    canonical_name: str = Field(min_length=1, description="Canonical path like buyer.name")
    field_type: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_field_name: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class ClassificationStage(BaseModel):
    document_type: str = Field(min_length=1)
    classified_fields: List[ClassifiedField] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TemplateStage(BaseModel):
    template_text: str = Field(min_length=1)
    placeholders: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class CompletenessStage(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    low_confidence_fields: List[str] = Field(default_factory=list)
    hallucination_flags: List[str] = Field(default_factory=list)
    required_coverage: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: List[str] = Field(default_factory=list)


class DynamicField(BaseModel):
    canonical_name: str = Field(description="Normalized path, e.g. buyer.name")
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, description="Exact quote from source text")


class AgreementExtraction(BaseModel):
    document_type: str = Field(description="For example: sale_agreement, lease_agreement")
    agreement_date: Optional[str] = None
    buyer: Party = Field(default_factory=Party)
    seller: Party = Field(default_factory=Party)
    total_amount: Optional[str] = None
    currency: Optional[str] = None
    normalized_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generic nested structure assembled from canonical field paths.",
    )
    dynamic_fields: List[DynamicField] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class StageRetries(BaseModel):
    raw_extraction: int = Field(ge=0, default=0)
    classification: int = Field(ge=0, default=0)
    template_generation: int = Field(ge=0, default=0)
    completeness_validation: int = Field(ge=0, default=0)


class PipelineResult(BaseModel):
    raw_fields: List[RawField] = Field(default_factory=list)
    classified_fields: List[ClassifiedField] = Field(default_factory=list)
    template_text: str
    placeholders: List[str] = Field(default_factory=list)
    completeness: CompletenessStage
    chunks_processed: int = Field(ge=1)


class TextExtractionRequest(BaseModel):
    text: str = Field(min_length=30)


class ExtractionResponse(BaseModel):
    extraction: AgreementExtraction
    pipeline: PipelineResult
    retries: StageRetries
    warnings: List[str] = Field(default_factory=list)
    review_session_id: Optional[str] = None


class RenderTemplateRequest(BaseModel):
    template_text: str = Field(min_length=1)
    payload: Dict[str, Any]


class RenderTemplateResponse(BaseModel):
    rendered_text: str


class ReviewDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    notes: Optional[str] = None


class ReviewSessionResponse(BaseModel):
    review_session_id: str
    status: Literal["pending", "approved", "rejected"]
    reviewer_notes: Optional[str] = None
    created_at: str
    updated_at: str
    extraction: AgreementExtraction
    template_text: str
    warnings: List[str] = Field(default_factory=list)


class CreateTemplateFromReviewRequest(BaseModel):
    template_name: Optional[str] = None


class TemplateFieldMapping(BaseModel):
    placeholder_name: str = Field(min_length=1, description="Canonical placeholder path in the template.")
    source_type: Literal["payload_path", "static_value"] = "payload_path"
    source_path: Optional[str] = Field(
        default=None,
        description="Dot path in transaction payload, e.g. deal.buyer.name",
    )
    static_value: Optional[str] = Field(
        default=None,
        description="Static literal value to reuse for this placeholder across generated documents.",
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Logical source entity, e.g. buyer, vendor, property, agent, order.",
    )
    required: bool = True
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self) -> "TemplateFieldMapping":
        if self.source_type == "payload_path":
            if not self.source_path or not self.source_path.strip():
                raise ValueError("payload_path mappings require source_path")
            self.source_path = self.source_path.strip()
            self.static_value = None
            return self

        if self.static_value is None or (isinstance(self.static_value, str) and not self.static_value.strip()):
            raise ValueError("static_value mappings require static_value")
        self.source_path = self.source_path or ""
        return self


class UpdateTemplateMappingsRequest(BaseModel):
    mappings: List[TemplateFieldMapping] = Field(default_factory=list)


class TemplateDefinitionResponse(BaseModel):
    template_id: str
    template_name: str
    source_review_session_id: str
    document_type: str
    status: Literal["draft", "ready"]
    created_at: str
    updated_at: str
    template_text: str
    placeholders: List[str] = Field(default_factory=list)
    mapped_placeholders: List[str] = Field(default_factory=list)
    unmapped_placeholders: List[str] = Field(default_factory=list)
    mappings: List[TemplateFieldMapping] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class GenerateDocumentRequest(BaseModel):
    transaction_payload: Dict[str, Any]


class GenerateDocumentResponse(BaseModel):
    template_id: str
    rendered_text: str
    mapped_placeholders: List[str] = Field(default_factory=list)


class SuggestTemplateMappingsRequest(BaseModel):
    sample_transaction_payload: Dict[str, Any]
    apply_suggestions: bool = False
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    allow_static_value_suggestions: bool = True


class MappingSuggestion(BaseModel):
    placeholder_name: str
    source_type: Literal["payload_path", "static_value", "unresolved"]
    source_path: Optional[str] = None
    static_value: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None


class SuggestTemplateMappingsResponse(BaseModel):
    template_id: str
    suggestions: List[MappingSuggestion] = Field(default_factory=list)
    applied_mappings: List[TemplateFieldMapping] = Field(default_factory=list)
    mapped_placeholders: List[str] = Field(default_factory=list)
    unmapped_placeholders: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TemplateIssueResponse(BaseModel):
    detail: str
    unmapped_placeholders: List[str] = Field(default_factory=list)
    missing_payload_fields: List[str] = Field(default_factory=list)
    invalid_placeholders: List[str] = Field(default_factory=list)
    duplicate_placeholders: List[str] = Field(default_factory=list)


class MetricsSummaryResponse(BaseModel):
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    parse_success_rate: float = 0.0
    average_retries_per_success: float = 0.0
    hallucination_rate: float = 0.0
    average_required_coverage: float = 0.0
    total_missing_fields: int = 0


class ErrorResponse(BaseModel):
    detail: str
