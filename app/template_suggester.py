import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

from app.schemas import MappingSuggestion

ENTITY_ALIAS_GROUPS = {
    "party_a": {"buyer", "client", "customer", "candidate", "applicant", "employee", "intern", "tenant"},
    "party_b": {"seller", "vendor", "company", "issuer", "employer", "provider", "landlord", "organization"},
    "agreement": {"agreement", "contract", "deal", "document", "order", "transaction", "offer"},
    "asset": {"property", "product", "asset", "unit", "listing", "item"},
    "representative": {"agent", "representative", "broker", "manager", "supervisor"},
}

TOKEN_SPLIT_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


class MappingSuggestionError(Exception):
    """Raised when mapping suggestion generation fails."""


@dataclass
class PayloadPathInfo:
    path: str
    value: Any
    value_type: str


class MappingSuggestionBatch(BaseModel):
    suggestions: List[MappingSuggestion]


def flatten_payload_paths(payload: Any, prefix: str = "") -> List[PayloadPathInfo]:
    if isinstance(payload, dict):
        items: List[PayloadPathInfo] = []
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            items.extend(flatten_payload_paths(value, child_prefix))
        return items

    if isinstance(payload, list):
        items = []
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            items.extend(flatten_payload_paths(value, child_prefix))
        return items

    return [PayloadPathInfo(path=prefix, value=payload, value_type=type(payload).__name__)]


def _path_tokens(path: str) -> List[str]:
    normalized = path.replace(".", "_").lower()
    return [token for token in TOKEN_SPLIT_PATTERN.split(normalized) if token]


def _entity_group(token: str) -> str:
    normalized = token.lower()
    for group_name, aliases in ENTITY_ALIAS_GROUPS.items():
        if normalized in aliases:
            return group_name
    return normalized


def _score_candidate(placeholder: str, payload_path: str) -> float:
    if placeholder == payload_path:
        return 0.99

    placeholder_segments = placeholder.split(".")
    payload_segments = payload_path.split(".")
    placeholder_tokens = set(_path_tokens(placeholder))
    payload_tokens = set(_path_tokens(payload_path))

    if not placeholder_tokens or not payload_tokens:
        return 0.0

    score = 0.0
    overlap = len(placeholder_tokens & payload_tokens) / float(len(placeholder_tokens | payload_tokens))
    score += overlap * 0.45

    if placeholder_segments[-1] == payload_segments[-1]:
        score += 0.25

    if _entity_group(placeholder_segments[0]) == _entity_group(payload_segments[0]):
        score += 0.2

    if placeholder_segments[-1] == "name" and payload_segments[-1] in {"name", "legal_name", "full_name"}:
        score += 0.15
    if placeholder_segments[-1] == "date" and payload_segments[-1] in {"date", "execution_date", "effective_date"}:
        score += 0.15
    if placeholder_segments[-1].endswith("amount") and payload_segments[-1] in {"amount", "total_amount", "stipend_amount"}:
        score += 0.15

    return min(score, 0.95)


def build_candidate_shortlist(
    placeholders: List[str],
    payload_paths: List[PayloadPathInfo],
    limit: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    shortlist: Dict[str, List[Dict[str, Any]]] = {}

    for placeholder in placeholders:
        ranked = sorted(
            (
                {
                    "path": payload_path.path,
                    "example_value": payload_path.value,
                    "value_type": payload_path.value_type,
                    "heuristic_score": round(_score_candidate(placeholder, payload_path.path), 4),
                }
                for payload_path in payload_paths
            ),
            key=lambda item: item["heuristic_score"],
            reverse=True,
        )
        shortlist[placeholder] = [item for item in ranked[:limit] if item["heuristic_score"] > 0.0]

    return shortlist


def heuristic_suggestions(
    placeholders: List[str],
    payload_paths: List[PayloadPathInfo],
    extracted_values: Dict[str, str],
    allow_static_value_suggestions: bool,
) -> List[MappingSuggestion]:
    shortlist = build_candidate_shortlist(placeholders, payload_paths)
    suggestions: List[MappingSuggestion] = []

    for placeholder in placeholders:
        candidates = shortlist.get(placeholder, [])
        if candidates and candidates[0]["heuristic_score"] >= 0.65:
            best = candidates[0]
            suggestions.append(
                MappingSuggestion(
                    placeholder_name=placeholder,
                    source_type="payload_path",
                    source_path=best["path"],
                    entity_type=best["path"].split(".")[0],
                    confidence=min(0.85, best["heuristic_score"]),
                    notes="Deterministic placeholder-to-payload path match.",
                )
            )
            continue

        extracted_value = extracted_values.get(placeholder)
        if allow_static_value_suggestions and extracted_value:
            suggestions.append(
                MappingSuggestion(
                    placeholder_name=placeholder,
                    source_type="static_value",
                    static_value=extracted_value,
                    entity_type=placeholder.split(".")[0],
                    confidence=0.4,
                    notes="Fallback static-value suggestion derived from extracted source text.",
                )
            )
            continue

        suggestions.append(
            MappingSuggestion(
                placeholder_name=placeholder,
                source_type="unresolved",
                confidence=0.0,
                notes="No reliable deterministic mapping candidate was found.",
            )
        )

    return suggestions


class TemplateMappingSuggester:
    def __init__(self, api_key: Optional[str], model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        if not self._api_key:
            return None
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise MappingSuggestionError(
                "Gemini SDK is not installed. Install `google-genai` and retry."
            ) from exc

        os.environ.setdefault("GEMINI_API_KEY", self._api_key)
        self._client = genai.Client()
        return self._client

    def _call_llm(
        self,
        document_type: str,
        placeholders: List[str],
        extracted_values: Dict[str, str],
        payload_paths: List[PayloadPathInfo],
        allow_static_value_suggestions: bool,
    ) -> List[MappingSuggestion]:
        client = self._get_client()
        if client is None:
            raise MappingSuggestionError("Gemini API key is not configured for mapping suggestions.")

        candidate_shortlist = build_candidate_shortlist(placeholders, payload_paths)
        request_payload = {
            "document_type": document_type,
            "placeholders": placeholders,
            "extracted_values": extracted_values,
            "payload_paths": [
                {
                    "path": item.path,
                    "example_value": item.value,
                    "value_type": item.value_type,
                }
                for item in payload_paths
            ],
            "candidate_shortlist": candidate_shortlist,
            "allow_static_value_suggestions": allow_static_value_suggestions,
        }

        system_prompt = (
            "You suggest mappings between document template placeholders and transaction payload paths. "
            "Work generically across agreements, letters, contracts, allotments, and transactional documents.\n"
            "Rules:\n"
            "1. Choose source_type `payload_path` when a placeholder should come from transaction payload.\n"
            "2. Choose source_type `static_value` only when the placeholder should stay fixed across transactions; reuse the exact extracted value.\n"
            "3. Choose source_type `unresolved` if there is not enough evidence.\n"
            "4. Never invent payload paths. Use only the provided payload paths.\n"
            "5. Keep confidence conservative.\n"
        )

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=json.dumps(request_payload),
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_json_schema": MappingSuggestionBatch.model_json_schema(),
                },
            )
        except Exception as exc:
            raise MappingSuggestionError(str(exc)) from exc

        response_text = getattr(response, "text", None)
        if not response_text:
            raise MappingSuggestionError("Suggestion model returned no text payload.")

        try:
            parsed = MappingSuggestionBatch.model_validate_json(response_text)
        except ValidationError as exc:
            raise MappingSuggestionError(str(exc)) from exc

        return parsed.suggestions

    def suggest(
        self,
        document_type: str,
        placeholders: List[str],
        extracted_values: Dict[str, str],
        sample_transaction_payload: Dict[str, Any],
        allow_static_value_suggestions: bool,
    ) -> Tuple[List[MappingSuggestion], List[str]]:
        payload_paths = flatten_payload_paths(sample_transaction_payload)
        deterministic = heuristic_suggestions(
            placeholders=placeholders,
            payload_paths=payload_paths,
            extracted_values=extracted_values,
            allow_static_value_suggestions=allow_static_value_suggestions,
        )

        warnings: List[str] = []
        try:
            llm_suggestions = self._call_llm(
                document_type=document_type,
                placeholders=placeholders,
                extracted_values=extracted_values,
                payload_paths=payload_paths,
                allow_static_value_suggestions=allow_static_value_suggestions,
            )
        except MappingSuggestionError as exc:
            warnings.append(f"LLM mapping suggestions were unavailable; deterministic fallback was used. Reason: {exc}")
            return deterministic, warnings

        payload_path_set = {item.path for item in payload_paths}
        deterministic_by_placeholder = {item.placeholder_name: item for item in deterministic}
        extracted_by_placeholder = extracted_values
        llm_by_placeholder = {item.placeholder_name: item for item in llm_suggestions}
        resolved: List[MappingSuggestion] = []

        for placeholder in placeholders:
            candidate = llm_by_placeholder.get(placeholder) or deterministic_by_placeholder[placeholder]

            if candidate.source_type == "payload_path":
                if not candidate.source_path or candidate.source_path not in payload_path_set:
                    candidate = deterministic_by_placeholder[placeholder]
            elif candidate.source_type == "static_value":
                extracted_value = extracted_by_placeholder.get(placeholder)
                if not extracted_value:
                    candidate = deterministic_by_placeholder[placeholder]
                else:
                    candidate.static_value = extracted_value
                    candidate.source_path = ""
            else:
                candidate = deterministic_by_placeholder[placeholder]

            resolved.append(candidate)

        return resolved, warnings
