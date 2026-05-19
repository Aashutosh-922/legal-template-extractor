import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from app.schemas import TemplateFieldMapping

PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*}}")


@dataclass
class GenerationPreparationResult:
    render_context: Dict[str, Any]
    mapped_placeholders: List[str]
    unmapped_placeholders: List[str]
    missing_payload_fields: List[str]


def extract_placeholders(template_text: str) -> List[str]:
    return sorted({match.group(1) for match in PLACEHOLDER_PATTERN.finditer(template_text)})


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _resolve_source_path(payload: Any, source_path: str) -> Any:
    current = payload
    for segment in source_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(source_path)
            current = current[segment]
            continue

        if isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                raise KeyError(source_path)
            current = current[index]
            continue

        raise KeyError(source_path)
    return current


def _set_target_path(target: Dict[str, Any], target_path: str, value: Any) -> None:
    current = target
    segments = [segment for segment in target_path.split(".") if segment]
    if not segments:
        return

    for segment in segments[:-1]:
        existing = current.get(segment)
        if existing is None:
            current[segment] = {}
            existing = current[segment]
        if not isinstance(existing, dict):
            raise ValueError(f"Placeholder path collides with scalar value: {target_path}")
        current = existing

    current[segments[-1]] = value


def prepare_generation_context(
    placeholders: Iterable[str],
    mappings: List[TemplateFieldMapping],
    transaction_payload: Dict[str, Any],
) -> GenerationPreparationResult:
    placeholder_list = sorted(set(placeholders))
    mapping_by_placeholder = {mapping.placeholder_name: mapping for mapping in mappings}
    unmapped_placeholders = [placeholder for placeholder in placeholder_list if placeholder not in mapping_by_placeholder]

    render_context: Dict[str, Any] = {}
    mapped_placeholders: List[str] = []
    missing_payload_fields: List[str] = []

    for placeholder in placeholder_list:
        mapping = mapping_by_placeholder.get(placeholder)
        if mapping is None:
            continue

        if mapping.source_type == "static_value":
            value = mapping.static_value
            if _is_missing_value(value):
                missing_payload_fields.append(f"{placeholder} -> static_value")
                continue
        else:
            try:
                value = _resolve_source_path(transaction_payload, mapping.source_path or "")
            except KeyError:
                missing_payload_fields.append(f"{placeholder} -> {mapping.source_path}")
                continue

            if _is_missing_value(value):
                missing_payload_fields.append(f"{placeholder} -> {mapping.source_path}")
                continue

        _set_target_path(render_context, placeholder, value)
        mapped_placeholders.append(placeholder)

    return GenerationPreparationResult(
        render_context=render_context,
        mapped_placeholders=mapped_placeholders,
        unmapped_placeholders=unmapped_placeholders,
        missing_payload_fields=sorted(set(missing_payload_fields)),
    )
