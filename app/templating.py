from jinja2 import Environment, StrictUndefined, TemplateError


class TemplateRenderingError(Exception):
    """Raised when template rendering fails."""


def render_template_text(template_text: str, payload: dict) -> str:
    try:
        environment = Environment(undefined=StrictUndefined, autoescape=False)
        template = environment.from_string(template_text)
        return template.render(**payload)
    except TemplateError as exc:
        raise TemplateRenderingError(str(exc)) from exc


def render_text_to_pdf_bytes(rendered_text: str) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise TemplateRenderingError("PDF rendering requires fpdf2.") from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    transliteration_map = str.maketrans(
        {
            "\u2013": "-",
            "\u2014": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\u00a0": " ",
        }
    )

    def _sanitize_pdf_text(value: str) -> str:
        # Core fonts in fpdf2 are latin-1 only. Normalize common smart punctuation
        # and replace unsupported codepoints so PDF generation never crashes.
        normalized = value.translate(transliteration_map)
        return normalized.encode("latin-1", errors="replace").decode("latin-1")

    lines = rendered_text.splitlines() or [rendered_text]
    for line in lines:
        sanitized_line = _sanitize_pdf_text(line if line else " ")
        pdf.multi_cell(0, 6, sanitized_line)

    output = pdf.output()
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)
