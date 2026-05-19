# AI-Powered Document Template Engine

FastAPI backend for extracting reusable document templates from legal/transaction PDFs, persisting review state, mapping template placeholders to business data, and generating filled text/PDF outputs.

## What It Covers

- PDF upload with OCR fallback
- Multi-stage LLM extraction with strict schema validation and retries
- Human review/approval workflow
- Reusable template creation from approved review sessions
- LLM-assisted mapping suggestions from a sample transaction payload
- Persistent field mappings from template placeholders to transaction payload paths
- Static placeholder values for boilerplate fields that should not vary by transaction
- Generation-time hard failures for unmapped placeholders or missing payload fields
- Docker Compose startup with SQLite persistence and Tesseract installed

## End-to-End Flow

```text
1. Upload PDF
2. Extract text + dynamic fields + Jinja template
3. Persist review session
4. Approve review session
5. Create reusable template from approved review
6. Auto-suggest mappings from a sample transaction payload
7. Review or adjust mappings
8. Submit real transaction payload
9. Generate filled text or PDF
```

## Why `gemini-3-flash-preview`

This service uses `gemini-3-flash-preview` by default because the task benefits more from reliable structured extraction than from open-ended generation.

- It supports Gemini structured outputs cleanly with JSON Schema / Pydantic.
- It is fast enough for staged extraction/retry workflows.
- It has a free tier suitable for development and demos.
- The pipeline compensates for occasional model misses with schema validation, retries, completeness checks, and deterministic fallback template generation.

If accuracy requirements rise further before shipping, the first change would be to benchmark the same pipeline against `gemini-3-pro-preview` on a fixed corpus of contracts and allotment letters.

## Stack

- FastAPI
- Google GenAI SDK with structured outputs
- Pydantic / Pydantic Settings
- PyMuPDF + pytesseract
- Jinja2 + fpdf2
- SQLite
- Docker Compose

## Quick Start

```bash
cp .env.example .env
# set GEMINI_API_KEY
python3.12 -m pip install -r requirements.txt
python3.12 -m uvicorn app.main:app --reload
```

Docs: `http://127.0.0.1:8000/docs`

Local Python for Gemini should be `3.9+`. On this machine, use `python3.12`. If your machine is older, use Docker.

## Docker

```bash
cp .env.example .env
# set GEMINI_API_KEY
docker compose up --build
```

Compose mounts a named volume for `/data/review_sessions.db`, so review sessions and template mappings persist across container restarts.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The current suite covers:

- placeholder extraction and mapping-context assembly
- template creation approval guardrails
- invalid mapping rejection
- generation failures for unmapped or missing fields
- successful text and PDF generation

## Primary API Flow

### 1. Extract from PDF

`POST /extract/pdf`

Returns:

- extracted normalized fields
- generated template text
- completeness warnings
- `review_session_id`

### 2. Approve the Review Session

`POST /review/sessions/{review_session_id}/decision`

```json
{
  "decision": "approved",
  "notes": "Template looks correct."
}
```

### 3. Create a Reusable Template

`POST /templates/from-review/{review_session_id}`

```json
{
  "template_name": "allotment letter v1"
}
```

Returns a `template_id`, placeholders, mapping status, and warnings.

### 4. Auto-Suggest Mappings From a Sample Payload

`POST /templates/{template_id}/suggest-mappings`

```json
{
  "sample_transaction_payload": {
    "buyer": {
      "legal_name": "John Doe"
    },
    "deal": {
      "execution_date": "2026-05-17"
    },
    "property": {
      "registered_address": "221B Baker Street, London"
    }
  },
  "apply_suggestions": true,
  "confidence_threshold": 0.75,
  "allow_static_value_suggestions": true
}
```

This endpoint uses Gemini plus deterministic matching to:

- suggest payload-path mappings
- suggest static values for boilerplate placeholders
- optionally auto-apply high-confidence suggestions

### 5. Save or Adjust Field Mappings Manually

`PUT /templates/{template_id}/mappings`

```json
{
  "mappings": [
    {
      "placeholder_name": "buyer.name",
      "source_path": "buyer.legal_name",
      "entity_type": "buyer"
    },
    {
      "placeholder_name": "agreement.date",
      "source_path": "deal.execution_date",
      "entity_type": "deal"
    },
    {
      "placeholder_name": "property.address",
      "source_path": "property.registered_address",
      "entity_type": "property"
    },
    {
      "placeholder_name": "seller.name",
      "source_type": "static_value",
      "static_value": "Acme Realty Pvt Ltd",
      "entity_type": "seller"
    }
  ]
}
```

The mapping model is intentionally flexible:

- a placeholder can point to any nested path in the submitted transaction payload
- or it can be fixed to a static value if it is issuer boilerplate rather than transaction data

### 6. Generate the Filled Document

`POST /templates/{template_id}/generate/pdf`

```json
{
  "transaction_payload": {
    "buyer": {
      "legal_name": "John Doe"
    },
    "deal": {
      "execution_date": "2026-05-17"
    },
    "property": {
      "registered_address": "221B Baker Street, London"
    }
  }
}
```

If any placeholder is unmapped or any mapped source path is missing/blank, the API returns `422` and explicitly lists the problem fields. Nothing is silently blanked.

## Core Endpoints

- `GET /health`
- `POST /extract/text`
- `POST /extract/pdf`
- `GET /review/sessions/{review_session_id}`
- `POST /review/sessions/{review_session_id}/decision`
- `POST /templates/from-review/{review_session_id}`
- `GET /templates/{template_id}`
- `POST /templates/{template_id}/suggest-mappings`
- `PUT /templates/{template_id}/mappings`
- `POST /templates/{template_id}/generate/text`
- `POST /templates/{template_id}/generate/pdf`
- `POST /render/text`
- `POST /render/pdf`
- `GET /metrics/summary`

## Reliability Notes

- Every LLM stage is Pydantic-validated.
- Invalid or unparseable stage output is retried up to `EXTRACTION_MAX_RETRIES`.
- Template generation has a deterministic local fallback.
- Required-field misses and hallucination risks are surfaced as warnings.
- Generation uses strict Jinja undefined handling and explicit pre-render validation.

## Configuration

Environment variables in `.env.example`:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `EXTRACTION_MAX_RETRIES`
- `MAX_INPUT_CHARACTERS`
- `CHUNK_SIZE_CHARACTERS`
- `CHUNK_OVERLAP_CHARACTERS`
- `MAX_CHUNKS`
- `REQUIRED_FIELDS`
- `ENABLE_OCR_FALLBACK`
- `REVIEW_DB_PATH`


## Postman

Import this collection for the complete API sequence:

- `postman/legal-template-extractor.postman_collection.json`

## Submission Notes

- The codebase is ready for a public GitHub repo.
- A Loom recording still needs to be created separately; that cannot be generated from this environment.
