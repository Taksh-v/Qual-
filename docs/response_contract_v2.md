# Response Contract v2

This document defines the normalized writing and response format contract introduced in the renovation.

## Goals
- Keep current API response shapes backward compatible.
- Add a normalized contract metadata block for consistency checks.
- Centralize schema, prompt format, validation, and normalization logic.

## Backward compatibility
- Existing keys such as `question`, `answer`, and `sources` are preserved.
- New contract metadata is appended under `_response_contract`.
- No endpoint path or request body changes are required.

## Contract metadata
Every normalized payload can include:

```json
{
	"_response_contract": {
		"schema_version": "v2",
		"mode": "brief",
		"validation_ok": true,
		"validation_warnings": []
	}
}
```

## Canonical response sections
The writing contract targets these sections:
- `Direct answer`
- `Data snapshot`
- `Causal chain`
- `What is happening`
- `Market impact`
- `Predicted events`
- `Scenarios`
- `What to watch`
- `Confidence`

Detailed mode additionally supports:
- `Executive summary`
- `Key risks`
- `Time horizons`

## Implementation modules
- `intelligence/response_schema.py`: dataclasses for structured response.
- `intelligence/response_builder.py`: mode-specific builders.
- `intelligence/prompt_templates.py`: centralized format instructions.
- `intelligence/response_normalizer.py`: text-to-schema parser.
- `intelligence/response_validator.py`: structural quality checks.
- `intelligence/response_middleware.py`: non-breaking payload normalization.
- `intelligence/response_contract.py`: facade helpers for integration.

## Validation rules (high level)
- Mandatory core fields: direct answer, data snapshot, market impact.
- Scenario probabilities should sum to approximately 100%.
- Predicted events should include trigger and invalidation.
- Warnings are attached in metadata, without breaking existing payloads.

