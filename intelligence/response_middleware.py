from __future__ import annotations

from typing import Any

from intelligence.response_normalizer import normalize_text_response
from intelligence.response_schema import StructuredResponse
from intelligence.response_validator import validate_structured_response


def normalize_api_payload(payload: dict[str, Any], mode: str = "brief") -> dict[str, Any]:
    """
    Non-breaking middleware helper:
    - keeps existing payload shape intact
    - attaches optional structured response metadata under `_response_contract`
    """
    answer = str(payload.get("answer", "") or "")
    structured: StructuredResponse = normalize_text_response(answer, mode=mode)
    report = validate_structured_response(structured)

    metadata = dict(payload.get("_response_contract", {}))
    metadata.update(
        {
            "mode": structured.metadata.mode,
            "validation_ok": report.ok,
            "validation_warnings": report.warnings,
            "schema_version": "v2",
        }
    )

    out = dict(payload)
    out["_response_contract"] = metadata
    return out
