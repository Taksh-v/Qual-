from __future__ import annotations

from typing import Any

from intelligence.response_builder import BriefResponseBuilder, DetailedResponseBuilder, HedgeResponseBuilder
from intelligence.response_middleware import normalize_api_payload
from intelligence.response_normalizer import normalize_text_response
from intelligence.response_schema import StructuredResponse
from intelligence.response_validator import ValidationReport, validate_structured_response


SCHEMA_VERSION = "v2"


def parse_text_to_contract(text: str, mode: str = "brief") -> StructuredResponse:
    return normalize_text_response(text, mode=mode)


def validate_contract(response: StructuredResponse) -> ValidationReport:
    return validate_structured_response(response)


def build_contract_payload(payload: dict[str, Any], mode: str = "brief") -> dict[str, Any]:
    return normalize_api_payload(payload, mode=mode)


def builder_for_mode(mode: str):
    m = (mode or "brief").strip().lower()
    if m == "detailed":
        return DetailedResponseBuilder()
    if m == "hedge":
        return HedgeResponseBuilder()
    return BriefResponseBuilder()

