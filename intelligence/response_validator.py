from __future__ import annotations

from dataclasses import dataclass, field

from intelligence.response_schema import StructuredResponse
from intelligence.writing_style import contains_vague_phrase


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        return [issue.message for issue in self.issues]


def validate_structured_response(resp: StructuredResponse) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not resp.direct_answer.strip():
        issues.append(ValidationIssue(code="missing_direct_answer", message="Direct answer is missing.", severity="error"))
    if not resp.data_snapshot.strip():
        issues.append(ValidationIssue(code="missing_data_snapshot", message="Data snapshot is missing.", severity="error"))
        
    if not resp.regime:
        issues.append(ValidationIssue(code="missing_regime", message="Regime classification is missing. Defaults to Transitional.", severity="warning"))
        
    if not resp.causal_architecture and not resp.causal_chain:
        issues.append(ValidationIssue(code="missing_causal_architecture", message="Causal architecture is missing.", severity="error"))

    if not resp.cross_asset_impacts and not resp.market_impact:
        issues.append(ValidationIssue(code="missing_market_impact", message="Cross-asset/Market impact section has no bullets.", severity="error"))
    
    if resp.metadata.mode == "detailed":
        if not resp.positioning:
            issues.append(ValidationIssue(code="missing_positioning", message="Positioning ideas are missing in detailed mode.", severity="warning"))
        if len(resp.cross_asset_impacts) < 3 and len(resp.market_impact) < 3:
            issues.append(ValidationIssue(code="insufficient_cross_asset", message="Fewer than 3 cross-asset impacts provided in detailed mode.", severity="warning"))

    scenario_sum = sum(x.probability_pct or 0 for x in resp.scenarios)
    if resp.scenarios and not (95 <= scenario_sum <= 105):
        issues.append(
            ValidationIssue(
                code="scenario_probability_sum",
                message=f"Scenario probabilities sum to {scenario_sum}% (expected ~100%).",
                severity="warning",
            )
        )

    legacy_impact = resp.market_impact + resp.what_is_happening + resp.what_to_watch
    for line in legacy_impact:
        if contains_vague_phrase(line):
            issues.append(
                ValidationIssue(
                    code="vague_phrase",
                    message=f"Vague wording detected: '{line[:80]}'",
                    severity="warning",
                )
            )
            
    # Also check new text fields for vagueness
    for ca in resp.cross_asset_impacts:
        if contains_vague_phrase(ca.mechanism):
            issues.append(ValidationIssue(code="vague_phrase_mechanism", message=f"Vague mechanism: '{ca.mechanism[:80]}'", severity="warning"))

    for event in resp.predicted_events:
        if not event.trigger or event.trigger == "N/A":
            issues.append(ValidationIssue(code="event_missing_trigger", message=f"{event.label} is missing trigger."))
        if not event.invalidation or event.invalidation == "N/A":
            issues.append(ValidationIssue(code="event_missing_invalidation", message=f"{event.label} is missing invalidation."))

    has_error = any(i.severity == "error" for i in issues)
    return ValidationReport(ok=not has_error, issues=issues)
