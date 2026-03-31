from __future__ import annotations

from intelligence.response_schema import (
    CausalArchitecture,
    CausalChainLink,
    CrossAssetImpact,
    KeyLevel,
    PositioningIdea,
    PredictedEvent,
    ResponseMetadata,
    Scenario,
    StructuredResponse,
)
from intelligence.response_validator import ValidationReport, validate_structured_response
from intelligence.writing_style import normalize_section_lines, normalize_whitespace


class BaseResponseBuilder:
    def __init__(self, mode: str = "brief") -> None:
        self.mode = mode
        self._response = StructuredResponse(
            direct_answer="",
            data_snapshot="",
            causal_chain="",
            metadata=ResponseMetadata(mode=mode),
        )

    def direct_answer(self, text: str) -> "BaseResponseBuilder":
        self._response.direct_answer = normalize_whitespace(text)
        return self

    def data_snapshot(self, text: str) -> "BaseResponseBuilder":
        self._response.data_snapshot = normalize_whitespace(text)
        return self

    def causal_chain(self, text: str) -> "BaseResponseBuilder":
        self._response.causal_chain = normalize_whitespace(text)
        return self

    def causal_architecture(self, primary: list[tuple[str, str]], secondary: list[tuple[str, str]], feedback: str) -> "BaseResponseBuilder":
        self._response.causal_architecture = CausalArchitecture(
            primary=[CausalChainLink(t, e) for t, e in primary],
            secondary=[CausalChainLink(t, e) for t, e in secondary],
            feedback_loop=normalize_whitespace(feedback),
        )
        return self

    def happening(self, bullets: list[str]) -> "BaseResponseBuilder":
        self._response.what_is_happening = normalize_section_lines(bullets)
        return self

    def market_impact(self, bullets: list[str]) -> "BaseResponseBuilder":
        self._response.market_impact = normalize_section_lines(bullets)
        return self

    def cross_asset_impact(self, asset_class: str, direction: str, mechanism: str) -> "BaseResponseBuilder":
        self._response.cross_asset_impacts.append(
            CrossAssetImpact(asset_class, normalize_whitespace(direction), normalize_whitespace(mechanism))
        )
        return self

    def predicted_event(
        self,
        label: str,
        horizon: str,
        probability_pct: int | None,
        narrative: str,
        trigger: str,
        invalidation: str,
    ) -> "BaseResponseBuilder":
        self._response.predicted_events.append(
            PredictedEvent(
                label=label,
                horizon=horizon,
                probability_pct=probability_pct,
                narrative=normalize_whitespace(narrative),
                trigger=normalize_whitespace(trigger),
                invalidation=normalize_whitespace(invalidation),
            )
        )
        return self

    def scenario(self, name: str, probability_pct: int | None, narrative: str) -> "BaseResponseBuilder":
        self._response.scenarios.append(
            Scenario(name=name, probability_pct=probability_pct, narrative=normalize_whitespace(narrative))
        )
        return self

    def watch(self, bullets: list[str]) -> "BaseResponseBuilder":
        self._response.what_to_watch = normalize_section_lines(bullets)
        return self

    def confidence(self, text: str) -> "BaseResponseBuilder":
        self._response.confidence = normalize_whitespace(text)
        return self

    def executive_summary(self, text: str) -> "BaseResponseBuilder":
        self._response.executive_summary = normalize_whitespace(text)
        return self

    def key_risks(self, bullets: list[str]) -> "BaseResponseBuilder":
        self._response.key_risks = normalize_section_lines(bullets)
        return self

    def time_horizons(self, bullets: list[str]) -> "BaseResponseBuilder":
        self._response.time_horizons = normalize_section_lines(bullets)
        return self

    def regime(self, text: str) -> "BaseResponseBuilder":
        self._response.regime = normalize_whitespace(text)
        return self

    def dominant_theme(self, text: str) -> "BaseResponseBuilder":
        self._response.dominant_theme = normalize_whitespace(text)
        return self

    def positioning_idea(self, action: str, instruments: str, rationale: str) -> "BaseResponseBuilder":
        self._response.positioning.append(
            PositioningIdea(normalize_whitespace(action), normalize_whitespace(instruments), normalize_whitespace(rationale))
        )
        return self
        
    def key_level(self, instrument: str, level: str, significance: str) -> "BaseResponseBuilder":
        self._response.key_levels.append(
            KeyLevel(normalize_whitespace(instrument), normalize_whitespace(level), normalize_whitespace(significance))
        )
        return self
        
    def historical_analog(self, text: str) -> "BaseResponseBuilder":
        self._response.historical_analog = normalize_whitespace(text)
        return self
        
    def data_gaps(self, gaps: list[str]) -> "BaseResponseBuilder":
        self._response.data_gaps = normalize_section_lines(gaps)
        return self

    def validate(self) -> ValidationReport:
        return validate_structured_response(self._response)

    def build(self) -> StructuredResponse:
        return self._response


class BriefResponseBuilder(BaseResponseBuilder):
    def __init__(self) -> None:
        super().__init__(mode="brief")


class DetailedResponseBuilder(BaseResponseBuilder):
    def __init__(self) -> None:
        super().__init__(mode="detailed")


class HedgeResponseBuilder(BaseResponseBuilder):
    def __init__(self) -> None:
        super().__init__(mode="hedge")
