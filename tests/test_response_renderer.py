"""
test_response_renderer.py
--------------------------
Unit tests for intelligence/response_renderer.py
"""

import pytest

from intelligence.response_schema import (
    PredictedEvent,
    ResponseMetadata,
    Scenario,
    StructuredResponse,
)
from intelligence.response_renderer import ResponseRenderer


@pytest.fixture
def sample_response() -> StructuredResponse:
    """Build a sample StructuredResponse for testing."""
    return StructuredResponse(
        direct_answer="Equities face downside pressure from rising yields.",
        data_snapshot="S&P500=4800, VIX=22, 10Y=4.5%, DXY=104",
        causal_chain="CPI above consensus → rate expectations reprice → long-duration assets sell off",
        what_is_happening=[
            "CPI came in at 3.5% vs 3.3% consensus [S1]",
            "Treasury yields spiked 12bps on the release [S2]",
        ],
        market_impact=[
            "Equities: S&P500 -1.2%; tech sectors worst hit as multiples compress.",
            "Rates/Bonds: 10Y yield rises to 4.62%; 2Y jumps 15bps.",
            "FX: DXY strengthens to 105; EUR/USD falls to 1.07.",
        ],
        predicted_events=[
            PredictedEvent(
                label="Rate Pressure",
                horizon="7-30d",
                probability_pct=45,
                narrative="Elevated yields keep pressure on growth equities",
                trigger="10Y > 4.7%",
                invalidation="10Y < 4.3% with soft CPI",
            ),
        ],
        scenarios=[
            Scenario(name="Base", probability_pct=55, narrative="Data in-line → range-bound markets"),
            Scenario(name="Bull", probability_pct=25, narrative="Soft PCE → rate cut pricing → rally"),
            Scenario(name="Bear", probability_pct=20, narrative="Hot data → stagflation fear → selloff"),
        ],
        what_to_watch=[
            "Next PCE print vs consensus",
            "10Y Treasury yield: above 4.7% signals renewed pressure",
        ],
        confidence="MEDIUM - Based on 5 indexed news chunks.",
        executive_summary="CPI surprised to the upside driving a rates-led selloff.",
        key_risks=["Payrolls surprise outside ±0.2% triggers rapid repricing."],
        time_horizons=["24-72h: Watch Fed speeches.", "1-4 weeks: Next CPI print."],
        metadata=ResponseMetadata(
            mode="detailed",
            model_used="groq:llama-3.1-70b",
            quality_score=82,
            citation_count=2,
        ),
    )


class TestRenderPlain:
    def test_matches_to_text(self, sample_response: StructuredResponse):
        """render_plain() should produce identical output to to_text()."""
        plain = ResponseRenderer.render_plain(sample_response)
        assert plain == sample_response.to_text()

    def test_contains_all_sections(self, sample_response: StructuredResponse):
        plain = ResponseRenderer.render_plain(sample_response)
        assert "Direct answer:" in plain
        assert "Data snapshot:" in plain
        assert "Causal chain:" in plain
        assert "What is happening:" in plain
        assert "Market impact:" in plain
        assert "Predicted events:" in plain
        assert "Scenarios" in plain
        assert "What to watch:" in plain
        assert "Confidence:" in plain


class TestRenderMarkdown:
    def test_has_bold_headers(self, sample_response: StructuredResponse):
        md = ResponseRenderer.render_markdown(sample_response)
        assert "## 🎯 Direct Answer" in md
        assert "## 📊 Data Snapshot" in md
        assert "## 📈 Market Impact" in md

    def test_has_scenario_table(self, sample_response: StructuredResponse):
        md = ResponseRenderer.render_markdown(sample_response)
        assert "| Scenario | Prob | Narrative | Trigger | Invalidation |" in md
        assert "**Base**" in md
        assert "~55%" in md

    def test_has_predicted_event_table(self, sample_response: StructuredResponse):
        md = ResponseRenderer.render_markdown(sample_response)
        assert "| Event | Horizon | Probability | Trigger | Invalidation |" in md
        assert "Rate Pressure" in md

    def test_has_executive_summary(self, sample_response: StructuredResponse):
        md = ResponseRenderer.render_markdown(sample_response)
        assert "## 📋 Executive Summary" in md

    def test_confidence_is_bold(self, sample_response: StructuredResponse):
        md = ResponseRenderer.render_markdown(sample_response)
        assert "**Confidence:**" in md


class TestRenderJson:
    def test_structure(self, sample_response: StructuredResponse):
        data = ResponseRenderer.render_json(sample_response)
        assert isinstance(data, dict)
        assert data["direct_answer"] == sample_response.direct_answer
        assert len(data["scenarios"]) == 3
        assert data["scenarios"][0]["name"] == "Base"
        assert data["predicted_events"][0]["label"] == "Rate Pressure"

    def test_optional_sections_present(self, sample_response: StructuredResponse):
        data = ResponseRenderer.render_json(sample_response)
        assert "key_risks" in data
        assert "time_horizons" in data
        assert "executive_summary" in data

    def test_optional_sections_absent(self):
        """Sections like key_risks should not appear if empty."""
        resp = StructuredResponse(
            direct_answer="Test",
            data_snapshot="Test",
            causal_chain="Test",
        )
        data = ResponseRenderer.render_json(resp)
        assert "key_risks" not in data
        assert "time_horizons" not in data


class TestRenderApi:
    def test_includes_metadata(self, sample_response: StructuredResponse):
        payload = ResponseRenderer.render_api(
            sample_response,
            model_used="groq:llama-3.1-70b",
            quality_score=82,
        )
        assert "metadata" in payload
        assert payload["metadata"]["model_used"] == "groq:llama-3.1-70b"
        assert payload["metadata"]["quality_score"] == 82
        assert payload["metadata"]["mode"] == "detailed"

    def test_includes_all_content(self, sample_response: StructuredResponse):
        payload = ResponseRenderer.render_api(sample_response)
        assert "direct_answer" in payload
        assert "scenarios" in payload
        assert "predicted_events" in payload
