"""
tests/test_bloomberg_formatter.py
----------------------------------
Unit tests for the BloombergFormatter output format validation.

Tests are pure-Python, no external dependencies or LLM calls.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intelligence.bloomberg_formatter import BloombergFormatter


SAMPLE_ANSWER = """Executive summary: US equities face headwinds as the Fed holds rates at 5.5%, with VIX at 22 signaling elevated uncertainty [S1].
Direct answer: Equity markets will likely face short-term pressure due to the sustained high rate environment, particularly in rate-sensitive sectors [S1][S2].
Data snapshot: S&P500=5200, VIX=22, yield_10y=4.5%, fed_funds_rate=5.5%, dxy=105 [S2].
Causal chain: Fed holds rates → borrowing costs stay elevated → consumer spending slows → earnings revise lower → equities reprice.
What is happening:
- The Federal Reserve maintained its federal funds rate at 5.25-5.5% at the September FOMC meeting [S1].
- Higher-for-longer rates are compressing equity multiples and slowing the housing market [S2].
Market impact:
- Equities: Bearish, particularly growth and real estate; defensives outperform [S1].
- Rates/Bonds: 10Y yields anchored near 4.5%; curve remains inverted [S2].
- FX: Dollar (DXY) at 105 — dollar strength pressuring EM equities [S3].
- Commodities: Gold bid as hedge; WTI range-bound near $80 [S1].
Scenarios (must sum to 100%):
- Base (~55%): Fed holds through Q1 2025; equities drift lower by 5-8%.
- Bull (~25%): CPI drops below 2.5% triggering rate cut expectations; S&P rallies to 5500.
- Bear (~20%): Inflation re-accelerates; Fed hikes again; S&P drops to 4800.
What to watch:
- November CPI release (Nov 13): critical for rate cut narrative.
- FOMC December meeting: any shift in dot plot.
Confidence: MEDIUM — 8 context chunks, 3 citations, 3/4 agents agree."""

SAMPLE_INDICATORS = {
    "sp500": 5200.0,
    "vix": 22.0,
    "yield_10y": 4.5,
    "yield_2y": 4.8,
    "yield_curve": -30.0,
    "dxy": 105.0,
    "fed_funds_rate": 5.5,
    "oil_wti": 80.0,
    "gold": 2000.0,
    "inflation_cpi": 3.2,
    "credit_hy": 450.0,
}


def _make_formatter():
    return BloombergFormatter()


class TestBloombergFormatterMorningNote:

    def test_morning_note_contains_header(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="What is the rate impact?")
        assert "MACRO AI INTELLIGENCE" in output
        assert "UTC" in output

    def test_morning_note_contains_executive_summary(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="Test?")
        assert "EXECUTIVE SUMMARY" in output

    def test_morning_note_has_live_market_snapshot(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, indicators=SAMPLE_INDICATORS, question="Test?")
        assert "LIVE MARKET SNAPSHOT" in output
        assert "VIX" in output
        assert "US 10Y Yield" in output

    def test_morning_note_has_causal_chain(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="Test?")
        assert "CAUSAL CHAIN" in output
        assert "Fed holds rates" in output

    def test_morning_note_has_scenario_matrix(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="Test?")
        assert "SCENARIO MATRIX" in output
        # Should contain base/bull/bear from parsed answer
        output_lower = output.lower()
        assert any(s in output_lower for s in ["base", "bull", "bear", "55%", "25%", "20%"])

    def test_morning_note_has_risks_section(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="Test?")
        assert "RISKS" in output or "WATCH" in output

    def test_morning_note_has_dividers(self):
        f = _make_formatter()
        output = f.morning_note(answer=SAMPLE_ANSWER, question="Test?")
        assert "━" in output

    def test_morning_note_regime_badge_mapping(self):
        f = _make_formatter()
        regime = {"regime": "RISK_OFF", "confidence": "HIGH"}
        output = f.morning_note(answer=SAMPLE_ANSWER, regime=regime, question="Test?")
        assert "RISK OFF" in output

    def test_morning_note_unknown_regime_graceful(self):
        f = _make_formatter()
        regime = {"regime": "SOMETHING_NEW", "confidence": "LOW"}
        output = f.morning_note(answer=SAMPLE_ANSWER, regime=regime, question="Test?")
        assert "MACRO AI" in output  # Should not crash

    def test_morning_note_empty_answer_graceful(self):
        f = _make_formatter()
        output = f.morning_note(answer="", question="Test?")
        assert "MACRO AI" in output
        assert len(output) > 50


class TestBloombergFormatterRiskMatrix:

    def test_risk_matrix_structure(self):
        f = _make_formatter()
        output = f.risk_matrix(answer=SAMPLE_ANSWER, indicators=SAMPLE_INDICATORS, question="Test?")
        assert "RISK MATRIX" in output
        assert "VIX" in output
        assert "HY Spread" in output
        assert "Yield Curve" in output

    def test_risk_matrix_scenario_breakdown(self):
        f = _make_formatter()
        output = f.risk_matrix(answer=SAMPLE_ANSWER, question="Test?")
        assert "Base Case" in output
        assert "Bull Case" in output
        assert "Bear Case" in output

    def test_risk_matrix_elevated_vix(self):
        f = _make_formatter()
        output = f.risk_matrix(
            answer=SAMPLE_ANSWER,
            indicators={"vix": 35.0},
            question="Test?"
        )
        assert "HIGH" in output

    def test_risk_matrix_low_vix(self):
        f = _make_formatter()
        output = f.risk_matrix(
            answer=SAMPLE_ANSWER,
            indicators={"vix": 12.0},
            question="Test?"
        )
        assert "LOW" in output


class TestBloombergFormatterTradeIdea:

    def test_trade_idea_structure(self):
        f = _make_formatter()
        output = f.trade_idea(answer=SAMPLE_ANSWER, question="Test?")
        assert "TRADE IDEA" in output
        assert "Thesis" in output
        assert "Rationale" in output
        assert "Confidence" in output

    def test_trade_idea_not_empty(self):
        f = _make_formatter()
        output = f.trade_idea(answer=SAMPLE_ANSWER, question="Rate impact on equities?")
        assert len(output) > 100


class TestBloombergFormatterBrief:

    def test_brief_has_header(self):
        f = _make_formatter()
        output = f.brief("This is the analysis.", question="Test?")
        assert "MACRO AI" in output
        assert "This is the analysis." in output

    def test_brief_with_model_used(self):
        f = _make_formatter()
        output = f.brief("Analysis.", model_used="groq:llama-3.3-70b")
        assert "groq:llama-3.3-70b" in output


class TestBloombergFormatterUnifiedEntry:

    def test_format_dispatch_morning_note(self):
        f = _make_formatter()
        output = f.format(mode="morning_note", answer=SAMPLE_ANSWER, question="Test?")
        assert "MACRO AI INTELLIGENCE" in output

    def test_format_dispatch_risk_matrix(self):
        f = _make_formatter()
        output = f.format(mode="risk_matrix", answer=SAMPLE_ANSWER, question="Test?")
        assert "RISK MATRIX" in output

    def test_format_dispatch_trade_idea(self):
        f = _make_formatter()
        output = f.format(mode="trade_idea", answer=SAMPLE_ANSWER, question="Test?")
        assert "TRADE IDEA" in output

    def test_format_dispatch_brief(self):
        f = _make_formatter()
        output = f.format(mode="brief", answer="My brief answer", question="Test?")
        assert "My brief answer" in output

    def test_format_unknown_mode_defaults_to_brief(self):
        f = _make_formatter()
        output = f.format(mode="unknown_mode", answer="Answer text")
        assert "Answer text" in output


class TestAnswerFieldParser:
    """Test the internal _parse_answer_fields method."""

    def test_parse_all_standard_fields(self):
        f = _make_formatter()
        fields = f._parse_answer_fields(SAMPLE_ANSWER)
        assert fields["executive_summary"]
        assert fields["direct_answer"]
        assert fields["causal_chain"]
        assert fields["what_is_happening"]
        assert fields["market_impact"]
        assert fields["confidence"]

    def test_parse_empty_text(self):
        f = _make_formatter()
        fields = f._parse_answer_fields("")
        assert fields["executive_summary"] == ""
        assert fields["direct_answer"] == ""

    def test_parse_bottom_line_alias(self):
        f = _make_formatter()
        fields = f._parse_answer_fields("Bottom line: Rates are rising.\nMarket impact: Equities fall.")
        assert "Rates are rising" in fields["direct_answer"]


class TestIndicatorTable:
    """Test the _build_indicator_table output."""

    def test_indicator_table_shows_known_indicators(self):
        f = _make_formatter()
        table = f._build_indicator_table(SAMPLE_INDICATORS)
        assert "S&P 500" in table
        assert "VIX" in table
        assert "US 10Y Yield" in table
        assert "DXY" in table

    def test_indicator_table_empty(self):
        f = _make_formatter()
        table = f._build_indicator_table({})
        assert "No live market data" in table

    def test_indicator_table_unknown_keys_ignored(self):
        f = _make_formatter()
        table = f._build_indicator_table({"totally_unknown_key": 999.0})
        assert "No live market data" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
