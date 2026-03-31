# Migration Guide: Response Contract v2

This guide outlines how to move from legacy response building (string manipulation) to the new `StructuredResponse` system.

## Key Changes
- **Builders**: Use `BriefResponseBuilder` or `DetailedResponseBuilder` to construct responses.
- **Centralized Prompts**: All prompt blocks (Mechanics, COT, Strict Rules) are now in `intelligence/prompt_templates.py`.
- **Validation**: Every response now includes a `quality_score` and `_response_contract` metadata block in the API.

## 1. Updating Prompt Building
Before:
```python
prompt = f"Use this format: ... {mechanics_text} ... Question: {q}"
```

After:
```python
from intelligence.prompt_templates import (
    get_response_format_block,
    FINANCIAL_MECHANICS_BLOCK,
    COT_REASONING_BLOCK,
    STRICT_RULES_BLOCK,
)

format_block = get_response_format_block("brief")
prompt = f"{FINANCIAL_MECHANICS_BLOCK}\n{COT_REASONING_BLOCK}\n{STRICT_RULES_BLOCK}\n{format_block}\n--- Question: {q}"
```

## 2. Using the Response Builders
Before (String Formatting):
```python
response_text = f"Direct answer: {answer}\nMarket impact: {impact_bullets}"
```

After (Structured):
```python
from intelligence.response_builder import BriefResponseBuilder

builder = BriefResponseBuilder()
builder.direct_answer("S&P 500 rose 1.2% after softer-than-expected CPI.")
builder.market_impact(["Regional banks (KRE) rallied on lower yields.", "Nasdaq outperformed growth names."])
# ... more sections ...
response_text = builder.build().to_text()
```

## 3. Integrating with the API Middleware
The `normalize_api_payload` function in `intelligence/response_middleware.py` handles the non-breaking transition for existing endpoints. It will:
1. Parse the text response into a schema object.
2. Validate the schema and compute a quality score.
3. Attach the metadata to the JSON response under `_response_contract`.

## 4. Normalizing External LLM Output
If you receive raw text from an external LLM, use the `response_normalizer.py` to convert it back to a structured object for validation:
```python
from intelligence.response_normalizer import normalize_text_response
from intelligence.response_validator import validate_structured_response

structured = normalize_text_response(llm_raw_text, mode="brief")
report = validate_structured_response(structured)
print(f"Is valid? {report.ok}. Score: {structured.metadata.quality_score}")
```
