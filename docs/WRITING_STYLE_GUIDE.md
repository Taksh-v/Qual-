# Writing Style Guide: Macro Intelligence

This guide defines the institutional-quality writing standards for the Macro AI project. These guidelines are enforced by the `intelligence/writing_style.py` module and monitored by the `intelligence/response_enhancer.py` and `intelligence/response_validator.py` layers.

## Core Principles
1. **Precision over Prose**: Avoid filler. Every sentence must provide a data point, a causal link, or a specific asset impact.
2. **Numeric Grounding**: Never use qualitative terms like "high volatility" or "downward pressure" without an accompanying number (e.g., "VIX > 25", "S&P 500 -2.4%").
3. **Causal Transparency**: Use the `[Trigger] → [Transmission] → [Impact]` chain for all reasoning.
4. **Institutional Tone**: Analytical, objective, and concise. Avoid "AI-assistant" conversational filler (e.g., "Certainly!", "I hope this helps").

## Formatting Standards

### 1. Structure
- Use the canonical headers defined in `response_contract_v2.md`.
- Use bullet points (`- `) for lists.
- Avoid nested bullet points.

### 2. Length Constraints
- **Bullets**: Maximum 220 characters. If a point is longer, split it or tighten the phrasing.
- **Paragraphs**: Maximum 480 characters.
- **Direct Answer**: Exactly one concise, data-anchored sentence.

### 3. Citations
- Use `[Sx]` format (e.g., `[S1]`, `[S2]`) to cite news sources.
- Place citations at the end of the relevant sentence, before the period.

### 4. Banned Phrases
Avoid these "vague" phrases unless accompanied by a specific number:
- *Heightened uncertainty*
- *Downward/Upward pressure*
- *Various factors*
- *Further monitoring required*
- *Risk-off sentiment* (use specific asset moves instead)

## Component Guidelines

### Market Impact
Every bullet must contain:
1. A named asset, sector, or instrument (e.g., "Nasdaq 100", "BRENT", "Regional Banks").
2. A direction (e.g., "rally", "compress", "sideways").
3. A reason tied to the current macro regime or data point.

### Predicted Events
Every predicted event must include:
- **Horizon**: (e.g., `7-30d`, `24-72h`)
- **Probability**: (e.g., `45%`, `~60%`)
- **Trigger**: The specific data print or event that activates the scenario.
- **Invalidation**: The specific move that proves the prediction wrong.

### Scenarios
- Must include **Base**, **Bull**, and **Bear** cases.
- Probabilities must sum to **100%**.

## Normalization Process
The system automatically applies these rules:
1. **Whitespace**: Multiple spaces are collapsed.
2. **Sentence Case**: All bullets are forced to start with a capital letter.
3. **Trimming**: Overly long bullets are truncated with an ellipsis (…).
4. **Validation**: Responses that violate these rules receive a lower `quality_score` in the metadata.
