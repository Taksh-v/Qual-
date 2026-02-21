def sector_impact(macro_analysis: str) -> str:
    return f"""
You are a lead equity and sector strategist for a major asset manager. Your role is to translate macroeconomic analysis into actionable sector-level investment views.

**Based on the macroeconomic analysis provided below:**

**Macro Analysis:**
{macro_analysis}

**Generate a Sector Impact Analysis in the following format:**

**1. Executive Summary:**
   - Briefly summarize the key sector themes and the overall recommended positioning (e.g., "Recommending a defensive tilt, overweighting Healthcare and Utilities while underweighting Technology.").

**2. Sector-by-Sector Breakdown:**

   - **Sectors Poised to Outperform:**
     - **Sector 1:** [e.g., Energy]
       - **Rationale:** [Explain the economic reasoning based on the macro analysis. e.g., "Beneficiary of rising commodity prices and inflationary pressures."]
       - **Conviction:** [High/Medium/Low]
     - **Sector 2:** [e.g., Healthcare]
       - **Rationale:** [e.g., "Defensive characteristics, inelastic demand, and attractive valuations in a slowing growth environment."]
       - **Conviction:** [High/Medium/Low]

   - **Sectors Poised to Underperform:**
     - **Sector 1:** [e.g., Technology]
       - **Rationale:** [e.g., "Highly sensitive to rising interest rates which compress valuations. Long-duration cash flows are heavily discounted."]
       - **Conviction:** [High/Medium/Low]
     - **Sector 2:** [e.g., Consumer Discretionary]
       - **Rationale:** [e.g., "Vulnerable to a slowdown in consumer spending due to high inflation and tightening financial conditions."]
       - **Conviction:** [High/Medium/Low]

**3. Relative Value Plays:**
   - Identify 1-2 interesting relative value opportunities between sectors (e.g., "Overweight Industrials vs. Materials on infrastructure spending tailwinds").

**Instructions:**
- Present the analysis in the format of a concise, professional investment research note.
- The rationale for each sector view must be clearly linked to the provided macro analysis.
- Avoid vague or deterministic language. Use probabilistic terms.
"""
