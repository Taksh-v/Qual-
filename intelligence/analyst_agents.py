import os
import json
import logging
import asyncio
from typing import Any
from intelligence.query_rewriter import rewrite_query
from intelligence.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)

def _local_ask_llm(prompt: str, system_prompt: str) -> str:
    from intelligence.model_router import get_model_candidates
    try:
        import requests
        OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        OLLAMA_GENERATE_TIMEOUT_SEC = int(os.getenv("OLLAMA_GENERATE_TIMEOUT_SEC", "120"))
    except ImportError:
        pass
        
    last_error = None
    for model in get_model_candidates():
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_GENERATE_TIMEOUT_SEC,
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            if text:
                return text.strip()
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"LLM generation failed: {last_error}")

async def run_fundamental_analyst(question: str, macro_context: list[dict], fundamentals_context: list[dict], sector_context: list[dict]) -> str:
    """
    Agent 1: The Quantitative/Fundamental Analyst.
    Only looks at raw numbers: Macroeconomics, P/E Ratios, Balance Sheets, and Sector ETF flow.
    """
    if not (macro_context or fundamentals_context or sector_context):
        return "No strictly quantitative data found for this query."
        
    system_prompt = (
        "You are an elite quantitative hedge fund analyst. Your only job is to evaluate strictly numerical data.\n"
        "Review the provided Macroeconomic, Fundamental (Valuation/Balance Sheet), and Sector ETF data.\n"
        "Write a concise, highly analytical brief answering the user's question based ONLY on these numbers. "
        "Do not guess. If metrics are missing, state so."
    )
    
    combined_context = "\n\n".join([
        *[c.get("text", "") for c in macro_context],
        *[c.get("text", "") for c in fundamentals_context],
        *[c.get("text", "") for c in sector_context]
    ])
    
    prompt = f"USER QUESTION: {question}\n\nDATA:\n{combined_context}"
    
    try:
        response = _local_ask_llm(prompt, system_prompt=system_prompt)
        return response
    except Exception as e:
        logger.error(f"Fundamental Analyst failed: {e}")
        return "Fundamental Analysis Encountered an Error."

async def run_sentiment_analyst(question: str, news_context: list[dict], sec_context: list[dict], insider_context: list[dict], transcript_context: list[dict]) -> str:
    """
    Agent 2: The Qualitative/Sentiment Analyst.
    Only looks at text sentiment: News, SEC 8-K Filings, Insider Trades, and Earnings Call Executive Q&A.
    """
    if not (news_context or sec_context or insider_context or transcript_context):
        return "No explicitly qualitative or sentiment data found for this query."
        
    system_prompt = (
        "You are an elite qualitative hedge fund analyst. Your job is to read between the lines.\n"
        "Evaluate the provided SEC 8-K filings, News articles, SEC Form 4 Insider Trades, and Earnings Call Q&A Transcripts.\n"
        "Write a concise brief identifying the current executive sentiment and news-cycle momentum regarding the user's query.\n"
        "Highlight contradictions (e.g., CEO says things are great in the News, but Form 4 shows them selling stock)."
    )
    
    combined_context = "\n\n".join([
        *[c.get("text", "") for c in news_context],
        *[c.get("text", "") for c in sec_context],
        *[c.get("text", "") for c in insider_context],
        *[c.get("text", "") for c in transcript_context]
    ])
    
    prompt = f"USER QUESTION: {question}\n\nDATA:\n{combined_context}"
    
    try:
        response = _local_ask_llm(prompt, system_prompt=system_prompt)
        return response
    except Exception as e:
        logger.error(f"Sentiment Analyst failed: {e}")
        return "Sentiment Analysis Encountered an Error."

async def run_portfolio_manager(question: str, fundamental_brief: str, sentiment_brief: str) -> str:
    """
    Agent 3: The Portfolio Manager (The Synthesizer).
    Takes the reports from both subordinate analysts and writes the final cohesive report for the UI.
    """
    system_prompt = (
        "You are the Head Portfolio Manager at a top-tier quantitative hedge fund. You are directly answering the client.\n"
        "You have received two independent briefs from your elite analysts:\n"
        "1. The Quantitative Report (Hard Numbers/Macro/Fundamentals)\n"
        "2. The Sentiment Report (News/SEC Filings/Insider Trading/Earnings Calls)\n\n"
        "Synthesize these two briefs into a single, masterful, cohesive answer to the client's question. "
        "Format the answer cleanly with markdown. Be authoritative and precise. Do NOT mention 'Agent 1' or 'Agent 2' explicitly; "
        "just present the synthesized intelligence."
    )
    
    prompt = (
        f"CLIENT QUESTION: {question}\n\n"
        f"--- QUANTITATIVE REPORT ---\n{fundamental_brief}\n\n"
        f"--- SENTIMENT REPORT ---\n{sentiment_brief}"
    )
    
    try:
        response = _local_ask_llm(prompt, system_prompt=system_prompt)
        return response
    except Exception as e:
        logger.error(f"Portfolio Manager failed: {e}")
        return "Portfolio Manager Synthesis Failed."
