#!/usr/bin/env python3
"""verify_enhancements.py — tests all enhanced modules"""
import sys
sys.path.insert(0, '/home/kali/Downloads/Qual')

errors = []

# 1. model_router
try:
    from intelligence.model_router import get_model_candidates, _mem_available_gb, _cpu_count
    print(f'[1] model_router OK - RAM: {_mem_available_gb():.1f}GB, CPUs: {_cpu_count()}')
    candidates = get_model_candidates()
    print(f'    Model candidates: {candidates}')
except Exception as e:
    errors.append(f'model_router: {e}')
    print(f'[1] model_router FAIL: {e}')

# 2. query_rewriter
try:
    from intelligence.query_rewriter import _deterministic_expand, extract_search_keywords
    expanded = _deterministic_expand('What does the fed rate hike mean for cpi and pmi?')
    kws = extract_search_keywords('What does Fed rate hike mean for CPI and PMI growth?')
    print(f'[2] query_rewriter OK')
    print(f'    Expanded query (first 100c): {expanded[:100]}')
    print(f'    Keywords: {kws}')
except Exception as e:
    errors.append(f'query_rewriter: {e}')
    print(f'[2] query_rewriter FAIL: {e}')

# 3. response_enhancer
try:
    from intelligence.response_enhancer import enhance_response, score_response
    sample = (
        "Direct answer: Fed raised rates 25bps to 5.5%, pressuring equities.\n"
        "Data snapshot: FedFunds=5.5%, 10Y=4.8%, VIX=22, S&P500=4100\n"
        "Causal chain: Rate hike -> higher discount rates -> PE compression -> equities fall\n"
        "What is happening:\n"
        "- Fed raised rates 25bps at FOMC meeting, 10Y yield rose 18bps [S1]\n"
        "- Higher borrowing costs reduce consumer spending, GDP growth at risk\n"
        "Market impact:\n"
        "- Equities: S&P500 down 1.2%, tech sector -2.1%\n"
        "- Rates/Bonds: 10Y at 4.8%, curve flattened 12bps\n"
        "- FX: DXY strengthened 0.4%, EUR/USD fell to 1.082\n"
        "Scenarios (probabilities must add to 100%):\n"
        "- Base (~55%): S&P holds above 4000\n"
        "- Bull (~25%): Fed pauses, S&P rallies to 4400\n"
        "- Bear (~20%): Recession signals mount, S&P falls to 3700\n"
        "What to watch:\n"
        "- CPI release on March 12, expected 3.1%\n"
        "Confidence: MEDIUM - Good data coverage but mixed signals\n"
    )
    enhanced, report = enhance_response(sample, mode='brief')
    print(f'[3] response_enhancer OK — Quality: {report.quality_score}/100')
    print(f'    Missing sections: {report.missing_sections}')
    print(f'    Vague bullets: {report.vague_bullets_count}')
    print(f'    Citations found: {report.citations_count}')
    print(f'    Scenario prob sum: {report.scenario_prob_sum}%')
    if report.warnings:
        print(f'    Warnings: {report.warnings}')
except Exception as e:
    errors.append(f'response_enhancer: {e}')
    print(f'[3] response_enhancer FAIL: {e}')
    import traceback; traceback.print_exc()

# 4. news_health_checker
try:
    from intelligence.news_health_checker import (
        check_news_health, check_news_health_quick,
        FeedStatus, NewsHealthReport, _check_index_status, _check_ollama
    )
    idx_status, idx_age, idx_count = _check_index_status()
    ollama_ok, models = _check_ollama()
    print(f'[4] news_health_checker OK')
    print(f'    Vector index: {idx_status}, age={idx_age}h, vectors={idx_count:,}')
    print(f'    Ollama: {"reachable" if ollama_ok else "unreachable"}, models={models}')
except Exception as e:
    errors.append(f'news_health_checker: {e}')
    print(f'[4] news_health_checker FAIL: {e}')
    import traceback; traceback.print_exc()

# 5. context_retriever (just import check — no FAISS call)
try:
    from intelligence.context_retriever import (
        _recency_score, _title_score, _relevance_score, retrieve_relevant_context
    )
    # Basic scoring test with a dummy item
    dummy_item = {
        'text': 'Federal Reserve rate hike 25bps inflation CPI',
        'metadata': {
            'title': 'Fed raises rates again amid inflation fears',
            'entities': ['Federal Reserve', 'inflation'],
            'extracted_at': '2026-03-08T10:00:00+00:00',
        }
    }
    r = _relevance_score('fed rate hike inflation', dummy_item)
    rec = _recency_score(dummy_item)
    print(f'[5] context_retriever OK — relevance={r:.2f}, recency={rec:.1f}')
except Exception as e:
    errors.append(f'context_retriever: {e}')
    print(f'[5] context_retriever FAIL: {e}')
    import traceback; traceback.print_exc()

# 6. macro_engine import check
try:
    from intelligence.macro_engine import macro_intelligence_pipeline, get_last_model_used
    print(f'[6] macro_engine OK')
except Exception as e:
    errors.append(f'macro_engine: {e}')
    print(f'[6] macro_engine FAIL: {e}')
    import traceback; traceback.print_exc()

# 7. API import check
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_check", "/home/kali/Downloads/Qual/api/app.py")
    # Just parse — don't exec (FastAPI would need server)
    import ast
    with open('/home/kali/Downloads/Qual/api/app.py') as f:
        src = f.read()
    ast.parse(src)
    print(f'[7] api/app.py syntax OK')
except Exception as e:
    errors.append(f'api/app.py: {e}')
    print(f'[7] api/app.py FAIL: {e}')

print()
if errors:
    print(f'FAILURES ({len(errors)}):')
    for err in errors:
        print(f'  ✗ {err}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')
