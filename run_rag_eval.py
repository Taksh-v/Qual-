from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from intelligence.data_quality import evaluate_retrieval_quality
from rag.query import (
    embed_query,
    load_index,
    load_metadata,
    retrieve_chunks,
    retrieve_chunks_lexical,
    run_query,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_CONFIG = ROOT / "config" / "rag_eval_queries.json"
DEFAULT_REPORT_PATH = ROOT / "data" / "vector_db" / "rag_eval_report.json"


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(t) > 2}


def _extract_citations(answer: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[S(\d+)\]", answer or "")]


def _parse_generation_diagnostics(answer: str) -> dict[str, Any]:
    mode = "unknown"
    reason = "unknown"
    deterministic_repair = False
    compact_retry = False
    match = re.search(r"Generation diagnostics:\s*([^\n]+)", answer or "", flags=re.IGNORECASE)
    if match:
        fields: dict[str, str] = {}
        for part in match.group(1).split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip().lower()] = value.strip().lower()
        mode = fields.get("mode", mode)
        reason = fields.get("reason", reason)
        deterministic_repair = fields.get("deterministic_repair", "no") in {"yes", "true", "1"}
        compact_retry = fields.get("compact_retry", "no") in {"yes", "true", "1"}
    return {
        "mode": mode,
        "reason": reason,
        "deterministic_repair": deterministic_repair,
        "compact_retry": compact_retry,
    }


def _split_claim_lines(answer: str) -> list[str]:
    lines = []
    for line in (answer or "").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned.lower().startswith(("executive summary:", "direct answer:", "confidence:")):
            cleaned = cleaned.split(":", 1)[1].strip()
        if cleaned and not cleaned.lower().startswith(("why this is likely:", "main risks:", "what to watch next:")):
            lines.append(cleaned)
    return lines


def _supported_claim_ratio(answer: str, chunks: list[dict[str, Any]]) -> float:
    cited_ids = _extract_citations(answer)
    if not cited_ids:
        return 0.0

    claim_lines = [line for line in _split_claim_lines(answer) if "[S" in line]
    if not claim_lines:
        return 0.0

    supported = 0
    for line in claim_lines:
        claim_tokens = _tokenize(re.sub(r"\[S\d+\]", "", line))
        line_cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not claim_tokens or not line_cites:
            continue

        best_overlap = 0.0
        for cidx in line_cites:
            if cidx <= 0 or cidx > len(chunks):
                continue
            src_tokens = _tokenize(chunks[cidx - 1].get("text", ""))
            if not src_tokens:
                continue
            overlap = len(claim_tokens.intersection(src_tokens)) / max(1, len(claim_tokens))
            best_overlap = max(best_overlap, overlap)
        if best_overlap >= 0.25:
            supported += 1

    return round(supported / max(1, len(claim_lines)), 4)


def _citation_valid_ratio(answer: str, chunks: list[dict[str, Any]]) -> float:
    cites = _extract_citations(answer)
    if not cites:
        return 0.0
    valid = sum(1 for c in cites if 1 <= c <= len(chunks))
    return round(valid / len(cites), 4)


def _hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    claim_lines = [line for line in _split_claim_lines(answer) if "[S" in line]
    if not claim_lines:
        return 0.0
    total_nums = 0
    missing_nums = 0
    for line in claim_lines:
        line_wo_cites = re.sub(r"\[S\d+\]", "", line)
        nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", line_wo_cites)
        filtered_nums = []
        for n in nums:
            if n.endswith("%"):
                filtered_nums.append(n)
                continue
            plain = n.replace(".", "")
            if plain.isdigit() and int(float(n)) >= 100:
                filtered_nums.append(n)
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not filtered_nums or not cites:
            continue
        cited_text = []
        for c in cites:
            if 1 <= c <= len(chunks):
                cited_text.append(chunks[c - 1].get("text", ""))
        corpus = " ".join(cited_text)
        for n in filtered_nums:
            total_nums += 1
            if n not in corpus:
                missing_nums += 1
    if total_nums == 0:
        return 0.0
    return round(missing_nums / total_nums, 4)


def _retrieval_hit(question: str, chunks: list[dict[str, Any]], must_include_any: list[str]) -> float:
    if not chunks:
        return 0.0
    lexical_hit = 1.0 if evaluate_retrieval_quality(question, chunks)["avg_token_overlap"] >= 0.08 else 0.0
    if not must_include_any:
        return lexical_hit
        
    corpus = " ".join(c.get("text", "") for c in chunks).lower()
    
    # Also check if it's explicitly retrieved in metadata
    metadata_corpus = " ".join([
        f"{c.get('sector', '')} {c.get('region', '')} {c.get('metadata', {}).get('company', '')}" 
        for c in chunks
    ]).lower()
    
    hits = 0
    for tok in must_include_any:
        tok_lower = tok.lower()
        if tok_lower in corpus or tok_lower in metadata_corpus:
            hits += 1
            
    keyword_hit = hits / len(must_include_any)
    # Blend weak keyword supervision with lexical-overlap signal.
    return round((0.5 * keyword_hit) + (0.5 * lexical_hit), 4)


def _load_eval_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("thresholds", {})
    data.setdefault("queries", [])
    return data


def _pass_fail(summary: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    checks = []
    checks.append(("retrieval_hit_rate", summary["retrieval_hit_rate"], thresholds.get("retrieval_hit_rate_min", 0.65), True))
    checks.append(("avg_token_overlap", summary["avg_token_overlap"], thresholds.get("avg_token_overlap_min", 0.08), True))
    checks.append(("citation_valid_ratio", summary["citation_valid_ratio"], thresholds.get("citation_valid_ratio_min", 0.9), True))
    checks.append(("grounding_ratio", summary["grounding_ratio"], thresholds.get("grounding_ratio_min", 0.75), True))
    checks.append(("hallucination_risk", summary["hallucination_risk"], thresholds.get("hallucination_risk_max", 0.2), False))

    failed = []
    for metric, value, target, is_min in checks:
        if is_min and value < target:
            failed.append(f"{metric}={value:.3f} < {target:.3f}")
        if not is_min and value > target:
            failed.append(f"{metric}={value:.3f} > {target:.3f}")
    return len(failed) == 0, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval + grounding quality with production-style gates.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_EVAL_CONFIG), help="Path to eval query config JSON.")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH), help="Path to write eval report JSON.")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit evaluated queries (0 means all).")
    args = parser.parse_args()

    config_path = Path(args.config)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_eval_config(config_path)
    thresholds = cfg.get("thresholds", {})
    queries = cfg.get("queries", [])
    if not queries:
        print("❌ No queries found in eval config.")
        return 1
    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]

    index = load_index()
    metadata = load_metadata()
    if index.ntotal <= 0 or not metadata:
        print("❌ Missing/empty index or metadata.")
        return 1

    results = []
    for q in queries:
        question = (q.get("question") or "").strip()
        if not question:
            continue
        must_include_any = q.get("must_include_any") or []
        category = q.get("category") or "general"

        retrieval_error = None
        try:
            qvec = asyncio.run(embed_query(question))
            chunks = retrieve_chunks(qvec, index, metadata, question=question)
        except Exception as exc:
            retrieval_error = str(exc)
            chunks = retrieve_chunks_lexical(question, metadata)

        rquality = evaluate_retrieval_quality(question, chunks)
        answer, answer_chunks = asyncio.run(run_query(question))
        generation_diag = _parse_generation_diagnostics(answer)
        # Use chunks tied to answer if available, else fallback to retrieval list.
        used_chunks = answer_chunks or chunks

        item = {
            "question": question,
            "category": category,
            "retrieval": {
                "status": rquality["status"],
                "score": rquality["score"],
                "avg_token_overlap": rquality["avg_token_overlap"],
                "chunk_count": rquality["chunk_count"],
                "issues": rquality["issues"],
                "hit_rate": _retrieval_hit(question, used_chunks, must_include_any),
            },
            "generation": {
                "citation_count": len(_extract_citations(answer)),
                "citation_valid_ratio": _citation_valid_ratio(answer, used_chunks),
                "grounding_ratio": _supported_claim_ratio(answer, used_chunks),
                "hallucination_risk": _hallucination_risk(answer, used_chunks),
                "mode": generation_diag["mode"],
                "fallback_reason": generation_diag["reason"],
                "deterministic_repair": generation_diag["deterministic_repair"],
                "compact_retry": generation_diag["compact_retry"],
            },
            "retrieval_fallback_used": bool(retrieval_error),
            "generation_fallback_used": generation_diag["mode"] == "fallback",
        }
        results.append(item)
        print(f"✓ Evaluated: {question}")

    n = max(1, len(results))
    summary = {
        "query_count": len(results),
        "retrieval_hit_rate": round(sum(r["retrieval"]["hit_rate"] for r in results) / n, 4),
        "avg_token_overlap": round(sum(r["retrieval"]["avg_token_overlap"] for r in results) / n, 4),
        "citation_valid_ratio": round(sum(r["generation"]["citation_valid_ratio"] for r in results) / n, 4),
        "grounding_ratio": round(sum(r["generation"]["grounding_ratio"] for r in results) / n, 4),
        "hallucination_risk": round(sum(r["generation"]["hallucination_risk"] for r in results) / n, 4),
        "retrieval_fallback_rate": round(sum(1 for r in results if r["retrieval_fallback_used"]) / n, 4),
        "generation_fallback_rate": round(sum(1 for r in results if r["generation_fallback_used"]) / n, 4),
        "deterministic_repair_rate": round(
            sum(1 for r in results if r["generation"].get("deterministic_repair")) / n,
            4,
        ),
        "compact_retry_rate": round(
            sum(1 for r in results if r["generation"].get("compact_retry")) / n,
            4,
        ),
    }

    fallback_reason_counts = Counter(
        r["generation"].get("fallback_reason", "unknown")
        for r in results
        if r.get("generation_fallback_used")
    )
    summary["generation_fallback_reasons"] = dict(fallback_reason_counts)

    categories: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        categories.setdefault(item.get("category", "general"), []).append(item)
    category_summary: dict[str, Any] = {}
    for cat, items in categories.items():
        m = max(1, len(items))
        category_summary[cat] = {
            "query_count": len(items),
            "grounding_ratio": round(sum(i["generation"]["grounding_ratio"] for i in items) / m, 4),
            "hallucination_risk": round(sum(i["generation"]["hallucination_risk"] for i in items) / m, 4),
            "retrieval_hit_rate": round(sum(i["retrieval"]["hit_rate"] for i in items) / m, 4),
        }

    passed, failures = _pass_fail(summary, thresholds)
    cat_ground_min = thresholds.get("category_grounding_ratio_min", 0.65)
    # Stricter requirement now that retrieval is stronger
    cat_hall_max = thresholds.get("category_hallucination_risk_max", 0.1)
    for cat, s in category_summary.items():
        if s["grounding_ratio"] < cat_ground_min:
            failures.append(f"{cat}.grounding_ratio={s['grounding_ratio']:.3f} < {cat_ground_min:.3f}")
        if s["hallucination_risk"] > cat_hall_max:
            failures.append(f"{cat}.hallucination_risk={s['hallucination_risk']:.3f} > {cat_hall_max:.3f}")
    passed = len(failures) == 0

    report = {
        "thresholds": thresholds,
        "summary": summary,
        "category_summary": category_summary,
        "industry_gate": {
            "passed": passed,
            "failed_checks": failures,
        },
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ RAG evaluation complete")
    print(json.dumps(report["summary"], indent=2))
    print(f"Industry gate: {'PASS' if passed else 'FAIL'}")
    if failures:
        print("Failed checks:")
        for fitem in failures:
            print(f"- {fitem}")
    print(f"\nReport saved to: {report_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
