"""Run a deterministic, provider-free RAG evaluation against the local corpus."""

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this offline evaluation independent of local credentials and .env.
os.environ["PYTHON_DOTENV_DISABLED"] = "true"
os.environ["GEMINI_API_KEY"] = "offline-evaluation-placeholder"
os.environ["TAVILY_API_KEY"] = "offline-evaluation-placeholder"

from huggingface_hub import try_to_load_from_cache

from src.rag import rag_pipeline
from src.rag.answer_language import AnswerLanguage, validate_answer_language
from src.rag.cancer_scope import filter_breast_cancer_evidence
from src.rag.context_builder import build_context
from src.rag.hybrid_retriever import (
    LOCAL_CANDIDATE_K,
    LOCAL_RESULT_LIMIT,
    select_diverse_results,
)
from src.rag.llm_generator import AnswerGenerationError, ProviderTimeoutError
from src.rag.local_retriever import MODEL_NAME, retrieve_documents_with_timings
from src.rag.relevance_checker import check_relevance
from src.rag.safety_guardrails import check_safety


CASES_PATH = ROOT / "evaluation" / "cases.json"
TOP_K = LOCAL_CANDIDATE_K


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 3)


def normalized(text):
    return " ".join(text.casefold().split())


def context_evidence(context):
    evidence = {}
    for block in re.split(r"(?=\[EVIDENCE\s+\d+\])", context):
        match = re.search(r"\[EVIDENCE\s+(\d+)\]", block, re.I)
        source = re.search(r"^(?:Source|Title):\s*(.+)$", block, re.M)
        content = re.search(r"Content:\s*(.*?)(?:\n\s*Similarity Score:|\Z)", block, re.S)
        if match:
            evidence[match.group(1)] = {
                "source": source.group(1).strip() if source else "",
                "content": normalized(content.group(1)) if content else "",
            }
    return evidence


def evaluate_case(case, top_k):
    safety = check_safety(case["question"])
    expected_behavior = case["expected_behavior"]
    observed_behavior = safety["risk_type"] if not safety["safe"] else "safe_answer"
    row = {
        "id": case["id"],
        "expected_behavior": expected_behavior,
        "observed_behavior": observed_behavior,
        "safety_pass": observed_behavior == expected_behavior,
        "expected_sources": case["expected_sources"],
        "retrieved_sources": [],
        "candidate_sources": [],
        "relevant": False,
        "best_similarity": None,
        "relevance_reason": None,
        "retrieval_ms": None,
        "model_initialization_ms": None,
        "embedding_ms": None,
        "faiss_search_ms": None,
        "index_load_ms": None,
        "metadata_load_ms": None,
        "context_construction_ms": 0.0,
        "context_chars": 0,
        "context_words": 0,
        "duplicate_passage_rate": None,
        "expected_source_recall_at_k": None,
        "expected_source_recall_selected": None,
        "expected_source_recall_in_context": None,
        "expected_source_mrr": None,
        "used_expected_source": None,
        "expected_source_ranks_candidate": [],
        "citation_traceability": None,
        "fallback_claims": None,
        "fallback_exact_support_rate": None,
    }
    if not safety["safe"]:
        return row, None

    query = case.get("retrieval_query", case["question"])
    start = time.perf_counter()
    results, _, timings = retrieve_documents_with_timings(query, top_k=top_k)
    row["retrieval_ms"] = (time.perf_counter() - start) * 1000
    candidates = filter_breast_cancer_evidence(results)
    row["candidate_sources"] = [item.get("source", "") for item in candidates]
    results = select_diverse_results(
        candidates,
        limit=LOCAL_RESULT_LIMIT,
    )
    relevance = check_relevance(results, query)
    row["relevant"] = relevance["relevant"]
    row["best_similarity"] = round(relevance["best_score"], 4)
    row["relevance_reason"] = relevance["reason"]
    row["retrieved_sources"] = [item.get("source", "") for item in results]
    for source_key, timing_key in (
        ("model_initialization_ms", "model_initialization"),
        ("embedding_ms", "embedding"),
        ("faiss_search_ms", "faiss_search"),
        ("index_load_ms", "index_load"),
        ("metadata_load_ms", "document_load"),
    ):
        if timing_key in timings:
            row[source_key] = round(timings[timing_key] * 1000, 3)

    expected = case["expected_sources"]
    candidate_ranks = [
        index
        for index, item in enumerate(candidates, start=1)
        if item.get("source") in expected
    ]
    row["expected_source_ranks_candidate"] = candidate_ranks
    if expected:
        row["expected_source_recall_at_k"] = round(
            len(set(item.get("source") for item in candidates) & set(expected)) / len(expected),
            3,
        )
        row["expected_source_recall_selected"] = round(
            len(set(item.get("source") for item in results) & set(expected)) / len(expected),
            3,
        )
        row["expected_source_mrr"] = (
            round(1 / min(candidate_ranks), 3) if candidate_ranks else 0.0
        )

    selected = results if relevance["relevant"] else []
    if expected:
        row["expected_source_recall_in_context"] = round(
            len(set(item.get("source") for item in selected) & set(expected))
            / len(expected),
            3,
        )
    row["used_expected_source"] = (
        bool(set(item.get("source") for item in selected) & set(expected))
        if expected
        else None
    )
    normalized_texts = [normalized(item.get("text", "")) for item in results]
    if normalized_texts:
        row["duplicate_passage_rate"] = round(
            1 - len(set(normalized_texts)) / len(normalized_texts), 3
        )

    context_start = time.perf_counter()
    context = build_context(local_results=selected)
    row["context_construction_ms"] = round(
        (time.perf_counter() - context_start) * 1000, 3
    )
    row["context_chars"] = len(context)
    row["context_words"] = len(context.split())
    mapped = context_evidence(context)
    row["citation_traceability"] = (
        round(
            sum(bool(item["source"] and item["content"]) for item in mapped.values())
            / len(mapped),
            3,
        )
        if mapped
        else None
    )

    fallback = rag_pipeline._evidence_extract(context, AnswerLanguage.ENGLISH, query)
    claim_section = (fallback or "").split("\n### Sources", 1)[0]
    claims = re.findall(r"^- (.+?) \[Evidence\s+(\d+)\]$", claim_section, re.M)
    supported = sum(
        normalized(claim) in mapped.get(evidence_id, {}).get("content", "")
        for claim, evidence_id in claims
    )
    if claims:
        row["fallback_claims"] = len(claims)
        row["fallback_exact_support_rate"] = round(supported / len(claims), 3)
    return row, context


def validate_fallback_paths(case, context):
    results = {
        "timeout": {"tested": False, "fallback_returned": False, "source": None},
        "provider_error": {"tested": False, "fallback_returned": False, "source": None},
    }
    if not context:
        return results

    retrieval = {"context": context, "has_evidence": True, "source": "local"}
    for label, error in (
        ("timeout", ProviderTimeoutError("evaluation timeout")),
        ("provider_error", AnswerGenerationError("evaluation provider error")),
    ):
        with (
            patch.object(rag_pipeline, "hybrid_retrieve", return_value=retrieval),
            patch.object(rag_pipeline, "generate_answer", side_effect=error),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = rag_pipeline.run_rag_pipeline(case["question"])
        results[label] = {
            "tested": True,
            "fallback_returned": bool(result.get("answer")),
            "source": result.get("source"),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    modules_path = try_to_load_from_cache(MODEL_NAME, "modules.json")
    if not isinstance(modules_path, str) or not Path(modules_path).is_file():
        raise RuntimeError(
            f"Offline evaluation requires the configured embedding model "
            f"({MODEL_NAME}) to be cached locally; no model download was attempted."
        )

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    timings = defaultdict(list)
    rows = []
    contexts = {}
    for case in cases:
        row, context = evaluate_case(case, args.top_k)
        rows.append(row)
        if context:
            contexts[case["id"]] = context
        for key in (
            "retrieval_ms",
            "model_initialization_ms",
            "embedding_ms",
            "faiss_search_ms",
            "index_load_ms",
            "metadata_load_ms",
            "context_construction_ms",
        ):
            if row[key] is not None:
                timings[key].append(row[key])

    first_safe = next((case for case in cases if case["id"] in contexts), None)
    repeated_query = None
    if first_safe:
        query = first_safe.get("retrieval_query", first_safe["question"])
        start = time.perf_counter()
        retrieve_documents_with_timings(query, top_k=args.top_k)
        repeated_query = round((time.perf_counter() - start) * 1000, 3)

    stage_names = {
        "retrieval_ms": "retrieval_ms_p50",
        "model_initialization_ms": "model_initialization_ms_first",
        "embedding_ms": "embedding_ms_p50",
        "faiss_search_ms": "faiss_search_ms_p50",
        "index_load_ms": "index_load_ms_p50",
        "metadata_load_ms": "metadata_load_ms_p50",
        "context_construction_ms": "context_construction_ms_p50",
    }
    latency = {
        output: (
            round(timings[input_key][0], 3)
            if output.endswith("_first") and timings[input_key]
            else percentile(timings[input_key], 0.5)
        )
        for input_key, output in stage_names.items()
    }
    latency.update({
        "first_request_retrieval_ms": (
            round(rows[next(i for i, row in enumerate(rows) if row["retrieval_ms"] is not None)]["retrieval_ms"], 3)
            if any(row["retrieval_ms"] is not None for row in rows)
            else None
        ),
        "retrieval_ms_p95": percentile(timings["retrieval_ms"], 0.95),
        "retrieval_ms_p95_warm": percentile(timings["retrieval_ms"][1:], 0.95),
        "retrieval_ms_same_query_warm": repeated_query,
        "gemini_generation_ms": None,
        "gemini_generation_note": "Not measured: evaluation makes no external Gemini/API calls.",
    })

    safe_rows = [row for row in rows if row["observed_behavior"] == "safe_answer"]
    labeled = [row for row in safe_rows if row["expected_sources"]]
    fallback_case = next(
        (case for case in cases if case["id"] in contexts and case["expected_sources"]),
        None,
    )
    fallback_metrics = validate_fallback_paths(
        fallback_case, contexts.get(fallback_case["id"]) if fallback_case else None
    ) if fallback_case else {}

    language_fixtures = {
        AnswerLanguage.ENGLISH: "## Answer\nBreast cancer is a disease. [Evidence 1]",
        AnswerLanguage.HINDI: "## उत्तर\nस्तन कैंसर एक गंभीर बीमारी है। [Evidence 1]",
        AnswerLanguage.BILINGUAL: (
            "## English\nBreast cancer is a disease. [Evidence 1]\n"
            "## हिंदी\nस्तन कैंसर एक बीमारी है। [Evidence 1]"
        ),
    }
    validation_samples = []
    for language, answer in language_fixtures.items():
        start = time.perf_counter()
        validate_answer_language(answer, language)
        validation_samples.append((time.perf_counter() - start) * 1_000_000)
    latency["language_validation_fixture_us_median"] = round(
        statistics.median(validation_samples), 3
    )
    latency["answer_validation_note"] = (
        "Language validation measured on valid fixtures; provider-generated claim "
        "faithfulness is not assessed offline."
    )

    source_recalls = [
        row["expected_source_recall_at_k"]
        for row in labeled
        if row["expected_source_recall_at_k"] is not None
    ]
    source_mrrs = [
        row["expected_source_mrr"]
        for row in labeled
        if row["expected_source_mrr"] is not None
    ]
    selected_source_recalls = [
        row["expected_source_recall_selected"]
        for row in labeled
        if row["expected_source_recall_selected"] is not None
    ]
    context_source_recalls = [
        row["expected_source_recall_in_context"]
        for row in labeled
        if row["expected_source_recall_in_context"] is not None
    ]
    traceability = [
        row["citation_traceability"]
        for row in safe_rows
        if row["citation_traceability"] is not None
    ]
    exact_support = [
        row["fallback_exact_support_rate"]
        for row in safe_rows
        if row["fallback_exact_support_rate"] is not None
    ]
    duplicate_rates = [
        row["duplicate_passage_rate"]
        for row in safe_rows
        if row["duplicate_passage_rate"] is not None
    ]
    context_sizes = [row["context_chars"] for row in safe_rows]
    context_words = [row["context_words"] for row in safe_rows]
    report = {
        "evaluation_version": 1,
        "provider_calls": 0,
        "index_dimensions": None,
        "index_records": None,
        "case_count": len(cases),
        "retrieval_top_k": args.top_k,
        "metrics": {
            "safety_pass_rate": round(
                sum(row["safety_pass"] for row in rows) / len(rows), 3
            ),
            "safety_failures": [
                {
                    "id": row["id"],
                    "expected": row["expected_behavior"],
                    "observed": row["observed_behavior"],
                }
                for row in rows
                if not row["safety_pass"]
            ],
            "mean_expected_source_recall_at_k": (
                round(statistics.mean(source_recalls), 3) if source_recalls else None
            ),
            "mean_expected_source_recall_in_selected_results": (
                round(statistics.mean(selected_source_recalls), 3)
                if selected_source_recalls else None
            ),
            "mean_expected_source_recall_in_context": (
                round(statistics.mean(context_source_recalls), 3)
                if context_source_recalls else None
            ),
            "mean_expected_source_mrr": (
                round(statistics.mean(source_mrrs), 3) if source_mrrs else None
            ),
            "expected_source_hit_case_rate": (
                round(sum(bool(row["used_expected_source"]) for row in labeled) / len(labeled), 3)
                if labeled else None
            ),
            "threshold_rejected_expected_source_cases": [
                {
                    "id": row["id"],
                    "best_similarity": row["best_similarity"],
                    "candidate_expected_source_ranks": row["expected_source_ranks_candidate"],
                    "candidate_sources": row["candidate_sources"],
                }
                for row in labeled
                if not row["relevant"]
                and set(row["candidate_sources"]) & set(row["expected_sources"])
            ],
            "context_citation_traceability_rate": (
                round(statistics.mean(traceability), 3) if traceability else None
            ),
            "fallback_exact_quote_support_rate": (
                round(statistics.mean(exact_support), 3) if exact_support else None
            ),
            "retrieval_duplicate_passage_rate": (
                round(statistics.mean(duplicate_rates), 3) if duplicate_rates else None
            ),
            "context_chars_mean": round(statistics.mean(context_sizes), 1) if context_sizes else None,
            "context_chars_max": max(context_sizes) if context_sizes else None,
            "context_words_mean": round(statistics.mean(context_words), 1) if context_words else None,
            "context_words_max": max(context_words) if context_words else None,
            "simulated_fallback_paths": fallback_metrics,
            "provider_error_rate": None,
            "provider_error_rate_note": "Not estimable without live provider requests.",
        },
        "latency": latency,
        "cases": rows,
        "limitations": [
            "Source labels are corpus-level relevance judgments, not clinical answer gold standards.",
            "Recall and MRR use source filename labels and do not establish passage entailment.",
            "Faithfulness is reported only as exact support for the deterministic extract fallback; no LLM answer is generated.",
            "No live web search, Gemini calls, or uploaded-file content is used.",
            "Cold-start timing is the first request in this evaluator process, not a full Streamlit server startup.",
        ],
    }

    try:
        from src.rag.local_retriever import get_faiss_index
        index = get_faiss_index()
        report["index_dimensions"] = index.d
        report["index_records"] = index.ntotal
    except (OSError, RuntimeError, ValueError):
        raise
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
