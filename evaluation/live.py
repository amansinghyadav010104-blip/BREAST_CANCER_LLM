"""Run a small live Gemini benchmark and evaluate citations against retrieved text."""

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from evaluation.answer_metrics import citation_traceability
from src.rag import rag_pipeline
from src.rag.answer_language import AnswerLanguage
from src.rag.llm_generator import (
    GEMINI_MODEL,
    _create_interaction,
    _read_text_response,
)
from src.rag.safety_guardrails import check_safety


CASES_PATH = ROOT / "evaluation" / "cases.json"
LIVE_CASE_IDS = (
    "definition-01",
    "symptoms-01",
    "treatment-types-01",
)
GENERATION_BUDGET_SECONDS = 30
JUDGE_BUDGET_SECONDS = 12


def judge_cited_claims(safety):
    claims = safety["claims"]
    evidence = safety["evidence"]
    prompt_items = []
    for index, claim in enumerate(claims, start=1):
        passages = []
        for evidence_id in claim["evidence_ids"]:
            item = evidence.get(evidence_id)
            if item:
                passages.append({
                    "evidence_id": evidence_id,
                    "source": item["source"],
                    "location": item["location"],
                    "text": item["content"],
                })
        prompt_items.append({
            "claim_index": index,
            "claim": claim["claim"],
            "cited_passages": passages,
        })

    if not prompt_items:
        return {"status": "no_cited_claims", "verdicts": []}

    prompt = (
        "You are evaluating citation support, not giving medical advice. "
        "For each claim, judge only whether the cited passage(s) support that claim. "
        "Use supported only when the passage entails the claim, unsupported when "
        "it conflicts or does not substantiate it, and unclear when evidence is "
        "ambiguous or insufficient. Do not use outside medical knowledge. "
        "Return JSON only with this exact shape: "
        '{"assessments":[{"claim_index":1,"verdict":"supported|unsupported|unclear"}]}. '
        "Assess every claim exactly once.\n\n"
        f"Question: {safety.get('question', '')}\n"
        f"Claims and actual cited passages:\n{json.dumps(prompt_items, ensure_ascii=True)}"
    )
    try:
        response = _create_interaction(
            prompt,
            deadline=time.perf_counter() + JUDGE_BUDGET_SECONDS,
            generation_config={"thinking_level": "low", "max_output_tokens": 500},
        )
        raw = _read_text_response(response)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {"status": "invalid_judge_output", "verdicts": []}
        parsed = json.loads(match.group())
        assessments = parsed.get("assessments")
        if not isinstance(assessments, list):
            return {"status": "invalid_judge_output", "verdicts": []}
        valid_indexes = set(range(1, len(claims) + 1))
        verdicts = {}
        for item in assessments:
            if not isinstance(item, dict):
                continue
            index = item.get("claim_index")
            verdict = item.get("verdict")
            if index in valid_indexes and verdict in {"supported", "unsupported", "unclear"}:
                verdicts[index] = verdict
        if set(verdicts) != valid_indexes:
            return {"status": "incomplete_judge_output", "verdicts": list(verdicts.values())}
        return {"status": "judged", "verdicts": [verdicts[i] for i in sorted(verdicts)]}
    except Exception as error:
        return {
            "status": "timeout" if "timeout" in type(error).__name__.casefold() else "error",
            "error_type": type(error).__name__,
            "verdicts": [],
        }


def run_case(case):
    started = time.perf_counter()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = rag_pipeline.run_rag_pipeline(
            case["question"],
            answer_language=AnswerLanguage.ENGLISH,
        )
    elapsed = time.perf_counter() - started
    answer = result.get("answer", "")
    context = result.get("context", "")
    citations = citation_traceability(answer, context)
    citations["question"] = case["question"]
    logging_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        judge = (
            judge_cited_claims(citations)
            if result.get("generation_status") == "generated"
            else {"status": "not_run", "verdicts": []}
        )
    finally:
        logging.disable(logging_disable_level)

    return {
        "case_id": case["id"],
        "generation_status": result.get("generation_status", "unknown"),
        "generation_failure": result.get("generation_failure"),
        "result_source": result.get("source"),
        "elapsed_seconds": round(elapsed, 3),
        "stage_timings_seconds": {
            name: round(value, 3)
            for name, value in result.get("timings", {}).items()
            if isinstance(value, (int, float))
        },
        "context_characters": len(context),
        "citation_claim_count": citations["claim_count"],
        "citation_count": citations["citation_count"],
        "valid_citation_count": citations["valid_citation_count"],
        "valid_citation_rate": citations["valid_citation_rate"],
        "faithfulness_judge_status": judge["status"],
        "faithfulness_verdicts": Counter(judge["verdicts"]),
        "faithfulness_judge_error_type": judge.get("error_type"),
        "retrieved_sources": sorted({
            source["source"] for source in citations["evidence"].values()
            if source["source"]
        }),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY is not configured. No provider request was made."
        )

    cases_by_id = {
        case["id"]: case
        for case in json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    }
    selected = [cases_by_id[case_id] for case_id in LIVE_CASE_IDS[:args.limit]]
    report = {
        "provider": "Gemini",
        "model": GEMINI_MODEL,
        "answer_request_limit": args.limit,
        "maximum_judge_requests": args.limit,
        "provider_calls_include": "At most one answer and one support-judge request per case.",
        "cases": [],
    }
    for case in selected:
        safety = check_safety(case["question"])
        if not safety["safe"]:
            report["cases"].append({
                "case_id": case["id"],
                "generation_status": "not_attempted",
                "generation_failure": "safety_guardrail",
            })
            continue
        try:
            report["cases"].append(run_case(case))
        except Exception as error:
            report["cases"].append({
                "case_id": case["id"],
                "generation_status": "error",
                "generation_failure": "unexpected_error",
                "error_type": type(error).__name__,
                "elapsed_seconds": None,
                "valid_citation_rate": None,
                "faithfulness_judge_status": "not_run",
            })

    counts = Counter(item.get("generation_status", "unknown") for item in report["cases"])
    failures = Counter(
        item["generation_failure"]
        for item in report["cases"]
        if item.get("generation_failure")
    )
    elapsed = [
        item["elapsed_seconds"]
        for item in report["cases"]
        if "elapsed_seconds" in item
    ]
    judge_counts = Counter(
        item.get("faithfulness_judge_status", "not_run")
        for item in report["cases"]
    )
    verdict_counts = Counter()
    for item in report["cases"]:
        verdict_counts.update(item.get("faithfulness_verdicts", {}))
    citation_rates = [
        item["valid_citation_rate"]
        for item in report["cases"]
        if item.get("valid_citation_rate") is not None
    ]
    successful_generation_times = [
        item["stage_timings_seconds"]["provider_generation"]
        for item in report["cases"]
        if item.get("generation_status") == "generated"
        and "provider_generation" in item.get("stage_timings_seconds", {})
    ]
    timeout_count = failures["provider_timeout"] + failures["application_deadline"]
    provider_error_count = failures["provider_error"] + failures["unexpected_error"]
    report["summary"] = {
        "generated": counts["generated"],
        "fallback": counts["fallback"],
        "not_attempted": counts["not_attempted"],
        "generation_failures": dict(failures),
        "timeout_count": timeout_count,
        "provider_error_count": provider_error_count,
        "fallback_frequency": (
            round(counts["fallback"] / len(report["cases"]), 3)
            if report["cases"] else None
        ),
        "elapsed_seconds_mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
        "successful_provider_generation_seconds_mean": (
            round(sum(successful_generation_times) / len(successful_generation_times), 3)
            if successful_generation_times else None
        ),
        "valid_citation_rate_mean": (
            round(sum(citation_rates) / len(citation_rates), 3)
            if citation_rates else None
        ),
        "faithfulness_judge_statuses": dict(judge_counts),
        "claim_support_verdicts": dict(verdict_counts),
        "provider_error_rate": (
            round(provider_error_count / len(report["cases"]), 3)
            if report["cases"] else None
        ),
        "timeout_rate": (
            round(timeout_count / len(report["cases"]), 3)
            if report["cases"] else None
        ),
    }
    report["limitations"] = [
        "The live sample uses fixed public-information questions and is not representative of user traffic.",
        "The support judge uses the same configured Gemini model/provider and is not an independent evaluator.",
        "A model-judged support result is a screening signal, not clinical validation.",
        "The report omits answer text, prompts, exception messages, and credential values.",
    ]
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
