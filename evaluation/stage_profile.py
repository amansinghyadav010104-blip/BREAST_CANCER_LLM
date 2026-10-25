"""Measure cold and warm local RAG stages in isolated processes."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def worker(warm_runs):
    from huggingface_hub import try_to_load_from_cache

    from src.rag.local_retriever import (
        MODEL_NAME,
        get_document_metadata,
        get_embedding_model,
        get_faiss_index,
        get_query_embedding,
    )

    model_path = try_to_load_from_cache(MODEL_NAME, "modules.json")
    if not isinstance(model_path, str) or not Path(model_path).is_file():
        raise RuntimeError(
            f"Profiling requires the configured model ({MODEL_NAME}) to be cached; "
            "no model download was attempted."
        )

    import importlib
    import time

    def elapsed(start):
        return round((time.perf_counter() - start) * 1000, 3)

    stage = {}
    started = time.perf_counter()
    importlib.import_module("sentence_transformers")
    stage["sentence_transformers_import_ms"] = elapsed(started)

    started = time.perf_counter()
    model = get_embedding_model()
    stage["model_construction_ms"] = elapsed(started)
    stage["model_initialization_total_ms"] = (
        stage["sentence_transformers_import_ms"] + stage["model_construction_ms"]
    )

    started = time.perf_counter()
    index = get_faiss_index()
    stage["index_load_ms"] = elapsed(started)
    started = time.perf_counter()
    metadata = get_document_metadata()
    stage["metadata_load_ms"] = elapsed(started)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata have different record counts.")

    import faiss
    from src.rag.cancer_scope import filter_breast_cancer_evidence
    from src.rag.context_builder import build_context
    from src.rag.relevance_checker import check_relevance

    query = "What are common signs and symptoms of breast cancer?"
    query_key = " ".join(query.split()).casefold()
    vector_key = query_key
    stage["cold"] = {}
    started = time.perf_counter()
    vector = get_query_embedding(vector_key, model).copy()
    faiss.normalize_L2(vector)
    stage["cold"]["query_embedding_ms"] = elapsed(started)

    started = time.perf_counter()
    scores, indices = index.search(vector, 10)
    results = [
        metadata[int(index_id)].copy() | {"score": float(score)}
        for score, index_id in zip(scores[0], indices[0])
        if index_id >= 0
    ]
    stage["cold"]["faiss_search_ms"] = elapsed(started)

    started = time.perf_counter()
    results = filter_breast_cancer_evidence(results)
    stage["cold"]["scope_filter_ms"] = elapsed(started)
    started = time.perf_counter()
    relevance = check_relevance(results, query)
    stage["cold"]["relevance_decision_ms"] = elapsed(started)
    started = time.perf_counter()
    context = build_context(
        local_results=results[:5] if relevance["relevant"] else None
    )
    stage["cold"]["context_construction_ms"] = elapsed(started)
    stage["cold"]["retrieval_context_total_ms"] = sum(stage["cold"].values())
    stage["cold"]["best_similarity"] = relevance["best_score"]
    stage["cold"]["context_characters"] = len(context)

    warm = []
    for _ in range(warm_runs):
        timing = {}
        started = time.perf_counter()
        model = get_embedding_model()
        timing["model_cache_lookup_ms"] = elapsed(started)
        started = time.perf_counter()
        index = get_faiss_index()
        timing["index_cache_lookup_ms"] = elapsed(started)
        started = time.perf_counter()
        metadata = get_document_metadata()
        timing["metadata_cache_lookup_ms"] = elapsed(started)
        started = time.perf_counter()
        vector = get_query_embedding(vector_key, model).copy()
        faiss.normalize_L2(vector)
        timing["query_embedding_ms"] = elapsed(started)
        started = time.perf_counter()
        scores, indices = index.search(vector, 10)
        results = [
            metadata[int(index_id)].copy() | {"score": float(score)}
            for score, index_id in zip(scores[0], indices[0])
            if index_id >= 0
        ]
        timing["faiss_search_ms"] = elapsed(started)
        started = time.perf_counter()
        results = filter_breast_cancer_evidence(results)
        timing["scope_filter_ms"] = elapsed(started)
        started = time.perf_counter()
        relevance = check_relevance(results, query)
        timing["relevance_decision_ms"] = elapsed(started)
        started = time.perf_counter()
        context = build_context(
            local_results=results[:5] if relevance["relevant"] else None
        )
        timing["context_construction_ms"] = elapsed(started)
        timing["retrieval_context_total_ms"] = sum(
            value for value in timing.values()
        )
        warm.append(timing)

    print(json.dumps({"cold": stage, "warm": warm}, separators=(",", ":")))


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.warm_runs)
        return
    if args.cold_runs < 1 or args.warm_runs < 1:
        parser.error("--cold-runs and --warm-runs must be positive")

    workers = []
    for _ in range(args.cold_runs):
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--warm-runs",
                str(args.warm_runs),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHON_DOTENV_DISABLED": "true"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"Cold profiling worker failed (exit {process.returncode}); "
                f"stderr type only: {process.stderr.splitlines()[-1][:160]}"
            )
        records = [
            line for line in process.stdout.splitlines()
            if line.startswith('{"cold":')
        ]
        if not records:
            raise RuntimeError("Cold profiling worker returned no measurement record.")
        workers.append(json.loads(records[-1]))

    cold_keys = workers[0]["cold"]
    cold = {}
    for key in cold_keys:
        values = [worker_result["cold"][key] for worker_result in workers]
        if isinstance(values[0], (int, float)):
            cold[key] = {
                "median": round(statistics.median(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
        else:
            cold[key] = values[0]

    warm_keys = workers[0]["warm"][0]
    warm = {}
    for key in warm_keys:
        values = [
            run[key]
            for worker_result in workers
            for run in worker_result["warm"]
        ]
        warm[key] = {
            "median": round(statistics.median(values), 3),
            "p95": percentile(values, 0.95),
        }

    print(json.dumps({
        "cold_process_count": args.cold_runs,
        "warm_requests_per_process": args.warm_runs,
        "cold_stages_ms": cold,
        "warm_stages_ms": warm,
        "profile_method": (
            "fresh interpreter processes per cold run; cached resources and repeated "
            "query in each process for warm runs"
        ),
        "limitations": [
            "Cold measurements include Python and native-library import behavior; "
            "they are not an OS-cache-flushed benchmark.",
            "No model download or provider request is performed.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
