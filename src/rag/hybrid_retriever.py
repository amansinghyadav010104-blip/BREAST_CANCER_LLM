import time

from .local_retriever import retrieve_documents_with_timings, retrieve_uploaded_documents
from .relevance_checker import check_relevance
from .web_search import web_search
from .context_builder import build_context
from .cancer_scope import (
    filter_breast_cancer_evidence,
    is_breast_cancer_related,
    normalize_breast_spelling,
)
from .llm_generator import translate_query_for_retrieval


LOCAL_CANDIDATE_K = 10
LOCAL_RESULT_LIMIT = 5
WEB_MAX_RESULTS = 5


def select_diverse_results(results, limit=LOCAL_RESULT_LIMIT):
    """Select the highest-ranked unique passages with at most one per source."""
    if limit <= 0:
        return []
    selected = []
    seen_sources = set()
    seen_texts = set()
    for result in results:
        source = result.get("source", "").strip()
        text = " ".join(result.get("text", "").casefold().split())
        if not source or not text or source in seen_sources or text in seen_texts:
            continue
        selected.append(result)
        seen_sources.add(source)
        seen_texts.add(text)
        if len(selected) == limit:
            break
    return selected


def hybrid_retrieve(query, uploaded_documents=None, translation_cache=None, deadline=None):
    """
    Hybrid retrieval pipeline.

    1. Search the local vector database.
    2. Check whether local results are relevant enough.
    3. If relevant, use local evidence.
    4. If not relevant, search trusted web sources.
    5. Build the final context.
    """

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL")
    print("=" * 60)

    # Step 1: Local retrieval
    print("\nSearching local knowledge base...")

    timings = {}
    normalization_start = time.perf_counter()
    normalized_query = normalize_breast_spelling(query)
    timings["query_normalization"] = time.perf_counter() - normalization_start

    translation_start = time.perf_counter()
    retrieval_query = translate_query_for_retrieval(
        normalized_query, translation_cache=translation_cache, deadline=deadline
    )
    timings["translation"] = time.perf_counter() - translation_start

    local_results, query_embedding, local_timings = retrieve_documents_with_timings(
        retrieval_query, top_k=LOCAL_CANDIDATE_K
    )
    local_results = select_diverse_results(
        filter_breast_cancer_evidence(local_results),
        limit=LOCAL_RESULT_LIMIT,
    )
    timings.update(local_timings)
    upload_search_start = time.perf_counter()
    uploaded_results = retrieve_uploaded_documents(
        retrieval_query,
        uploaded_documents or [],
        top_k=LOCAL_RESULT_LIMIT,
        query_embedding=query_embedding,
    )
    timings["uploaded_search"] = time.perf_counter() - upload_search_start

    # Step 2: Check relevance
    relevance = check_relevance(
        local_results,
        retrieval_query
    )

    print(
        "Best local similarity score:",
        round(relevance["best_score"], 4)
    )

    print(
        "Local evidence relevant:",
        relevance["relevant"]
    )

    print(
        "Reason:",
        relevance["reason"]
    )

    # Step 3: Decide retrieval source
    web_results = []

    if relevance["relevant"] or uploaded_results:
        source_label = (
            "local and uploaded evidence"
            if relevance["relevant"] and uploaded_results
            else "local knowledge base"
            if relevance["relevant"]
            else "uploaded evidence"
        )
        print(f"\nUsing {source_label}.")
        timings["web_search"] = 0.0

    else:

        print(
            "\nLocal evidence is not relevant enough."
        )

        print(
            "Searching trusted web sources..."
        )

        web_search_start = time.perf_counter()
        web_timeout = 6.0
        if deadline is not None:
            web_timeout = min(web_timeout, max(0.1, deadline - time.perf_counter()))
        web_results = web_search(
            retrieval_query,
            max_results=WEB_MAX_RESULTS,
            timeout=web_timeout,
        )
        timings["web_search"] = time.perf_counter() - web_search_start
        web_results = [
            result for result in web_results
            if result.get("content") and is_breast_cancer_related(
                result.get("content", ""),
                f"{result.get('title', '')} {result.get('url', '')}",
            )
        ]

        print(
            "Trusted web results found:",
            len(web_results)
        )

    # Step 4: Build final context
    context_start = time.perf_counter()
    context = build_context(
        local_results=local_results
        if relevance["relevant"]
        else None,

        web_results=web_results,
        uploaded_results=uploaded_results,
    )
    timings["context_construction"] = time.perf_counter() - context_start

    evidence_sources = []
    if relevance["relevant"]:
        evidence_sources.append("local")
    if uploaded_results:
        evidence_sources.append("upload")
    if web_results:
        evidence_sources.append("web")

    return {
        "query": query,
        "source": "+".join(evidence_sources) if evidence_sources else "none",
        "local_results": local_results,
        "web_results": web_results,
        "uploaded_results": uploaded_results,
        "context": context,
        "has_evidence": bool(evidence_sources),
        "timings": timings,
    }


if __name__ == "__main__":

    query = input(
        "Enter your breast cancer question: "
    )

    result = hybrid_retrieve(query)

    print("\n")
    print("=" * 60)
    print("FINAL RETRIEVAL SOURCE")
    print("=" * 60)

    print(result["source"])

    print("\n")
    print("=" * 60)
    print("FINAL CONTEXT")
    print("=" * 60)

    print(
        result["context"][:5000]
    )
