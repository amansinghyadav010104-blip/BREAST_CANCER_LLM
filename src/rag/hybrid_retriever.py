import re
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
MAX_RESULTS_PER_SOURCE = 1
MIN_SEMANTIC_SCORE = 0.45
QUERY_STOP_WORDS = {
    "about", "after", "and", "are", "can", "does", "for", "from", "how",
    "is", "it", "me", "of", "on", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "would",
    "क्या", "कैसे", "कब", "कहाँ", "कौन", "है", "हैं", "का", "की", "के",
    "में", "और", "लिए", "से",
}


def normalize_query(query):
    """Normalize whitespace, spelling and punctuation without dropping medical terms."""
    normalized = normalize_breast_spelling(" ".join(query.split())).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


def _query_terms(query):
    return {
        term for term in re.findall(r"[\w]+", query.casefold())
        if len(term) > 1 and term not in QUERY_STOP_WORDS
    }


def _lexical_overlap(query_terms, text):
    if not query_terms:
        return 0.0
    document_terms = set(re.findall(r"[\w]+", text.casefold()))
    return len(query_terms & document_terms) / len(query_terms)


def _near_duplicate(first, second):
    first_words = set(re.findall(r"[\w]+", first.casefold()))
    second_words = set(re.findall(r"[\w]+", second.casefold()))
    if not first_words or not second_words:
        return False
    return len(first_words & second_words) / len(first_words | second_words) >= 0.92


def select_diverse_results(results, limit=LOCAL_RESULT_LIMIT):
    """Keep strong, nonduplicate passages with one result per source."""
    if limit <= 0:
        return []
    selected = []
    source_counts = {}
    for result in results:
        source = result.get("source", "").strip().casefold()
        text = " ".join(result.get("text", "").casefold().split())
        if not source or not text or source_counts.get(source, 0) >= MAX_RESULTS_PER_SOURCE:
            continue
        if any(_near_duplicate(text, previous.get("text", "")) for previous in selected):
            continue
        selected.append(result)
        source_counts[source] = source_counts.get(source, 0) + 1
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
    normalized_query = normalize_query(query)
    timings["query_normalization"] = time.perf_counter() - normalization_start

    translation_start = time.perf_counter()
    retrieval_query = translate_query_for_retrieval(
        normalized_query, translation_cache=translation_cache, deadline=deadline
    )
    timings["translation"] = time.perf_counter() - translation_start

    local_results, query_embedding, local_timings = retrieve_documents_with_timings(
        retrieval_query, top_k=LOCAL_CANDIDATE_K
    )
    local_results = filter_breast_cancer_evidence(local_results)
    # FAISS returns cosine similarity because both document and query vectors
    # are normalized. Discard weak candidates before lexical reranking so a
    # keyword coincidence cannot promote an unrelated passage into context.
    local_results = [
        result for result in local_results
        if float(result.get("score", 0.0)) >= MIN_SEMANTIC_SCORE
    ]
    terms = _query_terms(retrieval_query)
    for result in local_results:
        semantic_score = float(result.get("score", 0.0))
        lexical_score = _lexical_overlap(terms, result.get("text", ""))
        result["semantic_score"] = semantic_score
        result["lexical_score"] = lexical_score
        result["score"] = semantic_score + min(0.04, lexical_score * 0.04)
    local_results.sort(key=lambda result: result["score"], reverse=True)
    local_results = select_diverse_results(
        local_results,
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
