from local_retriever import retrieve_documents
from relevance_checker import check_relevance
from web_search import web_search
from context_builder import build_context


LOCAL_TOP_K = 5
WEB_MAX_RESULTS = 5


def hybrid_retrieve(query):
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

    local_results = retrieve_documents(
        query,
        top_k=LOCAL_TOP_K
    )

    # Step 2: Check relevance
    relevance = check_relevance(
        local_results,
        query
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

    if relevance["relevant"]:

        print("\nUsing local knowledge base.")

    else:

        print(
            "\nLocal evidence is not relevant enough."
        )

        print(
            "Searching trusted web sources..."
        )

        web_results = web_search(
            query,
            max_results=WEB_MAX_RESULTS
        )

        print(
            "Trusted web results found:",
            len(web_results)
        )

    # Step 4: Build final context
    context = build_context(
        local_results=local_results
        if relevance["relevant"]
        else None,

        web_results=web_results
    )

    return {
        "query": query,
        "source": (
            "local"
            if relevance["relevant"]
            else "web"
        ),
        "local_results": local_results,
        "web_results": web_results,
        "context": context
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