# -----------------------------------------
# Relevance Checker
# -----------------------------------------

RELEVANCE_THRESHOLD = 0.60


def requires_web_search(query):
    """
    Detect queries that require current or updated information.
    """

    query = query.lower().strip()

    current_keywords = [
        "latest",
        "current",
        "recent",
        "new",
        "updated",
        "update",
        "2026",
        "2025",
        "guidelines",
        "new guidelines",
        "latest research",
        "recent research"
    ]

    return any(
        keyword in query
        for keyword in current_keywords
    )


def check_relevance(results, query=""):
    """
    Determine whether local retrieval contains
    sufficiently relevant information.

    Current-information queries are sent to
    web search instead of relying only on the
    local knowledge base.
    """

    if requires_web_search(query):

        best_score = (
            max(
                result["score"]
                for result in results
            )
            if results
            else 0.0
        )

        return {
            "relevant": False,
            "best_score": best_score,
            "reason": "Current information query"
        }

    if not results:

        return {
            "relevant": False,
            "best_score": 0.0,
            "reason": "No local results"
        }

    best_score = max(
        result["score"]
        for result in results
    )

    relevant = (
        best_score >= RELEVANCE_THRESHOLD
    )

    return {
        "relevant": relevant,
        "best_score": best_score,
        "reason": "Similarity threshold"
    }


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":

    test_results = [
        {"score": 0.7068},
        {"score": 0.6510},
        {"score": 0.6498},
        {"score": 0.6182},
        {"score": 0.5718}
    ]

    queries = [
        "What are the treatment options for breast cancer?",
        "What are the latest breast cancer treatment guidelines in 2026?"
    ]

    for query in queries:

        result = check_relevance(
            test_results,
            query
        )

        print("\nQuery:", query)

        print(
            "Best similarity score:",
            round(result["best_score"], 4)
        )

        print(
            "Use local evidence:",
            result["relevant"]
        )

        print(
            "Reason:",
            result["reason"]
        )