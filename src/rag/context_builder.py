def build_context(local_results=None, web_results=None):
    """
    Combine retrieved local documents and web search results
    into a single context for the LLM.
    """

    context_parts = []

    if local_results:
        context_parts.append(
            "LOCAL KNOWLEDGE BASE SOURCES"
        )

        for i, result in enumerate(local_results, start=1):

            context_parts.append(
                f"""
Source {i}
Title: {result.get('source', '')}
Page: {result.get('page', '')}
Similarity Score: {result.get('score', 0.0):.4f}

Content:
{result.get('text', '')}
"""
            )

    if web_results:
        context_parts.append(
            "\nWEB SEARCH SOURCES"
        )

        for i, result in enumerate(web_results, start=1):

            context_parts.append(
                f"""
Web Source {i}
Title: {result.get('title', '')}
URL: {result.get('url', '')}
Search Score: {result.get('score', 0.0):.4f}

Content:
{result.get('content', '')}
"""
            )

    if not context_parts:
        return "No relevant evidence was retrieved."

    return "\n".join(context_parts)


if __name__ == "__main__":

    test_local_results = [
        {
            "source": "Treatment for Breast Cancer - NCI.pdf",
            "page": 1,
            "score": 0.7068,
            "text": "Breast cancer treatment includes local and systemic treatments."
        }
    ]

    test_web_results = [
        {
            "title": "Breast Cancer Guidelines",
            "url": "https://pubmed.ncbi.nlm.nih.gov/",
            "score": 0.80,
            "content": "Recent research discusses updates in breast cancer treatment guidelines."
        }
    ]

    context = build_context(
        local_results=test_local_results,
        web_results=test_web_results
    )

    print("=" * 60)
    print("GENERATED CONTEXT")
    print("=" * 60)

    print(context)