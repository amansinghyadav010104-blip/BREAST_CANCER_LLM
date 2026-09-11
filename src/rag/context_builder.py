def build_context(local_results=None, web_results=None):
    """
    Build structured evidence context for the LLM.

    Each evidence item receives a unique ID so that
    the LLM can clearly identify its sources.
    """

    context_parts = []

    evidence_number = 1

    if local_results:
        context_parts.append("LOCAL KNOWLEDGE BASE EVIDENCE")

        for result in local_results:
            context_parts.append(
                f"""
[EVIDENCE {evidence_number}]

Source: {result.get('source', '')}
Page: {result.get('page', '')}
Similarity Score: {result.get('score', 0.0):.4f}

Content:
{result.get('text', '')}
"""
            )

            evidence_number += 1

    if web_results:
        context_parts.append("\nTRUSTED WEB EVIDENCE")

        for result in web_results:
            context_parts.append(
                f"""
[EVIDENCE {evidence_number}]

Title: {result.get('title', '')}
URL: {result.get('url', '')}
Search Score: {result.get('score', 0.0):.4f}

Content:
{result.get('content', '')}
"""
            )

            evidence_number += 1

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
            "title": "NCCN Guidelines for Patients",
            "url": "https://www.nccn.org/",
            "score": 0.8955,
            "content": "Treatment recommendations for invasive breast cancer."
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