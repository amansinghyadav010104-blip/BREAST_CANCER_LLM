import re


def build_context(local_results=None, web_results=None, uploaded_results=None):
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

    if uploaded_results:
        context_parts.append("SESSION UPLOAD EVIDENCE")

        for result in uploaded_results:
            location = (
                f"Page: {result['page']}"
                if result.get("page") is not None
                else f"Location: {result.get('section') or 'not available'}"
            )
            context_parts.append(
                f"""
[EVIDENCE {evidence_number}]

Source: {result.get('source', '')}
{location}
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


def extract_evidence_sources(context):
    """Return the source metadata attached to each evidence ID in the context."""
    sources = []
    for block in re.split(r"(?=\[EVIDENCE\s+\d+\])", context):
        evidence = re.search(r"\[EVIDENCE\s+(\d+)\]", block, re.I)
        source = re.search(r"^(?:Source|Title):\s*(.+)$", block, re.M)
        page = re.search(r"^Page:\s*(.+)$", block, re.M)
        location = re.search(r"^Location:\s*(.+)$", block, re.M)
        url = re.search(r"^URL:\s*(https?://\S+)\s*$", block, re.M)
        if not evidence or not source:
            continue
        reference = {
            "evidence_id": evidence.group(1),
            "source": source.group(1).strip(),
        }
        if page and page.group(1).strip():
            reference["location"] = f"Page {page.group(1).strip()}"
        elif location and location.group(1).strip():
            reference["location"] = location.group(1).strip()
        if url:
            reference["url"] = url.group(1)
        sources.append(reference)
    return sources


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