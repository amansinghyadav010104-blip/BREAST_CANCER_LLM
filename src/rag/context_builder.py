import re


MAX_CONTEXT_CHARACTERS = 12000
MAX_PASSAGE_CHARACTERS = 2800


def _normalized_text(text):
    return " ".join(re.sub(r"[^\w]+", " ", str(text).casefold()).split())


def _evidence_records(local_results, web_results, uploaded_results):
    records = []
    for kind, results in (
        ("local", local_results or []),
        ("uploaded", uploaded_results or []),
        ("web", web_results or []),
    ):
        for result in results:
            if kind == "web":
                content = result.get("content", "")
                source = result.get("title", "")
                if not content or not source:
                    continue
                records.append({
                    "kind": kind,
                    "source": source,
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                    "text": content,
                })
                continue

            content = result.get("text", "")
            source = result.get("source", "")
            if not content or not source:
                continue
            record = {
                "kind": kind,
                "source": source,
                "score": result.get("score", 0.0),
                "text": content,
            }
            if result.get("page") is not None:
                record["page"] = result["page"]
            elif result.get("section"):
                record["location"] = result["section"]
            records.append(record)
    return records


def _format_evidence(record, evidence_number, text):
    lines = [f"[EVIDENCE {evidence_number}]"]
    group_heading = {
        "local": "LOCAL KNOWLEDGE BASE EVIDENCE",
        "uploaded": "SESSION UPLOAD EVIDENCE",
        "web": "TRUSTED WEB EVIDENCE",
    }.get(record["kind"])
    if group_heading:
        lines.insert(0, group_heading)
    lines.append(f"Source: {record['source']}")
    if "page" in record:
        lines.append(f"Page: {record['page']}")
    elif "location" in record:
        lines.append(f"Location: {record['location']}")
    if record.get("url"):
        lines.append(f"URL: {record['url']}")
    lines.extend((f"Similarity Score: {float(record.get('score', 0.0)):.4f}", "", "Content:", text))
    return "\n".join(lines)


def build_context(
    local_results=None,
    web_results=None,
    uploaded_results=None,
    max_characters=MAX_CONTEXT_CHARACTERS,
):
    """
    Build structured evidence context for the LLM.

    Keep the most relevant retrieved passages first and preserve source metadata.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero.")

    records = _evidence_records(local_results, web_results, uploaded_results)
    context_parts = []
    seen_passages = set()
    used_characters = 0

    for record in records:
        normalized = _normalized_text(record["text"])
        duplicate_key = (
            record["kind"],
            record["source"].casefold(),
            str(record.get("page", record.get("location", ""))).casefold(),
            normalized,
        )
        if not normalized or duplicate_key in seen_passages:
            continue

        text = " ".join(record["text"].split())[:MAX_PASSAGE_CHARACTERS]
        evidence_number = len(context_parts) + 1
        block = _format_evidence(record, evidence_number, text)
        separator = "\n\n" if context_parts else ""
        remaining = max_characters - used_characters - len(separator) - len(block)
        if remaining < 0:
            text = text[:max(0, len(text) + remaining)].rsplit(" ", 1)[0]
            if not text:
                break
            block = _format_evidence(record, evidence_number, text)
            if used_characters + len(separator) + len(block) > max_characters:
                break
        context_parts.append(block)
        seen_passages.add(duplicate_key)
        used_characters += len(separator) + len(block)

    if not context_parts:
        return "No relevant evidence was retrieved."

    return "\n\n".join(context_parts)


def extract_evidence_sources(context):
    """Return unique source/location references while retaining their evidence IDs."""
    sources = []
    references = {}
    for block in re.split(r"(?=\[EVIDENCE\s+\d+\])", context):
        evidence = re.search(r"\[EVIDENCE\s+(\d+)\]", block, re.I)
        source = re.search(r"^(?:Source|Title):\s*(.+)$", block, re.M)
        page = re.search(r"^Page:\s*(.+)$", block, re.M)
        location = re.search(r"^Location:\s*(.+)$", block, re.M)
        url = re.search(r"^URL:\s*(https?://\S+)\s*$", block, re.M)
        if not evidence or not source:
            continue
        source_name = source.group(1).strip()
        source_location = None
        if page and page.group(1).strip():
            source_location = f"Page {page.group(1).strip()}"
        elif location and location.group(1).strip():
            source_location = location.group(1).strip()
        source_url = url.group(1) if url else None
        key = (source_name.casefold(), (source_location or "").casefold(), source_url or "")
        if key not in references:
            reference = {
                "evidence_id": evidence.group(1),
                "evidence_ids": [],
                "source": source_name,
            }
            if source_location:
                reference["location"] = source_location
            if source_url:
                reference["url"] = source_url
            references[key] = reference
            sources.append(reference)
        references[key]["evidence_ids"].append(evidence.group(1))
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
