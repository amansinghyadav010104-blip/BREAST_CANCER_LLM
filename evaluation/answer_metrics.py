"""Parsing helpers for validating citations against actual retrieved evidence."""

import re


def evidence_by_id(context):
    evidence = {}
    for block in re.split(r"(?=\[EVIDENCE\s+\d+\])", context):
        match = re.search(r"\[EVIDENCE\s+(\d+)\]", block, re.I)
        source = re.search(r"^(?:Source|Title):\s*(.+)$", block, re.M)
        location = re.search(r"^(?:Page|Location):\s*(.+)$", block, re.M)
        url = re.search(r"^URL:\s*(https?://\S+)\s*$", block, re.M)
        content = re.search(
            r"Content:\s*(.*?)(?:\n\s*Similarity Score:|\Z)",
            block,
            re.S,
        )
        if not match:
            continue
        evidence[match.group(1)] = {
            "source": source.group(1).strip() if source else "",
            "location": location.group(1).strip() if location else "",
            "url": url.group(1) if url else "",
            "content": " ".join(content.group(1).split()) if content else "",
        }
    return evidence


def cited_claims(answer):
    claims = []
    in_sources = False
    for line in answer.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,4}\s*Sources?\b", stripped, re.I):
            in_sources = True
        elif re.match(r"^#{1,4}\s+", stripped):
            in_sources = False
        if in_sources:
            continue
        citations = re.findall(r"\[Evidence\s+(\d+)\]", stripped, re.I)
        if not citations:
            continue
        claim = re.sub(r"\[Evidence\s+\d+\]", "", stripped, flags=re.I)
        claim = re.sub(r"^[-*•]\s*", "", claim).strip()
        if claim:
            claims.append({"claim": claim, "evidence_ids": list(dict.fromkeys(citations))})
    return claims


def citation_traceability(answer, context):
    evidence = evidence_by_id(context)
    claims = cited_claims(answer)
    total = sum(len(claim["evidence_ids"]) for claim in claims)
    valid = sum(
        evidence_id in evidence
        for claim in claims
        for evidence_id in claim["evidence_ids"]
    )
    return {
        "claim_count": len(claims),
        "citation_count": total,
        "valid_citation_count": valid,
        "valid_citation_rate": round(valid / total, 3) if total else None,
        "claims": claims,
        "evidence": evidence,
    }
