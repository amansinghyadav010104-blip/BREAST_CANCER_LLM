
import logging
import re
import time

from .hybrid_retriever import hybrid_retrieve
from .llm_generator import (
    ApplicationDeadlineError,
    AnswerGenerationError,
    ProviderTimeoutError,
    QueryTranslationError,
    generate_answer,
)
from .safety_guardrails import check_safety
from .user_mode import UserMode
from .answer_language import AnswerLanguage, AnswerLanguageGenerationError
from .safety_guardrails import insufficient_evidence_message, localized_safety_message
from .cancer_scope import normalize_breast_spelling

logger = logging.getLogger(__name__)
REQUEST_BUDGET_SECONDS = 30.0


def _log_stage_timings(timings):
    labels = (
        "retrieval", "translation", "model_initialization", "embedding",
        "index_load", "document_load", "faiss_search",
        "uploaded_search", "web_search", "context_construction", "provider_generation",
        "validation", "generation", "total",
    )
    print("Request stage timings (seconds):")
    for stage in labels:
        if stage in timings:
            print(f"  {stage}: {timings[stage]:.3f}")


def _evidence_extract(context, language, question=""):
    """Build a clearly labeled extract using only quoted retrieved evidence."""
    blocks = re.split(r"(?=\[EVIDENCE\s+\d+\])", context)
    items = []
    query_terms = set()
    for raw_term in re.findall(r"[a-zA-Z]{4,}", question):
        term = raw_term.casefold()
        if term in {"what", "when", "where", "which", "common", "about", "from"}:
            continue
        query_terms.add(term[:-1] if term.endswith("s") else term)
    if "treatment" in query_terms:
        query_terms.difference_update({"breast", "cancer"})
    for block in blocks:
        match = re.search(r"\[EVIDENCE\s+(\d+)\]", block)
        if not match:
            continue
        evidence_id = match.group(1)
        source = re.search(r"^(?:Source|Title):\s*(.+)$", block, re.M)
        location = re.search(r"^(?:Page|Location):\s*(.+)$", block, re.M)
        content = re.search(r"Content:\s*(.*?)(?:\n\s*Similarity Score:|\Z)", block, re.S)
        text = re.sub(r"\s+", " ", content.group(1)).strip() if content else ""
        # Skip common PDF page furniture when a clean section heading is available.
        overview = re.search(r"\bOverview\s+", text, re.I)
        if overview:
            text = text[overview.end():]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = []
        for sentence in sentences:
            sentence = sentence.strip(" \t\r\n-•")
            if len(sentence) < 24 or len(sentence) > 400:
                continue
            if re.search(r"(?i)\b(?:donate|reading time|key facts|contents|table of contents)\b", sentence):
                continue
            if re.search(r"\b[bcdfghjklmnpqrstvwxyz]{1,2}\s+[a-z]{2,}\b", sentence):
                continue
            if sum("\u0600" <= character <= "\u06ff" for character in sentence) > 5:
                continue
            sentence_terms = {
                term.casefold() for term in re.findall(r"[a-zA-Z]{4,}", sentence)
            }
            sentence_terms |= {term[:-1] for term in sentence_terms if term.endswith("s")}
            if query_terms and not query_terms.intersection(sentence_terms):
                continue
            selected.append(sentence)
            if len(selected) >= 2:
                break
        if not selected:
            continue
        reference = (source.group(1).strip() if source else "Source details unavailable")
        if location and location.group(1).strip():
            reference += f", {location.group(1).strip()}"
        items.extend((sentence, evidence_id, reference) for sentence in selected)
        if len(items) >= 3:
            items = items[:3]
            break
    if not items:
        return None
    if AnswerLanguage(language) == AnswerLanguage.HINDI:
        heading = "## उपलब्ध साक्ष्य से उद्धरण (स्वतंत्र रूप से सत्यापित उत्तर नहीं)"
        safety = "यह उपलब्ध स्रोतों से लिया गया अंश है, सत्यापित चिकित्सीय उत्तर नहीं। व्यक्तिगत निर्णयों के लिए अपनी स्वास्थ्य टीम से बात करें।"
        sources_heading = "### स्रोत"
    elif AnswerLanguage(language) == AnswerLanguage.BILINGUAL:
        heading = "## Evidence-based extract (not a verified generated answer)"
        safety = "यह उपलब्ध स्रोतों से लिया गया अंश है, सत्यापित चिकित्सीय उत्तर नहीं। व्यक्तिगत निर्णयों के लिए अपनी स्वास्थ्य टीम से बात करें।"
        sources_heading = "### Sources / स्रोत"
    else:
        heading = "## Evidence-based extract (not a verified generated answer)"
        safety = "This is an extract from retrieved sources, not a verified generated answer. Consult your care team for personal decisions."
        sources_heading = "### Sources"
    lines = [heading, ""]
    lines.extend(f"- {sentence} [Evidence {number}]" for sentence, number, _ in items)
    lines.extend(["", safety, "", sources_heading])
    lines.extend(f"- {reference} [Evidence {number}]" for _, number, reference in items)
    return "\n".join(lines)


def run_rag_pipeline(
    question,
    mode=UserMode.PATIENT,
    answer_language=AnswerLanguage.ENGLISH,
    uploaded_documents=None,
    translation_cache=None,
):

    total_start = time.perf_counter()
    deadline = total_start + REQUEST_BUDGET_SECONDS
    timings = {}

    print("\n" + "=" * 60)
    print("BREAST CANCER HYBRID RAG + GENAI")
    print("=" * 60)

    print("Selected Mode:", mode.value)

    # Step 0: Safety check
    start = time.perf_counter()
    safety_result = check_safety(question, answer_language)
    timings["safety"] = time.perf_counter() - start

    if not safety_result["safe"]:
        timings["total"] = time.perf_counter() - total_start

        print("\nSafety Guardrail Triggered")
        print("Risk Type:", safety_result["risk_type"])
        print(f"Safety time: {timings['safety']:.2f}s")
        print(f"Total time: {timings['total']:.2f}s")

        return {
            "question": question,
            "mode": mode.value,
            "source": "safety_guardrail",
            "generation_status": "not_attempted",
            "generation_failure": None,
            "answer": safety_result["message"],
            "context": "",
            "timings": timings
        }

    # Step 1: Hybrid retrieval
    print("\nSearching medical documents...")
    start = time.perf_counter()

    try:
        retrieval_result = hybrid_retrieve(
            question, uploaded_documents, translation_cache=translation_cache,
            deadline=deadline,
        )
    except QueryTranslationError as error:
        timings["retrieval"] = time.perf_counter() - start
        timings["total"] = time.perf_counter() - total_start
        print(f"Query translation failed ({type(error).__name__}).")
        _log_stage_timings(timings)
        return {
            "question": question,
            "mode": mode.value,
            "source": "insufficient_evidence",
            "generation_status": "not_attempted",
            "generation_failure": None,
            "answer": insufficient_evidence_message(answer_language),
            "context": "No relevant evidence was retrieved.",
            "timings": timings,
        }
    except (TimeoutError, ProviderTimeoutError) as error:
        timings["retrieval"] = time.perf_counter() - start
        timings["total"] = time.perf_counter() - total_start
        logger.warning("Retrieval exceeded its time limit: %s", type(error).__name__)
        _log_stage_timings(timings)
        return {
            "question": question,
            "mode": mode.value,
            "source": "generation_timeout",
            "generation_status": "not_attempted",
            "generation_failure": "retrieval_timeout",
            "answer": localized_safety_message("generation_unavailable", answer_language)
            + "\n\nThe request exceeded its 30-second time limit during retrieval.",
            "context": "",
            "timings": timings,
        }

    timings["retrieval"] = time.perf_counter() - start
    timings.update(retrieval_result.get("timings", {}))
    timings["context_construction"] = retrieval_result.get("timings", {}).get("context_construction", 0.0)

    print(f"Retrieval time: {timings['retrieval']:.2f}s")

    # Step 2: Get retrieved context
    context = retrieval_result["context"]

    if not retrieval_result["has_evidence"]:
        timings["generation"] = 0.0
        timings["total"] = time.perf_counter() - total_start
        return {
            "question": question,
            "mode": mode.value,
            "source": "insufficient_evidence",
            "generation_status": "not_attempted",
            "generation_failure": None,
            "answer": insufficient_evidence_message(answer_language),
            "context": context,
            "timings": timings,
        }

    # Step 3: Generate answer using Gemini
    print("\nGenerating answer with Gemini...")
    start = time.perf_counter()
    generation_timings = {}

    try:
        if time.perf_counter() >= deadline:
            raise ApplicationDeadlineError("The 30-second request budget expired during retrieval.")
        answer = generate_answer(
            normalize_breast_spelling(question),
            context,
            mode=mode,
            answer_language=answer_language,
            deadline=deadline,
            timings=generation_timings,
        )
    except (AnswerGenerationError, AnswerLanguageGenerationError) as error:
        timings["generation"] = time.perf_counter() - start
        timings.update(generation_timings)
        timings["total"] = time.perf_counter() - total_start
        logger.error("Answer generation failed (%s): %s", type(error).__name__, error)
        _log_stage_timings(timings)
        extract = _evidence_extract(context, answer_language, question)
        if extract:
            fallback_answer = extract
            source = "evidence_extract"
            if isinstance(error, ProviderTimeoutError):
                fallback_answer += "\n\nGeneration did not complete within the request time limit."
        elif isinstance(error, ProviderTimeoutError):
            fallback_answer = (
                localized_safety_message("generation_unavailable", answer_language)
                + "\n\nGeneration exceeded the 30-second request limit, and the retrieved passages could not be safely summarized."
            )
            source = "generation_timeout"
        else:
            fallback_answer = localized_safety_message("generation_unavailable", answer_language)
            source = "generation_unavailable"
        return {
            "question": question,
            "mode": mode.value,
            "source": source,
            "generation_status": "fallback",
            "generation_failure": (
                "application_deadline"
                if isinstance(error, ApplicationDeadlineError)
                else "provider_timeout"
                if isinstance(error, ProviderTimeoutError)
                else "provider_error"
            ),
            "answer": fallback_answer,
            "context": context,
            "timings": timings,
        }

    timings["generation"] = time.perf_counter() - start
    timings.update(generation_timings)

    print(f"Generation time: {timings['generation']:.2f}s")

    # Total time
    timings["total"] = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Safety Check: {timings['safety']:.2f}s")
    print(f"Retrieval:    {timings['retrieval']:.2f}s")
    for stage in (
        "query_normalization",
        "translation",
        "model_initialization",
        "embedding",
        "index_load",
        "document_load",
        "faiss_search",
        "uploaded_search",
        "web_search",
        "context_construction",
        "provider_generation",
        "validation",
    ):
        if stage in timings:
            print(f"{stage.replace('_', ' ').title():<24}{timings[stage]:.2f}s")
    print(f"Generation:   {timings['generation']:.2f}s")
    print(f"Total:        {timings['total']:.2f}s")

    return {
        "question": question,
        "mode": mode.value,
        "source": retrieval_result["source"],
        "generation_status": "generated",
        "generation_failure": None,
        "answer": answer,
        "context": context,
        "timings": timings
    }


if __name__ == "__main__":

    print("=" * 60)
    print("SELECT USER MODE")
    print("=" * 60)
    print("1. Patient Mode")
    print("2. Doctor Mode")

    mode_choice = input("\nEnter your choice (1 or 2): ").strip()

    if mode_choice == "2":
        selected_mode = UserMode.DOCTOR
    else:
        selected_mode = UserMode.PATIENT

    question = input("\nEnter your breast cancer question: ")

    result = run_rag_pipeline(
        question,
        mode=selected_mode
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("EVIDENCE SOURCE")
    print("=" * 60)
    print(result["source"])
