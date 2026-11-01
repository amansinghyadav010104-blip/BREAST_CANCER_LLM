
import os
import re
import sys
import traceback
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import run_rag_pipeline
from src.rag.context_builder import extract_evidence_sources
from src.rag.user_mode import UserMode
from src.rag.answer_language import AnswerLanguage
from src.rag.answer_language import AnswerLanguageGenerationError
from src.rag.document_ingestion import get_or_embed_uploaded_chunks
from src.rag.speech_component import speech_controls
from src.rag.local_retriever import get_embedding_model
st.set_page_config(
    page_title="Breast Cancer AI Assistant",
    page_icon="🎗️",
    layout="centered"
)

st.title("🎗️ Breast Cancer AI Assistant")
st.caption(
    "Evidence-based breast cancer education "
    "and clinical support"
)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    mode_name = st.selectbox(
        "User mode",
        ["Patient Mode", "Doctor Mode"],
    )
    answer_language = st.selectbox(
        "Answer language",
        list(AnswerLanguage),
        format_func=lambda language: language.value,
    )
    voice_locale = st.selectbox(
        "Voice input language",
        ["en-IN", "hi-IN"],
        format_func=lambda locale: "English" if locale == "en-IN" else "हिंदी",
    )
    uploaded_files = st.file_uploader(
        "Add evidence files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        max_upload_size=20,
    )

    upload_cache = st.session_state.get("uploaded_chunks_cache", {})
    active_upload_cache = {}
    for uploaded_file in uploaded_files or []:
        content = uploaded_file.getvalue()
        try:
            file_digest, chunks = get_or_embed_uploaded_chunks(
                uploaded_file.name,
                content,
                upload_cache,
                get_embedding_model().encode,
            )
            active_upload_cache[file_digest] = chunks
        except ValueError as error:
            st.error(f"{uploaded_file.name}: {error}")
    st.session_state.uploaded_chunks_cache = active_upload_cache
    uploaded_documents = [
        chunk
        for chunks in active_upload_cache.values()
        for chunk in chunks
    ]
    if active_upload_cache:
        st.caption(f"{len(active_upload_cache)} file(s) ready for this session")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            generation_status = message.get("generation_status")
            if generation_status == "fallback":
                st.caption("Evidence extract · Gemini did not return a verified answer")
                failure = message.get("generation_failure")
                if failure in {"provider_timeout", "application_deadline"}:
                    st.info("Gemini took too long to respond. This extract comes from retrieved sources.")
                else:
                    st.info("Gemini could not generate an answer. This extract comes from retrieved sources.")
            elif generation_status == "generated":
                st.caption("Generated answer · grounded in retrieved evidence")
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"Retrieved sources ({len(sources)})"):
                    for source in sources:
                        evidence_ids = source.get("evidence_ids") or [source["evidence_id"]]
                        evidence_label = ", ".join(evidence_ids)
                        reference = (
                            f"Evidence {evidence_label}: {source['source']}"
                        )
                        if source.get("location"):
                            reference += f" — {source['location']}"
                        if source.get("url"):
                            reference += f" — {source['url']}"
                        st.text(reference)
            response_time = message.get("response_time")
            if isinstance(response_time, (int, float)):
                st.caption(f"Response time: {response_time:.1f} seconds")

latest_answer = next(
    (message["content"] for message in reversed(st.session_state.messages)
     if message["role"] == "assistant"),
    "",
)
transcript = speech_controls(voice_locale, latest_answer)
if transcript:
    st.session_state["question_input"] = transcript

# User input
question = st.chat_input(
    "Ask a breast cancer question...",
    key="question_input",
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    mode = (
        UserMode.PATIENT
        if mode_name == "Patient Mode"
        else UserMode.DOCTOR
    )

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching evidence and preparing a response...",
                show_time=True,
            ):
                result = run_rag_pipeline(
                    question,
                    mode=mode,
                    answer_language=answer_language,
                    uploaded_documents=uploaded_documents,
                    translation_cache=st.session_state.setdefault(
                        "retrieval_translation_cache", {}
                    ),
                )

            answer = result.get(
                "answer",
                "No answer was generated."
            )

            st.markdown(answer)
            generation_status = result.get("generation_status", "not_attempted")
            if generation_status == "fallback":
                st.caption("Evidence extract · Gemini did not return a verified answer")
                failure = result.get("generation_failure")
                if failure == "provider_timeout" or failure == "application_deadline":
                    st.info("Gemini took too long to respond. The evidence extract below is from retrieved sources and is not a generated answer.")
                else:
                    st.info("Gemini could not generate an answer. The evidence extract below is from retrieved sources and is not a generated answer.")
            elif generation_status == "generated":
                st.caption("Generated answer · grounded in retrieved evidence")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": extract_evidence_sources(result.get("context", "")),
                "generation_status": generation_status,
                "generation_failure": result.get("generation_failure"),
                "response_time": result.get("timings", {}).get("total"),
            })
            st.rerun()

        except AnswerLanguageGenerationError as error:
            st.warning(str(error))
        except Exception as error:
            error_traceback = "".join(traceback.format_exception(error))
            for variable in ("GEMINI_API_KEY", "TAVILY_API_KEY"):
                secret = os.getenv(variable)
                if secret:
                    error_traceback = error_traceback.replace(secret, "[REDACTED]")
            error_traceback = re.sub(
                r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+",
                r"\1[REDACTED]",
                error_traceback,
            )
            print(
                "RAG question processing failed:\n" + error_traceback,
                file=sys.stderr,
                flush=True,
            )
            st.error(
                "Something went wrong while processing "
                "your question. Please try again."
            )

st.divider()

st.caption(
    "For educational support only. "
    "This system does not replace professional "
    "medical advice."
)
