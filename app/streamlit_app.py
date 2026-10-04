
import streamlit as st

from src.rag.rag_pipeline import run_rag_pipeline
from src.rag.user_mode import UserMode

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

# Select user mode
mode_name = st.selectbox(
    "Select User Mode",
    ["Patient Mode", "Doctor Mode"]
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input(
    "Ask your breast cancer question..."
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
            with st.spinner("🔎 Searching medical documents..."):
                result = run_rag_pipeline(
                    question,
                    mode=mode
                )

            answer = result.get(
                "answer",
                "No answer was generated."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception:
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
