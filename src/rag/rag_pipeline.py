from hybrid_retriever import hybrid_retrieve
from llm_generator import generate_answer
from safety_guardrails import check_safety
from user_mode import UserMode


def run_rag_pipeline(question, mode=UserMode.PATIENT):
    """
    Complete Hybrid RAG + GenAI pipeline.

    Question
        ↓
    Safety Guardrail
        ↓
    Hybrid Retrieval
        ↓
    Local FAISS / Trusted Web
        ↓
    Context
        ↓
    Gemini LLM with Patient/Doctor Mode
        ↓
    Grounded Answer
    """

    print("\n" + "=" * 60)
    print("BREAST CANCER HYBRID RAG + GENAI")
    print("=" * 60)

    print("\nUser Question:")
    print(question)

    print("\nSelected Mode:")
    print(mode.value)

    # Step 0: Safety check
    safety_result = check_safety(question)

    if not safety_result["safe"]:
        print("\nSafety Guardrail Triggered")
        print("Risk Type:", safety_result["risk_type"])

        return {
            "question": question,
            "mode": mode.value,
            "source": "safety_guardrail",
            "answer": safety_result["message"],
            "context": ""
        }

    # Step 1: Hybrid retrieval
    retrieval_result = hybrid_retrieve(question)

    # Step 2: Get retrieved context
    context = retrieval_result["context"]

    # Step 3: Generate answer using Gemini
    print("\nGenerating answer with Gemini...")

    answer = generate_answer(
        question,
        context,
        mode=mode
    )

    return {
        "question": question,
        "mode": mode.value,
        "source": retrieval_result["source"],
        "answer": answer,
        "context": context
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

    question = input(
        "\nEnter your breast cancer question: "
    )

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