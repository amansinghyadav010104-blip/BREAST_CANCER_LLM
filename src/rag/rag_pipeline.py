from hybrid_retriever import hybrid_retrieve
from llm_generator import generate_answer
from safety_guardrails import check_safety


def run_rag_pipeline(question):
    """
    Complete Hybrid RAG + GenAI pipeline.

    Question
        ↓
    Hybrid Retrieval
        ↓
    Local FAISS / Trusted Web
        ↓
    Context
        ↓
    Gemini LLM
        ↓
    Grounded Answer
    """

    print("\n" + "=" * 60)
    print("BREAST CANCER HYBRID RAG + GENAI")
    print("=" * 60)

    print("\nUser Question:")
    print(question)
    
        # Step 0: Safety check
    safety_result = check_safety(question)

    if not safety_result["safe"]:
        print("\nSafety Guardrail Triggered")
        print("Risk Type:", safety_result["risk_type"])

        return {
            "question": question,
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
        context
    )

    return {
        "question": question,
        "source": retrieval_result["source"],
        "answer": answer,
        "context": context
    }


if __name__ == "__main__":

    question = input(
        "\nEnter your breast cancer question: "
    )

    result = run_rag_pipeline(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("EVIDENCE SOURCE")
    print("=" * 60)

    print(result["source"])