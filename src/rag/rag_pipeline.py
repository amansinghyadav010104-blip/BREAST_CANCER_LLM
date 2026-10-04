
import time

from .hybrid_retriever import hybrid_retrieve
from .llm_generator import generate_answer
from .safety_guardrails import check_safety
from .user_mode import UserMode


def run_rag_pipeline(question, mode=UserMode.PATIENT):

    total_start = time.perf_counter()
    timings = {}

    print("\n" + "=" * 60)
    print("BREAST CANCER HYBRID RAG + GENAI")
    print("=" * 60)

    print("\nUser Question:", question)
    print("Selected Mode:", mode.value)

    # Step 0: Safety check
    start = time.perf_counter()
    safety_result = check_safety(question)
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
            "answer": safety_result["message"],
            "context": "",
            "timings": timings
        }

    # Step 1: Hybrid retrieval
    print("\nSearching medical documents...")
    start = time.perf_counter()

    retrieval_result = hybrid_retrieve(question)

    timings["retrieval"] = time.perf_counter() - start

    print(f"Retrieval time: {timings['retrieval']:.2f}s")

    # Step 2: Get retrieved context
    context = retrieval_result["context"]

    # Step 3: Generate answer using Gemini
    print("\nGenerating answer with Gemini...")
    start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        mode=mode
    )

    timings["generation"] = time.perf_counter() - start

    print(f"Generation time: {timings['generation']:.2f}s")

    # Total time
    timings["total"] = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Safety Check: {timings['safety']:.2f}s")
    print(f"Retrieval:    {timings['retrieval']:.2f}s")
    print(f"Generation:   {timings['generation']:.2f}s")
    print(f"Total:        {timings['total']:.2f}s")

    return {
        "question": question,
        "mode": mode.value,
        "source": retrieval_result["source"],
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

import time

from .hybrid_retriever import hybrid_retrieve
from .llm_generator import generate_answer
from .safety_guardrails import check_safety
from .user_mode import UserMode


def run_rag_pipeline(question, mode=UserMode.PATIENT):

    total_start = time.perf_counter()
    timings = {}

    print("\n" + "=" * 60)
    print("BREAST CANCER HYBRID RAG + GENAI")
    print("=" * 60)

    print("\nUser Question:", question)
    print("Selected Mode:", mode.value)

    # Step 0: Safety check
    start = time.perf_counter()
    safety_result = check_safety(question)
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
            "answer": safety_result["message"],
            "context": "",
            "timings": timings
        }

    # Step 1: Hybrid retrieval
    print("\nSearching medical documents...")
    start = time.perf_counter()

    retrieval_result = hybrid_retrieve(question)

    timings["retrieval"] = time.perf_counter() - start

    print(f"Retrieval time: {timings['retrieval']:.2f}s")

    # Step 2: Get retrieved context
    context = retrieval_result["context"]

    # Step 3: Generate answer using Gemini
    print("\nGenerating answer with Gemini...")
    start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        mode=mode
    )

    timings["generation"] = time.perf_counter() - start

    print(f"Generation time: {timings['generation']:.2f}s")

    # Total time
    timings["total"] = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Safety Check: {timings['safety']:.2f}s")
    print(f"Retrieval:    {timings['retrieval']:.2f}s")
    print(f"Generation:   {timings['generation']:.2f}s")
    print(f"Total:        {timings['total']:.2f}s")

    return {
        "question": question,
        "mode": mode.value,
        "source": retrieval_result["source"],
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
