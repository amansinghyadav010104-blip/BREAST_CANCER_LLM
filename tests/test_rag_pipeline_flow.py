import unittest
from unittest.mock import patch

from src.rag.answer_language import AnswerLanguage
from src.rag import hybrid_retriever, llm_generator, rag_pipeline
from src.rag.llm_generator import AnswerGenerationError, QueryTranslationError
from src.rag.local_retriever import MODEL_NAME, get_embedding_model, get_vector_store


class RagPipelineFlowTests(unittest.TestCase):
    def test_faiss_resources_are_cached_between_reruns(self):
        first = get_vector_store()
        second = get_vector_store()

        self.assertIs(first, second)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_query_embedding_is_cached_for_repeated_question(self):
        import numpy as np

        class Model:
            def __init__(self):
                self.calls = 0

            def encode(self, questions):
                self.calls += 1
                return np.asarray([[1.0, 0.0]], dtype="float32")

        model = Model()
        from src.rag.local_retriever import get_query_embedding
        get_query_embedding.clear()
        try:
            first = get_query_embedding("what is breast cancer?", model)
            second = get_query_embedding("what is breast cancer?", model)

            self.assertEqual(model.calls, 1)
            np.testing.assert_array_equal(first, second)
        finally:
            get_query_embedding.clear()

    def test_cached_model_load_avoids_hub_metadata_requests(self):
        get_embedding_model.clear()
        cached_model = object()
        try:
            with (
                patch(
                    "huggingface_hub.try_to_load_from_cache",
                    return_value="C:/models/snapshot/modules.json",
                ),
                patch("pathlib.Path.is_file", return_value=True),
                patch(
                    "sentence_transformers.SentenceTransformer",
                    return_value=cached_model,
                ) as constructor,
            ):
                first = get_embedding_model()
                second = get_embedding_model()

            self.assertIs(first, cached_model)
            self.assertIs(first, second)
            constructor.assert_called_once_with(MODEL_NAME, local_files_only=True)
        finally:
            get_embedding_model.clear()

    def test_typo_is_corrected_for_retrieval_without_changing_other_text(self):
        with (
            patch.object(
                hybrid_retriever,
                "retrieve_documents_with_timings",
                return_value=([], [1.0, 0.0], {}),
            ) as retrieve,
            patch.object(hybrid_retriever, "retrieve_uploaded_documents", return_value=[]),
            patch.object(hybrid_retriever, "web_search", return_value=[]),
        ):
            hybrid_retriever.hybrid_retrieve("what is brest cancer")

        self.assertEqual(retrieve.call_args.args[0], "what is breast cancer")

    def test_hindi_question_is_translated_only_for_retrieval(self):
        interaction = type("Interaction", (), {"output_text": "breast cancer symptoms"})()
        with patch.object(llm_generator.client.interactions, "create", return_value=interaction):
            translated = llm_generator.translate_query_for_retrieval("स्तन कैंसर के लक्षण क्या हैं?")

        self.assertEqual(translated, "breast cancer symptoms")
        self.assertEqual(
            llm_generator.translate_query_for_retrieval("What are breast cancer symptoms?"),
            "What are breast cancer symptoms?",
        )

    def test_hindi_treatment_question_is_translated_for_retrieval(self):
        interaction = type(
            "Interaction", (), {"output_text": "breast cancer treatment options"}
        )()
        question = "स्तन कैंसर के उपचार के विकल्प क्या हैं?"
        with patch.object(
            llm_generator.client.interactions, "create", return_value=interaction
        ) as create:
            translated = llm_generator.translate_query_for_retrieval(question)

        self.assertEqual(translated, "breast cancer treatment options")
        self.assertEqual(
            create.call_args.kwargs["timeout"],
            llm_generator.GENAI_REQUEST_TIMEOUT_SECONDS,
        )

    def test_hindi_translation_is_cached_only_in_supplied_session_cache(self):
        interaction = type("Interaction", (), {"output_text": "breast cancer symptoms"})()
        cache = {}
        with patch.object(
            llm_generator.client.interactions, "create", return_value=interaction
        ) as create:
            first = llm_generator.translate_query_for_retrieval(
                "स्तन कैंसर के लक्षण?", translation_cache=cache
            )
            second = llm_generator.translate_query_for_retrieval(
                "स्तन कैंसर के लक्षण?", translation_cache=cache
            )

        self.assertEqual(first, second)
        create.assert_called_once()

    def test_relevant_upload_is_used_without_web_fallback(self):
        upload = {
            "source": "breast-notes.pdf",
            "page": 2,
            "section": None,
            "chunk_id": 1,
            "text": "Breast cancer evidence about treatment options.",
        }
        ranked_upload = {**upload, "score": 0.8}
        with (
            patch.object(
                hybrid_retriever,
                "retrieve_documents_with_timings",
                return_value=([], [1.0, 0.0], {}),
            ),
            patch.object(
                hybrid_retriever,
                "retrieve_uploaded_documents",
                return_value=[ranked_upload],
            ),
            patch.object(hybrid_retriever, "web_search") as web_search,
        ):
            result = hybrid_retriever.hybrid_retrieve(
                "What are breast cancer treatment options?", [upload]
            )

        self.assertTrue(result["has_evidence"])
        self.assertEqual(result["source"], "upload")
        self.assertIn("breast-notes.pdf", result["context"])
        self.assertIn("Page: 2", result["context"])
        web_search.assert_not_called()

    def test_local_retrieval_uses_ten_candidates_and_limits_context_to_distinct_sources(self):
        candidates = [
            {"source": "Breast cancer A.pdf", "page": 1, "text": "Breast cancer first A passage", "score": 0.9},
            {"source": "Breast cancer A.pdf", "page": 2, "text": "Breast cancer second A passage", "score": 0.8},
            {"source": "Breast cancer B.pdf", "page": 1, "text": "Breast cancer first B passage", "score": 0.7},
            {"source": "Breast cancer C.pdf", "page": 1, "text": "Breast cancer first C passage", "score": 0.6},
            {"source": "Breast cancer D.pdf", "page": 1, "text": "Breast cancer first D passage", "score": 0.5},
            {"source": "Breast cancer E.pdf", "page": 1, "text": "Breast cancer first E passage", "score": 0.4},
        ]
        with (
            patch.object(
                hybrid_retriever,
                "retrieve_documents_with_timings",
                return_value=(candidates, [1.0, 0.0], {}),
            ) as retrieve,
            patch.object(hybrid_retriever, "retrieve_uploaded_documents", return_value=[]),
            patch.object(hybrid_retriever, "web_search"),
        ):
            result = hybrid_retriever.hybrid_retrieve("What is breast cancer?")

        self.assertEqual(retrieve.call_args.kwargs["top_k"], 10)
        self.assertEqual(
            [item["source"] for item in result["local_results"]],
            [
                "Breast cancer A.pdf",
                "Breast cancer B.pdf",
                "Breast cancer C.pdf",
                "Breast cancer D.pdf",
                "Breast cancer E.pdf",
            ],
        )
        self.assertLessEqual(len(result["local_results"]), 5)
        self.assertEqual(hybrid_retriever.select_diverse_results(candidates, limit=0), [])

    def test_empty_retrieval_returns_localized_insufficiency_without_gemini(self):
        with (
            patch.object(
                rag_pipeline,
                "hybrid_retrieve",
                return_value={
                    "context": "No relevant evidence was retrieved.",
                    "has_evidence": False,
                    "source": "none",
                },
            ),
            patch.object(rag_pipeline, "generate_answer") as generate_answer,
        ):
            result = rag_pipeline.run_rag_pipeline(
                "What are the treatment options for breast cancer?",
                answer_language=AnswerLanguage.HINDI,
            )

        self.assertEqual(result["source"], "insufficient_evidence")
        self.assertIn("पर्याप्त नहीं", result["answer"])
        generate_answer.assert_not_called()

    def test_general_education_questions_reach_generation_with_citations(self):
        evidence = {
            "context": "[EVIDENCE 1]\nSource: NCI\nContent: Breast cancer treatment evidence.",
            "has_evidence": True,
            "source": "local",
        }
        questions = [
            ("What is breast cancer?", AnswerLanguage.ENGLISH),
            ("What are the treatment options for breast cancer?", AnswerLanguage.ENGLISH),
            ("What factors affect breast cancer treatment?", AnswerLanguage.ENGLISH),
            ("स्तन कैंसर के उपचार के विकल्प क्या हैं?", AnswerLanguage.HINDI),
        ]

        with (
            patch.object(rag_pipeline, "hybrid_retrieve", return_value=evidence),
            patch.object(
                rag_pipeline,
                "generate_answer",
                return_value="## Answer\nEvidence-based response. [Evidence 1]",
            ) as generate_answer,
        ):
            for question, language in questions:
                with self.subTest(question=question.encode("unicode_escape").decode()):
                    result = rag_pipeline.run_rag_pipeline(
                        question, answer_language=language
                    )

                    self.assertEqual(result["source"], "local")
                    self.assertIn("[Evidence 1]", result["answer"])
                    self.assertIn("[EVIDENCE 1]", result["context"])
                    self.assertEqual(result["generation_status"], "generated")

        self.assertEqual(generate_answer.call_count, len(questions))

    def test_generation_failure_returns_localized_message_and_keeps_evidence(self):
        evidence = {
            "context": "[EVIDENCE 1]\nSource: NCI\nContent: Breast cancer treatment evidence describes several possible approaches.",
            "has_evidence": True,
            "source": "local",
        }
        with (
            patch.object(rag_pipeline, "hybrid_retrieve", return_value=evidence),
            patch.object(
                rag_pipeline,
                "generate_answer",
                side_effect=AnswerGenerationError("provider unavailable"),
            ),
        ):
            result = rag_pipeline.run_rag_pipeline(
                "What are the treatment options for breast cancer?"
            )

        self.assertEqual(result["source"], "evidence_extract")
        self.assertIn("Evidence-based extract", result["answer"])
        self.assertIn("[EVIDENCE 1]", result["context"])
        self.assertEqual(result["generation_status"], "fallback")
        self.assertEqual(result["generation_failure"], "provider_error")

    def test_hindi_translation_failure_returns_insufficiency_without_crashing(self):
        with (
            patch.object(
                rag_pipeline,
                "hybrid_retrieve",
                side_effect=QueryTranslationError("translation unavailable"),
            ),
            patch.object(rag_pipeline, "generate_answer") as generate_answer,
        ):
            result = rag_pipeline.run_rag_pipeline(
                "स्तन कैंसर के उपचार के विकल्प क्या हैं?",
                answer_language=AnswerLanguage.HINDI,
            )

        self.assertEqual(result["source"], "insufficient_evidence")
        self.assertIn("पर्याप्त नहीं", result["answer"])
        generate_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
