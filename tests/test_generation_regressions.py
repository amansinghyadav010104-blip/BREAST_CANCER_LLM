import unittest
import asyncio
import time
from unittest.mock import patch

from src.rag import llm_generator, rag_pipeline
from src.rag.answer_language import AnswerLanguage
from src.rag.llm_generator import (
    AnswerGenerationError,
    ApplicationDeadlineError,
    CitationValidationError,
    ProviderTimeoutError,
)


CONTEXT = "[EVIDENCE 1]\nSource: NCI\nPage: 2\nContent:\nBreast cancer begins when cells in the breast grow out of control."


class GenerationRegressionTests(unittest.TestCase):
    def test_empty_response_is_reported_as_generation_failure(self):
        with patch.object(llm_generator.client.interactions, "create", return_value=type("R", (), {"output_text": " "})()):
            with self.assertRaises(AnswerGenerationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_provider_timeout_is_classified(self):
        with patch.object(llm_generator.client.interactions, "create", side_effect=TimeoutError()):
            with self.assertRaises(ProviderTimeoutError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_deadline_stops_waiting_for_slow_provider(self):
        started = time.perf_counter()
        class SlowInteractions:
            async def create(self, **kwargs):
                await asyncio.sleep(0.3)

        class FakeClient:
            aio = type("Aio", (), {"interactions": SlowInteractions()})()

        with self.assertRaises(ProviderTimeoutError):
            asyncio.run(llm_generator._await_interaction(FakeClient(), {}, 0.25, True))
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_request_is_not_started_without_cleanup_budget(self):
        deadline = time.perf_counter() + 2.0
        with (
            patch.object(llm_generator.genai, "Client") as client_constructor,
            self.assertRaises(ApplicationDeadlineError),
        ):
            llm_generator._create_interaction("prompt", deadline=deadline)

        client_constructor.assert_not_called()

    def test_authentication_error_does_not_escape_as_raw_provider_exception(self):
        with patch.object(llm_generator.client.interactions, "create", side_effect=RuntimeError("401 unauthorized")):
            with self.assertRaises(AnswerGenerationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_success_requires_genuine_evidence_citation_and_records_timings(self):
        response = type("R", (), {"output_text": "## Answer\nBreast cancer begins when breast cells grow out of control. [Evidence 1]"})()
        timings = {}
        with patch.object(llm_generator.client.interactions, "create", return_value=response):
            answer = llm_generator.generate_answer(
                "What is breast cancer?", CONTEXT, timings=timings
            )
        self.assertIn("[Evidence 1]", answer)
        self.assertIn("provider_generation", timings)
        self.assertIn("validation", timings)

    def test_fabricated_citation_is_rejected(self):
        response = type("R", (), {"output_text": "Breast cancer is a disease. [Evidence 9]"})()
        with patch.object(llm_generator.client.interactions, "create", return_value=response):
            with self.assertRaises(CitationValidationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_timeout_fallback_is_clear_and_keeps_context(self):
        evidence = {"context": CONTEXT, "has_evidence": True, "source": "local"}
        with (
            patch.object(rag_pipeline, "hybrid_retrieve", return_value=evidence),
            patch.object(rag_pipeline, "generate_answer", side_effect=ProviderTimeoutError("deadline")),
        ):
            result = rag_pipeline.run_rag_pipeline("What is breast cancer?")
        self.assertEqual(result["source"], "evidence_extract")
        self.assertIn("Evidence-based extract", result["answer"])
        self.assertIn("time limit", result["answer"])
        self.assertIn("[EVIDENCE 1]", result["context"])
        self.assertEqual(result["generation_status"], "fallback")
        self.assertEqual(result["generation_failure"], "provider_timeout")

    def test_expired_end_to_end_deadline_skips_provider_and_records_total(self):
        evidence = {"context": CONTEXT, "has_evidence": True, "source": "local"}
        with (
            patch.object(rag_pipeline, "REQUEST_BUDGET_SECONDS", -1),
            patch.object(rag_pipeline, "hybrid_retrieve", return_value=evidence),
            patch.object(rag_pipeline, "generate_answer") as generate,
        ):
            result = rag_pipeline.run_rag_pipeline("What is breast cancer?")
        self.assertEqual(result["source"], "evidence_extract")
        self.assertIn("[Evidence 1]", result["answer"])
        self.assertGreaterEqual(result["timings"]["total"], 0)
        generate.assert_not_called()

    def test_evidence_fallback_skips_unreadable_pdf_extraction(self):
        from src.rag.rag_pipeline import _evidence_extract

        noisy = "[EVIDENCE 1]\nSource: scan.pdf\nContent:\nDonate © Breast cancer reading time 7 min. Key facts."
        self.assertIsNone(_evidence_extract(noisy, AnswerLanguage.ENGLISH))

    def test_definition_fallback_extracts_readable_claim_after_pdf_header(self):
        from src.rag.rag_pipeline import _evidence_extract

        context = (
            "[EVIDENCE 1]\nSource: Breast cancer fact sheet.pdf\nPage: 1\nContent:\n"
            "Donate © Breast cancer 3 July 2026 Reading time: 7 min. Key facts. "
            "Overview Breast cancer is a disease in which abnormal cells in the breast "
            "grow out of control and form tumours. If left unchecked, the tumours can "
            "spread throughout the body and become fatal."
        )
        answer = _evidence_extract(context, AnswerLanguage.ENGLISH, "What is breast cancer?")
        self.assertIn("Evidence-based extract", answer)
        self.assertIn("Breast cancer is a disease", answer)
        self.assertIn("Breast cancer fact sheet.pdf, 1 [Evidence 1]", answer)

    def test_treatment_fallback_cites_retrieved_general_treatment_passage(self):
        from src.rag.rag_pipeline import _evidence_extract

        context = (
            "[EVIDENCE 1]\nSource: Breast cancer fact sheet.pdf\nPage: 1\nContent:\n"
            "Overview Breast cancer is a disease in which abnormal cells grow out of control. "
            "Treatment is based on the person, the type of cancer and its spread. "
            "Treatment combines surgery, radiation therapy and medications."
        )
        answer = _evidence_extract(
            context, AnswerLanguage.ENGLISH,
            "What are the common treatments for breast cancer?",
        )
        self.assertIn("Treatment is based on the person", answer)
        self.assertIn("Treatment combines surgery", answer)
        self.assertIn("[Evidence 1]", answer)

    def test_hindi_generation_preserves_language_validation_and_citation(self):
        response = type("R", (), {"output_text": "## उत्तर\nस्तन कैंसर तब शुरू होता है जब कोशिकाएँ बढ़ती हैं। [Evidence 1]"})()
        with patch.object(llm_generator.client.interactions, "create", return_value=response):
            answer = llm_generator.generate_answer(
                "स्तन कैंसर क्या है?", CONTEXT,
                answer_language=AnswerLanguage.HINDI,
            )
        self.assertIn("[Evidence 1]", answer)


if __name__ == "__main__":
    unittest.main()
