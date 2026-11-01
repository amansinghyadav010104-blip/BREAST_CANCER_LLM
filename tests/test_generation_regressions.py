import unittest
import asyncio
import heapq
import socket
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


class _TimerOnlyEventLoop(asyncio.SelectorEventLoop):
    """Run coroutine timers without Windows' socketpair-based loop wakeup."""

    def _make_self_pipe(self):
        self._ssock = self._csock = None

    def _write_to_self(self):
        # This test schedules work only on the loop thread.
        pass

    def _close_self_pipe(self):
        pass

    def _run_once(self):
        while self._scheduled and self._scheduled[0]._cancelled:
            heapq.heappop(self._scheduled)._scheduled = False
        timeout = 0.0 if self._ready else (
            max(0.0, self._scheduled[0]._when - self.time())
            if self._scheduled else 0.001
        )
        if timeout:
            time.sleep(timeout)
        end_time = self.time() + self._clock_resolution
        while self._scheduled and self._scheduled[0]._when < end_time:
            handle = heapq.heappop(self._scheduled)
            handle._scheduled = False
            if not handle._cancelled:
                self._ready.append(handle)
        for _ in range(len(self._ready)):
            handle = self._ready.popleft()
            if not handle._cancelled:
                handle._run()


class _HttpStatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__("sanitized test response")
        self.status_code = status_code


class GenerationRegressionTests(unittest.TestCase):
    def setUp(self):
        provider_patch = patch.object(llm_generator, "LLM_PROVIDER", "gemini")
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

    def test_pipeline_and_generator_share_the_configured_request_budget(self):
        self.assertEqual(
            rag_pipeline.REQUEST_BUDGET_SECONDS,
            llm_generator.REQUEST_BUDGET_SECONDS,
        )

    def test_empty_response_is_reported_as_generation_failure(self):
        with patch.object(llm_generator.client.models, "generate_content", return_value=type("R", (), {"text": " "})()):
            with self.assertRaises(AnswerGenerationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_provider_timeout_is_classified(self):
        with patch.object(llm_generator.client.models, "generate_content", side_effect=TimeoutError()):
            with self.assertRaises(ProviderTimeoutError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_http_504_is_classified_as_provider_timeout(self):
        with patch.object(
            llm_generator.client.models, "generate_content", side_effect=_HttpStatusError(504)
        ):
            with self.assertRaises(ProviderTimeoutError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_provider_5xx_is_classified_without_leaking_raw_error(self):
        error = _HttpStatusError(503)
        self.assertEqual(llm_generator._provider_error_category(error), "provider_failure")
        with patch.object(llm_generator.client.models, "generate_content", side_effect=error):
            with self.assertRaisesRegex(AnswerGenerationError, "Gemini could not generate"):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_transient_primary_failure_uses_one_gemini_model_fallback(self):
        response = type("R", (), {"text": "Generated answer. [Evidence 1]"})()
        with patch.object(
            llm_generator.client.models,
            "generate_content",
            side_effect=[_HttpStatusError(503), response],
        ) as generate:
            answer = llm_generator.generate_answer("What is breast cancer?", CONTEXT)

        self.assertIn("Generated answer", answer)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in generate.call_args_list],
            [llm_generator.GEMINI_MODEL, llm_generator.GEMINI_FALLBACK_MODEL],
        )

    def test_invalid_api_key_does_not_retry_with_fallback(self):
        error = _HttpStatusError(401)
        with patch.object(
            llm_generator.client.models, "generate_content", side_effect=error
        ) as generate:
            with self.assertRaises(AnswerGenerationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(llm_generator._provider_error_category(error), "authentication_failure")

    def test_dns_failure_is_classified_and_allows_only_configured_fallback(self):
        error = socket.gaierror("temporary name resolution failure")
        self.assertEqual(llm_generator._provider_error_category(error), "dns_failure")
        self.assertTrue(llm_generator._should_failover(error))
        self.assertFalse(llm_generator._should_failover(ApplicationDeadlineError("expired")))

    def test_deadline_stops_waiting_for_slow_provider(self):
        started = time.perf_counter()
        class SlowInteractions:
            def __init__(self):
                self.cancelled = False

            async def generate_content(self, **kwargs):
                try:
                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        class FakeClient:
            aio = type("Aio", (), {"models": SlowInteractions()})()

        fake_client = FakeClient()
        loop = _TimerOnlyEventLoop()
        with self.assertRaises(ProviderTimeoutError):
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    llm_generator._await_generation_response(fake_client, {}, 0.25, True)
                )
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        self.assertLess(time.perf_counter() - started, 0.25)
        self.assertTrue(fake_client.aio.models.cancelled)

    def test_request_is_not_started_without_cleanup_budget(self):
        deadline = time.perf_counter() + 2.0
        with (
            patch.object(llm_generator.genai, "Client") as client_constructor,
            self.assertRaises(ApplicationDeadlineError),
        ):
            llm_generator._create_generation_response("prompt", deadline=deadline)

        client_constructor.assert_not_called()

    def test_authentication_error_does_not_escape_as_raw_provider_exception(self):
        with patch.object(llm_generator.client.models, "generate_content", side_effect=_HttpStatusError(401)):
            with self.assertRaises(AnswerGenerationError):
                llm_generator.generate_answer("What is breast cancer?", CONTEXT)

    def test_success_requires_genuine_evidence_citation_and_records_timings(self):
        response = type("R", (), {"output_text": "## Answer\nBreast cancer begins when breast cells grow out of control. [Evidence 1]"})()
        timings = {}
        with patch.object(llm_generator.client.models, "generate_content", return_value=response):
            answer = llm_generator.generate_answer(
                "What is breast cancer?", CONTEXT, timings=timings
            )
        self.assertIn("[Evidence 1]", answer)
        self.assertIn("provider_generation", timings)
        self.assertIn("validation", timings)

    def test_fabricated_citation_is_rejected(self):
        response = type("R", (), {"output_text": "Breast cancer is a disease. [Evidence 9]"})()
        with patch.object(llm_generator.client.models, "generate_content", return_value=response):
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

    def test_fallback_deduplicates_claims_and_source_page_references(self):
        from src.rag.rag_pipeline import _evidence_extract

        sentence = "Breast cancer is a disease in which cells grow out of control."
        context = (
            "[EVIDENCE 1]\nSource: Breast cancer fact sheet.pdf\nPage: 1\nContent:\n"
            f"{sentence}\n"
            "[EVIDENCE 2]\nSource: Breast cancer fact sheet.pdf\nPage: 1\nContent:\n"
            f"{sentence}"
        )
        answer = _evidence_extract(context, AnswerLanguage.ENGLISH, "What is breast cancer?")

        self.assertEqual(answer.count(sentence), 1)
        self.assertEqual(answer.count("Breast cancer fact sheet.pdf, 1"), 1)
        self.assertIn("[Evidence 1]", answer)
        self.assertIn("[Evidence 2]", answer)

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
        with patch.object(llm_generator.client.models, "generate_content", return_value=response):
            answer = llm_generator.generate_answer(
                "स्तन कैंसर क्या है?", CONTEXT,
                answer_language=AnswerLanguage.HINDI,
            )
        self.assertIn("[Evidence 1]", answer)


if __name__ == "__main__":
    unittest.main()
