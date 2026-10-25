import unittest
from unittest.mock import patch

from src.rag.answer_language import (
    AnswerLanguage,
    AnswerLanguageGenerationError,
    get_language_instructions,
    validate_answer_language,
)
from src.rag.cancer_scope import check_cancer_scope, filter_breast_cancer_evidence
from src.rag import llm_generator
from src.rag.safety_guardrails import check_safety, insufficient_evidence_message
from src.rag.user_mode import UserMode


class SafetyAndLanguageTests(unittest.TestCase):
    def test_breast_cancer_scope_and_unrelated_cancers(self):
        self.assertTrue(check_cancer_scope("What is breast cancer?")["in_scope"])
        self.assertTrue(check_cancer_scope("what is brest cancer")["in_scope"])
        self.assertTrue(check_safety("what is brest cancer")["safe"])
        unrelated = check_cancer_scope("What are lung cancer symptoms?")
        self.assertFalse(unrelated["in_scope"])
        self.assertEqual(unrelated["reason"], "unrelated_cancer")
        self.assertFalse(
            check_cancer_scope("Compare breast cancer and lung cancer")["in_scope"]
        )
        self.assertFalse(check_cancer_scope("How do I bake bread?")["in_scope"])

    def test_evidence_filter_excludes_unrelated_documents(self):
        results = [
            {"source": "breast.txt", "text": "Breast cancer evidence."},
            {"source": "lung.txt", "text": "Lung cancer evidence."},
        ]

        self.assertEqual(filter_breast_cancer_evidence(results), results[:1])

    def test_answer_language_instructions_cover_all_choices(self):
        english = get_language_instructions(AnswerLanguage.ENGLISH)
        hindi = get_language_instructions(AnswerLanguage.HINDI)
        bilingual = get_language_instructions(AnswerLanguage.BILINGUAL)

        self.assertIn("English only", english)
        self.assertIn("हिंदी", hindi)
        self.assertIn("अंग्रेज़ी", bilingual)

    def test_language_validator_accepts_only_requested_output_sections(self):
        english = "## Answer\nBreast cancer is a disease. [Evidence 1]"
        hindi = "### उत्तर\nस्तन कैंसर एक बीमारी है। [Evidence 1]"
        bilingual = (
            "## English\nBreast cancer is a disease. [Evidence 1]\n\n"
            "## हिंदी\nस्तन कैंसर एक बीमारी है। [Evidence 1]"
        )

        self.assertEqual(validate_answer_language(english, AnswerLanguage.ENGLISH), english)
        self.assertEqual(validate_answer_language(hindi, AnswerLanguage.HINDI), hindi)
        self.assertEqual(validate_answer_language(bilingual, AnswerLanguage.BILINGUAL), bilingual)
        with self.assertRaises(AnswerLanguageGenerationError):
            validate_answer_language(english, AnswerLanguage.HINDI)
        with self.assertRaises(AnswerLanguageGenerationError):
            validate_answer_language(hindi, AnswerLanguage.BILINGUAL)

    def test_invalid_language_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            get_language_instructions("French")

    def test_generator_raises_instead_of_returning_english_for_hindi(self):
        interaction = type(
            "Interaction", (), {"output_text": "Breast cancer is a disease. [Evidence 1]"}
        )()
        with patch.object(
            llm_generator.client.interactions, "create", return_value=interaction
        ):
            with self.assertRaises(AnswerLanguageGenerationError):
                llm_generator.generate_answer(
                    "स्तन कैंसर क्या है?",
                    "[EVIDENCE 1] Breast cancer is a disease.",
                    answer_language=AnswerLanguage.HINDI,
                )

    def test_doctor_mode_instructions_reach_gemini(self):
        interaction = type(
            "Interaction", (), {"output_text": "## Answer\nBreast cancer is a disease. [Evidence 1]"}
        )()
        with patch.object(
            llm_generator.client.interactions, "create", return_value=interaction
        ) as create:
            llm_generator.generate_answer(
                "What is breast cancer?",
                "[EVIDENCE 1] Breast cancer is a disease.",
                mode=UserMode.DOCTOR,
                answer_language=AnswerLanguage.ENGLISH,
            )

        prompt = create.call_args.kwargs["input"]
        self.assertIn("Clinical Support Mode", prompt)
        self.assertIn("medically precise and professional language", prompt)

    def test_hindi_diagnosis_and_medication_requests_are_blocked(self):
        diagnosis = check_safety("क्या मुझे स्तन कैंसर है?", AnswerLanguage.HINDI)
        medication = check_safety("दवा की खुराक कितनी होनी चाहिए?", AnswerLanguage.BILINGUAL)

        self.assertEqual(diagnosis["risk_type"], "diagnosis")
        self.assertIn("निदान", diagnosis["message"])
        self.assertEqual(medication["risk_type"], "medication")
        self.assertIn("dose", medication["message"])
        self.assertIn("खुराक", medication["message"])

    def test_personal_treatment_decisions_are_blocked_in_hindi(self):
        result = check_safety("मेरा इलाज बदलना चाहिए?", AnswerLanguage.HINDI)

        self.assertEqual(result["risk_type"], "treatment_decision")
        self.assertIn("उपचार", result["message"])

    def test_general_breast_cancer_treatment_questions_are_allowed(self):
        questions = [
            "What are the treatment options for breast cancer?",
            "What factors affect breast cancer treatment?",
            "What questions can I ask my oncologist about breast cancer treatment?",
            "स्तन कैंसर के उपचार के विकल्प क्या हैं?",
        ]

        for question in questions:
            with self.subTest(question=question.encode("unicode_escape").decode()):
                self.assertTrue(check_safety(question)["safe"])

    def test_personalized_treatment_selection_is_blocked(self):
        result = check_safety("What treatment should I choose for my breast cancer?")

        self.assertFalse(result["safe"])
        self.assertEqual(result["risk_type"], "treatment_decision")

    def test_unrelated_question_is_blocked_by_pipeline_safety_check(self):
        result = check_safety("What are the treatment options for lung cancer?")

        self.assertFalse(result["safe"])
        self.assertEqual(result["risk_type"], "unrelated_cancer")

    def test_unspecified_cancer_diagnosis_uses_diagnosis_guardrail(self):
        result = check_safety("Do I have cancer?")

        self.assertEqual(result["risk_type"], "diagnosis")

    def test_unrelated_cancer_and_insufficient_evidence_messages_localize(self):
        unrelated = check_safety("Tell me about prostate cancer", AnswerLanguage.BILINGUAL)

        self.assertEqual(unrelated["risk_type"], "unrelated_cancer")
        self.assertIn("breast cancer", unrelated["message"])
        self.assertIn("स्तन कैंसर", unrelated["message"])
        self.assertIn("पर्याप्त नहीं", insufficient_evidence_message(AnswerLanguage.HINDI))


if __name__ == "__main__":
    unittest.main()
