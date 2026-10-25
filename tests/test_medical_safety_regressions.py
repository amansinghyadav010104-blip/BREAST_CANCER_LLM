import unittest

from src.rag.safety_guardrails import check_safety


class MedicalSafetyRegressionTests(unittest.TestCase):
    def test_personal_breast_lump_diagnosis_question_is_blocked(self):
        result = check_safety("Could this new breast lump be cancer?")

        self.assertFalse(result["safe"])
        self.assertEqual(result["risk_type"], "diagnosis")

    def test_general_breast_lump_education_remains_allowed(self):
        result = check_safety("Can breast lumps be benign?")

        self.assertTrue(result["safe"])

    def test_personal_hindi_lump_diagnosis_question_is_blocked(self):
        result = check_safety("क्या यह नई स्तन की गांठ कैंसर हो सकती है?")

        self.assertFalse(result["safe"])
        self.assertEqual(result["risk_type"], "diagnosis")


if __name__ == "__main__":
    unittest.main()
