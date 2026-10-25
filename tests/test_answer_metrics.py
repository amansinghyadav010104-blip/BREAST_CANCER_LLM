import unittest

from evaluation.answer_metrics import citation_traceability, cited_claims, evidence_by_id


CONTEXT = (
    "[EVIDENCE 1]\nSource: NCI.pdf\nPage: 2\nContent:\n"
    "Breast cancer treatment can include surgery.\n"
    "[EVIDENCE 2]\nTitle: WHO breast cancer\nURL: https://www.who.int/example\n"
    "Content:\nBreast cancer screening can help detect disease early.\n"
)


class AnswerMetricTests(unittest.TestCase):
    def test_evidence_map_retains_actual_source_location_and_url(self):
        evidence = evidence_by_id(CONTEXT)

        self.assertEqual(evidence["1"]["source"], "NCI.pdf")
        self.assertEqual(evidence["1"]["location"], "2")
        self.assertEqual(evidence["2"]["source"], "WHO breast cancer")
        self.assertEqual(evidence["2"]["url"], "https://www.who.int/example")

    def test_cited_claims_omit_source_list(self):
        claims = cited_claims(
            "## Answer\nTreatment can include surgery. [Evidence 1]\n"
            "### Sources\n- NCI.pdf [Evidence 1]"
        )

        self.assertEqual(
            claims,
            [{"claim": "Treatment can include surgery.", "evidence_ids": ["1"]}],
        )

    def test_traceability_detects_fabricated_evidence_id(self):
        metrics = citation_traceability(
            "Treatment can include surgery. [Evidence 1]\n"
            "A second claim. [Evidence 9]",
            CONTEXT,
        )

        self.assertEqual(metrics["citation_count"], 2)
        self.assertEqual(metrics["valid_citation_count"], 1)
        self.assertEqual(metrics["valid_citation_rate"], 0.5)

    def test_uncited_text_does_not_count_as_cited_claim(self):
        metrics = citation_traceability("This answer has no evidence citation.", CONTEXT)

        self.assertEqual(metrics["claim_count"], 0)
        self.assertIsNone(metrics["valid_citation_rate"])


if __name__ == "__main__":
    unittest.main()
