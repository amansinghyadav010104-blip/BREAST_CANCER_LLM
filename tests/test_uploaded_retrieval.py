import unittest
from unittest.mock import Mock

from src.rag.context_builder import build_context, extract_evidence_sources
from src.rag.hybrid_retriever import normalize_query, select_diverse_results
from src.rag.uploaded_retriever import embed_uploaded_chunks, rank_uploaded_documents


class UploadedRetrievalTests(unittest.TestCase):
    def test_query_normalization_keeps_medical_terms_and_cleans_punctuation(self):
        self.assertEqual(
            normalize_query("  What IS brest-cancer?  "),
            "what is breast cancer",
        )

    def test_diversity_filter_removes_near_duplicate_chunks_across_pages(self):
        results = [
            {"source": "NCI.pdf", "page": 1, "text": "Breast cancer cells grow out of control in the breast."},
            {"source": "NCI.pdf", "page": 2, "text": "Breast cancer cells grow out of control in the breast."},
            {"source": "WHO.pdf", "page": 4, "text": "Breast cancer begins when abnormal cells grow."},
        ]
        selected = select_diverse_results(results, limit=3)
        self.assertEqual(len(selected), 2)
        self.assertEqual([item["source"] for item in selected], ["NCI.pdf", "WHO.pdf"])

    def test_ranks_relevant_session_upload_chunks_and_excludes_other_cancers(self):
        chunks = [
            {"source": "notes.pdf", "page": 3, "text": "Breast cancer evidence"},
            {"source": "lung.pdf", "page": 1, "text": "Lung cancer evidence"},
        ]

        def encode(texts):
            return [[1.0, 0.0] if "Breast" in text or "question" in text else [0.0, 1.0]
                    for text in texts]

        results = rank_uploaded_documents("question", chunks, encode)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "notes.pdf")
        self.assertEqual(results[0]["score"], 1.0)

    def test_uploaded_evidence_carries_file_page_and_citation_id(self):
        context = build_context(uploaded_results=[{
            "source": "my evidence.pdf",
            "page": 4,
            "section": None,
            "score": 0.8,
            "text": "Breast cancer evidence",
        }])

        self.assertIn("[EVIDENCE 1]", context)
        self.assertIn("my evidence.pdf", context)
        self.assertIn("Page: 4", context)
        self.assertIn("Breast cancer evidence", context)

    def test_context_keeps_local_and_uploaded_sources_separate(self):
        context = build_context(
            local_results=[{
                "source": "NCI breast cancer.pdf",
                "page": 2,
                "score": 0.8,
                "text": "Local breast cancer evidence",
            }],
            uploaded_results=[{
                "source": "presentation-notes.txt",
                "page": None,
                "section": "Text file",
                "score": 0.9,
                "text": "Uploaded breast cancer test fact",
            }],
        )

        self.assertIn("LOCAL KNOWLEDGE BASE EVIDENCE", context)
        self.assertIn("SESSION UPLOAD EVIDENCE", context)
        self.assertIn("[EVIDENCE 1]", context)
        self.assertIn("[EVIDENCE 2]", context)
        self.assertIn("NCI breast cancer.pdf", context)
        self.assertIn("presentation-notes.txt", context)
        self.assertIn("Location: Text file", context)

    def test_context_deduplicates_identical_passages_but_preserves_distinct_pages(self):
        text = "Breast cancer evidence about treatment options."
        context = build_context(local_results=[
            {"source": "NCI.pdf", "page": 2, "score": 0.9, "text": text},
            {"source": "NCI.pdf", "page": 2, "score": 0.8, "text": text},
            {"source": "NCI.pdf", "page": 3, "score": 0.7, "text": text},
        ])

        self.assertEqual(context.count("[EVIDENCE "), 2)
        self.assertIn("Page: 2", context)
        self.assertIn("Page: 3", context)

    def test_evidence_sources_map_each_id_to_retrieved_source_and_location(self):
        context = build_context(
            local_results=[{
                "source": "NCI.pdf",
                "page": 3,
                "score": 0.8,
                "text": "Breast cancer evidence",
            }],
            uploaded_results=[{
                "source": "notes.docx",
                "page": None,
                "section": "Paragraph 4",
                "score": 0.9,
                "text": "Uploaded breast cancer evidence",
            }],
            web_results=[{
                "title": "NCI Breast Cancer",
                "url": "https://www.cancer.gov/example",
                "score": 0.9,
                "content": "Trusted breast cancer evidence",
            }],
        )

        self.assertEqual(
            extract_evidence_sources(context),
            [
                {"evidence_id": "1", "evidence_ids": ["1"], "source": "NCI.pdf", "location": "Page 3"},
                {"evidence_id": "2", "evidence_ids": ["2"], "source": "notes.docx", "location": "Paragraph 4"},
                {
                    "evidence_id": "3",
                    "evidence_ids": ["3"],
                    "source": "NCI Breast Cancer",
                    "url": "https://www.cancer.gov/example",
                },
            ],
        )

    def test_source_list_deduplicates_filename_page_and_keeps_evidence_ids(self):
        context = build_context(local_results=[
            {"source": "NCI.pdf", "page": 3, "score": 0.9, "text": "Breast cancer evidence about screening."},
            {"source": "NCI.pdf", "page": 3, "score": 0.8, "text": "Breast cancer evidence about diagnosis."},
        ])
        sources = extract_evidence_sources(context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["evidence_ids"], ["1", "2"])

    def test_embedded_session_chunks_are_reused_without_reencoding(self):
        encoder = Mock(return_value=[[3.0, 4.0]])
        chunks = [{"source": "breast.txt", "page": None, "text": "Breast cancer fact"}]
        embedded = embed_uploaded_chunks(chunks, encoder)
        encoder.reset_mock()

        results = rank_uploaded_documents(
            "question", embedded, query_embedding=[0.6, 0.8]
        )

        encoder.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["embedding"][0], 0.6)
        self.assertAlmostEqual(results[0]["embedding"][1], 0.8)


if __name__ == "__main__":
    unittest.main()
