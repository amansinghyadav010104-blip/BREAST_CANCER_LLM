import io
import unittest

from docx import Document

from src.rag.document_ingestion import (
    chunk_uploaded_document,
    get_or_embed_uploaded_chunks,
    parse_uploaded_document,
)


def make_text_pdf(text):
    stream = f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


class DocumentIngestionTests(unittest.TestCase):
    def test_extracts_pdf_text_and_preserves_page(self):
        sections = parse_uploaded_document("breast.pdf", make_text_pdf("Breast cancer evidence"))

        self.assertEqual(sections[0]["page"], 1)
        self.assertIn("Breast cancer evidence", sections[0]["text"])

    def test_extracts_utf8_txt(self):
        sections = parse_uploaded_document("notes.txt", "स्तन कैंसर".encode("utf-8"))

        self.assertIsNone(sections[0]["page"])
        self.assertEqual(sections[0]["text"], "स्तन कैंसर")

    def test_extracts_docx_paragraphs_and_tables(self):
        document = Document()
        document.add_paragraph("Breast cancer evidence")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "ER positive"
        table.cell(0, 1).text = "Evidence detail"
        stream = io.BytesIO()
        document.save(stream)

        sections = parse_uploaded_document("notes.docx", stream.getvalue())

        self.assertEqual(sections[0]["section"], "Paragraph 1")
        self.assertEqual(sections[1]["section"], "Table 1, row 1")

    def test_chunks_retain_filename_and_page(self):
        chunks = chunk_uploaded_document("breast.pdf", make_text_pdf("Breast cancer evidence"))

        self.assertEqual(chunks[0]["source"], "breast.pdf")
        self.assertEqual(chunks[0]["page"], 1)

    def test_rejects_unsupported_and_empty_files(self):
        with self.assertRaises(ValueError):
            parse_uploaded_document("notes.csv", b"content")
        with self.assertRaises(ValueError):
            parse_uploaded_document("notes.txt", b"")

    def test_session_upload_cache_does_not_reembed_unchanged_file(self):
        encoder_calls = []

        def encode(texts):
            encoder_calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

        cache = {}
        content = b"Breast cancer presentation test fact."
        key, chunks = get_or_embed_uploaded_chunks("session.txt", content, cache, encode)
        cache[key] = chunks
        reused_key, reused_chunks = get_or_embed_uploaded_chunks(
            "session.txt", content, cache, encode
        )

        self.assertEqual(key, reused_key)
        self.assertIs(chunks, reused_chunks)
        self.assertEqual(len(encoder_calls), 1)


if __name__ == "__main__":
    unittest.main()