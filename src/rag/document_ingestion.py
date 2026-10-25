from io import BytesIO
import hashlib
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from .uploaded_retriever import embed_uploaded_chunks


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def parse_uploaded_document(filename, content):
    """Extract text sections while retaining source and available locations."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Upload a PDF, TXT, or DOCX file.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Each uploaded file must be 20 MB or smaller.")

    sections = []
    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    sections.append({"page": page_number, "section": None, "text": text})
        except Exception as error:
            raise ValueError("The PDF could not be read.") from error
    elif extension == ".txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("TXT files must use UTF-8 encoding.") from error
        if text:
            sections.append({"page": None, "section": "Text file", "text": text})
    else:
        try:
            document = Document(BytesIO(content))
            for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
                text = paragraph.text.strip()
                if text:
                    sections.append({
                        "page": None,
                        "section": f"Paragraph {paragraph_number}",
                        "text": text,
                    })
            for table_number, table in enumerate(document.tables, start=1):
                for row_number, row in enumerate(table.rows, start=1):
                    text = " | ".join(cell.text.strip() for cell in row.cells).strip(" |")
                    if text:
                        sections.append({
                            "page": None,
                            "section": f"Table {table_number}, row {row_number}",
                            "text": text,
                        })
        except Exception as error:
            raise ValueError("The DOCX file could not be read.") from error

    if not sections:
        raise ValueError("No selectable text was found in the uploaded file.")
    return sections


def chunk_uploaded_document(filename, content, chunk_size=300, overlap=40):
    """Split extracted sections into evidence records without losing locations."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    chunks = []
    for section in parse_uploaded_document(filename, content):
        words = section["text"].split()
        start = 0
        chunk_number = 1
        while start < len(words):
            text = " ".join(words[start:start + chunk_size])
            if text:
                chunks.append({
                    "source": filename,
                    "page": section["page"],
                    "section": section["section"],
                    "chunk_id": chunk_number,
                    "text": text,
                })
                chunk_number += 1
            start += chunk_size - overlap
    return chunks


def get_or_embed_uploaded_chunks(filename, content, session_cache, encode):
    """Reuse session-scoped embedded chunks or parse/embed a newly changed file."""
    digest = f"{filename}:{hashlib.sha256(content).hexdigest()}"
    if digest in session_cache:
        return digest, session_cache[digest]
    chunks = chunk_uploaded_document(filename, content)
    return digest, embed_uploaded_chunks(chunks, encode)