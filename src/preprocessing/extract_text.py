from pathlib import Path

from pypdf import PdfReader
from .clean_text import clean_text


def extract_pdf_text(pdf_path):
    """
    Extract and clean text from a PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        raw_text = page.extract_text()

        if raw_text:

            cleaned_text = clean_text(raw_text)

            pages.append({
                "page": page_number,
                "text": cleaned_text
            })

    return pages


if __name__ == "__main__":

    pdf_path = Path(
        "data/raw/nci/Treatment for Breast Cancer - NCI.pdf"
    )

    pages = extract_pdf_text(pdf_path)

    print("Number of pages:", len(pages))

    for page in pages:

        print("\n")
        print("=" * 60)
        print(f"PAGE {page['page']}")
        print("=" * 60)

        print(page["text"][:2000])