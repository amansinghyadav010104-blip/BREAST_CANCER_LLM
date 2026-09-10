from pathlib import Path
import json

from extract_text import extract_pdf_text


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def process_all_pdfs():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(RAW_DIR.rglob("*.pdf"))

    print("PDF files found:", len(pdf_files))

    for pdf_path in pdf_files:

        print("\nProcessing:", pdf_path)

        pages = extract_pdf_text(pdf_path)

        document = {
            "source": pdf_path.name,
            "path": str(pdf_path),
            "pages": pages
        }

        output_name = pdf_path.stem + ".json"

        output_path = PROCESSED_DIR / output_name

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                document,
                f,
                ensure_ascii=False,
                indent=4
            )

        print("Saved:", output_path)


if __name__ == "__main__":
    process_all_pdfs()