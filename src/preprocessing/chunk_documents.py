from pathlib import Path
import json


PROCESSED_DIR = Path("data/processed")
CHUNKED_DIR = Path("data/processed/chunks")

CHUNK_SIZE = 500
OVERLAP = 50


def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_json_files():

    CHUNKED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_files = list(PROCESSED_DIR.glob("*.json"))

    print("JSON files found:", len(json_files))

    for json_file in json_files:

        print("\nProcessing:", json_file.name)

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            document = json.load(f)

        all_chunks = []

        for page in document["pages"]:

            page_chunks = create_chunks(
                page["text"],
                CHUNK_SIZE,
                OVERLAP
            )

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1
            ):

                all_chunks.append({
                    "source": document["source"],
                    "page": page["page"],
                    "chunk_id": chunk_number,
                    "text": chunk
                })

        output_file = CHUNKED_DIR / (
            json_file.stem + "_chunks.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                all_chunks,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(
            "Chunks created:",
            len(all_chunks)
        )

        print(
            "Saved:",
            output_file
        )


if __name__ == "__main__":
    process_json_files()