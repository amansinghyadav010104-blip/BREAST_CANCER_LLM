from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


# -----------------------------------------
# Paths
# -----------------------------------------

CHUNKS_DIR = Path("data/processed/chunks")
OUTPUT_DIR = Path("data/processed/embeddings")


# -----------------------------------------
# Embedding model
# -----------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading embedding model...")
    
    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    json_files = list(
        CHUNKS_DIR.glob("*_chunks.json")
    )

    print("Chunk files found:", len(json_files))


    for json_file in json_files:

        print("\nProcessing:", json_file.name)

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)


        texts = [
            chunk["text"]
            for chunk in chunks
        ]


        print("Generating embeddings...")

        embeddings = model.encode(
            texts,
            show_progress_bar=True
        )


        output_data = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            output_data.append({

                "source": chunk["source"],

                "page": chunk["page"],

                "chunk_id": chunk["chunk_id"],

                "text": chunk["text"],

                "embedding": embedding.tolist()

            })


        output_file = OUTPUT_DIR / (
            json_file.stem + "_embeddings.json"
        )


        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output_data,
                f,
                ensure_ascii=False
            )


        print(
            "Saved:",
            output_file
        )


if __name__ == "__main__":

    create_embeddings()