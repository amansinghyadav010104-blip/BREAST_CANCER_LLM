from pathlib import Path
import json

import faiss
import numpy as np


# -----------------------------------------
# Paths
# -----------------------------------------

EMBEDDINGS_DIR = Path("data/processed/embeddings")
VECTOR_DB_DIR = Path("vector_db")

INDEX_FILE = VECTOR_DB_DIR / "index.faiss"
METADATA_FILE = VECTOR_DB_DIR / "metadata.json"


def build_faiss_index():

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_files = list(
        EMBEDDINGS_DIR.glob("*_embeddings.json")
    )

    print("Embedding files found:", len(embedding_files))

    if not embedding_files:
        print("No embedding files found.")
        print("Run create_embeddings.py first.")
        return

    all_embeddings = []
    metadata = []

    for embedding_file in embedding_files:

        print("\nProcessing:", embedding_file.name)

        with open(
            embedding_file,
            "r",
            encoding="utf-8"
        ) as f:

            records = json.load(f)

        for record in records:

            all_embeddings.append(
                record["embedding"]
            )

            metadata.append({
                "source": record["source"],
                "page": record["page"],
                "chunk_id": record["chunk_id"],
                "text": record["text"]
            })

    # Convert embeddings to NumPy array
    embeddings = np.array(
        all_embeddings,
        dtype="float32"
    )

    print("\nEmbedding matrix shape:", embeddings.shape)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index using inner product
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("Total vectors added:", index.ntotal)

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFAISS index saved:", INDEX_FILE)
    print("Metadata saved:", METADATA_FILE)


if __name__ == "__main__":

    build_faiss_index()