from pathlib import Path
import json

import faiss
import numpy as np


# -----------------------------------------
# Paths
# -----------------------------------------

EMBEDDINGS_DIR = Path("data/processed/embeddings")
VECTOR_DB_DIR = Path("vector_db")


# -----------------------------------------
# Build FAISS database
# -----------------------------------------

def build_vector_database():

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_files = list(
        EMBEDDINGS_DIR.glob("*_embeddings.json")
    )

    print("Embedding files found:", len(embedding_files))

    all_embeddings = []
    metadata = []

    # -----------------------------------------
    # Read all embedding files
    # -----------------------------------------

    for embedding_file in embedding_files:

        print("\nLoading:", embedding_file.name)

        with open(
            embedding_file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        for chunk in chunks:

            all_embeddings.append(
                chunk["embedding"]
            )

            metadata.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            })


    # -----------------------------------------
    # Convert embeddings to NumPy
    # -----------------------------------------

    embeddings = np.array(
        all_embeddings,
        dtype="float32"
    )

    print("\nTotal embeddings:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])


    # -----------------------------------------
    # Normalize embeddings
    # -----------------------------------------

    faiss.normalize_L2(embeddings)


    # -----------------------------------------
    # Create FAISS index
    # Inner Product + normalized vectors
    # = cosine similarity
    # -----------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)


    # -----------------------------------------
    # Save FAISS index
    # -----------------------------------------

    index_path = VECTOR_DB_DIR / "breast_cancer.index"

    faiss.write_index(
        index,
        str(index_path)
    )


    # -----------------------------------------
    # Save metadata
    # -----------------------------------------

    metadata_path = VECTOR_DB_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )


    # -----------------------------------------
    # Final information
    # -----------------------------------------

    print("\nVector database created successfully!")

    print("Total vectors:", index.ntotal)

    print(
        "FAISS index:",
        index_path
    )

    print(
        "Metadata:",
        metadata_path
    )


if __name__ == "__main__":

    build_vector_database()