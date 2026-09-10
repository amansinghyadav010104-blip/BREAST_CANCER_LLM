from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# Paths
# -----------------------------------------

VECTOR_DB_DIR = Path("vector_db")

INDEX_PATH = VECTOR_DB_DIR / "breast_cancer.index"
METADATA_PATH = VECTOR_DB_DIR / "metadata.json"


# -----------------------------------------
# Embedding model
# -----------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# -----------------------------------------
# Load FAISS database
# -----------------------------------------

index = faiss.read_index(
    str(INDEX_PATH)
)


# -----------------------------------------
# Load metadata
# -----------------------------------------

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# -----------------------------------------
# Local retrieval function
# -----------------------------------------

def retrieve_documents(query, top_k=5):

    # Convert question to embedding
    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(
        query_embedding
    )

    # Search vector database
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        # Ignore invalid FAISS indexes
        if idx < 0:
            continue

        result = metadata[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# -----------------------------------------
# Test Local Retriever
# -----------------------------------------

if __name__ == "__main__":

    print("Local Breast Cancer Retriever")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    )

    results = retrieve_documents(
        query,
        top_k=5
    )

    print("\n")
    print("=" * 60)
    print("LOCAL RETRIEVAL RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n")
        print(f"RESULT {i}")
        print("-" * 60)

        print(
            "Similarity Score:",
            round(result["score"], 4)
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "Chunk ID:",
            result["chunk_id"]
        )

        print("\nText:")

        print(result["text"])