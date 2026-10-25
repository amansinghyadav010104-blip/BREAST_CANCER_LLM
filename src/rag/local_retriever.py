from pathlib import Path
import json
import time

import faiss
import numpy as np
import streamlit as st
from .uploaded_retriever import rank_uploaded_documents


# -----------------------------------------
# Paths
# -----------------------------------------

VECTOR_DB_DIR = Path("vector_db")

INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
METADATA_PATH = VECTOR_DB_DIR / "metadata.json"


# -----------------------------------------
# Embedding model
# -----------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import try_to_load_from_cache

    local_modules = try_to_load_from_cache(MODEL_NAME, "modules.json")
    local_files_only = isinstance(local_modules, str) and Path(local_modules).is_file()
    return SentenceTransformer(MODEL_NAME, local_files_only=local_files_only)


@st.cache_resource(show_spinner=False)
def get_faiss_index():
    index = faiss.read_index(str(INDEX_PATH))
    return index


@st.cache_resource(show_spinner=False)
def get_document_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata


@st.cache_resource(show_spinner=False)
def get_vector_store():
    index = get_faiss_index()
    metadata = get_document_metadata()
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata have different record counts.")
    return index, metadata


@st.cache_data(show_spinner=False, max_entries=512, ttl=3600)
def get_query_embedding(query_key, _model):
    """Cache deterministic local query embeddings for repeated questions."""
    return np.asarray(_model.encode([query_key]), dtype="float32")


# -----------------------------------------
# Local retrieval function
# -----------------------------------------

def retrieve_documents_with_timings(query, top_k=5):
    timings = {}
    model_start = time.perf_counter()
    model = get_embedding_model()
    timings["model_initialization"] = time.perf_counter() - model_start

    # Normalize whitespace and case so harmless input formatting does not
    # create duplicate cached embeddings. Embedding and retrieval use the same
    # normalized form, keeping cache identity and vector identity aligned.
    query_key = " ".join(query.split()).casefold()
    embedding_start = time.perf_counter()
    query_embedding = get_query_embedding(query_key, model).copy()
    faiss.normalize_L2(query_embedding)
    timings["embedding"] = time.perf_counter() - embedding_start

    index_load_start = time.perf_counter()
    index = get_faiss_index()
    timings["index_load"] = time.perf_counter() - index_load_start

    document_load_start = time.perf_counter()
    metadata = get_document_metadata()
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata have different record counts.")
    timings["document_load"] = time.perf_counter() - document_load_start

    search_start = time.perf_counter()
    scores, indices = index.search(
        query_embedding,
        top_k
    )
    timings["faiss_search"] = time.perf_counter() - search_start

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

    return results, query_embedding[0], timings


def retrieve_documents(query, top_k=5):
    results, _, _ = retrieve_documents_with_timings(query, top_k=top_k)
    return results


def retrieve_uploaded_documents(query, documents, top_k=5, query_embedding=None):
    return rank_uploaded_documents(
        query,
        documents,
        encode=get_embedding_model().encode if query_embedding is None else None,
        query_embedding=query_embedding,
        top_k=top_k,
    )


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
