import numpy as np

from .cancer_scope import is_breast_cancer_related


UPLOAD_RELEVANCE_THRESHOLD = 0.30


def embed_uploaded_chunks(chunks, encode):
    if not chunks:
        return []
    vectors = np.asarray(encode([chunk["text"] for chunk in chunks]), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return [
        {**chunk, "embedding": vector.tolist()}
        for chunk, vector in zip(chunks, vectors)
    ]


def rank_uploaded_documents(
    query,
    documents,
    encode=None,
    top_k=5,
    query_embedding=None,
):
    """Rank session-only uploaded chunks without changing the persistent FAISS index."""
    documents_by_source = {}
    for document in documents:
        documents_by_source.setdefault(document.get("source", ""), []).append(document)

    scoped_sources = {
        source
        for source, source_documents in documents_by_source.items()
        if is_breast_cancer_related(
            " ".join(item.get("text", "") for item in source_documents), source
        )
    }
    eligible = [item for item in documents if item.get("source", "") in scoped_sources]
    if not eligible:
        return []

    if all("embedding" in item for item in eligible):
        document_vectors = np.asarray(
            [item["embedding"] for item in eligible], dtype="float32"
        )
    else:
        if encode is None:
            raise ValueError("An embedding encoder is required for unembedded uploads.")
        document_vectors = np.asarray(
            encode([item["text"] for item in eligible]), dtype="float32"
        )

    if query_embedding is None:
        if encode is None:
            raise ValueError("An embedding encoder is required for the query.")
        query_embedding = np.asarray(encode([query])[0], dtype="float32")
    query_embedding = np.asarray(query_embedding, dtype="float32").reshape(-1)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm:
        query_embedding = query_embedding / query_norm

    document_norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    document_vectors = np.divide(
        document_vectors,
        document_norms,
        out=np.zeros_like(document_vectors),
        where=document_norms != 0,
    )
    scores = document_vectors @ query_embedding

    ranked = []
    for index in np.argsort(scores)[::-1][:top_k]:
        score = float(scores[index])
        if score >= UPLOAD_RELEVANCE_THRESHOLD:
            result = eligible[index].copy()
            result["score"] = score
            ranked.append(result)
    return ranked