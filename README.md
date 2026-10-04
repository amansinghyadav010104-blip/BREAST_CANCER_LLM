# Breast Cancer Clinical Support and Patient Education RAG

A research prototype that retrieves evidence from a local breast-cancer document collection and, when local evidence is insufficient, searches a configured list of trusted medical domains before generating an evidence-grounded response with Gemini.

## Current repository contents
- Raw PDFs from NCI, WHO, ASCO, and research papers.
- Extracted/cleaned page-level JSON and precomputed 500-word chunks with 50-word overlap.
- Sentence-transformer embedding generation and FAISS index-building scripts.
- Local retrieval, relevance routing, trusted-domain web retrieval, evidence context construction, Gemini answer generation, patient/doctor response modes, and basic medical safety guardrails.

## Setup
Use Python 3.10+ in a virtual environment.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the repository root (never commit it):
```dotenv
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Build the local knowledge base
Run commands from the repository root:

```powershell
python -m src.preprocessing.process_documents
python -m src.preprocessing.chunk_documents
python -m src.embeddings.create_embeddings
python -m src.embeddings.build_index
```

The last two steps create `data/processed/embeddings/` and `vector_db/index.faiss` plus `vector_db/metadata.json`. The uploaded archive includes processed JSON and chunks, but does not include embeddings or a built FAISS index, so these steps are required before retrieval.

## Run
```powershell
python -m src.rag.rag_pipeline
```
Choose Patient or Doctor mode and enter a question. The pipeline is a command-line prototype; no web app or voice interface is included in this archive.

## Important research limitations
- The safety checks are keyword-based and are not a substitute for clinical review or a validated medical safety system.
- Local relevance uses a fixed similarity threshold; this needs evaluation against a labeled query set.
- Web fallback is limited to the configured domains and depends on API availability and returned search snippets.
- No automated tests, benchmark/evaluation dataset, UI, authentication, or audit logging are included in this archive.
- This prototype is for educational/research use only, not diagnosis or treatment decisions.

## Suggested next development milestones
1. Build the embedding/index artifacts and run end-to-end smoke tests.
2. Add a labeled retrieval/answer evaluation set (recall@k, MRR, citation correctness, groundedness, refusal/safety cases).
3. Improve query routing and safety checks with explicit tests and clinical review.
4. Add structured source display and a Streamlit interface only after the core pipeline is reproducibly tested.
