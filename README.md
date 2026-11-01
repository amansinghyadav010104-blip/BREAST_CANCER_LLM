# Breast Cancer Clinical Support and Patient Education RAG

A research prototype that retrieves evidence from a local breast-cancer document collection and, when local evidence is insufficient, searches a configured list of trusted medical domains before generating an evidence-grounded response with Gemini.

## Current repository contents
- Raw PDFs from NCI, WHO, ASCO, and research papers.
- Extracted/cleaned page-level JSON and precomputed 500-word chunks with 50-word overlap.
- Sentence-transformer embedding generation and FAISS index-building scripts.
- A Streamlit interface with Patient/Doctor modes, Gemini generation, citations, and medical safety guardrails.
- Local FAISS retrieval, trusted-domain web fallback, and session-only PDF, TXT, and DOCX evidence uploads.
- English-only, Hindi-only, and bilingual answers with browser-based English/Hindi speech input and answer playback.

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
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.6-flash
REQUEST_BUDGET_SECONDS=60
GEMINI_TIMEOUT_SECONDS=45
TAVILY_API_KEY=your_tavily_api_key
```

`REQUEST_BUDGET_SECONDS` limits the full safety, retrieval, and answer-generation
request. `GEMINI_TIMEOUT_SECONDS` caps an individual Gemini request; the overall
request budget takes precedence. The configured model is attempted once, then
the application can fail over once to `GEMINI_FALLBACK_MODEL` for transient
timeouts, rate limits, or provider failures. Both models use the Gemini API.
`google-genai` is pinned to the tested 2.28.0 SDK release, and `LLM_PROVIDER`
is fixed to `gemini`.

## Build the local knowledge base
Run commands from the repository root:

```powershell
python -m src.preprocessing.process_documents
python -m src.preprocessing.chunk_documents
python -m src.embeddings.create_embeddings
python -m src.embeddings.build_index
```

The last two steps create `data/processed/embeddings/` and `vector_db/index.faiss` plus `vector_db/metadata.json`. These generated artifacts are ignored by Git, so run the steps when they are not already present locally.

## Run
```powershell
streamlit run app/streamlit_app.py
```

Choose Patient or Doctor mode and an answer language. Uploaded files are extracted and searched in the current Streamlit session only; they are not written to disk or added to the persistent FAISS index. PDF page numbers and original upload names are retained in citations. TXT and DOCX citations include a section location when available.

Voice input uses the browser's Web Speech API and requires microphone permission in a compatible browser. Hindi recognition and playback depend on the browser and installed speech services/voices. Typed questions and on-screen answers remain available when speech features are unavailable. Image-only/scanned PDFs are not OCR processed and may yield no evidence.

Run the automated tests with:

```powershell
python -m unittest discover -s tests -v
```

On Ubuntu, create and activate the environment with `python3 -m venv .venv`,
then run `source .venv/bin/activate`, `python -m pip install --upgrade pip`,
`pip install -r requirements.txt`, and `streamlit run app/streamlit_app.py`.

## Offline RAG evaluation
Run the fixed 39-question evaluation set from the repository root:

```powershell
python evaluation/run.py
```

The cases and source-level relevance labels are in `evaluation/cases.json`. The
runner reports retrieval recall/MRR, context/source traceability, exact support
for deterministic evidence-extract fallback claims, simulated timeout/provider
fallback behavior, safety outcomes, and stage timings. It uses the existing
FAISS index and the configured embedding model already cached locally; it does
not rebuild the index, read `.env`, call Gemini or web search, or download a
model. It fails with an explicit message if the configured model is not cached.
Expected sources are evaluation labels, not clinical gold answers. The fallback
support score checks exact quoted text only and does not validate generated
Gemini answers or establish clinical faithfulness.

Profile local cold and warm stages in separate processes:

```powershell
python evaluation/stage_profile.py --cold-runs 3 --warm-runs 5
```

This uses the already-cached model and index, makes no provider requests, and
does not flush operating-system file caches. Cold totals include imports and
should be interpreted as startup observations on the current machine.

For up to three live end-to-end Gemini requests, with at most one additional
Gemini citation-support judging request per generated answer, run:

```powershell
python evaluation/live.py --limit 3
```

This uses configured credentials but never prints them, response text, or raw
provider errors. The report gives answer latency, timeout/provider-error and
fallback counts, citation-ID traceability against actual retrieved passages,
and model-judged claim-support results. The judge uses the same provider/model,
so its result is not independent evidence or clinical validation. Avoid running
this if provider requests are not permitted or may incur unacceptable costs.

## Important research limitations
- The safety checks are keyword-based and are not a substitute for clinical review or a validated medical safety system.
- Local relevance uses a fixed similarity threshold; this needs evaluation against a labeled query set.
- Local retrieval searches 10 candidates and places at most five unique-source passages in the generation context. The offline evaluation reports corpus-source recall, but source-level labels do not establish clinical relevance or answer correctness.
- Web fallback is limited to the configured domains and depends on API availability and returned search snippets.
- The original documents are preserved; evidence and questions are filtered to the breast-cancer scope, but users should review uploaded sources before relying on them.
- No authentication or audit logging is included.
- This prototype is for educational/research use only, not diagnosis or treatment decisions.

## Suggested next development milestones
1. Build the embedding/index artifacts and run end-to-end smoke tests.
2. Add a labeled retrieval/answer evaluation set (recall@k, MRR, citation correctness, groundedness, refusal/safety cases).
3. Review the language, scope, and safety behavior with qualified breast-cancer clinicians before any clinical use.
