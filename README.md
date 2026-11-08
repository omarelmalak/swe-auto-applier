# AutoApply AI

Autonomous, human-in-the-loop job application agent. Phase 1 focuses on the local knowledge base + form filler; Discord queue, scraping, and EC2 come later.

## Setup

```bash
cd swe-auto-applier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set COHERE_API_KEY
```

Always use the project venv (`source .venv/bin/activate` or prefix commands with `.venv/bin/python`). Homebrew/system `python3` will not see these packages.

---

## Commands by module

### `core/knowledge_base.py`

FAISS index over `data/resume.pdf` + `data/qa_bank.json` using Cohere `embed-english-v3.0`.

```bash
# Rebuild the index from scratch (resume PDF + qa_bank.json)
# Also writes data/faiss_index/chunk_preview.html — open it to see each chunk boxed
python -m core.knowledge_base rebuild
open data/faiss_index/chunk_preview.html

# Regenerate the HTML preview from the saved index (no Cohere call)
python -m core.knowledge_base preview

# Semantic search (default top 3 — not the whole resume)
python -m core.knowledge_base search "HubSpot internship"

# Custom top-k + optional weak-hit filter
python -m core.knowledge_base search "email" -k 2
python -m core.knowledge_base search "work authorization" -k 3 --min-score 0.35
```

Chunking is **section/role-aware**: contact, Skills, Education, and each job
(title + dates + all bullets) are separate FAISS vectors so a role is never split.

Examples of useful queries:

```bash
python -m core.knowledge_base search "email phone linkedin"
python -m core.knowledge_base search "University of Toronto GPA"
python -m core.knowledge_base search "Scotiabank frontend bullets"
python -m core.knowledge_base search "Python FastAPI React skills"
```

---

### `core/dom_parser.py`

Opens a job apply URL with Playwright, isolates form controls, minifies the HTML
(strips scripts/styles/class noise), and detects whether the current step’s CTA is
**Next/Continue** or **Submit/Apply**.

```bash
# Headless parse — prints JSON summary (fields + navigation)
python -m core.dom_parser "https://boards.greenhouse.io/example/jobs/123"

# Show the browser while parsing
python -m core.dom_parser "https://jobs.lever.co/example/abcd" --headed

# Write full JSON payload and/or minified HTML to disk
python -m core.dom_parser "https://example.com/apply" --out /tmp/form.json --html-out /tmp/form.min.html
```

Navigation `kind` is one of `next` | `submit` | `unknown`. When both Next and Submit
exist on the page, **Next wins** (multi-step wizard).

Unit tests (no network / no browser):

```bash
pytest tests/test_dom_parser.py -v
```

---

### `config.py`

No CLI. Loaded automatically via `python-dotenv` from `.env`.

Relevant env vars (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `COHERE_API_KEY` | Embeddings + chat |
| `COHERE_EMBEDDING_MODEL` | Default `embed-english-v3.0` |
| `COHERE_CHAT_MODEL` | Default `command-r-plus` (LLM router, upcoming) |
| `EMBEDDING_DIM` | Default `1024` |

---

### Tests

```bash
# Knowledge-base unit tests (no live Cohere calls; fake embeddings)
pytest tests/test_knowledge_base.py -v

# DOM parser unit tests (static HTML; no Playwright launch)
pytest tests/test_dom_parser.py -v

# All tests
pytest -v
```

After `pip install -r requirements.txt`, install browser binaries once:

```bash
playwright install chromium
```

---

## Upcoming (Phase 1 — not implemented yet)

```bash
# RAG + Cohere → JSON selector/answer map
# python -m core.llm_router "<job_apply_url>"

# Visual Chrome fill + multi-step Next loop; pause before Submit
# python modes/manual_apply.py "<job_apply_url>"
```

---

## Data files

| Path | Role |
| --- | --- |
| `data/resume.pdf` | Upload artifact + source text for FAISS |
| `data/qa_bank.json` | Learned Q&A pairs (`[]` until Discord corrections) |
| `data/faiss_index/index.faiss` | Persisted FAISS vectors (gitignored) |
| `data/faiss_index/metadata.json` | Chunk text sidecar (gitignored) |
| `data/faiss_index/chunk_preview.html` | Visual board of each chunk in a colored box (gitignored) |
