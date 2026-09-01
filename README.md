# DeepVision

A local-first research and study assistant for academic papers.

Give it an arXiv ID or your own PDF. It runs a seven-stage pipeline — download,
text extraction, figure extraction, OCR, vision analysis, embeddings, report —
and produces an interactive, cited study guide you can read, chat with, and
revise from with flashcards and quizzes.

Everything runs on your machine. Papers, vectors and study progress live in a
local SQLite database and a local Chroma store; no account, no cloud sync.

---

## What it does

**Reads figures, not just text.** Every page is rendered and every embedded
image extracted, then passed through OCR and a vision model, so charts, tables
and diagrams become searchable content rather than gaps in the text.

**Writes a report for someone new to the topic.** Eleven fixed sections — At a
Glance, Overview, Background, Key Concepts, Methods, Key Results, Figures,
Limitations & Open Questions, Why It Matters, Key Takeaways, Study Questions —
each answering exactly one question, so nothing is said twice. Claims carry
inline citations that resolve to the page and passage they came from, and a
figure referenced in the prose renders as the actual image.

**Answers questions about the paper.** Chat routes each question before it
retrieves: citations, metadata and structure are answered directly from the
paper's record, and only genuine content questions go through retrieval. Answers
are grounded in the paper and cite what they used.

**Generates study material from the whole paper.** Flashcards and quizzes are
built from a sweep of the entire document, not just the report, so regenerating
a deck surfaces material you have not seen. Quizzes mix multiple choice,
true/false and short answer across three difficulty tiers, with per-question
explanations and server-side grading.

**Schedules review by session, not by date.** Three ratings — *Again*, *Almost*,
*Got it* — decide where a card goes in the current sitting. There are no due
dates and no intervals; how you rate a card orders your next session, so the
cards you are worst at come first.

**Reads a chapter at a time.** Long PDFs are split by their embedded outline,
and a report can be scoped to a single chapter's page range.

---

## Tech stack

| Layer | Built with |
|---|---|
| Backend | Python 3.11+, FastAPI, SQLModel, Pydantic v2 |
| Frontend | React 18, TypeScript, Vite |
| Storage | SQLite (metadata, reports, study state), ChromaDB (vectors) |
| PDF / vision | PyMuPDF, Pillow, Tesseract OCR |
| Models | Ollama (local) or the Anthropic / OpenAI APIs |

---

## Quickstart

**Requirements:** Python 3.11+, Node 18+, and [Tesseract](https://github.com/tesseract-ocr/tesseract)
for OCR (`brew install tesseract` on macOS).

```bash
git clone <your-repo-url> && cd DeepVision

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd frontend && npm install && cd ..

./start.sh            # backend :8000, frontend :5173
```

Open <http://localhost:5173>. `./stop.sh` stops both; logs are written to
`backend.log` and `frontend.log`.

Running the two services by hand instead:

```bash
.venv/bin/python -m uvicorn deepvision.api.main:app --port 8000
cd frontend && npm run dev
```

---

## Choosing models

Nothing is bundled — pick one of these in **Settings**.

**Local (free, private, slower).** Install [Ollama](https://ollama.com) and pull
a model:

```bash
ollama pull llama3.1:8b
```

A full report takes around 20 minutes on `llama3.1:8b`, so the **Concise**
detail level is recommended locally. Vision and embedding models download from
Hugging Face to `~/.cache` on first use.

**API (fast, costs money).** Add an Anthropic or OpenAI key in Settings. Reports
finish in under a minute. The key is stored only in your local database, which
is gitignored and never leaves your machine.

The app degrades rather than breaks: if a model call fails or times out, the
affected section falls back to text extracted directly from the paper and is
flagged in the UI as extracted rather than written.

---

## Layout

```
deepvision/
  models/       Pydantic contracts shared across every layer
  db/           SQLModel tables and session handling
  providers/    LLM / vision / embedding interfaces + local and API adapters
  ingestion/    Download, parse, OCR, vision, and the 7-stage orchestrator
  rag/          Chunking, vector store, retrieval, chunk-quality filtering
  agents/       Research, summarizer, media, citation, synthesis, flashcard, quiz
  study/        Session scheduler, deck generation, grading
  report/       Report assembly, citation formatting, figure linking, export
  api/          FastAPI app, routers, request/response schemas
frontend/       React + TypeScript single-page app
scripts/        smoke_check.py — offline end-to-end check
```

---

## Verifying a change

`smoke_check.py` runs the entire pipeline against a generated fixture PDF using
offline stubs, so it needs no models, no network and no API key:

```bash
.venv/bin/python scripts/smoke_check.py     # 141 checks
cd frontend && npm run build                # typecheck + production build
```

---

## Configuration

Copy `.env.example` to `.env` to override defaults — data directory, database
path, request timeouts. Every variable is prefixed `DEEPVISION_`.

Your database (`deepvision/data/deepvision.db`), downloaded PDFs, extracted
images and vector store are all gitignored.
