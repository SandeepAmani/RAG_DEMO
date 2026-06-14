# RAG Chatbot 🤖

A local **Retrieval-Augmented Generation (RAG)** demo that lets you upload PDF documents and ask questions about them through an interactive chat UI.

Built with **Groq LLaMA-3.1-8B** (fast free LLM API) and **HuggingFace MiniLM** (offline embeddings) — no OpenAI key required.

---

## What It Does

```
Your PDF
   ↓
Chunking (split into ~500-char pieces)
   ↓
Embedding (HuggingFace all-MiniLM-L6-v2 → 384-dim vectors, runs offline)
   ↓
Vector Store (in-memory cosine similarity search)
   ↓
Your Question → embed → search → top-3 chunks retrieved
   ↓
Groq LLaMA-3.1-8B generates an answer grounded in those chunks
```

---

## Project Structure

```
RAG_DEMO/
├── streamlit_app.py   # Streamlit chat UI (main entry point)
├── rag_demo.py        # Core RAG pipeline (CLI version + reusable functions)
├── run_app.ps1        # One-click launch script for the UI
├── setup_and_run.ps1  # First-time setup + CLI launcher
├── .env               # Groq API key (not committed to git)
└── .gitignore
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 + |
| sentence-transformers | 5.5.1 |
| numpy | 2.4.6 |
| groq | 1.4.0 |
| pypdf | 6.13.0 |
| python-docx | 1.2.0 |
| streamlit | 1.58.0 |

You also need a free **Groq API key** — get one at [console.groq.com](https://console.groq.com).

---

## Setup

### 1. Clone / download the repo

```powershell
cd wherever\you\keep\projects
# place the RAG_DEMO folder here
```

### 2. Install dependencies

```powershell
pip install sentence-transformers numpy groq pypdf python-docx streamlit
```

Or using the included setup script (also prompts for your API key):

```powershell
.\setup_and_run.ps1
```

### 3. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

The app reads this file automatically — no need to set environment variables manually.  
Alternatively, paste the key directly in the sidebar of the Streamlit UI at runtime.

> Get a free key at [console.groq.com](https://console.groq.com) → Create API Key.

---

## Running the Streamlit UI

```powershell
# Option A — PowerShell helper script
.\run_app.ps1

# Option B — run directly
python -m streamlit run streamlit_app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

## Using the UI

1. **Enter your Groq API key** in the sidebar (auto-loaded from `.env` if present)
2. **Upload a PDF** using the file uploader in the sidebar
3. Adjust **Chunk size** and **Overlap** sliders if needed (defaults work well)
4. Click **⚡ Process Document** — the app reads, chunks, and embeds the PDF
5. **Type a question** in the chat box and press **Send 🚀**
6. The answer appears as a chat bubble; the **Retrieved Chunks** tab on the right shows exactly which passages from your PDF were used

### UI Overview

```
┌──────────────────────────────────────────────────────────────┐
│  🤖 RAG Chat Bot                               [SB Sandeep]  │
│  ✓ API Key  |  ✓ Document Ready  |  💬 3 questions           │
├─────────────────────────────┬────────────────────────────────┤
│  💬 Chat                    │  📑 Document Chunks            │
│                             │  ┌──────────┬───────────────┐  │
│  [user bubble]              │  │All Chunks│Retrieved Chunks│ │
│  [bot bubble]               │  └──────────┴───────────────┘  │
│  ...                        │  📌 Chunk 1  Similarity: 0.72  │
│  ────────────────────────   │  📌 Chunk 2  Similarity: 0.61  │
│  [text area]  [Send 🚀]     │  📌 Chunk 3  Similarity: 0.48  │
└─────────────────────────────┴────────────────────────────────┘

Sidebar:
  🔑 Groq API Key
  📂 Upload PDF
  ⚡ Process Document
  🕘 Chat History (last 20 questions)
```

---

## Running the CLI Version

The original terminal-based pipeline is in `rag_demo.py`. To use it with your own PDF, edit the `EXTERNAL_FILES` list at the top of the file:

```python
EXTERNAL_FILES = [
    "C:\\Users\\you\\Downloads\\your_document.pdf",
]
```

Then run:

```powershell
python rag_demo.py
```

---

## How It Works — Key Components

| Component | What it does | Library |
|---|---|---|
| **Document Loader** | Extracts text from PDF / DOCX files | `pypdf`, `python-docx` |
| **Chunker** | Splits text into overlapping fixed-size pieces | built-in |
| **Embedding Model** | Converts text to 384-dim vectors (runs fully offline after first download ~80 MB) | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Store** | In-memory cosine similarity search | `numpy` |
| **Retriever** | Embeds the question, finds top-K matching chunks | built-in |
| **Generator** | Sends context + question to LLaMA-3.1-8B and returns the answer | `groq` |

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunk size | Sidebar slider or `chunk_documents(chunk_size=…)` | 500 chars |
| Chunk overlap | Sidebar slider or `chunk_documents(chunk_overlap=…)` | 50 chars |
| Top-K chunks retrieved | `VectorStore.search(top_k=…)` | 3 |
| LLM model | `generate_answer(model=…)` | `llama-3.1-8b-instant` |
| Max response tokens | `generate_answer` → `max_tokens` | 512 |
| Temperature | `generate_answer` → `temperature` | 0.1 |

---

## Troubleshooting

**`ModuleNotFoundError`** — run `pip install sentence-transformers numpy groq pypdf python-docx streamlit`

**`streamlit` not on PATH** — use `python -m streamlit run streamlit_app.py` instead of `streamlit run …`

**Groq API error / invalid key** — double-check your key starts with `gsk_` and has no extra spaces in `.env`

**PDF shows "No text found"** — the PDF is likely scanned images. You'll need OCR preprocessing (e.g. `pytesseract`) before it can be used.

**Embedding model slow on first run** — it downloads ~80 MB to `~/.cache/huggingface/` once; subsequent runs are instant.

---

## License

MIT — free to use and modify.
