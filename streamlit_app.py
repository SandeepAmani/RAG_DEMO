"""
RAG Chatbot — Streamlit UI
Wraps the core pipeline from rag_demo.py with an interactive chat interface.
"""

import os
import time
import hashlib
from pathlib import Path

import streamlit as st

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global font & background ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Header bar ── */
  .rag-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.25);
  }
  .rag-header h1 {
    color: white;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
  }
  .rag-header p {
    color: #bfdbfe;
    font-size: 0.82rem;
    margin: 0;
  }
  .rag-header .icon { font-size: 2.2rem; }

  /* ── User avatar (top-right) ── */
  .user-pill {
    background: #dbeafe;
    color: #1e3a5f;
    border-radius: 999px;
    padding: 4px 12px 4px 6px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .user-pill .avatar {
    background: #2563eb;
    color: white;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
  }

  /* ── Chat bubbles ── */
  .msg-user {
    background: #2563eb;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2);
    font-size: 0.92rem;
    line-height: 1.5;
  }
  .msg-bot {
    background: #f1f5f9;
    color: #1e293b;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 0.92rem;
    line-height: 1.5;
  }
  .msg-label {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-bottom: 2px;
    font-weight: 500;
    letter-spacing: 0.03em;
  }
  .msg-label-right { text-align: right; }

  /* ── Chunk card ── */
  .chunk-card {
    background: #fefce8;
    border-left: 4px solid #eab308;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #422006;
    line-height: 1.6;
  }
  .chunk-card .chunk-header {
    font-weight: 700;
    color: #854d0e;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .chunk-card.active {
    background: #fef9c3;
    border-left-color: #ca8a04;
    box-shadow: 0 2px 8px rgba(202,138,4,0.18);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0f172a;
  }
  section[data-testid="stSidebar"] .block-container {
    padding: 18px 12px;
  }

  /* ── History item ── */
  .hist-item {
    background: #1e293b;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: border-color 0.15s;
  }
  .hist-item:hover { border-color: #3b82f6; }
  .hist-item .q { color: #e2e8f0; font-size: 0.82rem; font-weight: 500; }
  .hist-item .ts { color: #64748b; font-size: 0.7rem; }

  /* ── Status badge ── */
  .badge {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }
  .badge-green { background:#dcfce7; color:#166534; }
  .badge-blue  { background:#dbeafe; color:#1e40af; }
  .badge-yellow{ background:#fef9c3; color:#854d0e; }

  /* ── Divider ── */
  hr.styled { border: none; border-top: 1px solid #e2e8f0; margin: 12px 0; }

  /* Hide default Streamlit chrome padding overhead */
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Lazy imports (avoid crashing if deps missing) ───────────────────────────
@st.cache_resource(show_spinner=False)
def _load_deps():
    """Import heavy ML dependencies once and cache them."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from groq import Groq
        from pypdf import PdfReader
        return True, None
    except ImportError as e:
        return False, str(e)

# ─── Pipeline helpers (pulled from rag_demo.py logic) ────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(uploaded_file) -> str:
    """Extract text from a Streamlit UploadedFile (PDF)."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(t.strip() for t in pages if t.strip())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size character chunking with overlap."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings = None

    def build(self, chunks: list[str], model):
        self.chunks = chunks
        self.embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    def search(self, query: str, model, top_k: int = 3) -> list[dict]:
        q_vec = model.encode(query, convert_to_numpy=True)
        dot = self.embeddings.dot(q_vec)
        q_norm = np.linalg.norm(q_vec)
        d_norms = np.maximum(np.linalg.norm(self.embeddings, axis=1), 1e-10)
        scores = dot / (q_norm * d_norms)
        idx = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.chunks[i], "score": float(scores[i])} for i in idx]


def generate_answer(question: str, chunks: list[dict], groq_client) -> str:
    ctx_parts = [
        f"[Chunk {i+1} | similarity: {r['score']:.3f}]\n{r['chunk']}"
        for i, r in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(ctx_parts)
    system = (
        "You are a helpful AI assistant. Answer the user's question ONLY based on "
        "the provided context. If the context lacks enough information, say so clearly. "
        "Do not use outside knowledge."
    )
    user_msg = (
        f"CONTEXT (retrieved from knowledge base):\n"
        f"{'='*60}\n{context}\n{'='*60}\n\n"
        f"QUESTION: {question}"
    )
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ─── Session state initialisation ────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],           # list of {role, content, chunks, ts}
        "vector_store": None,
        "all_chunks": [],
        "doc_name": None,
        "history": [],            # condensed list for sidebar
        "groq_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Groq key helper ─────────────────────────────────────────────────────────
def _get_groq_key() -> str:
    key = st.session_state.groq_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="text-align:center; margin-bottom:18px;">
      <div style="font-size:2.2rem;">🤖</div>
      <div style="color:#e2e8f0; font-weight:700; font-size:1.1rem;">RAG Chatbot</div>
      <div style="color:#64748b; font-size:0.72rem;">Powered by Groq + HuggingFace</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── API Key ──
    with st.expander("🔑 Groq API Key", expanded=not bool(_get_groq_key())):
        key_input = st.text_input(
            "Enter your Groq API key",
            value=st.session_state.groq_key,
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if key_input:
            st.session_state.groq_key = key_input
        if _get_groq_key():
            st.markdown('<span class="badge badge-green">✓ Key loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">⚠ Key missing</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── PDF Upload ──
    st.markdown('<div style="color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.06em; margin-bottom:6px;">📂 UPLOAD DOCUMENT</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Overlap (chars)", 0, 200, 50, 10)

    process_btn = st.button("⚡ Process Document", use_container_width=True, type="primary")

    if process_btn and uploaded:
        if not _get_groq_key():
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Reading PDF…"):
                uploaded.seek(0)
                raw_text = load_pdf(uploaded)

            if not raw_text.strip():
                st.error("No text found in PDF. It may be image-based.")
            else:
                with st.spinner("Chunking document…"):
                    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    model = load_embedding_model()
                    vs = VectorStore()
                    vs.build(chunks, model)

                st.session_state.vector_store = vs
                st.session_state.all_chunks = chunks
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"✅ Ready — {len(chunks)} chunks indexed")

    if st.session_state.doc_name:
        st.markdown(f"""
        <div style="background:#1e293b; border-radius:8px; padding:8px 12px; margin-top:8px;">
          <div style="color:#7dd3fc; font-size:0.75rem; font-weight:600;">📄 Active document</div>
          <div style="color:#e2e8f0; font-size:0.8rem; word-break:break-all;">{st.session_state.doc_name}</div>
          <div style="color:#64748b; font-size:0.7rem; margin-top:2px;">{len(st.session_state.all_chunks)} chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Chat History ──
    st.markdown('<div style="color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.06em; margin-bottom:6px;">🕘 CHAT HISTORY</div>', unsafe_allow_html=True)

    if st.session_state.history:
        for item in reversed(st.session_state.history[-20:]):
            st.markdown(f"""
            <div class="hist-item">
              <div class="q">💬 {item['q'][:55]}{'…' if len(item['q'])>55 else ''}</div>
              <div class="ts">🕐 {item['ts']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569; font-size:0.78rem; text-align:center; padding:12px 0;">No questions yet</div>', unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


# ─── MAIN AREA ───────────────────────────────────────────────────────────────

# Header
col_h, col_user = st.columns([5, 1])
with col_h:
    st.markdown("""
    <div class="rag-header">
      <div class="icon">🤖</div>
      <div>
        <h1>RAG Chat Bot</h1>
        <p>Retrieval-Augmented Generation · Groq LLaMA-3.1-8B · HuggingFace MiniLM</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_user:
    st.markdown("""
    <div style="display:flex; justify-content:flex-end; align-items:center; height:100%;">
      <div class="user-pill">
        <div class="avatar">SB</div>
        Sandeep
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Status strip ──
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    key_ok = bool(_get_groq_key())
    st.markdown(
        f'<div style="text-align:center"><span class="badge {"badge-green" if key_ok else "badge-yellow"}">{"✓ API Key" if key_ok else "⚠ No API Key"}</span></div>',
        unsafe_allow_html=True,
    )
with status_col2:
    doc_ok = bool(st.session_state.vector_store)
    st.markdown(
        f'<div style="text-align:center"><span class="badge {"badge-green" if doc_ok else "badge-blue"}">{"✓ Document Ready" if doc_ok else "📄 No Document"}</span></div>',
        unsafe_allow_html=True,
    )
with status_col3:
    n_q = len(st.session_state.history)
    st.markdown(
        f'<div style="text-align:center"><span class="badge badge-blue">💬 {n_q} question{"s" if n_q != 1 else ""}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr class='styled'>", unsafe_allow_html=True)


# ── Two-column layout: chat | chunks ──
chat_col, chunk_col = st.columns([3, 2], gap="large")

with chat_col:
    st.markdown("#### 💬 Chat")

    # Render conversation
    chat_area = st.container()
    with chat_area:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding:48px 0; color:#94a3b8;">
              <div style="font-size:3rem; margin-bottom:12px;">💬</div>
              <div style="font-weight:600; font-size:1rem; margin-bottom:6px;">Start a conversation</div>
              <div style="font-size:0.82rem;">Upload a PDF and ask questions about it below.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-label msg-label-right">You · {msg["ts"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="msg-label">🤖 RAG Bot · {msg["ts"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Input box ──
    with st.form("chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_area(
                "Ask a question",
                placeholder="What is this document about? / Summarise key points…",
                label_visibility="collapsed",
                height=72,
                key="question_input",
            )
        with btn_col:
            send = st.form_submit_button("Send 🚀", use_container_width=True)

    if send and user_q.strip():
        user_q = user_q.strip()

        if not _get_groq_key():
            st.error("⚠ Please enter your Groq API key in the sidebar.")
        elif not st.session_state.vector_store:
            st.warning("📄 Please upload and process a PDF first.")
        else:
            ts = time.strftime("%H:%M")

            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_q, "ts": ts})

            with st.spinner("🔍 Retrieving relevant chunks…"):
                em_model = load_embedding_model()
                results = st.session_state.vector_store.search(user_q, em_model, top_k=3)

            with st.spinner("🧠 Generating answer…"):
                try:
                    from groq import Groq
                    client = Groq(api_key=_get_groq_key())
                    answer = generate_answer(user_q, results, client)
                except Exception as e:
                    answer = f"❌ Error calling Groq API: {e}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": results,
                "ts": ts,
            })
            st.session_state.history.append({"q": user_q, "ts": ts})
            st.rerun()


with chunk_col:
    st.markdown("#### 📑 Document Chunks")

    tab_all, tab_active = st.tabs(["All Chunks", "Retrieved Chunks"])

    with tab_all:
        if st.session_state.all_chunks:
            st.markdown(
                f'<div style="color:#64748b; font-size:0.78rem; margin-bottom:8px;">Total: <b>{len(st.session_state.all_chunks)}</b> chunks</div>',
                unsafe_allow_html=True,
            )
            # Show up to 30 chunks with scroll
            scroll_chunks = st.session_state.all_chunks[:30]
            for i, chunk in enumerate(scroll_chunks):
                preview = chunk[:220].replace("\n", " ")
                with st.expander(f"Chunk {i+1} · {len(chunk)} chars", expanded=False):
                    st.markdown(
                        f'<div style="font-size:0.8rem; color:#334155; line-height:1.6; white-space:pre-wrap;">{chunk}</div>',
                        unsafe_allow_html=True,
                    )
            if len(st.session_state.all_chunks) > 30:
                st.caption(f"… and {len(st.session_state.all_chunks)-30} more chunks")
        else:
            st.markdown(
                '<div style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:24px 0;">📄 Upload a document to see chunks here.</div>',
                unsafe_allow_html=True,
            )

    with tab_active:
        # Show chunks from the last bot message that has chunks
        last_chunks = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and msg.get("chunks"):
                last_chunks = msg["chunks"]
                break

        if last_chunks:
            st.markdown(
                f'<div style="color:#64748b; font-size:0.78rem; margin-bottom:10px;">Chunks used to answer the <b>last question</b>:</div>',
                unsafe_allow_html=True,
            )
            for i, r in enumerate(last_chunks):
                score_color = "#16a34a" if r["score"] > 0.5 else "#ca8a04" if r["score"] > 0.3 else "#dc2626"
                st.markdown(f"""
                <div class="chunk-card active">
                  <div class="chunk-header">
                    📌 Chunk {i+1}
                    <span style="font-size:0.75rem; color:{score_color}; font-weight:700;">
                      Similarity: {r['score']:.3f}
                    </span>
                  </div>
                  <div style="font-size:0.8rem; line-height:1.6; white-space:pre-wrap;">{r['chunk']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:24px 0;">🔍 Ask a question to see which chunks are retrieved.</div>',
                unsafe_allow_html=True,
            )
