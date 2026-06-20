"""
RAG Chatbot — Streamlit UI
Wraps the core pipeline from rag_demo.py with an interactive chat interface.

[CHANGED] Now supports three upload types:
  - PDF documents     (.pdf)
  - Word documents    (.docx)
  - Image files       (.png, .jpg, .jpeg, .gif, .webp)

Models upgraded:
  - Embeddings: clip-ViT-B-32  (was: all-MiniLM-L6-v2)
  - LLM:        meta-llama/llama-4-scout-17b-16e-instruct  (was: llama-3.1-8b-instant)
"""

import os
import io
import time
import base64
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

  /* ── Upload section cards ── */
  .upload-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
    border: 1px solid #334155;
  }
  .upload-card-title {
    color: #7dd3fc;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
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
  .badge-purple{ background:#ede9fe; color:#5b21b6; }

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
        from PIL import Image
        return True, None
    except ImportError as e:
        return False, str(e)

# ─── Pipeline helpers ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the CLIP multimodal embedding model.

    [CHANGED] Switched from 'all-MiniLM-L6-v2' (text-only, 384-dim) to
    'clip-ViT-B-32' (multimodal text+image, 512-dim). CLIP maps text and images
    into the same vector space, enabling semantic search across both modalities.

    [OLD] return SentenceTransformer("all-MiniLM-L6-v2")
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("clip-ViT-B-32")  # [NEW] multimodal CLIP model


def load_pdf(uploaded_file) -> tuple[str, list[dict]]:
    """
    Extract content from a Streamlit UploadedFile (PDF).

    Strategy:
      - For each page, first try pypdf text extraction (fast, works for digital PDFs).
      - If a page has no extractable text (scanned / image-based), render it to a
        PNG image using pymupdf (fitz) and store it as an image dict for the vision LLM.
        pymupdf is self-contained — no Poppler or Tesseract installation needed.

    Returns:
        text      (str)       : all extractable text joined across pages
        img_pages (list[dict]): image dicts (type='image') for pages with no text
    """
    from pypdf import PdfReader
    import fitz  # pymupdf — renders PDF pages to pixel images

    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Open the same bytes with pymupdf so we can render pages when needed
    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages_text = []
    img_pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Page has a text layer — use it directly
            pages_text.append(text.strip())
        else:
            # Page is image-based (scanned, diagram, screenshot in PDF, etc.)
            # Render it to a PNG at 150 DPI and store as an image chunk.
            # The vision LLM will read the visual content directly.
            try:
                fitz_page = fitz_doc[i]
                # Matrix(2, 2) = 2x zoom → 144 DPI (72 DPI base × 2); good balance of quality vs size
                mat = fitz.Matrix(2, 2)
                pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
                png_bytes = pix.tobytes("png")
                b64_data = base64.b64encode(png_bytes).decode("utf-8")
                img_pages.append({
                    "type": "image",
                    "path": f"{uploaded_file.name} — page {i+1}",
                    "mime_type": "image/png",
                    "b64_data": b64_data,
                })
            except Exception as e:
                st.warning(f"Could not render page {i+1} of {uploaded_file.name}: {e}")

    fitz_doc.close()
    combined_text = "\n\n".join(pages_text)
    return combined_text, img_pages


def load_docx(uploaded_file) -> str:
    """
    [NEW] Extract text from a Streamlit UploadedFile (Word .docx).
    Previously only PDFs were supported in the Streamlit UI.
    Uses python-docx to read paragraph text from the Word document.
    """
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(uploaded_file.read()))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_image_file(uploaded_file) -> dict | None:
    """
    [NEW] Load an image UploadedFile and encode it as base64 for the vision LLM.

    Why base64? The Groq vision API sends images as base64 data URIs inside the
    message content. This function prepares the image in that format.

    Returns a dict with 'type': 'image' so the vector store and LLM can distinguish
    image chunks from text chunks and handle them differently.
    """
    from PIL import Image as PILImage
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    mime_type = mime_map.get(suffix, "image/jpeg")

    try:
        img = PILImage.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG" if mime_type != "image/png" else "PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image",
            "path": uploaded_file.name,
            "mime_type": mime_type,
            "b64_data": b64_data,
        }
    except Exception as e:
        st.error(f"Could not load image {uploaded_file.name}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size character chunking with overlap. [UNCHANGED]"""
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
    """
    In-memory vector store using numpy cosine similarity.

    [CHANGED] The store now holds a mix of text strings and image dicts.
    The search() method is unchanged — the caller handles the mixed result types.
    """
    def __init__(self):
        self.chunks: list = []    # str or image dict
        self.embeddings = None

    def build(self, chunks: list, model):
        """
        Build the vector store from a mixed list of text strings and image dicts.

        [CHANGED] Previously only text strings were passed. Now we check each chunk:
        - text chunks  → passed as string to model.encode()
        - image chunks → decoded from base64 back to PIL Image, then passed to CLIP
        CLIP produces the same 512-dim embedding regardless of input type.
        """
        from PIL import Image as PILImage

        self.chunks = chunks
        inputs = []
        for chunk in chunks:
            if isinstance(chunk, dict) and chunk.get("type") == "image":
                # [NEW] Decode base64 back to PIL Image for CLIP encoding
                img_bytes = base64.b64decode(chunk["b64_data"])
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                inputs.append(pil_img)
            else:
                inputs.append(chunk if isinstance(chunk, str) else "")

        self.embeddings = model.encode(inputs, show_progress_bar=False, convert_to_numpy=True)

    def search(self, query: str, model, top_k: int = 3) -> list[dict]:
        """Search the vector store with a text query. [UNCHANGED logic]"""
        q_vec = model.encode(query, convert_to_numpy=True)
        dot = self.embeddings.dot(q_vec)
        q_norm = np.linalg.norm(q_vec)
        d_norms = np.maximum(np.linalg.norm(self.embeddings, axis=1), 1e-10)
        scores = dot / (q_norm * d_norms)
        idx = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.chunks[i], "score": float(scores[i])} for i in idx]


def generate_answer(question: str, chunks: list[dict], groq_client) -> str:
    """
    Generate an answer using the Groq vision LLM, grounded in retrieved chunks.

    [CHANGED]
    - Model: 'meta-llama/llama-4-scout-17b-16e-instruct'  (was: 'llama-3.1-8b-instant')
    - User message: list of content blocks     (was: plain string)
    Image chunks are embedded as base64 data URIs in 'image_url' content blocks.
    Text chunks are embedded as 'text' content blocks, same as before.
    """
    system = (
        "You are a helpful AI assistant. Answer the user's question ONLY based on "
        "the provided context. If the context lacks enough information, say so clearly. "
        "Do not use outside knowledge."
    )

    # [CHANGED] Build content as a list of blocks (supports images) instead of one string.
    # [OLD] user_msg = f"CONTEXT:\n{'='*60}\n{context}\n{'='*60}\n\nQUESTION: {question}"
    content_blocks = []
    content_blocks.append({"type": "text", "text": f"CONTEXT (retrieved from knowledge base):\n{'='*60}"})

    for i, r in enumerate(chunks):
        chunk = r["chunk"]
        score = r["score"]

        if isinstance(chunk, dict) and chunk.get("type") == "image":
            # [NEW] Image chunk — label + base64 image_url block
            content_blocks.append({
                "type": "text",
                "text": f"\n[Image {i+1} | similarity: {score:.3f}] ({chunk['path']})"
            })
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{chunk['mime_type']};base64,{chunk['b64_data']}"}
            })
        else:
            # Text chunk — same content as before, now in block format
            content_blocks.append({
                "type": "text",
                "text": f"\n[Chunk {i+1} | similarity: {score:.3f}]\n{chunk}"
            })

    content_blocks.append({"type": "text", "text": f"\n{'='*60}\n\nQUESTION: {question}"})

    # [CHANGED] model='meta-llama/llama-4-scout-17b-16e-instruct'  (was: 'llama-3.1-8b-instant')
    # [CHANGED] content=content_blocks (list)           (was: content=user_msg (string))
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",   # [NEW] vision-capable model
        # [OLD] model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": content_blocks},  # [NEW] list of blocks
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ─── Session state initialisation ────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "all_chunks": [],
        "doc_name": None,
        "history": [],
        "groq_key": "",
        # [NEW] Track counts per document type for the status display
        "n_pdf_chunks": 0,
        "n_word_chunks": 0,
        "n_image_chunks": 0,
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
      <div style="color:#64748b; font-size:0.72rem;">Powered by Groq + CLIP (Multimodal)</div>
    </div>
    """, unsafe_allow_html=True)
    # [OLD] <div style="color:#64748b; font-size:0.72rem;">Powered by Groq + HuggingFace</div>

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

    # ──────────────────────────────────────────────────────────────────────────
    # [NEW] Three separate upload sections: PDF, Word, Images
    # Previously there was only one file_uploader accepting PDF files.
    # Now each document type has its own upload widget and preview area.
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown('<div style="color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.06em; margin-bottom:8px;">📂 UPLOAD DOCUMENTS</div>', unsafe_allow_html=True)

    # ── Section 1: PDF Upload ──────────────────────────────────────────────
    st.markdown('<div class="upload-card"><div class="upload-card-title">📄 PDF Documents</div>', unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,    # [NEW] accept multiple files at once
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    # Show a small preview of each uploaded PDF file name
    if uploaded_pdfs:
        for f in uploaded_pdfs:
            st.markdown(
                f'<div style="color:#94a3b8; font-size:0.72rem; padding:2px 0;">📄 {f.name} ({f.size//1024} KB)</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Word Document Upload ──────────────────────────────────
    # [NEW] Word document upload was not available in the previous UI.
    st.markdown('<div class="upload-card"><div class="upload-card-title">📝 Word Documents</div>', unsafe_allow_html=True)
    uploaded_docs = st.file_uploader(
        "Upload Word files",
        type=["docx"],
        accept_multiple_files=True,
        key="docx_uploader",
        label_visibility="collapsed",
    )
    if uploaded_docs:
        for f in uploaded_docs:
            st.markdown(
                f'<div style="color:#94a3b8; font-size:0.72rem; padding:2px 0;">📝 {f.name} ({f.size//1024} KB)</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Image Upload ───────────────────────────────────────────
    # [NEW] Image upload was not available in the previous UI.
    # Images are encoded as base64 and sent directly to the vision LLM.
    st.markdown('<div class="upload-card"><div class="upload-card-title">🖼️ Images</div>', unsafe_allow_html=True)
    uploaded_images = st.file_uploader(
        "Upload image files",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )
    # [NEW] Show a thumbnail preview for each uploaded image
    if uploaded_images:
        preview_cols = st.columns(min(len(uploaded_images), 3))
        for idx, img_file in enumerate(uploaded_images):
            col = preview_cols[idx % 3]
            with col:
                img_file.seek(0)
                col.image(img_file, use_container_width=True, caption=img_file.name[:12])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chunking sliders ──
    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Overlap (chars)", 0, 200, 50, 10)

    process_btn = st.button("⚡ Process All Documents", use_container_width=True, type="primary")

    # ── Process all uploaded files ──
    if process_btn:
        any_uploaded = uploaded_pdfs or uploaded_docs or uploaded_images
        if not any_uploaded:
            st.warning("Please upload at least one file first.")
        elif not _get_groq_key():
            st.error("Please enter your Groq API key first.")
        else:
            all_chunks = []
            n_pdf = n_word = n_img = 0

            # --- Process PDF files ---
            # load_pdf() now returns (text, img_pages).
            # Text pages are chunked normally; image pages (scanned/diagram pages)
            # are passed through as image dicts for the vision LLM to read directly.
            if uploaded_pdfs:
                with st.spinner(f"Reading {len(uploaded_pdfs)} PDF file(s)…"):
                    for f in uploaded_pdfs:
                        f.seek(0)
                        raw_text, img_pages = load_pdf(f)
                        if raw_text.strip():
                            text_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
                            all_chunks.extend(text_chunks)
                            n_pdf += len(text_chunks)
                        if img_pages:
                            # Image pages from scanned/image-based PDF — add as image chunks
                            all_chunks.extend(img_pages)
                            n_img += len(img_pages)
                            st.info(f"📄→🖼️ {f.name}: {len(img_pages)} image page(s) will be read by the vision model.")
                        if not raw_text.strip() and not img_pages:
                            st.warning(f"Could not extract any content from {f.name}.")

            # --- Process Word files ---
            # [NEW] Word processing was not available before
            if uploaded_docs:
                with st.spinner(f"Reading {len(uploaded_docs)} Word file(s)…"):
                    for f in uploaded_docs:
                        f.seek(0)
                        raw_text = load_docx(f)
                        if raw_text.strip():
                            text_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
                            all_chunks.extend(text_chunks)
                            n_word += len(text_chunks)
                        else:
                            st.warning(f"No text found in {f.name}.")

            # --- Process Image files ---
            # [NEW] Image processing was not available before
            if uploaded_images:
                with st.spinner(f"Encoding {len(uploaded_images)} image(s)…"):
                    for f in uploaded_images:
                        f.seek(0)
                        img_doc = load_image_file(f)
                        if img_doc:
                            all_chunks.append(img_doc)  # image dicts pass through as single chunks
                            n_img += 1

            if not all_chunks:
                st.error("No content could be extracted from the uploaded files.")
            else:
                # Embed all chunks using CLIP (handles both text strings and image dicts)
                with st.spinner(f"Embedding {len(all_chunks)} chunks with CLIP…"):
                    model = load_embedding_model()
                    vs = VectorStore()
                    vs.build(all_chunks, model)

                st.session_state.vector_store = vs
                st.session_state.all_chunks = all_chunks
                st.session_state.n_pdf_chunks = n_pdf
                st.session_state.n_word_chunks = n_word
                st.session_state.n_image_chunks = n_img
                # Build a combined name from all uploaded files
                names = (
                    [f.name for f in (uploaded_pdfs or [])] +
                    [f.name for f in (uploaded_docs or [])] +
                    [f.name for f in (uploaded_images or [])]
                )
                st.session_state.doc_name = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
                st.session_state.messages = []
                st.success(
                    f"✅ Ready — {n_pdf} PDF chunks · {n_word} Word chunks · {n_img} image(s) indexed"
                )

    # Active document summary
    if st.session_state.doc_name:
        st.markdown(f"""
        <div style="background:#1e293b; border-radius:8px; padding:8px 12px; margin-top:8px;">
          <div style="color:#7dd3fc; font-size:0.75rem; font-weight:600;">📂 Active documents</div>
          <div style="color:#e2e8f0; font-size:0.8rem; word-break:break-all;">{st.session_state.doc_name}</div>
          <div style="color:#64748b; font-size:0.7rem; margin-top:4px;">
            📄 {st.session_state.n_pdf_chunks} PDF chunks &nbsp;·&nbsp;
            📝 {st.session_state.n_word_chunks} Word chunks &nbsp;·&nbsp;
            🖼️ {st.session_state.n_image_chunks} images
          </div>
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
        <p>Multimodal RAG · Groq LLaMA-3.2-11B-Vision · CLIP-ViT-B-32 · PDF · Word · Images</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # [OLD] <p>Retrieval-Augmented Generation · Groq LLaMA-3.1-8B · HuggingFace MiniLM</p>
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
# [CHANGED] Added a 4th status column for image count
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    key_ok = bool(_get_groq_key())
    st.markdown(
        f'<div style="text-align:center"><span class="badge {"badge-green" if key_ok else "badge-yellow"}">{"✓ API Key" if key_ok else "⚠ No API Key"}</span></div>',
        unsafe_allow_html=True,
    )
with status_col2:
    doc_ok = bool(st.session_state.vector_store)
    st.markdown(
        f'<div style="text-align:center"><span class="badge {"badge-green" if doc_ok else "badge-blue"}">{"✓ Docs Ready" if doc_ok else "📄 No Docs"}</span></div>',
        unsafe_allow_html=True,
    )
with status_col3:
    n_img = st.session_state.n_image_chunks
    st.markdown(
        f'<div style="text-align:center"><span class="badge {"badge-purple" if n_img else "badge-blue"}">{"🖼️ " + str(n_img) + " image(s)" if n_img else "🖼️ No Images"}</span></div>',
        unsafe_allow_html=True,
    )
with status_col4:
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
              <div style="font-size:0.82rem;">Upload PDFs, Word documents, or images and ask questions about them.</div>
            </div>
            """, unsafe_allow_html=True)
            # [OLD] <div style="font-size:0.82rem;">Upload a PDF and ask questions about it below.</div>
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
                placeholder="What is this document about? / Describe the image / Summarise key points…",
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
            st.warning("📄 Please upload and process at least one file first.")
        else:
            ts = time.strftime("%H:%M")

            st.session_state.messages.append({"role": "user", "content": user_q, "ts": ts})

            with st.spinner("🔍 Retrieving relevant chunks…"):
                em_model = load_embedding_model()
                results = st.session_state.vector_store.search(user_q, em_model, top_k=3)

            with st.spinner("🧠 Generating answer (vision model)…"):
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
            text_count = sum(1 for c in st.session_state.all_chunks if isinstance(c, str))
            img_count = sum(1 for c in st.session_state.all_chunks if isinstance(c, dict))
            st.markdown(
                f'<div style="color:#64748b; font-size:0.78rem; margin-bottom:8px;">'
                f'Total: <b>{len(st.session_state.all_chunks)}</b> chunks '
                f'({text_count} text · {img_count} image)</div>',
                unsafe_allow_html=True,
            )
            # Show up to 30 chunks
            scroll_chunks = st.session_state.all_chunks[:30]
            for i, chunk in enumerate(scroll_chunks):
                if isinstance(chunk, dict) and chunk.get("type") == "image":
                    # [NEW] Show image preview for image chunks
                    with st.expander(f"Image {i+1} · {chunk['path']}", expanded=False):
                        img_bytes = base64.b64decode(chunk["b64_data"])
                        from PIL import Image as PILImage
                        pil_img = PILImage.open(io.BytesIO(img_bytes))
                        st.image(pil_img, use_container_width=True)
                        st.caption(f"Type: {chunk['mime_type']} · File: {chunk['path']}")
                else:
                    # Text chunk (same as before)
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
                '<div style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:24px 0;">📄 Upload documents to see chunks here.</div>',
                unsafe_allow_html=True,
            )

    with tab_active:
        # Show chunks from the last bot message
        last_chunks = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and msg.get("chunks"):
                last_chunks = msg["chunks"]
                break

        if last_chunks:
            st.markdown(
                '<div style="color:#64748b; font-size:0.78rem; margin-bottom:10px;">Chunks used to answer the <b>last question</b>:</div>',
                unsafe_allow_html=True,
            )
            for i, r in enumerate(last_chunks):
                chunk = r["chunk"]
                score = r["score"]
                score_color = "#16a34a" if score > 0.5 else "#ca8a04" if score > 0.3 else "#dc2626"

                if isinstance(chunk, dict) and chunk.get("type") == "image":
                    # [NEW] Display retrieved image chunks with a preview thumbnail
                    st.markdown(f"""
                    <div class="chunk-card active">
                      <div class="chunk-header">
                        🖼️ Image {i+1}
                        <span style="font-size:0.75rem; color:{score_color}; font-weight:700;">
                          Similarity: {score:.3f}
                        </span>
                      </div>
                      <div style="font-size:0.75rem; color:#64748b;">{chunk['path']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    img_bytes = base64.b64decode(chunk["b64_data"])
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(io.BytesIO(img_bytes))
                    st.image(pil_img, use_container_width=True)
                else:
                    # Text chunk (same as before)
                    st.markdown(f"""
                    <div class="chunk-card active">
                      <div class="chunk-header">
                        📌 Chunk {i+1}
                        <span style="font-size:0.75rem; color:{score_color}; font-weight:700;">
                          Similarity: {score:.3f}
                        </span>
                      </div>
                      <div style="font-size:0.8rem; line-height:1.6; white-space:pre-wrap;">{chunk}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:24px 0;">🔍 Ask a question to see which chunks are retrieved.</div>',
                unsafe_allow_html=True,
            )
