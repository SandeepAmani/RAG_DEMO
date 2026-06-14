# =============================================================================
# RAG DEMO — Retrieval-Augmented Generation for Study Purposes
# =============================================================================
#
# WHAT IS RAG?
# ------------
# RAG stands for Retrieval-Augmented Generation.
# It combines two ideas:
#   1. RETRIEVAL  — search a knowledge base to find relevant text chunks
#   2. GENERATION — send those chunks to an LLM so it can answer grounded
#                   in actual content (not just its training memory)
#
# WHY USE RAG?
# ------------
# LLMs like GPT or LLaMA are trained on fixed data. They can hallucinate
# (make things up) or not know recent/private facts. RAG fixes this by
# giving the LLM fresh, relevant context at query time.
#
# PIPELINE (end-to-end flow):
# ---------------------------
#   Documents (text)
#        ↓
#   Chunking (split into small pieces)
#        ↓
#   Embedding (convert each chunk to a vector / numbers)
#        ↓
#   Vector Store (store vectors in memory for fast search)
#        ↓
#   User asks a question
#        ↓
#   Question → Embedding → Search vector store → Top-K chunks retrieved
#        ↓
#   Prompt = "Use this context: <chunks> \n Answer: <question>"
#        ↓
#   LLM (Groq / LLaMA3) generates the final answer
#
# COMPONENTS USED IN THIS SCRIPT:
# --------------------------------
#   • sentence-transformers  — HuggingFace library to create text embeddings
#   • numpy                  — for vector math (cosine similarity search)
#   • groq (Python SDK)      — to call the Groq LLM API
#   • os / getpass           — to read the API key from environment variables
#
# HOW TO RUN:
# -----------
#   1. Install dependencies:
#        pip install sentence-transformers numpy groq
#
#   2. Set your Groq API key (get a free one at https://console.groq.com):
#        Windows PowerShell:   $env:GROQ_API_KEY = "your_key_here"
#        Windows CMD:          set GROQ_API_KEY=your_key_here
#        macOS/Linux bash:     export GROQ_API_KEY=your_key_here
#
#   3. Run the script:
#        python rag_demo.py
# =============================================================================


# --- STANDARD LIBRARY IMPORTS ------------------------------------------------
import os        # Used to read environment variables (e.g., GROQ_API_KEY)
import sys       # Used to exit the script cleanly on errors
import textwrap  # Used to wrap long text output neatly in the terminal
from pathlib import Path  # Used to locate the .env file relative to this script

# --- THIRD-PARTY IMPORTS -----------------------------------------------------
# These must be installed via: pip install sentence-transformers numpy groq pypdf python-docx

try:
    import numpy as np
    # numpy is a numerical computing library.
    # We use it here to:
    #   - Store embedding vectors as arrays
    #   - Compute cosine similarity (how similar two vectors are)
except ImportError:
    print("ERROR: 'numpy' is not installed. Run: pip install numpy")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    # SentenceTransformer loads a pre-trained HuggingFace embedding model.
    # It converts a string of text → a list of numbers (a "vector").
    # Similar texts produce vectors that are close together in vector space.
    # Model used: 'all-MiniLM-L6-v2' — small (80MB), fast, runs fully offline.
except ImportError:
    print("ERROR: 'sentence-transformers' is not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

try:
    from groq import Groq
    # Groq is the Python SDK for calling the Groq LLM API.
    # Groq provides free, very fast inference for open-source models like LLaMA3.
    # We use it to generate the final natural-language answer.
except ImportError:
    print("ERROR: 'groq' is not installed. Run: pip install groq")
    sys.exit(1)

try:
    from pypdf import PdfReader
    # pypdf reads PDF files page by page and extracts their text content.
    # PdfReader opens the file; .pages gives a list of page objects;
    # page.extract_text() returns the text on that page as a plain string.
except ImportError:
    print("ERROR: 'pypdf' is not installed. Run: pip install pypdf")
    sys.exit(1)

try:
    from docx import Document as DocxDocument
    # python-docx reads .docx (Word) files.
    # Document() opens the file; .paragraphs gives a list of paragraph objects;
    # paragraph.text returns the text of that paragraph as a plain string.
except ImportError:
    print("ERROR: 'python-docx' is not installed. Run: pip install python-docx")
    sys.exit(1)


# =============================================================================
# STEP 0 — DOCUMENT LOADER (load text from PDF or Word files)
# =============================================================================
#
# This section adds support for loading EXTERNAL documents (PDF / .docx) so
# the RAG system can answer questions about YOUR files instead of (or in
# addition to) the hardcoded study documents.
#
# HOW IT WORKS:
#   - load_pdf(path)  → reads every page of a PDF and returns text as a string
#   - load_docx(path) → reads every paragraph of a Word doc and returns text
#   - load_documents_from_files(paths) → auto-detects file type, returns a
#                                        list of text strings (one per file)
#
# The returned text strings are then added to the DOCUMENTS list and go
# through the same chunking → embedding → vector store pipeline as the
# hardcoded documents.

def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Parameters:
        file_path (str): absolute or relative path to the .pdf file

    Returns:
        text (str): all page text joined together, or empty string on failure
    """
    print(f"  [PDF] Reading: {file_path}")
    reader = PdfReader(file_path)
    # reader.pages is a list — one PageObject per page in the PDF
    pages_text = []
    for i, page in enumerate(reader.pages):
        # extract_text() returns the raw text on this page.
        # Some PDFs (scanned images) return empty string — we skip those.
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(text.strip())
    combined = "\n\n".join(pages_text)  # join pages with blank line between them
    print(f"  [PDF] Extracted {len(reader.pages)} pages, "
          f"{len(combined):,} characters of text.")
    return combined


def load_docx(file_path: str) -> str:
    """
    Extract all text from a Word (.docx) file.

    Parameters:
        file_path (str): absolute or relative path to the .docx file

    Returns:
        text (str): all paragraph text joined together, or empty string on failure
    """
    print(f"  [DOCX] Reading: {file_path}")
    doc = DocxDocument(file_path)
    # doc.paragraphs is a list — one Paragraph object per paragraph block.
    # paragraph.text gives the plain text (no formatting codes).
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    combined = "\n\n".join(paragraphs)
    print(f"  [DOCX] Extracted {len(paragraphs)} paragraphs, "
          f"{len(combined):,} characters of text.")
    return combined


def load_documents_from_files(file_paths: list) -> list:
    """
    Load text from a list of PDF and/or Word file paths.

    Parameters:
        file_paths (list of str): paths to .pdf or .docx files

    Returns:
        documents (list of str): one text string per successfully loaded file
    """
    documents = []
    for path in file_paths:
        # Normalise: strip 'file:///' prefix if the user pastes a browser URL
        # e.g. "file:///C:/Users/..." → "C:/Users/..."
        if path.startswith("file:///"):
            path = path[8:]            # remove "file:///"
        elif path.startswith("file://"):
            path = path[7:]            # remove "file://"

        path = path.replace("/", "\\") # normalise slashes on Windows

        if not Path(path).exists():
            print(f"  [WARNING] File not found, skipping: {path}")
            continue

        suffix = Path(path).suffix.lower()  # e.g. ".pdf" or ".docx"

        if suffix == ".pdf":
            text = load_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = load_docx(path)
        else:
            print(f"  [WARNING] Unsupported file type '{suffix}', skipping: {path}")
            continue

        if text.strip():
            documents.append(text)
        else:
            print(f"  [WARNING] No text extracted from: {path}")

    return documents


# =============================================================================
# STEP 1 — KNOWLEDGE BASE (Hardcoded study documents about RAG)
# =============================================================================
#
# In a real production RAG system, these documents would come from PDFs,
# databases, web pages, or uploaded files. Here we hardcode them directly
# in the script so you can study without any external files.
#
# Each string in this list is one "document" — a self-contained piece of text.
# The RAG system will split these into smaller chunks, embed them, and search
# through them to answer your questions.
DOCUMENTS = []
# DOCUMENTS = [
#     """
#     RAG stands for Retrieval-Augmented Generation. It is a technique in AI
#     that combines information retrieval with text generation. Instead of
#     relying solely on what an LLM learned during training, RAG allows the model
#     to look up relevant information from an external knowledge base at query
#     time. This makes responses more accurate, factual, and up-to-date.
#     """,

#     """
#     The RAG pipeline has three main stages: indexing, retrieval, and generation.
#     During indexing, documents are split into chunks, and each chunk is converted
#     into a vector embedding using an embedding model. During retrieval, the user's
#     question is also embedded, and the most similar document chunks are found using
#     cosine similarity or other vector search methods. During generation, the
#     retrieved chunks are passed to the LLM as context so it can generate a
#     grounded answer.
#     """,

#     """
#     An embedding is a numerical representation of text — a list of floating-point
#     numbers called a vector. Semantically similar texts produce vectors that are
#     close to each other in high-dimensional space. For example, the sentences
#     "What is machine learning?" and "Define ML" would have very similar embedding
#     vectors. This property allows us to search for relevant text by comparing
#     vectors rather than doing keyword matching.
#     """,

#     """
#     Cosine similarity is the most common metric used to compare embedding vectors
#     in RAG systems. It measures the angle between two vectors — a score of 1.0
#     means the vectors point in the same direction (very similar), while a score
#     near 0 means they are unrelated. Given a user query embedding and a set of
#     document chunk embeddings, we rank the chunks by cosine similarity and return
#     the top-K most relevant chunks.
#     """,

#     """
#     Chunking is the process of splitting long documents into smaller pieces before
#     embedding them. This is important because embedding models have a maximum token
#     limit (e.g., 512 tokens). Also, smaller chunks lead to more precise retrieval —
#     you return only the paragraph that is relevant, not an entire 10-page document.
#     Common chunking strategies include: fixed-size chunking (every N characters),
#     sentence-level chunking, and semantic chunking (split on topic boundaries).
#     """,

#     """
#     The prompt sent to the LLM in a RAG system typically has three parts:
#     1. A system instruction: "You are a helpful assistant. Answer only based on
#        the provided context."
#     2. The retrieved context: the top-K document chunks found by the retriever.
#     3. The user question.
#     By instructing the LLM to answer only from context, we reduce hallucination
#     and keep responses grounded in the actual knowledge base content.
#     """,

#     """
#     Vector stores (also called vector databases) are specialized databases for
#     storing and searching embedding vectors efficiently. Examples include:
#     FAISS (Meta, open-source), ChromaDB, Pinecone, Weaviate, and Qdrant.
#     In this demo script, we implement a simple in-memory vector store using
#     a plain Python list and numpy cosine similarity — no external database needed.
#     For large-scale production use (millions of documents), a proper vector DB
#     like FAISS or Pinecone would be used instead.
#     """,

#     """
#     Groq is a company that provides extremely fast inference for open-source LLMs
#     using their custom LPU (Language Processing Unit) hardware. Their free API
#     supports models like LLaMA3 (by Meta) and Mixtral (by Mistral AI).
#     In a RAG pipeline, Groq serves as the GENERATION component — it takes the
#     retrieved context and the user question and generates a final natural-language
#     answer. The model used in this demo is llama3-8b-8192 (8 billion parameters,
#     8192 token context window).
#     """,

#     """
#     HuggingFace is an AI company and open-source platform that hosts thousands of
#     pre-trained models for NLP, computer vision, and more. The `sentence-transformers`
#     library, built on top of HuggingFace, provides easy-to-use embedding models.
#     The model used in this demo is `all-MiniLM-L6-v2` — a distilled BERT model
#     that maps sentences to a 384-dimensional embedding space. It is small (~80MB),
#     fast, runs fully offline (no API key needed), and works well for semantic search.
#     """,

#     """
#     Hallucination in LLMs refers to when the model generates plausible-sounding
#     but factually incorrect information. This happens because LLMs predict the
#     next token based on patterns in training data, not because they have verified
#     knowledge. RAG reduces hallucination by providing the model with actual source
#     text at inference time, and by instructing it to answer only from that context.
#     This is one of the core reasons RAG is widely used in production AI applications.
#     """,
# ]

# =============================================================================
# EXTERNAL FILES — Add your PDF / Word documents here
# =============================================================================
#
# HOW TO USE:
#   Add the full path to your file(s) in the list below.
#   You can paste either a Windows path or a browser file:// URL — both work.
#
# EXAMPLES:
#   "C:\\Users\\sande\\Downloads\\my_document.pdf"
#   "file:///C:/Users/sande/Downloads/my_document.pdf"
#   "C:\\Users\\sande\\Documents\\notes.docx"
#
# You can add multiple files — one path per line, separated by commas.
# Set the list to [] (empty) to use only the hardcoded study documents above.

EXTERNAL_FILES = [
    "file:///C:/Users/sande/Downloads/Learn_Coding_in_2026_12_Week_Guide.pdf",
]

# Load text from the external files and append to the DOCUMENTS list.
# This happens at import time so the rest of the pipeline sees one unified list.
if EXTERNAL_FILES:
    print("\n[FILES] Loading external documents...")
    loaded = load_documents_from_files(EXTERNAL_FILES)
    DOCUMENTS.extend(loaded)
    print(f"[FILES] Added {len(loaded)} external document(s) to knowledge base.")


# =============================================================================
# STEP 2 — CHUNKING
# =============================================================================
#
# We split each document into smaller chunks.
# In this demo, documents are already short, but we still apply chunking
# to demonstrate the concept and handle any longer documents correctly.
#
# Strategy used: fixed-size character chunking with overlap.
# - chunk_size   : max number of characters per chunk
# - chunk_overlap: how many characters to repeat between consecutive chunks
#                  Overlap helps avoid cutting a sentence mid-thought and
#                  losing context at chunk boundaries.

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split a list of documents into smaller overlapping text chunks.

    Parameters:
        documents    (list of str) : the raw text documents
        chunk_size   (int)         : maximum characters per chunk
        chunk_overlap(int)         : characters of overlap between adjacent chunks

    Returns:
        chunks (list of str) : all resulting text chunks across all documents
    """
    chunks = []  # will collect all chunks from all documents

    for doc in documents:
        # Strip leading/trailing whitespace from the document
        doc = doc.strip()

        # If the document fits within one chunk, add it directly
        if len(doc) <= chunk_size:
            chunks.append(doc)
            continue

        # Otherwise, slide a window across the document
        # start : where the current chunk begins
        start = 0
        while start < len(doc):
            # end : where the current chunk ends (capped at doc length)
            end = min(start + chunk_size, len(doc))

            # Extract the chunk and add it to our list
            chunk = doc[start:end].strip()
            if chunk:  # skip empty strings
                chunks.append(chunk)

            # Move the window forward, stepping back by overlap amount
            # so adjacent chunks share some context
            start += chunk_size - chunk_overlap

    return chunks


# =============================================================================
# STEP 3 — EMBEDDING MODEL (HuggingFace / sentence-transformers)
# =============================================================================
#
# The embedding model converts text → vector (a list of numbers).
# We load the model ONCE at startup (it downloads ~80MB on first run,
# then caches locally at ~/.cache/huggingface/).
# Subsequent runs use the local cache — no internet needed after first load.

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Load and return a SentenceTransformer embedding model.

    Parameters:
        model_name (str): HuggingFace model identifier

    Returns:
        model (SentenceTransformer): loaded embedding model
    """
    print(f"\n[EMBEDDING] Loading model '{model_name}' from HuggingFace...")
    print("  (First run downloads ~80MB. Subsequent runs use local cache.)")

    # SentenceTransformer automatically downloads the model from HuggingFace Hub
    # and caches it at: C:\Users\<you>\.cache\huggingface\hub\  (Windows)
    model = SentenceTransformer(model_name)

    print(f"[EMBEDDING] Model loaded successfully.")
    return model


def embed_texts(model, texts):
    """
    Convert a list of text strings into embedding vectors.

    Parameters:
        model (SentenceTransformer): the loaded embedding model
        texts (list of str)        : text strings to embed

    Returns:
        embeddings (np.ndarray): 2D array of shape (N, D)
                                 N = number of texts
                                 D = embedding dimension (384 for MiniLM)
    """
    # model.encode() takes a list of strings and returns a numpy array
    # show_progress_bar=True prints a tqdm progress bar during encoding
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


# =============================================================================
# STEP 4 — VECTOR STORE (in-memory, using numpy)
# =============================================================================
#
# A vector store stores:
#   - the original text chunks
#   - their corresponding embedding vectors
#
# At search time, we:
#   1. Embed the user's query
#   2. Compare the query vector to all stored chunk vectors using cosine similarity
#   3. Return the top-K most similar chunks
#
# Cosine similarity formula:
#   similarity = (A · B) / (||A|| * ||B||)
# Where A and B are vectors, · is dot product, ||.|| is the L2 norm (magnitude).
# Result is between -1 and 1. For text, usually between 0 and 1.

class SimpleVectorStore:
    """
    A minimal in-memory vector store using numpy for cosine similarity search.

    Attributes:
        chunks     (list of str)  : the text chunks
        embeddings (np.ndarray)   : shape (N, D) — one row per chunk
    """

    def __init__(self):
        self.chunks = []           # stores the text of each chunk
        self.embeddings = None     # stores all embedding vectors as a 2D numpy array

    def add(self, chunks, embeddings):
        """
        Add chunks and their precomputed embeddings to the store.

        Parameters:
            chunks     (list of str) : text chunks
            embeddings (np.ndarray)  : shape (N, D)
        """
        self.chunks = chunks
        self.embeddings = embeddings
        print(f"\n[VECTOR STORE] Indexed {len(chunks)} chunks.")
        print(f"[VECTOR STORE] Each embedding has {embeddings.shape[1]} dimensions.")

    def cosine_similarity(self, query_vec, doc_vecs):
        """
        Compute cosine similarity between one query vector and many document vectors.

        Parameters:
            query_vec (np.ndarray): shape (D,) — the query embedding
            doc_vecs  (np.ndarray): shape (N, D) — all chunk embeddings

        Returns:
            similarities (np.ndarray): shape (N,) — similarity score per chunk
        """
        # Compute dot products: query_vec · each row of doc_vecs
        # np.dot(doc_vecs, query_vec) gives shape (N,)
        dot_products = np.dot(doc_vecs, query_vec)

        # Compute the L2 norm (magnitude) of the query vector
        # np.linalg.norm computes sqrt(sum of squares)
        query_norm = np.linalg.norm(query_vec)

        # Compute the L2 norm of each document vector
        # axis=1 means compute along each row → result shape (N,)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        # Avoid division by zero: replace any 0 norm with a tiny number
        # np.maximum keeps the larger of two values element-wise
        doc_norms = np.maximum(doc_norms, 1e-10)

        # Final cosine similarity: dot / (norm_query * norm_doc)
        similarities = dot_products / (query_norm * doc_norms)
        return similarities

    def search(self, query_embedding, top_k=3):
        """
        Retrieve the top-K most relevant chunks for a given query embedding.

        Parameters:
            query_embedding (np.ndarray): shape (D,) — the embedded user query
            top_k (int)                 : number of top results to return

        Returns:
            results (list of dict): each dict has 'chunk' (str) and 'score' (float)
        """
        # Compute similarity of the query to all stored chunks
        scores = self.cosine_similarity(query_embedding, self.embeddings)

        # argsort returns indices that would sort the array in ascending order
        # [::-1] reverses it to get descending order (highest similarity first)
        ranked_indices = np.argsort(scores)[::-1]

        # Take only the top-K indices
        top_indices = ranked_indices[:top_k]

        # Build result list with chunk text and similarity score
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })

        return results


# =============================================================================
# STEP 5 — RETRIEVER
# =============================================================================
#
# The retriever wraps the vector store and the embedding model together.
# Given a raw text query (string), it:
#   1. Embeds the query using the same model used to index documents
#   2. Searches the vector store for top-K similar chunks
# This is the "R" in RAG.

def retrieve(query, embedding_model, vector_store, top_k=3):
    """
    Retrieve the most relevant document chunks for a user query.

    Parameters:
        query           (str)               : the user's question as plain text
        embedding_model (SentenceTransformer): the loaded embedding model
        vector_store    (SimpleVectorStore) : the indexed vector store
        top_k           (int)               : how many chunks to retrieve

    Returns:
        results (list of dict): top-K chunks with similarity scores
    """
    # Embed the query — same model used for indexing, so vectors are comparable
    # encode() can take a single string or a list; returns shape (D,) for one string
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    # Search the vector store for the most similar chunks
    results = vector_store.search(query_embedding, top_k=top_k)

    return results


# =============================================================================
# STEP 6 — GENERATOR (Groq + LLaMA3)
# =============================================================================
#
# The generator takes:
#   - the retrieved context chunks
#   - the user's question
# and calls the Groq API to produce a final answer.
#
# We construct a prompt with:
#   - a SYSTEM message: sets the LLM's behavior (answer only from context)
#   - a USER message  : contains the context + question
#
# The model: llama3-8b-8192
#   "llama3"   = Meta's LLaMA 3 model family
#   "8b"       = 8 billion parameters (size of the model)
#   "8192"     = maximum context window in tokens

def generate_answer(question, retrieved_chunks, groq_client, model="llama-3.1-8b-instant"):
    """
    Generate an answer using the Groq LLM, grounded in retrieved context.

    Parameters:
        question         (str)        : the user's original question
        retrieved_chunks (list of dict): output of retrieve() — chunks + scores
        groq_client      (Groq)       : authenticated Groq API client
        model            (str)        : Groq model ID to use

    Returns:
        answer (str): the LLM's generated response
    """
    # --- Build the context string from retrieved chunks ---
    # Join all retrieved chunk texts, numbered for clarity
    context_parts = []
    for i, result in enumerate(retrieved_chunks, start=1):
        context_parts.append(f"[Chunk {i} | similarity: {result['score']:.3f}]\n{result['chunk']}")

    # Join all parts with a separator line
    context_text = "\n\n---\n\n".join(context_parts)

    # --- System message ---
    # This tells the LLM how to behave. "Only use the context" reduces hallucination.
    system_message = (
        "You are a helpful AI assistant specializing in explaining RAG concepts. "
        "Answer the user's question ONLY based on the provided context below. "
        "If the context does not contain enough information to answer, say so clearly. "
        "Do not use any knowledge outside of what is in the context."
    )

    # --- User message ---
    # This is what the LLM "sees" as the human turn. Context comes first, then question.
    user_message = (
        f"CONTEXT (retrieved from knowledge base):\n"
        f"{'='*60}\n"
        f"{context_text}\n"
        f"{'='*60}\n\n"
        f"QUESTION: {question}"
    )

    # --- Call the Groq API ---
    # groq_client.chat.completions.create() follows the OpenAI Chat Completions format.
    # messages: a list of role/content dicts — "system" sets behavior, "user" is the input.
    # max_tokens: cap the response length (prevents runaway generation).
    # temperature: controls randomness. 0.2 = mostly deterministic, focused answers.
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=512,   # maximum tokens in the generated response
        temperature=0.9,  # low temperature → more focused, less creative
    )

    # Extract the answer text from the response object
    # response.choices[0] — first (and usually only) completion choice
    # .message.content    — the generated text string
    answer = response.choices[0].message.content
    return answer


# =============================================================================
# STEP 7 — MAIN: Wire Everything Together
# =============================================================================
#
# This function runs the full RAG pipeline:
#   1. Load and chunk documents
#   2. Load embedding model
#   3. Embed all chunks and build the vector store (indexing phase)
#   4. Set up the Groq LLM client
#   5. Enter an interactive Q&A loop in the terminal

def main():
    print("=" * 65)
    print("  RAG DEMO — Retrieval-Augmented Generation (Study Edition)")
    print("  LLM: Groq / LLaMA-3.1-8B-Instant   |   Embeddings: MiniLM-L6-v2")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 7a. Read the Groq API key — checks two places in order:
    #       1. Environment variable GROQ_API_KEY  (set in your terminal session)
    #       2. A .env file in the same folder as this script
    #
    # The .env file format is simply one line:
    #       GROQ_API_KEY=your_key_here
    # -------------------------------------------------------------------------

    api_key = os.environ.get("GROQ_API_KEY")  # Check environment variable first

    if not api_key:
        # Path(__file__) is the absolute path to this script file.
        # .parent gives the folder that contains it.
        # / ".env" appends the filename — this builds: <script_folder>/.env
        env_file = Path(__file__).parent / ".env"

        if env_file.exists():
            # Read every line in the .env file
            for line in env_file.read_text().splitlines():
                line = line.strip()
                # Skip blank lines and comment lines (starting with #)
                if not line or line.startswith("#"):
                    continue
                # Split on the first "=" only, so values can contain "=" characters
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "GROQ_API_KEY":
                        api_key = value.strip()
                        print("\n[API] Groq API key loaded from .env file.")
                        break

    if not api_key:
        print("\nERROR: GROQ_API_KEY not found.")
        print("Either:")
        print("  1. Create a .env file in this folder with the line:")
        print("       GROQ_API_KEY=your_key_here")
        print("  2. Or set the environment variable:")
        print("       PowerShell :  $env:GROQ_API_KEY = 'your_key_here'")
        print("       CMD        :  set GROQ_API_KEY=your_key_here")
        print("\nGet a free key at: https://console.groq.com")
        sys.exit(1)

    if not api_key.startswith("gsk_"):
        # Groq API keys always start with "gsk_" — catch obvious mistakes early
        print(f"\nWARNING: Key doesn't look like a Groq key (expected 'gsk_...').")
        print("Continuing anyway — the API call will fail if the key is wrong.")
    else:
        print(f"\n[API] Groq API key found (ends with ...{api_key[-6:]}).")

    # -------------------------------------------------------------------------
    # 7b. Chunk the hardcoded documents
    # -------------------------------------------------------------------------
    print("\n[CHUNKING] Splitting documents into chunks...")
    chunks = chunk_documents(DOCUMENTS, chunk_size=500, chunk_overlap=50)
    print(f"[CHUNKING] Created {len(chunks)} chunks from {len(DOCUMENTS)} documents.")

    # Show a sample chunk so the user understands what chunking produces
    print(f"\n[CHUNKING] Sample chunk (chunk #1 of {len(chunks)}):")
    print("-" * 40)
    print(textwrap.fill(chunks[0], width=60))
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 7c. Load HuggingFace embedding model
    # -------------------------------------------------------------------------
    embedding_model = load_embedding_model("all-MiniLM-L6-v2")

    # -------------------------------------------------------------------------
    # 7d. Embed all chunks and build the vector store (INDEXING phase)
    # -------------------------------------------------------------------------
    print("\n[INDEXING] Embedding all chunks — this is the INDEXING phase...")
    print("  (Each chunk is converted to a 384-dimensional vector.)")

    # embed_texts() returns shape (N, 384) where N = number of chunks
    chunk_embeddings = embed_texts(embedding_model, chunks)

    # Build the in-memory vector store
    vector_store = SimpleVectorStore()
    vector_store.add(chunks, chunk_embeddings)

    # -------------------------------------------------------------------------
    # 7e. Set up the Groq API client
    # -------------------------------------------------------------------------
    # Groq(api_key=...) creates an authenticated client.
    # All subsequent API calls go through this object.
    print("\n[LLM] Connecting to Groq API...")
    groq_client = Groq(api_key=api_key)
    print("[LLM] Groq client ready. Model: llama-3.1-8b-instant")

    # -------------------------------------------------------------------------
    # 7f. Interactive Q&A loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RAG SYSTEM READY — Ask questions about RAG!")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 65)

    # Suggested starter questions the user can try:
    print("\nSuggested questions to try:")
    print("  • What is RAG?")
    print("  • How does cosine similarity work?")
    print("  • What is chunking and why is it needed?")
    print("  • What is an embedding?")
    print("  • How does Groq relate to RAG?")
    print("  • What causes hallucination in LLMs?")

    while True:
        # Prompt the user for input in the terminal
        print()
        try:
            user_question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\n\n[EXIT] Interrupted. Goodbye!")
            break

        # Check for exit commands
        if user_question.lower() in ("quit", "exit", "q", ""):
            print("\n[EXIT] Goodbye! Thanks for studying RAG.")
            break

        print(f"\n[RETRIEVAL] Searching for relevant chunks for: '{user_question}'")

        # --- RETRIEVAL phase ---
        # Find the top-3 most relevant chunks from the vector store
        results = retrieve(
            query=user_question,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=3
        )

        # Show the retrieved chunks so the user can see what the LLM receives
        print("\n[RETRIEVAL] Top-3 chunks retrieved:")
        for i, r in enumerate(results, start=1):
            print(f"\n  Chunk {i} (similarity score: {r['score']:.4f}):")
            print("  " + "-" * 50)
            # textwrap.fill wraps long text to a given line width
            wrapped = textwrap.fill(r['chunk'], width=60, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            print("  " + "-" * 50)

        # --- GENERATION phase ---
        print("\n[GENERATION] Sending context + question to Groq (LLaMA-3.1-8B-Instant)...")
        try:
            answer = generate_answer(
                question=user_question,
                retrieved_chunks=results,
                groq_client=groq_client,
                model="llama-3.1-8b-instant"
            )
        except Exception as e:
            # Catch API errors (network issues, invalid key, rate limits, etc.)
            print(f"\n[ERROR] Groq API call failed: {e}")
            continue

        # Print the final answer
        print("\n" + "=" * 65)
        print("  ANSWER (generated by LLaMA3-8B via Groq):")
        print("=" * 65)
        # textwrap.fill wraps the answer to 65 chars wide for clean terminal output
        print(textwrap.fill(answer, width=65))
        print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block ensures main() only runs when the script is executed directly
# (not when it is imported as a module by another script).

if __name__ == "__main__":
    main()
