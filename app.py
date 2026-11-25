import os
import re
import hashlib
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ------------------------------ App setup --------------------------------------
st.set_page_config(page_title="ResearchLens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0d111b; color: #e8edf6; }
  [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #273244; }
  .hero { padding: 1.1rem 1.3rem; border: 1px solid #283548; border-radius: 18px;
           background: linear-gradient(120deg, #17253a, #131b2a); margin-bottom: 1rem; }
  .hero h1 { margin: 0; color: #eaf2ff; font-size: 2rem; }
  .hero p { margin: .35rem 0 0; color: #9fb1c9; }
  .source-card { padding: .8rem 1rem; border: 1px solid #29384d; border-radius: 12px;
                 background: #131c2a; margin-bottom: .55rem; }
  .small-muted { color: #9aabc3; font-size: .9rem; }
  div[data-testid="stChatMessage"] { border: 1px solid #27364a; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------ Core helpers ---------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (first run may take a moment)...")
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)



def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_number, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append({"page": page_number, "text": text})
    doc.close()
    return pages


def chunk_page(text: str, page_number: int, chunk_size: int, overlap: int) -> List[Dict]:
    # Chunk by word count to preserve readable passages and avoid breaking mid-word.
    words = text.split()
    if not words:
        return []
    chunk_size = max(80, int(chunk_size))
    overlap = min(max(0, int(overlap)), chunk_size - 1)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        passage = " ".join(words[start:end]).strip()
        if passage:
            chunks.append({"text": passage, "page": page_number})
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_document_index(pdf_bytes: bytes, chunk_size: int, overlap: int, embedder) -> Tuple[List[Dict], faiss.IndexFlatIP]:
    pages = extract_pdf_pages(pdf_bytes)
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page["text"], page["page"], chunk_size, overlap))
    if not chunks:
        raise ValueError("No selectable text found in this PDF. It may be scanned; try an OCR-ready PDF.")
    texts = [item["text"] for item in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return chunks, index


def retrieve(question: str, chunks: List[Dict], index, embedder, top_k: int) -> List[Dict]:
    q_vec = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    q_vec = np.asarray(q_vec, dtype="float32")
    scores, ids = index.search(q_vec, min(top_k, len(chunks)))
    found = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)].copy()
        chunk["score"] = float(score)
        found.append(chunk)
    return found


def ask_ollama(question: str, passages: List[Dict], model: str, ollama_url: str, temperature: float) -> str:
    context = "\n\n".join(
        f"[Source {i} | page {p['page']}]\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    prompt = f"""You are ResearchLens, an assistant that answers questions using the user's research paper.
Use only the context below for factual claims about the paper. If the context does not contain enough information, say so clearly.
Cite supporting passages using [Source 1, page 2] style. Do not invent citations. Give a clear, student-friendly answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer grounded in provided passages and cite source/page. Be concise but helpful."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload, timeout=240)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "No response text was returned.").strip()



def reset_document_state():
    for key in ["chunks", "faiss_index", "doc_hash", "doc_name", "page_count", "chat_history"]:
        st.session_state.pop(key, None)


# ------------------------------ Sidebar ---------------------------------------
st.sidebar.markdown("### ⚙️ Settings")
ollama_url = st.sidebar.text_input("Ollama API URL", value="http://localhost:11434")
ollama_model = st.sidebar.text_input("Ollama model", value="llama3.2")
embedding_model_name = st.sidebar.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
top_k = st.sidebar.slider("Retrieved passages", min_value=2, max_value=8, value=4)
chunk_size = st.sidebar.slider("Chunk size (words)", min_value=100, max_value=600, value=260, step=20)
chunk_overlap = st.sidebar.slider("Chunk overlap (words)", min_value=0, max_value=120, value=50, step=10)
temperature = st.sidebar.slider("Answer creativity", 0.0, 1.0, 0.2, 0.1)
st.sidebar.caption("Models run locally through Ollama and Sentence Transformers.")

# ------------------------------ Main UI ---------------------------------------
st.markdown("""
<div class="hero">
  <h1>🔎 ResearchLens</h1>
  <p>Ask questions from your research papers — with retrieved passages and page references.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("📄 Your paper")
    uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.getvalue()
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if st.session_state.get("doc_hash") != digest:
            with st.spinner("Reading PDF, creating chunks, and building semantic index..."):
                try:
                    embedder = load_embedding_model(embedding_model_name)
                    chunks, index = build_document_index(pdf_bytes, chunk_size, chunk_overlap, embedder)
                    pages = extract_pdf_pages(pdf_bytes)
                    st.session_state["chunks"] = chunks
                    st.session_state["faiss_index"] = index
                    st.session_state["doc_hash"] = digest
                    st.session_state["doc_name"] = uploaded_pdf.name
                    st.session_state["page_count"] = len(pages)
                    st.session_state["chat_history"] = []
                except Exception as e:
                    st.error(f"Could not process this PDF: {e}")
        if st.session_state.get("doc_hash") == digest:
            st.success("Paper is ready for questions")
            st.caption(f"**{uploaded_pdf.name}** · {st.session_state.get('page_count', 0)} pages · {len(st.session_state.get('chunks', []))} text chunks")
            if st.button("Remove paper / start over", use_container_width=True):
                reset_document_state()
                st.rerun()
    else:
        st.info("Upload a PDF to begin. Text-based PDFs work best; scanned papers may need OCR first.")

with right:
    st.subheader("✨ What you can ask")
    st.markdown("- Summarize the paper's main idea\n- Explain a method in simple terms\n- Find limitations or future work\n- Compare two concepts discussed in the paper\n- Ask about a result, figure caption, or equation")
    st.markdown("<div class='small-muted'>ResearchLens retrieves relevant passages before asking the local LLM to write an answer.</div>", unsafe_allow_html=True)

st.divider()
if st.session_state.get("chunks") and st.session_state.get("faiss_index"):
    st.subheader("💬 Ask ResearchLens")
    for entry in st.session_state.get("chat_history", []):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("sources"):
                with st.expander("Retrieved passages"):
                    for n, passage in enumerate(entry["sources"], start=1):
                        st.markdown(f"**Source {n} · page {passage['page']} · similarity {passage['score']:.3f}**")
                        st.write(passage["text"])
    question = st.chat_input("Ask a question about your uploaded paper...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching passages and generating a grounded answer..."):
                try:
                    embedder = load_embedding_model(embedding_model_name)
                    passages = retrieve(question, st.session_state.chunks, st.session_state.faiss_index, embedder, top_k)
                    answer = ask_ollama(question, passages, ollama_model, ollama_url, temperature)
                    st.markdown(answer)
                    with st.expander("Retrieved passages"):
                        for n, passage in enumerate(passages, start=1):
                            st.markdown(f"**Source {n} · page {passage['page']} · similarity {passage['score']:.3f}**")
                            st.write(passage["text"])
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": passages})
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to Ollama. Make sure Ollama is installed and running, then check the API URL.")
                except requests.exceptions.Timeout:
                    st.error("Ollama took too long to respond. Try a smaller model or a shorter question.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Ollama returned an HTTP error: {e}. Check that the selected model is pulled.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
else:
    st.markdown("### 👋 Ready when you are")
    st.caption("Your question box will appear after a PDF has been processed.")


st.markdown("<div class='small-muted' style='text-align:center;margin-top:2rem'>ResearchLens · Local PDF RAG · Built with Streamlit, FAISS, Sentence Transformers, PyMuPDF & Ollama</div>", unsafe_allow_html=True)
