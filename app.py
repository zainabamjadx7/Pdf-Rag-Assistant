import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, build a FAISS vector index, and ask questions using an open-weight LLM through Groq.")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for item in pages:
        text = item["text"]
        page_number = item["page"]

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_number,
                })

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# -----------------------------
# FAISS indexing
# -----------------------------
def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(query, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(question, retrieved_chunks, client):
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the supplied document context.
If the answer is not present in the context, clearly say that the
information was not found in the uploaded document.

Do not invent facts or sources.
When useful, mention the page number(s) supporting the answer.
"""

    user_prompt = f"""DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{LLM_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}`")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP}`")
    st.write(f"**Top-K retrieval:** `{TOP_K}`")

    st.divider()

    if st.session_state.index is not None:
        st.success(
            f"Indexed: {st.session_state.document_name}\n\n"
            f"{len(st.session_state.chunks)} chunks"
        )
    else:
        st.info("Upload and process a PDF to create the vector index.")


# -----------------------------
# Upload + indexing
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned/image-only PDFs may require OCR.",
)

if uploaded_file is not None:
    if (
        st.session_state.document_name != uploaded_file.name
        or st.session_state.index is None
    ):
        if st.button("🔨 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting PDF text..."):
                pages = extract_pdf_text(uploaded_file)

            if not pages:
                st.error(
                    "No selectable text was found. This app currently works with "
                    "text-based PDFs; scanned PDFs need an OCR step."
                )
                st.stop()

            with st.spinner("Creating chunks..."):
                chunks = chunk_text(pages)

            with st.spinner("Creating embeddings and FAISS index..."):
                embedding_model = load_embedding_model()
                index, _ = build_faiss_index(chunks, embedding_model)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []

            st.success(
                f"Done! Extracted {len(pages)} pages and created "
                f"{len(chunks)} chunks."
            )


# -----------------------------
# Chat
# -----------------------------
if st.session_state.index is not None:
    st.divider()
    st.subheader("💬 Ask questions about your PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the uploaded document...")

    if question:
        client = load_groq_client()

        if client is None:
            st.error(
                "GROQ_API_KEY is missing. Add it to Streamlit Secrets before "
                "asking questions."
            )
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                embedding_model = load_embedding_model()

                retrieved = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    embedding_model,
                )

                answer = generate_answer(
                    question,
                    retrieved,
                    client,
                )

            st.markdown(answer)

            with st.expander("🔎 Retrieved sources"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} — Page {item['page']} — "
                        f"Similarity: {item['score']:.3f}**"
                    )
                    st.write(item["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👆 Upload a PDF and click **Process PDF** to get started.")
