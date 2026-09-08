import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Mini RAG App",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Mini RAG — Ask Questions from Your PDF")
st.write(
    "A small RAG application built to understand the complete flow: "
    "PDF → chunks → embeddings → similarity search → LLM answer."
)


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Small, free, local embedding model.
    # It runs on the Streamlit server; it is NOT the LLM.
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Groq client
# -----------------------------
def get_groq_api_key():
    # Streamlit Cloud: add GROQ_API_KEY in App Settings → Secrets.
    # Local: you can also use an environment variable.
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GROQ_API_KEY")


def get_groq_client():
    api_key = get_groq_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )
        st.stop()

    return Groq(api_key=api_key)


# -----------------------------
# 1. Extract text from PDF
# -----------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                f"[Page {page_number}]\n{text.strip()}"
            )

    return "\n\n".join(pages)


# -----------------------------
# 2. Split text into chunks
# -----------------------------
def create_chunks(text, chunk_size=800, overlap=120):
    """
    Simple character-based chunking.

    We use characters here to keep the code easy to understand.
    In production RAG systems, chunking is often token-aware.
    """

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------
# 3. Create embeddings
# -----------------------------
def create_embeddings(chunks):
    """
    Convert every chunk into a numerical vector.
    """

    return embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# -----------------------------
# 4. Retrieve relevant chunks
# -----------------------------
def retrieve_chunks(question, chunks, chunk_embeddings, top_k=3):
    """
    Convert the question to an embedding and compare it
    with all chunk embeddings using cosine similarity.

    Because embeddings are normalized, dot product = cosine similarity.
    """

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = np.dot(chunk_embeddings, question_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "chunk": chunks[index],
                "score": float(scores[index]),
            }
        )

    return results


# -----------------------------
# 5. Generate answer with LLM
# -----------------------------
def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join(
        [
            f"Source {i + 1}:\n{item['chunk']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question ONLY using the CONTEXT below.

Rules:
1. If the answer is clearly present in the context, answer it directly.
2. If the answer is not present in the context, say:
   "This information is not present in the provided PDF."
3. Do not use outside knowledge to fill missing information.
4. Do not invent facts.
5. Keep the answer clear and concise.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only supplied document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=700,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -----------------------------
# Sidebar: PDF ingestion
# -----------------------------
with st.sidebar:
    st.header("1️⃣ Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
    )

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=400,
        max_value=1500,
        value=800,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=300,
        value=120,
        step=20,
    )

    process_button = st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
    )

    if process_button:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Reading PDF → creating chunks → creating embeddings..."):
                text = extract_pdf_text(uploaded_file)
                chunks = create_chunks(
                    text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                if not chunks:
                    st.error(
                        "No readable text was found in this PDF. "
                        "Try a text-based PDF rather than a scanned image PDF."
                    )
                else:
                    embeddings = create_embeddings(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.chunk_embeddings = embeddings
                    st.session_state.document_name = uploaded_file.name

            st.success(f"Processed {len(chunks)} chunks.")


# -----------------------------
# Main area
# -----------------------------
if st.session_state.chunks:
    st.success(
        f"Document ready: **{st.session_state.document_name}** "
        f"({len(st.session_state.chunks)} chunks)"
    )

    st.subheader("2️⃣ Ask a question")

    question = st.text_input(
        "Question",
        placeholder="Example: What is the main topic of this PDF?",
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=min(5, len(st.session_state.chunks)),
        value=min(3, len(st.session_state.chunks)),
    )

    ask_button = st.button(
        "Ask",
        type="primary",
    )

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the document..."):
                retrieved = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    top_k=top_k,
                )

            # Simple relevance threshold for this learning project.
            # This helps reject clearly unrelated questions.
            best_score = retrieved[0]["score"]

            if best_score < 0.30:
                st.info(
                    "🔎 No sufficiently relevant information was found "
                    "in the PDF."
                )
            else:
                with st.spinner("Generating answer with GPT-OSS 120B..."):
                    answer = generate_answer(question, retrieved)

                st.subheader("3️⃣ Answer")
                st.write(answer)

                st.subheader("🔍 Retrieved chunks")

                for i, item in enumerate(retrieved, start=1):
                    with st.expander(
                        f"Chunk {i} — similarity score: {item['score']:.3f}"
                    ):
                        st.write(item["chunk"])

else:
    st.info(
        "Upload a PDF from the sidebar and click **Process PDF** to start."
    )


# -----------------------------
# Learning section
# -----------------------------
with st.expander("🧠 Understand the RAG flow"):
    st.markdown(
        """
### Ingestion phase

**PDF**
→ extract text
→ split text into **chunks**
→ create **embeddings**
→ keep chunks + vectors in memory

### Question phase

**User question**
→ create question **embedding**
→ compare it with chunk embeddings
→ retrieve the most similar chunks
→ send question + retrieved chunks to **GPT-OSS 120B**
→ generate the final answer

### Important

This app does **not train GPT-OSS on your PDF**.

The PDF is retrieved at question time and supplied as context to the model.
        """
    )
