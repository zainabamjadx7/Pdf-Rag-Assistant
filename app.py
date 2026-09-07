import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Page Configuration
st.set_page_config(page_title="RAG PDF Q&A App", page_icon="📚", layout="wide")

st.title("📚 RAG Based PDF Question Answering App")
st.write("Upload a PDF document, build a vector index using FAISS, and ask questions powered by Groq!")

# Sidebar for API Key input
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))

if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to proceed.")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store using FAISS and open-source HuggingFace embeddings
@st.cache_resource(show_spinner=False)
def get_vector_store(_text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(_text_chunks, embedding=embeddings)
    return vector_store

# Sidebar File Upload
st.sidebar.header("Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Process Document"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Extracting text and creating vector embeddings..."):
            raw_text = get_pdf_text(uploaded_files)
            if not raw_text.strip():
                st.error("Could not extract any text from the provided PDF(s).")
            else:
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.sidebar.success("PDF processed and vector database initialized successfully!")

# Main QA Interface
st.subheader("Ask a Question from Your Document")
user_question = st.text_input("Enter your question:")

if user_question:
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.error("Please upload and process a PDF document first!")
    elif not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar!")
    else:
        with st.spinner("Searching document context and generating answer..."):
            try:
                # Perform similarity search
                docs = st.session_state.vector_store.similarity_search(user_question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Query Groq API
                client = Groq(api_key=groq_api_key)
                prompt = f"""You are a helpful AI assistant. Answer the user's question based strictly on the provided context. If the answer cannot be found in the context, state clearly that the information is not available in the document.

Context:
{context}

Question:
{user_question}
"""
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                )
                
                answer = chat_completion.choices[0].message.content

                st.markdown("### Answer:")
                st.write(answer)

                with st.expander("View Retrieved Context"):
                    st.write(context)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
