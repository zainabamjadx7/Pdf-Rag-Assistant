import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Page Configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

# Custom CSS to center and match image design
st.markdown("""
    <style>
    .block-container {
        max-width: 800px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-text {
        font-size: 1rem;
        color: #B0B0B0;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Subtitle as per image
st.markdown('<div class="main-title">📄 PDF RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a PDF and ask questions about its content using FAISS retrieval and an open-weight model through Groq.</div>', unsafe_allow_html=True)

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Helper Functions
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
            
    if not text.strip():
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Handle File Processing Automatically on Upload
if uploaded_file is not None and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF and creating vector database..."):
        try:
            st.session_state.vector_store = process_pdf(uploaded_file)
            if st.session_state.vector_store:
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
            else:
                st.error("Could not extract text from the uploaded PDF.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Display info box if no PDF uploaded
if uploaded_file is None:
    st.info("Upload a PDF to begin.")
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None

# Question Answering Interface
if st.session_state.pdf_processed and st.session_state.vector_store is not None:
    st.markdown("---")
    user_question = st.text_input("Ask a question about your document:")
    
    if user_question:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter your Groq API Key:", type="password")
            
        if api_key:
            with st.spinner("Searching document and generating answer..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(user_question, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    client = Groq(api_key=api_key)
                    prompt = f"""Answer the question based strictly on the provided context. If the answer is not present, state that clearly.

Context:
{context}

Question:
{user_question}
"""
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                       model="ollama launch claude --model llama3.1"
                    )
                    
                    st.markdown("### Answer")
                    st.write(chat_completion.choices[0].message.content)
                    
                    with st.expander("View Retrieved Chunks"):
                        st.write(context)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
