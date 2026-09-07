# 📚 RAG-Based PDF Question Answering System

A high-performance **Retrieval-Augmented Generation (RAG)** application built to chat with PDF documents. This project extracts text from uploaded PDFs, generates vector embeddings using open-source models, stores them in a local **FAISS** vector database, and uses the **Groq API** (Llama 3.3 70B) for ultra-fast response generation.

---

## 🌟 Key Features

* **PDF Text Extraction**: Efficiently extracts raw text from single or multi-page PDF documents.
* **Smart Chunking**: Splits extracted text into optimal chunks with context overlap using LangChain.
* **Open-Source Embeddings**: Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for generating dense vector representations.
* **Vector Storage**: Utilizes **FAISS** (Facebook AI Similarity Search) for fast similarity search and retrieval.
* **Groq API Integration**: Leverages ultra-fast LLM inference using Groq for accurate answers based strictly on retrieved document context.
* **Streamlit Web UI**: Simple, intuitive user interface for uploading files and querying documents.

---

## 🛠️ Tech Stack

* **Frontend / Web Framework**: [Streamlit](https://streamlit.io/)
* **Text Extraction**: PyPDF2
* **RAG Framework**: [LangChain](https://www.langchain.com/)
* **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
* **Embeddings**: [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
* **LLM Provider**: [Groq API](https://groq.com/)

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Get a free API key from [Groq Console](https://console.groq.com/).

You can pass the API key directly through the Streamlit sidebar or set it as an environment variable:

```bash
# On Linux/macOS
export GROQ_API_KEY="your_groq_api_key_here"

# On Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📖 How It Works

1. **Upload PDF**: Upload your document via the sidebar.
2. **Process Document**: Click **Process Document** to extract text, divide it into chunks, and store vector embeddings in FAISS.
3. **Ask Questions**: Type your query into the text box. The system retrieves relevant chunks from FAISS and sends them alongside your question to Groq LLM to generate an accurate answer.

---

## ☁️ Deployment

This project is ready for deployment on **Streamlit Community Cloud**:

1. Push your code to GitHub (`app.py`, `requirements.txt`, and `README.md`).
2. Log in to [Streamlit Share](https://share.streamlit.io/).
3. Connect your repository and set `app.py` as the main entry point.
4. Add `GROQ_API_KEY` under **Advanced Settings > Secrets**.
5. Deploy!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
