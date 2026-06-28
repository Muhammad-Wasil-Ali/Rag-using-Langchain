# 🔍 RAG Pipeline using LangChain

A fully free Retrieval-Augmented Generation (RAG) pipeline built with LangChain, OpenRouter (free LLMs), Google Gemini Embeddings, and FAISS vector store — running entirely on Google Colab.

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique where instead of relying on an LLM's training data alone, we:

1. **Store** your documents in a vector database
2. **Retrieve** the most relevant chunks when a question is asked
3. **Generate** an answer using an LLM — grounded in your actual documents

This means less hallucination, more accurate answers.

---

## 🛠️ Tech Stack (100% Free)

| Component | Tool | Cost |
|---|---|---|
| **Framework** | LangChain | Free |
| **LLM** | OpenRouter (Llama 3.3 70B / GPT-OSS 120B) | Free |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) | Free |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Free |
| **Environment** | Google Colab | Free |

---

## 🏗️ Architecture

```
Your Document (PDF/TXT)
        ↓
   TextLoader
        ↓
RecursiveCharacterTextSplitter (chunks)
        ↓
Google Gemini Embeddings
        ↓
FAISS Vector Store
        ↓
User Question → Retriever (top-k chunks)
        ↓
Prompt Template (context + question)
        ↓
OpenRouter LLM (Llama / GPT-OSS)
        ↓
Answer ✅
```

---

## 📦 Installation

```bash
pip install langchain langchain-community langchain-openai langchain-google-genai faiss-cpu pypdf -q
```

---

## 🔑 API Keys Required

### 1. OpenRouter (Free LLM)
- Go to [openrouter.ai](https://openrouter.ai)
- Sign up (no credit card needed)
- Go to **Keys** → Create Key → Copy

### 2. Google Gemini (Free Embeddings)
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **Get API Key** → Copy

> **Tip:** In Google Colab, save keys in 🔑 **Secrets** (left sidebar) instead of hardcoding them.

---

## 🚀 Usage

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google.colab import userdata

# 1. Load Document
loader = TextLoader("/content/your_file.txt")
documents = loader.load()

# 2. Split into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=userdata.get("GOOGLE_API_KEY")
)

# 4. FAISS Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Prompt Template
template = """You are a helpful assistant.
Use ONLY the given context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. OpenRouter LLM
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_key=userdata.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# 7. RAG Chain (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Ask Questions!
result = rag_chain.invoke("What is this document about?")
print(result)
```

---

## 📁 Project Structure

```
Rag-using-Langchain/
│
├── rag_pipeline.ipynb    # Main Colab notebook
├── README.md             # Project documentation
└── sample_data/          # Sample documents (optional)
```

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Install the missing package e.g. `pip install langchain-community` |
| `FileNotFoundError` | Use `/content/` path in Colab, not `./content/` |
| `AuthenticationError 401` | Check API key is correct and Secrets toggle is ON |
| `RateLimitError 429` | Switch model or wait 30 seconds — free tier limit |
| `Model ID not valid` | Check current free models at `openrouter.ai/models` |

---

## 🔄 Alternative Free Models (OpenRouter)

```python
# Try these if one is rate-limited
"meta-llama/llama-3.3-70b-instruct:free"
"openai/gpt-oss-120b:free"
"meta-llama/llama-4-scout:free"
"openrouter/free"  # auto-selects best available free model
```

---

## 👨‍💻 Author

**Muhammad Wasil Ali**
- GitHub: [@Muhammad-Wasil-Ali](https://github.com/Muhammad-Wasil-Ali)
- LinkedIn: [Muhammad Wasil Ali]((https://www.linkedin.com/in/muhammad-wasil-ali/)

---

## 📄 License

MIT License — free to use, modify, and distribute.
