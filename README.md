# 🧠 Alex AI — Production-Grade RAG Assistant (LangChain + Custom Retrieval)

A **production-ready Retrieval-Augmented Generation (RAG) system** built to simulate a real-world AI assistant that represents a candidate’s professional profile.

This project combines **LangChain capabilities** with a **custom high-performance retrieval pipeline**, demonstrating strong system design beyond basic framework usage.

---

## 🎯 Project Goal

To build a **real-time, streaming AI assistant** that:

* Answers recruiter-style questions
* Uses **grounded resume data**
* Minimizes hallucinations
* Exposes **evaluation metrics per response**

---

## ⚡ Key Highlights

* 🔥 **Custom Multi-Query Retrieval (Beyond LangChain Retriever)**
* ⚡ **Streaming LLM Responses (low latency UX)**
* 🧠 **Semantic Cache (fast repeated responses)**
* 📊 **RAG Evaluation Metrics (context relevance, faithfulness, hallucination)**
* 🧵 **Async + Non-blocking architecture**
* 🧪 **Production-grade retry + rate limit handling**

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Query Routing (LLM)
   ↓
Multi-Query Generation
   ↓
Custom Vector Store (FAISS)
   ↓
Reranking + Filtering
   ↓
Prompt Construction
   ↓
LLM (Streaming)
   ↓
Evaluation Metrics
```

---

## 🧠 Why Not Just LangChain?

This project intentionally **goes beyond default LangChain abstractions**:

| Feature         | LangChain Default | This System             |
| --------------- | ----------------- | ----------------------- |
| Retriever       | Basic similarity  | 🔥 Multi-query + rerank |
| Context Quality | Moderate          | High                    |
| Control         | Limited           | Full                    |
| Evaluation      | None              | Built-in                |
| Streaming       | Basic             | Optimized               |

---

## ⚙️ Tech Stack

* **Backend**: FastAPI (async streaming)
* **LLM**: OpenRouter / Gemini API
* **Embeddings**: HuggingFace (MiniLM)
* **Vector DB**: FAISS
* **Framework**: LangChain (partial usage)
* **Observability**: LangSmith (optional)

---

## 🔥 Core Features

### 1. Multi-Query Retrieval

Generates multiple semantic variations of a query to improve recall.

```text
"What are your skills?"
→ ["technical skills", "technologies used", "tools & frameworks"]
```

---

### 2. Reranking Engine

* Combines results from multiple queries
* Deduplicates chunks
* Ranks using similarity scores

---

### 3. Semantic Cache

* Avoids recomputation for similar queries
* Improves latency significantly

---

### 4. Streaming Responses

* Token-by-token output
* TOFT (Time to First Token)
* TOLT (Time to Last Token)

---

### 5. RAG Evaluation Metrics

Each response includes:

* **Context Relevance**
* **Answer Relevance**
* **Faithfulness**
* **Hallucination Risk**

---

## 📊 Example Metrics

```json
{
  "context_score": 0.72,
  "answer_relevance": 0.81,
  "faithfulness": 0.78,
  "hallucination": "LOW"
}
```

---

## 🚀 How It Works

1. User sends query
2. LLM classifies intent + generates multi-queries
3. FAISS retrieves relevant chunks
4. System reranks and filters context
5. Prompt is constructed dynamically
6. LLM streams response
7. Evaluation metrics are computed

---

## 🧩 Key Engineering Decisions

### ✅ Replaced LangChain Retriever

* Implemented **custom multi-query + reranking**
* Improved context relevance from ~0.3 → ~0.7+

---

### ✅ Lazy Model Loading

* Prevents deployment failure on low-resource environments
* Enables smooth cloud deployment (Render)

---

### ✅ Async + Non-blocking Design

* Uses `asyncio` + streaming generators
* Handles concurrent requests efficiently

---

## 📁 Project Structure

```
server/
├── core/
│   ├── assistant.py
│   ├── session_manager.py
│
├── rag/
│   ├── vector_store.py
│   ├── rag_chain.py
│   ├── document_processor.py
│
├── services/
│   ├── semantic_cache.py
│   ├── rag_evaluator.py
│
├── llm/
│   ├── streaming_llm.py
│   ├── langchain_llm.py
│
├── main.py
```

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload
```

---

## 🌍 Deployment Notes

* Uses **lazy loading** to avoid cold start failures
* Compatible with **Render / low-CPU environments**
* Handles **rate limits and retries gracefully**

---

## 💡 What This Demonstrates (SDE-2 Level)

* System design over framework dependency
* Async + streaming architecture
* Retrieval optimization (multi-query + rerank)
* Observability and evaluation
* Production-ready error handling

---

## 📌 Future Improvements

* Hybrid search (BM25 + vector)
* Cross-encoder reranking
* Persistent FAISS index
* Memory-aware conversations


---

## ⭐ Final Note

> This project starts with LangChain — but intentionally evolves beyond it to demonstrate deeper system-level thinking.

---
