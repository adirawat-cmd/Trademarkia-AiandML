# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight **semantic search system** built on the **20 Newsgroups dataset**.
The system enables users to query a corpus of documents using natural language while efficiently reusing previously computed results through a **semantic caching mechanism**.

The system includes:

* Sentence embeddings for semantic understanding
* A FAISS-based vector store for efficient similarity search
* Fuzzy clustering to capture overlapping semantic topics
* A custom semantic cache implemented from first principles
* A FastAPI service exposing the system via REST API

---

# System Architecture
```
Dataset
↓
Text Cleaning & Preprocessing
↓
Sentence Embeddings (Sentence Transformers)
↓
Vector Store (FAISS)
↓
Fuzzy Clustering (Gaussian Mixture Model)
↓
Semantic Cache
↓
FastAPI Service
```
---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 news articles across 20 categories**.

Preprocessing decisions:

* Removed **headers, footers, and quoted replies** to eliminate email metadata and discussion chains
* Removed **very short documents** that do not provide useful semantic content
* Normalized whitespace to reduce noise

These steps ensure the embedding model receives meaningful textual input.

---

# Embedding Model

The system uses the **all-MiniLM-L6-v2** model from Sentence Transformers.

Reasons for choosing this model:

* Lightweight and fast
* Produces high-quality semantic embeddings
* Suitable for similarity search tasks
* 384-dimensional vector representation

Each document and query is converted into a **dense vector embedding** that captures semantic meaning.

---

# Vector Database (FAISS)

Document embeddings are stored in a **FAISS index** to enable fast similarity search.

FAISS was chosen because:

* It provides efficient nearest-neighbor search for high-dimensional vectors
* It scales well to large datasets
* It is widely used in production semantic search systems

The system uses **Inner Product search**, which corresponds to cosine similarity when embeddings are normalized.

---

# Fuzzy Clustering

To uncover the semantic structure of the corpus, the system uses a **Gaussian Mixture Model (GMM)**.

Unlike traditional clustering, which assigns each document to a single cluster, GMM produces **soft cluster memberships**.

Example:

A document may belong to:

* 70% politics
* 20% firearms
* 10% law

This probabilistic representation better captures the reality that documents often span multiple topics.

The dominant cluster for each query is determined by selecting the cluster with the highest probability.

---

# Semantic Cache

Traditional caches rely on exact query matches, which fail when the same question is phrased differently.

This system implements a **semantic cache** that compares query embeddings.

Example:

Query 1:
"What are gun control laws?"

Query 2:
"What are firearm regulations?"

These queries produce **similar embeddings**, allowing the cache to detect semantic similarity and reuse previously computed results.

The cache works as follows:

1. Generate embedding for the incoming query
2. Compare it with stored query embeddings
3. If similarity exceeds a threshold (0.85), return cached result
4. Otherwise compute the result and store it in the cache

Cache statistics are also tracked, including:

* total entries
* hit count
* miss count
* hit rate

---

# API Endpoints

The system exposes the following FastAPI endpoints.

## POST /query

Submit a natural language query.

Request:

{
"query": "space exploration missions"
}

Response:

{
"query": "...",
"cache_hit": true,
"matched_query": "...",
"similarity_score": 0.91,
"result": "...",
"dominant_cluster": 3
}

---

## GET /cache/stats

Returns cache statistics.

Example response:

{
"total_entries": 10,
"hit_count": 4,
"miss_count": 6,
"hit_rate": 0.4
}

---

## DELETE /cache

Clears all cached entries and resets statistics.

---

# Running the Project

## 1. Create a virtual environment

python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

---

## 2. Install dependencies

pip install -r requirements.txt

---

## 3. Run the API

uvicorn app.main:app --reload

---

## 4. Open API documentation

http://127.0.0.1:8000/docs

FastAPI automatically generates an interactive Swagger UI for testing endpoints.

---

# Key Design Decisions

Embedding Model
Chosen for speed and semantic quality.

Vector Store
FAISS provides efficient similarity search.

Clustering
Gaussian Mixture Model allows probabilistic cluster membership.

Semantic Cache
Implemented from scratch to recognize semantically similar queries rather than exact matches.

---

# Future Improvements

Possible extensions include:

* Persistent vector storage
* Approximate nearest neighbor indexes for larger datasets
* Adaptive cache thresholds
* Distributed caching for large-scale systems

---

# Author

Adi Rawat
