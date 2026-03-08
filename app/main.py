from fastapi import FastAPI
from pydantic import BaseModel

from data.dataset_loader import load_dataset
from app.embeddings import embed_documents, embed_query
from app.search import VectorStore
from app.clustering import FuzzyClusterer
from app.cache import SemanticCache

# -----------------------------

# Initialize FastAPI

# -----------------------------

app = FastAPI(title="Semantic Search with Cache")

# -----------------------------

# Load system components

# -----------------------------

print("Loading dataset...")
documents = load_dataset()

print("Generating embeddings...")
doc_embeddings = embed_documents(documents)

print("Building FAISS vector store...")
vector_store = VectorStore(doc_embeddings)

print("Training fuzzy clustering...")
clusterer = FuzzyClusterer(n_clusters=10)
clusterer.fit(doc_embeddings)

print("Initializing semantic cache...")
cache = SemanticCache(similarity_threshold=0.85)

# -----------------------------

# Request model

# -----------------------------

class QueryRequest(BaseModel):
    query: str

# -----------------------------

# POST /query

# -----------------------------

@app.post("/query")
def query_system(request: QueryRequest):
    query_text = request.query

# Generate query embedding
    query_embedding = embed_query(query_text)

# 1️⃣ Check semantic cache
    cache_result = cache.lookup(query_text, query_embedding)

    if cache_result:
        return {
        "query": query_text,
        **cache_result
    }

# 2️⃣ Cache miss → perform search
    scores, indices = vector_store.search(query_embedding, top_k=3)

    results = []

    for idx in indices:
        results.append(documents[idx][:300])

# Combine top results into one response
    combined_result = "\n\n".join(results)

# 3️⃣ Determine dominant cluster
    cluster = clusterer.dominant_cluster(query_embedding)

# 4️⃣ Store result in cache
    cache.store(
    query=query_text,
    embedding=query_embedding,
    result=combined_result,
    cluster=cluster
)

    return {
    "query": query_text,
    "cache_hit": False,
    "matched_query": None,
    "similarity_score": None,
    "result": combined_result,
    "dominant_cluster": cluster
}

# -----------------------------

# GET /cache/stats

# -----------------------------

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

# -----------------------------

# DELETE /cache

# -----------------------------

@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {
    "message": "Cache cleared successfully"
}
