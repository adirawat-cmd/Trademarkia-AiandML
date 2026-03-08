import faiss
import numpy as np

class VectorStore:
    """
Simple FAISS-based vector store for semantic search.

This class stores document embeddings and allows efficient
similarity search using FAISS.
"""

    def __init__(self, embeddings: np.ndarray):

    # FAISS requires float32
        self.embeddings = embeddings.astype("float32")

    # Dimension of embeddings (MiniLM produces 384-dim vectors)
        self.dimension = self.embeddings.shape[1]

    # Build FAISS index using Inner Product (works with normalized embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)

    # Add embeddings to the index
        self.index.add(self.embeddings)

        print(f"FAISS index created with {self.index.ntotal} vectors")


    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
    Perform semantic search using FAISS.

    Args:
        query_embedding: embedding vector of query
        top_k: number of nearest neighbors to retrieve

    Returns:
        scores: similarity scores
        indices: indices of matching documents
    """

        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        return scores[0], indices[0]

if __name__ == "__main__":

# Example usage test

    from data.dataset_loader import load_dataset
    from app.embeddings import embed_documents, embed_query

    print("Loading dataset...")
    docs = load_dataset()

    print("Generating embeddings...")
    doc_embeddings = embed_documents(docs[:1000])  # small subset for test

    print("Building FAISS index...")
    store = VectorStore(doc_embeddings)

    query = "space exploration missions"

    print("\nEmbedding query...")
    q_emb = embed_query(query)

    scores, indices = store.search(q_emb, top_k=5)

    print("\nTop results:\n")

    for i, idx in enumerate(indices):
        print(f"Rank {i+1} | Score: {scores[i]:.4f}")
        print(docs[idx][:200])
        print()
