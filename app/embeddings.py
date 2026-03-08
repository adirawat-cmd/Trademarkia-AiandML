from sentence_transformers import SentenceTransformer
import numpy as np

"""
Embedding module

This module is responsible for converting text into vector embeddings
using a pre-trained sentence transformer model.

## Model choice:

We use "all-MiniLM-L6-v2" because:

* It is lightweight and fast
* Produces good semantic embeddings
* Works well for similarity search tasks
  """

# Load embedding model once at startup

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(docs: list[str]) -> np.ndarray:
    """
Generate embeddings for a list of documents.

Args:
    docs (list[str]): list of text documents

Returns:
    np.ndarray: matrix of document embeddings
                shape -> (num_documents, embedding_dimension)
"""

    embeddings = model.encode(
    docs,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
    )

    return embeddings

def embed_query(query: str) -> np.ndarray:
    """
Generate embedding for a single query.

```
Args:
    query (str): natural language search query

Returns:
    np.ndarray: query embedding vector
"""

    embedding = model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True
)

    return embedding

if __name__ == "__main__":

# Quick test to verify embeddings work
    sample_docs = [
    "Gun control laws in the United States",
    "Space exploration and NASA missions",
    "Computer graphics and GPU rendering"
]

    print("Generating embeddings for sample documents...\n")

    doc_embeddings = embed_documents(sample_docs)

    print("Embedding shape:", doc_embeddings.shape)

    sample_query = "firearm regulations"

    query_embedding = embed_query(sample_query)

    print("Query embedding shape:", query_embedding.shape)
