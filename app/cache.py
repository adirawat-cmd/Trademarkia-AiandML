import numpy as np

class SemanticCache:
    """
Semantic cache that stores previous query results and retrieves
them if a new query is semantically similar.
"""

    def __init__(self, similarity_threshold: float = 0.85):

        self.cache = []
        self.similarity_threshold = similarity_threshold

    # statistics
        self.hit_count = 0
        self.miss_count = 0


    def cosine_similarity(self, a: np.ndarray, b: np.ndarray):
        """
    Compute cosine similarity between two vectors.
    """

        return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


    def lookup(self, query: str, query_embedding: np.ndarray):
        """
    Check if a semantically similar query exists in cache.
    """

        best_match = None
        best_score = 0

        for item in self.cache:

            score = self.cosine_similarity(
            query_embedding,
            item["embedding"]
        )

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= self.similarity_threshold:

            self.hit_count += 1

            return {
            "cache_hit": True,
            "matched_query": best_match["query"],
            "similarity_score": float(best_score),
            "result": best_match["result"],
            "dominant_cluster": best_match["cluster"]
        }
        self.miss_count += 1

        return None


    def store(self, query, embedding, result, cluster):
        """
    Store a query result in the cache.
    """

        entry = {
        "query": query,
        "embedding": embedding,
        "result": result,
        "cluster": cluster
    }

        self.cache.append(entry)


    def stats(self):
        """
    Return cache statistics.
    """

        total = len(self.cache)

        hit_rate = (
        self.hit_count / (self.hit_count + self.miss_count)
        if (self.hit_count + self.miss_count) > 0
        else 0
    )

        return {
        "total_entries": total,
        "hit_count": self.hit_count,
        "miss_count": self.miss_count,
        "hit_rate": round(hit_rate, 3)
    }


    def clear(self):
        """
    Clear the cache.
    """

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0

if __name__ == "__main__":

    from app.embeddings import embed_query

    cache = SemanticCache()

    q1 = "gun control laws"
    q2 = "firearm regulations"

    emb1 = embed_query(q1)
    emb2 = embed_query(q2)

# store first query
    cache.store(q1, emb1, "Example result", cluster=3)

# lookup second query
    result = cache.lookup(q2, emb2)

    print(result)

    print("\nCache stats:")
    print(cache.stats())
