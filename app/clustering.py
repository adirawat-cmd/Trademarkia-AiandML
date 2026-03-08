import numpy as np
from sklearn.mixture import GaussianMixture

class FuzzyClusterer:
    """
Implements fuzzy clustering using Gaussian Mixture Models.

```
Each document receives a probability distribution over clusters
instead of belonging to a single cluster.
"""

    def __init__(self, n_clusters: int = 10):

        self.n_clusters = n_clusters
        self.model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=42
    )


    def fit(self, embeddings: np.ndarray):
        """
    Train clustering model on document embeddings.
    """

        print("Training Gaussian Mixture clustering model...")

        self.model.fit(embeddings)

        print("Clustering training complete.")


    def get_membership(self, embeddings: np.ndarray):
        """
    Returns probability distribution of each document across clusters.

    Example output for a document:
    [0.1, 0.7, 0.05, 0.15]

    Meaning:
    10% cluster 0
    70% cluster 1
    5% cluster 2
    15% cluster 3
    """

        probabilities = self.model.predict_proba(embeddings)

        return probabilities


    def dominant_cluster(self, embedding: np.ndarray):
        """
    Return the cluster with highest probability for a given embedding.
    """

        probs = self.model.predict_proba(embedding.reshape(1, -1))

        return int(np.argmax(probs))

if __name__ == "__main__":

    from data.dataset_loader import load_dataset
    from app.embeddings import embed_documents

    print("Loading dataset...")
    docs = load_dataset()

    print("Generating embeddings...")
    embeddings = embed_documents(docs[:2000])  # smaller subset for testing

    print("Initializing fuzzy clustering...")
    clusterer = FuzzyClusterer(n_clusters=10)

    clusterer.fit(embeddings)

    print("Computing cluster memberships...")

    probs = clusterer.get_membership(embeddings)

    print("\nExample cluster distribution for first document:\n")

    print(probs[0])

    print("\nDominant cluster:", np.argmax(probs[0]))
