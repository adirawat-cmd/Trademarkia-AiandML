import re
from sklearn.datasets import fetch_20newsgroups

def clean_text(text: str) -> str:
    """
Perform light cleaning on text documents.
Steps:
- remove excessive whitespace
- remove very long sequences of punctuation
- normalize spaces
"""

# Remove multiple newlines
    text = re.sub(r"\n+", " ", text)

# Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

# Strip leading/trailing spaces
    text = text.strip()

    return text

def load_dataset(min_length: int = 50):
    """
Load and clean the 20 Newsgroups dataset.
Cleaning decisions:
1. Remove headers, footers, and quotes since they often contain
   email metadata and reply chains that do not represent the
   document's semantic content.
2. Remove extremely short documents because they do not provide
   meaningful semantic signals for embedding models.
3. Apply light normalization to reduce noise.

Returns:
    documents (list[str]): cleaned list of news posts
"""

    print("Downloading / loading 20 Newsgroups dataset...")

    dataset = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes")
    )

    raw_documents = dataset.data

    cleaned_documents = []

    for doc in raw_documents:

        if not doc:
            continue

        cleaned = clean_text(doc)

    # Filter very short documents
        if len(cleaned) < min_length:
            continue

        cleaned_documents.append(cleaned)

    print(f"Total raw documents: {len(raw_documents)}")
    print(f"Documents after cleaning: {len(cleaned_documents)}")

    return cleaned_documents


if __name__ == "__main__":
    docs = load_dataset()

    print("\nExample document:\n")
    print(docs[0][:500])  # print first 500 characters

