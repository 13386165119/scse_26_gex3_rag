"""Lecture 11 vector retrieval plus an explicit offline BM25 diagnostic.

No changes to policy_loader.py or university_it_policies.txt are necessary.
"""
from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from typing import Sequence

from policy_loader import load_policy_documents

# Required by the assignment: load the sole knowledge-base document this way.
documents = load_policy_documents()

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "our", "should", "that", "the", "their", "there", "this", "to", "university",
    "us", "use", "was", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "exact", "please", "tell",
}
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.I)


def normalize_term(word: str) -> str:
    # Tiny morphology normalizer for short university IT queries.
    # A user asking about a "password" should find policies saying "passwords".
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [normalize_term(w) for w in WORD.findall(text.casefold()) if w not in STOPWORDS]


class PolicyRetriever:
    """Small, dependency-free BM25 index. Retrieval is independent of model weights."""

    def __init__(self, policy_documents: Sequence):
        if not policy_documents:
            raise ValueError("The policy knowledge base is empty")
        # Deduplicate repeated instructional entries to avoid filling the context
        # with identical passages. Original loader outputs remain unchanged.
        unique = {}
        for doc in policy_documents:
            key = " ".join(doc.page_content.casefold().split())
            unique.setdefault(key, doc)
        self.documents = list(unique.values())
        self.term_freqs = [Counter(tokenize(doc.page_content)) for doc in self.documents]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_len = max(sum(self.lengths) / len(self.lengths), 1)
        self.doc_freq = Counter()
        for tf in self.term_freqs:
            self.doc_freq.update(tf.keys())
        self.n = len(self.documents)

    def search(self, question: str, top_k: int = 5) -> list:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = list(dict.fromkeys(tokenize(question)))
        if not terms:
            return []
        scored = []
        k1, b = 1.5, 0.75
        for idx, tf in enumerate(self.term_freqs):
            score = 0.0
            overlaps = 0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                overlaps += 1
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                score += idf * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * self.lengths[idx] / self.avg_len)
                )
            # At least two meaningful query terms are needed when the query has
            # more than one term; prevents weak, off-topic retrieval.
            needed = 1 if len(terms) == 1 else 2
            if overlaps >= needed and score > 0.0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return [self.documents[idx] for _, idx in scored[:top_k]]


retriever = PolicyRetriever(documents)


class VectorPolicyRetriever:
    """Ollama nomic embeddings and Chroma, as in Lecture 11.

    Exact duplicate content is indexed once with its first original policy ID.
    """

    def __init__(self, policy_documents: Sequence, persist_directory=None):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        if not policy_documents:
            raise ValueError("The policy knowledge base is empty")
        unique = {}
        for doc in policy_documents:
            unique.setdefault(" ".join(doc.page_content.casefold().split()), doc)
        self.documents = list(unique.values())
        digest = hashlib.sha256("\n".join(doc.page_content for doc in self.documents).encode()).hexdigest()[:16]
        self.store = Chroma(
            collection_name=f"university_support_{digest}",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434", keep_alive="30m"),
            persist_directory=persist_directory,
        )
        existing = set(self.store.get()["ids"])
        missing = [doc for doc in self.documents if doc.metadata["policy_id"] not in existing]
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            self.store.add_documents(batch, ids=[doc.metadata["policy_id"] for doc in batch])

    def search(self, question: str, top_k: int = 5) -> list:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return self.store.similarity_search(question, k=min(top_k, len(self.documents)))


def retrieve_policies(question: str, top_k: int = 5) -> list:
    return retriever.search(question, top_k=top_k)


def format_context(found_documents: Sequence) -> str:
    return "\n\n".join(
        "[Policy ID: {id}; topic: {topic}]\n{text}".format(
            id=doc.metadata.get("policy_id", "unknown"),
            topic=doc.metadata.get("topic", "unknown"),
            text=doc.page_content,
        )
        for doc in found_documents
    )
