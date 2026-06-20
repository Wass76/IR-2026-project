import re
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
)
from utils import setup_logger, save_json, load_json

logger = setup_logger("Embeddings")


class EmbeddingModel:
    """Dense text encoder using sentence-transformers."""

    def __init__(self, model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def prepare_text(text):
        """Minimal text prep for dense encoding (no stemming/stopwords)."""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def encode_documents(self, texts, batch_size=None):
        """
        Batch-encode a list of document texts.

        Args:
            texts: list of raw document strings
            batch_size: encoding batch size (defaults to EMBEDDING_BATCH_SIZE)

        Returns:
            float32 numpy array of shape (N, EMBEDDING_DIM)
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        prepared = [self.prepare_text(t) for t in texts]
        logger.info(f"Encoding {len(prepared)} documents (batch_size={batch_size})...")

        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, queries_dict, batch_size=None):
        """
        Encode a dict of queries {query_id: raw_text}.

        Returns:
            dict {query_id: float32 vector of shape (EMBEDDING_DIM,)}
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        if not queries_dict:
            return {}

        query_ids = list(queries_dict.keys())
        texts = [self.prepare_text(queries_dict[qid]) for qid in query_ids]
        logger.info(f"Encoding {len(texts)} queries (batch_size={batch_size})...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        return {qid: embeddings[i] for i, qid in enumerate(query_ids)}

    @staticmethod
    def build_metadata(doc_count, max_docs, model_name=None, dim=None):
        """Build metadata dict for cache invalidation."""
        return {
            "model": model_name or EMBEDDING_MODEL,
            "dim": dim or EMBEDDING_DIM,
            "doc_count": doc_count,
            "max_docs": max_docs,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def metadata_matches(cached, doc_count, max_docs, model_name=None):
        """Return True if cached metadata matches current run config."""
        if not cached:
            return False
        expected_model = model_name or EMBEDDING_MODEL
        return (
            cached.get("model") == expected_model
            and cached.get("doc_count") == doc_count
            and cached.get("max_docs") == max_docs
            and cached.get("normalize_embeddings") == NORMALIZE_EMBEDDINGS
        )

    @staticmethod
    def save_metadata(metadata, path):
        save_json(metadata, path)

    @staticmethod
    def load_metadata(path):
        try:
            return load_json(path)
        except (FileNotFoundError, OSError):
            return None


if __name__ == "__main__":
    samples = [
        "What is Python programming language?",
        "How do I learn machine learning?",
        "Best practices for information retrieval systems",
    ]
    embedder = EmbeddingModel()
    vecs = embedder.encode_documents(samples, batch_size=3)
    print(f"Shape: {vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1)
    print(f"L2 norms: {norms}")
