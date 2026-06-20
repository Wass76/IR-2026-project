import faiss
import numpy as np

from config import EMBEDDING_DIM
from utils import setup_logger, save_json, load_json

logger = setup_logger("VectorIndex")


class VectorIndex:
    """FAISS-based vector index for dense retrieval (cosine via inner product)."""

    def __init__(self):
        self.index = None
        self.doc_ids = []

    def build(self, doc_ids, embeddings):
        """
        Build FAISS IndexFlatIP from document embeddings.

        Args:
            doc_ids: ordered list of document IDs aligned with embedding rows
            embeddings: float32 array of shape (N, dim)
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != embeddings rows ({embeddings.shape[0]})"
            )

        dim = embeddings.shape[1]
        self.doc_ids = list(doc_ids)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"Vector index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query_vector, top_k=10):
        """
        Search for nearest neighbors.

        Args:
            query_vector: 1-D float32 array of shape (dim,)
            top_k: number of results to return

        Returns:
            list of (doc_id, score) tuples ranked by descending score
        """
        if self.index is None or not self.doc_ids:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    def save(self, index_path, doc_ids_path):
        """Persist FAISS index and doc-id mapping."""
        if self.index is None:
            raise RuntimeError("Cannot save: index not built.")
        faiss.write_index(self.index, str(index_path))
        save_json(self.doc_ids, doc_ids_path)
        logger.info(f"Vector index saved to {index_path}")

    def load(self, index_path, doc_ids_path):
        """Load FAISS index and doc-id mapping."""
        self.index = faiss.read_index(str(index_path))
        self.doc_ids = load_json(doc_ids_path)
        logger.info(f"Vector index loaded: {self.index.ntotal} vectors from {index_path}")

    @property
    def total_docs(self):
        return self.index.ntotal if self.index else 0

    @property
    def dim(self):
        return self.index.d if self.index else EMBEDDING_DIM


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    doc_ids = ["d1", "d2", "d3"]
    embeddings = rng.standard_normal((3, 8)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    vi = VectorIndex()
    vi.build(doc_ids, embeddings)
    query = embeddings[0]
    print(vi.search(query, top_k=2))
