"""
Library-backed BM25 and TF-IDF models (rank_bm25 + scikit-learn).
"""

import math
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from utils import setup_logger

logger = setup_logger("LexicalScoring")

LEXICAL_CACHE_VERSION = 3
_EMPTY_DOC_TOKEN = "__empty__"


def _tokens_to_text(tokens):
    return " ".join(_safe_tokens(tokens))


def _safe_tokens(tokens):
    """rank_bm25 divides by avg document length; empty docs can produce NaN scores."""
    return tokens if tokens else [_EMPTY_DOC_TOKEN]


def _is_valid_score(score):
    try:
        return math.isfinite(float(score)) and float(score) != 0.0
    except (TypeError, ValueError):
        return False


def _identity_preprocess(text):
    return text


def _split_tokens(text):
    return text.split()


@dataclass
class BM25Cache:
    """rank_bm25 BM25Okapi model aligned with doc_ids."""

    doc_ids: list
    model: BM25Okapi
    bm25_k1: float
    bm25_b: float

    @classmethod
    def build(cls, processed_docs, bm25_k1, bm25_b):
        doc_ids = list(processed_docs.keys())
        corpus = [_safe_tokens(processed_docs[doc_id]) for doc_id in doc_ids]
        model = BM25Okapi(corpus, k1=bm25_k1, b=bm25_b)
        return cls(
            doc_ids=doc_ids,
            model=model,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    def score(self, query_tokens):
        if not query_tokens:
            return {}
        scores = self.model.get_scores(_safe_tokens(query_tokens))
        return {
            self.doc_ids[i]: float(scores[i])
            for i in range(len(self.doc_ids))
            if _is_valid_score(scores[i])
        }


@dataclass
class TfidfCache:
    """scikit-learn TfidfVectorizer + sparse document matrix."""

    doc_ids: list
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def build(cls, processed_docs):
        doc_ids = list(processed_docs.keys())
        corpus = [_tokens_to_text(processed_docs[doc_id]) for doc_id in doc_ids]
        vectorizer = TfidfVectorizer(
            tokenizer=_split_tokens,
            preprocessor=_identity_preprocess,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True, #TF = 1 + log(TF)
            norm="l2",
        )
        matrix = vectorizer.fit_transform(corpus)
        return cls(doc_ids=doc_ids, vectorizer=vectorizer, matrix=matrix)

    def score(self, query_tokens):
        if not query_tokens:
            return {}
        query_vec = self.vectorizer.transform([_tokens_to_text(query_tokens)])
        scores = linear_kernel(query_vec, self.matrix).ravel()
        return {
            self.doc_ids[i]: float(scores[i])
            for i in range(len(self.doc_ids))
            if _is_valid_score(scores[i])
        }
