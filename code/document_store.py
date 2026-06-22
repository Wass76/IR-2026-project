"""
Lightweight SQLite persistence for original and processed corpus documents.
"""

import json
import sqlite3
from contextlib import contextmanager

from config import BATCH_SIZE, DATASET_NAME, DOCUMENTS_DB_PATH
from utils import setup_logger

logger = setup_logger("DocumentStore")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    original_content TEXT NOT NULL,
    processed_tokens TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS corpus_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class DocumentStore:
    """Persist and retrieve original and processed documents by doc_id."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or DOCUMENTS_DB_PATH)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._ensure_schema(conn)

    @staticmethod
    def _ensure_schema(conn):
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "processed_tokens" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN processed_tokens TEXT")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _max_docs_key(max_docs):
        return "full" if max_docs is None else str(max_docs)

    @staticmethod
    def _preprocessing_meta(preprocessor):
        return {
            "preprocessing_stemming": str(preprocessor.use_stemming).lower(),
            "preprocessing_lemmatization": str(preprocessor.use_lemmatization).lower(),
        }

    def get_corpus_meta(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM corpus_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def corpus_meta_matches(self, dataset_name, max_docs):
        meta = self.get_corpus_meta()
        return (
            meta.get("dataset") == dataset_name
            and meta.get("max_docs") == self._max_docs_key(max_docs)
            and int(meta.get("doc_count", 0)) > 0
        )

    def processed_corpus_matches(self, dataset_name, max_docs, preprocessor):
        if not self.corpus_meta_matches(dataset_name, max_docs):
            return False
        meta = self.get_corpus_meta()
        if meta.get("processed_cached") != "true":
            return False
        expected = self._preprocessing_meta(preprocessor)
        return (
            meta.get("preprocessing_stemming") == expected["preprocessing_stemming"]
            and meta.get("preprocessing_lemmatization")
            == expected["preprocessing_lemmatization"]
        )

    def corpus_is_cached(self, dataset_name, max_docs, doc_count):
        if not self.corpus_meta_matches(dataset_name, max_docs):
            return False
        return int(self.get_corpus_meta().get("doc_count", 0)) == doc_count

    def _iter_document_rows(self, conn, include_processed=False):
        columns = "doc_id, original_content"
        if include_processed:
            columns += ", processed_tokens"
        offset = 0
        while True:
            rows = conn.execute(
                f"SELECT {columns} FROM documents LIMIT ? OFFSET ?",
                (BATCH_SIZE, offset),
            ).fetchall()
            if not rows:
                break
            yield from rows
            offset += len(rows)
            if offset % 10000 == 0 or len(rows) < BATCH_SIZE:
                logger.info(f"Loaded {offset} documents from store...")

    def load_corpus(self, dataset_name=DATASET_NAME, max_docs=None):
        """
        Load original documents from SQLite when the stored corpus matches.
        Returns {doc_id: text} or None.
        """
        if not self.corpus_meta_matches(dataset_name, max_docs):
            return None

        expected = int(self.get_corpus_meta().get("doc_count", 0))
        logger.info(
            f"Document store cache hit — loading {expected} originals from SQLite..."
        )
        docs = {}
        with self._connect() as conn:
            for row in self._iter_document_rows(conn, include_processed=False):
                docs[row["doc_id"]] = row["original_content"]

        if len(docs) != expected:
            logger.warning(
                f"Document store mismatch ({len(docs)} stored vs {expected} expected); "
                "will reload from ir_datasets."
            )
            return None

        logger.info(f"Loaded {len(docs)} original documents from document store.")
        return docs

    def load_corpus_bundle(self, dataset_name=DATASET_NAME, max_docs=None, preprocessor=None):
        """
        Load originals and processed tokens from SQLite when fully cached.
        Returns ({doc_id: text}, {doc_id: [tokens]}) or None.
        """
        if preprocessor is None or not self.processed_corpus_matches(
            dataset_name, max_docs, preprocessor
        ):
            return None

        expected = int(self.get_corpus_meta().get("doc_count", 0))
        logger.info(
            f"Processed corpus cache hit — loading {expected} docs from SQLite..."
        )
        docs = {}
        processed_docs = {}
        with self._connect() as conn:
            for row in self._iter_document_rows(conn, include_processed=True):
                docs[row["doc_id"]] = row["original_content"]
                if not row["processed_tokens"]:
                    logger.warning(
                        "Missing processed tokens in store; will reprocess corpus."
                    )
                    return None
                processed_docs[row["doc_id"]] = json.loads(row["processed_tokens"])

        if len(docs) != expected or len(processed_docs) != expected:
            logger.warning(
                "Processed corpus count mismatch in store; will rebuild."
            )
            return None

        logger.info(
            f"Loaded {len(docs)} original + processed documents from document store."
        )
        return docs, processed_docs

    def _write_corpus_meta(self, conn, dataset_name, max_docs, doc_count, preprocessor=None):
        meta = {
            "dataset": dataset_name,
            "max_docs": self._max_docs_key(max_docs),
            "doc_count": str(doc_count),
        }
        if preprocessor is not None:
            meta["processed_cached"] = "true"
            meta.update(self._preprocessing_meta(preprocessor))
        for key, value in meta.items():
            conn.execute(
                "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def save_corpus(
        self,
        docs,
        dataset_name=DATASET_NAME,
        max_docs=None,
        extra_metadata=None,
    ):
        """Persist original documents only."""
        doc_count = len(docs)
        if doc_count == 0:
            return 0

        if self.corpus_is_cached(dataset_name, max_docs, doc_count):
            logger.info(
                f"Document store cache hit ({doc_count} docs) — skipping re-insert."
            )
            return doc_count

        logger.info(f"Persisting {doc_count} original documents to {self.db_path}...")
        base_meta = {"dataset": dataset_name}
        if extra_metadata:
            base_meta.update(extra_metadata)
        meta_json = json.dumps(base_meta, ensure_ascii=False)
        rows = [(doc_id, content, meta_json) for doc_id, content in docs.items()]

        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM corpus_meta WHERE key LIKE 'preprocessing_%'")
            conn.execute(
                "DELETE FROM corpus_meta WHERE key = 'processed_cached'"
            )
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start : start + BATCH_SIZE]
                conn.executemany(
                    "INSERT INTO documents (doc_id, original_content, metadata) "
                    "VALUES (?, ?, ?)",
                    batch,
                )
            self._write_corpus_meta(conn, dataset_name, max_docs, doc_count)

        logger.info(f"Persisted {doc_count} original documents.")
        return doc_count

    def save_corpus_bundle(
        self,
        docs,
        processed_docs,
        dataset_name=DATASET_NAME,
        max_docs=None,
        preprocessor=None,
        extra_metadata=None,
    ):
        """Persist originals and processed tokens together."""
        doc_count = len(docs)
        if doc_count == 0:
            return 0

        if preprocessor and self.processed_corpus_matches(
            dataset_name, max_docs, preprocessor
        ):
            logger.info(
                f"Processed corpus cache hit ({doc_count} docs) — skipping re-insert."
            )
            return doc_count

        base_meta = {"dataset": dataset_name}
        if extra_metadata:
            base_meta.update(extra_metadata)
        meta_json = json.dumps(base_meta, ensure_ascii=False)

        if self.corpus_meta_matches(dataset_name, max_docs) and int(
            self.get_corpus_meta().get("doc_count", 0)
        ) == doc_count:
            logger.info("Updating processed tokens in document store...")
            rows = [
                (json.dumps(processed_docs[doc_id], ensure_ascii=False), doc_id)
                for doc_id in docs
            ]
            with self._connect() as conn:
                for start in range(0, len(rows), BATCH_SIZE):
                    batch = rows[start : start + BATCH_SIZE]
                    conn.executemany(
                        "UPDATE documents SET processed_tokens = ? WHERE doc_id = ?",
                        batch,
                    )
                self._write_corpus_meta(
                    conn, dataset_name, max_docs, doc_count, preprocessor
                )
            logger.info(f"Updated processed tokens for {doc_count} documents.")
            return doc_count

        logger.info(
            f"Persisting {doc_count} original + processed documents to {self.db_path}..."
        )
        rows = [
            (
                doc_id,
                docs[doc_id],
                json.dumps(processed_docs[doc_id], ensure_ascii=False),
                meta_json,
            )
            for doc_id in docs
        ]

        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM corpus_meta")
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start : start + BATCH_SIZE]
                conn.executemany(
                    "INSERT INTO documents "
                    "(doc_id, original_content, processed_tokens, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    batch,
                )
            self._write_corpus_meta(
                conn, dataset_name, max_docs, doc_count, preprocessor
            )

        logger.info(f"Persisted {doc_count} original + processed documents.")
        return doc_count

    def get_document(self, doc_id):
        """Return original (+ processed) document record for a doc_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id, original_content, processed_tokens, metadata "
                "FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        record = {
            "doc_id": row["doc_id"],
            "original_content": row["original_content"],
            "metadata": metadata,
        }
        if row["processed_tokens"]:
            record["processed_tokens"] = json.loads(row["processed_tokens"])
        return record

    def get_processed_tokens(self, doc_id):
        record = self.get_document(doc_id)
        if record is None:
            return None
        return record.get("processed_tokens")

    def document_count(self):
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
