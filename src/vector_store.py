"""
vector_store.py
Manages persistent vector storage using ChromaDB.
Handles adding, updating, and deleting document chunks with stable unique IDs,
duplicate prevention, metadata preservation, and persistence across restarts.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from src.config import config
from src.text_splitter import TextChunk
from src.embeddings import EmbeddingGenerator
from src.bm25 import BM25Index


class VectorStore:
    """
    Interface for persistent ChromaDB vector storage.
    Configured with cosine distance metric for intuitive similarity comparisons.
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the persistent ChromaDB client and collection.

        Args:
            persist_dir: Directory path for ChromaDB SQLite & HNSW files.
            collection_name: Name of the vector collection.
            embedding_generator: Optional EmbeddingGenerator instance.
        """
        # Disable anonymized telemetry pings
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        self.persist_dir = Path(persist_dir) if persist_dir else config.chroma_db_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or config.collection_name
        self.embedding_generator = embedding_generator

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or fetch collection using cosine distance
        # Cosine distance = 1.0 - Cosine Similarity
        # Range: [0.0, 2.0] (0.0 = identical direction, 1.0 = orthogonal/unrelated)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize persistent BM25 sparse keyword index
        self.bm25_path = self.persist_dir / "bm25_index.json"
        self.bm25 = BM25Index.load(self.bm25_path)

        # Synchronize BM25 if ChromaDB has records but BM25 is not populated yet
        if self.collection.count() > 0 and len(self.bm25.corpus) == 0:
            existing_data = self.collection.get(include=["documents", "metadatas"])
            e_ids = existing_data.get("ids", [])
            e_docs = existing_data.get("documents", [])
            e_metas = existing_data.get("metadatas", [])
            sync_batch = [
                {"chunk_id": e_ids[i], "text": e_docs[i], "metadata": e_metas[i] or {}}
                for i in range(len(e_ids))
            ]
            self.bm25.add_documents(sync_batch)
            self.bm25.save(self.bm25_path)

        print(
            f"[VectorStore] Connected to ChromaDB at '{self.persist_dir}'. "
            f"Collection: '{self.collection_name}' (Current records: {self.collection.count()}, BM25 indexed: {len(self.bm25.corpus)})"
        )

    def _ensure_embedding_generator(self) -> EmbeddingGenerator:
        """Lazy loader for embedding generator to avoid unnecessary overhead."""
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()
        return self.embedding_generator

    def count(self) -> int:
        """Returns total number of chunks currently stored in the collection."""
        return self.collection.count()

    def is_document_ingested(self, filename: str) -> bool:
        """Check whether chunks from a given file already exist in the store."""
        results = self.collection.get(where={"filename": filename}, limit=1)
        return len(results.get("ids", [])) > 0

    def get_document_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """
        Retrieve all stored chunk data and metadata for a specific document.
        """
        results = self.collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        chunks = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i in range(len(ids)):
            chunks.append({
                "chunk_id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            })
        return chunks

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: Optional[List[List[float]]] = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Add document chunks to the vector database.
        Prevents duplicate embedding generation if chunks are already ingested.

        Args:
            chunks: List of TextChunk objects to store.
            embeddings: Pre-computed embeddings. If None, computes them automatically.
            force_reindex: If True, replaces existing document chunks unconditionally.

        Returns:
            Number of chunks successfully added or updated.
        """
        if not chunks:
            return 0

        # Group chunks by filename to check for existing records
        chunks_by_file: Dict[str, List[TextChunk]] = {}
        for c in chunks:
            chunks_by_file.setdefault(c.filename, []).append(c)

        chunks_to_insert: List[TextChunk] = []

        for filename, file_chunks in chunks_by_file.items():
            already_ingested = self.is_document_ingested(filename)

            if already_ingested and not force_reindex:
                existing_chunks = self.get_document_chunks(filename)
                print(
                    f"[VectorStore] Document '{filename}' is already ingested "
                    f"({len(existing_chunks)} existing chunks). Skipping redundant embedding."
                )
                continue

            if already_ingested and force_reindex:
                print(f"[VectorStore] Force re-indexing '{filename}'. Removing old chunks...")
                self.delete_document(filename)

            chunks_to_insert.extend(file_chunks)

        if not chunks_to_insert:
            print("[VectorStore] No new chunks to insert.")
            return 0

        # Compute embeddings if not provided
        if embeddings is None:
            generator = self._ensure_embedding_generator()
            print(f"[VectorStore] Generating embeddings for {len(chunks_to_insert)} new chunk(s)...")
            texts = [c.text for c in chunks_to_insert]
            computed_embeddings = generator.embed_documents(texts)
        else:
            computed_embeddings = embeddings

        ids = [c.chunk_id for c in chunks_to_insert]
        documents = [c.text for c in chunks_to_insert]
        metadatas = [c.metadata for c in chunks_to_insert]

        # Use upsert to cleanly handle new and updated chunks
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=computed_embeddings,
        )

        # Update persistent BM25 index
        bm25_docs = [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            for c in chunks_to_insert
        ]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.bm25_path)

        print(
            f"[VectorStore] Successfully stored {len(chunks_to_insert)} chunk(s). "
            f"Total records in collection: {self.collection.count()}"
        )
        return len(chunks_to_insert)

    def delete_document(self, filename: str) -> int:
        """
        Delete all chunks and embeddings belonging to a specific document.
        Supports matching by filename or doc_id.

        Args:
            filename: Name or doc_id of the document file to delete.

        Returns:
            Number of chunks removed.
        """
        existing = self.collection.get(where={"filename": filename})
        ids_to_delete = existing.get("ids", [])
        if not ids_to_delete:
            existing = self.collection.get(where={"doc_id": filename})
            ids_to_delete = existing.get("ids", [])

        if not ids_to_delete:
            print(f"[VectorStore] No chunks found for document '{filename}'. Nothing to delete.")
            return 0

        self.collection.delete(ids=ids_to_delete)

        # Remove from BM25 index
        for cid in ids_to_delete:
            self.bm25.remove_document(cid)
        self.bm25.save(self.bm25_path)

        print(f"[VectorStore] Deleted {len(ids_to_delete)} chunk(s) for document '{filename}'.")
        return len(ids_to_delete)

    def update_document(self, filename: str, new_chunks: List[TextChunk]) -> int:
        """
        Update a document by replacing its existing chunks with new ones.

        Args:
            filename: Target document filename.
            new_chunks: New list of TextChunk objects.

        Returns:
            Number of new chunks added.
        """
        self.delete_document(filename)
        return self.add_chunks(new_chunks, force_reindex=True)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Summarize all unique documents currently stored in the vector database.

        Returns:
            List of summary dictionaries with keys: filename, doc_id, total_chunks, pages.
        """
        records = self.collection.get(include=["metadatas"])
        metadatas = records.get("metadatas", [])

        doc_summary: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            if not m:
                continue
            fname = m.get("filename", "unknown")
            doc_id = m.get("doc_id", fname)
            page_num = m.get("page_number", 1)
            total_pages = m.get("total_pages", 1)

            if fname not in doc_summary:
                doc_summary[fname] = {
                    "filename": fname,
                    "doc_id": doc_id,
                    "doc_name": fname,
                    "total_chunks": 0,
                    "total_pages": total_pages,
                    "pages": set(),
                }
            doc_summary[fname]["total_chunks"] += 1
            doc_summary[fname]["pages"].add(page_num)

        result = []
        for fname, data in doc_summary.items():
            result.append({
                "filename": fname,
                "doc_id": data["doc_id"],
                "doc_name": data["doc_name"],
                "total_chunks": data["total_chunks"],
                "total_pages": data["total_pages"],
                "indexed_pages": sorted(list(data["pages"])),
            })
        return result

    def get_collection_fingerprint(self, doc_ids: Optional[List[str]] = None) -> str:
        """
        Generate a deterministic version fingerprint of the vector store / selected documents.
        Enables future Redis caching to accurately invalidate or hit cache without redesigning.

        Args:
            doc_ids: Optional document scope.

        Returns:
            16-character hexadecimal version fingerprint.
        """
        import hashlib
        docs = self.list_documents()
        if doc_ids:
            doc_set = set(doc_ids)
            docs = [d for d in docs if d["filename"] in doc_set or d.get("doc_id") in doc_set]

        fingerprint_raw = "|".join(
            f"{d['filename']}:{d['total_chunks']}:{d.get('total_pages', 1)}"
            for d in sorted(docs, key=lambda x: x["filename"])
        )
        return hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()[:16]

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform approximate nearest neighbor search using cosine distance,
        with optional document scope filtering.

        Args:
            query_embedding: Normalized query vector.
            n_results: Number of nearest neighbors to retrieve.
            where: Optional metadata filter dictionary.
            doc_ids: Optional list of document filenames/IDs to scope the query.

        Returns:
            ChromaDB query results dictionary containing ids, documents, metadatas, distances.
        """
        total_records = self.collection.count()
        if total_records == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Construct or merge document scope filter
        final_where = dict(where) if where else {}
        if doc_ids:
            if len(doc_ids) == 1:
                final_where["filename"] = doc_ids[0]
            else:
                final_where["filename"] = {"$in": doc_ids}

        # n_results cannot exceed total existing records in ChromaDB
        effective_n = min(n_results, total_records)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if final_where:
            kwargs["where"] = final_where

        results = self.collection.query(**kwargs)
        return results

    def query_bm25(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for keyword matches with optional document scope filtering.

        Args:
            query: User's search query or keyword phrase.
            top_k: Max results to return.
            doc_ids: Optional list of document filenames/IDs to restrict search to.

        Returns:
            List of dicts containing chunk_id, text, metadata, and bm25_score.
        """
        ranked = self.bm25.search(query, top_k=top_k, filter_doc_ids=doc_ids)
        results = []
        for chunk_id, score in ranked:
            doc_data = self.bm25.corpus.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "text": doc_data.get("text", ""),
                "metadata": doc_data.get("metadata", {}),
                "bm25_score": score,
            })
        return results

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single chunk by its ID."""
        if chunk_id in self.bm25.corpus:
            return self.bm25.corpus[chunk_id]
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if res.get("ids"):
            return {
                "chunk_id": chunk_id,
                "text": res["documents"][0],
                "metadata": res["metadatas"][0] if res["metadatas"] else {},
            }
        return None

