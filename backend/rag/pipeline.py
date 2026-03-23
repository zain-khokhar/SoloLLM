"""
RAG Pipeline Orchestrator for SoloLLM.

Coordinates the full RAG pipeline:
1. Document upload & ingestion
2. Chunking
3. Embedding
4. Indexing (vector + keyword)
5. Retrieval (hybrid)
6. Re-ranking
7. Citation building
8. Context distillation (Phase 3)
"""

import hashlib
import logging
import uuid
from pathlib import Path

from rag.ingest import ingest_file, ParsedDocument
from rag.chunking import chunking_engine, Chunk
from rag.embeddings import embedding_engine
from rag.vectorstore import vector_store
from rag.keyword_index import keyword_index
from rag.retriever import hybrid_retriever, RetrievalResult
from rag.reranker import reranker
from rag.citations import citation_tracker, CitedContext
from core.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Full RAG pipeline orchestrator.

    Provides high-level methods for:
    - Document ingestion (upload → parse → chunk → embed → index)
    - Query processing (retrieve → rerank → cite → format context)
    """

    async def init(self):
        """Initialize all stores."""
        await vector_store.init()
        await keyword_index.init()
        await vector_store.create_workspace("default", "Default Workspace", "Your default document workspace")
        logger.info("RAG pipeline initialized")

    # ── Document Ingestion ──────────────────────────────────

    async def ingest_document(
        self,
        file_path: str,
        workspace_id: str = "default",
    ) -> dict:
        """
        Full ingestion pipeline: parse → chunk → embed → index.

        Returns a summary dict with document_id, chunk_count, etc.
        """
        # Step 1: Parse the document
        logger.info(f"Ingesting: {file_path}")
        parsed = await ingest_file(file_path)

        if parsed.errors:
            logger.error(f"Parse errors: {parsed.errors}")
            return {
                "success": False,
                "errors": parsed.errors,
                "filename": parsed.filename,
            }

        if not parsed.content.strip():
            return {
                "success": False,
                "errors": ["Document has no extractable content"],
                "filename": parsed.filename,
            }

        # Generate document ID
        content_hash = hashlib.sha256(parsed.content.encode()).hexdigest()[:16]
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Step 2: Chunk the document
        chunks = chunking_engine.chunk_document(
            content=parsed.content,
            sections=parsed.sections if parsed.sections else None,
            document_id=document_id,
            document_title=parsed.title,
        )

        max_chunks = max(100, int(getattr(settings, "rag_max_chunks_per_document", 8000)))
        if len(chunks) > max_chunks:
            logger.warning(
                "Chunk cap applied for '%s': %d -> %d",
                parsed.filename,
                len(chunks),
                max_chunks,
            )
            chunks = chunks[:max_chunks]

        if not chunks:
            return {
                "success": False,
                "errors": ["No chunks could be created from document"],
                "filename": parsed.filename,
            }

        # Step 3: Generate embeddings in batches with progress logs
        chunk_texts = [c.content for c in chunks]
        batch_size = max(8, int(getattr(settings, "rag_embedding_batch_size", 128)))
        embeddings: list[list[float]] = []
        total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
        logger.info(
            "Embedding '%s': %d chunks in %d batches (batch_size=%d, fallback=%s)",
            parsed.filename,
            len(chunk_texts),
            total_batches,
            batch_size,
            embedding_engine.is_fallback,
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunk_texts))
            batch = chunk_texts[start:end]
            batch_embeddings = embedding_engine.embed_documents(batch)
            embeddings.extend(batch_embeddings)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                logger.info(
                    "Embedding progress '%s': batch %d/%d (%d/%d chunks)",
                    parsed.filename,
                    batch_idx + 1,
                    total_batches,
                    end,
                    len(chunk_texts),
                )

        # Step 4: Prepare chunk dicts for storage
        chunk_dicts = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_dicts.append({
                "id": chunk.id,
                "document_id": document_id,
                "content": chunk.content,
                "embedding": embedding,
                "document_title": chunk.document_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "parent_chunk_id": chunk.parent_chunk_id,
                "metadata": chunk.metadata,
            })

        # Step 5: Store in vector DB + keyword index
        logger.info("Indexing '%s': writing %d chunks to stores", parsed.filename, len(chunk_dicts))
        await vector_store.add_document(
            document_id=document_id,
            filename=parsed.filename,
            file_type=parsed.file_type,
            title=parsed.title,
            content_hash=content_hash,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
            workspace_id=workspace_id,
            metadata=parsed.metadata,
        )
        await vector_store.add_chunks(chunk_dicts, workspace_id)
        await keyword_index.add_chunks(chunk_dicts, workspace_id)

        result = {
            "success": True,
            "document_id": document_id,
            "filename": parsed.filename,
            "title": parsed.title,
            "file_type": parsed.file_type,
            "chunk_count": len(chunks),
            "page_count": parsed.page_count,
            "content_length": len(parsed.content),
            "table_count": len(parsed.tables),
            "workspace_id": workspace_id,
        }

        # Phase 4: Extract entities for the knowledge graph
        try:
            from memory.knowledge_graph import knowledge_graph
            graph_result = await knowledge_graph.extract_and_store(
                parsed.content, document_id
            )
            result["entities_extracted"] = graph_result.get("entities", 0)
            result["relationships_extracted"] = graph_result.get("relationships", 0)
        except Exception as e:
            logger.warning(f"Knowledge graph extraction skipped: {e}")

        logger.info(
            f"Ingested '{parsed.filename}': {len(chunks)} chunks, "
            f"{parsed.page_count} pages, {len(parsed.content)} chars"
        )

        return result

    async def delete_document(self, document_id: str):
        """Delete a document and all its chunks from all indexes."""
        await vector_store.delete_document(document_id)
        await keyword_index.delete_document(document_id)
        logger.info(f"Deleted document: {document_id}")

    async def list_documents(self, workspace_id: str = "default") -> list[dict]:
        """List all documents in a workspace."""
        return await vector_store.list_documents(workspace_id)

    async def get_document(self, document_id: str) -> dict | None:
        """Get document details."""
        return await vector_store.get_document(document_id)

    # ── Query / Retrieval ───────────────────────────────────

    async def query(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = True,
    ) -> CitedContext:
        """
        Full RAG query pipeline: retrieve → rerank → build citations.

        Returns a CitedContext with formatted context and source references.
        """
        # Step 1: Hybrid retrieval (precision-aware)
        results, _diag = await hybrid_retriever.retrieve(
            query=query,
            workspace_id=workspace_id,
            top_k=top_k * 3,  # Over-fetch for reranking
            document_id=document_id,
            document_ids=document_ids,
            precision_mode=settings.rag_precision_mode,
            vector_min_score=settings.rag_vector_min_score,
            lexical_required_coverage=settings.rag_lexical_required_coverage,
            per_document_cap=settings.rag_per_document_cap,
            use_mmr=settings.rag_use_mmr,
            mmr_lambda=settings.rag_mmr_lambda,
            candidate_pool_size=settings.rag_candidate_pool_size,
        )

        if not results:
            return CitedContext(context_text="", citations=[], instruction="")

        # Step 2: Rerank
        if rerank:
            rerank_out = reranker.rerank(query=query, results=results, top_k=top_k)
            results = rerank_out["results"]
        else:
            results = results[:top_k]

        # Step 3: Build cited context
        cited = citation_tracker.build_cited_context(results)

        logger.info(
            f"RAG query: '{query[:50]}...', "
            f"retrieved={len(results)}, citations={len(cited.citations)}"
        )

        return cited

    async def retrieve_for_distillation(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks for use in the distillation pipeline.

        Returns raw RetrievalResult objects (not CitedContext).

        Args:
            document_ids: Filter to only these documents (thread-scoped)
        """
        results, _diag = await hybrid_retriever.retrieve(
            query=query,
            workspace_id=workspace_id,
            top_k=top_k * 3,
            document_ids=document_ids,
            precision_mode=settings.rag_precision_mode,
            vector_min_score=settings.rag_vector_min_score,
            lexical_required_coverage=settings.rag_lexical_required_coverage,
            per_document_cap=settings.rag_per_document_cap,
            use_mmr=settings.rag_use_mmr,
            mmr_lambda=settings.rag_mmr_lambda,
            candidate_pool_size=settings.rag_candidate_pool_size,
        )

        if not results:
            return []

        rerank_out = reranker.rerank(query=query, results=results, top_k=top_k)
        return rerank_out["results"]

    async def distilled_query(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        conversation_messages: list[dict] | None = None,
        base_system_prompt: str = "",
        document_ids: list[str] | None = None,
    ) -> dict:
        """
        Full distillation-enhanced RAG query.

        Runs the distillation pipeline: decompose → retrieve → dedup →
        multi-hop → compress → score confidence → adaptive prompt.

        Args:
            document_ids: Filter retrieval to only these documents (thread-scoped)

        Returns:
            dict with keys: system_prompt, confidence, query_type, sub_queries,
            compression_ratio, hops_used, cited_context
        """
        from core.distillation import distillation_pipeline

        async def _retriever(q, ws, k):
            return await self.retrieve_for_distillation(q, ws, k, document_ids=document_ids)

        result = await distillation_pipeline.process(
            query=query,
            retriever_fn=_retriever,
            workspace_id=workspace_id,
            top_k=top_k,
            conversation_messages=conversation_messages,
            base_system_prompt=base_system_prompt,
        )

        # Also build citations from the retrieval results
        cited = CitedContext(context_text="", citations=[], instruction="")
        if result.retrieval_results:
            cited = citation_tracker.build_cited_context(result.retrieval_results[:top_k])

        return {
            "system_prompt": result.system_prompt,
            "processed_context": result.processed_context,
            "confidence": result.confidence,
            "query_type": result.query_type,
            "sub_queries": result.sub_queries,
            "compression_ratio": result.compression_ratio,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "hops_used": result.hops_used,
            "chunks_before_dedup": result.chunks_before_dedup,
            "chunks_after_dedup": result.chunks_after_dedup,
            "cited_context": cited,
        }

    async def get_context_for_chat(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
    ) -> str:
        """
        Get formatted RAG context suitable for injection into a chat system prompt.
        Includes graph-augmented context from the knowledge graph.

        Returns an empty string if no relevant documents are found.
        """
        cited = await self.query(query=query, workspace_id=workspace_id, top_k=top_k)
        base_context = citation_tracker.format_system_context(cited)

        # Phase 4: Add knowledge graph context
        graph_context = ""
        try:
            from memory.knowledge_graph import knowledge_graph
            graph_context = await knowledge_graph.get_related_context(query)
        except Exception as e:
            logger.debug(f"Graph context skipped: {e}")

        if graph_context and base_context:
            return f"{base_context}\n\n{graph_context}"
        return base_context or graph_context

    # ── Workspace Management ────────────────────────────────

    async def create_workspace(self, name: str, description: str = "") -> dict:
        """Create a new workspace."""
        workspace_id = f"ws_{uuid.uuid4().hex[:12]}"
        await vector_store.create_workspace(workspace_id, name, description)
        return {"id": workspace_id, "name": name, "description": description}

    async def list_workspaces(self) -> list[dict]:
        """List all workspaces."""
        return await vector_store.list_workspaces()

    async def get_stats(self, workspace_id: str = "default") -> dict:
        """Get stats for a workspace."""
        docs = await vector_store.list_documents(workspace_id)
        chunk_count = await vector_store.get_chunk_count(workspace_id)
        backend_info = await vector_store.get_backend_info()
        return {
            "workspace_id": workspace_id,
            "document_count": len(docs),
            "chunk_count": chunk_count,
            "embedding_info": embedding_engine.get_info(),
            "vector_backend": backend_info,
        }


# Singleton
rag_pipeline = RAGPipeline()
