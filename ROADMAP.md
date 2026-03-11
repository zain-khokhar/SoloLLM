# SoloLLM — Project Roadmap & Architecture Draft (v2.0)

> A local-first, GPU+CPU hybrid LLM platform — like AnythingLLM, but faster, leaner, smarter, and built to make small models produce enterprise-grade outputs.

---

## 1. Vision

Build a **self-hosted, privacy-first** application that lets users chat with LLMs, ingest documents (RAG), manage knowledge bases, and run agents — all locally. The core thesis: **a 7B model with the right infrastructure can outperform a raw 70B model** on document-heavy tasks.

| Area | AnythingLLM | SoloLLM (Ours) |
|---|---|---|
| **Inference** | Delegates to external engines (Ollama, LM Studio, etc.) | **Built-in inference engine** with GPU/CPU shared-memory support |
| **Memory** | Basic vector store | **Tiered memory** — vector + graph + keyword + compressed long-term memory |
| **RAG pipeline** | Single-pass retrieval | **Multi-hop RAG** with re-ranking, citation tracking, and recursive summarization |
| **Small model boost** | None | **Context Distillation Engine** — compresses retrieved context so small models get more signal per token |
| **Response limits** | Hard stop at token limit | **Auto-continuation** — detects truncation and offers seamless resume from exact stop point |
| **Document handling** | Basic parsing | **Deep document intelligence** — layout-aware parsing, table extraction, OCR, hierarchical chunking |
| **Agents** | Basic tool-use | **Multi-agent orchestration** with plans, loops, tool chaining, and self-reflection |
| **UI** | Electron/web | **Lightweight web UI** (SvelteKit or Next.js) + optional desktop via Tauri |
| **Performance** | No shared-memory optimization | **GPU↔CPU shared-memory inference** for machines with limited VRAM |
| **Plugin system** | Limited | **Hot-loadable plugin architecture** for models, tools, and data sources |
| **Output quality** | Depends entirely on model size | **Output amplification pipeline** — chain-of-density prompting, iterative refinement, self-verification |

---

## 2. Core Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Web / Desktop UI                           │
│               (SvelteKit or Next.js + Tauri)                 │
│   ┌──────────────────────────────────────────────────────┐   │
│   │  Streaming Chat · Doc Manager · Agent Dashboard      │   │
│   │  Response Continuation UI · Memory Inspector         │   │
│   └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │ REST / WebSocket / SSE
┌────────────────────────▼─────────────────────────────────────┐
│                   API Gateway (FastAPI)                        │
│         Auth · Rate-limit · Session · Request Router          │
└──┬──────────┬──────────┬──────────┬───────────┬──────────────┘
   │          │          │          │           │
   ▼          ▼          ▼          ▼           ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
│ Chat │ │  RAG   │ │ Agent  │ │  Admin   │ │  Output    │
│  Svc │ │Pipeline│ │Orchestr│ │ & Config │ │  Amplifier │
└──┬───┘ └───┬────┘ └───┬────┘ └──────────┘ └─────┬──────┘
   │         │          │                          │
   ▼         ▼          ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│               Inference Engine (Python/C++)                    │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  llama.cpp / vLLM / CTransformers bindings             │   │
│  │  GPU (CUDA/ROCm/Metal) ◄──shared mem──► CPU            │   │
│  │  Auto-Continuation Controller · Token Budget Manager    │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              Context Distillation Engine                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Chunk       │  │ Relevance    │  │ Compression &       │  │
│  │ Prioritizer │  │ Scorer       │  │ Summarization       │  │
│  └─────────────┘  └──────────────┘  └─────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                    Storage Layer                               │
│  ┌──────────────┐  ┌────────────┐  ┌───────────────────────┐ │
│  │ Vector DB    │  │ SQLite /   │  │ Graph Store           │ │
│  │ (Qdrant /    │  │ Postgres   │  │ (NetworkX / neo4j-   │ │
│  │  ChromaDB)   │  │            │  │  embedded)            │ │
│  ├──────────────┤  ├────────────┤  ├───────────────────────┤ │
│  │ Tiered Index │  │ Session &  │  │ Entity-Relation       │ │
│  │ Manager      │  │ Memory DB  │  │ Knowledge Graph       │ │
│  └──────────────┘  └────────────┘  └───────────────────────┘ │
│  ┌──────────────┐  ┌────────────────────────────────────────┐ │
│  │ File / Blob  │  │ Compressed Long-Term Memory Store      │ │
│  │ Store        │  │ (Summarized conversation embeddings)   │ │
│  └──────────────┘  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. GPU + CPU Shared-Memory Strategy

This is the primary performance differentiator.

### 3.1 Problem
Most consumer machines have 4–12 GB VRAM. Large models (13B+) don't fit entirely in VRAM.

### 3.2 Solution — Layer-Split Inference
- **Model layer partitioning**: Automatically profile available VRAM and RAM, then split transformer layers across GPU and CPU.
- **Shared-memory IPC**: Use OS-level shared memory (`mmap` on Linux/macOS, `CreateFileMapping` on Windows) to pass activation tensors between GPU and CPU layers with **zero-copy** where possible.
- **Adaptive offloading**: Dynamically move layers to/from GPU based on real-time memory pressure (e.g., when a user opens another GPU-heavy app).
- **Quantization-aware splitting**: Prefer offloading less-critical layers (early layers) to CPU in lower precision (Q4) while keeping attention-heavy layers on GPU in higher precision (Q5/Q6).
- **KV-cache optimization**: Use paged attention (inspired by vLLM) for the KV cache — allocate non-contiguous memory blocks so long contexts don't cause OOM. This is critical for heavy PDF processing.

### 3.3 Implementation
- Wrap **llama.cpp** (via `llama-cpp-python`) which already supports `n_gpu_layers` splitting.
- Add an **auto-profiler** that benchmarks the system on first run and selects optimal layer split.
- Expose a config knob: `gpu_memory_budget_mb` — the engine auto-adjusts around this target.
- Implement a **model warm-start cache** — keep the last-used model's weights in shared memory so switching back is instant.

---

## 4. Advanced Vector Memory System

This is what makes small models handle heavy documents. The goal: **store smarter, retrieve less, but retrieve the right things**.

### 4.1 Tiered Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   QUERY ARRIVES                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 1: Hot Cache (In-Memory)                          │
│  ── Recently accessed chunks, pinned chunks             │
│  ── Latency: <1ms                                       │
│  ── Eviction: LRU with relevance weighting              │
└────────────────────────┬────────────────────────────────┘
                         │ miss
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 2: Vector Index (Qdrant / ChromaDB)               │
│  ── All document chunks with embeddings                 │
│  ── HNSW index for approximate nearest neighbor         │
│  ── Metadata filtering (date, source, section, type)    │
│  ── Latency: 5-50ms                                     │
└────────────────────────┬────────────────────────────────┘
                         │ augmented with
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 3: Keyword Index (BM25 via tantivy / SQLite FTS5) │
│  ── Full-text search for exact matches                  │
│  ── Catches what embeddings miss (names, codes, IDs)    │
│  ── Fused with vector results via Reciprocal Rank Fusion│
└────────────────────────┬────────────────────────────────┘
                         │ enriched by
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 4: Knowledge Graph (NetworkX / embedded neo4j)    │
│  ── Entity→Relation→Entity triples                      │
│  ── Traversal for multi-hop questions                   │
│  ── "Who is the manager of the person who signed X?"    │
└────────────────────────┬────────────────────────────────┘
                         │ combined with
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 5: Compressed Long-Term Memory                    │
│  ── Summarized embeddings of past conversations         │
│  ── Rolling window compression: keep recent in full,    │
│     older turns as summaries, oldest as embeddings-only │
│  ── Allows infinite conversation history                │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Smart Chunking Strategy

Instead of naive fixed-size chunking, use a **hierarchical multi-strategy** approach:

| Strategy | When | How |
|---|---|---|
| **Structural chunking** | PDFs, DOCX, HTML | Split by headings, sections, paragraphs — respect document hierarchy |
| **Semantic chunking** | All text | Use embedding similarity: when adjacent sentences diverge in meaning, split there |
| **Table-aware chunking** | Tables in PDFs/DOCX | Extract tables as structured data (markdown), embed as complete units |
| **Sliding window overlap** | Dense text | 20% overlap between chunks to prevent information loss at boundaries |
| **Parent-child linking** | All documents | Each chunk stores a reference to its parent section and sibling chunks, so retrieval can "expand" context on demand |

### 4.3 Embedding Optimizations

- **Matryoshka embeddings**: Use models that support variable-dimension embeddings (e.g., `nomic-embed-text-v1.5`). Store full 768-dim vectors but search using truncated 256-dim for speed, then re-rank with full dimensions.
- **Quantized embeddings**: Store embeddings as `int8` or `binary` quantized vectors — 4-32x less memory, negligible quality loss. This is what allows managing thousands of heavy PDFs locally.
- **Batch embedding pipeline**: Process document chunks in GPU-accelerated batches, not one at a time. Use the same GPU/CPU split strategy as inference.
- **Incremental indexing**: When new documents are added, only embed and index the new chunks — never re-process existing data.

---

## 5. Context Distillation Engine (Small Model Supercharger)

This is the key innovation that makes small models (3B–7B) produce outputs rivaling much larger models on RAG tasks.

### 5.1 Problem
Small models have limited context windows (2K–8K tokens). If you stuff 20 retrieved chunks into the prompt, the model drowns — it can't find the relevant information in all that noise.

### 5.2 Solution — Retrieve → Distill → Generate

```
User Query
    │
    ▼
Retrieve 20-50 candidate chunks (cast a wide net)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Cross-Encoder Re-ranking          │
│  Score each chunk against the query          │
│  Keep top-K most relevant (e.g., top 10)    │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Context Compression               │
│  For each chunk, extract ONLY the sentences │
│  that are relevant to the query.            │
│  Tool: Small extractive model or rule-based │
│  sentence scoring (TF-IDF against query).   │
│  Reduces each chunk by ~60-70%.             │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Deduplication & Ordering          │
│  Remove redundant information across chunks │
│  Order by document structure, not relevance │
│  (preserves narrative flow)                 │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
Distilled context (fits in 1-2K tokens)
    │
    ▼
Small model generates HIGH QUALITY answer
(because every token of context is relevant)
```

### 5.3 Why This Works
- A 7B model with 2K tokens of **perfectly relevant** context outperforms a 70B model with 8K tokens of **mostly irrelevant** context.
- The compression is computationally cheap — cross-encoder re-ranking and sentence extraction are fast operations.
- The user never sees this pipeline — they just see a small model giving suspiciously good answers.

---

## 6. Auto-Continuation System (Never Lose an Answer)

### 6.1 Problem
Small models have limited output token budgets (often 512–2048 tokens). When generating long answers (document summaries, detailed analyses), the model hits its limit mid-sentence. Users lose the rest of the answer.

### 6.2 Solution — Intelligent Auto-Continuation

```
┌─────────────────────────────────────────────────────────┐
│  INFERENCE ENGINE — Token Budget Manager                 │
│                                                          │
│  1. Before generation, estimate required output length   │
│     based on query type:                                 │
│     • Simple question → ~200 tokens                     │
│     • Summary request → ~800-2000 tokens                │
│     • "Explain in detail" → ~1500-4000 tokens           │
│                                                          │
│  2. If estimated output > model's max_tokens:            │
│     • Set continuation_mode = true                       │
│     • Generate up to token limit                         │
│     • Detect if output is COMPLETE or TRUNCATED:         │
│       ── Check for sentence completion                   │
│       ── Check for structural markers (headers, lists)   │
│       ── Check for concluding phrases                    │
│                                                          │
│  3. If TRUNCATED:                                        │
│     • Save generation state (KV cache snapshot)          │
│     • Show user: "Response was truncated. Continue?"     │
│                                                          │
│  4. On "Continue":                                       │
│     • Restore KV cache (no re-processing of prompt!)     │
│     • If KV cache unavailable: rebuild context with      │
│       original prompt + generated text so far            │
│     • Resume generation from exact stop point            │
│     • Stitch output seamlessly in the UI                 │
│                                                          │
│  5. Repeat until answer is complete or user stops        │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Smart Continuation Features
- **KV-cache preservation**: When possible, keep the key-value cache alive between continuations so the model doesn't have to re-process the entire conversation. This makes continuation nearly instant.
- **Context window management**: If the accumulated output grows too long for the context window, use a **sliding window** approach — summarize earlier parts and keep the most recent output in full context.
- **Automatic mode**: User can toggle "auto-continue" to automatically resume without asking. The UI shows a progress indicator: "Part 2 of ~3 generating..."
- **Structured continuation**: For multi-section outputs (summaries, reports), the system detects the structure and continues section-by-section, ensuring no section is cut off.

---

## 7. Output Amplification Pipeline (Small Model, Big Results)

Additional techniques to maximize output quality from small models:

### 7.1 Chain-of-Density Prompting
Instead of asking the model to generate a final answer directly, use iterative densification:
1. Generate a rough draft answer.
2. Ask the model: "Make this more specific, adding key details from the source."
3. Repeat 1-2 times. Each pass adds density without increasing length much.
4. The user sees only the final, polished version.

### 7.2 Self-Verification Loop
After generating an answer:
1. Extract factual claims from the answer.
2. Check each claim against the retrieved source chunks.
3. If a claim isn't supported, regenerate that part with explicit instructions to stick to sources.
4. Flag any answer that couldn't be fully verified with a confidence indicator.

### 7.3 Query Decomposition
For complex questions, break them down:
- "Compare the financial performance of Q1 and Q3" becomes:
  1. "What was the financial performance in Q1?"
  2. "What was the financial performance in Q3?"
  3. "Compare these results."
- Each sub-query gets its own retrieval pass, reducing the chance of missing information.

### 7.4 Adaptive Prompt Templates
- Maintain a library of optimized prompt templates per task type (Q&A, summary, analysis, extraction, comparison).
- Automatically select the right template based on query classification.
- Templates are tuned for small models — shorter system prompts, clearer instructions, explicit output format requirements.

---

## 8. Deep Document Intelligence

### 8.1 Layout-Aware PDF Processing
- Use **PyMuPDF (fitz)** + **pdfplumber** for text extraction with positional metadata.
- Detect headers, footers, page numbers, columns, sidebars — and exclude noise.
- **Table extraction**: Detect tables using layout analysis, extract as structured markdown or CSV. Tables are embedded as complete units, not split across chunks.
- **Image/figure extraction**: Pull embedded images, store separately, and link them to the surrounding text chunks. Future multimodal models can process these.

### 8.2 OCR Pipeline (for scanned PDFs)
- Integrate **Tesseract OCR** (via `pytesseract`) or **EasyOCR** as a fallback for scanned documents.
- OCR results are post-processed with language model correction to fix common OCR errors.
- Scanned pages are flagged in metadata so the UI can warn about potential quality issues.

### 8.3 Document Hierarchy Preservation
```
Document: "Q3 Financial Report.pdf"
├── Metadata: {author, date, pages, type: "financial_report"}
├── Section: "Executive Summary" (pages 1-2)
│   ├── Chunk 1: "Revenue grew 15%..."
│   └── Chunk 2: "Key risks include..."
├── Section: "Revenue Breakdown" (pages 3-7)
│   ├── Chunk 3: "North America: $2.1B..."
│   ├── Table: "Revenue by Region" (structured data)
│   └── Chunk 4: "APAC growth driven by..."
└── Section: "Outlook" (pages 8-9)
    ├── Chunk 5: "We expect Q4 revenue..."
    └── Chunk 6: "Capital expenditure planned..."
```
- Every chunk knows its **document → section → subsection → position**.
- Retrieval can **expand context**: if Chunk 3 is relevant, the system can optionally pull the parent section header and sibling chunks for better context.

---

## 9. Feature Breakdown

### Phase 1 — Foundation (Sprint 1)
- [ ] **Project scaffolding**: Python (FastAPI) backend, SvelteKit frontend, Tauri shell
- [ ] **Inference engine wrapper**: llama.cpp integration with GPU/CPU layer splitting
- [ ] **Auto-profiler**: Detect GPU, VRAM, RAM; auto-set optimal `n_gpu_layers` and quantization
- [ ] **Model manager**: Download, quantize, and manage GGUF models (HuggingFace integration)
- [ ] **Basic chat UI**: Streaming responses via SSE, conversation history, model selection
- [ ] **Token budget manager**: Track token usage per response, detect truncation
- [ ] **Auto-continuation system**: Detect truncated responses, offer "Continue" button, stitch seamlessly
- [ ] **SQLite storage**: Conversations, settings, user preferences (via `aiosqlite`)
- [ ] **KV-cache manager**: Preserve and restore KV cache across continuations

### Phase 2 — RAG Pipeline & Document Intelligence (Sprint 2)
- [ ] **Document ingestion**: PDF (layout-aware), DOCX, TXT, MD, HTML, CSV, EPUB, code files
- [ ] **Table extraction engine**: Detect and extract tables from PDFs as structured markdown
- [ ] **OCR pipeline**: Tesseract/EasyOCR integration for scanned documents
- [ ] **Hierarchical chunking engine**: Structural + semantic + table-aware chunking with parent-child linking
- [ ] **Embedding engine**: Local embeddings (`nomic-embed-text`, BGE) with GPU/CPU split, Matryoshka support
- [ ] **Quantized embedding storage**: int8/binary quantized vectors for memory efficiency
- [ ] **Vector store**: Qdrant (primary, supports quantized vectors natively) or ChromaDB, with metadata filtering
- [ ] **BM25 keyword index**: SQLite FTS5 or tantivy for exact-match search
- [ ] **Hybrid retrieval**: Vector + BM25 fused via Reciprocal Rank Fusion
- [ ] **Cross-encoder re-ranking**: Score retrieved chunks against query for precision
- [ ] **Citation tracking**: Every answer cites source chunks with page/section references
- [ ] **Workspace/collections**: Organize documents into scoped knowledge bases

### Phase 3 — Context Distillation & Output Amplification (Sprint 3) ✅
- [x] **Context compression pipeline**: Extract only query-relevant sentences from retrieved chunks
- [x] **Deduplication engine**: Remove redundant info across retrieved chunks
- [x] **Adaptive prompt templates**: Task-aware prompt selection (Q&A, summary, analysis, comparison)
- [x] **Query decomposition**: Break complex questions into sub-queries with separate retrieval passes
- [x] **Chain-of-density prompting**: Iterative answer refinement for richer outputs
- [x] **Self-verification loop**: Cross-check generated claims against source material
- [x] **Confidence scoring**: Flag answers with low source support
- [x] **Multi-hop retrieval**: Iterative retrieval for complex reasoning chains
- [x] **Conversation memory**: Long-term memory with rolling compression (recent=full, older=summary, oldest=embedding)

### Phase 4 — Knowledge Graph & Advanced Memory (Sprint 4)
- [ ] **Entity extraction**: Extract named entities and relationships from ingested documents
- [ ] **Knowledge graph builder**: Automatically build entity-relationship graphs (NetworkX / embedded)
- [ ] **Graph-augmented retrieval**: Use graph traversal to answer multi-hop questions
- [ ] **Graph visualization UI**: Interactive knowledge graph explorer in the frontend
- [ ] **Memory inspector UI**: Visualize what the system "remembers" — chunks, summaries, entities
- [ ] **Web scraping**: Ingest from URLs, sitemaps, RSS feeds
- [ ] **Incremental re-indexing**: Update vector store and graph when documents change, without full rebuild

### Phase 5 — Agents & Tools (Sprint 5)
- [x] **Tool framework**: Define tools as Python functions with JSON schema — `ToolDefinition`, `ToolParameter`, `ToolResult` dataclasses with `ToolRegistry`
- [x] **Built-in tools**: Calculator, code runner (sandboxed subprocess), file reader/writer, web search (DuckDuckGo), datetime utility
- [x] **Agent loop**: ReAct-style reasoning (Thought → Action → Observation → Final Answer) with configurable max steps (default 10) and streaming SSE
- [x] **Agent-RAG integration**: `rag_search` tool queries the document pipeline; `knowledge_graph` tool queries entity relationships
- [ ] **Multi-agent orchestration**: Define agent teams with roles and handoff protocols
- [x] **Agent memory**: Persistent memory store (SQLite) — agents can store/recall/list facts across sessions via `memory` tool; manual memory management UI
- [ ] **Scheduled tasks**: Cron-like agent runs for monitoring, summarization, report generation
- [x] **Agent API**: `/api/agent/run` (sync), `/api/agent/run/stream` (SSE), `/api/agent/tools`, `/api/agent/execute`, `/api/agent/memory` (CRUD), `/api/agent/runs` (history)
- [x] **Agent UI**: Full-featured Agent Mode view with ReAct step visualization, tool browser, memory manager, and run history

### Phase 6 — Polish & Extensibility (Sprint 6)
- [ ] **Plugin system**: Hot-loadable Python plugins for new tools, data sources, and model backends
- [ ] **API compatibility**: OpenAI-compatible `/v1/chat/completions` endpoint (drop-in replacement)
- [ ] **Multi-user support**: Basic auth, per-user workspaces, per-user API keys
- [ ] **Theming & UX**: Dark/light mode, keyboard shortcuts, mobile-responsive
- [ ] **Export/import**: Conversations, workspaces, knowledge bases as portable archives
- [ ] **Telemetry-free**: Zero tracking, fully offline-capable after initial model download
- [ ] **One-click installer**: Bundled executable for Windows/macOS/Linux (Tauri + PyInstaller)
- [ ] **Performance dashboard**: Show inference speed (tokens/sec), memory usage, GPU utilization in real-time

---

## 10. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Backend** | Python 3.11+ / FastAPI | Async, fast, huge ML ecosystem |
| **Inference** | llama.cpp (via llama-cpp-python) | Best GGUF performance, GPU/CPU split built-in |
| **Embeddings** | sentence-transformers / llama.cpp | Local, fast, GPU-accelerated, Matryoshka support |
| **Vector DB** | Qdrant (embedded mode) | Quantized vector support, advanced filtering, fast HNSW |
| **Keyword Search** | SQLite FTS5 or tantivy | BM25 search without external dependencies |
| **Re-ranking** | cross-encoder (via sentence-transformers) | Precision re-ranking after initial retrieval |
| **Knowledge Graph** | NetworkX (embedded) | No server needed, fast for local-scale graphs |
| **Database** | SQLite (aiosqlite) | No external deps, single-file, fast, WAL mode for concurrency |
| **PDF Processing** | PyMuPDF + pdfplumber | Layout-aware text extraction, table detection |
| **OCR** | Tesseract / EasyOCR | Scanned document support |
| **Frontend** | SvelteKit | Fast, small bundle, SSR-ready |
| **Desktop** | Tauri v2 | Rust-based, ~10x smaller than Electron |
| **Task queue** | asyncio + in-process queue | No Redis/Celery overhead for single-user |
| **Packaging** | PyInstaller + Tauri bundler | Single executable distribution |

---

## 11. What Makes This Different From AnythingLLM

1. **Context Distillation**: AnythingLLM sends raw chunks to the model. SoloLLM compresses and distills context so small models get pure signal — no noise.

2. **Auto-Continuation**: AnythingLLM stops when the model stops. SoloLLM detects truncation, preserves the KV cache, and lets the model seamlessly continue — making a 2K output-token model produce 10K+ token documents.

3. **Tiered Vector Memory**: AnythingLLM uses a flat vector store. SoloLLM has 5 tiers — hot cache, vector, keyword, graph, compressed long-term memory — each optimized for different query types.

4. **Quantized Embeddings**: AnythingLLM stores full float32 embeddings. SoloLLM stores int8/binary quantized embeddings — handling 10-30x more documents in the same memory.

5. **Deep Document Intelligence**: AnythingLLM does basic text extraction. SoloLLM preserves document hierarchy, extracts tables as structured data, handles scanned PDFs via OCR, and maintains parent-child chunk relationships.

6. **Output Amplification**: Chain-of-density prompting, self-verification, query decomposition — these are techniques that make a 7B model produce outputs that look like they came from GPT-4.

7. **Hardware Optimization**: Auto-profiling, adaptive GPU/CPU splitting, KV-cache paging, and model warm-start caching — every optimization compounds to make inference faster on consumer hardware.

---

## 12. Unique Feature Ideas (Beyond AnythingLLM)

### 12.1 "Focus Mode" — Targeted Document Q&A
- User highlights a specific section/page range in a PDF.
- System restricts retrieval to only that section — no noise from the rest of the document.
- Perfect for: "What does Section 3.2 say about liability?"

### 12.2 "Deep Dive" — Exhaustive Document Analysis
- User uploads a document and requests "Deep Dive."
- System processes every section independently, generates a comprehensive section-by-section analysis.
- Uses auto-continuation to produce arbitrarily long outputs.
- Output: A full document summary with per-section breakdowns, key findings, extracted entities, and a knowledge graph visualization.

### 12.3 "Compare Mode" — Multi-Document Comparison
- User selects 2+ documents and a comparison criteria.
- System retrieves relevant sections from each document, aligns them by topic, and generates a structured comparison table.
- Example: "Compare the terms of Contract A and Contract B."

### 12.4 "Smart Bookmarks" — Pin Important Context
- Users can pin specific chunks or paragraphs as "always include" context for a workspace.
- Pinned context is always injected into the prompt, regardless of retrieval results.
- Useful for standing instructions, key definitions, or reference data.

### 12.5 "History-Aware RAG" — Learn From Past Conversations
- The system tracks which retrieved chunks actually led to useful answers (based on user feedback — thumbs up/down).
- Over time, retrieval quality improves for that workspace — frequently useful chunks get boosted, irrelevant ones get demoted.
- A lightweight feedback loop that improves without any model fine-tuning.

### 12.6 "Explain Like I'm..." — Adaptive Response Complexity
- User can set a complexity level: Technical Expert, Professional, Student, Beginner.
- System adjusts the prompt template to match the level.
- Same retrieval, different output style — all controlled through prompt engineering.

### 12.7 Real-Time Inference Dashboard
- Show live metrics during generation: tokens/second, GPU utilization, memory pressure, VRAM/RAM usage.
- Visual indicator of which layers are on GPU vs CPU.
- Useful for power users who want to optimize their setup.

---

## 13. Rough File Structure

```
solollm/
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── api/
│   │   ├── chat.py                 # Chat endpoints (streaming SSE)
│   │   ├── rag.py                  # Document & retrieval endpoints
│   │   ├── agents.py               # Agent endpoints
│   │   ├── models.py               # Model management endpoints
│   │   ├── documents.py            # Document upload & management
│   │   └── admin.py                # Settings & system info
│   ├── core/
│   │   ├── inference.py            # LLM inference engine (GPU/CPU split)
│   │   ├── continuation.py         # Auto-continuation controller
│   │   ├── token_budget.py         # Token budget estimation & management
│   │   ├── kv_cache.py             # KV-cache preservation & restoration
│   │   ├── profiler.py             # Hardware auto-detection & optimization
│   │   ├── embeddings.py           # Embedding engine (Matryoshka, quantized)
│   │   └── config.py               # App configuration
│   ├── rag/
│   │   ├── ingest.py               # Document parsing (PDF, DOCX, etc.)
│   │   ├── pdf_processor.py        # Layout-aware PDF processing
│   │   ├── table_extractor.py      # Table detection & extraction
│   │   ├── ocr.py                  # OCR pipeline for scanned docs
│   │   ├── chunking.py             # Hierarchical multi-strategy chunking
│   │   ├── vectorstore.py          # Vector DB interface (Qdrant/ChromaDB)
│   │   ├── keyword_index.py        # BM25 keyword search (FTS5/tantivy)
│   │   ├── retriever.py            # Hybrid search + rank fusion
│   │   ├── reranker.py             # Cross-encoder re-ranking
│   │   ├── distiller.py            # Context compression & distillation
│   │   └── citations.py            # Source tracking & citation generation
│   ├── memory/
│   │   ├── conversation.py         # Conversation memory with compression
│   │   ├── longterm.py             # Long-term memory store
│   │   └── graph.py                # Knowledge graph (entity-relation)
│   ├── amplifier/
│   │   ├── chain_of_density.py     # Iterative answer densification
│   │   ├── self_verify.py          # Claim verification against sources
│   │   ├── query_decompose.py      # Complex query breakdown
│   │   └── prompt_templates.py     # Task-adaptive prompt library
│   ├── agents/
│   │   ├── loop.py                 # ReAct agent loop
│   │   ├── tools/                  # Built-in tools
│   │   │   ├── web_search.py
│   │   │   ├── calculator.py
│   │   │   ├── code_runner.py      # Sandboxed code execution
│   │   │   └── file_io.py
│   │   └── orchestrator.py         # Multi-agent coordination
│   ├── storage/
│   │   ├── database.py             # SQLite models & migrations
│   │   └── filestore.py            # Document blob storage
│   └── plugins/
│       └── loader.py               # Plugin discovery & loading
├── frontend/
│   ├── src/
│   │   ├── routes/                 # SvelteKit pages
│   │   │   ├── chat/               # Chat interface
│   │   │   ├── documents/          # Document manager
│   │   │   ├── knowledge/          # Knowledge graph explorer
│   │   │   ├── agents/             # Agent dashboard
│   │   │   └── settings/           # Configuration
│   │   ├── lib/
│   │   │   ├── components/         # Shared UI components
│   │   │   │   ├── ContinuationBanner.svelte  # "Continue generating?" UI
│   │   │   │   ├── CitationPopover.svelte     # Click-to-see-source
│   │   │   │   ├── MemoryInspector.svelte     # Memory visualization
│   │   │   │   └── InferenceDashboard.svelte  # Real-time metrics
│   │   │   └── utils/              # API client, helpers
│   │   └── stores/                 # State management
│   └── static/
├── desktop/                         # Tauri shell
├── models/                          # Downloaded model files
├── data/                            # SQLite DB, vector store, documents
│   ├── db/                          # SQLite databases
│   ├── vectors/                     # Qdrant/ChromaDB storage
│   ├── documents/                   # Original uploaded files
│   └── cache/                       # KV cache, model warm-start cache
└── plugins/                         # User-installed plugins
```

---

## 14. Stretch Goals (Post-MVP)

- **Vision models**: LLaVA / multimodal support for image understanding from PDFs
- **Voice interface**: Whisper (STT) + local TTS for voice conversations
- **Fine-tuning UI**: QLoRA fine-tuning on your documents, in-app
- **Collaborative mode**: LAN-based multi-user with shared knowledge bases
- **Mobile companion**: Lightweight mobile app connecting to your local instance
- **Model arena**: Side-by-side model comparison on the same prompt + context
- **Document versioning**: Track changes across document versions, show diffs in knowledge base
- **API marketplace**: Share and install community plugins, prompt templates, and agent configurations
- **Batch processing**: Upload 100 PDFs and ask "Extract all financial figures" — runs overnight with agents

---

## 15. Getting Started (First Steps)

Once you approve this direction, the build order would be:

1. **Set up the backend** — FastAPI + llama-cpp-python with a `/chat` endpoint and SSE streaming
2. **Build the profiler** — detect GPU/VRAM/RAM, auto-set `n_gpu_layers` and quantization
3. **Implement auto-continuation** — token budget tracking, truncation detection, KV-cache preservation, and seamless resume
4. **Add a minimal chat UI** — SvelteKit page with streaming, continuation banner, and real-time token counter
5. **Wire up model management** — download GGUF models from HuggingFace, auto-select based on hardware profile
6. **Build the RAG pipeline** — layout-aware PDF processing → hierarchical chunking → quantized embedding → tiered retrieval → context distillation → generate
7. **Add output amplification** — chain-of-density, self-verification, query decomposition
8. **Build knowledge graph** — entity extraction, graph construction, graph-augmented retrieval

Each step produces a **working, testable increment**.

---

*This system is designed so that a $200 laptop running a 7B model can give answers that make people ask "Wait, what model is this running?" — that's the goal.*
