# SoloLLM Vector Search Precision Upgrade Draft

## 1) Problem Summary (Current Behavior)

The current retrieval pipeline is strong in structure (hybrid retrieval, reranking hooks, distillation), but precision drops for broad terms like "function" because retrieval is still too permissive at multiple layers.

Observed causes in current implementation:

1. Keyword search is OR-heavy.
- In `backend/rag/keyword_index.py`, `_escape_fts_query()` converts query terms into an OR query.
- Example impact: searching "explain function calling" can match any chunk containing only "function".

2. No strict similarity floor in vector retrieval.
- In `backend/rag/vectorstore.py`, FAISS returns top candidates by rank, then results are truncated by top_k.
- There is no minimum cosine/IP threshold gate before results proceed.

3. Fusion relies on rank position (RRF), not calibrated relevance.
- In `backend/rag/retriever.py`, Reciprocal Rank Fusion combines rank slots from vector and keyword retrieval.
- Rank-only fusion can over-promote loosely related chunks if they appear in either list.

4. Reranker is disabled by default.
- In `backend/core/config.py`, `reranker_enabled=False`.
- Without cross-encoder reranking, precision depends heavily on upstream retrieval.

5. Context compression can keep weakly related text.
- In `backend/core/distillation.py`, query filtering keeps sentences with any token overlap.
- Generic terms can retain noise.

6. Retrieval count is user-controlled via top_k but not confidence-adaptive.
- Thread setting uses `rag_top_k`; there is no dynamic reduction when confidence is low.

Net effect: high recall, lower precision for ambiguous or generic queries.

---

## 2) Target Retrieval Architecture (Advanced + Precise)

Move from "hybrid rank merge" to "precision-gated multi-stage retrieval":

1. Query understanding and intent typing.
2. Candidate generation (vector + lexical) with stricter semantics.
3. Score calibration and hard gating.
4. MMR/diversity and per-document caps.
5. Cross-encoder reranking on small candidate set.
6. Confidence-based context packing and fallback behavior.

### 2.1 Stage A: Query Understanding

For each query, derive:
- query_type: factual / definition / procedural / code / comparison.
- keyphrases: multi-word phrases (e.g., "function calling").
- required terms vs optional terms.
- ambiguity score (high if query has generic tokens).

Implementation note:
- Add `backend/rag/query_understanding.py` with a lightweight extractor.
- Keep it deterministic first (regex + heuristics) to avoid extra LLM latency.

### 2.2 Stage B: Candidate Generation (Stricter)

#### Lexical branch improvements
Current OR strategy should be replaced with weighted AND/phrase strategy:
- Required terms must appear for high-confidence lexical candidates.
- Exact phrase matches get strong boost.
- Optional terms contribute secondary score.

Suggested lexical scoring (conceptual):

score_lex = 0.55 * bm25_norm + 0.30 * phrase_match + 0.15 * term_coverage

where:
- bm25_norm is normalized BM25 score.
- phrase_match is binary/graded phrase hit.
- term_coverage = matched_required_terms / total_required_terms.

#### Vector branch improvements
- Fetch larger pool (e.g., 50-120) but apply strict threshold before passing to fusion.
- Add dynamic minimum similarity:
  - factual/definition: higher threshold (e.g., 0.30-0.40 normalized).
  - broad or exploratory query: lower threshold (e.g., 0.20-0.28).

### 2.3 Stage C: Score Calibration + Fusion

Replace rank-only fusion with calibrated score fusion:

score_fused = wv * score_vec_norm + wl * score_lex_norm + wp * phrase_bonus + ws * section_signal

Recommended initial weights:
- wv=0.45, wl=0.35, wp=0.15, ws=0.05.

Hard gates before accepting candidate:
- min(score_vec_norm, if vector-eligible) >= vector_threshold OR
- lexical term_coverage >= required_coverage_threshold.

This prevents weak one-word matches from surviving.

### 2.4 Stage D: Diversity + Redundancy Control

Add post-fusion MMR on top candidates:
- Keep relevance high, reduce near-duplicate chunks.
- Enforce per-document cap (e.g., max 2 chunks per document in final set).

MMR objective:

MMR(d) = lambda * Rel(d, q) - (1 - lambda) * max_{s in selected} Sim(d, s)

with lambda ~ 0.65 as start.

### 2.5 Stage E: Cross-Encoder Reranking

Run reranker only on a small candidate pool (e.g., top 15-30 after gating/MMR).
- This is where precision becomes strict.
- Final response context should typically include only 3-6 chunks, not broad 10-20 chunk windows.

### 2.6 Stage F: Confidence-Guided Context Packing

Context builder should:
- Select only chunks above confidence floor.
- Stop adding context when marginal gain is low.
- Return "insufficient evidence" state when evidence is weak.

Behavior change:
- If confidence < threshold, do not flood prompt with weak chunks.
- Ask a follow-up question or state limited evidence.

---

## 3) Concrete Code Changes Required

## 3.1 Backend RAG Core

1. `backend/rag/keyword_index.py`
- Replace OR-only query construction.
- Add parser for required terms and quoted phrases.
- Return richer lexical metadata: `term_coverage`, `phrase_match`, normalized score.

2. `backend/rag/vectorstore.py`
- Add support for `score_threshold`, `fetch_k`, and optional score normalization.
- Return both raw and normalized vector scores.

3. `backend/rag/retriever.py`
- Replace pure RRF path with configurable `precision_fusion` mode.
- Implement:
  - calibrated score fusion,
  - hard threshold gates,
  - per-document cap,
  - MMR selection,
  - debug diagnostics (why a chunk was accepted/rejected).

4. `backend/rag/reranker.py`
- Keep heuristic fallback.
- Add rerank cutoffs and confidence outputs.
- Enable model warm check to avoid first-request latency spikes.

5. `backend/rag/pipeline.py`
- Make final top_k adaptive by confidence.
- Add retrieval diagnostics object in response for observability.
- Thread-aware retrieval should preserve strict document scope as already done.

6. `backend/core/distillation.py`
- Tighten query filtering logic:
  - use keyphrase matching and weighted overlap,
  - do not keep sentence solely because of one generic token.
- Update confidence scorer to include:
  - calibrated retrieval score,
  - phrase coverage,
  - contradiction/dispersion penalty.

## 3.2 Configuration and Per-Thread Controls

1. `backend/core/config.py`
- Add defaults:
  - `rag_vector_min_score`
  - `rag_lexical_required_coverage`
  - `rag_candidate_pool_size`
  - `rag_per_document_cap`
  - `rag_use_mmr`
  - `rag_mmr_lambda`
  - `rag_precision_mode` (legacy_rrf | precision_fusion)

2. `backend/storage/database.py`
- Add migrations for new thread settings columns.
- Extend `update_thread_settings()` allowed fields.

3. `backend/storage/schemas.py`
- Extend `ThreadSettingsUpdate` with new precision controls.

4. `backend/api/threads.py`
- No major logic change; ensure new settings are validated and exposed.

## 3.3 Frontend Controls

1. `frontend/src/types/index.ts`
- Extend `ThreadSettings` type with new precision parameters.

2. `frontend/src/components/chat/ThreadSettingsPanel.tsx`
- Add advanced RAG precision section:
  - min similarity threshold
  - per-document cap
  - candidate pool size
  - MMR toggle + lambda
  - precision mode

3. `frontend/src/lib/api.ts`
- Ensure settings payload supports the new fields.

---

## 4) Hugging Face Model Upgrade Recommendations

Current models:
- Embedding: `all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (currently disabled by default)

These are lightweight and fast, but not best-in-class for precision.

### 4.1 Embedding Model Upgrade Path

Tier A (balanced speed/quality):
1. `BAAI/bge-small-en-v1.5`
2. `intfloat/e5-base-v2`

Tier B (higher quality, more memory/latency):
1. `BAAI/bge-base-en-v1.5`
2. `mixedbread-ai/mxbai-embed-large-v1`

If multilingual requirements exist:
- `BAAI/bge-m3`

Recommendation for this project hardware profile:
- Start with `BAAI/bge-small-en-v1.5` or `intfloat/e5-base-v2`.
- Keep batch size conservative on low-memory systems.
- Re-index documents after embedding model switch (mandatory).

### 4.2 Reranker Upgrade Path

Tier A:
1. `cross-encoder/ms-marco-MiniLM-L-6-v2` (current, good baseline)
2. `BAAI/bge-reranker-base`

Tier B:
1. `BAAI/bge-reranker-large`
2. `jinaai/jina-reranker-v2-base-multilingual` (if multilingual needed)

Practical guidance:
- Enable reranker by default in precision mode.
- Keep candidate set small (15-30) to control latency.
- Use local cache and warmup for startup smoothness.

---

## 5) Rollout Plan (Step-by-Step)

### Phase 0: Baseline Measurement (before changes)

Create evaluation dataset from real queries:
- 100-300 queries from app logs (anonymized).
- Label relevant chunks for each query (top 3-5 truth chunks).

Metrics:
- Recall@k
- Precision@k
- nDCG@k
- MRR
- latency p50/p95

Deliverable:
- baseline report and regression guardrails.

### Phase 1: Retrieval Gating + Fusion Upgrade

Implement in `keyword_index.py`, `vectorstore.py`, `retriever.py`:
- strict lexical parsing,
- vector score threshold,
- calibrated fusion,
- per-doc cap.

Add feature flag:
- `rag_precision_mode=legacy_rrf | precision_fusion`

Goal:
- large precision gain without frontend changes.

### Phase 2: MMR + Adaptive Context Packing

Implement:
- MMR selector,
- confidence-aware final chunk count (3-6 typical),
- insufficient-evidence fallback behavior.

Goal:
- reduce noisy context injection into LLM prompt.

### Phase 3: Model Upgrade + Reindex

1. swap embedding model,
2. rebuild embeddings and FAISS indexes,
3. enable reranker by default,
4. compare latency/quality against baseline.

Goal:
- maximize relevance quality while keeping acceptable latency.

### Phase 4: UI and Thread-Level Precision Controls

Expose settings in thread panel:
- user can tune strictness per thread/document set.

Goal:
- practical control for different use cases (strict QA vs exploratory research).

### Phase 5: Evaluation Automation

Add retrieval evaluation script:
- run nightly or pre-release,
- fail build on significant precision regression.

---

## 6) Data and Index Migration Requirements

1. Embedding model changes require full re-embedding and FAISS rebuild.
2. DB migration required for new thread settings columns.
3. Add backward compatibility defaults so existing threads remain functional.
4. Keep legacy RRF mode available during transition.

---

## 7) Risk Analysis and Mitigation

1. Risk: precision up, recall down too much.
- Mitigation: tune thresholds by query type and keep fallback mode.

2. Risk: reranker latency increase.
- Mitigation: rerank only top 15-30 candidates, cache model, warmup on startup.

3. Risk: model upgrade memory pressure.
- Mitigation: tiered model options; keep lightweight fallback in config.

4. Risk: overfitting to benchmark set.
- Mitigation: include production-like long-tail queries and periodic refresh.

---

## 8) Acceptance Criteria

The upgrade is successful when all are true:

1. Precision@5 improves by at least 25% on labeled evaluation set.
2. nDCG@5 improves by at least 20%.
3. For ambiguous term queries (like "function"), irrelevant chunk rate drops significantly.
4. p95 retrieval latency increase is within agreed budget (for example <35%).
5. User-visible answer quality improves (fewer hallucination-like context mismatches).

---

## 9) Recommended Default Values (Initial)

For precision mode defaults:
- candidate_pool_size: 80
- vector_min_score: 0.28
- lexical_required_coverage: 0.5
- per_document_cap: 2
- use_mmr: true
- mmr_lambda: 0.65
- pre_rerank_limit: 24
- final_context_chunks: 4

These should be tuned via offline evaluation after initial implementation.

---

## 10) Final Recommendation

Implement precision_fusion architecture behind a feature flag, run A/B evaluation against current RRF, then promote as default once metrics pass.

Most impactful sequence:
1. strict lexical parsing + score thresholds,
2. calibrated fusion + MMR,
3. enable reranker by default,
4. embedding model upgrade + reindex.

This sequence gives immediate precision improvements while preserving deployment safety.
