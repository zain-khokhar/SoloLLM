# Highlighted Handout Rebuild Plan (No Top-K, Page-Batch Flow)

## Goal
Build a deterministic highlighted-handout pipeline where:
1. The syllabus scope is explicit (`midterm` or `final`).
2. Processing is sequential and page-based (5 pages per batch).
3. The AI receives both review evidence and full extracted text for each 5-page batch in order.
4. The AI predicts topic importance from review patterns + batch text, then returns exact text spans to highlight.
5. Only exact returned text is highlighted in the source PDF.

This replaces top-k retrieval behavior for highlighted generation.

## Non-Negotiable Rules
1. No vector top-k retrieval for highlighted mode.
2. Process pages in strict order.
3. Midterm mode starts at page 1 and ends at midterm boundary.
4. Final mode starts at final boundary and runs to last page.
5. Batch size default is 5 pages, user-adjustable (with safe min/max).
6. Highlighting decisions must be review-informed and AI-predicted for each batch.
7. Model output must be exact snippets present in the current batch text.
8. No paraphrase highlighting.
9. If AI output is invalid/non-matching, skip safely and log why.
10. Do not use hardcoded topic scoring formulas for highlight selection in this mode.

## Syllabus Boundary Model

### Inputs
1. `split_mode`: `auto` | `manual` | `none`
2. `manual_midterm_end_page`: optional
3. `syllabus_scope`: `midterm` | `final` | `all`

### Boundary Resolution
1. If `split_mode=manual`, use `manual_midterm_end_page`.
2. If `split_mode=auto`, apply current mids formula (e.g. 500 -> 237).
3. If `split_mode=none`, treat entire document as `all`.

### Processing Ranges
1. `midterm`: pages `1..mid_end`
2. `final`: pages `mid_end+1..total_pages`
3. `all`: pages `1..total_pages`

## New Highlighting Pipeline (Target Design)

### Stage 1: Build Ordered Page Text Map
1. Extract text per page from source PDF.
2. Build `page_text_map[page_no]` with cleaned but faithful text.
3. Persist in checkpoint for resume/restart.

### Stage 2: Batch Generator (5 Pages at a Time)
1. Create ordered windows from selected range:
   - Batch 1: pages 1-5
   - Batch 2: pages 6-10
   - ...
2. Each batch payload includes:
   - page range
   - full text of those 5 pages
   - active scope (`midterm`/`final`)
   - review evidence pack (course reviews + extracted patterns)

### Stage 2.5: Review Evidence Pack (AI Input)
1. Build a compact review packet before batch loop starts.
2. Packet includes:
   - raw high-signal review lines
   - normalized topic mentions and co-occurrence hints
   - urgency/repeat indicators from reviews
   - recent vs older review trend snapshot
3. This packet is provided to every batch prompt so predictions stay review-informed.

### Stage 3: AI Prediction + Extraction Contract (Strict)
Per batch, prompt must instruct:
1. Predict what is important using BOTH:
   - review patterns/evidence
   - the current 5-page batch text
2. Select spans that best match review-driven exam patterns.
3. Return exact text copied from current batch only.
4. Return machine-parseable JSON lines/array.
5. Include prediction rationale tied to review evidence.
6. Include confidence as metadata only (not sole selection criterion).
7. Reject uncertain spans instead of guessing.

#### Required Output Schema
```json
[
  {
    "text": "exact copied sentence or phrase",
    "importance": "high|medium",
      "reason": "definition|formula|rule|concept|frequent_exam_point",
      "review_signal": "matched_review_pattern_or_phrase",
      "prediction_note": "why this span is important given reviews + current batch",
      "confidence": 0.0
  }
]
```

### Prediction Policy (Critical)
1. Selection is NOT confidence-only.
2. Final selection priority is a combined AI judgment of:
    - review support strength
    - semantic importance in current batch text
    - exam-likelihood pattern fit from reviews
3. Confidence is used as a tie-breaker/metadata, not as the primary gate.

### Stage 4: Deterministic Match + Highlight
1. For each returned span, run exact search on batch pages.
2. If exact search fails, run controlled fallback:
   - normalize whitespace only
   - no semantic fuzzy matching
3. Highlight all valid matches in yellow.
4. Record unmatched spans in checkpoint logs.

### Stage 5: Live Preview + Persistence
1. After each processed batch, save live preview PDF snapshot.
2. Update checkpoint with:
   - last completed batch index
   - pages processed
   - highlights applied
   - unmatched spans
   - errors
3. Resume starts from next pending batch.

## UI/UX Plan

### Generation Controls
1. Model selector (downloaded models only).
2. Scope selector: `midterm` / `final` / `all`.
3. Split selector: `auto` / `manual` / `none`.
4. Manual midterm end page input (when manual).
5. Batch size input (default 5; safe range e.g. 1-10).
6. Prompt text limit per batch (default optimized for 1.5B models).

### Runtime Controls
1. Start
2. Pause
3. Resume
4. Stop
5. Delete failed/completed job

### Live Progress Panel
1. Current scope and page range.
2. Current batch (`k / N`).
3. Pages processed count.
4. Highlights applied count.
5. Unmatched spans count.
6. Latest errors (if any).
7. Open/download live preview snapshot.

## Prompt + Model Budget Strategy (for 1.5B models)
1. Keep batch text compact but complete for 5 pages.
2. Always include review evidence packet with each batch prompt.
3. Use low temperature (stable prediction + extraction behavior).
4. Hard cap returned spans per batch to avoid over-highlighting.
5. Prefer smaller structured output over verbose reasoning.

### Default Safe Settings
1. `batch_pages = 5`
2. `max_batch_chars = 7000` (tunable)
3. `max_review_evidence_chars = 2500`
4. `max_spans_per_batch = 20`
4. `temperature = 0.1`

## Error Handling Strategy
1. If AI response invalid JSON: retry once with repair prompt.
2. If still invalid: mark batch failed, continue or stop based on policy.
3. If no spans matched: log as `batch_no_match`, continue.
4. If file lock during preview save: save to alternate filename and continue.
5. All failures written to checkpoint for post-mortem.

## Data Model / Checkpoint Additions
Checkpoint should include:
1. `scope`, `split_mode`, `mid_end_page`
2. `batch_pages`, `current_batch_index`, `total_batches`
3. `processed_batches[]`
4. `batch_results[]`:
   - pages
   - spans_returned
   - spans_matched
   - highlights_applied
   - errors
5. `preview_file_path`
6. `review_packet_hash`
7. `prediction_policy_version`

## Acceptance Criteria
1. Midterm mode processes only pages `1..mid_end`.
2. Final mode starts exactly at `mid_end+1`.
3. Processing order is strictly sequential, 5 pages at a time.
4. No vector top-k retrieval used in highlighted mode.
5. Every highlighted span is traceable to AI prediction using reviews + batch text.
6. Highlighted spans are exact text from model output and found in pages.
6. Pause/resume continues from correct next batch.
7. Live preview updates after each batch.
8. Job survives restart via checkpoint resume.

## Rollout Plan

### Phase 1: Core Engine Swap
1. Remove top-k path for highlighted mode.
2. Add sequential batch processor and review evidence packet builder.
3. Add AI prediction + extraction prompt contract (reviews + batch text).
3. Add deterministic matcher and yellow highlighter.

### Phase 2: UI + Controls
1. Add scope/split/batch controls.
2. Add runtime controls and progress panel.
3. Add live preview behavior tied to batch completion.

### Phase 3: Reliability
1. Add checkpoint completeness validation.
2. Add robust retry + error classification.
3. Add telemetry counters for match quality.

### Phase 4: Quality Tuning
1. Tune max spans per batch to avoid clutter.
2. Tune defaults per model family (deepseek/qwen 1.5B).
3. Add optional “strictness” slider for conservative highlights.

## Test Plan (Must Pass)
1. 500-page PDF with auto split gives expected mid/final boundaries.
2. Midterm mode highlights only midterm pages.
3. Final mode highlights only final pages.
4. Batch order correctness (`1-5`, `6-10`, ...).
5. Pause/resume during long run resumes correctly.
6. Simulated crash resumes from last completed batch.
7. Predicted spans show review-linked reasons in output JSON/logs.
8. Invalid model output handled without corrupting run.
9. Preview file lock scenario handled without job failure.

## Open Decisions (to finalize before coding)
1. Exact formula function text for mids boundary (if different from current 47.4%).
2. Continue-on-error vs stop-on-error policy at batch level.
3. Maximum allowed highlights per page to prevent visual clutter.
4. Whether to include a second validation pass model (optional).
