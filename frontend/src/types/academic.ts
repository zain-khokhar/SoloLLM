// TypeScript types for VU Academic Auto-Generation

export interface AcademicCourse {
  id: string;
  code: string;
  title: string;
  department: string;
  total_lectures: number;
  workspace_id: string;
  metadata: string;
  review_count?: number;
  created_at: string;
  updated_at: string;
}

export interface AcademicReview {
  id: string;
  course_id: string;
  source_id: string | null;
  review_text: string;
  reviewer_token: string;
  semester: string;
  urgency_score: number;
  sentiment_score: number;
  is_duplicate: number;
  is_spam: number;
  created_at: string;
}

export interface AcademicTopicScore {
  id: string;
  course_id: string;
  topic_name: string;
  exam_probability: number;
  weight_bucket: 'high' | 'medium' | 'low';
  midterm_relevance: number;
  final_relevance: number;
  review_frequency: number;
  urgency_signal: number;
  consensus_score: number;
  syllabus_importance: number;
  recency_weight: number;
  llm_confidence: number;
  evidence?: any[];
  scored_at: string;
}

export interface AcademicAiLog {
  content_type: string;
  topic_name?: string;
  model?: string;
  request: string;
  response?: string;
  error?: string;
}

export interface AcademicRenderStats {
  match_rate?: number;
  highlight_count?: number;
  unmatched_spans?: number;
  failed_batches?: number;
}

export interface AcademicJob {
  id: string;
  course_id: string;
  job_type: string;
  output_types: string[];
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_stage: string;
  stages_completed: string[];
  checkpoint_json?: {
    options?: {
      generation_model?: string | null;
      max_context_chars?: number;
      syllabus_scope?: 'all' | 'midterm' | 'final';
      split_mode?: 'auto' | 'manual' | 'none';
      manual_midterm_end_page?: number | null;
      retrieval_top_k?: number;
      batch_pages?: number;
      max_batch_chars?: number;
      max_review_evidence_chars?: number;
      max_spans_per_batch?: number;
    };
    preview?: {
      processed_pages?: number[];
      processed_topics?: number;
      highlights_applied?: number;
      live_file_path?: string;
      last_updated?: string;
      unmatched_spans?: number;
    };
    render_stats?: AcademicRenderStats;
    ai_logs?: AcademicAiLog[];
    model_warning?: string;
    quality_warning?: string;
  };
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface AcademicOutput {
  id: string;
  course_id: string;
  job_id: string;
  output_type: string;
  version: number;
  file_path: string;
  file_size: number;
  topic_count: number;
  generation_params: Record<string, any>;
  created_at: string;
}

export interface AcademicGenerateOptions {
  generation_model?: string;
  max_context_chars?: number;
  syllabus_scope?: 'all' | 'midterm' | 'final';
  split_mode?: 'auto' | 'manual' | 'none';
  manual_midterm_end_page?: number;
  retrieval_top_k?: number;
  batch_pages?: number;
  max_batch_chars?: number;
  max_review_evidence_chars?: number;
  max_spans_per_batch?: number;
}

export interface AcademicFeedback {
  id: string;
  output_id: string;
  rating: number;
  comment: string;
  topic_accuracy_pct: number | null;
  created_at: string;
}

export interface AcademicOverviewStats {
  total_courses: number;
  total_reviews: number;
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_outputs: number;
  system_status: string;
}

export interface CourseMetrics {
  topic_count: number;
  high_confidence: number;
  medium_confidence: number;
  low_confidence: number;
  avg_exam_probability: number;
  output_count: number;
  avg_feedback_rating: number;
  feedback_count: number;
}

export interface ScoringWeights {
  review_frequency: number;
  urgency_signal: number;
  consensus: number;
  syllabus_importance: number;
  recency: number;
  llm_confidence: number;
}

export interface BulkImportResult {
  total_files: number;
  success_count: number;
  failed_count: number;
  results: Array<{
    filename: string;
    course_code?: string;
    course_id?: string;
    workspace_id?: string;
    confidence?: number;
    error?: string;
    needs_manual_mapping?: boolean;
    ingest?: any;
  }>;
}

export interface ReviewIngestResult {
  source_id: string;
  course_id: string;
  total_added: number;
  skipped_spam?: number;
  skipped_duplicate?: number;
  filename: string;
  error?: string;
}

export type AcademicTab = 'overview' | 'courses' | 'reviews' | 'generation' | 'outputs' | 'evaluation';
