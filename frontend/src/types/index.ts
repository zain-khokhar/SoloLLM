export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  token_count: number;
  is_continuation: boolean;
  continuation_of: string | null;
  documents_used?: string[];
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  system_prompt: string;
  created_at: string;
  updated_at: string;
}

export interface ModelInfo {
  name: string;
  size: number | null;
  digest: string;
  modified_at: string | null;
  parameter_size: string | null;
  quantization_level: string | null;
}

export interface SystemProfile {
  gpu_name: string | null;
  vram_mb: number | null;
  ram_mb: number | null;
  cpu_name: string | null;
  cpu_cores: number | null;
  os_info: string | null;
  profiled_at: string | null;
  recommended_models: string[];
}

export interface AppSettings {
  ollama_base_url: string;
  default_model: string;
  max_tokens: number;
  temperature: number;
  auto_continue: boolean;
  system_prompt: string;
}

// SSE event types
export interface TokenEvent {
  content: string;
}

export interface DoneEvent {
  message_id: string;
  conversation_id: string;
  tokens_used: number;
  truncated: boolean;
}

export interface TruncatedEvent {
  message_id: string;
  conversation_id: string;
  tokens_used: number;
  reason: string;
  confidence: number;
  last_content: string;
}

export interface ErrorEvent {
  error: string;
}

// Phase 3 — Distillation types

export interface ConfidenceScore {
  overall: number;
  retrieval_quality: number;
  coverage: number;
  source_diversity: number;
  level: "high" | "medium" | "low";
}

export interface DistillationMeta {
  confidence: ConfidenceScore;
  query_type: string;
  sub_queries: string[];
  compression_ratio: number;
  original_tokens: number;
  compressed_tokens: number;
  hops_used: number;
  chunks_before_dedup: number;
  chunks_after_dedup: number;
  citations: {
    index: number;
    document_title: string;
    section_title: string;
    page_number: number | null;
    document_id?: string;
  }[];
}

export interface DistillationSettings {
  distillation_enabled: boolean;
  context_compression: boolean;
  compression_target_ratio: number;
  deduplication_enabled: boolean;
  dedup_similarity_threshold: number;
  adaptive_prompts: boolean;
  query_decomposition: boolean;
  multi_hop_retrieval: boolean;
  multi_hop_max_hops: number;
  self_verification: boolean;
  chain_of_density: boolean;
  chain_of_density_iterations: number;
  confidence_scoring: boolean;
  conversation_memory_compression: boolean;
  max_recent_messages: number;
  max_memory_tokens: number;
}

// Phase 4 — Knowledge Graph & Memory types

export interface GraphStats {
  entity_count: number;
  relationship_count: number;
  entity_types: Record<string, number>;
}

export interface GraphNode {
  id: string;
  name: string;
  entity_type: string;
  mention_count: number;
}

export interface GraphEdge {
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  weight: number;
}

export interface GraphVisualization {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphAnalysis {
  node_count: number;
  edge_count: number;
  density: number;
  communities: {
    id: number;
    size: number;
    members: { id: string; name: string }[];
  }[];
  hub_entities: {
    id: string;
    name: string;
    entity_type: string;
    pagerank: number;
    centrality: number;
    degree: number;
  }[];
}

export interface EntityNeighbors {
  entity: GraphNode | null;
  neighbors: {
    id: string;
    source_entity_id: string;
    target_entity_id: string;
    relation_type: string;
    weight: number;
    target_name: string;
    target_type: string;
  }[];
}

export interface TimelineEntity {
  id: string;
  name: string;
  entity_type: string;
  mention_count: number;
  source_document_id: string;
  created_at: string;
  updated_at: string;
}

export interface ScrapeResult {
  success: boolean;
  url: string;
  document_id?: string;
  title?: string;
  chunk_count?: number;
  content_length?: number;
  link_count?: number;
  entities_extracted?: number;
  relationships_extracted?: number;
  errors?: string[];
}

export interface ScrapePreview {
  url: string;
  title: string;
  content_length: number;
  content_preview: string;
  link_count: number;
  links: string[];
  metadata: Record<string, string>;
}

// Phase 5 — Agent types

export interface AgentTool {
  name: string;
  description: string;
  category: string;
  is_dangerous: boolean;
  parameters: {
    name: string;
    type: string;
    description: string;
    required: boolean;
  }[];
}

export interface AgentStep {
  step: number;
  thought: string;
  action: string;
  action_input: Record<string, unknown>;
  observation: string;
  is_final: boolean;
}

export interface AgentRunResult {
  answer: string;
  success: boolean;
  error: string | null;
  total_steps: number;
  tools_used: string[];
  steps: AgentStep[];
}

export interface AgentMemory {
  id: string;
  content: string;
  category: string;
  created_at: string;
  updated_at: string;
}

export interface AgentRun {
  id: string;
  query: string;
  answer: string;
  model: string;
  total_steps: number;
  tools_used: string;
  success: number;
  error: string | null;
  created_at: string;
}

// Phase 6 — Dashboard & Export/Import types

export interface DashboardSummary {
  uptime_seconds: number;
  total_requests: number;
  total_tokens_in: number;
  total_tokens_out: number;
  total_tokens: number;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  p95_latency_ms: number;
  error_rate: number;
  requests_per_minute: number;
  model_stats: Record<
    string,
    { requests: number; tokens_in: number; tokens_out: number; avg_latency_ms: number }
  >;
  endpoint_stats: Record<string, number>;
  system?: {
    cpu_percent: number;
    memory_percent: number;
    memory_used_mb: number;
    memory_total_mb: number;
  };
}

export interface ExportData {
  version: string;
  app: string;
  exported_at: string;
  conversation_count: number;
  conversations: Conversation[];
}

export interface ImportResult {
  success: boolean;
  imported_count: number;
}

export interface SettingsExport {
  version: string;
  app: string;
  exported_at: string;
  settings: Record<string, string>;
}

// ── Runtime / Ollama Manager types ─────────────────────────

export interface OllamaHealth {
  available: boolean;
  version: string | null;
  managed: boolean;
  binary_exists: boolean;
  binary_path: string;
  models_dir: string;
  port: number;
  base_url: string;
  process_running: boolean;
}

export interface RuntimeStatus {
  ollama: OllamaHealth;
  auto_start_enabled: boolean;
}

export interface AuthConfig {
  private_access_enabled: boolean;
  owner_username: string;
  session_ttl_hours: number;
  admin_token_enabled: boolean;
  public_base_url: string;
}

export interface AuthSession {
  authenticated: boolean;
  private_access_enabled: boolean;
  owner_username: string;
  expires_at: string | null;
  authenticated_via: "session" | "owner_password" | "admin_token" | "disabled" | null;
  admin_token_enabled: boolean;
}

export interface BackendCapabilities {
  deployment: {
    public_base_url: string;
    admin_route_protection: boolean;
    private_access_enabled: boolean;
    runtime_management_enabled: boolean;
  };
  features: {
    agent: boolean;
    academic: boolean;
    export_import: boolean;
    training: boolean;
    quantize: boolean;
  };
  network: {
    cors_origins: string[];
    cors_origin_regex: string | null;
    allowed_hosts: string[];
  };
}

export interface CatalogModel {
  name: string;
  display_name: string;
  family: string;
  parameters: string;
  size_gb: number;
  description: string;
  recommended_vram_mb: number;
  recommended_ram_mb: number;
  compatible: boolean;
  performance_note: string;
}

export interface ModelCatalogResponse {
  catalog: CatalogModel[];
  hardware: {
    vram_mb: number | null;
    ram_mb: number | null;
  };
}

// ── Thread types ────────────────────────────────────────────

export interface Thread {
  id: string;
  conversation_id: string;
  title: string;
  system_prompt: string;
  context_mode: "isolated" | "shared";
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface ThreadSettings {
  thread_id: string;
  max_tokens: number | null;
  temperature: number | null;
  rag_enabled: boolean;
  rag_top_k: number;
  compression_enabled: boolean;
  memory_layers: number;
  max_history_messages: number;
  compression_ratio: number;
  // KV / memory fields
  kv_compression_enabled: boolean;
  memory_layer_mode: "none" | "sliding_window" | "virtual_paging";
  memory_page_size: number;
  context_window_size: number;
  // Precision mode controls
  rag_precision_mode: string;
  rag_vector_min_score: number;
  rag_lexical_required_coverage: number;
  rag_candidate_pool_size: number;
  rag_per_document_cap: number;
  rag_use_mmr: boolean;
  rag_mmr_lambda: number;
}

export interface ThreadContextPage {
  id: string;
  thread_id: string;
  page_number: number;
  content: string;
  token_count: number;
  is_active: boolean;
  created_at: string;
}

// ── Document / RAG types ────────────────────────────────────

export interface DocumentInfo {
  document_id: string;
  filename: string;
  title: string;
  file_type: string;
  chunk_count: number;
  page_count: number;
  workspace_id: string;
  created_at: string;
}

export interface Citation {
  index: number;
  document_title: string;
  section_title: string;
  page_number: number | null;
  document_id: string;
  chunk_id: string;
  relevance_score: number;
  excerpt: string;
}

export interface EmbeddingInfo {
  model: string;
  dimension: number;
  fallback: boolean;
}

export interface RAGStats {
  workspace_id: string;
  document_count: number;
  chunk_count: number;
  embedding_info: EmbeddingInfo;
}

// ── Training types ──────────────────────────────────────────

export interface TrainingStatus {
  status: "idle" | "preparing_data" | "downloading_base_model" | "training" | "exporting_gguf" | "registering_with_ollama" | "complete" | "error";
  current_step: number;
  total_steps: number;
  loss: number;
  val_loss?: number;
  best_val_loss?: number;
  quality_passed?: boolean;
  epoch: number;
  device?: "gpu" | "cpu";
  message: string;
  error: string | null;
}

export interface TrainingDataPreview {
  total_examples: number;
  source_mode?: "conversation" | "documents" | "mixed";
  documents_used?: string[];
  sequence?: Array<{
    index: number;
    source_type: "conversation" | "document";
    source_name: string;
    document_ids: string[];
  }>;
  preview: Array<{
    instruction: string;
    input: string;
    output: string;
    conversation_id: string;
    source_type?: "conversation" | "document";
    source_name?: string;
    document_ids?: string[];
  }>;
  conversations_used: number;
}

// ── Model Info types ────────────────────────────────────────

export interface ModelDetailInfo {
  name: string;
  context_length: number | null;
  details: Record<string, unknown>;
  parameters: string;
}

export interface TrainingCapabilities {
  gpu_available: boolean;
  gpu_memory_gb: number | null;
  cpu_ram_gb: number;
  recommended_device: "gpu" | "cpu";
  required_gpu_memory_gb: number;
  required_cpu_ram_gb: number;
  can_train_on_gpu: boolean;
  can_train_on_cpu: boolean;
  message: string;
}

export interface FinetunedModel {
  id: string;
  name: string;
  display_name: string;
  base_model: string;
  base_model_hf: string;
  training_examples: number;
  final_loss: number | null;
  model_path: string;
  is_registered: boolean;
  created_at: string;
  registered_at: string | null;
}
