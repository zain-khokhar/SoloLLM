export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  token_count: number;
  is_continuation: boolean;
  continuation_of: string | null;
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
