from pydantic import BaseModel, Field
from typing import Optional


# ── Chat ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    workspace_id: Optional[str] = "default"
    documents_only: Optional[bool] = False


class ContinueRequest(BaseModel):
    conversation_id: str
    message_id: str


# ── Conversations ───────────────────────────────────────────

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    model: str
    system_prompt: str = ""
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    token_count: int = 0
    is_continuation: bool = False
    continuation_of: Optional[str] = None
    documents_used: list[str] = []
    created_at: str


class ConversationWithMessages(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse]


# ── Models ──────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class PullModelRequest(BaseModel):
    name: str = Field(..., min_length=1)


# ── System ──────────────────────────────────────────────────

class SystemProfile(BaseModel):
    gpu_name: Optional[str] = None
    vram_mb: Optional[int] = None
    ram_mb: Optional[int] = None
    cpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    os_info: Optional[str] = None
    profiled_at: Optional[str] = None
    recommended_models: list[str] = []


class SettingsUpdate(BaseModel):
    ollama_base_url: Optional[str] = None
    default_model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    auto_continue: Optional[bool] = None
    system_prompt: Optional[str] = None


# ── Distillation ────────────────────────────────────────────

class DistillationSettings(BaseModel):
    distillation_enabled: Optional[bool] = None
    context_compression: Optional[bool] = None
    compression_target_ratio: Optional[float] = None
    deduplication_enabled: Optional[bool] = None
    adaptive_prompts: Optional[bool] = None
    query_decomposition: Optional[bool] = None
    multi_hop_retrieval: Optional[bool] = None
    multi_hop_max_hops: Optional[int] = None
    self_verification: Optional[bool] = None
    chain_of_density: Optional[bool] = None
    chain_of_density_iterations: Optional[int] = None
    confidence_scoring: Optional[bool] = None
    conversation_memory_compression: Optional[bool] = None


class ChainOfDensityRequest(BaseModel):
    content: str = Field(..., min_length=1)
    model: Optional[str] = None
    iterations: int = 2


class SelfVerifyRequest(BaseModel):
    response: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    model: Optional[str] = None


class DistilledQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    workspace_id: str = "default"
    top_k: int = 5
    conversation_id: Optional[str] = None


# ── Agent (Phase 5) ────────────────────────────────────────

class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: Optional[str] = None
    max_steps: int = 10


class AgentMemoryCreate(BaseModel):
    content: str = Field(..., min_length=1)
    category: str = "general"


class AgentSettings(BaseModel):
    agent_enabled: Optional[bool] = None
    agent_max_steps: Optional[int] = None
    agent_temperature: Optional[float] = None


# ── Threads ─────────────────────────────────────────────────

class ThreadCreate(BaseModel):
    title: str = "New Thread"
    system_prompt: Optional[str] = ""
    context_mode: str = "isolated"


class ThreadUpdate(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    context_mode: Optional[str] = None


class ThreadSettingsUpdate(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    rag_enabled: Optional[bool] = None
    rag_top_k: Optional[int] = None
    compression_enabled: Optional[bool] = None
    memory_layers: Optional[int] = None
    max_history_messages: Optional[int] = None
    compression_ratio: Optional[float] = None
    # Precision mode controls
    rag_precision_mode: Optional[str] = None
    rag_vector_min_score: Optional[float] = None
    rag_lexical_required_coverage: Optional[float] = None
    rag_candidate_pool_size: Optional[int] = None
    rag_per_document_cap: Optional[int] = None
    rag_use_mmr: Optional[bool] = None
    rag_mmr_lambda: Optional[float] = None
