import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_BASE_DIR = (
    WORKSPACE_ROOT
    if (WORKSPACE_ROOT / "backend").exists() and (WORKSPACE_ROOT / "frontend").exists()
    else PACKAGE_ROOT
)
DEFAULT_ENV_FILES = (
    str(PACKAGE_ROOT / ".env"),
    str(WORKSPACE_ROOT / ".env"),
)
DEFAULT_CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOLOLLM_",
        env_file=DEFAULT_ENV_FILES,
        env_ignore_empty=True,
        extra="ignore",
    )

    # Application
    app_name: str = "SoloLLM"
    app_version: str = "0.1.0"
    debug: bool = True

    # Paths
    base_dir: Path = DEFAULT_BASE_DIR
    data_dir: Path = DEFAULT_BASE_DIR / "data"
    db_path: Path = data_dir / "db" / "solollm.db"
    public_base_url: str = "http://127.0.0.1:8000"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "llama3.2:1b"

    # Embedded Ollama
    ollama_auto_start: bool = True
    ollama_binary_dir: Path = data_dir / "ollama"
    ollama_models_dir: Path = data_dir / "models"
    ollama_port: int = 11434

    # Inference
    max_tokens: int = 2048
    temperature: float = 0.7
    context_window: int = 4096
    auto_continue: bool = True
    gpu_layers: int = -1  # -1 = auto (Ollama decides), 0 = CPU only, N = N layers on GPU
    max_power_mode: bool = True  # Force max GPU+CPU on every prompt (--gpu --no-cpu-offload)

    # Continuation
    continuation_overlap_chars: int = 200
    max_continuations: int = 5
    truncation_detection: bool = True

    # Phase 3 — Context Distillation
    distillation_enabled: bool = True
    context_compression: bool = True
    compression_target_ratio: float = 0.6
    deduplication_enabled: bool = True
    dedup_similarity_threshold: float = 0.85
    adaptive_prompts: bool = True
    query_decomposition: bool = True
    multi_hop_retrieval: bool = True
    multi_hop_max_hops: int = 2
    self_verification: bool = False  # Off by default — doubles LLM calls
    chain_of_density: bool = False   # Off by default — multiple LLM passes
    chain_of_density_iterations: int = 2
    confidence_scoring: bool = True
    conversation_memory_compression: bool = True
    max_recent_messages: int = 10
    max_memory_tokens: int = 4000

    # Phase 4 — Knowledge Graph & Memory
    knowledge_graph_enabled: bool = True
    entity_extraction_on_ingest: bool = True
    graph_augmented_retrieval: bool = True
    web_scraping_enabled: bool = True
    web_scrape_timeout: int = 30
    max_scrape_content_mb: int = 5

    # RAG ingestion guardrails
    rag_max_chunks_per_document: int = 8000
    rag_embedding_batch_size: int = 128
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    vector_index_backend: str = "faiss"

    # RAG reranker controls (avoid long first-request model downloads by default)
    reranker_enabled: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_local_files_only: bool = True

    # RAG Precision Mode
    rag_precision_mode: str = "legacy_rrf"  # "legacy_rrf" | "precision_fusion"
    rag_vector_min_score: float = 0.28
    rag_lexical_required_coverage: float = 0.5
    rag_candidate_pool_size: int = 80
    rag_per_document_cap: int = 2
    rag_use_mmr: bool = True
    rag_mmr_lambda: float = 0.65
    rag_pre_rerank_limit: int = 24
    rag_final_context_chunks: int = 4

    # Cold-start warmup
    cold_start_warmup_enabled: bool = True
    cold_start_warmup_delay_seconds: float = 1.5
    cold_start_warmup_query: str = "warmup"
    cold_start_warmup_run_rag_probe: bool = True

    # Phase 5 — Agent Framework
    agent_enabled: bool = True
    agent_max_steps: int = 10
    agent_temperature: float = 0.2
    agent_tools_enabled: list[str] = [
        "calculator", "code_runner", "file_reader", "file_writer",
        "web_search", "datetime", "rag_search", "knowledge_graph", "memory",
    ]

    # Academic Auto-Generation
    academic_enabled: bool = True
    academic_data_dir: Path = data_dir / "academic_outputs"
    academic_preprocessing_model: str = ""  # empty = use default_model
    academic_generation_model: str = ""  # empty = use default_model
    academic_max_concurrent_jobs: int = 2
    export_import_enabled: bool = True
    training_enabled: bool = True
    quantize_enabled: bool = True
    runtime_management_enabled: bool = True
    private_access_enabled: bool = False
    owner_username: str = "admin"
    owner_password: str = ""
    session_secret: str = ""
    session_ttl_hours: int = 168
    admin_api_token: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    cors_origin_regex: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _parse_string_list(cls, value: Any) -> Any:
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value


settings = Settings()

# Ensure data directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "db").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "documents").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "cache").mkdir(parents=True, exist_ok=True)
settings.ollama_binary_dir.mkdir(parents=True, exist_ok=True)
settings.ollama_models_dir.mkdir(parents=True, exist_ok=True)
