import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "SoloLLM"
    app_version: str = "0.1.0"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    db_path: Path = data_dir / "db" / "solollm.db"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
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

    # Phase 5 — Agent Framework
    agent_enabled: bool = True
    agent_max_steps: int = 10
    agent_temperature: float = 0.2
    agent_tools_enabled: list[str] = [
        "calculator", "code_runner", "file_reader", "file_writer",
        "web_search", "datetime", "rag_search", "knowledge_graph", "memory",
    ]

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_prefix = "SOLOLLM_"
        env_file = ".env"


settings = Settings()

# Ensure data directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "db").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "documents").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "cache").mkdir(parents=True, exist_ok=True)
settings.ollama_binary_dir.mkdir(parents=True, exist_ok=True)
settings.ollama_models_dir.mkdir(parents=True, exist_ok=True)
