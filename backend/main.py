import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from storage.database import init_db
from api import chat, models, conversations, system, distillation, documents, dashboard, graph, agent, openai_compat, export_import, threads, training

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    await init_db()
    logger.info("Database initialized")

    # Ensure Ollama is running (auto-download + start if needed)
    if settings.ollama_auto_start:
        try:
            from core.ollama_manager import ollama_manager
            success = await ollama_manager.ensure_running()
            if success:
                logger.info("Ollama is ready")
            else:
                logger.warning("Ollama could not be started — models won't be available until Ollama is running")
        except Exception as e:
            logger.warning(f"Ollama auto-start failed: {e}")

    # Auto-detect default model: if the configured one isn't installed, pick the first available
    try:
        from core.inference import ollama_client
        from storage.database import get_setting, set_setting
        installed = await ollama_client.list_models()
        installed_names = [m["name"] for m in installed]
        db_default = await get_setting("default_model")
        current_default = db_default or settings.default_model
        if installed_names and current_default not in installed_names:
            new_default = installed_names[0]
            await set_setting("default_model", new_default)
            settings.default_model = new_default
            logger.info(f"Default model '{current_default}' not installed — switched to '{new_default}'")
        elif db_default and db_default in installed_names:
            settings.default_model = db_default
    except Exception as e:
        logger.warning(f"Model auto-detect skipped: {e}")

    # Initialize RAG pipeline
    try:
        from rag.pipeline import rag_pipeline
        await rag_pipeline.init()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.warning(f"RAG pipeline init skipped: {e}")

    # Initialize Knowledge Graph (Phase 4)
    try:
        from memory.knowledge_graph import knowledge_graph
        await knowledge_graph.init()
        logger.info("Knowledge graph initialized")
    except Exception as e:
        logger.warning(f"Knowledge graph init skipped: {e}")

    logger.info(f"Phase 3 — Context Distillation: {'enabled' if settings.distillation_enabled else 'disabled'}")
    logger.info(f"Phase 4 — Knowledge Graph: {'enabled' if settings.knowledge_graph_enabled else 'disabled'}")
    logger.info(f"Phase 5 — Agent Framework: {'enabled' if settings.agent_enabled else 'disabled'}")
    logger.info("Phase 6 — OpenAI-compat API, Export/Import, Dashboard: enabled")

    # ── Max-Power Runner: force-load the default model onto GPU ──
    if settings.max_power_mode:
        try:
            from core.max_power_runner import max_power_runner
            logger.info("[MaxPower] Force GPU reload — unloading CPU-cached model and reloading with num_gpu=999")
            await max_power_runner.force_gpu_reload(settings.default_model)
            logger.info(f"[MaxPower] ✓ Model '{settings.default_model}' is on GPU — max power active")
        except Exception as e:
            logger.warning(f"[MaxPower] Warm failed (non-fatal): {e}")

    yield

    # Shutdown: stop managed Ollama
    if settings.ollama_auto_start:
        try:
            from core.ollama_manager import ollama_manager
            await ollama_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping Ollama: {e}")

    # Shutdown: release warmed models
    if settings.max_power_mode:
        try:
            from core.max_power_runner import max_power_runner
            await max_power_runner.shutdown()
        except Exception as e:
            logger.warning(f"[MaxPower] Shutdown error: {e}")

    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(conversations.router, prefix="/api", tags=["Conversations"])
app.include_router(system.router, prefix="/api", tags=["System"])
app.include_router(distillation.router, tags=["Distillation"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(dashboard.router, tags=["Dashboard"])
app.include_router(graph.router, tags=["Knowledge Graph"])
app.include_router(agent.router, tags=["Agent"])
app.include_router(openai_compat.router, tags=["OpenAI-Compatible"])
app.include_router(export_import.router, tags=["Export/Import"])
app.include_router(threads.router, tags=["Threads"])
app.include_router(training.router, tags=["Training"])


# Dashboard metrics middleware
@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time as _time
    start = _time.time()
    response = await call_next(request)
    latency_ms = round((_time.time() - start) * 1000, 1)
    endpoint = f"{request.method} {request.url.path}"
    dashboard.metrics.record(
        endpoint=endpoint,
        latency_ms=latency_ms,
        success=response.status_code < 400,
    )
    return response


@app.get("/api/health")
async def health_check():
    from core.inference import ollama_client
    ollama_ok = await ollama_client.is_available()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "ollama_connected": ollama_ok,
    }
