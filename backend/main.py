import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.auth import has_admin_access, is_private_access_enabled, is_request_authenticated
from storage.database import init_db
from api import academic, agent, auth, chat, conversations, dashboard, distillation, documents, export_import, graph, models, openai_compat, quantize, system, threads, training

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_GATES = (
    ("agent_enabled", "/api/agent"),
    ("training_enabled", "/api/training"),
    ("quantize_enabled", "/api/quantize"),
    ("academic_enabled", "/api/academic"),
    ("export_import_enabled", "/api/export"),
)

ADMIN_ROUTE_RULES = (
    ({"POST", "PUT", "DELETE"}, "/api/runtime"),
    ({"PUT"}, "/api/settings"),
    ({"POST"}, "/api/system/profile"),
    ({"POST"}, "/api/models/pull"),
    ({"DELETE"}, "/api/models"),
    ({"*"}, "/api/training"),
    ({"*"}, "/api/quantize"),
    ({"*"}, "/api/export"),
)

PUBLIC_ROUTE_PREFIXES = (
    "/api/auth",
)

PUBLIC_ROUTE_PATHS = {
    "/api/health",
}

# Multipart parser can produce extremely verbose per-chunk debug logs on file upload.
logging.getLogger("python_multipart").setLevel(logging.INFO)
logging.getLogger("python_multipart.multipart").setLevel(logging.INFO)


@dataclass
class _WarmupResult:
    content: str
    score: float = 0.0
    section_title: str = ""


def _path_matches(prefix: str, path: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def _feature_enabled(setting_name: str) -> bool:
    return bool(getattr(settings, setting_name, True))


def _requires_admin_token(method: str, path: str) -> bool:
    normalized_method = method.upper()
    normalized_path = path.rstrip("/") or "/"
    for methods, prefix in ADMIN_ROUTE_RULES:
        if _path_matches(prefix, normalized_path) and ("*" in methods or normalized_method in methods):
            return True
    return False


def _is_public_route(path: str) -> bool:
    if path in PUBLIC_ROUTE_PATHS:
        return True
    return any(_path_matches(prefix, path) for prefix in PUBLIC_ROUTE_PREFIXES)


async def _run_cold_start_warmup():
    """Preload expensive RAG pieces so first user request stays responsive."""
    if settings.cold_start_warmup_delay_seconds > 0:
        await asyncio.sleep(settings.cold_start_warmup_delay_seconds)

    logger.info("Cold-start warmup: started")

    # 1) Warm embedding model (can trigger initial model load/cache)
    try:
        from rag.embeddings import embedding_engine
        await asyncio.to_thread(embedding_engine.embed_query, settings.cold_start_warmup_query)
        logger.info("Cold-start warmup: embedding engine ready")
    except Exception as e:
        logger.warning(f"Cold-start warmup embedding skipped: {e}")

    # 2) Warm reranker path (cross-encoder if enabled, heuristic otherwise)
    try:
        from rag.reranker import reranker
        warmup_results = [_WarmupResult(content="warmup context")]
        await asyncio.to_thread(
            reranker.rerank,
            settings.cold_start_warmup_query,
            warmup_results,
            1,
        )
        logger.info("Cold-start warmup: reranker path ready")
    except Exception as e:
        logger.warning(f"Cold-start warmup reranker skipped: {e}")

    # 3) Optional lightweight RAG probe if documents exist
    if settings.cold_start_warmup_run_rag_probe:
        try:
            from rag.vectorstore import vector_store
            from rag.pipeline import rag_pipeline

            docs = await vector_store.list_documents("default")
            if docs:
                await rag_pipeline.query(
                    query=settings.cold_start_warmup_query,
                    workspace_id="default",
                    top_k=1,
                    rerank=False,
                )
                logger.info("Cold-start warmup: RAG probe complete")
            else:
                logger.info("Cold-start warmup: no documents, skipping RAG probe")
        except Exception as e:
            logger.warning(f"Cold-start warmup RAG probe skipped: {e}")

    logger.info("Cold-start warmup: finished")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    warmup_task: asyncio.Task | None = None

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(
        "Paths: base_dir=%s data_dir=%s db_path=%s",
        settings.base_dir,
        settings.data_dir,
        settings.db_path,
    )
    logger.info(
        "Network: public_base_url=%s host=%s port=%s ollama_base_url=%s",
        settings.public_base_url,
        settings.host,
        settings.port,
        settings.ollama_base_url,
    )
    logger.info(
        "CORS: origins=%s regex=%s allowed_hosts=%s",
        settings.cors_origins,
        settings.cors_origin_regex,
        settings.allowed_hosts,
    )
    logger.info(
        "Features: agent=%s academic=%s export_import=%s training=%s quantize=%s runtime=%s",
        settings.agent_enabled,
        settings.academic_enabled,
        settings.export_import_enabled,
        settings.training_enabled,
        settings.quantize_enabled,
        settings.runtime_management_enabled,
    )
    logger.info("Admin route protection: %s", "enabled" if settings.admin_api_token else "disabled")
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
    logger.info(
        "Reranker: %s (model=%s, local_only=%s)",
        "cross-encoder" if settings.reranker_enabled else "heuristic",
        settings.reranker_model_name,
        settings.reranker_local_files_only,
    )
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

    # Warm expensive components in the background so first chat is fast.
    if settings.cold_start_warmup_enabled:
        warmup_task = asyncio.create_task(_run_cold_start_warmup())
        logger.info(
            "Cold-start warmup scheduled (delay=%.1fs)",
            settings.cold_start_warmup_delay_seconds,
        )

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

    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

if settings.allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(conversations.router, prefix="/api", tags=["Conversations"])
app.include_router(auth.router, tags=["Auth"])
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
app.include_router(academic.router, tags=["Academic"])
app.include_router(quantize.router, tags=["Quantize"])


@app.middleware("http")
async def deployment_guard_middleware(request: Request, call_next):
    if request.method.upper() == "OPTIONS":
        return await call_next(request)

    path = request.url.path.rstrip("/") or "/"

    if is_private_access_enabled() and not _is_public_route(path) and not is_request_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"detail": "Owner authentication required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    for setting_name, prefix in FEATURE_GATES:
        if not _feature_enabled(setting_name) and _path_matches(prefix, path):
            return JSONResponse(status_code=404, content={"detail": "Feature disabled"})

    if not settings.runtime_management_enabled and _path_matches("/api/runtime", path):
        return JSONResponse(status_code=404, content={"detail": "Runtime management disabled"})

    if _requires_admin_token(request.method, path) and not has_admin_access(request):
        return JSONResponse(
            status_code=401,
            content={"detail": "Admin token required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


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
