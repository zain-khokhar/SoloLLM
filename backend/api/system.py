import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.profiler import profile_system
from core.config import settings
from storage.database import (
    save_system_profile, get_system_profile,
    get_all_settings, set_setting, get_setting,
)
from storage.schemas import SettingsUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Runtime / Ollama Manager Endpoints ──────────────────────

@router.get("/runtime/status")
async def runtime_status():
    """Get Ollama runtime status including health and manager state."""
    from core.ollama_manager import ollama_manager
    health = await ollama_manager.health_check()
    return {
        "ollama": health,
        "auto_start_enabled": settings.ollama_auto_start,
    }


@router.get("/capabilities")
async def get_capabilities():
    """Expose deployment flags so the frontend and operators can adapt to server mode."""
    return {
        "deployment": {
            "public_base_url": settings.public_base_url,
            "admin_route_protection": bool(settings.admin_api_token),
            "private_access_enabled": settings.private_access_enabled and bool(settings.owner_password),
            "runtime_management_enabled": settings.runtime_management_enabled,
        },
        "features": {
            "agent": settings.agent_enabled,
            "academic": settings.academic_enabled,
            "export_import": settings.export_import_enabled,
            "training": settings.training_enabled,
            "quantize": settings.quantize_enabled,
        },
        "network": {
            "cors_origins": settings.cors_origins,
            "cors_origin_regex": settings.cors_origin_regex,
            "allowed_hosts": settings.allowed_hosts,
        },
    }


@router.post("/runtime/setup")
async def runtime_setup():
    """Trigger Ollama download + start (for first-time setup)."""
    from core.ollama_manager import ollama_manager
    try:
        success = await ollama_manager.ensure_running()
        health = await ollama_manager.health_check()
        return {
            "success": success,
            "ollama": health,
        }
    except Exception as e:
        logger.error(f"Runtime setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runtime/setup/progress")
async def runtime_setup_progress():
    """Stream Ollama download / setup progress via SSE."""
    from core.ollama_manager import ollama_manager

    async def _stream():
        prev = None
        while True:
            progress = ollama_manager.download_progress
            if progress != prev:
                yield {
                    "event": "progress",
                    "data": json.dumps(progress),
                }
                prev = {**progress}
                if progress["stage"] in ("ready", "error", "idle"):
                    break
            await asyncio.sleep(0.5)

    return EventSourceResponse(_stream(), media_type="text/event-stream")


@router.post("/runtime/restart")
async def runtime_restart():
    """Restart the managed Ollama instance."""
    from core.ollama_manager import ollama_manager
    if not ollama_manager.is_managed:
        raise HTTPException(status_code=400, detail="Ollama is not managed by SoloLLM (system-installed Ollama is running)")
    try:
        success = await ollama_manager.restart()
        return {"success": success}
    except Exception as e:
        logger.error(f"Ollama restart failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runtime/models/catalog")
async def get_model_catalog():
    """Get the full model catalog with hardware compatibility info."""
    from core.ollama_manager import ollama_manager

    # Get system profile for compatibility checking
    profile = await get_system_profile()
    vram_mb = None
    ram_mb = None
    if profile:
        vram_mb = profile.get("vram_mb")
        ram_mb = profile.get("ram_mb")

    if not ram_mb:
        # Quick profile if not done yet
        from core.profiler import profile_system
        hw = profile_system()
        vram_mb = hw.get("vram_mb")
        ram_mb = hw.get("ram_mb")
        try:
            await save_system_profile(hw)
        except Exception:
            pass

    catalog = ollama_manager.get_model_catalog(vram_mb=vram_mb, ram_mb=ram_mb)
    return {
        "catalog": catalog,
        "hardware": {
            "vram_mb": vram_mb,
            "ram_mb": ram_mb,
        },
    }


@router.get("/system/profile")
async def get_profile():
    """Get cached system hardware profile."""
    profile = await get_system_profile()
    if not profile:
        # Auto-profile on first request
        profile = profile_system()
        await save_system_profile(profile)
        profile = await get_system_profile()
    return {"profile": profile}


@router.post("/system/profile")
async def run_profiler():
    """Re-run the hardware profiler and update the cached profile."""
    profile = profile_system()
    await save_system_profile(profile)
    stored = await get_system_profile()

    # Include model recommendations
    result = dict(stored) if stored else profile
    result["recommended_models"] = profile.get("recommended_models", [])
    return {"profile": result}


@router.get("/settings")
async def get_settings():
    """Get all application settings."""
    db_settings = await get_all_settings()

    # Merge with defaults
    return {
        "settings": {
            "ollama_base_url": db_settings.get("ollama_base_url", settings.ollama_base_url),
            "default_model": db_settings.get("default_model", settings.default_model),
            "max_tokens": int(db_settings.get("max_tokens", settings.max_tokens)),
            "temperature": float(db_settings.get("temperature", settings.temperature)),
            "auto_continue": db_settings.get("auto_continue", str(settings.auto_continue)).lower() == "true",
            "system_prompt": db_settings.get("system_prompt", ""),
        }
    }


@router.put("/settings")
async def update_settings(request: SettingsUpdate):
    """Update application settings."""
    updates = request.model_dump(exclude_none=True)

    # Validate default_model is actually installed before saving
    if "default_model" in updates:
        from core.inference import ollama_client
        try:
            installed = await ollama_client.list_models()
            installed_names = [m["name"] for m in installed]
            requested = updates["default_model"]
            if requested not in installed_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{requested}' is not installed. Available models: {', '.join(installed_names) or 'none'}",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")

    for key, value in updates.items():
        await set_setting(key, str(value))
        logger.info(f"Setting updated: {key}")

    # Sync default_model to runtime settings
    if "default_model" in updates:
        settings.default_model = updates["default_model"]

    # Return updated settings
    return await get_settings()


# ── Max-Power Runner Endpoints ──────────────────────────────

@router.get("/runtime/maxpower/status")
async def maxpower_status():
    """Get the current max-power runner status."""
    from core.max_power_runner import max_power_runner
    return {
        "enabled": settings.max_power_mode,
        **max_power_runner.status(),
    }


@router.post("/runtime/maxpower/warm")
async def maxpower_warm_model(model: str | None = None):
    """
    Force-load a model onto GPU at full power.
    Unloads any CPU-cached version first, then reloads with num_gpu=999.
    If model is omitted, warms the default model.
    """
    from core.max_power_runner import max_power_runner
    target = model or settings.default_model
    ok = await max_power_runner.force_gpu_reload(target)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to warm model '{target}' on GPU")
    return {
        "success": True,
        "model": target,
        **max_power_runner.status(),
    }


@router.post("/runtime/maxpower/unload")
async def maxpower_unload_model(model: str | None = None):
    """Unload a model from GPU VRAM."""
    from core.max_power_runner import max_power_runner
    target = model or settings.default_model
    await max_power_runner.unload_model(target)
    return {"success": True, "model": target}
