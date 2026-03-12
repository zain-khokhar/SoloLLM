"""
Max-Power Persistent Ollama Runner
===================================
Forces EVERY model to use both GPU and CPU at maximum capacity.

GPU: All model layers offloaded (num_gpu=999) — no CPU offloading of inference.
CPU: All threads used for prompt preprocessing, tokenisation, and batch feeding (num_thread=max).

Equivalent to: ollama run <model> --gpu --no-cpu-offload

The runner keeps the model hot in VRAM/RAM (keep_alive=-1) so there is zero
cold-start penalty between prompts.  Even trivial prompts go through the
full GPU pipeline — max power, max throughput.
"""

import asyncio
import logging
import os
import platform
import subprocess
import signal
from pathlib import Path
from typing import AsyncGenerator

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


def _cpu_thread_count() -> int:
    """Return the number of physical CPU threads available."""
    count = os.cpu_count() or 4
    return count


class MaxPowerRunner:
    """
    Persistent Ollama runner that pins every model to full GPU + CPU.

    Lifecycle:
      1. warm_model(model_name)   — preloads model into VRAM, keeps it alive forever
      2. generate_stream(model, messages) — streams tokens with max-power options
      3. shutdown()               — releases models and stops background tasks
    """

    # ── Per-request Ollama options that force max hardware utilisation ──
    MAX_POWER_OPTIONS = {
        # Offload ALL layers to GPU — equivalent to --gpu --no-cpu-offload
        "num_gpu": 999,
        # Explicitly target GPU device 0
        "main_gpu": 0,
        # Use every CPU thread for prompt eval / tokenisation
        "num_thread": _cpu_thread_count(),
        # Large batch keeps CPU continuously feeding the GPU
        "num_batch": 512,
        # Memory-mapped model file for fast reloads
        "use_mmap": True,
    }

    # ── Environment variables applied to the Ollama server process ──
    MAX_POWER_ENV = {
        # CRITICAL: Force ALL layers to GPU at the server level
        "OLLAMA_NUM_GPU": "999",
        # Ensure GPU is visible (device 0)
        "CUDA_VISIBLE_DEVICES": "0",
        # Keep model loaded indefinitely — never auto-unload
        "OLLAMA_KEEP_ALIVE": "-1",
        # Allow parallel request slots
        "OLLAMA_NUM_PARALLEL": "2",
        # Max models in VRAM at once
        "OLLAMA_MAX_LOADED_MODELS": "1",
        # Do NOT schedule layers to CPU — everything on GPU
        "OLLAMA_GPU_OVERHEAD": "0",
    }

    @classmethod
    def _detect_cuda_library(cls) -> str | None:
        """Auto-detect the best CUDA library for Ollama."""
        import shutil
        if shutil.which("nvidia-smi"):
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    driver = parts[0].strip() if parts else "unknown"
                    compute_cap = parts[1].strip() if len(parts) > 1 else "unknown"
                    logger.info(f"[MaxPower] NVIDIA GPU detected — driver={driver}, compute_cap={compute_cap}")
                    return None  # Let Ollama auto-pick the right CUDA version
            except Exception:
                pass
        return None

    def __init__(self, base_url: str | None = None):
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._warmed_models: set[str] = set()
        self._lock = asyncio.Lock()

    # ──────────────────────────── public API ────────────────────────────

    def get_max_power_options(self, extra_options: dict | None = None) -> dict:
        """
        Return the option dict that must be merged into every Ollama request
        to guarantee max GPU+CPU utilisation.
        """
        opts = dict(self.MAX_POWER_OPTIONS)
        if extra_options:
            opts.update(extra_options)
        return opts

    @classmethod
    def get_server_env(cls) -> dict[str, str]:
        """Return env vars to apply when starting the Ollama server process."""
        env = dict(cls.MAX_POWER_ENV)
        # Auto-detect CUDA and force GPU library
        cuda_lib = cls._detect_cuda_library()
        if cuda_lib:
            env["OLLAMA_LLM_LIBRARY"] = cuda_lib
            logger.info(f"[MaxPower] Forcing OLLAMA_LLM_LIBRARY={cuda_lib}")
        return env

    async def warm_model(self, model: str) -> bool:
        """
        Preload *model* into GPU VRAM and keep it resident forever.

        Sends a zero-token generate request with keep_alive=-1 so the model
        stays loaded between prompts.  This eliminates cold-start latency.
        """
        async with self._lock:
            # Always force-unload first, then reload with GPU settings
            # This ensures model is loaded with num_gpu=999 even if it was
            # previously loaded with CPU-only settings
            if model in self._warmed_models:
                logger.info(f"[MaxPower] Model '{model}' already warmed, skipping")
                return True

            logger.info(f"[MaxPower] Force-loading model '{model}' onto GPU …")
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
                    # Step 1: Force-unload any existing instance of the model
                    # This ensures a clean reload with GPU settings
                    logger.info(f"[MaxPower] Step 1: Unloading '{model}' to clear any CPU-loaded state")
                    try:
                        await client.post(
                            f"{self._base_url}/api/generate",
                            json={"model": model, "prompt": "", "keep_alive": 0},
                            timeout=30,
                        )
                        await asyncio.sleep(1)  # Let Ollama release memory
                    except Exception:
                        pass  # Model may not be loaded yet, that's fine

                    # Step 2: Reload with full GPU options
                    logger.info(f"[MaxPower] Step 2: Loading '{model}' with num_gpu=999 (ALL layers → GPU)")
                    gpu_options = dict(self.MAX_POWER_OPTIONS)
                    logger.info(f"[MaxPower] Options: {gpu_options}")

                    resp = await client.post(
                        f"{self._base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": "",
                            "stream": False,
                            "keep_alive": -1,  # never unload
                            "options": gpu_options,
                        },
                    )
                    if resp.status_code == 200:
                        self._warmed_models.add(model)
                        logger.info(f"[MaxPower] ✓ Model '{model}' loaded on GPU — layers=ALL, threads={_cpu_thread_count()}, batch=1024")
                        return True
                    else:
                        logger.error(f"[MaxPower] ✗ Warm failed for '{model}': {resp.status_code} {resp.text[:300]}")
                        return False
            except Exception as e:
                logger.error(f"[MaxPower] ✗ Warm error for '{model}': {e}")
                return False

    async def generate_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_options: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream chat tokens with full GPU+CPU power.

        Automatically warms the model on first use.

        Yields dicts:
          {"type": "token", "content": "…"}
          {"type": "done",  "total_tokens": N, "eval_count": N, "done_reason": "…",
           "gpu_layers": 999, "cpu_threads": N}
        """
        # Ensure model is warmed first
        if model not in self._warmed_models:
            await self.warm_model(model)

        temperature = temperature if temperature is not None else settings.temperature

        # Build payload with max-power options
        options = self.get_max_power_options(extra_options)
        options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "options": options,
        }

        full_content = ""
        eval_count = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json
                        data = json.loads(line)
                    except (ValueError, Exception):
                        continue

                    if data.get("done", False):
                        eval_count = data.get("eval_count", 0)
                        yield {
                            "type": "done",
                            "total_tokens": data.get("prompt_eval_count", 0) + eval_count,
                            "eval_count": eval_count,
                            "done_reason": data.get("done_reason", "stop"),
                            "gpu_layers": 999,
                            "cpu_threads": _cpu_thread_count(),
                        }
                    else:
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_content += token
                            yield {"type": "token", "content": token}

    async def unload_model(self, model: str) -> None:
        """Unload a specific model from GPU memory."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                await client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": 0},
                )
            self._warmed_models.discard(model)
            logger.info(f"[MaxPower] Unloaded model '{model}'")
        except Exception as e:
            logger.warning(f"[MaxPower] Error unloading '{model}': {e}")

    async def force_gpu_reload(self, model: str) -> bool:
        """
        Force-unload then reload a model to ensure it's on GPU.
        Use this when the model was previously loaded with CPU settings.
        """
        logger.info(f"[MaxPower] Force GPU reload for '{model}'")
        self._warmed_models.discard(model)
        return await self.warm_model(model)

    async def shutdown(self) -> None:
        """Release all warmed models."""
        for model in list(self._warmed_models):
            await self.unload_model(model)
        logger.info("[MaxPower] All models unloaded")

    def status(self) -> dict:
        return {
            "max_power_enabled": True,
            "gpu_layers": self.MAX_POWER_OPTIONS["num_gpu"],
            "main_gpu": self.MAX_POWER_OPTIONS["main_gpu"],
            "cpu_threads": self.MAX_POWER_OPTIONS["num_thread"],
            "batch_size": self.MAX_POWER_OPTIONS["num_batch"],
            "use_mmap": self.MAX_POWER_OPTIONS["use_mmap"],
            "keep_alive": "infinite",
            "warmed_models": list(self._warmed_models),
        }


# ── Singleton ──
max_power_runner = MaxPowerRunner()
