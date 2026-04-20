"""
Ollama Lifecycle Manager — Auto-download, start, and manage the Ollama binary.
Ensures Ollama is available before the backend tries to connect.
"""

import asyncio
import logging
import platform
import shutil
import signal
import subprocess
import sys
import zipfile
import tarfile
import io
from pathlib import Path
from typing import Callable

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

# Ollama release URLs per platform
OLLAMA_VERSION = "0.6.2"
OLLAMA_RELEASES = {
    "Windows": f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-windows-amd64.zip",
    "Linux": f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tgz",
    "Darwin": f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-darwin",
}

# Comprehensive catalog of popular Ollama models
AVAILABLE_MODELS_CATALOG = [
    # Llama family
    {"name": "llama3.2:1b", "display_name": "Llama 3.2 1B", "family": "Llama", "parameters": "1B", "size_gb": 0.8, "description": "Ultra-light, fast responses. Great for simple tasks.", "recommended_vram_mb": 1500, "recommended_ram_mb": 4000},
    {"name": "llama3.2:latest", "display_name": "Llama 3.2 3B", "family": "Llama", "parameters": "3B", "size_gb": 2.0, "description": "Fast and capable. Best balance for most users.", "recommended_vram_mb": 3000, "recommended_ram_mb": 6000},
    {"name": "llama3.1:8b", "display_name": "Llama 3.1 8B", "family": "Llama", "parameters": "8B", "size_gb": 4.7, "description": "High quality general-purpose model.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    # Gemma family
    {"name": "gemma2:2b", "display_name": "Gemma 2 2B", "family": "Gemma", "parameters": "2B", "size_gb": 1.6, "description": "Google's compact model. Great for quick tasks.", "recommended_vram_mb": 2000, "recommended_ram_mb": 4000},
    {"name": "gemma2:9b", "display_name": "Gemma 2 9B", "family": "Gemma", "parameters": "9B", "size_gb": 5.4, "description": "Google's powerful model. Excellent quality.", "recommended_vram_mb": 7000, "recommended_ram_mb": 12000},
    {"name": "gemma2:27b", "display_name": "Gemma 2 27B", "family": "Gemma", "parameters": "27B", "size_gb": 16.0, "description": "Google's largest Gemma. Best quality, needs strong hardware.", "recommended_vram_mb": 18000, "recommended_ram_mb": 32000},
    # Qwen family
    {"name": "qwen2.5:0.5b", "display_name": "Qwen 2.5 0.5B", "family": "Qwen", "parameters": "0.5B", "size_gb": 0.4, "description": "Tiny and fast. Good for basic tasks.", "recommended_vram_mb": 1000, "recommended_ram_mb": 2000},
    {"name": "qwen2.5:1.5b", "display_name": "Qwen 2.5 1.5B", "family": "Qwen", "parameters": "1.5B", "size_gb": 1.0, "description": "Small but capable. Good for everyday use.", "recommended_vram_mb": 1500, "recommended_ram_mb": 4000},
    {"name": "qwen2.5:3b", "display_name": "Qwen 2.5 3B", "family": "Qwen", "parameters": "3B", "size_gb": 1.9, "description": "Balanced quality and speed from Alibaba.", "recommended_vram_mb": 3000, "recommended_ram_mb": 6000},
    {"name": "qwen2.5:7b", "display_name": "Qwen 2.5 7B", "family": "Qwen", "parameters": "7B", "size_gb": 4.4, "description": "Strong multi-language model.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    {"name": "qwen2.5:14b", "display_name": "Qwen 2.5 14B", "family": "Qwen", "parameters": "14B", "size_gb": 9.0, "description": "Powerful reasoning. Needs good hardware.", "recommended_vram_mb": 10000, "recommended_ram_mb": 18000},
    # Mistral family
    {"name": "mistral:7b", "display_name": "Mistral 7B", "family": "Mistral", "parameters": "7B", "size_gb": 4.1, "description": "Excellent general model from Mistral AI.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    {"name": "mixtral:8x7b", "display_name": "Mixtral 8x7B", "family": "Mistral", "parameters": "47B (MoE)", "size_gb": 26.0, "description": "Mixture of Experts. Top-tier quality, needs 32 GB+ RAM.", "recommended_vram_mb": 28000, "recommended_ram_mb": 48000},
    # Phi family
    {"name": "phi3:mini", "display_name": "Phi-3 Mini 3.8B", "family": "Phi", "parameters": "3.8B", "size_gb": 2.3, "description": "Microsoft's small model. Good reasoning for its size.", "recommended_vram_mb": 3000, "recommended_ram_mb": 6000},
    {"name": "phi3:medium", "display_name": "Phi-3 Medium 14B", "family": "Phi", "parameters": "14B", "size_gb": 7.9, "description": "Microsoft's larger Phi. Strong at math and code.", "recommended_vram_mb": 10000, "recommended_ram_mb": 16000},
    # Code-specialized
    {"name": "codellama:7b", "display_name": "Code Llama 7B", "family": "Code Llama", "parameters": "7B", "size_gb": 3.8, "description": "Specialized for code generation and understanding.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    {"name": "codellama:13b", "display_name": "Code Llama 13B", "family": "Code Llama", "parameters": "13B", "size_gb": 7.4, "description": "Larger code model. Better for complex programming tasks.", "recommended_vram_mb": 10000, "recommended_ram_mb": 16000},
    {"name": "deepseek-coder:6.7b", "display_name": "DeepSeek Coder 6.7B", "family": "DeepSeek", "parameters": "6.7B", "size_gb": 3.8, "description": "Strong code generation from DeepSeek.", "recommended_vram_mb": 5000, "recommended_ram_mb": 10000},
    {"name": "starcoder2:3b", "display_name": "StarCoder2 3B", "family": "StarCoder", "parameters": "3B", "size_gb": 1.7, "description": "Compact code model. Fast and accurate.", "recommended_vram_mb": 3000, "recommended_ram_mb": 6000},
    # Small / Lightweight
    {"name": "tinyllama:latest", "display_name": "TinyLlama 1.1B", "family": "TinyLlama", "parameters": "1.1B", "size_gb": 0.6, "description": "Smallest usable model. Runs on almost anything.", "recommended_vram_mb": 1000, "recommended_ram_mb": 2000},
    # Vision models
    {"name": "llava:7b", "display_name": "LLaVA 7B", "family": "LLaVA", "parameters": "7B", "size_gb": 4.5, "description": "Vision + language model. Can understand images.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    # DeepSeek
    {"name": "deepseek-r1:1.5b", "display_name": "DeepSeek R1 1.5B", "family": "DeepSeek", "parameters": "1.5B", "size_gb": 1.0, "description": "DeepSeek's reasoning model. Compact edition.", "recommended_vram_mb": 2000, "recommended_ram_mb": 4000},
    {"name": "deepseek-r1:7b", "display_name": "DeepSeek R1 7B", "family": "DeepSeek", "parameters": "7B", "size_gb": 4.7, "description": "Strong reasoning and math capabilities.", "recommended_vram_mb": 6000, "recommended_ram_mb": 10000},
    {"name": "deepseek-r1:14b", "display_name": "DeepSeek R1 14B", "family": "DeepSeek", "parameters": "14B", "size_gb": 9.0, "description": "DeepSeek's powerful reasoning model.", "recommended_vram_mb": 10000, "recommended_ram_mb": 18000},
]


class OllamaManager:
    """Manages the embedded Ollama binary lifecycle."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._managed = False
        self._port = settings.ollama_port
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._binary_dir = settings.ollama_binary_dir
        self._models_dir = settings.ollama_models_dir
        self._max_restarts = 3
        self._restart_count = 0
        # Download progress tracking
        self._download_progress: dict = {
            "stage": "idle",  # idle | downloading | extracting | starting | ready | error
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "percent": 0,
            "error": None,
        }

    @property
    def download_progress(self) -> dict:
        return {**self._download_progress}

    @property
    def is_managed(self) -> bool:
        """True if we started the Ollama process (vs system-installed)."""
        return self._managed

    @property
    def binary_path(self) -> Path:
        """Path to the Ollama binary."""
        system = platform.system()
        if system == "Windows":
            return self._binary_dir / "ollama.exe"
        return self._binary_dir / "ollama"

    @property
    def status(self) -> dict:
        """Current status of the Ollama manager."""
        return {
            "managed": self._managed,
            "binary_exists": self.binary_path.exists(),
            "binary_path": str(self.binary_path),
            "models_dir": str(self._models_dir),
            "port": self._port,
            "base_url": self._base_url,
            "process_running": self._process is not None and self._process.poll() is None,
        }

    async def ensure_running(self) -> bool:
        """
        Main entry point. Ensures Ollama is available.
        1. Check if already running (system-wide)
        2. If not, check for our binary
        3. If no binary, download it
        4. Start Ollama subprocess
        """
        # 1. Is Ollama already available?
        if await self._check_available():
            logger.info("Ollama is already running (system or external)")
            return True

        # 2. Do we have the binary?
        if not self.binary_path.exists():
            logger.info("Ollama binary not found, downloading...")
            try:
                await self._download_binary()
            except Exception as e:
                logger.error(f"Failed to download Ollama: {e}")
                self._download_progress.update(stage="error", error=str(e))
                return False

        # 3. Start our managed instance
        self._download_progress["stage"] = "starting"
        try:
            ok = await self._start()
            if ok:
                self._download_progress["stage"] = "ready"
            return ok
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            self._download_progress.update(stage="error", error=str(e))
            return False

    async def _check_available(self) -> bool:
        """Check if Ollama is already running on the configured port."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=3)
                return resp.status_code == 200
        except Exception:
            return False

    async def _download_binary(self, progress_callback: Callable | None = None) -> Path:
        """Download the Ollama binary for the current OS."""
        system = platform.system()
        url = OLLAMA_RELEASES.get(system)
        if not url:
            self._download_progress.update(stage="error", error=f"Unsupported OS: {system}")
            raise RuntimeError(f"Unsupported OS: {system}. Only Windows, Linux, and macOS are supported.")

        self._binary_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading Ollama v{OLLAMA_VERSION} for {system}...")
        self._download_progress.update(stage="downloading", downloaded_bytes=0, total_bytes=0, percent=0, error=None)

        async with httpx.AsyncClient(timeout=httpx.Timeout(600), follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                self._download_progress["total_bytes"] = total
                downloaded = 0
                chunks = []

                async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    pct = round((downloaded / total) * 100, 1) if total else 0
                    self._download_progress.update(downloaded_bytes=downloaded, percent=pct)
                    if progress_callback and total:
                        progress_callback(downloaded, total)

                data = b"".join(chunks)

        # Extract or save based on format
        self._download_progress["stage"] = "extracting"
        if url.endswith(".zip"):
            self._extract_zip(data)
        elif url.endswith(".tgz"):
            self._extract_tgz(data)
        else:
            # Direct binary (macOS)
            self.binary_path.write_bytes(data)

        # Make executable on Unix
        if system != "Windows":
            self.binary_path.chmod(0o755)

        logger.info(f"Ollama binary saved to {self.binary_path}")
        return self.binary_path

    def _extract_zip(self, data: bytes) -> None:
        """Extract ALL files from zip archive (Windows).
        
        IMPORTANT: Must extract everything, not just ollama.exe.
        CUDA runner DLLs and other GPU libraries are bundled in the zip.
        """
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(self._binary_dir)
            logger.info(f"Extracted {len(zf.namelist())} files to {self._binary_dir}")
            for name in zf.namelist():
                if 'cuda' in name.lower() or 'runner' in name.lower() or name.endswith('.dll'):
                    logger.info(f"  GPU file: {name}")

    def _extract_tgz(self, data: bytes) -> None:
        """Extract Ollama from a tar.gz archive (Linux)."""
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(self._binary_dir, filter="data")

        if self.binary_path.exists():
            return

        for candidate in self._binary_dir.rglob("ollama"):
            if candidate.is_file():
                shutil.copy2(candidate, self.binary_path)
                self.binary_path.chmod(0o755)
                return

        raise FileNotFoundError("Could not locate Ollama binary after Linux archive extraction")

    async def _start(self) -> bool:
        """Start Ollama as a subprocess."""
        if not self.binary_path.exists():
            logger.error(f"Ollama binary not found at {self.binary_path}")
            return False

        self._models_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **dict(__import__("os").environ),
            "OLLAMA_HOST": f"127.0.0.1:{self._port}",
            "OLLAMA_MODELS": str(self._models_dir),
        }

        # ── Max-Power: inject server-level env vars for full GPU+CPU ──
        if settings.max_power_mode:
            from core.max_power_runner import MaxPowerRunner
            mp_env = MaxPowerRunner.get_server_env()
            env.update(mp_env)
            logger.info(f"[MaxPower] Server env applied: {mp_env}")
            if env.get("OLLAMA_LLM_LIBRARY"):
                logger.info(f"[MaxPower] Using OLLAMA_LLM_LIBRARY={env['OLLAMA_LLM_LIBRARY']}")
        else:
            env["OLLAMA_NUM_PARALLEL"] = "1"

        logger.info(f"Starting Ollama on port {self._port} (models: {self._models_dir})")

        try:
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(
                [str(self.binary_path), "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
        except OSError as e:
            logger.error(f"Failed to start Ollama process: {e}")
            return False

        # Wait for it to be ready
        ready = await self._wait_for_ready(timeout=30)
        if ready:
            self._managed = True
            self._restart_count = 0
            logger.info(f"Ollama is ready on {self._base_url}")
        else:
            logger.error("Ollama failed to become ready within timeout")
            await self.stop()

        return ready

    async def _wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for Ollama health endpoint to respond."""
        for i in range(timeout * 2):
            # Check if process died
            if self._process and self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(errors="replace")[:500]
                logger.error(f"Ollama process exited with code {self._process.returncode}: {stderr}")
                return False

            if await self._check_available():
                return True

            await asyncio.sleep(0.5)

        return False

    async def restart(self) -> bool:
        """Restart Ollama (up to max_restarts times)."""
        if self._restart_count >= self._max_restarts:
            logger.error(f"Ollama restart limit ({self._max_restarts}) reached")
            return False

        self._restart_count += 1
        logger.warning(f"Restarting Ollama (attempt {self._restart_count}/{self._max_restarts})")
        await self.stop()
        return await self._start()

    async def stop(self) -> None:
        """Stop the managed Ollama subprocess."""
        if self._process is None:
            return

        logger.info("Stopping managed Ollama process...")
        try:
            if platform.system() == "Windows":
                self._process.terminate()
            else:
                self._process.send_signal(signal.SIGTERM)

            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error stopping Ollama: {e}")
        finally:
            self._process = None
            self._managed = False
            logger.info("Ollama stopped")

    async def health_check(self) -> dict:
        """Detailed health check of Ollama."""
        available = await self._check_available()
        version = None

        if available:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self._base_url}/api/version", timeout=5)
                    if resp.status_code == 200:
                        version = resp.json().get("version")
            except Exception:
                pass

        return {
            "available": available,
            "version": version,
            **self.status,
        }

    def get_model_catalog(self, vram_mb: int | None = None, ram_mb: int | None = None) -> list[dict]:
        """
        Return the model catalog with compatibility info based on system hardware.
        Each model gets a 'compatible' and 'performance_note' field.
        """
        catalog = []
        for model in AVAILABLE_MODELS_CATALOG:
            entry = {**model}

            # Determine compatibility
            if vram_mb and vram_mb >= model["recommended_vram_mb"]:
                entry["compatible"] = True
                entry["performance_note"] = "Runs on GPU — best performance"
            elif ram_mb and ram_mb >= model["recommended_ram_mb"]:
                entry["compatible"] = True
                if vram_mb and vram_mb > 0:
                    entry["performance_note"] = "Uses GPU + CPU (shared) — good performance"
                else:
                    entry["performance_note"] = "CPU only — slower but functional"
            elif ram_mb and ram_mb >= model["recommended_ram_mb"] * 0.7:
                entry["compatible"] = True
                entry["performance_note"] = "Tight fit — may be slow"
            else:
                entry["compatible"] = False
                entry["performance_note"] = "Insufficient hardware — not recommended"

            catalog.append(entry)

        return catalog


# Singleton
ollama_manager = OllamaManager()
