"""
GGUF Model Quantization Manager.

Handles downloading llama.cpp tools, HuggingFace model download,
GGUF conversion, quantization, and Ollama import.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_USE_THREAD_SUBPROCESS = sys.platform == "win32"

# ── Quantization types ────────────────────────────────────

QUANT_TYPES = {
    "Q2_K": {
        "description": "Extreme compression",
        "size_ratio": 0.14,
        "quality_note": "Noticeable quality loss, only for experimentation",
        "bits_per_weight": 2.63,
    },
    "Q3_K_M": {
        "description": "Very small",
        "size_ratio": 0.19,
        "quality_note": "Some quality loss, usable for simple tasks",
        "bits_per_weight": 3.44,
    },
    "Q4_0": {
        "description": "Small, good quality",
        "size_ratio": 0.24,
        "quality_note": "Good for most tasks, fast inference",
        "bits_per_weight": 4.50,
    },
    "Q4_K_M": {
        "description": "Recommended balance",
        "size_ratio": 0.27,
        "quality_note": "Best default — minimal quality loss, great speed",
        "bits_per_weight": 4.83,
    },
    "Q5_0": {
        "description": "Medium size",
        "size_ratio": 0.31,
        "quality_note": "Very good quality, moderate speed",
        "bits_per_weight": 5.50,
    },
    "Q5_K_M": {
        "description": "Medium, excellent quality",
        "size_ratio": 0.34,
        "quality_note": "Near-imperceptible quality loss",
        "bits_per_weight": 5.69,
    },
    "Q6_K": {
        "description": "Large, near-lossless",
        "size_ratio": 0.40,
        "quality_note": "Practically lossless, slower inference",
        "bits_per_weight": 6.56,
    },
    "Q8_0": {
        "description": "Largest quantized",
        "size_ratio": 0.53,
        "quality_note": "Minimal quality loss, largest file size",
        "bits_per_weight": 8.50,
    },
    "F16": {
        "description": "Full precision (no quantization)",
        "size_ratio": 1.0,
        "quality_note": "No quantization applied, original quality",
        "bits_per_weight": 16.0,
    },
}


# ── Job dataclass ─────────────────────────────────────────

@dataclass
class QuantizeJob:
    id: str
    status: str  # pending | running | complete | error | cancelled
    source_type: str  # local_gguf | huggingface
    source: str
    quant_level: str
    output_name: str
    import_to_ollama: bool = True
    progress: float = 0.0
    stage: str = "pending"  # pending | downloading | converting | quantizing | importing | complete
    message: str = "Waiting to start..."
    error: str | None = None
    output_path: str | None = None
    output_size: int | None = None
    created_at: str = ""
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Manager ───────────────────────────────────────────────

class QuantizeManager:
    """Manages GGUF quantization jobs."""

    def __init__(self):
        self.tools_dir = Path(settings.data_dir) / "quantize_tools"
        self.output_dir = Path(settings.data_dir) / "quantize_outputs"
        self.hf_cache_dir = Path(settings.data_dir) / "quantize_hf_cache"
        self._jobs: dict[str, QuantizeJob] = {}
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}

        for d in (self.tools_dir, self.output_dir, self.hf_cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Tool management ───────────────────────────────────

    @property
    def _quantize_bin(self) -> Path:
        return self.tools_dir / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")

    @property
    def _convert_script(self) -> Path:
        return self.tools_dir / "convert_hf_to_gguf.py"

    def tools_ready(self) -> bool:
        return self._quantize_bin.exists()

    def get_tools_status(self) -> dict:
        return {
            "ready": self.tools_ready(),
            "tools_path": str(self.tools_dir),
            "has_quantize": self._quantize_bin.exists(),
            "has_convert": self._convert_script.exists(),
        }

    async def setup_tools(self) -> AsyncGenerator[dict, None]:
        """Download llama.cpp tools from GitHub. Yields progress events."""
        try:
            yield {"stage": "checking", "message": "Checking GitHub for latest llama.cpp release...", "percent": 5}

            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get("https://api.github.com/repos/ggerganov/llama.cpp/releases/latest")
                resp.raise_for_status()
                release = resp.json()

            tag = release.get("tag_name", "")
            assets = release.get("assets", [])

            # Find Windows CPU binary zip
            zip_asset = None
            for a in assets:
                name = a.get("name", "").lower()
                if "win" in name and "x64" in name and name.endswith(".zip"):
                    if "cuda" not in name and "vulkan" not in name:
                        zip_asset = a
                        break
            # Fallback: accept any Windows zip
            if not zip_asset:
                for a in assets:
                    name = a.get("name", "").lower()
                    if "win" in name and name.endswith(".zip"):
                        zip_asset = a
                        break

            if not zip_asset:
                yield {"stage": "error", "message": f"No Windows binary found in release {tag}. Assets: {[a['name'] for a in assets[:10]]}", "percent": 0}
                return

            download_url = zip_asset["browser_download_url"]
            total_bytes = zip_asset.get("size", 0)
            zip_name = zip_asset["name"]

            yield {"stage": "downloading", "message": f"Downloading {zip_name}...", "percent": 10, "total_bytes": total_bytes}

            # Download the zip with progress
            zip_path = self.tools_dir / zip_name
            downloaded = 0
            async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
                async with client.stream("GET", download_url) as stream:
                    with open(zip_path, "wb") as f:
                        async for chunk in stream.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = int(10 + (downloaded / max(total_bytes, 1)) * 50) if total_bytes else 30
                            yield {"stage": "downloading", "message": f"Downloading... {downloaded // (1024*1024)} MB", "percent": min(pct, 60), "downloaded_bytes": downloaded, "total_bytes": total_bytes}

            yield {"stage": "extracting", "message": "Extracting llama-quantize...", "percent": 65}

            # Extract llama-quantize from the zip
            found_quantize = False
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                for name in zf.namelist():
                    basename = os.path.basename(name)
                    if basename.lower() in ("llama-quantize.exe", "llama-quantize"):
                        with zf.open(name) as src, open(str(self._quantize_bin), "wb") as dst:
                            dst.write(src.read())
                        if sys.platform != "win32":
                            os.chmod(str(self._quantize_bin), 0o755)
                        found_quantize = True
                        break

            if not found_quantize:
                # Try alternative name patterns
                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    for name in zf.namelist():
                        basename = os.path.basename(name).lower()
                        if "quantize" in basename and (basename.endswith(".exe") or "." not in basename):
                            with zf.open(name) as src, open(str(self._quantize_bin), "wb") as dst:
                                dst.write(src.read())
                            if sys.platform != "win32":
                                os.chmod(str(self._quantize_bin), 0o755)
                            found_quantize = True
                            break

            # Clean up zip
            try:
                zip_path.unlink()
            except Exception:
                pass

            if not found_quantize:
                yield {"stage": "error", "message": "llama-quantize binary not found in the release archive.", "percent": 0}
                return

            yield {"stage": "downloading_convert", "message": "Downloading convert_hf_to_gguf.py...", "percent": 75}

            # Download convert script from llama.cpp repo
            convert_url = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{tag}/convert_hf_to_gguf.py"
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(convert_url)
                    if resp.status_code == 200:
                        self._convert_script.write_text(resp.text, encoding="utf-8")
                    else:
                        logger.warning("Could not download convert script (status %d), HF conversion will be unavailable", resp.status_code)
            except Exception as e:
                logger.warning("Could not download convert script: %s", e)

            yield {"stage": "installing_deps", "message": "Installing gguf Python package...", "percent": 85}

            # Install gguf package
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", "gguf", "--quiet",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.wait()
            except Exception as e:
                logger.warning("Could not install gguf package: %s", e)

            yield {"stage": "ready", "message": "Tools ready!", "percent": 100}

        except Exception as e:
            logger.exception("setup_tools failed")
            yield {"stage": "error", "message": str(e), "percent": 0}

    # ── Job management ────────────────────────────────────

    def start_job(
        self,
        source_type: str,
        source: str,
        quant_level: str,
        output_name: str,
        import_to_ollama: bool = True,
    ) -> str:
        if quant_level not in QUANT_TYPES:
            raise ValueError(f"Invalid quantization level: {quant_level}")
        if source_type not in ("local_gguf", "huggingface"):
            raise ValueError("source_type must be 'local_gguf' or 'huggingface'")
        if source_type == "local_gguf":
            p = Path(source)
            if not p.exists():
                raise ValueError(f"File not found: {source}")
            if not p.suffix.lower() == ".gguf":
                raise ValueError("File must be a .gguf file")
        if not output_name.strip():
            raise ValueError("Output name is required")

        if not self.tools_ready() and quant_level != "F16":
            raise ValueError("Quantization tools not set up. Run setup first.")

        job = QuantizeJob(
            id=uuid.uuid4().hex[:16],
            status="running",
            source_type=source_type,
            source=source.strip(),
            quant_level=quant_level,
            output_name=output_name.strip(),
            import_to_ollama=import_to_ollama,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs[job.id] = job

        task = asyncio.create_task(self._run_job(job))
        self._active_tasks[job.id] = task
        return job.id

    async def _run_job(self, job: QuantizeJob):
        """Full quantization pipeline."""
        try:
            input_gguf: Path

            if job.source_type == "huggingface":
                # Stage 1: Download model from HuggingFace
                model_path = await self._download_hf_model(job, job.source)

                # Stage 2: Convert to GGUF F16
                if job.quant_level == "F16":
                    input_gguf = await self._convert_to_gguf(job, model_path, "f16")
                    # For F16, conversion IS the final output
                    final_out = self.output_dir / f"{job.output_name}.gguf"
                    if input_gguf != final_out:
                        shutil.copy2(str(input_gguf), str(final_out))
                    job.output_path = str(final_out)
                    job.output_size = final_out.stat().st_size
                else:
                    input_gguf = await self._convert_to_gguf(job, model_path, "f16")
                    # Stage 3: Quantize
                    output_path = await self._quantize_gguf(job, input_gguf, job.quant_level)
                    job.output_path = str(output_path)
                    job.output_size = output_path.stat().st_size
                    # Clean up intermediate F16
                    try:
                        input_gguf.unlink()
                    except Exception:
                        pass
            else:
                # Local GGUF
                input_gguf = Path(job.source)
                if job.quant_level == "F16":
                    # Nothing to quantize, just copy
                    final_out = self.output_dir / f"{job.output_name}.gguf"
                    shutil.copy2(str(input_gguf), str(final_out))
                    job.output_path = str(final_out)
                    job.output_size = final_out.stat().st_size
                else:
                    output_path = await self._quantize_gguf(job, input_gguf, job.quant_level)
                    job.output_path = str(output_path)
                    job.output_size = output_path.stat().st_size

            # Stage 4: Optional Ollama import
            if job.import_to_ollama and job.output_path:
                await self._import_to_ollama(job, Path(job.output_path), job.output_name)

            job.status = "complete"
            job.stage = "complete"
            job.progress = 1.0
            job.message = "Quantization complete!"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info("Quantization job %s completed: %s", job.id, job.output_path)

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "Job cancelled by user"
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.message = f"Error: {e}"
            logger.exception("Quantization job %s failed", job.id)
        finally:
            self._active_processes.pop(job.id, None)
            self._active_tasks.pop(job.id, None)

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of all files in a directory (bytes)."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    async def _download_hf_model(self, job: QuantizeJob, model_id: str) -> Path:
        """Download model from HuggingFace."""
        job.stage = "downloading"
        job.message = f"Downloading {model_id} from HuggingFace..."
        job.progress = 0.0

        local_dir = self.hf_cache_dir / model_id.replace("/", "--")

        try:
            from huggingface_hub import snapshot_download, model_info as hf_model_info
        except ImportError:
            raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub")

        # Use a dedicated thread pool so we don't starve FastAPI's default executor
        from concurrent.futures import ThreadPoolExecutor
        _dl_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-dl")

        # Get expected total size from HuggingFace API
        expected_size = 0
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(_dl_executor, lambda: hf_model_info(model_id))
            ignore_ext = {".msgpack", ".h5", ".ot", ".onnx"}
            for sib in (info.siblings or []):
                if sib.rfilename and not any(sib.rfilename.endswith(ext) for ext in ignore_ext):
                    expected_size += (sib.size or 0)
        except Exception:
            expected_size = 0

        # Monitor download progress in background
        download_done = asyncio.Event()

        async def _monitor_progress():
            while not download_done.is_set():
                current_size = await asyncio.get_event_loop().run_in_executor(
                    _dl_executor, self._get_dir_size, local_dir
                )
                gb_done = current_size / (1024 ** 3)
                if expected_size > 0:
                    pct = min(current_size / expected_size, 1.0)
                    gb_total = expected_size / (1024 ** 3)
                    job.progress = round(pct * 0.3, 3)
                    job.message = f"Downloading {model_id}: {gb_done:.1f} / {gb_total:.1f} GB"
                else:
                    job.message = f"Downloading {model_id}: {gb_done:.1f} GB downloaded..."
                try:
                    await asyncio.wait_for(download_done.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        monitor_task = asyncio.create_task(_monitor_progress())

        def _download_full():
            return snapshot_download(
                model_id,
                local_dir=str(local_dir),
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
            )

        loop = asyncio.get_event_loop()
        try:
            result_path = await loop.run_in_executor(_dl_executor, _download_full)
            download_done.set()
            await monitor_task
            job.message = f"Downloaded {model_id}"
            job.progress = 0.3
            return Path(result_path)
        except Exception as e:
            download_done.set()
            await monitor_task
            raise RuntimeError(f"Failed to download {model_id}: {e}") from e
        finally:
            _dl_executor.shutdown(wait=False)

    async def _convert_to_gguf(self, job: QuantizeJob, model_path: Path, out_type: str = "f16") -> Path:
        """Convert HuggingFace model to GGUF format."""
        job.stage = "converting"
        job.message = f"Converting to GGUF ({out_type})... This may take a while."
        job.progress = 0.35

        output_file = self.output_dir / f"{job.output_name}-{out_type}.gguf"

        if not self._convert_script.exists():
            raise RuntimeError("convert_hf_to_gguf.py not found. Run tool setup first.")

        cmd = [
            sys.executable,
            str(self._convert_script),
            str(model_path),
            "--outfile", str(output_file),
            "--outtype", out_type,
        ]

        logger.info("Running GGUF conversion: %s", " ".join(cmd))

        if _USE_THREAD_SUBPROCESS:
            result_path = await self._run_subprocess_threaded(job, cmd, "converting", 0.35, 0.6)
        else:
            result_path = await self._run_subprocess_async(job, cmd, "converting", 0.35, 0.6)

        if not output_file.exists():
            raise RuntimeError(f"GGUF conversion failed — output file not created: {output_file}")

        job.message = "Conversion to GGUF complete"
        job.progress = 0.6
        return output_file

    async def _quantize_gguf(self, job: QuantizeJob, input_gguf: Path, quant_type: str) -> Path:
        """Quantize a GGUF file using llama-quantize."""
        job.stage = "quantizing"
        job.message = f"Quantizing to {quant_type}..."
        job.progress = 0.6

        output_file = self.output_dir / f"{job.output_name}-{quant_type.lower()}.gguf"

        if not self._quantize_bin.exists():
            raise RuntimeError("llama-quantize not found. Run tool setup first.")

        cmd = [str(self._quantize_bin), str(input_gguf), str(output_file), quant_type]

        logger.info("Running quantization: %s", " ".join(cmd))

        if _USE_THREAD_SUBPROCESS:
            await self._run_subprocess_threaded(job, cmd, "quantizing", 0.6, 0.95, parse_quantize_progress=True)
        else:
            await self._run_subprocess_async(job, cmd, "quantizing", 0.6, 0.95, parse_quantize_progress=True)

        if not output_file.exists():
            raise RuntimeError(f"Quantization failed — output file not created: {output_file}")

        job.message = f"Quantization to {quant_type} complete"
        job.progress = 0.95
        return output_file

    def _find_ollama_binary(self) -> str:
        """Find the Ollama binary — checks PATH, embedded, and common install locations."""
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            return ollama_bin

        import platform
        # Check embedded binary managed by OllamaManager
        if platform.system() == "Windows":
            embedded = settings.ollama_binary_dir / "ollama.exe"
        else:
            embedded = settings.ollama_binary_dir / "ollama"
        if embedded.exists():
            return str(embedded)

        # Check common Windows install locations
        if platform.system() == "Windows":
            candidates = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)

        raise RuntimeError(
            "Ollama binary not found. Make sure Ollama is running "
            "or go to Settings → Runtime to set up the embedded Ollama."
        )

    async def _import_to_ollama(self, job: QuantizeJob, gguf_path: Path, model_name: str):
        """Import quantized model into Ollama."""
        job.stage = "importing"
        job.message = f"Importing {model_name} to Ollama..."
        job.progress = 0.95

        # Create Modelfile
        modelfile_path = self.output_dir / f"Modelfile_{model_name}"
        abs_gguf = str(gguf_path.resolve()).replace("\\", "/")
        modelfile_path.write_text(f'FROM "{abs_gguf}"\n', encoding="utf-8")

        ollama_bin = self._find_ollama_binary()

        cmd = [ollama_bin, "create", model_name, "-f", str(modelfile_path)]
        logger.info("Running Ollama import: %s", " ".join(cmd))

        if _USE_THREAD_SUBPROCESS:
            await self._run_subprocess_threaded(job, cmd, "importing", 0.95, 0.99)
        else:
            await self._run_subprocess_async(job, cmd, "importing", 0.95, 0.99)

        job.message = f"Imported as '{model_name}' in Ollama"
        job.progress = 1.0

        # Clean up Modelfile
        try:
            modelfile_path.unlink()
        except Exception:
            pass

    async def import_gguf_to_ollama(self, gguf_path: str, model_name: str) -> dict:
        """Import a local GGUF file directly into Ollama (no quantization)."""
        p = Path(gguf_path)
        if not p.exists():
            raise ValueError(f"File not found: {gguf_path}")
        if p.suffix.lower() != ".gguf":
            raise ValueError("File must be a .gguf file")
        if not model_name.strip():
            raise ValueError("Model name is required")

        model_name = model_name.strip()

        # Create Modelfile
        self.output_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = self.output_dir / f"Modelfile_{model_name}"
        abs_gguf = str(p.resolve()).replace("\\", "/")
        modelfile_path.write_text(f'FROM "{abs_gguf}"\n', encoding="utf-8")

        ollama_bin = self._find_ollama_binary()

        cmd = [ollama_bin, "create", model_name, "-f", str(modelfile_path)]
        logger.info("Quick import GGUF to Ollama: %s", " ".join(cmd))

        loop = asyncio.get_event_loop()

        def _run():
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            raw_output, _ = proc.communicate(timeout=1800)
            output = raw_output.decode("utf-8", errors="replace") if raw_output else ""
            return proc.returncode, output

        returncode, output = await loop.run_in_executor(None, _run)

        # Clean up Modelfile
        try:
            modelfile_path.unlink()
        except Exception:
            pass

        if returncode != 0:
            raise RuntimeError(f"Ollama import failed (exit {returncode}): {output[-500:] if output else 'no output'}")

        return {
            "model_name": model_name,
            "gguf_path": str(p.resolve()),
            "size": p.stat().st_size,
        }

    async def _run_subprocess_threaded(
        self, job: QuantizeJob, cmd: list[str], stage: str,
        progress_start: float, progress_end: float,
        parse_quantize_progress: bool = False,
    ) -> str:
        """Run subprocess in a thread (Windows-compatible)."""
        loop = asyncio.get_event_loop()

        def _run():
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self._active_processes[job.id] = proc
            output_lines = []
            progress_re = re.compile(r'\[\s*(\d+)/\s*(\d+)\]')

            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                output_lines.append(line)

                if parse_quantize_progress:
                    m = progress_re.search(line)
                    if m:
                        current, total = int(m.group(1)), int(m.group(2))
                        if total > 0:
                            ratio = current / total
                            job.progress = progress_start + ratio * (progress_end - progress_start)
                            job.message = f"Quantizing... {current}/{total} tensors ({int(ratio*100)}%)"

                if "error" in line.lower() and "warning" not in line.lower():
                    job.message = line[:200]

            proc.wait()
            return proc.returncode, "\n".join(output_lines[-20:])

        returncode, output = await loop.run_in_executor(None, _run)
        self._active_processes.pop(job.id, None)

        if returncode != 0:
            raise RuntimeError(f"{stage} failed (exit code {returncode}): {output[-500:]}")
        return output

    async def _run_subprocess_async(
        self, job: QuantizeJob, cmd: list[str], stage: str,
        progress_start: float, progress_end: float,
        parse_quantize_progress: bool = False,
    ) -> str:
        """Run subprocess asynchronously (non-Windows)."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        output_lines = []
        progress_re = re.compile(r'\[\s*(\d+)/\s*(\d+)\]')

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            output_lines.append(line)

            if parse_quantize_progress:
                m = progress_re.search(line)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    if total > 0:
                        ratio = current / total
                        job.progress = progress_start + ratio * (progress_end - progress_start)
                        job.message = f"Quantizing... {current}/{total} tensors ({int(ratio*100)}%)"

        await proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"{stage} failed (exit code {proc.returncode}): {chr(10).join(output_lines[-20:])}")
        return "\n".join(output_lines[-20:])

    # ── Job queries ───────────────────────────────────────

    def get_job(self, job_id: str) -> QuantizeJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    async def cancel_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return

        proc = self._active_processes.get(job_id)
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass

        task = self._active_tasks.get(job_id)
        if task:
            task.cancel()

        job.status = "cancelled"
        job.message = "Cancelled by user"

    def delete_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        if job.status == "running":
            return
        # Optionally clean up output file
        if job.output_path:
            try:
                Path(job.output_path).unlink(missing_ok=True)
            except Exception:
                pass
        del self._jobs[job_id]


# Singleton
quantize_manager = QuantizeManager()
