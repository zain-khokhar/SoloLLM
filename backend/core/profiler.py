import platform
import psutil
import subprocess
import logging

logger = logging.getLogger(__name__)


def _get_gpu_info() -> tuple[str | None, int | None]:
    """Detect GPU name and VRAM in MB."""
    # Try GPUtil first
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return gpu.name, int(gpu.memoryTotal)
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram = int(float(parts[1].strip()))
            return name, vram
    except Exception:
        pass

    return None, None


def _get_cpu_info() -> tuple[str, int]:
    """Get CPU name and core count."""
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = psutil.cpu_count(logical=True) or 0

    # Try to get a better CPU name on Windows
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and l.strip() != "Name"]
                if lines:
                    cpu_name = lines[0]
        except Exception:
            pass

    return cpu_name, cpu_cores


def profile_system() -> dict:
    """Profile the current system hardware."""
    gpu_name, vram_mb = _get_gpu_info()
    cpu_name, cpu_cores = _get_cpu_info()
    ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    profile = {
        "gpu_name": gpu_name,
        "vram_mb": vram_mb,
        "ram_mb": ram_mb,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "os_info": os_info,
    }

    profile["recommended_models"] = _recommend_models(vram_mb, ram_mb)
    logger.info(f"System profile: GPU={gpu_name} ({vram_mb}MB), RAM={ram_mb}MB, CPU={cpu_name} ({cpu_cores} cores)")
    return profile


def _recommend_models(vram_mb: int | None, ram_mb: int) -> list[str]:
    """Recommend Ollama models based on hardware."""
    models = []

    # VRAM-based recommendations
    if vram_mb and vram_mb >= 8000:
        models.extend(["llama3.1:8b", "mistral:7b", "codellama:13b", "llama3.2:latest"])
    elif vram_mb and vram_mb >= 4000:
        models.extend(["llama3.2:latest", "mistral:7b", "phi3:mini", "gemma2:2b"])
    elif vram_mb and vram_mb >= 2000:
        models.extend(["llama3.2:1b", "phi3:mini", "gemma2:2b", "tinyllama:latest"])
    else:
        # CPU-only based on RAM
        if ram_mb >= 16000:
            models.extend(["llama3.2:latest", "mistral:7b", "phi3:mini"])
        elif ram_mb >= 8000:
            models.extend(["llama3.2:1b", "phi3:mini", "gemma2:2b", "tinyllama:latest"])
        else:
            models.extend(["tinyllama:latest", "gemma2:2b"])

    return models
