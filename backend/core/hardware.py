"""
Hardware detection for training: GPU VRAM and CPU RAM.
Used to decide whether a model can train on GPU or must use CPU.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_gpu_memory_gb() -> float | None:
    """Return total GPU memory in GB for the first GPU, or None if no GPU / detection failed."""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return round(gpus[0].memoryTotal / 1024, 2)
    except Exception as e:
        logger.debug("GPUtil GPU detection failed: %s", e)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            val = float(result.stdout.strip().split("\n")[0].strip().split()[0])
            return round(val / 1024, 2)
    except Exception as e:
        logger.debug("nvidia-smi GPU detection failed: %s", e)

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 2)
    except Exception as e:
        logger.debug("torch.cuda GPU detection failed: %s", e)

    return None


def get_cpu_ram_gb() -> float:
    """Return total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception as e:
        logger.debug("psutil RAM detection failed: %s", e)
    return 0.0
