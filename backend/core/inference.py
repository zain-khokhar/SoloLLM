import httpx
import json
import logging
from typing import AsyncGenerator

from core.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for the Ollama API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=5)
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        """List locally available models."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size": m.get("size"),
                    "digest": m.get("digest", "")[:12],
                    "modified_at": m.get("modified_at"),
                    "parameter_size": m.get("details", {}).get("parameter_size"),
                    "quantization_level": m.get("details", {}).get("quantization_level"),
                })
            return models

    async def pull_model(self, name: str) -> AsyncGenerator[dict, None]:
        """Pull a model from Ollama registry. Yields progress events."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": name},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    async def show_model(self, name: str) -> dict:
        """Get detailed model information including context window size."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/api/show",
                json={"name": name},
            )
            resp.raise_for_status()
            data = resp.json()
            model_info = data.get("model_info", {})

            context_length = None
            for key, value in model_info.items():
                if "context_length" in key:
                    context_length = value
                    break

            return {
                "name": name,
                "context_length": context_length,
                "details": data.get("details", {}),
                "parameters": data.get("parameters", ""),
            }

    async def delete_model(self, name: str) -> bool:
        """Delete a local model."""
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                "DELETE",
                f"{self.base_url}/api/delete",
                json={"name": name},
                timeout=30,
            )
            return resp.status_code == 200

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream a chat completion from Ollama.

        Yields dicts with keys:
          - For tokens: {"type": "token", "content": "..."}
          - For done:   {"type": "done", "total_tokens": N, "eval_count": N, "done_reason": "..."}
        """
        model = model or settings.default_model
        temperature = temperature if temperature is not None else settings.temperature

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        # Do NOT set num_predict — let the model generate until natural completion

        # ── Max-Power mode: force all layers to GPU, max CPU threads ──
        if settings.max_power_mode:
            from core.max_power_runner import max_power_runner
            gpu_opts = max_power_runner.get_max_power_options()
            payload["options"].update(gpu_opts)
            payload["keep_alive"] = -1
            logger.info(f"[MaxPower] STREAM request → model={model}, num_gpu={gpu_opts['num_gpu']}, "
                        f"num_thread={gpu_opts['num_thread']}, num_batch={gpu_opts['num_batch']}")
        elif settings.gpu_layers >= 0:
            payload["options"]["num_gpu"] = settings.gpu_layers

        full_content = ""
        eval_count = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("done", False):
                        eval_count = data.get("eval_count", 0)
                        yield {
                            "type": "done",
                            "total_tokens": data.get("prompt_eval_count", 0) + eval_count,
                            "eval_count": eval_count,
                            "done_reason": data.get("done_reason", "stop"),
                        }
                    else:
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_content += token
                            yield {"type": "token", "content": token}

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Non-streaming chat completion. Returns the full response."""
        model = model or settings.default_model
        temperature = temperature if temperature is not None else settings.temperature

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        # Do NOT set num_predict — let the model generate until natural completion

        # ── Max-Power mode: force all layers to GPU, max CPU threads ──
        if settings.max_power_mode:
            from core.max_power_runner import max_power_runner
            gpu_opts = max_power_runner.get_max_power_options()
            payload["options"].update(gpu_opts)
            payload["keep_alive"] = -1
            logger.info(f"[MaxPower] NON-STREAM request → model={model}, num_gpu={gpu_opts['num_gpu']}, "
                        f"num_thread={gpu_opts['num_thread']}, num_batch={gpu_opts['num_batch']}")
        elif settings.gpu_layers >= 0:
            payload["options"]["num_gpu"] = settings.gpu_layers

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {
                "content": data.get("message", {}).get("content", ""),
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "done_reason": data.get("done_reason", "stop"),
            }


# Singleton instance
ollama_client = OllamaClient()
