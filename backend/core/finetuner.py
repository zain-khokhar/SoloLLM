"""
LoRA Fine-Tuning Engine for SoloLLM.

Uses standard HuggingFace transformers + PEFT for LoRA training.
Auto-detects CUDA: uses native-precision LoRA on GPU, full-precision LoRA on CPU.
Runs training as an isolated subprocess to manage memory.
Saves merged model and registers with Ollama.
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

from core.config import settings
from core.hardware import get_cpu_ram_gb, get_gpu_memory_gb
from core.training import TrainingConfig, TrainingProgress, TrainingStatus
from storage import database as db

logger = logging.getLogger(__name__)

# On Windows, asyncio subprocess often fails with NotImplementedError when the event
# loop is SelectorEventLoop (e.g. under uvicorn). Use synchronous subprocess in a thread instead.
_USE_THREAD_SUBPROCESS = sys.platform == "win32"

# Mapping from Ollama model names to standard HuggingFace model IDs
OLLAMA_TO_HF_MAP = {
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "phi3:mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Approximate VRAM (GB) needed for native-precision GPU LoRA training.
MODEL_VRAM_GPU_GB = {
    "llama3.2:1b": 2.5,
    "llama3.2:3b": 5.5,
    "llama3.1:8b": 12.0,
    "llama3:8b": 12.0,
    "mistral:7b": 10.0,
    "gemma2:2b": 3.5,
    "gemma2:9b": 16.0,
    "phi3:mini": 8.0,
    "qwen2.5:0.5b": 1.5,
    "qwen2.5:1.5b": 3.0,
    "qwen2.5:3b": 5.5,
    "qwen2.5:7b": 12.0,
    "tinyllama": 2.5,
}

# Approximate system RAM (GB) needed for full-precision (fp32) CPU training.
MODEL_RAM_CPU_GB = {
    "llama3.2:1b": 6,
    "llama3.2:3b": 14,
    "llama3.1:8b": 34,
    "llama3:8b": 34,
    "mistral:7b": 30,
    "gemma2:2b": 10,
    "gemma2:9b": 40,
    "phi3:mini": 20,
    "qwen2.5:0.5b": 4,
    "qwen2.5:1.5b": 8,
    "qwen2.5:3b": 14,
    "qwen2.5:7b": 34,
    "tinyllama": 6,
}


class FineTuner:
    """LoRA fine-tuning engine using HuggingFace + PEFT."""

    def __init__(self):
        self.output_dir = Path(settings.data_dir) / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress = TrainingProgress()
        self._process: asyncio.subprocess.Process | None = None
        self._proc_popen: subprocess.Popen | None = None  # used on Windows for cancel
        self._training_task: asyncio.Task | None = None
        self._last_script_error: str = ""

    @property
    def progress(self) -> TrainingProgress:
        return self._progress

    @property
    def is_training(self) -> bool:
        return self._progress.status not in (TrainingStatus.IDLE, TrainingStatus.COMPLETE, TrainingStatus.ERROR)

    def resolve_base_model(self, ollama_model: str) -> str | None:
        """Resolve an Ollama model name to a HuggingFace model ID."""
        if ollama_model in OLLAMA_TO_HF_MAP:
            return OLLAMA_TO_HF_MAP[ollama_model]
        base = ollama_model.split(":")[0]
        for key, val in OLLAMA_TO_HF_MAP.items():
            if key.startswith(base):
                return val
        return None

    def _get_model_memory_requirements(self, ollama_model: str) -> tuple[float, float]:
        """Return (vram_gpu_gb, ram_cpu_gb) for the model. Estimates from param count for unknown models."""
        vram = MODEL_VRAM_GPU_GB.get(ollama_model)
        ram = MODEL_RAM_CPU_GB.get(ollama_model)
        if vram is not None and ram is not None:
            return (vram, ram)
        # Try prefix match
        base = ollama_model.split(":")[0]
        for key, v in MODEL_VRAM_GPU_GB.items():
            if key.startswith(base):
                return (v, MODEL_RAM_CPU_GB.get(key, 16.0))
        # Dynamic fallback: parse parameter count from tag (e.g. "0.5b", "1.5b", "7b")
        tag = ollama_model.split(":")[-1] if ":" in ollama_model else ""
        m = re.match(r"(\d+(?:\.\d+)?)\s*[bB]", tag)
        if m:
            params_b = float(m.group(1))
            estimated_vram = round(params_b * 1.5 + 0.75, 1)
            estimated_ram = round(params_b * 4 + 2, 0)
            logger.info("Estimated memory for %s (%.1fB params): VRAM=%.1f GB, RAM=%.0f GB",
                        ollama_model, params_b, estimated_vram, estimated_ram)
            return (estimated_vram, estimated_ram)
        return (6.0, 16.0)

    def get_training_capabilities(self, ollama_model: str) -> dict:
        """
        Return capabilities for the given model: GPU/CPU availability, required resources,
        and recommended device. No hybrid: either full GPU or full CPU.
        """
        gpu_gb = get_gpu_memory_gb()
        cpu_gb = get_cpu_ram_gb()
        vram_need, ram_need = self._get_model_memory_requirements(ollama_model)

        can_gpu = gpu_gb is not None and gpu_gb >= vram_need
        can_cpu = cpu_gb >= ram_need

        if can_gpu:
            recommended = "gpu"
            message = f"Training will use GPU. Requires {vram_need} GB VRAM; you have {gpu_gb} GB."
        elif can_cpu:
            recommended = "cpu"
            message = f"Not enough GPU VRAM for this model. Training will use CPU. Requires {ram_need} GB RAM; you have {cpu_gb} GB."
        else:
            recommended = "cpu"
            message = (
                f"GPU has insufficient VRAM ({gpu_gb} GB available, {vram_need} GB needed). "
                f"CPU may be tight ({cpu_gb} GB available, {ram_need} GB recommended). Training will attempt CPU."
            )

        return {
            "gpu_available": gpu_gb is not None,
            "gpu_memory_gb": gpu_gb,
            "cpu_ram_gb": cpu_gb,
            "recommended_device": recommended,
            "required_gpu_memory_gb": vram_need,
            "required_cpu_ram_gb": ram_need,
            "can_train_on_gpu": can_gpu,
            "can_train_on_cpu": can_cpu,
            "message": message,
        }

    async def start_training(
        self,
        training_data: list[dict],
        config: TrainingConfig,
        validation_data: list[dict] | None = None,
    ):
        """Start training as a background asyncio task."""
        if self.is_training:
            raise RuntimeError("Training is already in progress")

        self._progress = TrainingProgress(
            status=TrainingStatus.PREPARING,
            message="Preparing training data...",
        )
        self._training_task = asyncio.create_task(
            self._run_training(training_data, config, validation_data=validation_data)
        )

    async def _run_training(
        self,
        training_data: list[dict],
        config: TrainingConfig,
        validation_data: list[dict] | None = None,
    ):
        """Run the full training pipeline."""
        try:
            self._last_script_error = ""
            # 1. Save training data as JSONL
            data_path = self.output_dir / "train_data.jsonl"
            with open(data_path, "w", encoding="utf-8") as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(training_data)} training examples to {data_path}")

            val_path = self.output_dir / "val_data.jsonl"
            if validation_data:
                with open(val_path, "w", encoding="utf-8") as f:
                    for item in validation_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info("Saved %d validation examples to %s", len(validation_data), val_path)
            else:
                val_path = None

            # 2. Resolve HF model
            hf_model = self.resolve_base_model(config.ollama_model_name)
            if not hf_model:
                hf_model = config.base_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            self._progress.status = TrainingStatus.DOWNLOADING_BASE
            self._progress.message = f"Loading base model: {hf_model}"

            # 3. Decide device: full GPU or full CPU. No hybrid (causes meta-tensor errors).
            caps = self.get_training_capabilities(config.ollama_model_name)
            train_on_gpu = caps["recommended_device"] == "gpu"
            self._progress.device = "gpu" if train_on_gpu else "cpu"
            logger.info(
                "Training device: %s (GPU available=%s, can_train_on_gpu=%s)",
                "GPU" if train_on_gpu else "CPU",
                caps["gpu_available"],
                caps["can_train_on_gpu"],
            )

            # 4. Write standalone training script
            script_path = self.output_dir / "train_script.py"
            model_output_dir = self.output_dir / "output"

            # Clear old checkpoints to avoid shape mismatch when switching models
            if model_output_dir.exists():
                logger.info("Clearing old training output directory: %s", model_output_dir)
                shutil.rmtree(model_output_dir)
            model_output_dir.mkdir(parents=True, exist_ok=True)

            self._write_training_script(
                path=script_path,
                config=config,
                hf_model=hf_model,
                data_path=str(data_path),
                val_path=str(val_path) if val_path else "",
                output_dir=str(model_output_dir),
                train_on_gpu=train_on_gpu,
            )

            # 5. Run as subprocess (thread-based on Windows to avoid asyncio NotImplementedError)
            self._progress.status = TrainingStatus.TRAINING
            self._progress.message = "Training in progress..."

            if _USE_THREAD_SUBPROCESS:
                returncode, stderr_text = await self._run_training_subprocess_thread(script_path)
            else:
                returncode, stderr_text = await self._run_training_subprocess_asyncio(script_path)

            if returncode != 0:
                error_msg = self._last_script_error or self._extract_relevant_error(stderr_text)
                if not error_msg:
                    error_msg = "Process exited with non-zero code (no stderr). Check backend logs."
                raise RuntimeError(f"Training script failed:\n{error_msg}")

            # 7. Quality gate + register with Ollama
            effective_val_loss = self._progress.best_val_loss or self._progress.val_loss
            if effective_val_loss and effective_val_loss > config.quality_loss_threshold:
                raise RuntimeError(
                    "Training quality gate failed. "
                    f"best_val_loss={effective_val_loss:.4f} exceeds threshold={config.quality_loss_threshold:.4f}."
                )
            self._progress.quality_passed = True

            self._progress.status = TrainingStatus.REGISTERING
            self._progress.message = "Registering model with Ollama..."
            await self._register_with_ollama(config, str(model_output_dir))

            # Save model info to database
            await db.save_finetuned_model(
                name=config.output_name,
                display_name=config.output_name.replace("-", " ").title(),
                base_model=config.ollama_model_name,
                base_model_hf=hf_model,
                model_path=str(model_output_dir),
                training_examples=len(training_data),
                final_loss=effective_val_loss,
                is_registered=True,
            )

            # Done
            self._progress.status = TrainingStatus.COMPLETE
            self._progress.message = f"Training complete! Model: {config.output_name}"
            logger.info(f"Training complete: {config.output_name}")

        except asyncio.CancelledError:
            self._progress.status = TrainingStatus.IDLE
            self._progress.message = "Training cancelled"
            logger.info("Training cancelled by user")
        except Exception as e:
            self._progress.status = TrainingStatus.ERROR
            err_str = str(e).strip() or "Unknown error (check backend logs)"
            self._progress.error = err_str
            self._progress.message = f"Error: {err_str[:200]}"
            logger.exception("Training failed: %s", err_str)

    def _apply_progress_line(self, text: str) -> None:
        """Parse a stdout line and update progress (PROGRESS: / RESULT:)."""
        if text.startswith("PROGRESS:"):
            try:
                data = json.loads(text[9:])
                self._progress.current_step = data.get("step", 0)
                self._progress.total_steps = data.get("total_steps", 0)
                self._progress.loss = data.get("loss", 0.0)
                self._progress.val_loss = data.get("val_loss", self._progress.val_loss)
                self._progress.best_val_loss = data.get("best_val_loss", self._progress.best_val_loss)
                self._progress.epoch = data.get("epoch", 0.0)
                self._progress.learning_rate = data.get("lr", 0.0)
                self._progress.message = data.get("message", "Training...")
                if data.get("status"):
                    try:
                        self._progress.status = TrainingStatus(data["status"])
                    except ValueError:
                        pass
            except json.JSONDecodeError:
                pass
        elif text.startswith("ERROR:"):
            try:
                payload = json.loads(text[6:])
                msg = (payload.get("message") or "").strip()
                tb = (payload.get("traceback") or "").strip()
                self._last_script_error = f"{msg}\n{tb}".strip() if tb else msg
            except json.JSONDecodeError:
                self._last_script_error = text[6:].strip()
        elif text.startswith("RESULT:"):
            try:
                result = json.loads(text[7:])
                self._progress.val_loss = result.get("final_eval_loss", self._progress.val_loss)
                self._progress.best_val_loss = result.get("best_eval_loss", self._progress.best_val_loss)
            except json.JSONDecodeError:
                pass
        else:
            logger.debug("[train] %s", text)

    def _extract_relevant_error(self, stderr_text: str) -> str:
        """Extract a useful traceback or final error line from stderr text."""
        text = (stderr_text or "").strip()
        if not text:
            return ""

        trace_idx = text.rfind("Traceback (most recent call last):")
        if trace_idx >= 0:
            return text[trace_idx:].strip()[-4000:]

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        filtered = [ln for ln in lines if "|" not in ln or "%" not in ln]
        if filtered:
            return "\n".join(filtered[-20:])[-4000:]
        return "\n".join(lines[-20:])[-4000:]

    async def _run_training_subprocess_asyncio(self, script_path: Path) -> tuple[int, str]:
        """Run training script via asyncio subprocess (Unix)."""
        self._process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_chunks: list[bytes] = []

        async def read_stderr():
            if self._process and self._process.stderr:
                while True:
                    chunk = await self._process.stderr.read(8192)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(read_stderr())
        try:
            if self._process and self._process.stdout:
                async for line in self._process.stdout:
                    text = line.decode("utf-8", errors="replace").strip()
                    self._apply_progress_line(text)
        finally:
            if self._process:
                await self._process.wait()
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        return (self._process.returncode or -1), stderr_text

    async def _run_training_subprocess_thread(self, script_path: Path) -> tuple[int, str]:
        """Run training script in a thread using subprocess.Popen (Windows-safe)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._proc_popen = proc
            stderr_list: list[bytes] = []

            def read_stderr() -> None:
                if proc.stderr:
                    while True:
                        chunk = proc.stderr.read(8192)
                        if not chunk:
                            break
                        stderr_list.append(chunk)

            err_thread = threading.Thread(target=read_stderr, daemon=True)
            err_thread.start()
            try:
                if proc.stdout:
                    for line in iter(proc.stdout.readline, b""):
                        if not line:
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, ("line", line))
            finally:
                proc.wait()
                err_thread.join(timeout=2.0)
                stderr_bytes = b"".join(stderr_list)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", proc.returncode or -1, stderr_bytes))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item[0] == "done":
                self._proc_popen = None
                _, returncode, stderr_bytes = item
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                return returncode, stderr_text
            if item[0] == "line":
                text = item[1].decode("utf-8", errors="replace").strip()
                self._apply_progress_line(text)

    def _write_training_script(
        self,
        path: Path,
        config: TrainingConfig,
        hf_model: str,
        data_path: str,
        val_path: str,
        output_dir: str,
        train_on_gpu: bool = True,
    ):
        """Write a standalone Python training script. Either full GPU or full CPU — no hybrid."""
        use_gpu_flag = "True" if train_on_gpu else "False"
        # GPU path: native precision (bf16/fp16), single device (cuda:0). CPU path: fp32, device_map="cpu".
        if train_on_gpu:
            load_model_block = """# GPU: native precision, no quantization
load_kwargs = {
    "trust_remote_code": True,
    "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    "device_map": "cuda:0",
}
model = AutoModelForCausalLM.from_pretrained(hf_model, **load_kwargs)
model.gradient_checkpointing_enable()
"""
        else:
            load_model_block = """# CPU: full precision, no quantization
load_kwargs = {"trust_remote_code": True, "dtype": torch.float32, "device_map": "cpu"}
model = AutoModelForCausalLM.from_pretrained(hf_model, **load_kwargs)
"""
        base_indent = ""
        load_model_indented = "\n".join(base_indent + line for line in load_model_block.strip().split("\n"))

        script = textwrap.dedent(f"""\
            import json, sys, os
            import traceback
            import gc
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            def progress(step=0, total_steps=0, loss=0.0, val_loss=0.0, best_val_loss=0.0, epoch=0.0, lr=0.0, message="", status=""):
                print("PROGRESS:" + json.dumps({{
                    "step": step, "total_steps": total_steps, "loss": loss,
                    "val_loss": val_loss, "best_val_loss": best_val_loss,
                    "epoch": epoch, "lr": lr, "message": message, "status": status,
                }}), flush=True)

            import torch
            hf_model = "{hf_model}"
            TRAIN_ON_GPU = {use_gpu_flag}
            USE_CUDA = TRAIN_ON_GPU and torch.cuda.is_available()
            device_label = "GPU" if USE_CUDA else "CPU"
            progress(message=f"Loading model: {{hf_model}} ({{device_label}})", status="downloading_base_model")

            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from transformers.trainer_callback import TrainerCallback
            from peft import LoraConfig, get_peft_model, TaskType

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model — full GPU or full CPU only
            {load_model_indented}

            progress(message="Applying LoRA adapters...", status="training")

            # Apply LoRA
            lora_config = LoraConfig(
                r={config.lora_rank},
                lora_alpha={config.lora_alpha},
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Load training data
            progress(message="Loading training data...")
            data = []
            with open(r"{data_path}", "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))

            val_data = []
            val_path = r"{val_path}"
            if val_path and os.path.exists(val_path):
                with open(val_path, "r", encoding="utf-8") as f:
                    for line in f:
                        val_data.append(json.loads(line))

            # Format with chat template
            formatted = []
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                formatted.append({{"text": text}})

            formatted_val = []
            for item in val_data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                formatted_val.append({{"text": text}})

            from datasets import Dataset
            dataset = Dataset.from_list(formatted)
            eval_dataset = Dataset.from_list(formatted_val) if formatted_val else None

            from trl import SFTTrainer, SFTConfig

            output_dir = r"{output_dir}"
            os.makedirs(output_dir, exist_ok=True)

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        eval_loss = logs.get("eval_loss", 0.0)
                        progress(
                            step=state.global_step,
                            total_steps=state.max_steps,
                            loss=logs.get("loss", 0.0),
                            val_loss=eval_loss,
                            best_val_loss=state.best_metric or 0.0,
                            epoch=state.epoch or 0.0,
                            lr=logs.get("learning_rate", 0.0),
                            message=f"Step {{state.global_step}}/{{state.max_steps}}",
                            status="training",
                        )

            # Precision flags
            use_fp16 = False
            use_bf16 = False
            if USE_CUDA:
                use_bf16 = torch.cuda.is_bf16_supported()
                use_fp16 = not use_bf16

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                args=SFTConfig(
                    per_device_train_batch_size={config.batch_size},
                    gradient_accumulation_steps={config.gradient_accumulation_steps},
                    warmup_ratio={config.warmup_ratio},
                    num_train_epochs={config.num_epochs},
                    learning_rate={config.learning_rate},
                    fp16=use_fp16,
                    bf16=use_bf16,
                    disable_tqdm=True,
                    logging_steps=1,
                    output_dir=output_dir,
                    eval_strategy="epoch" if eval_dataset is not None else "no",
                    save_strategy="epoch" if eval_dataset is not None else "steps",
                    save_total_limit=3,
                    load_best_model_at_end=True if eval_dataset is not None else False,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    weight_decay={config.weight_decay},
                    lr_scheduler_type="cosine",
                    seed=42,
                    use_cpu=not USE_CUDA,
                    max_length={config.max_seq_length},
                    remove_unused_columns=False,
                ),
                callbacks=[ProgressCallback()],
            )

            progress(message="Starting training...", status="training")
            try:
                trainer.train()

                # Get eval metrics from training (no need to re-evaluate - SFTTrainer already did it during training)
                best_eval_loss = float(trainer.state.best_metric or 0.0)
                print("RESULT:" + json.dumps({{
                    "final_eval_loss": best_eval_loss,
                    "best_eval_loss": best_eval_loss,
                    "eval_examples": len(formatted_val),
                }}), flush=True)

                # Merge and save on CPU to avoid late-stage CUDA OOM/crash.
                progress(message="Merging LoRA and saving model...", status="exporting_gguf")
                if USE_CUDA:
                    model = model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                gc.collect()

                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                progress(message="Model saved!", status="registering_with_ollama")
            except Exception as e:
                print("ERROR:" + json.dumps({{
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }}), flush=True)
                raise
        """)

        path.write_text(script, encoding="utf-8")
        logger.info(f"Wrote training script to {path}")

    async def _register_with_ollama(self, config: TrainingConfig, model_dir: str):
        """Create a Modelfile and register the trained model with Ollama."""
        ollama_bin = shutil.which("ollama")
        embedded_bin = settings.ollama_binary_dir / ("ollama.exe" if sys.platform == "win32" else "ollama")
        if embedded_bin.exists():
            ollama_bin = str(embedded_bin)
        if not ollama_bin:
            raise RuntimeError(
                "Ollama binary not found. Expected PATH entry or embedded binary at "
                f"{embedded_bin}"
            )

        modelfile_path = self.output_dir / "Modelfile"
        modelfile_content = f"""FROM {model_dir}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a fine-tuned version of {config.ollama_model_name}, trained on user conversation data with SoloLLM."
"""
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        if _USE_THREAD_SUBPROCESS:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [ollama_bin, "create", config.output_name, "-f", str(modelfile_path)],
                    capture_output=True,
                    text=True,
                ),
            )
            if result.returncode != 0:
                raise RuntimeError(f"Ollama create failed: {result.stderr or result.stdout or 'unknown'}")
        else:
            process = await asyncio.create_subprocess_exec(
                ollama_bin, "create", config.output_name, "-f", str(modelfile_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                error = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama create failed: {error}")

        logger.info("Registered model '%s' with Ollama", config.output_name)

    async def cancel_training(self):
        """Cancel the current training."""
        if self._proc_popen is not None and self._proc_popen.poll() is None:
            self._proc_popen.terminate()
            self._proc_popen = None
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
        self._progress = TrainingProgress(
            status=TrainingStatus.IDLE,
            message="Training cancelled",
        )


# Singleton
fine_tuner = FineTuner()
