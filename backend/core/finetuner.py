"""
QLoRA Fine-Tuning Engine for SoloLLM.

Uses Unsloth for efficient 4-bit LoRA training.
Runs training as an isolated subprocess to manage GPU memory.
Exports trained model as GGUF and registers with Ollama.
"""

import asyncio
import json
import logging
import sys
import textwrap
from pathlib import Path

from core.config import settings
from core.training import TrainingConfig, TrainingProgress, TrainingStatus

logger = logging.getLogger(__name__)

# Mapping from Ollama model names to Unsloth-compatible HuggingFace model IDs
OLLAMA_TO_HF_MAP = {
    "llama3.2:1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3.1:8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3:8b": "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "mistral:7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "gemma2:2b": "unsloth/gemma-2-2b-it-bnb-4bit",
    "gemma2:9b": "unsloth/gemma-2-9b-it-bnb-4bit",
    "phi3:mini": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "qwen2.5:3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "qwen2.5:7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "tinyllama": "unsloth/tinyllama-bnb-4bit",
}


class FineTuner:
    """QLoRA fine-tuning engine using Unsloth."""

    def __init__(self):
        self.output_dir = Path(settings.data_dir) / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress = TrainingProgress()
        self._process: asyncio.subprocess.Process | None = None
        self._training_task: asyncio.Task | None = None

    @property
    def progress(self) -> TrainingProgress:
        return self._progress

    @property
    def is_training(self) -> bool:
        return self._progress.status not in (TrainingStatus.IDLE, TrainingStatus.COMPLETE, TrainingStatus.ERROR)

    def resolve_base_model(self, ollama_model: str) -> str | None:
        """Resolve an Ollama model name to a HuggingFace model ID."""
        # Try exact match
        if ollama_model in OLLAMA_TO_HF_MAP:
            return OLLAMA_TO_HF_MAP[ollama_model]
        # Try base name (without tag)
        base = ollama_model.split(":")[0]
        for key, val in OLLAMA_TO_HF_MAP.items():
            if key.startswith(base):
                return val
        return None

    async def start_training(self, training_data: list[dict], config: TrainingConfig):
        """Start training as a background asyncio task."""
        if self.is_training:
            raise RuntimeError("Training is already in progress")

        self._progress = TrainingProgress(
            status=TrainingStatus.PREPARING,
            message="Preparing training data...",
        )
        self._training_task = asyncio.create_task(self._run_training(training_data, config))

    async def _run_training(self, training_data: list[dict], config: TrainingConfig):
        """Run the full training pipeline."""
        try:
            # 1. Save training data as JSONL
            data_path = self.output_dir / "train_data.jsonl"
            with open(data_path, "w", encoding="utf-8") as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(training_data)} training examples to {data_path}")

            # 2. Resolve HF model
            hf_model = self.resolve_base_model(config.ollama_model_name)
            if not hf_model:
                hf_model = config.base_model or "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

            self._progress.status = TrainingStatus.DOWNLOADING_BASE
            self._progress.message = f"Loading base model: {hf_model}"

            # 3. Write standalone training script
            script_path = self.output_dir / "train_script.py"
            model_output_dir = self.output_dir / "output"
            gguf_path = model_output_dir / f"{config.output_name}.gguf"

            self._write_training_script(
                path=script_path,
                config=config,
                hf_model=hf_model,
                data_path=str(data_path),
                output_dir=str(model_output_dir),
            )

            # 4. Run as subprocess
            self._progress.status = TrainingStatus.TRAINING
            self._progress.message = "Training in progress..."

            self._process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 5. Read progress from stdout
            async for line in self._process.stdout:
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("PROGRESS:"):
                    try:
                        data = json.loads(text[9:])
                        self._progress.current_step = data.get("step", 0)
                        self._progress.total_steps = data.get("total_steps", 0)
                        self._progress.loss = data.get("loss", 0.0)
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
                else:
                    logger.debug(f"[train] {text}")

            await self._process.wait()

            if self._process.returncode != 0:
                stderr = await self._process.stderr.read()
                error_msg = stderr.decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(f"Training script failed: {error_msg}")

            # 6. Register with Ollama
            self._progress.status = TrainingStatus.REGISTERING
            self._progress.message = "Registering model with Ollama..."
            await self._register_with_ollama(config, str(gguf_path))

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
            self._progress.error = str(e)
            self._progress.message = f"Error: {str(e)[:200]}"
            logger.error(f"Training failed: {e}")

    def _write_training_script(
        self,
        path: Path,
        config: TrainingConfig,
        hf_model: str,
        data_path: str,
        output_dir: str,
    ):
        """Write a standalone Python training script (runs in subprocess)."""
        script = textwrap.dedent(f"""\
            import json, sys, os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            def progress(step=0, total_steps=0, loss=0.0, epoch=0.0, lr=0.0, message="", status=""):
                print(f"PROGRESS:" + json.dumps({{
                    "step": step, "total_steps": total_steps, "loss": loss,
                    "epoch": epoch, "lr": lr, "message": message, "status": status,
                }}), flush=True)

            progress(message="Loading model: {hf_model}", status="downloading_base_model")

            from unsloth import FastLanguageModel
            import torch

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="{hf_model}",
                max_seq_length={config.max_seq_length},
                dtype=None,
                load_in_4bit=True,
            )

            progress(message="Applying LoRA adapters...", status="training")

            model = FastLanguageModel.get_peft_model(
                model,
                r={config.lora_rank},
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha={config.lora_alpha},
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )

            # Load training data
            progress(message="Loading training data...")
            data = []
            with open(r"{data_path}", "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))

            # Format with chat template
            formatted = []
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                formatted.append({{"text": text}})

            from datasets import Dataset
            dataset = Dataset.from_list(formatted)

            # Training
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from transformers.trainer_callback import TrainerCallback

            output_dir = r"{output_dir}"
            os.makedirs(output_dir, exist_ok=True)

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        progress(
                            step=state.global_step,
                            total_steps=state.max_steps,
                            loss=logs.get("loss", 0.0),
                            epoch=state.epoch or 0.0,
                            lr=logs.get("learning_rate", 0.0),
                            message=f"Step {{state.global_step}}/{{state.max_steps}}",
                            status="training",
                        )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length={config.max_seq_length},
                dataset_num_proc=2,
                args=TrainingArguments(
                    per_device_train_batch_size={config.batch_size},
                    gradient_accumulation_steps={config.gradient_accumulation_steps},
                    warmup_ratio={config.warmup_ratio},
                    num_train_epochs={config.num_epochs},
                    learning_rate={config.learning_rate},
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=1,
                    output_dir=output_dir,
                    weight_decay={config.weight_decay},
                    lr_scheduler_type="cosine",
                    seed=42,
                    report_to="none",
                ),
                callbacks=[ProgressCallback()],
            )

            progress(message="Starting training...", status="training")
            trainer.train()

            # Export to GGUF
            progress(message="Exporting to GGUF format...", status="exporting_gguf")
            model.save_pretrained_gguf(
                output_dir,
                tokenizer,
                quantization_method="{config.gguf_quantization}",
            )
            progress(message="Export complete!", status="registering_with_ollama")
        """)

        path.write_text(script, encoding="utf-8")
        logger.info(f"Wrote training script to {path}")

    async def _register_with_ollama(self, config: TrainingConfig, gguf_path: str):
        """Create a Modelfile and register the trained model with Ollama."""
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a fine-tuned version of {config.ollama_model_name}, trained on user conversation data with SoloLLM."
"""
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            "ollama", "create", config.output_name, "-f", str(modelfile_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama create failed: {error}")

        logger.info(f"Registered model '{config.output_name}' with Ollama")

    async def cancel_training(self):
        """Cancel the current training."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
        self._progress = TrainingProgress(
            status=TrainingStatus.IDLE,
            message="Training cancelled",
        )


# Singleton
fine_tuner = FineTuner()
