"""
Training Data Pipeline for SoloLLM Self-Training.

Extracts conversation data from the database and formats it
for fine-tuning with SFTTrainer (instruction-tuning format).
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from storage.database import get_messages, list_conversations

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    DOWNLOADING_BASE = "downloading_base"
    TRAINING = "training"
    EXPORTING_GGUF = "exporting_gguf"
    REGISTERING = "registering"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TrainingConfig:
    base_model: str = ""
    ollama_model_name: str = ""
    output_name: str = "solollm-custom"
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    quantization_bits: int = 4
    gguf_quantization: str = "q4_k_m"


@dataclass
class TrainingProgress:
    status: TrainingStatus = TrainingStatus.IDLE
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    epoch: float = 0.0
    learning_rate: float = 0.0
    message: str = ""
    error: str = ""


class TrainingDataPreparer:
    """Extracts and formats conversation data for fine-tuning."""

    async def extract_from_conversations(
        self,
        conversation_ids: Optional[list[str]] = None,
        min_quality_score: float = 0.5,
    ) -> list[dict]:
        """
        Extract user/assistant pairs from conversations.

        Quality filter:
        - Skip assistant responses shorter than 20 chars
        - Skip user messages shorter than 5 chars

        Builds context from up to 2 prior messages.

        Returns list of dicts with keys:
            instruction, input, output, conversation_id
        """
        examples = []

        if conversation_ids:
            conv_ids = conversation_ids
        else:
            convos = await list_conversations(limit=500)
            conv_ids = [c["id"] for c in convos]

        for conv_id in conv_ids:
            try:
                messages = await get_messages(conv_id)
            except Exception as e:
                logger.warning(f"Failed to load messages for conversation {conv_id}: {e}")
                continue

            # Filter out system messages
            msgs = [m for m in messages if m["role"] in ("user", "assistant")]

            # Build pairs: each user message followed by an assistant response
            for i in range(len(msgs) - 1):
                user_msg = msgs[i]
                assistant_msg = msgs[i + 1]

                if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                    continue

                # Quality filters
                if len(user_msg["content"].strip()) < 5:
                    continue
                if len(assistant_msg["content"].strip()) < 20:
                    continue

                # Build context from up to 2 prior messages
                context_parts = []
                start = max(0, i - 2)
                for j in range(start, i):
                    prior = msgs[j]
                    context_parts.append(f"{prior['role'].capitalize()}: {prior['content']}")
                context = "\n".join(context_parts)

                examples.append({
                    "instruction": user_msg["content"],
                    "input": context,
                    "output": assistant_msg["content"],
                    "conversation_id": conv_id,
                })

        logger.info(f"Extracted {len(examples)} training examples from {len(conv_ids)} conversations")
        return examples

    def format_for_training(
        self,
        examples: list[dict],
        chat_template: str = "llama-3",
    ) -> list[dict]:
        """
        Format examples into {"messages": [...]} format used by SFTTrainer.

        Each example becomes a messages list with system, user, and assistant turns.
        """
        formatted = []

        for ex in examples:
            messages = []

            # System message with context if available
            if ex.get("input"):
                messages.append({
                    "role": "system",
                    "content": f"You are a helpful assistant. Previous context:\n{ex['input']}",
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant.",
                })

            messages.append({
                "role": "user",
                "content": ex["instruction"],
            })

            messages.append({
                "role": "assistant",
                "content": ex["output"],
            })

            formatted.append({"messages": messages})

        logger.info(f"Formatted {len(formatted)} examples for {chat_template} template")
        return formatted
