"""
Training Data Pipeline for SoloLLM Self-Training.

Extracts conversation data from the database and formats it
for fine-tuning with SFTTrainer (instruction-tuning format).
"""

import logging
import random
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import aiosqlite

from core.config import settings
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
    validation_split: float = 0.1
    quality_loss_threshold: float = 1.8
    min_validation_examples: int = 20


@dataclass
class TrainingProgress:
    status: TrainingStatus = TrainingStatus.IDLE
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    epoch: float = 0.0
    learning_rate: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = 0.0
    quality_passed: bool = False
    device: str = ""
    message: str = ""
    error: str = ""


class TrainingDataPreparer:
    """Extracts and formats conversation data for fine-tuning."""

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned

    def _dedupe_examples(self, examples: list[dict]) -> list[dict]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for ex in examples:
            key = (
                self._clean_text(ex.get("instruction", ""))[:240],
                self._clean_text(ex.get("input", ""))[:240],
                self._clean_text(ex.get("output", ""))[:240],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ex)
        return deduped

    async def extract_from_documents(
        self,
        document_ids: Optional[list[str]] = None,
        workspace_id: str = "default",
        max_chunks_per_document: int = 120,
    ) -> list[dict]:
        """Extract instruction-style examples from uploaded documents/chunks."""
        vectors_db_path = str(settings.data_dir / "db" / "vectors.db")
        examples: list[dict] = []

        db = await aiosqlite.connect(vectors_db_path)
        db.row_factory = aiosqlite.Row
        try:
            doc_params: list[object] = [workspace_id]
            doc_filter = ""
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                doc_filter = f" AND d.id IN ({placeholders})"
                doc_params.extend(document_ids)

            cursor = await db.execute(
                f"""
                SELECT
                    d.id AS document_id,
                    d.filename,
                    d.created_at,
                    dc.content,
                    dc.chunk_index
                FROM documents d
                JOIN document_chunks dc ON dc.document_id = d.id
                WHERE d.workspace_id = ? {doc_filter}
                ORDER BY d.created_at ASC, dc.chunk_index ASC
                """,
                tuple(doc_params),
            )
            rows = await cursor.fetchall()
        finally:
            await db.close()

        chunk_counts: dict[str, int] = {}
        for row in rows:
            document_id = row["document_id"]
            filename = row["filename"]
            chunk_counts.setdefault(document_id, 0)
            if chunk_counts[document_id] >= max_chunks_per_document:
                continue

            excerpt = self._clean_text(row["content"])
            if len(excerpt) < 120 or len(excerpt) > 2400:
                continue
            if excerpt.count(" ") < 12:
                continue

            snippet = excerpt[:900]
            first_sentence = re.split(r"(?<=[.!?])\s+", excerpt, maxsplit=1)[0][:180]

            examples.append(
                {
                    "instruction": f"Summarize the key points from '{filename}'.",
                    "input": f"Document: {filename}\nExcerpt:\n{snippet}",
                    "output": snippet,
                    "conversation_id": "",
                    "source_type": "document",
                    "source_name": filename,
                    "document_ids": [document_id],
                    "sequence_key": f"{row['created_at']}::{row['chunk_index']:06d}",
                }
            )

            # Add synthetic QA-style example for richer instruction tuning.
            examples.append(
                {
                    "instruction": f"What does '{filename}' explain about: {first_sentence}",
                    "input": f"Use only this excerpt from {filename}.",
                    "output": snippet,
                    "conversation_id": "",
                    "source_type": "document",
                    "source_name": filename,
                    "document_ids": [document_id],
                    "sequence_key": f"{row['created_at']}::{row['chunk_index']:06d}::qa",
                }
            )
            chunk_counts[document_id] += 1

        deduped = self._dedupe_examples(examples)
        logger.info("Extracted %d document-based examples", len(deduped))
        return deduped

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

                instruction = self._clean_text(user_msg["content"])
                output = self._clean_text(assistant_msg["content"])
                docs_used = assistant_msg.get("documents_used") or []

                examples.append({
                    "instruction": instruction,
                    "input": context,
                    "output": output,
                    "conversation_id": conv_id,
                    "source_type": "conversation",
                    "source_name": conv_id,
                    "document_ids": docs_used,
                    "sequence_key": assistant_msg.get("created_at", ""),
                })

        deduped = self._dedupe_examples(examples)
        logger.info(f"Extracted {len(deduped)} training examples from {len(conv_ids)} conversations")
        return deduped

    async def extract_examples(
        self,
        source_mode: str = "conversation",
        conversation_ids: Optional[list[str]] = None,
        document_ids: Optional[list[str]] = None,
        workspace_id: str = "default",
    ) -> list[dict]:
        """Extract examples from conversations/documents in deterministic sequence."""
        source_mode = (source_mode or "conversation").lower()
        if source_mode not in {"conversation", "documents", "mixed"}:
            raise ValueError(f"Unsupported training source_mode: {source_mode}")

        examples: list[dict] = []
        if source_mode in {"conversation", "mixed"}:
            conv_examples = await self.extract_from_conversations(conversation_ids=conversation_ids)
            examples.extend(conv_examples)
        if source_mode in {"documents", "mixed"}:
            doc_examples = await self.extract_from_documents(
                document_ids=document_ids,
                workspace_id=workspace_id,
            )
            examples.extend(doc_examples)

        # Stable source-first ordering for predictable training sequence.
        source_priority = {"document": 0, "conversation": 1}
        examples.sort(
            key=lambda ex: (
                source_priority.get(ex.get("source_type", "conversation"), 9),
                ex.get("source_name", ""),
                ex.get("sequence_key", ""),
            )
        )
        return examples

    def split_train_validation(
        self,
        formatted_examples: list[dict],
        validation_split: float,
        min_validation_examples: int,
    ) -> tuple[list[dict], list[dict]]:
        """Deterministic split for validation-driven quality checks."""
        total = len(formatted_examples)
        if total < 20:
            return formatted_examples, []

        split_count = max(min_validation_examples, int(total * validation_split))
        split_count = min(split_count, max(1, total // 4))

        indices = list(range(total))
        random.Random(42).shuffle(indices)
        val_set = set(indices[:split_count])

        train_data: list[dict] = []
        val_data: list[dict] = []
        for idx, item in enumerate(formatted_examples):
            if idx in val_set:
                val_data.append(item)
            else:
                train_data.append(item)
        return train_data, val_data

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
