"""Training API endpoints for SoloLLM self-training system."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from core.training import TrainingDataPreparer, TrainingConfig
from core.finetuner import fine_tuner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training")


class StartTrainingRequest(BaseModel):
    model: str = Field(..., min_length=1, description="Ollama model name to fine-tune")
    output_name: Optional[str] = "solollm-custom"
    conversation_ids: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    source_mode: str = Field(
        default="conversation",
        description="Training source mode: conversation, documents, mixed",
    )
    workspace_id: str = "default"
    lora_rank: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    validation_split: float = 0.1
    quality_loss_threshold: float = 1.8


preparer = TrainingDataPreparer()


@router.post("/start")
async def start_training(request: StartTrainingRequest):
    """Start fine-tuning a model on conversation data."""
    if fine_tuner.is_training:
        raise HTTPException(status_code=409, detail="Training is already in progress")

    # Resolve HF model
    hf_model = fine_tuner.resolve_base_model(request.model)
    if not hf_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not supported for training. "
                   f"Supported: {', '.join(fine_tuner.OLLAMA_TO_HF_MAP.keys()) if hasattr(fine_tuner, 'OLLAMA_TO_HF_MAP') else 'llama3.2, mistral, gemma2, phi3, qwen2.5'}"
        )

    mode = (request.source_mode or "conversation").lower()
    if mode not in {"conversation", "documents", "mixed"}:
        raise HTTPException(status_code=400, detail="source_mode must be one of: conversation, documents, mixed")

    # Extract training data
    examples = await preparer.extract_examples(
        source_mode=mode,
        conversation_ids=request.conversation_ids,
        document_ids=request.document_ids,
        workspace_id=request.workspace_id,
    )
    if len(examples) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data. Found {len(examples)} examples, need at least 10. "
                   "Add more conversations/documents first."
        )

    # Format for training
    formatted_data = preparer.format_for_training(examples)
    train_data, validation_data = preparer.split_train_validation(
        formatted_data,
        validation_split=request.validation_split,
        min_validation_examples=20,
    )

    # Build config
    config = TrainingConfig(
        base_model=hf_model,
        ollama_model_name=request.model,
        output_name=request.output_name or "solollm-custom",
        lora_rank=request.lora_rank,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
        max_seq_length=request.max_seq_length,
        validation_split=request.validation_split,
        quality_loss_threshold=request.quality_loss_threshold,
    )

    # Start training
    await fine_tuner.start_training(train_data, config, validation_data=validation_data)

    documents_used = sorted({
        doc_id
        for ex in examples
        for doc_id in (ex.get("document_ids") or [])
    })

    return {
        "status": "started",
        "examples": len(examples),
        "training_examples": len(train_data),
        "validation_examples": len(validation_data),
        "base_model": hf_model,
        "source_mode": mode,
        "documents_used": documents_used,
    }


@router.get("/status")
async def get_training_status():
    """Get current training progress."""
    p = fine_tuner.progress
    return {
        "status": p.status.value,
        "current_step": p.current_step,
        "total_steps": p.total_steps,
        "loss": p.loss,
        "val_loss": p.val_loss,
        "best_val_loss": p.best_val_loss,
        "quality_passed": p.quality_passed,
        "epoch": p.epoch,
        "device": p.device,
        "message": p.message,
        "error": p.error,
    }


@router.post("/cancel")
async def cancel_training():
    """Cancel the current training."""
    if not fine_tuner.is_training:
        raise HTTPException(status_code=400, detail="No training in progress")
    await fine_tuner.cancel_training()
    return {"status": "cancelled"}


@router.get("/data/preview")
async def preview_training_data(
    conversation_ids: str | None = None,
    document_ids: str | None = None,
    source_mode: str = "conversation",
    workspace_id: str = "default",
):
    """Preview the training data that would be extracted."""
    conv_ids = conversation_ids.split(",") if conversation_ids else None
    doc_ids = document_ids.split(",") if document_ids else None
    mode = (source_mode or "conversation").lower()
    examples = await preparer.extract_examples(
        source_mode=mode,
        conversation_ids=conv_ids,
        document_ids=doc_ids,
        workspace_id=workspace_id,
    )

    documents_used = sorted({
        doc_id
        for ex in examples
        for doc_id in (ex.get("document_ids") or [])
    })

    sequence = [
        {
            "index": idx + 1,
            "source_type": ex.get("source_type", "conversation"),
            "source_name": ex.get("source_name", ""),
            "document_ids": ex.get("document_ids", []),
        }
        for idx, ex in enumerate(examples[:50])
    ]

    return {
        "total_examples": len(examples),
        "preview": examples[:5],
        "conversations_used": len({e["conversation_id"] for e in examples if e.get("conversation_id")}),
        "documents_used": documents_used,
        "source_mode": mode,
        "sequence": sequence,
    }
