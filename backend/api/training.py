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
    lora_rank: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 2048


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

    # Extract training data
    examples = await preparer.extract_from_conversations(request.conversation_ids)
    if len(examples) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data. Found {len(examples)} examples, need at least 10. "
                   "Have more conversations first."
        )

    # Format for training
    training_data = preparer.format_for_training(examples)

    # Build config
    config = TrainingConfig(
        base_model=hf_model,
        ollama_model_name=request.model,
        output_name=request.output_name or "solollm-custom",
        lora_rank=request.lora_rank,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
        max_seq_length=request.max_seq_length,
    )

    # Start training
    await fine_tuner.start_training(training_data, config)

    return {
        "status": "started",
        "examples": len(examples),
        "base_model": hf_model,
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
        "epoch": p.epoch,
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
async def preview_training_data(conversation_ids: str | None = None):
    """Preview the training data that would be extracted."""
    conv_ids = conversation_ids.split(",") if conversation_ids else None
    examples = await preparer.extract_from_conversations(conv_ids)
    return {
        "total_examples": len(examples),
        "preview": examples[:5],
        "conversations_used": len(set(e["conversation_id"] for e in examples)),
    }
