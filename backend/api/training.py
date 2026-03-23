"""Training API endpoints for SoloLLM self-training system."""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from core.training import TrainingDataPreparer, TrainingConfig
from core.finetuner import fine_tuner
from storage import database as db

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


@router.get("/capabilities")
async def get_training_capabilities(model: str):
    """Get training capabilities for a specific model."""
    return fine_tuner.get_training_capabilities(model)


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


# ── Fine-tuned Models Management ────────────────────────────


@router.get("/models")
async def list_finetuned_models():
    """List all fine-tuned SoloLLM models."""
    models = await db.list_finetuned_models()
    return {"models": models}


@router.get("/models/{model_name}")
async def get_finetuned_model(model_name: str):
    """Get details of a specific fine-tuned model."""
    model = await db.get_finetuned_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return {"model": model}


class RegisterModelRequest(BaseModel):
    name: str = Field(..., description="Name of the fine-tuned model to register")


class ImportAndRegisterModelRequest(BaseModel):
    name: str = Field(..., description="Model name to register in Ollama")
    model_path: str = Field(..., description="Path to exported HF model directory")
    base_model: str = Field(default="", description="Original Ollama base model name")
    base_model_hf: str = Field(default="", description="Original HuggingFace base model id")
    display_name: Optional[str] = None
    training_examples: int = 0
    final_loss: Optional[float] = None


@router.post("/models/register")
async def register_finetuned_model(request: RegisterModelRequest):
    """Register a fine-tuned model with Ollama."""
    model = await db.get_finetuned_model(request.name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found")

    if model.get("is_registered"):
        return {"status": "already_registered", "model": request.name}

    # Register with Ollama
    try:
        from core.training import TrainingConfig
        config = TrainingConfig(
            output_name=request.name,
            base_model=model.get("base_model_hf", ""),
            ollama_model_name=model.get("base_model", ""),
        )
        await fine_tuner._register_with_ollama(config, model["model_path"])
        await db.update_finetuned_model_registration(request.name, True)
        return {"status": "registered", "model": request.name}
    except Exception as e:
        logger.exception("Failed to register model with Ollama")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/import-register")
async def import_and_register_finetuned_model(request: ImportAndRegisterModelRequest):
    """Import an existing exported model directory, register it with Ollama, and save it in DB."""
    path = Path(request.model_path)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {request.model_path}")

    if not (path / "config.json").exists():
        raise HTTPException(
            status_code=400,
            detail="Model path does not look like a merged HF model directory (missing config.json)",
        )

    try:
        config = TrainingConfig(
            output_name=request.name,
            base_model=request.base_model_hf,
            ollama_model_name=request.base_model,
        )
        await fine_tuner._register_with_ollama(config, str(path))
        await db.save_finetuned_model(
            name=request.name,
            display_name=request.display_name or request.name.replace("-", " ").title(),
            base_model=request.base_model,
            base_model_hf=request.base_model_hf,
            model_path=str(path),
            training_examples=request.training_examples,
            final_loss=request.final_loss,
            is_registered=True,
        )
        return {"status": "registered", "model": request.name}
    except Exception as e:
        logger.exception("Failed to import/register fine-tuned model")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unregister")
async def unregister_finetuned_model(request: RegisterModelRequest):
    """Unregister a fine-tuned model from Ollama (delete from Ollama)."""
    model = await db.get_finetuned_model(request.name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found")

    if not model.get("is_registered"):
        return {"status": "not_registered", "model": request.name}

    # Delete from Ollama
    try:
        from core.inference import ollama_client
        success = await ollama_client.delete_model(request.name)
        if success:
            await db.update_finetuned_model_registration(request.name, False)
            return {"status": "unregistered", "model": request.name}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model from Ollama")
    except Exception as e:
        logger.exception("Failed to unregister model from Ollama")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_finetuned_model(model_name: str):
    """Delete a fine-tuned model completely (Ollama + database + files)."""
    import shutil
    from pathlib import Path

    model = await db.get_finetuned_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Unregister from Ollama if registered
    if model.get("is_registered"):
        try:
            from core.inference import ollama_client
            await ollama_client.delete_model(model_name)
        except Exception as e:
            logger.warning("Failed to delete model from Ollama: %s", e)

    # Delete model files
    model_path = model.get("model_path")
    if model_path:
        path = Path(model_path)
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.warning("Failed to delete model files: %s", e)

    # Delete from database
    await db.delete_finetuned_model(model_name)

    return {"status": "deleted", "model": model_name}
