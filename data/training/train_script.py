import json, sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def progress(step=0, total_steps=0, loss=0.0, val_loss=0.0, best_val_loss=0.0, epoch=0.0, lr=0.0, message="", status=""):
    print("PROGRESS:" + json.dumps({
        "step": step, "total_steps": total_steps, "loss": loss,
        "val_loss": val_loss, "best_val_loss": best_val_loss,
        "epoch": epoch, "lr": lr, "message": message, "status": status,
    }), flush=True)

import torch
hf_model = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_ON_GPU = True
USE_CUDA = TRAIN_ON_GPU and torch.cuda.is_available()
device_label = "GPU (4-bit)" if USE_CUDA else "CPU"
progress(message=f"Loading model: {hf_model} ({device_label})", status="downloading_base_model")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model — full GPU (4-bit) or full CPU only
            # GPU: 4-bit QLoRA, all layers on GPU (no CPU offload)
load_kwargs = {"trust_remote_code": True}
try:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    load_kwargs["device_map"] = "cuda:0"
except ImportError:
    load_kwargs["dtype"] = torch.float16
    load_kwargs["device_map"] = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(hf_model, **load_kwargs)
try:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
except ImportError:
    pass
model.gradient_checkpointing_enable()

progress(message="Applying LoRA adapters...", status="training")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
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
with open(r"C:\Users\KLH\Desktop\Solollm\data\training\train_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

val_data = []
val_path = r"C:\Users\KLH\Desktop\Solollm\data\training\val_data.jsonl"
if val_path and os.path.exists(val_path):
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))

# Format with chat template
formatted = []
for item in data:
    text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
    formatted.append({"text": text})

formatted_val = []
for item in val_data:
    text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
    formatted_val.append({"text": text})

from datasets import Dataset
dataset = Dataset.from_list(formatted)
eval_dataset = Dataset.from_list(formatted_val) if formatted_val else None

from trl import SFTTrainer, SFTConfig

output_dir = r"C:\Users\KLH\Desktop\Solollm\data\training\output"
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
                message=f"Step {state.global_step}/{state.max_steps}",
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=2,
        learning_rate=0.0002,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=1,
        output_dir=output_dir,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "steps",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        use_cpu=not USE_CUDA,
        max_length=2048,
        remove_unused_columns=False,
    ),
    callbacks=[ProgressCallback()],
)

progress(message="Starting training...", status="training")
trainer.train()

# Get eval metrics from training (no need to re-evaluate - SFTTrainer already did it during training)
best_eval_loss = float(trainer.state.best_metric or 0.0)
print("RESULT:" + json.dumps({
    "final_eval_loss": best_eval_loss,
    "best_eval_loss": best_eval_loss,
    "eval_examples": len(formatted_val),
}), flush=True)

# Merge LoRA into base model and save
progress(message="Merging LoRA and saving model...", status="exporting_gguf")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
progress(message="Model saved!", status="registering_with_ollama")
