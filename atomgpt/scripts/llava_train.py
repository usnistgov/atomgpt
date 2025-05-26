import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
)

# Load the dataset
dataset = load_dataset("unsloth/llava-instruct-mix-vsft-mini", split="train")

# Load processor and model with 4-bit quantization
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForVision2Seq.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
    ],  # Adjust based on the model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)


def preprocess(example):
    # Build prompt in LLaVA style: "USER: <image>\nquestion\nASSISTANT:"
    user_turns = [m for m in example["messages"] if m["role"] == "user"]
    assistant_turns = [
        m for m in example["messages"] if m["role"] == "assistant"
    ]

    if not user_turns or not assistant_turns:
        return {}  # Skip malformed examples

    user_text = " ".join([c["text"] for m in user_turns for c in m["content"]])
    assistant_text = " ".join(
        [c["text"] for m in assistant_turns for c in m["content"]]
    )

    prompt = f"USER: <image>\n{user_text}\nASSISTANT:"
    full_text = f"{prompt} {assistant_text}"

    processed = processor(
        images=example["images"][0],  # use first image
        text=full_text,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    processed["labels"] = processed["input_ids"].clone()
    return {k: v.squeeze(0) for k, v in processed.items()}


# Preprocessing function
def preprocess2(example):
    processed = processor(
        text=example["text"],
        images=example["image"],
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    processed["labels"] = processed["input_ids"].clone()
    return {k: v.squeeze(0) for k, v in processed.items()}


# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-llava",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save final model and processor
model.save_pretrained("./finetuned-llava")
processor.save_pretrained("./finetuned-llava")
