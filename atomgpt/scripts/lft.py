import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime
import numpy as np

# Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME = "liuhaotian/LLaVA-Instruct-150K"  # An actual multimodal instruction dataset
OUTPUT_DIR = f"./finetuned-{MODEL_ID.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M')}"
USE_WANDB = False  # Set to True if you want to use wandb for logging
WANDB_PROJECT = "llava-finetune"

# Training parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=False,
    report_to="wandb" if USE_WANDB else "none",
    remove_unused_columns=False,
    fp16=True,
)

# Initialize wandb if enabled
if USE_WANDB:
    wandb.init(project=WANDB_PROJECT)

# Load dataset
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
print(f"Dataset loaded: {dataset}")

# Let's create a small validation split if there isn't one
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    dataset = {"train": dataset["train"], "validation": dataset["test"]}

# Setup QLoRA configuration for memory-efficient fine-tuning
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model and processor
print(f"Loading model and processor: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare the model for QLoRA training
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Preprocess function for the dataset
def preprocess_function(examples):
    # The llava_instruct_150k dataset uses a specific format:
    # Each example has 'image', 'conversations' with alternating user/assistant messages
    inputs = []
    labels = []

    for i in range(len(examples["image"])):
        # Extract the conversation
        conversation = examples["conversations"][i]

        # In this dataset, the first message is usually from the user with image
        user_prompt = conversation[0]["value"]
        assistant_response = (
            conversation[1]["value"] if len(conversation) > 1 else ""
        )

        # Format for LLaVA instruction
        prompt = f"USER: <image>\n{user_prompt}\nASSISTANT:"

        inputs.append({"image": examples["image"][i], "text": prompt})
        labels.append(assistant_response)

    # Process inputs with the processor
    processed_inputs = processor(
        text=[inp["text"] for inp in inputs],
        images=[inp["image"] for inp in inputs],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Create labels for causal language modeling
    processed_labels = processor.tokenizer(
        labels,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).input_ids

    # Replace padding token id with -100 so they are ignored in loss computation
    processed_labels = [
        [
            -100 if token == processor.tokenizer.pad_token_id else token
            for token in label
        ]
        for label in processed_labels
    ]

    processed_inputs["labels"] = processed_labels
    return processed_inputs


# Tokenize and prepare datasets - use a smaller subset for faster processing if needed
print("Preprocessing dataset...")
# For demonstration purposes, let's use a smaller subset
if len(dataset["train"]) > 5000:
    dataset["train"] = dataset["train"].select(range(5000))
if len(dataset["validation"]) > 500:
    dataset["validation"] = dataset["validation"].select(range(500))

tokenized_dataset = {}
for split in dataset:
    tokenized_dataset[split] = dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset[split].column_names,
        num_proc=4,
        desc=f"Processing {split}",
    )


# Create data collator that handles the pixel values and other inputs
def custom_data_collator(examples):
    batch = {}
    for key in examples[0].keys():
        if key == "pixel_values":
            batch[key] = torch.stack([example[key] for example in examples])
        else:
            batch[key] = torch.tensor([example[key] for example in examples])
    return batch


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=custom_data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)


# Test the model with a sample image
def test_finetuned_model(image_path, prompt):
    from PIL import Image
    from transformers import pipeline

    # Load fine-tuned model
    pipe = pipeline(
        "image-to-text",
        model=OUTPUT_DIR,
        model_kwargs={"quantization_config": quantization_config},
    )

    # Load test image
    image = Image.open(image_path)

    # Generate response
    outputs = pipe(
        image, prompt=prompt, generate_kwargs={"max_new_tokens": 200}
    )
    return outputs[0]["generated_text"]


# Example usage of the fine-tuned model
if os.path.exists("test_image.jpg"):
    test_prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"
    result = test_finetuned_model("test_image.jpg", test_prompt)
    print(f"Fine-tuned model output:\n{result}")

print(f"Fine-tuning completed. Model saved to {OUTPUT_DIR}")
