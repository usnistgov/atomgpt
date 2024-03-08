from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = (
    True  # Use 4bit quantization to reduce memory usage. Can be False.
)

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

nm = "unsloth/mistral-7b-bnb-4bit"
nm = fourbit_models[-2]
nm = fourbit_models[0]
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=nm,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


dataset = load_dataset(
    "json", data_files="alpaca_Tc_supercon.json", split="train"
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        overwrite_output_dir=True,
        # max_steps = 60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        num_train_epochs=10,
        report_to="none",
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("lora_model_m")
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Below is a description of a superconductor material.",  # instruction
            "The chemical formula is YCI The  Tc_supercon is 6.483. The spacegroup is 12. Generate atomic structure description with lattice lengths, angles, coordinates and atom types.",  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
# outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print("xyz", tokenizer.batch_decode(outputs))
