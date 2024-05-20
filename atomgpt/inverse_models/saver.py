from atomgpt.inverse_models.loader import FastLanguageModel
from atomgpt.inverse_models.save import save_to_gguf,unsloth_save_pretrained_gguf
import json
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
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = (
    True  # Use 4bit quantization to reduce memory usage. Can be False.
)
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
model.save_pretrained("unsloth_finetuned_model")
tokenizer.save_pretrained("unsloth_finetuned_model")
#model.save_pretrained_gguf("xyz",tokenizer=tokenizer)
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  #
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "xyz-unsloth.Q8_0.gguf", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto"

)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
print(model)
