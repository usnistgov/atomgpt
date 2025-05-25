import torch
import os
from jarvis.db.figshare import data
from tqdm import tqdm
from jarvis.core.atoms import Atoms, get_supercell_dims, crop_square
from jarvis.analysis.stem.convolution_apprx import STEMConv
from jarvis.core.specie import Specie
from PIL import Image
import numpy as np
from jarvis.io.vasp.inputs import Poscar
from jarvis.analysis.defects.surface import Surface
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from vision_dataset import generate_dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
from atomgpt.inverse_models.callbacks import (
    PrintGPUUsageCallback,
    ExampleTrainerCallback,
)

# from inference import db_inference
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from jarvis.db.jsonutils import loadjson

from unsloth import FastVisionModel  # FastLanguageModel for LLMs


# model_name="unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
# model_name="unsloth/Llama-3.2-11B-Vision-Instruct"
# model_name="unsloth/Pixtral-12B-2409"


def get_model(model_name="unsloth/Pixtral-12B-2409"):

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    try:
        model = FastVisionModel.get_peft_model(
            model,
            # We do NOT finetune vision & attention layers since Pixtral uses more memory!
            finetune_vision_layers=False,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=False,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=8,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=8,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

    except:
        pass

    return model, tokenizer


def run(datasets=["dft_2d"], model_name="unsloth/Pixtral-12B-2409"):
    # def run(datasets=["dft_3d", "dft_2d"], model_name="unsloth/Pixtral-12B-2409"):
    train_datasets = []
    test_datasets = []
    for i in datasets:
        train_dataset, test_dataset = generate_dataset(
            dataset_name=i, output_folder="formula_based"
        )
        train_datasets.extend(train_dataset)
        test_datasets.extend(test_dataset)
    # model, tokenizer = get_model(model_name=model_name)
    model, tokenizer = get_model(model_name=model_name)

    FastVisionModel.for_training(model)  # Enable for training!
    output_dir = "formula_output_dir_" + "_".join(datasets) + "_" + model_name
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,  # Reduce to 1 to make Pixtral fit!
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # max_steps = 30,
            num_train_epochs=5,  # Set this instead of max_steps for full training runs
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            save_steps=100,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",  # For Weights and Biases
            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=1048,
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
    )
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    """
    callback = ExampleTrainerCallback(
        some_tokenized_dataset=tokenized_eval,
        # some_tokenized_dataset=tokenized_eval,
        tokenizer=tokenizer,
        max_length=1048,
        callback_samples=2,
    )
    trainer.add_callback(callback)
    """
    gpu_usage = PrintGPUUsageCallback()
    trainer.add_callback(gpu_usage)

    trainer_stats = trainer.train()

    # print([i[id_tag] for i in test_dataset][0:10])
    d = loadjson(os.path.join(output_dir, "dft_2d_test_dataset.json"))
    db_inference(d=d)
    d = loadjson(os.path.join(output_dir, "dft_3d_test_dataset.json"))
    db_inference(d=d)


if __name__ == "__main__":
    run()
    # d = loadjson(os.path.join("formula_based", "dft_2d_test_dataset.json"))
    # db_inference(d=d,model_path="formula_output_dir_dft_3d_dft_2d_unsloth/Pixtral-12B-2409/checkpoint-1240",output_folder="formula_based")
