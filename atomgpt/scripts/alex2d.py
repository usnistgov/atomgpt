from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from atomgpt.inverse_models.vision_dataset import generate_dataset
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
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerState, TrainerControl
from atomgpt.inverse_models.callbacks import (
    PrintGPUUsageCallback,
    ExampleTrainerCallback,
)
import json
from tqdm import tqdm

# from inference import db_inference
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from jarvis.db.jsonutils import loadjson
from datasets import Dataset

from unsloth import FastVisionModel  # FastLanguageModel for LLMs


# model_name="unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
# model_name="unsloth/Llama-3.2-11B-Vision-Instruct"
# model_name="unsloth/Pixtral-12B-2409"

from transformers import TrainerCallback
from jarvis.core.atoms import Atoms
import numpy as np
from jarvis.core.lattice import Lattice


def text2atoms(response):
    response = response.split("assistant<|end_header_id|>\n")[1].split(
        ". The "
    )[0]
    tmp_atoms_array = response.strip("</s>").split("\n")
    # tmp_atoms_array= [element for element in tmp_atoms_array  if element != '']
    # print("tmp_atoms_array", tmp_atoms_array)
    lat_lengths = np.array(tmp_atoms_array[1].split(), dtype="float")
    lat_angles = np.array(tmp_atoms_array[2].split(), dtype="float")

    lat = Lattice.from_parameters(
        lat_lengths[0],
        lat_lengths[1],
        lat_lengths[2],
        lat_angles[0],
        lat_angles[1],
        lat_angles[2],
    )
    elements = []
    coords = []
    for ii, i in enumerate(tmp_atoms_array):
        if ii > 2 and ii < len(tmp_atoms_array):
            # if ii>2 and ii<len(tmp_atoms_array)-1:
            tmp = i.split()
            elements.append(tmp[0])
            coords.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])

    atoms = Atoms(
        coords=coords,
        elements=elements,
        lattice_mat=lat.lattice(),
        cartesian=False,
    )
    return atoms


class EnhancedTrainingMonitor(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset=None,
        print_frequency=50,
        generate_text_frequency=500,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.print_frequency = print_frequency
        self.generate_text_frequency = (
            generate_text_frequency  # Less frequent text generation
        )
        self.epoch = 0
        self.processor = None  # Will hold the image processor if needed

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch += 1
        print(f"\n===== Beginning Epoch {self.epoch} =====\n")

    def on_step_end(self, args, state, control, **kwargs):
        """Prints training progress and sample information"""
        if state.global_step % self.print_frequency != 0:
            return

        print(f"\n=== Training Progress at Step {state.global_step} ===")

        # Print loss if available
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            loss = last_log.get("loss", "N/A")
            learning_rate = last_log.get("learning_rate", "N/A")
            print(f"Loss: {loss}")
            print(f"Learning rate: {learning_rate}")

        # Sample examination
        if self.eval_dataset and len(self.eval_dataset) > 0:
            sample_idx = state.global_step % len(self.eval_dataset)
            sample = self.eval_dataset[sample_idx]

            # Print sample info
            if isinstance(sample, dict) and "id" in sample:
                print(f"\nSample ID: {sample['id']}")

            # Print input prompt
            input_prompt = None
            if (
                isinstance(sample, dict)
                and "messages" in sample
                and len(sample["messages"]) > 0
            ):
                for content in sample["messages"][0]["content"]:
                    if content["type"] == "text":
                        input_prompt = content["text"]
                        print(f"Input prompt: {input_prompt}")
                        break

            # Print expected output
            expected_output = None
            if (
                isinstance(sample, dict)
                and "messages" in sample
                and len(sample["messages"]) > 1
            ):
                for content in sample["messages"][1]["content"]:
                    if content["type"] == "text":
                        expected_output = content["text"]
                        print(f"Expected output: {expected_output}")
                        break
            model = kwargs["model"]
            # print("model",model)
            model.eval()
            # Generate text periodically (less frequently than progress reports)
            print("\n--- Attempting Text Generation ---")
            print("sample", sample)
            instruction = sample["messages"][0]["content"][0]["text"]
            image = sample["images"][0]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                        # {"type": "text", "text": "The atomic structure is represented as follows:"}
                    ],
                }
            ]
            # print()
            # print("messages", messages)

            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            outputs = model.generate(
                **inputs, max_new_tokens=1554, use_cache=True
            )
            response = self.tokenizer.batch_decode(outputs)[
                0
            ]  # .split("[/INST]")[1]
            # print("response",response)
            try:
                response = text2atoms(response)
                print("Predicted", response)
            except Exception as exp:
                print("exp", exp, "\n", response)
                pass
            # print("----------------------------")


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


def evaluate_and_save(
    model, tokenizer, test_dataset, output_path="test_predictions.json"
):
    results = []
    model.eval()
    print(f"\n🔍 Running evaluation on {len(test_dataset)} samples...")

    for example in tqdm(test_dataset):
        try:
            sample_id = example.get("id", None)
            instruction = None
            image = example["images"][0]

            for content in example["messages"][0]["content"]:
                if content["type"] == "text":
                    instruction = content["text"]

            # Format prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            input_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1554)

            generated = tokenizer.batch_decode(outputs)[0]

            # Target atoms
            target_text = None
            for content in example["messages"][1]["content"]:
                if content["type"] == "text":
                    target_text = content["text"]
                    break

            target_atoms = text2atoms(
                "assistant<|end_header_id|>\n" + target_text
            )
            pred_atoms = text2atoms(generated)

            results.append(
                {
                    "id": sample_id,
                    "target": target_atoms.to_dict(),
                    "predicted": pred_atoms.to_dict(),
                }
            )

        except Exception as e:
            print(f"⚠️ Error in sample {example.get('id', 'unknown')}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"\n✅ Evaluation complete. Results saved to: {output_path}")


def run(
    datasets=["dft_2d"],
    id_tag="jid",
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    output_folder="dft_2d_formula_based",
):
    # def run(datasets=["dft_3d", "dft_2d"], model_name="unsloth/Pixtral-12B-2409"):
    train_datasets = []
    test_datasets = []
    for i in datasets:
        train_dataset, test_dataset = generate_dataset(
            dataset_name=i, output_folder=output_folder, id_tag=id_tag
        )
        train_datasets.extend(train_dataset)
        test_datasets.extend(test_dataset)
    # model, tokenizer = get_model(model_name=model_name)
    model, tokenizer = get_model(model_name=model_name)
    # train_dataset = Dataset.from_list(train_dataset)

    FastVisionModel.for_training(model)  # Enable for training!
    output_dir = output_folder + "_" + "_".join(datasets) + "_" + model_name
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,  # Reduce to 1 to make Pixtral fit!
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # save_strategy="epoch",
            # save_steps=1,
            # max_steps = 30,
            num_train_epochs=3,  # Set this instead of max_steps for full training runs
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            save_steps=100,
            # optim="adamw_torch_fused",
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
            max_seq_length=1248,
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

    # """
    callback = EnhancedTrainingMonitor(
        eval_dataset=test_dataset[0:1],
        # some_tokenized_dataset=tokenized_eval,
        tokenizer=tokenizer,
        print_frequency=1,
        # max_length=1048,
        # callback_samples=2,
    )
    trainer.add_callback(callback)
    # """
    gpu_usage = PrintGPUUsageCallback()
    trainer.add_callback(gpu_usage)

    trainer_stats = trainer.train()
    evaluate_and_save(
        model,
        tokenizer,
        test_datasets,
        output_path=f"{output_dir}/test_predictions.json",
    )

    # print([i[id_tag] for i in test_dataset][0:10])
    # d = loadjson(os.path.join(output_dir, "dft_2d_test_dataset.json"))
    # db_inference(d=d)
    # d = loadjson(os.path.join(output_dir, "dft_3d_test_dataset.json"))
    # db_inference(d=d)


if __name__ == "__main__":
    # run(model_name="formula_output_dir_dft_2d_unsloth/Llama-3.2-11B-Vision-Instruct/checkpoint-620")
    run(
        model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
        datasets=["alex_pbe_2d_all"],
        id_tag="id",
        output_folder="alex2d_formula_based",
    )
    # d = loadjson(os.path.join("formula_based", "dft_2d_test_dataset.json"))
    # db_inference(d=d,model_path="formula_output_dir_dft_3d_dft_2d_unsloth/Pixtral-12B-2409/checkpoint-1240",output_folder="formula_based")
