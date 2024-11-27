from jarvis.db.jsonutils import loadjson
from typing import Optional
from atomgpt.inverse_models.loader import FastLanguageModel
from atomgpt.inverse_models.custom_trainer import CustomSFTTrainer
from atomgpt.inverse_models.utils import (
    gen_atoms,
    text2atoms,
    get_crystal_string_t,
    get_figlet,
)
import torch
from peft import PeftModel
from datasets import load_dataset
from functools import partial
from transformers import TrainingArguments
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
from tqdm import tqdm
import pprint
from jarvis.io.vasp.inputs import Poscar
import csv
import os
from pydantic_settings import BaseSettings
import sys
import json
import argparse
import torch.nn as nn
from typing import Literal
import time

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)


# Adapted from https://github.com/unslothai/unsloth
class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    id_prop_path: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"
    prefix: str = "atomgpt_run"
    model_name: str = "knc6/atomgpt_mistral_tc_supercon"
    batch_size: int = 2
    num_epochs: int = 2
    logging_steps: int = 1
    dataset_num_proc: int = 2
    seed_val: int = 3407
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train: Optional[int] = 2
    num_val: Optional[int] = 2
    num_test: Optional[int] = 2
    model_save_path: str = "lora_model_m"
    loss_type: str = "default"
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    property_name: str = "prop"
    output_dir: str = "outputs"
    model_save_path: str = "lora_model_m"
    csv_out: str = "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    chem_info: Literal["none", "formula", "element_list"] = "formula"
    file_format: Literal["poscar", "xyz", "pdb"] = "poscar"
    max_seq_length: int = (
        2048  # Choose any! We auto support RoPE Scaling internally!
    )
    dtype: Optional[str] = (
        None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    )
    load_in_4bit: bool = (
        True  # True  # Use 4bit quantization to reduce memory usage. Can be False.
    )
    instruction: str = "Below is a description of a superconductor material."
    alpaca_prompt: str = (
        "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
    )
    output_prompt: str = (
        " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )


def make_alpaca_json(
    dataset=[],
    jids=[],
    prop="Tc_supercon",
    instruction="",
    include_jid=False,
    chem_info="",
    output_prompt="",
):
    mem = []
    for i in dataset:
        if i[prop] != "na" and i["id"] in jids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            if include_jid:
                info["id"] = i["id"]
            info["instruction"] = instruction
            if chem_info == "none":
                prefix = ""
            elif chem_info == "element_list":
                prefix = (
                    "The chemical elements are "
                    + atoms.composition.search_string
                    + " . "
                )
            elif chem_info == "formula":
                prefix = (
                    "The chemical formula is "
                    + atoms.composition.reduced_formula
                    + " . "
                )

            info["input"] = (
                prefix
                + "The  "
                + prop
                + " is "
                + i[prop]
                + "."
                + output_prompt
            )
            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


def formatting_prompts_func(examples, alpaca_prompt):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = "</s>"
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


def evaluate(
    test_set=[], model="", tokenizer="", csv_out="out.csv", config=""
):
    print("Testing\n", len(test_set))
    f = open(csv_out, "w")
    f.write("id,target,prediction\n")

    for i in tqdm(test_set, total=len(test_set)):
        # try:
        prompt = i["input"]
        print("prompt", prompt)
        gen_mat = gen_atoms(
            prompt=i["input"],
            tokenizer=tokenizer,
            model=model,
            alpaca_prompt=config.alpaca_prompt,
            instruction=config.instruction,
        )
        target_mat = text2atoms("\n" + i["output"])
        print("target_mat", target_mat)
        print("genmat", gen_mat)
        line = (
            i["id"]
            + ","
            + Poscar(target_mat).to_string().replace("\n", "\\n")
            + ","
            + Poscar(gen_mat).to_string().replace("\n", "\\n")
            + "\n"
        )
        f.write(line)
        print()
    # except Exception as exp:
    #    print("Error", exp)
    #    pass
    f.close()


def batch_evaluate(
    test_set=[],
    prompts=[],
    model="",
    tokenizer="",
    csv_out="out.csv",
    config="",
):
    print("Testing\n", len(test_set))
    f = open(csv_out, "w")
    if not prompts:
        target_exists = True
        prompts = [i["input"] for i in test_set]
        ids = [i["id"] for i in test_set]
    else:
        target_exists = False
        ids = ["id-" + str(i) for i in range(len(prompts))]
    inputs = tokenizer(
        [
            config.alpaca_prompt.format(config.instruction, msg, "")
            for msg in prompts
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_seq_length,
        use_cache=True,
        # top_k=50,
        # top_p=0.9,
        # temperature=0.7,
        # num_beams=5
    )
    outputs_decoded = tokenizer.batch_decode(outputs)
    outputs_decoded = [
        output.replace("<unk>", "").split("### Output:")[1]
        for output in outputs_decoded
    ]
    print("outputs_decoded", outputs_decoded)
    f.write("id,target,prediction\n")

    for ii, i in tqdm(enumerate(test_set), total=len(test_set)):
        # try:

        gen_mat = (
            Poscar(text2atoms(outputs_decoded[ii]))
            .to_string()
            .replace("\n", "\\n")
        )
        if target_exists:
            target_mat = (
                Poscar(text2atoms("\n" + i["output"]))
                .to_string()
                .replace("\n", "\\n")
            )
        else:
            target_mat = ""
        print("target_mat", target_mat)
        print("genmat", gen_mat)
        line = ids[ii] + "," + target_mat + "," + gen_mat + "\n"
        f.write(line)
        print()
    # except Exception as exp:
    #    print("Error", exp)
    #    pass
    f.close()


def run_atomgpt_inverse(config_file="config.json"):
    figlet = get_figlet()
    print(figlet)
    t1 = time.time()
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config.dict())
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    f = open(os.path.join(config.model_save_path, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    id_prop_path = config.id_prop_path
    run_path = os.path.dirname(id_prop_path)
    num_train = config.num_train
    num_test = config.num_test
    model_name = config.model_name
    # loss_function = config.loss_function
    # id_prop_path = os.path.join(run_path, id_prop_path)
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    dat = []
    ids = []
    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        ids.append(i[0])
        tmp = [float(j) for j in i[1:]]
        print("tmp", tmp)
        if len(tmp) == 1:
            tmp = str(float(tmp[0]))
        else:
            tmp = "\n".join(map(str, tmp))

        # if ";" in i[1]:
        #    tmp = "\n".join([str(round(float(j), 2)) for j in i[1].split(";")])
        # else:
        #    tmp = str(round(float(i[1]), 3))
        info["prop"] = (
            tmp  # float(i[1])  # [float(j) for j in i[1:]]  # float(i[1]
        )
        pth = os.path.join(run_path, info["id"])
        if config.file_format == "poscar":
            atoms = Atoms.from_poscar(pth)
        elif config.file_format == "xyz":
            atoms = Atoms.from_xyz(pth)
        elif config.file_format == "cif":
            atoms = Atoms.from_cif(pth)
        elif config.file_format == "pdb":
            # not tested well
            atoms = Atoms.from_pdb(pth)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    train_ids = ids[0:num_train]
    test_ids = ids[num_train : num_train + num_test]
    # test_ids = ids[num_train:]

    m_train = make_alpaca_json(
        dataset=dat,
        jids=train_ids,
        prop=config.property_name,
        instruction=config.instruction,
        chem_info=config.chem_info,
        output_prompt=config.output_prompt,
    )
    dumpjson(data=m_train, filename="alpaca_prop_train.json")
    print("Sample:\n", m_train[0])

    m_test = make_alpaca_json(
        dataset=dat,
        jids=test_ids,
        prop="prop",
        include_jid=True,
        instruction=config.instruction,
        chem_info=config.chem_info,
        output_prompt=config.output_prompt,
    )
    dumpjson(data=m_test, filename="alpaca_prop_test.json")

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if not isinstance(model, PeftModel):
        # import sys
        print("Not Peft model")
        # sys.exit()
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

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    dataset = load_dataset(
        "json", data_files="alpaca_prop_train.json", split="train"
    )
    formatting_prompts_func_with_prompt = partial(
        formatting_prompts_func, alpaca_prompt=config.alpaca_prompt
    )

    dataset = dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        loss_type=config.loss_type,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=5,
            overwrite_output_dir=True,
            # max_steps = 60,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            optim=config.optim,
            weight_decay=0.01,
            lr_scheduler_type=config.lr_scheduler_type,  # "linear",
            seed=config.seed_val,
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained(config.model_save_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_save_path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    # batch_evaluate(
    #    prompts=[i["input"] for i in m_test],
    #    model=model,
    #    tokenizer=tokenizer,
    #    csv_out=config.csv_out,
    #    config=config,
    # )
    # t1 = time.time()
    # batch_evaluate(
    #    test_set=m_test,
    #    model=model,
    #    tokenizer=tokenizer,
    #    csv_out=config.csv_out,
    #    config=config,
    # )
    # t2 = time.time()
    # t1a = time.time()
    evaluate(
        test_set=m_test,
        model=model,
        tokenizer=tokenizer,
        csv_out=config.csv_out,
        config=config,
    )
    t2 = time.time()
    print("Time taken:", t2 - t1)


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    args = parser.parse_args(sys.argv[1:])
    run_atomgpt_inverse(config_file=args.config_name)
    #    config_file="config.json"
    # )
