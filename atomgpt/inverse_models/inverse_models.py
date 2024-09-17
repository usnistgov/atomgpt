from jarvis.db.jsonutils import loadjson
from typing import Optional
from atomgpt.inverse_models.loader import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
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
import argparse

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

    id_prop_path: Optional[str] = "id_prop.csv"
    prefix: str = "atomgpt_run"
    model_name: str = "unsloth/mistral-7b-bnb-4bit"
    batch_size: int = 2
    num_epochs: int = 2
    seed_val: int = 42
    num_train: Optional[int] = 2
    num_val: Optional[int] = 2
    num_test: Optional[int] = 2
    model_save_path: str = "lora_model_m"


instruction = "Below is a description of a superconductor material."
# model_save_path = "lora_model_m"

alpaca_prompt1 = (
    '"""\n'
    + instruction
    + '\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Output:\n{}"""'
)
alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    # crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str


def make_alpaca_json(
    dataset=[], jids=[], prop="Tc_supercon", include_jid=False
):
    mem = []
    for i in dataset:
        if i[prop] != "na" and i["id"] in jids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            if include_jid:
                info["id"] = i["id"]
            info["instruction"] = instruction

            info["input"] = (
                "The chemical formula is "
                + atoms.composition.reduced_formula
                + ". The  "
                + prop
                + " is "
                + str(round(i[prop], 3))
                + "."
                + " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


def formatting_prompts_func(examples):
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


def text2atoms(response):
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


def gen_atoms(prompt="", max_new_tokens=512, model="", tokenizer=""):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,
                prompt,  # input
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )
    response = tokenizer.batch_decode(outputs)[0].split("# Output:")[1]
    atoms = None
    try:
        atoms = text2atoms(response)
    except Exception as exp:

        print(exp)
        pass
    return atoms


#######################################


def run_atomgpt_inverse(config_file="config.json"):
    run_path = os.path.abspath(config_file).split("config.json")[0]
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config)
    id_prop_path = config.id_prop_path
    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    id_prop_path = os.path.join(run_path, id_prop_path)
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    dat = []
    ids = []
    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        ids.append(i[0])
        info["prop"] = float(i[1])  # [float(j) for j in i[1:]]  # float(i[1]
        # pth = os.path.join(run_path, info["id"])
        pth = os.path.join(id_prop_path.split("id_prop.csv")[0], info["id"])
        atoms = Atoms.from_poscar(pth)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    train_ids = ids[0:num_train]
    val_ids = (
        ids[-(num_val + num_test) : -num_test]
        if num_test > 0
        else ids[-(num_val + num_test) :]
    )  # noqa:E203
    test_ids = ids[-num_test:] if num_test > 0 else []
    # test_ids = ids[num_train:]

    m_train = make_alpaca_json(dataset=dat, jids=train_ids, prop="prop")
    dumpjson(data=m_train, filename="alpaca_prop_train.json")
    if num_val > 0:
        m_val = make_alpaca_json(
            dataset=dat, jids=val_ids, prop="prop", include_jid=True
        )
        dumpjson(data=m_val, filename="alpaca_prop_val.json")

    m_test = make_alpaca_json(
        dataset=dat, jids=test_ids, prop="prop", include_jid=True
    )
    dumpjson(data=m_test, filename="alpaca_prop_test.json")
    # m_test = make_alpaca_json(dataset=dft_3d, jids=test_ids, prop="Tc_supercon",include_jid=True)
    # dumpjson(data=m_val, filename="alpaca_Tc_supercon_test.json")

    max_seq_length = (
        2048  # Choose any! We auto support RoPE Scaling internally!
    )
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = (
        True  # Use 4bit quantization to reduce memory usage. Can be False.
    )

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
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

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    dataset = load_dataset(
        "json", data_files="alpaca_prop_train.json", split="train"
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
            num_train_epochs=config.num_epochs,
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained(config.model_save_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_save_path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    f = open("AI-AtomGen-prop-dft_3d-test-rmse.csv", "w")
    f.write("id,target,prediction\n")

    for i in tqdm(m_test):
        prompt = i["input"]
        print("prompt", prompt)
        gen_mat = gen_atoms(
            prompt=i["input"], tokenizer=tokenizer, model=model
        )
        target_mat = text2atoms("\n" + i["output"])
        print("target_mat", target_mat)
        print("genmat", gen_mat)
        # print(target_mat.composition.reduced_formula,gen_mat.composition.reduced_formula,target_mat.density,gen_mat.density )
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
    f.close()


def main():
    args = parser.parse_args(sys.argv[1:])
    run_atomgpt_inverse(config_file=args.config_name)


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    #    config_file="config.json"
    # )
    main()
