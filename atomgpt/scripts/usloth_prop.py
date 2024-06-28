from sklearn.metrics import mean_absolute_error
import pandas as pd
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import re
import os
import json
import zipfile
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
from collections import defaultdict


def atoms_describer(
    atoms=[],
    xrd_peaks=5,
    xrd_round=1,
    cutoff=4,
    take_n_bomds=2,
    include_spg=True,
):
    """Describe an atomic structure."""
    if include_spg:
        spg = Spacegroup3D(atoms)
    theta, d_hkls, intens = XRD().simulate(atoms=(atoms))
    #     x = atoms.atomwise_angle_and_radial_distribution()
    #     bond_distances = {}
    #     for i, j in x[-1]["different_bond"].items():
    #         bond_distances[i.replace("_", "-")] = ", ".join(
    #             map(str, (sorted(list(set([round(jj, 2) for jj in j])))))
    #         )
    dists = defaultdict(list)
    elements = atoms.elements
    for i in atoms.get_all_neighbors(r=cutoff):
        for j in i:
            key = "-".join(sorted([elements[j[0]], elements[j[1]]]))
            dists[key].append(j[2])
    bond_distances = {}
    for i, j in dists.items():
        dist = sorted(set([round(k, 2) for k in j]))
        if len(dist) >= take_n_bomds:
            dist = dist[0:take_n_bomds]
        bond_distances[i] = ", ".join(map(str, dist))
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    info = {}
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": (fracs),
        "atomic_X": ", ".join(
            map(str, [Specie(s).X for s in atoms.uniq_species])
        ),
        "atomic_Z": ", ".join(
            map(str, [Specie(s).Z for s in atoms.uniq_species])
        ),
    }
    struct_info = {
        "lattice_parameters": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.abc])
        ),
        "lattice_angles": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.angles])
        ),
        # "spg_number": spg.space_group_number,
        # "spg_symbol": spg.space_group_symbol,
        "top_k_xrd_peaks": ", ".join(
            map(
                str,
                sorted(list(set([round(i, xrd_round) for i in theta])))[
                    0:xrd_peaks
                ],
            )
        ),
        "density": round(atoms.density, 3),
        # "crystal_system": spg.crystal_system,
        # "point_group": spg.point_group_symbol,
        # "wyckoff": ", ".join(list(set(spg._dataset["wyckoffs"]))),
        "bond_distances": bond_distances,
        # "natoms_primitive": spg.primitive_atoms.num_atoms,
        # "natoms_conventional": spg.conventional_standard_structure.num_atoms,
    }
    if include_spg:
        struct_info["spg_number"] = spg.space_group_number
        struct_info["spg_symbol"] = spg.space_group_symbol
        struct_info["crystal_system"] = spg.crystal_system
        struct_info["point_group"] = spg.point_group_symbol
        struct_info["wyckoff"] = ", ".join(list(set(spg._dataset["wyckoffs"])))
        struct_info["natoms_primitive"] = spg.primitive_atoms.num_atoms
        struct_info["natoms_conventional"] = (
            spg.conventional_standard_structure.num_atoms
        )
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    line = "The number of atoms are: " + str(
        atoms.num_atoms
    )  # +"., The elements are: "+",".join(atoms.elements)+". "
    for i, j in info.items():
        if not isinstance(j, dict):
            line += "The " + i + " is " + j + ". "
        else:
            # print("i",i)
            # print("j",j)
            for ii, jj in j.items():
                tmp = ""
                if isinstance(jj, dict):
                    for iii, jjj in jj.items():
                        tmp += iii + ": " + str(jjj) + " "
                else:
                    tmp = jj
                line += "The " + ii + " is " + str(tmp) + ". "
    return line


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


def make_alpaca_json_gen(dataset=[], prop="Tc_supercon"):
    alpaca_prompt = """Below is a description of a material..

    ### Instruction:
    {}

    ### Input:
    {}

    ### Output:
    {}"""

    mem = []
    all_ids = []
    for i in dataset:
        if i[prop] != "na":
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["instruction"] = (
                "Below is a description of a superconductor material."
            )
            info["input"] = (
                "The chemical formula is "
                + atoms.composition.reduced_formula
                + " The  "
                + prop
                + " is "
                + str(round(i[prop], 3))
                + ". The spacegroup is "
                + i["spg_number"]
                + "."
                + " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


def make_alpaca_json_pred(
    dataset=[], prop="Tc_supercon", id_tag="jid", ids=[]
):
    alpaca_prompt = """Below is a description of a material..

    ### Instruction:
    {}

    ### Input:
    {}

    ### Output:
    {}"""
    all_ids = []
    mem = []
    for i in dataset:
        if i[prop] != "na" and i[id_tag] in ids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["instruction"] = (
                "Predict " + prop + " property of this material"
            )
            info["input"] = get_crystal_string_t(atoms)
            info["output"] = str(round(i[prop], 2))
            mem.append(info)
            all_ids.append(i[id_tag])
    return alpaca_prompt, mem, all_ids


benchmark_file = (
    "AI-SinglePropertyPrediction-PBE_gap-halide_peroskites-test-mae.csv.zip"
)
root_dir = "/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard"
method = benchmark_file.split("-")[0]
task = benchmark_file.split("-")[1]
prop = benchmark_file.split("-")[2]
dataset = benchmark_file.split("-")[3]
temp = dataset + "_" + prop + ".json.zip"
temp2 = dataset + "_" + prop + ".json"
fname = os.path.join(root_dir, "benchmarks", method, task, temp)
zp = zipfile.ZipFile(fname)
bench = json.loads(zp.read(temp2))
dft_3d = data(dataset)
id_tag = "jid"
if "jid" in dft_3d[0]:
    id_tag = "jid"
else:
    id_tag = "id"

# train_atoms = []
# val_atoms = []
# test_atoms = []
# train_targets = []
# val_targets = []
# test_targets = []
train_ids = list(bench["train"].keys())
test_ids = list(bench["test"].keys())
if "val" in bench:
    val_ids = list(bench["val"].keys())
else:
    val_ids = test_ids
print("total", len(dft_3d))
print("test_ids", len(test_ids))
print("val_ids", len(val_ids))
print("train_ids", len(train_ids))
alpaca_prompt, train_data, train_ids = make_alpaca_json_pred(
    dataset=dft_3d, prop=prop, id_tag=id_tag, ids=train_ids
)
alpaca_prompt, test_data, test_ids = make_alpaca_json_pred(
    dataset=dft_3d, prop=prop, id_tag=id_tag, ids=test_ids
)
dumpjson(data=train_data, filename="train_data.json")
dumpjson(data=test_data, filename="test_data.json")
model_path = "lora_model_train"

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
# nm = fourbit_models[0]
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


dataset = load_dataset("json", data_files="train_data.json", split="train")
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
        num_train_epochs=3,
        report_to="none",
    ),
)
trainer_stats = trainer.train()
model.save_pretrained(model_path)

model_x, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model_x)  # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

f = open("sloth_prop.csv", "w")
f.write("id,target,prediction\n")
for ii, i in enumerate(test_data):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Predict "
                + prop
                + " property of this material",  # instruction
                i["input"],  # input
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = tokenizer.batch_decode(
        model_x.generate(**inputs, max_new_tokens=64, use_cache=True)
    )[0].split("### Output:\n")[-1]
    floats = [float(j) for j in re.findall(r"\b\d+\.\d+\b", outputs)]
    print(test_ids[ii], ",", i["output"], ",", floats[0])
    line = (
        str(test_ids[ii])
        + ","
        + str(i["output"])
        + ","
        + str(floats[0])
        + "\n"
    )
    f.write(line)
    # print(test_ids[ii], ",",i["output"].split("## Output:\\n")[1].split("</s>")[0], ",",tokenizer.batch_decode(outputs))
f.close()
df = pd.read_csv("sloth_prop.csv")
print("mae", mean_absolute_error(df["target"], df["prediction"]))
