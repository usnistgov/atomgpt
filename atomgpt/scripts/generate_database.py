from jarvis.core.specie import atomic_numbers_to_symbols
import numpy as np
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.composition import Composition
from tqdm import tqdm
import os
from models import load_model, get_model, get_trainer, save_model
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
from utils import eval_prompts
from sample_funcs import parse_fn
from transformers import TextStreamer
import time
import random
import numpy as np
import random

fourbit_model = "knc6/mistral-7b-bnb-4bit"
path = os.path.join("./2_models_mbj_bandgap", fourbit_model.split("/")[1])
print("Models Name", fourbit_model, path)

llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(model_name=path)
FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

Z = np.arange(100) + 1
els = atomic_numbers_to_symbols(Z)
m = 1
n = 2

import pandas as pd

data = {
    "prompt": [],
    "response": [],
    "formula": [],
    "expected_prop_val": [],
    "after_alignn_prop_val": [],
    "gen_material_str": [],
    "gen_material_cif": [],
}
df = pd.DataFrame(data)
elements = [
    "Si",
    "O",
    "C",
    "N",
    "Al",
    "Ga",
    "Cd",
    "Te",
    "Ge",
    "Se",
    "S",
    "As",
    "B",
    "Zn",
    "Cu",
    "P",
    "Pb",
    "Sn",
    "Mo",
    "In",
    "Ag",
]


def gen_binary_samples(llm_model, llm_tokenizer, element=elements):
    index = 0
    for m in np.arange(1, 4):
        for n in np.arange(1, 4):
            for i in tqdm(els):
                for z in elements:
                    try:
                        comp = Composition.from_dict({i: m, z: n})
                        mbj_value = round(random.uniform(2.5, 5), 3)
                        prompt_example = (
                            "Below is a description of a superconductor material. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate atomic structure description with lattice lengths, angles, coordinates and atom types.\n\n### Input:\nThe chemical formula is "
                            + comp.reduced_formula
                            + ". The  mbj_bandgap value is "
                            + str(mbj_value)
                            + ".\n\n### Response:\n"
                        )
                        batch = llm_tokenizer(
                            prompt_example, return_tensors="pt"
                        )
                        batch = {k: v.cuda() for k, v in batch.items()}
                        print(comp.reduced_formula, str(mbj_value))
                        generate_ids = llm_model.generate(
                            **batch,
                            do_sample=False,
                            max_new_tokens=4096,
                            pad_token_id=llm_tokenizer.eos_token_id,
                            use_cache=True,
                        )

                        gen_strs = llm_tokenizer.batch_decode(
                            generate_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        try:
                            material_str = gen_strs[0].replace(
                                prompt_example, ""
                            )
                            cif_str = parse_fn(material_str)
                        except Exception as e:
                            cif_str = None
                            print(index, material_str, cif_str, e)
                        try:
                            df.loc[index, "prompt"] = prompt_example
                            df.loc[index, "response"] = gen_strs
                            df.loc[index, "formula"] = comp.reduced_formula
                            df.loc[index, "expected_prop_val"] = mbj_value
                            df.loc[index, "gen_material_str"] = material_str
                            df.loc[index, "gen_material_cif"] = cif_str
                            index += 1
                        except Exception as e:
                            print(index, e)
                        print(index, comp.reduced_formula, mbj_value)
                        df.to_csv(
                            f"./2_gen_mbj_bandgap/material_generated_samples.csv",
                            index=False,
                        )
                    except:
                        pass


def gen_ternary_samples(llm_model, llm_tokenizer, element=elements):
    index = 0
    for m in np.arange(1, 3):
        for n in np.arange(1, 3):
            for t in np.arange(1, 3):
                for p in elements:
                    for i in tqdm(elements):
                        for z in elements:
                            try:
                                comp = Composition.from_dict(
                                    {i: m, p: t, z: n}
                                )
                                mbj_value = round(random.uniform(2.5, 5), 3)
                                prompt_example = (
                                    "Below is a description of a superconductor material. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate atomic structure description with lattice lengths, angles, coordinates and atom types.\n\n### Input:\nThe chemical formula is "
                                    + comp.reduced_formula
                                    + ". The  mbj_bandgap value is "
                                    + str(mbj_value)
                                    + ".\n\n### Response:\n"
                                )
                                batch = llm_tokenizer(
                                    prompt_example, return_tensors="pt"
                                )
                                batch = {k: v.cuda() for k, v in batch.items()}
                                generate_ids = llm_model.generate(
                                    **batch,
                                    do_sample=False,
                                    max_new_tokens=4096,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    use_cache=True,
                                )

                                gen_strs = llm_tokenizer.batch_decode(
                                    generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True,
                                )
                                try:
                                    material_str = gen_strs[0].replace(
                                        prompt_example, ""
                                    )
                                    cif_str = parse_fn(material_str)
                                except Exception as e:
                                    cif_str = None
                                    print(index, material_str, cif_str, e)
                                try:
                                    df.loc[index, "prompt"] = prompt_example
                                    df.loc[index, "response"] = gen_strs
                                    df.loc[index, "formula"] = (
                                        comp.reduced_formula
                                    )
                                    df.loc[index, "expected_prop_val"] = (
                                        mbj_value
                                    )
                                    df.loc[index, "gen_material_str"] = (
                                        material_str
                                    )
                                    df.loc[index, "gen_material_cif"] = cif_str
                                    index += 1
                                except Exception as e:
                                    print(index, e)
                                print(index, comp.reduced_formula, mbj_value)
                                df.to_csv(
                                    f"./2_gen_mbj_bandgap/material_generated_samples_ternary_onlyelementlist1.csv",
                                    index=False,
                                )
                            except:
                                pass


gen_ternary_samples(llm_model, llm_tokenizer)
print("Finished ...")
