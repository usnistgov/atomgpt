from atomgpt.inverse_models.loader import FastLanguageModel
from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from jarvis.db.jsonutils import loadjson, dumpjson
import os
import pprint
from atomgpt.inverse_models.utils import gen_atoms, main_spectra, load_exp_file
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
import time
from jarvis.core.atoms import ase_to_atoms

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer"
    + " Inverse Model Predictor."
)
parser.add_argument(
    "--output_dir",
    default="outputs",
    help="Name of the output directory",
)
parser.add_argument(
    "--pred_csv",
    default="pred_list_inverse.csv",
    help="CSV file for prediction list.",
)
parser.add_argument(
    "--intvl",
    default="0.3",
    help="XRD 2 theta bin",
)
parser.add_argument(
    "--relax",
    default="True",
    help="Relax cell or not",
)


def relax_atoms(
    atoms=None,
    fmax=0.05,
    nsteps=150,
    constant_volume=False,
):
    from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

    calculator = AlignnAtomwiseCalculator(path=default_path(), device="cpu")
    t1 = time.time()
    # if calculator is None:
    #  return atoms
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
    # TODO: Make it work with any other optimizer
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)
    en = ase_atoms.atoms.get_potential_energy()
    final_atoms = ase_to_atoms(ase_atoms.atoms)
    t2 = time.time()
    return final_atoms


def predict(
    output_dir="outputs",
    # config_name="outputs/config.json",
    pred_csv="pred_list_inverse.csv",
    fname="out_inv.json",
    device="cuda",
    intvl=0.3,
    tol=0.1,
    relax=False,
    background_subs=False,
    filename="Q4_K_M.gguf",
    load_in_4bit=False,  # temp_config["load_in_4bit"]
):
    # if not os.path.exists("config_name"):

    #    config_name=os.path.join(output_dir,"config.json")
    config_name = os.path.join(output_dir, "config.json")
    parent = Path(output_dir).parent
    if not os.path.exists(config_name):
        config_name = os.path.join(parent, "config.json")
    print("config used", config_name)
    temp_config = loadjson(config_name)
    print("config used", temp_config)
    temp_config = TrainingPropConfig(**temp_config).dict()
    max_seq_length = temp_config["max_seq_length"]
    model_name = temp_config["model_name"]
    # output_dir = temp_config["output_dir"]
    dtype = temp_config["dtype"]
    load_in_4bit = load_in_4bit  # temp_config["load_in_4bit"]
    adapter = os.path.join(output_dir, "adapter_config.json")

    if os.path.exists(adapter):

        model_name = output_dir  # temp_config["model_name"]
    print("Model used:", model_name)
    pprint.pprint(temp_config)
    model = None
    tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, gguf_file=filename
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, gguf_file=filename
        )
        pass
    atoms_arr = []
    f = open(pred_csv, "r")
    lines = f.read().splitlines()
    f.close()
    mem = []

    for i in lines:
        prompt = i
        if ".dat" in i:
            parent = Path(pred_csv).parent
            fname_csv = os.path.join(parent, i)
            formula, x, y = load_exp_file(
                filename=fname_csv,
                intvl=intvl,
                tol=tol,
                background_subs=background_subs,
            )
            # y[y < 0.1] = 0
            y_new_str = "\n".join(["{0:.2f}".format(x) for x in y])
            formula = str(formula.split("/")[-1].split(".dat")[0])
            # gen_mat = main_spectra(spectra=[[y_new_str,y]],formulas=[formula],model=model,tokenizer=tokenizer,device='cuda')[0]
            prompt = (
                "The chemical formula is "
                + formula
                + " The XRD is "
                + y_new_str
                + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
        print("prompt", prompt.replace("\n", ","))
        gen_mat = gen_atoms(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=temp_config["alpaca_prompt"],
            instruction=temp_config["instruction"],
            device=device,
        )
        print("gen atoms", gen_mat)
        print("gen atoms spacegroup", gen_mat.spacegroup())
        print("intvl", intvl)
        if relax:
            gen_mat = relax_atoms(atoms=gen_mat)
            print("gen atoms relax", gen_mat, gen_mat.spacegroup())
        atoms_arr.append(gen_mat.to_dict())
        info = {}
        info["prompt"] = prompt
        info["atoms"] = gen_mat.to_dict()
        mem.append(info)
    dumpjson(data=mem, filename=fname)


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    args = parser.parse_args(sys.argv[1:])
    predict(
        output_dir=args.output_dir,
        pred_csv=args.pred_csv,
        intvl=float(args.intvl),
        # config_name=args.config_name,
    )
