from atomgpt.inverse_models import FastLanguageModel
from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from jarvis.db.jsonutils import loadjson, dumpjson
import os
import pprint
from atomgpt.inverse_models.utils import gen_atoms, main_spectra, load_exp_file
import argparse
import sys
from pathlib import Path


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


def predict(
    output_dir="outputs",
    pred_csv="pred_list_inverse.csv",
    fname="out_inv.json",
    device="cuda",
):
    temp_config = loadjson(os.path.join(output_dir, "atomgpt_config.json"))
    temp_config = TrainingPropConfig(**temp_config).dict()
    max_seq_length = temp_config["max_seq_length"]
    model_name = temp_config["model_name"]
    output_dir = temp_config["output_dir"]
    dtype = temp_config["dtype"]
    load_in_4bit = temp_config["load_in_4bit"]
    model_name = temp_config["model_name"]
    pprint.pprint(temp_config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    atoms_arr = []
    f = open(pred_csv, "r")
    lines = f.read().splitlines()
    f.close()
    mem = []

    for i in lines:
        prompt = i
        if ".dat" in i:
            parent = Path(pred_csv).parent
            fname = os.path.join(parent, i)
            formula, x, y = load_exp_file(filename=fname, intvl=0.3)
            y[y < 0] = 0
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
        print("prompt", prompt)
        gen_mat = gen_atoms(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=temp_config["alpaca_prompt"],
            instruction=temp_config["instruction"],
            device=device,
        )
        print("gen atoms", gen_mat)
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
    predict(output_dir=args.output_dir, pred_csv=args.pred_csv)
