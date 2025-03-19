from jarvis.io.vasp.inputs import Poscar
import os
import sys
from jarvis.db.jsonutils import loadjson
import csv
from jarvis.core.atoms import Atoms
import transformers, torch, os, json, zipfile
from tqdm import tqdm
from atomgpt.forward_models.forward_models import AtomGPTDataset
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer"
    + " Forward Model Predictor."
)
parser.add_argument(
    "--output_dir",
    default="out",
    help="Name of the output directory",
)
parser.add_argument(
    "--pred_csv",
    default="pred_list.csv",
    help="CSV file for prediction list.",
)


def predict(
    output_dir="out", pred_csv="pred_list.csv", fname="out.csv", device="cuda"
):
    temp_config = loadjson(os.path.join(output_dir, "config.json"))
    max_length = temp_config["max_length"]
    model_name = temp_config["model_name"]
    output_dir = temp_config["output_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "t5" in model_name:
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.lm_head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, temp_config["latent_dim"]),
        torch.nn.Linear(temp_config["latent_dim"], 1),
    )
    model.load_state_dict(
        torch.load(output_dir + "/best_model.pt", map_location=device)
    )
    model.to(device)
    with open(pred_csv, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    dat = []
    test_texts = []
    test_targets = []
    test_ids_temp = []

    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        parent = Path(pred_csv).parent
        # info["prop"] = [float(j) for j in i[1:]]  # float(i[1])
        pth = os.path.join(
            os.path.abspath(parent), i[0]
        )  # os.path.join(run_path, info["id"])
        # print('path',pth)
        atoms = Atoms.from_poscar(pth)
        desc = atoms.describe()[temp_config["desc_type"]]
        info["desc"] = desc
        info["prop"] = ""
        dat.append(info)
        test_texts.append(desc)
        test_targets.append(-999999)
        test_ids_temp.append(i[0])

    test_dataset = AtomGPTDataset(
        texts=test_texts,
        targets=test_targets,
        tokenizer=tokenizer,
        ids=test_ids_temp,
        max_length=max_length,
    )
    batch_size = temp_config["batch_size"]
    if len(test_dataset) < temp_config["batch_size"]:
        batch_size = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # Lets load the model first

    f = open(fname, "w")
    f.write("id,predictions\n")
    for batch in test_dataloader:
        input_ids = batch[0]["input_ids"].squeeze()  # .squeeze(0)
        if "t5" in model_name:
            predictions = (
                model(
                    input_ids.to(device),
                    decoder_input_ids=input_ids.to(device),
                )
                .logits.squeeze()
                .mean(dim=-1)
            )

        else:
            predictions = (
                model(
                    input_ids.to(device),
                )
                .logits.squeeze()
                .mean(dim=-1)
            )
        ids = batch[1]
        predictions = predictions.cpu().detach().numpy().tolist()
        for j, k in zip(ids, predictions):
            line = str(j) + "," + str(k) + "\n"
            f.write(line)
    f.close()


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    args = parser.parse_args(sys.argv[1:])
    predict(output_dir=args.output_dir, pred_csv=args.pred_csv)
    #    config_file="config.json"
    # )

# pred()
