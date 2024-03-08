"""Module for fin tuning LLM model for materials chemsitry."""

from jarvis.db.figshare import data
import transformers
import torch
from jarvis.db.jsonutils import dumpjson
import sys
import argparse
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from jarvis.core.atoms import Atoms
import pandas as pd
from sklearn.metrics import mean_absolute_error

# from tqdm import tqdm
import time
import json
import zipfile

os.environ["WANDB_ANONYMOUS"] = "must"
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
torch.use_deterministic_algorithms(True)
#torch.set_default_dtype(torch.float16)
IGNORE_INDEX = -100
torch.cuda.empty_cache()


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
parser = argparse.ArgumentParser(description="AtomGPT")
parser.add_argument(
    "--benchmark_file",
    default="AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip",
    # default="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae",
    help="Benchmarks available in jarvis_leaderboard/benchmarks/*/*.zip",
)




def get_crystal_string_1225(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        +" ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) +" "+" ".join(["{0:.2f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )
    #extra=atoms.composition.reduced_formula
    #crystal_str+=" "+extra
    return crystal_str

def get_crystal_string(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        +" ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) +" "+" ".join(["{0:.3f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )
    extra=atoms.composition.reduced_formula
    crystal_str+="\n"+extra
    return crystal_str


# Define a custom dataset class for regression
class AtomGPTDataset(Dataset):
    def __init__(
        self, texts=[], targets=[], ids=[], tokenizer="", max_length=128
    ):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not ids:
            ids = ["text-" + str(i) for i in range(len(texts))]
        self.ids = ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return (
            inputs,
            self.ids[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# Example usage


def run_atomgpt(
    prefix="ss",
    model_name="gpt2",
    benchmark_file="AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae.csv.zip",
    root_dir="/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard",
    batch_size=8,
    max_length=512,
    num_epochs=500,
    latent_dim=1024,
    learning_rate=1e-3,
    # learning_rate=1e-3,
    test_each_run=True,
    #learning_rate=5e-5,
    pretrained_path="",
):
    # Load pre-trained tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2")

    # dft_3d = data("dft_3d")
    # root_dir="/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard"
    # benchmark_file="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv.zip"
    # benchmark_file = "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae.csv.zip"
    print("benchmark_file", benchmark_file)
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
    val_ids = list(bench["val"].keys())
    test_ids = list(bench["test"].keys())

    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    # model_name = "gpt2"
    if "t5" in model_name:
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            #load_in_8bit=False,
            #torch_dtype=torch.float16,
            #load_in_8bit=True,
            # device_map="auto"
        )
    # device = model.device
    if "t5" in model_name:
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # batch_size = 16
    # max_length = 128
    # num_epochs = 100
    # learning_rate = 5e-5
    criterion = torch.nn.L1Loss()
    # Define example regression data (texts and corresponding numeric targets)
    train_texts = []
    train_targets = []
    train_ids_temp = []
    val_texts = []
    val_targets = []
    val_ids_temp = []
    test_texts = []
    test_targets = []
    test_ids_temp = []

    for i in dft_3d:
        if i[prop] != "na":
            atoms = Atoms.from_dict(i["atoms"])
            tmp = get_crystal_string(atoms)
            if i[id_tag] in train_ids:
                train_texts.append(tmp)
                train_targets.append(i[prop])
                train_ids_temp.append(i[id_tag])
            elif i[id_tag] in val_ids:
                val_texts.append(tmp)
                val_targets.append(i[prop])
                val_ids_temp.append(i[id_tag])
            elif i[id_tag] in test_ids:
                test_texts.append(tmp)
                test_targets.append(i[prop])
                test_ids_temp.append(i[id_tag])
    print("test_texts:",test_texts[0])
    """    
    ##############################
    ###Fast test###    
    train_texts = [
        "This is the first example text.",
        "Second example is a bit longer than the first one, but still within the max length.",
        "Third example is the longest among these three examples. It exceeds the max length and will be truncated.",
        "Second example is a bit longer than the first one, but still within the max length.",
    ]
    train_targets = [10.2, 15.5, 20.1, 15.5]  # Example regression targets
    val_texts = test_texts = train_texts
    val_targets = test_targets = train_targets
    train_ids_temp=['a','b','c','d']
    val_ids_temp = test_ids_temp = train_ids_temp
    batch_size = 2
    num_epochs = 3
    
    ##############################
    ##############################
    """

    # Fine-tune the last layer of GPT-2 for regression
    # fine_tune_gpt2_regression(train_texts, train_targets, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.add_special_tokens({"unk_token": "<ATOM>"})
        model.resize_token_embeddings(len(tokenizer))
    model.lm_head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, latent_dim),
        # torch.nn.Linear(latent_dim, latent_dim),
        torch.nn.Linear(latent_dim, latent_dim),
        torch.nn.Linear(latent_dim, 1),
    )  
    if pretrained_path!="":
       model.load_state_dict(torch.load(pretrained_path,map_location=device))     
    # model.lm_head = torch.nn.Sequential(torch.nn.Linear( model.config.hidden_size, 256),torch.nn.SiLU(),torch.nn.Linear( 256, 1) )
    model.to(device)
    if torch.cuda.device_count() > 1:
       device_ids = [d for d in range(torch.cuda.device_count())]
       model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
    # Prepare datasets and dataloaders with data collator
    train_dataset = AtomGPTDataset(
        texts=train_texts,
        targets=train_targets,
        ids=train_ids_temp,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = AtomGPTDataset(
        texts=val_texts,
        targets=val_targets,
        tokenizer=tokenizer,
        ids=val_ids_temp,
        max_length=max_length,
    )
    test_dataset = AtomGPTDataset(
        texts=test_texts,
        targets=test_targets,
        tokenizer=tokenizer,
        ids=test_ids_temp,
        max_length=max_length,
    )

    # val_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    val_dataloader = test_dataloader
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=pct_start,
        pct_start=0.3,
    )
    #scheduler = torch.optim.lr_scheduler.StepLR(
    #        optimizer,
    #        step_size=30,
    #    )
    print("train_data", len(train_texts))
    print("test_data", len(test_texts))
    output_dir = prefix + "_out_" + model_name + "_" + dataset + "_" + prop
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_loss = np.inf
    tot_time_start = time.time()
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()
        t1 = time.time()
        for batch in train_dataloader:
            optimizer.zero_grad()
            train_loss = 0
            # train_result = []
            input_ids = batch[0]["input_ids"].squeeze()  # .squeeze(0)
            # if 't5' in model_name:
            #   decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(device)
            #   decoder_input_ids = model._shift_right(decoder_input_ids)
            #   predictions = (
            #       model(input_ids = input_ids.to(device),decoder_input_ids=decoder_input_ids).logits.squeeze().mean(dim=-1)
            #   )
            # else:
            # predictions = (
            #     model(input_ids.to(device)).logits.squeeze().mean(dim=-1)
            # )
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
            targets = batch[2].squeeze()
            loss = criterion(
                predictions.squeeze(), targets.squeeze().to(device)
            )
            #print('train',predictions,targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # optimizer.zero_grad()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        t2 = time.time()
        train_time = round(t2 - t1, 3)
        model.eval()

        # total_eval_mae_loss = 0
        # predictions_list = []
        # targets_list = []
        val_loss = 0
        t1 = time.time()
        for batch in val_dataloader:
            with torch.no_grad():
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
                targets = batch[2].squeeze()
                # print('val',predictions,targets)
                loss = criterion(
                    predictions.squeeze(), targets.squeeze().to(device)
                )
                val_loss += loss.item()
        saving_tag = ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_name = "best_model.pt"
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, best_model_name),
            )
            # print("Saving model for epoch", epoch)
            saving_tag = " saving model:" + str(epoch)
        val_loss = val_loss / len(val_dataloader)
        t2 = time.time()
        val_time = round(t2 - t1, 3)
        train_history.append(train_loss)
        val_history.append(val_loss)
        history = os.path.join(output_dir, "history.json")

        dumpjson(
            data={"train": train_history, "val": val_history}, filename=history
        )
        mae = ""
        if test_each_run:
            t1_test = time.time()
            model.eval()
            fname = os.path.join(output_dir, "test_results.csv")
            f = open(fname, "w")
            f.write("id,target,predictions\n")
            for batch in test_dataloader:
                with torch.no_grad():
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
                    targets = batch[2].squeeze()
                    if len(ids) == 1:
                        targets = [targets]
                        predictions = [predictions]
                        # ids=[ids]
                    for ii, jj, kk in zip(targets, predictions, ids):
                        # print(kk,ii.cpu().detach().numpy().tolist(),jj.cpu().detach().numpy().tolist())
                        line = (
                            str(kk)
                            + ","
                            + str(round(ii.cpu().detach().numpy().tolist(), 3))
                            + ","
                            + str(round(jj.cpu().detach().numpy().tolist(), 3))
                            + "\n"
                        )
                        # f.write("%s, %6f, %6f\n" % (kk, ii.cpu().detach().numpy().tolist(), jj.cpu().detach().numpy().tolist()))
                        # print(line)
                        f.write(line)
            t2_test = time.time()
            test_time = round(t2_test - t1_test, 3)
            f.close()
            df = pd.read_csv(fname)
            mae = mean_absolute_error(df["target"], df["predictions"])
        if mae == "":
            print(
                "Epoch, train loss, val loss, train_time, val_time",
                epoch,
                train_loss,
                val_loss,
                train_time,
                val_time,
                saving_tag,
            )
        else:
            print(
                "Epoch,  train loss, val loss, test loss, train_time, val_time, test_time",
                epoch,
                train_loss,
                val_loss,
                mae,
                train_time,
                val_time,
                test_time,
                saving_tag,
            )

    model.eval()
    fname = os.path.join(output_dir, "test_results.csv")
    f = open(fname, "w")
    f.write("id,target,predictions\n")
    for batch in test_dataloader:
        with torch.no_grad():
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
                    model(input_ids.to(device)).logits.squeeze().mean(dim=-1)
                )
            ids = batch[1]
            targets = batch[2].squeeze()
            if len(ids) == 1:
                targets = [targets]
                predictions = [predictions]
                # ids=[ids]
            for ii, jj, kk in zip(targets, predictions, ids):
                # print(kk,ii.cpu().detach().numpy().tolist(),jj.cpu().detach().numpy().tolist())
                line = (
                    str(kk)
                    + ","
                    + str(round(ii.cpu().detach().numpy().tolist(), 3))
                    + ","
                    + str(round(jj.cpu().detach().numpy().tolist(), 3))
                    + "\n"
                )
                # f.write("%s, %6f, %6f\n" % (kk, ii.cpu().detach().numpy().tolist(), jj.cpu().detach().numpy().tolist()))
                # print(line)
                f.write(line)
    f.close()
    tot_time_end = time.time()
    tot_time = tot_time_end - tot_time_start
    print("tot_time", tot_time)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    benchmark_file = args.benchmark_file
    model_name="meta-llama/Llama-2-7b-hf",
    model_name="google/flan-t5-base"
    model_name = "facebook/opt-350m"
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name="google/flan-t5-small"
    model_name="google-t5/t5-small"
    model_name="gpt2"
    run_atomgpt(
        model_name=model_name,
        benchmark_file=benchmark_file,
        #num_epochs=300, 
        #pretrained_path="xyz_out_google/flan-t5-small_tinnet_N_ead/best_model.pt",
        #pretrained_path="ss_out_google/flan-t5-small_tinnet_N_ead/best_model.pt",
        prefix="xyz1",
        batch_size=16
    )
