"""Module for fine tuning LLM model for materials chemsitry."""

from jarvis.db.figshare import data
import transformers
import torch
import random
from jarvis.db.jsonutils import loadjson, dumpjson
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from jarvis.core.atoms import Atoms
import pandas as pd
from sklearn.metrics import mean_absolute_error
import json
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
import pprint
from collections import defaultdict
from tqdm import tqdm
import time
import json
import zipfile
from typing import Optional
from pydantic_settings import BaseSettings
import csv
import pprint
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


class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    id_prop_path: Optional[str] = "robo_desc.json.zip"
    prefix: str = "atomgpt_run"
    model_name: str = "gpt2"
    batch_size: int = 16
    max_length: int = 512
    num_epochs: int = 500
    latent_dim: int = 1024
    learning_rate: float = 1e-3
    test_each_run: bool = True
    include_struct: bool = False
    pretrained_path: str = ""
    seed_val: int = 42
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    output_dir: str = "out_temp"
    desc_type: str = "desc_3"
    convert: bool = False  # raw files for false
    train_ratio: Optional[float] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    keep_data_order: bool = True


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=True,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = (
        ids[-(n_val + n_test) : -n_test]
        if n_test > 0
        else ids[-(n_val + n_test) :]
    )  # noqa:E203
    id_test = ids[-n_test:] if n_test > 0 else []
    return id_train, id_val, id_test


def make_id_prop(
    benchmark_file="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv.zip",
    desc_file="robo_desc.json.zip",
    leaderboard_dir="/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard",
    # leaderboard_dir="/work/03943/kamalch/ls6/Software/atomgpt/jarvis_leaderboard/jarvis_leaderboard/",
    output_dir="test_id_prop",
):
    print("benchmark_file", benchmark_file)
    method = benchmark_file.split("-")[0]
    task = benchmark_file.split("-")[1]
    prop_name = benchmark_file.split("-")[2]
    dataset = benchmark_file.split("-")[3]
    temp = dataset + "_" + prop_name + ".json.zip"
    temp2 = dataset + "_" + prop_name + ".json"
    fname = os.path.join(leaderboard_dir, "benchmarks", method, task, temp)
    zp = zipfile.ZipFile(fname)
    bench = json.loads(zp.read(temp2))
    dft_3d = data(dataset)
    id_tag = "jid"
    output_dir = prop_name + "_" + dataset
    if "jid" in dft_3d[0]:
        id_tag = "jid"
    else:
        id_tag = "id"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_ids = list(bench["train"].keys())
    test_ids = list(bench["test"].keys())
    if "val" in bench:
        val_ids = list(bench["val"].keys())
    else:
        val_ids = test_ids
    print("Saving files in", output_dir)
    if ".zip" in desc_file:
        zp = zipfile.ZipFile(desc_file)
        dat = json.loads(zp.read(desc_file.split(".zip")[0].split("/")[-1]))

    else:
        dat = loadjson(desc_file)

    dat2 = {}
    for i in dat:
        dat2[i["id"]] = i["desc"]
    dft_3d2 = {}
    for i in dft_3d:
        dft_3d2[i[id_tag]] = i
    mem = []
    for i in train_ids:
        desc = dat2[i]
        prop = dft_3d2[i][prop_name]
        info = {}
        info["id"] = i
        info["desc"] = desc
        info["prop"] = prop
        mem.append(info)
    for i in val_ids:
        desc = dat2[i]

        prop = dft_3d2[i][prop_name]
        info = {}
        info["id"] = i
        info["desc"] = desc
        info["prop"] = prop
        mem.append(info)
    for i in test_ids:
        desc = dat2[i]
        prop = dft_3d2[i][prop_name]
        info = {}
        info["id"] = i
        info["desc"] = desc
        info["prop"] = prop
        mem.append(info)
    print("total", len(dft_3d))
    print("test_ids", len(test_ids))
    print("val_ids", len(val_ids))
    print("train_ids", len(train_ids))
    filename = os.path.join(output_dir, "id_prop_llm.json")
    filename_config = os.path.join(output_dir, "config.json")
    minfo = {}
    minfo["n_train"] = len(train_ids)
    minfo["n_val"] = len(val_ids)
    minfo["n_test"] = len(test_ids)
    minfo["id_prop_path"] = os.path.abspath(filename)
    minfo["output_dir"] = os.path.abspath(output_dir)

    dumpjson(data=minfo, filename=filename_config)
    dumpjson(data=mem, filename=filename)
    return output_dir


##
os.environ["WANDB_ANONYMOUS"] = "must"
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
try:
    import torch_xla.core.xla_model as xm

    xm.set_rng_state(random_seed)
except ImportError:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(random_seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
torch.use_deterministic_algorithms(True)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
# device = "cpu"


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
        # torch.tensor(inputs*10,dtype=inputs.dtype)
        return (
            inputs,
            self.ids[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# Example usage


def run_atomgpt(config_file="config.json"):
    print("Running AtomGPT prop predictor.")
    run_path = os.path.abspath(config_file).split("config.json")[0]
    print("PATH", run_path)
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config)
    id_prop_path = config.id_prop_path
    convert = config.convert
    # if convert:
    #    model = get_figshare_model(
    #        model_name="jv_formation_energy_peratom_alignn"
    #    )
    if ".zip" in id_prop_path:
        zp = zipfile.ZipFile(id_prop_path)
        dat = json.loads(zp.read(id_prop_path.split(".zip")[0]))
    elif ".csv" in id_prop_path:
        with open(id_prop_path, "r") as f:
            reader = csv.reader(f)
            dt = [row for row in reader]

        dat = []
        for i in tqdm(dt, total=len(dt)):
            info = {}
            info["id"] = i[0]
            info["prop"] = [float(j) for j in i[1:]]  # float(i[1])
            # pth=os.path.join(run_path,info['id'])
            pth = os.path.join(
                id_prop_path.split("id_prop.csv")[0], info["id"]
            )
            if convert:
                atoms = Atoms.from_poscar(pth)
                lines = atoms.describe()[config.desc_type]
                # lines = atoms.describe(model=model)[config.desc_type]
            else:

                with open(pth, "r") as f:
                    lines = f.read()
            info["desc"] = lines
            dat.append(info)

    else:
        dat = loadjson(id_prop_path)
    print("len", len(dat))
    prefix = config.prefix
    model_name = config.model_name
    batch_size = config.batch_size
    max_length = config.max_length
    num_epochs = config.num_epochs
    latent_dim = config.latent_dim
    learning_rate = config.learning_rate
    test_each_run = config.test_each_run
    pretrained_path = config.pretrained_path
    seed_val = config.seed_val
    include_struct = config.include_struct
    n_train = config.n_train
    n_val = config.n_val
    n_test = config.n_test
    train_ratio = config.train_ratio
    val_ratio = config.val_ratio
    test_ratio = config.test_ratio
    output_dir = config.output_dir
    keep_data_order = config.keep_data_order
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(config.dict(), indent=4))
    f.close()

    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(dat),
        split_seed=seed_val,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
    )

    train_texts = []
    train_targets = []
    train_ids_temp = []
    val_texts = []
    val_targets = []
    val_ids_temp = []
    test_texts = []
    test_targets = []
    test_ids_temp = []
    train_info = []
    val_info = []
    test_info = []
    for ii, i in enumerate(dat):
        if ii in id_train:
            train_texts.append(i["desc"])
            train_targets.append(i["prop"])
            train_ids_temp.append(i["id"])
            train_info.append(i)
        if ii in id_test:
            test_texts.append(i["desc"])
            test_targets.append(i["prop"])
            test_ids_temp.append(i["id"])
            val_info.append(i)
        if ii in id_val:
            val_texts.append(i["desc"])
            val_targets.append(i["prop"])
            val_ids_temp.append(i["id"])
            test_info.append(i)
    print("test_texts:", len(test_texts))
    print("val_texts example:", val_texts[0])
    print("test_texts example:", test_texts[0])

    print("Train\n", pd.DataFrame(train_info))
    print("Val\n", pd.DataFrame(val_info))
    print("test\n", pd.DataFrame(test_info))

    print("total", len(dat))
    print("test_ids", len(id_test))
    print("val_ids", len(id_val))
    print("train_ids", len(id_train))
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
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            # load_in_8bit=True,
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
        tokenizer.add_special_tokens({"unk_token": "#"})
        tokenizer.add_special_tokens({"unk_token": "&"})
        tokenizer.add_special_tokens({"unk_token": "@"})
        model.resize_token_embeddings(len(tokenizer))
    model.lm_head = torch.nn.Sequential(
        # torch.nn.Linear(model.config.hidden_size, 1),
        torch.nn.Linear(model.config.hidden_size, latent_dim),
        # torch.nn.Linear( latent_dim,256),
        # torch.nn.Transformer(d_model=latent_dim, nhead=1, num_encoder_layers=1, num_decoder_layers=1),
        # torch.nn.Linear(latent_dim, latent_dim),
        # torch.nn.Linear(latent_dim, latent_dim),
        # torch.nn.ReLU(),
        # torch.nn.LeakyReLU(),
        # torch.nn.Dropout(p=0.2),
        # torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4), num_layers=2),
        # torch.nn.Linear(256, 1),
        torch.nn.Linear(latent_dim, 1),
    )
    if pretrained_path != "":
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    # model.lm_head = torch.nn.Sequential(torch.nn.Linear( model.config.hidden_size, 256),torch.nn.SiLU(),torch.nn.Linear( 256, 1) )
    # set_seed(seed)
    # set_deterministic()
    model.to(device)
    if torch.cuda.device_count() > 1:
        device_ids = [d for d in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
    # Prepare datasets and dataloaders with data collator
    # TODO: knc6 change later
    train_dataset = AtomGPTDataset(
        texts=train_texts,
        targets=train_targets,
        ids=train_ids_temp,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = AtomGPTDataset(
        texts=val_texts,
        targets=val_targets,
        tokenizer=tokenizer,
        ids=val_ids_temp,
        max_length=max_length,
    )
    val_dataset = AtomGPTDataset(
        texts=test_texts,
        targets=test_targets,
        tokenizer=tokenizer,
        ids=test_ids_temp,
        max_length=max_length,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=pct_start,
        pct_start=0.3,
    )
    # output_dir = prefix + "_out"  # + model_name + "_" + dataset + "_" + prop
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
            # print('input_ids',input_ids.shape)
            if "t5" in model_name:
                input_ids = batch[0]["input_ids"].squeeze(1)  # .squeeze(0)
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
            # print('train',predictions,targets)
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            train_loss += loss.item()
        scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        t2 = time.time()
        train_time = round(t2 - t1, 3)

        # total_eval_mae_loss = 0
        # predictions_list = []
        # targets_list = []
        model.eval()
        val_loss = 0
        t1 = time.time()
        fname = os.path.join(output_dir, "val_results.csv")
        f = open(fname, "w")
        f.write("id,target,predictions\n")
        with torch.no_grad():
            for batch in val_dataloader:
                # input_ids = batch[0]["input_ids"].squeeze()  # .squeeze(0)
                input_ids = batch[0]["input_ids"].squeeze(1)  # .squeeze(0)
                ids = batch[1]
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
                    f.write(line)
        f.close()
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
        model.eval()
        with torch.no_grad():
            if test_each_run:
                t1_test = time.time()
                # model.eval()
                fname = os.path.join(output_dir, "test_results.csv")
                f = open(fname, "w")
                f.write("id,target,predictions\n")
                test_loss = 0
                for batch in test_dataloader:
                    input_ids = batch[0]["input_ids"].squeeze()  # .squeeze(0)
                    if "t5" in model_name:
                        input_ids = batch[0]["input_ids"].squeeze(
                            1
                        )  # .squeeze(0)
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
                    loss = criterion(
                        predictions.squeeze(), targets.squeeze().to(device)
                    )
                    test_loss += loss.item()
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
                        f.write(line)
                test_loss = test_loss / len(test_dataloader)
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
                # mae,
                test_loss,
                train_time,
                val_time,
                test_time,
                saving_tag,
            )

    model.eval()
    fname = os.path.join(output_dir, "test_results_final.csv")
    f = open(fname, "w")
    f.write("id,target,predictions\n")
    for batch in test_dataloader:
        optimizer.zero_grad()
        input_ids = batch[0]["input_ids"].squeeze()  # .squeeze(0)
        if "t5" in model_name:
            input_ids = batch[0]["input_ids"].squeeze(1)  # .squeeze(0)
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


def main():
    args = parser.parse_args(sys.argv[1:])
    run_atomgpt(config_file=args.config_name)


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    # args = parser.parse_args(sys.argv[1:])
    # run_atomgpt(config_file=args.config_name)
    #    config_file="config.json"
    # )
    main()
