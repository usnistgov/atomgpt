#!/usr/bin/env python
"""Module to train properties."""
import transformers
from atomgpt.data.dataset import data_from_benchmark_file, data_from_id_prop
from atomgpt.config import TrainingPropConfig
import os
import json
import zipfile
import torch
from jarvis.db.figshare import data
import time
import pandas as pd
from sklearn.metrics import mean_absolute_error
import random
import numpy as np
import os
from jarvis.db.jsonutils import loadjson, dumpjson
import sys
import argparse
import pprint

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

parser = argparse.ArgumentParser(description="AtomGPT")
parser.add_argument(
    "--config_file",
    default="config.json",
    help="Config file",
)


def set_seed(random_seed=42):
    os.environ["WANDB_ANONYMOUS"] = "must"
    # random_seed = 42
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


def run_atomgpt(config_file=""):
    print("Running AtomGPT prop predictor.")
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    benchmark_file = config.benchmark_file
    id_prop_path = config.id_prop_path
    prefix = config.prefix
    model_name = config.model_name
    leaderboard_dir = config.leaderboard_dir
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
    keep_data_order = config.keep_data_order
    output_dir = config.output_dir
    print("configs", pprint.pprint(config.dict()))
    set_seed(random_seed=seed_val)
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

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.add_special_tokens({"unk_token": "#"})
        tokenizer.add_special_tokens({"unk_token": "&"})
        tokenizer.add_special_tokens({"unk_token": "@"})
        model.resize_token_embeddings(len(tokenizer))
    model.lm_head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, latent_dim),
        torch.nn.Linear(latent_dim, 1),
    )
    if benchmark_file is not None:
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = data_from_benchmark_file(
            benchmark_file=benchmark_file,
            leaderboard_dir=leaderboard_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            include_struct=include_struct,
        )
    elif id_prop_path is not None:
        train_dataloader, val_dataloader, test_dataloader = data_from_id_prop(
            id_prop_path=id_prop_path,
            tokenizer=tokenizer,
            max_length=max_length,
            split_seed=seed_val,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            keep_data_order=keep_data_order,
            batch_size=batch_size,
            include_struct=include_struct,
            calc_desc=False,
        )
    else:
        raise ValueError("Provide id_prop_path or benchmark_file")

    val_dataloader = test_dataloader  # for now
    if pretrained_path != "":
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    if torch.cuda.device_count() > 1:
        device_ids = [d for d in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    criterion = torch.nn.L1Loss()
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=pct_start,
        pct_start=0.3,
    )
    # print("train_data", len(train_texts))
    # print("test_data", len(test_texts))
    # output_dir = prefix + "_out_" + model_name
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
            # print('train',predictions,targets)
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
    # box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
    # coords = [[0, 0, 0], [0.25, 0.2, 0.25]]
    # elements = ["Si", "Si"]
    # Si = Atoms(lattice_mat=box, coords=coords, elements=elements)
    # tmp=atoms_describer(Si)
    # print(tmp)
    # import sys
    # sys.exit()
    args = parser.parse_args(sys.argv[1:])
    config_file = args.config_file
    # "AI-SinglePropertyPrediction-PBE_gap-halide_peroskites-test-mae.csv.zip"
    # "AI-SinglePropertyPrediction-Tc_supercon-dft_3d-test-mae.csv.zip"
    # id_prop_path = (
    #    "/wrk/knc6/Software/mini_alignn/alignn/alignn/examples/sample_data"
    # )
    # "AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip"
    # "AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"
    # args.benchmark_file
    model_name = "facebook/opt-350m"
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    model_name = "google/flan-t5-small"
    model_name = "google/flan-t5-base"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name = "google-t5/t5-small"
    model_name = "xlnet/xlnet-base-cased"
    model_name = "afmck/testing-llama-tiny"
    model_name = "EleutherAI/gpt-neo-125m"
    model_name = "openai-community/gpt2-medium"
    model_name = "meta-llama/Llama-2-7b-hf"
    model_name = "stas/tiny-random-llama-2"
    model_name = "ahxt/llama2_xs_460M_experimental"
    model_name = "gpt2"
    run_atomgpt(
        config_file=config_file,
    )
