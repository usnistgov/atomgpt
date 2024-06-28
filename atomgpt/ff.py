import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    AdamW,
    get_linear_schedule_with_warmup,
)
import numpy as np
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AtomGPTFF(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name="gpt2",
        latent_dim=512,
        n_out=3,
        tokenizer="",
        include_stress=True,
        stress_weight=0.1,
        force_weight=1,
    ):
        super(AtomGPTFF, self).__init__()
        self.config = transformers.GPT2Config.from_pretrained(
            pretrained_model_name
        )
        self.gpt2 = transformers.GPT2Model.from_pretrained(
            pretrained_model_name, config=self.config
        )
        self.tokenizer = tokenizer
        self.regressor_forces = torch.nn.Sequential(
            torch.nn.Linear(self.config.n_embd, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, n_out),
        )
        self.regressor_energies = torch.nn.Sequential(
            torch.nn.Linear(self.config.n_embd, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 1),
        )
        self.include_stress = include_stress
        self.force_weight = force_weight
        self.stress_weight = stress_weight

    def forward(self, sample):
        if isinstance(sample, dict):
            sample = sample["text"]
        texts = sample.split("@\n")[1].strip("&").split("\n")

        forces = []
        energies = []
        coords = []
        for text in texts:

            # print('coords',text.split()[1:])
            coord = np.array(text.strip("&").split()[1:], dtype=float)
            coords.append(coord)
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
                device
            )
            outputs = self.gpt2(input_ids=input_ids)
            last_hidden_state = outputs.last_hidden_state
            force = self.regressor_forces(last_hidden_state[:, -1, :])
            forces.append(force)
            energy = self.regressor_energies(last_hidden_state[:, -1, :])
            energies.append(energy)

        # Concatenate and pad forces to match the longest sequence in the batch
        max_len = max(len(f) for f in forces)
        forces = torch.cat(forces).to(device)

        coords = torch.tensor(np.array(coords), dtype=torch.float).to(device)
        net_energy = torch.sum(torch.cat(energies))
        # print('forces',forces,forces.shape)
        stress_tensor = torch.empty(1)
        if self.include_stress and self.stress_weight > 0:
            vol = float(sample.split("The volume is ")[1].split(".")[0])
            stress_tensor = torch.flatten(
                torch.sum(torch.einsum("ij,ik->ijk", coords, forces), dim=0)
                / vol
            ).squeeze(0)
        info = {}
        # info['forces'] = forces_padded
        # print('stress_tensor',stress_tensor,stress_tensor.shape)
        info["forces"] = forces
        info["coords"] = coords
        info["energy"] = net_energy
        info["stress"] = stress_tensor
        return info


class AtomGPTFFDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def collate_fn(batch):
    max_len = max(len(sample["forces"]) for sample in batch)
    padded_batch = []
    # print('batch',batch)
    for sample in batch:
        padded_sample = sample.copy()
        padded_sample["forces"] += [[0, 0, 0]] * (
            max_len - len(sample["forces"])
        )  # Pad the targets
        padded_batch.append(padded_sample)
    # print('padded_batch',padded_batch)
    return padded_batch


def train(
    tokenizer=None,
    latent_dim=512,
    train_array=[],
    val_array=[],
    test_array=[],
    include_stress=True,
    force_weight=1,
    stress_weight=0.1,
    batch_size=2,
    num_epochs=10,
    pretrained_model_name="gpt2",
):
    model = AtomGPTFF(
        tokenizer=tokenizer,
        include_stress=include_stress,
        latent_dim=latent_dim,
        pretrained_model_name=pretrained_model_name,
        force_weight=force_weight,
        stress_weight=stress_weight,
    )

    train_dataset = AtomGPTFFDataset(train_array)
    print("Instance train", train_dataset[0])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataset = AtomGPTFFDataset(val_array)
    print("Instance val", val_dataset[0])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if test_array:
        test_dataset = AtomGPTFFDataset(test_array)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = torch.nn.L1Loss()
    best_loss = np.inf
    for epoch in range(num_epochs):
        t1 = time.time()
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            batch_loss = 0
            optimizer.zero_grad()
            for sample in batch:
                pred = model(sample["text"])
                target_forces = torch.tensor(sample["forces"])[
                    0 : pred["forces"].shape[0]
                ].to(device)
                target_stress = torch.tensor(sample["stress"]).to(device)
                energy_loss = criterion(
                    pred["energy"], torch.tensor(sample["energy"]).to(device)
                )
                force_loss = force_weight * criterion(
                    pred["forces"], target_forces
                )
                # stress_loss = stress_weight * criterion(
                #    pred["stress"], target_stress.to(device)
                # )
                if include_stress:
                    loss = energy_loss + force_loss + stress_loss
                else:
                    loss = energy_loss + force_loss
                loss.backward()
                batch_loss += loss.item()
            train_loss += batch_loss
            optimizer.step()
            scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        model.eval()
        val_loss = 0
        for batch_idx, batch in enumerate(val_dataloader):
            batch_loss = 0
            # optimizer.zero_grad()
            for sample in batch:
                pred = model(sample["text"])
                target_forces = torch.tensor(sample["forces"])[
                    0 : pred["forces"].shape[0]
                ].to(device)
                target_stress = torch.tensor(sample["stress"]).to(device)
                energy_loss = (
                    torch.mean(
                        pred["energy"]
                        - torch.tensor(sample["energy"]).to(device)
                    )
                ) ** 2
                force_loss = (
                    force_weight * torch.mean(pred["forces"] - target_forces)
                ) ** 2
                # stress_loss = (
                #    stress_weight * torch.mean(pred["stress"] - target_stress)
                # ) ** 2
                include_stress = False
                if include_stress:
                    loss = energy_loss + force_loss + stress_loss
                else:
                    loss = energy_loss + force_loss
                loss.backward()
                batch_loss += loss.item()
            val_loss += batch_loss
            # optimizer.step()
            # scheduler.step()
        val_loss = val_loss / len(val_dataloader)
        output_dir = "./"
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_name = "best_model.pt"
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, best_model_name),
            )
        t2 = time.time()
        epoch_time = t2 - t1
        print(
            f"Epoch {epoch + 1}, Train Loss, Val Loss, Time:"
            f" {train_loss:.4f}, {val_loss:.4f}, {epoch_time:.4f}"
        )


if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    latent_dim = 512

    samples = [
        {
            "text": "The volume is 60.@\nGa 0 0 0&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1]],
            "energy": 1,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1]],
            "energy": 1,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2\nAl 3 3 3&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1]],
            "energy": 2,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2 \nAl 3 3 3 \nXe 4 4 4&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1]],
            "energy": 3,
        },
    ]
    train(
        tokenizer=tokenizer,
        latent_dim=512,
        train_array=samples,
        val_array=samples,
        test_array=samples,
        include_stress=True,
        batch_size=2,
        num_epochs=10,
        pretrained_model_name="gpt2",
    )
