from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from trl import SFTTrainer
from atomgpt.inverse_models.utils import text2atoms
from transformers import TrainerCallback, TrainerState, TrainerControl
from jarvis.core.specie import Specie
import re
import torch
from trl import SFTTrainer
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_rdf(
    atomic_positions, lattice_params, num_bins=100, max_distance=10.0
):
    """
    Compute the Radial Distribution Function (RDF) for a given set of atomic positions.
    """
    num_atoms = atomic_positions.shape[0]
    distances = []

    # Compute pairwise distances considering periodic boundary conditions (PBC)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            delta = atomic_positions[i] - atomic_positions[j]
            delta -= (
                torch.round(delta / lattice_params) * lattice_params
            )  # Apply PBC
            distance = torch.norm(delta)
            if distance < max_distance:
                distances.append(distance.item())

    # Histogram the distances
    distances = torch.tensor(distances, device=atomic_positions.device)
    hist = torch.histc(distances, bins=num_bins, min=0.0, max=max_distance)

    # Normalize RDF
    bin_edges = torch.linspace(
        0, max_distance, steps=num_bins + 1, device=atomic_positions.device
    )
    bin_width = bin_edges[1] - bin_edges[0]
    normalization = (
        (4 / 3)
        * torch.pi
        * (torch.pow(bin_edges[1:], 3) - torch.pow(bin_edges[:-1], 3))
    )
    normalization *= num_atoms / (
        torch.prod(lattice_params) * num_atoms * (num_atoms - 1) / 2
    )

    rdf = hist / normalization
    return rdf, bin_edges[:-1]


def parse_structure(structure_str, num_bins=100, max_distance=10.0):
    """
    Parse a structure string and compute relevant quantities, including RDF and atomic number differences.
    """
    lines = structure_str.split("\n")
    lattice_params = torch.tensor(
        [float(x) for x in lines[0].split()], requires_grad=True, device=device
    )
    angles = torch.tensor(
        [float(x) for x in lines[1].split()], requires_grad=True, device=device
    )
    # print("lattice_params", lattice_params)
    # print("angles", angles)
    # Parse atomic positions and chemical elements
    atomic_positions = []
    chemical_elements = []
    for line in lines[2:]:
        if line.strip():  # Ignore empty lines
            tokens = line.split()
            chemical_elements.append(tokens[0])
            atomic_positions.append([float(x) for x in tokens[1:]])
    atomic_positions = torch.tensor(
        atomic_positions, requires_grad=True, device=device
    )

    # Map chemical elements to atomic numbers
    atomic_numbers = torch.tensor(
        [Specie(el).Z for el in chemical_elements],
        dtype=torch.float32,
        device=device,
    )

    # Compute RDF
    rdf, bin_edges = compute_rdf(
        atomic_positions, lattice_params, num_bins, max_distance
    )

    return (
        lattice_params,
        angles,
        atomic_positions,
        chemical_elements,
        atomic_numbers,
        rdf,
        bin_edges,
    )


def compute_losses(
    target_data,
    pred_data,
    weights={
        "atomic_numbers": 1.0,
        "lattice_params": 1.0,
        "angles": 0.001,
        "coordinates": 1.0,
        "rdf": 1.0,
    },
):
    """
    Compute weighted losses for atomic numbers, lattice parameters, angles, coordinates, and RDF.
    """
    # Unpack data
    (
        target_lattice_params,
        target_angles,
        target_coords,
        _,
        target_atomic_numbers,
        target_rdf,
        _,
    ) = target_data
    (
        pred_lattice_params,
        pred_angles,
        pred_coords,
        _,
        pred_atomic_numbers,
        pred_rdf,
        _,
    ) = pred_data

    # Atomic number difference loss
    atomic_number_loss = torch.mean(
        (pred_atomic_numbers - target_atomic_numbers) ** 2
    )

    # Lattice parameters loss
    lattice_params_loss = torch.mean(
        (pred_lattice_params - target_lattice_params) ** 2
    )

    # Angles loss
    angles_loss = torch.mean((pred_angles - target_angles) ** 2)

    # Atomic coordinates loss
    coordinates_loss = torch.mean((pred_coords - target_coords) ** 2)

    # RDF loss
    rdf_loss = torch.mean((pred_rdf - target_rdf) ** 2)

    # Combine losses with weights
    total_loss = (
        weights["atomic_numbers"] * atomic_number_loss
        + weights["lattice_params"] * lattice_params_loss
        + weights["angles"] * angles_loss
        + weights["coordinates"] * coordinates_loss
        + weights["rdf"] * rdf_loss
    )
    # Return individual losses and total loss
    p = {
        "atomic_number_loss": atomic_number_loss.item(),
        "lattice_params_loss": lattice_params_loss.item(),
        "angles_loss": angles_loss.item(),
        "coordinates_loss": coordinates_loss.item(),
        "rdf_loss": rdf_loss.item(),
        "total_loss": total_loss.item(),
    }
    # print('p',p)
    return total_loss


def a_compute_loss(target_structure, pred_structure):
    target_lattice, target_angles, target_positions = parse_structure(
        target_structure
    )
    pred_lattice, pred_angles, pred_positions = parse_structure(pred_structure)

    # Loss for lattice parameters
    lattice_loss = torch.nn.functional.mse_loss(pred_lattice, target_lattice)

    # Loss for angles
    angle_loss = torch.nn.functional.mse_loss(pred_angles, target_angles)

    # Loss for atomic positions
    position_loss = torch.nn.functional.mse_loss(
        pred_positions, target_positions
    )

    # Combine losses (weighted sum)
    total_loss = lattice_loss + angle_loss + position_loss
    return total_loss


class AtomicStructureCallback(TrainerCallback):
    def __init__(self, extractor_function, tokenizer, device="cuda"):
        super().__init__()
        self.extractor_function = extractor_function
        self.tokenizer = tokenizer
        self.device = device
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        print("Successfully connected to trainer in set_trainer")

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started - AtomicStructureCallback initialized")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return

        print(f"\n===== Epoch {state.epoch:.0f} End =====")

        model = self.trainer.model

        if not hasattr(self, "train_dataloader"):
            print("Creating dataloader iterator")
            self.train_dataloader = self.trainer.get_train_dataloader()
            self.dataloader_iter = iter(self.train_dataloader)

        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            print("Resetting dataloader iterator")
            self.dataloader_iter = iter(self.train_dataloader)
            batch = next(self.dataloader_iter)

        self._process_batch(model, batch)

    def _process_batch(self, model, batch):
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        was_training = model.training
        model.eval()

        with torch.no_grad():
            input_ids = batch["input_ids"][:2]
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask[:2]

            input_texts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_return_sequences=1,
            )

        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        atomic_structures = self.extractor_function(generated_texts)
        for i, (input_text, gen_text, structure) in enumerate(
            zip(input_texts, generated_texts, atomic_structures)
        ):
            print(f"\nExample {i+1}:")
            print(f"Input: {input_text[:100]}...")
            print(f"Generated: {gen_text[:100]}...")
            print(f"Extracted atomic structure:\n{structure}\n")
            print("-" * 40)

        if was_training:
            model.train()


class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Initialize any variables you need

    def on_train_begin(self, args, state, control, **kwargs):
        # Called at the beginning of training
        print("Training has begun!")

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Called at the beginning of each epoch
        print(f"Epoch {state.epoch} started")

    def on_step_end(self, args, state, control, **kwargs):
        # Called at the end of each optimization step
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step} completed")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Called after evaluation
        print(f"Evaluation results: {metrics}")

    def on_train_end(self, args, state, control, **kwargs):
        # Called at the end of training
        print("Training completed!")

    # Other methods you can override:
    # on_log, on_prediction_step, on_save, etc.


class ExampleTrainerCallback(TrainerCallback):
    """Custom ExampleTrainerCallback that accesses the model after each epoch"""

    def __init__(self, some_tokenized_dataset):
        """Initializes the ExampleTrainerCallback instance."""
        super().__init__()
        # --------------------- Add custom code here ------------------------------------
        self.some_tokenized_dataset = some_tokenized_dataset
        # ------------------------------------------------------------------------------

    # Overriding the on_epoch_end() function
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an epoch.
        """
        # --------------------- Add custom code here ------------------------------------
        print("Hello an epoch has ended!")

        # Access the current state of the model after the epoch ends:
        model = kwargs["model"]

        # Add some custom code here...
        model.eval()

        # Perform inference on some dataset
        with torch.no_grad():
            for item in self.some_tokenized_dataset:
                print("item", item)
                print("item k", item.keys())
                input_ids = item[
                    "input_ids"
                ]  # .unsqueeze(0)  # Add batch dimension
                attention_mask = item[
                    "attention_mask"
                ]  # .unsqueeze(0)  # Add batch dimension

                # Forward pass, assuming model is a BertForSequenceClassification type
                # i.e. model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                print("x", prediction)
                # Do something with prediction
        # ------------------------------------------------------------------------------


def extract_atomic_structure(target_texts):
    atomic_structures = []
    for text in target_texts:
        if "### Output:" in text:
            structure_part = text.split("### Output:")[1].strip()
            # Basic validation: must contain at least 3 lines (lattice, angles, 1 atom)
            lines = structure_part.splitlines()
            if len(lines) >= 3:
                atomic_structures.append(structure_part)
            else:
                print("⚠️ Output too short, skipping.")
        else:
            print("⚠️ No '### Output:' found in the text.")
    return atomic_structures


def parse_structure_text(text):
    if "### Output:" in text:
        text = text.split("### Output:")[1]
    else:
        raise ValueError("No '### Output:' section found.")

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    if len(lines) < 3:
        raise ValueError("Too few lines to be a valid atomic structure.")

    try:
        lattice = list(map(float, lines[0].split()))
        if len(lattice) != 3:
            raise ValueError("Lattice line must contain exactly 3 values.")

        angles = list(map(float, lines[1].split()))
        if len(angles) != 3:
            raise ValueError("Angles line must contain exactly 3 values.")
    except Exception as e:
        raise ValueError(f"Failed to parse lattice/angles: {e}")

    # FIXED: capture 4 groups (element + 3 floats)
    atom_line_pattern = re.compile(
        r"^\s*([A-Z][a-z]?)\s+"
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+"
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+"
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    )

    atoms = []
    for i, line in enumerate(lines[2:], start=3):
        match = atom_line_pattern.match(line)
        if match:
            try:
                element = match.group(1)
                coords = [
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                ]
                atoms.append((element, coords))
            except Exception as e:
                print(f"⚠️ Failed to parse atom line {i}: {line} | Error: {e}")
        else:
            print(f"⚠️ Line {i} does not match atom pattern: {line}")
            continue

    if len(atoms) == 0:
        raise ValueError("No valid atom lines found.")

    return lattice, angles, atoms


class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        dataset_text_field,
        max_seq_length,
        dataset_num_proc,
        packing,
        args: TrainingArguments,
        loss_type="default",  # Default to MSE
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            args=args,
        )
        # self.model=model
        self.loss_type = loss_type.lower()
        # self.use_bare_trainer = use_bare_trainer

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "default":  # crossentropy
            # Currently recommneded to use default
            return super().compute_loss(model, inputs, return_outputs)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        input_ids = inputs[
            "input_ids"
        ]  # assuming input includes only prompt (not full answer)
        attention_mask = inputs.get("attention_mask", None)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=108,
            do_sample=False,  # greedy decoding
        )

        pred_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Clean and decode target texts
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        cleaned_labels = [[t for t in seq if t != -100] for seq in labels]
        target_texts = self.tokenizer.batch_decode(
            cleaned_labels, skip_special_tokens=True
        )

        # Device where model tensors live
        device = logits.device
        total_loss = None

        for pred, target in zip(pred_texts, target_texts):
            # try:
            if target:
                print("target", target)
                tgt_lat, tgt_ang, tgt_atoms = parse_structure_text(target)
                print("pred", pred)
                pred_lat, pred_ang, pred_atoms = parse_structure_text(pred)
                print("tgt_atoms", tgt_atoms)
                print("pred_atoms", pred_atoms)
                print()
                print()
                print()

                # Convert values to tensors
                pred_lat = torch.tensor(
                    pred_lat, device=device, dtype=torch.float32
                )
                tgt_lat = torch.tensor(
                    tgt_lat, device=device, dtype=torch.float32
                )
                pred_ang = torch.tensor(
                    pred_ang, device=device, dtype=torch.float32
                )
                tgt_ang = torch.tensor(
                    tgt_ang, device=device, dtype=torch.float32
                )

                # Lattice and angle MSE loss
                loss_lat = F.mse_loss(pred_lat, tgt_lat)
                loss_ang = F.mse_loss(pred_ang, tgt_ang)

                # Atom type mismatch
                type_mismatch_count = sum(
                    1 for p, t in zip(pred_atoms, tgt_atoms) if p[0] != t[0]
                )
                loss_type = torch.tensor(
                    [type_mismatch_count / max(1, len(tgt_atoms))],
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )

                # Coordinate MSE loss
                pred_coords = torch.tensor(
                    [p[1] for p in pred_atoms],
                    device=device,
                    dtype=torch.float32,
                )
                tgt_coords = torch.tensor(
                    [t[1] for t in tgt_atoms],
                    device=device,
                    dtype=torch.float32,
                )
                loss_coord = F.mse_loss(pred_coords, tgt_coords)

                # Combined structured loss
                loss = loss_lat + loss_ang + loss_type + loss_coord

            # except Exception as e:
            #    # Parsing or shape mismatch fallback
            #    loss = torch.tensor([10.0], device=device, requires_grad=True)

            # Accumulate total loss
            if total_loss is None:
                total_loss = loss.squeeze()
            else:
                total_loss = total_loss + loss.squeeze()

        final_loss = total_loss / len(pred_texts)
        print("final_loss", final_loss)
        return (final_loss, outputs) if return_outputs else final_loss
