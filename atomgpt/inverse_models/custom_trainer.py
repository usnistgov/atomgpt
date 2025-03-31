from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from trl import SFTTrainer
from atomgpt.inverse_models.utils import text2atoms
from transformers.trainer_callback import TrainerCallback
from jarvis.core.specie import Specie


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


from transformers.trainer_callback import TrainerCallback
import torch
from transformers.trainer_callback import TrainerCallback
import torch
from transformers.trainer_callback import TrainerCallback
import torch


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


class AtomicStructureCallbackddd(TrainerCallback):
    def __init__(
        self, extractor_function, tokenizer, device="cuda", step_frequency=100
    ):
        super().__init__()
        self.extractor_function = extractor_function
        self.tokenizer = tokenizer
        self.device = device
        self.step_frequency = step_frequency
        self.trainer = None

    def set_trainer(self, trainer):
        """Gets called automatically when the trainer is initialized"""
        self.trainer = trainer
        print("Successfully connected to trainer in set_trainer")

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started - AtomicStructureCallback initialized")
        print("dddddd", self.trainer.get_train_dataloader())

    def on_step_end(self, args, state, control, **kwargs):
        if (
            state.global_step % self.step_frequency != 0
            or self.trainer is None
        ):
            return

        print(f"\n----- Step {state.global_step} -----")

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


class AtomicStructureCallbackX(TrainerCallback):
    def __init__(
        self, extractor_function, tokenizer, device="cuda", step_frequency=100
    ):
        super().__init__()
        self.extractor_function = extractor_function
        self.tokenizer = tokenizer
        self.device = device
        self.step_frequency = step_frequency
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Store reference to the trainer when training begins"""
        print("Training started - AtomicStructureCallback initialized")
        if "trainer" in kwargs:
            self.trainer = kwargs["trainer"]
            print("Successfully connected to trainer")
        else:
            print("Warning: Could not access trainer")

    def on_step_end(self, args, state, control, **kwargs):
        """Extract structures at regular intervals during training"""
        if (
            state.global_step % self.step_frequency != 0
            or self.trainer is None
        ):
            return

        print(f"\n----- Step {state.global_step} -----")

        # Access the model from the trainer
        model = self.trainer.model

        # Get a batch from the training dataloader
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

        # Process the batch
        self._process_batch(model, batch)

    def _process_batch(self, model, batch):
        """Process a batch to extract atomic structures"""
        # Move to correct device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Save model state
        was_training = model.training
        model.eval()

        # Generate predictions
        with torch.no_grad():
            # Take just a few examples
            input_ids = batch["input_ids"][:2]
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask[:2]

            # Get input texts for reference
            input_texts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

            # Generate outputs
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Adjust as needed
                num_return_sequences=1,
            )

        # Decode generated outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        # Extract atomic structures
        atomic_structures = self.extractor_function(generated_texts)

        # Print the results
        for i, (input_text, gen_text, structure) in enumerate(
            zip(input_texts, generated_texts, atomic_structures)
        ):
            print(f"\nExample {i+1}:")
            print(f"Input: {input_text[:100]}...")
            print(f"Generated: {gen_text[:100]}...")
            print(f"Extracted atomic structure:\n{structure}\n")
            print("-" * 40)

        # Restore model state
        if was_training:
            model.train()


def extract_atomic_structure(target_texts):
    atomic_structures = []
    for text in target_texts:
        if "### Output:" in text:
            structure_part = text.split("### Output:")[1].strip()
            atomic_structures.append(structure_part)
        else:
            atomic_structures.append("No structure found")
    return atomic_structures


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


def extract_atomic_structure(target_texts):
    atomic_structures = []
    for text in target_texts:
        # Split the text at "### Output:"
        if "### Output:" in text:
            structure_part = text.split("### Output:")[1].strip()
            atomic_structures.append(structure_part)
        else:
            print("No '### Output:' found in the text.")
    return atomic_structures


from transformers import TrainerCallback, TrainerState, TrainerControl


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


class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        dataset_text_field,
        max_seq_length,
        dataset_num_proc,
        packing,
        args: TrainingArguments,
        loss_type="default",  # Default to MSE
        # callbacks=CustomCallback(),
        # callbacks=CustomCallback(),
    ):
        """
        Initialize CustomSFTTrainer with explicit parameters and loss type.

        :param model: The model to train.
        :param tokenizer: The tokenizer to preprocess the data.
        :param train_dataset: The dataset for training.
        :param dataset_text_field: The text field in the dataset.
        :param max_seq_length: Maximum sequence length for tokenization.
        :param dataset_num_proc: Number of processes for dataset preprocessing.
        :param packing: Whether to use packing for sequences.
        :param args: TrainingArguments object.
        :param loss_type: The type of loss function ('mse', 'l1', 'cross_entropy').
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            args=args,
            # callbacks=[AtomicStructureCallback(extractor_function=extract_atomic_structure, tokenizer=tokenizer)]
        )

        # atomic_callback = AtomicStructureCallback(
        #    extractor_function=extract_atomic_structure,  # Your function
        #    tokenizer=tokenizer,  # Your tokenizer
        #    device=model.device  # Get the device from your model
        # )
        # self.model=model
        # if callbacks:
        #    self.add_callback(callbacks)
        # self.add_callback(atomic_callback)
        # cback = AtomicStructureCallback(extractor_function=extract_atomic_structure, tokenizer=tokenizer)
        def tokenize_function(example):
            return tokenizer(
                example["text"], padding="max_length", truncation=True
            )

        some_tokenized_dataset = train_dataset.map(
            tokenize_function, batched=True
        )
        example_callback = ExampleTrainerCallback(
            some_tokenized_dataset=some_tokenized_dataset
        )
        self.add_callback(example_callback)
        # self.add_callback(cback)
        self.loss_type = loss_type.lower()
        # self.use_bare_trainer = use_bare_trainer

    def calculate_density(self, atomic_structure):
        return len(
            atomic_structure
        )  # Placeholder: use actual calculation logic

    def extract_atomic_structure(self, target_texts):
        atomic_structures = []
        for text in target_texts:
            # Split the text at "### Output:"
            if "### Output:" in text:
                structure_part = text.split("### Output:")[1].strip()
                atomic_structures.append(structure_part)
            else:
                print("No '### Output:' found in the text.")
        return atomic_structures

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation based on the selected loss type or the bare trainer.
        """
        if self.loss_type == "default":  # crossentropy
            # Currently recommneded to use default
            return super().compute_loss(model, inputs, return_outputs)

        # Custom loss computation
        labels = inputs.get("labels")
        # print("Labels:", labels)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Select appropriate loss function
        if self.loss_type == "mse":
            loss_fn = nn.MSELoss()
            target = labels.float()
            loss = loss_fn(logits.view(-1), target.view(-1))
        elif self.loss_type == "l1":
            loss_fn = nn.L1Loss()
            target = labels.float()
            loss = loss_fn(logits.view(-1), target.view(-1))
        elif self.loss_type == "atomgpt_structure":
            print("Not tested, use with caution")
            if labels is not None:
                # labels = labels.cpu().numpy()
                # print("self.tokenizer", self.tokenizer)
                # print("inputs", inputs,inputs['input_ids'].shape)
                logits = logits.argmax(-1)  # .view(-1)
                # print('logits',logits,logits.shape)
                # print('labels1',labels,labels.shape)
                # Need to make generalized
                # labels[labels == -100] = 0 #self.tokenizer.eos_token_id
                # print('labels2',labels,labels.shape)
                # Generate outputs
                # Decode generated text (example for illustration)
                # loss_fn = nn.CrossEntropyLoss()
                loss_fn = nn.L1Loss()
                # x = logits.view(-1, logits.size(-1))
                # y = labels.view(-1)
                loss = loss_fn(logits, labels)
                return loss

                target_texts = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                pred_texts = self.tokenizer.batch_decode(
                    logits, skip_special_tokens=True
                )
                # print('target_texts',target_texts)
                # print('pred_texts',pred_texts)
                # Extract atomic structures (or manipulate the texts)
                target_atomic_structures = self.extract_atomic_structure(
                    target_texts
                )
                pred_atomic_structures = self.extract_atomic_structure(
                    pred_texts
                )
                # print('target_atomic_structures',target_atomic_structures)
                # print('pred_atomic_structures',pred_atomic_structures)
                total_loss = 0
                for target, pred in zip(
                    target_atomic_structures, pred_atomic_structures
                ):
                    total_loss += compute_losses(
                        parse_structure(target), parse_structure(pred)
                    )
                # print('loss',total_loss)
                return total_loss

        elif self.loss_type == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
            x = logits.view(-1, logits.size(-1))
            y = labels.view(-1)
            # print('x',x.shape)
            # print('y',y.shape)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            # print('loss',loss,loss.shape)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss
