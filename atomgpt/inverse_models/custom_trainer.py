from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from trl import SFTTrainer
from atomgpt.inverse_models.utils import text2atoms


def extract_atomic_structure(target_texts):
    """
    Extracts the atomic structure description from a list of target texts.

    :param target_texts: List of strings containing target texts with atomic structure details.
    :return: List of strings with only the atomic structure descriptions.
    """
    atomic_structures = []

    for text in target_texts:
        # Split the text at "### Output:"
        if "### Output:" in text:
            structure_part = text2atoms(
                text.split("### Output:")[1]
            )  # .strip()
            atomic_structures.append(structure_part)
        else:
            print("No '### Output:' found in the text.")

    return atomic_structures


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
        )
        # self.model=model
        self.loss_type = loss_type.lower()
        # self.use_bare_trainer = use_bare_trainer

    def calculate_density(self, atomic_structure):
        # Example of a function to calculate density (or any other feature from atomic structure)
        # You can implement this based on your domain knowledge.
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
            # if self.use_bare_trainer:
            # Use the bare SFTTrainer's loss computation
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
        elif self.loss_type == "density":

            if labels is not None:
                # labels = labels.cpu().numpy()
                # print("self.tokenizer", self.tokenizer)
                # print("inputs", inputs,inputs['input_ids'].shape)
                # print('logits',logits,logits.shape)
                # print('labels1',labels,labels.shape)
                # Need to make generalized
                labels[labels == -100] = 0
                # print('labels2',labels,labels.shape)
                # Generate outputs
                # Decode generated text (example for illustration)
                target_texts = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                pred_texts = self.tokenizer.batch_decode(
                    logits.argmax(-1), skip_special_tokens=True
                )

                # Extract atomic structures (or manipulate the texts)
                target_atomic_structures = self.extract_atomic_structure(
                    target_texts
                )
                pred_atomic_structures = self.extract_atomic_structure(
                    pred_texts
                )

                # For demonstration, let's calculate the L1 loss between target and predicted atomic structures
                # Assuming that the atomic structures are numerical or encoded in a way that we can directly compare
                # For the sake of this example, let's assume you have a function to calculate density or other features
                # Example: comparing the density (or other features) of the predicted and target atomic structures
                target_densities = torch.tensor(
                    [
                        self.calculate_density(struct)
                        for struct in target_atomic_structures
                    ]
                )
                pred_densities = torch.tensor(
                    [
                        self.calculate_density(struct)
                        for struct in pred_atomic_structures
                    ]
                )

                # Ensure the tensors are on the correct device
                target_densities = target_densities.to(logits.device)
                pred_densities = pred_densities.to(logits.device)

                # Custom loss: L1 loss between target and predicted densities
                loss_fn = nn.L1Loss()
                loss = loss_fn(pred_densities, target_densities)
                print(loss)
                return loss
                import sys

                sys.exit()
                target_out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=2024,
                    use_cache=True,
                )
                # print("target_out", target_out)

                # Decode the generated outputs for analysis or debugging
                target_texts = self.tokenizer.batch_decode(
                    target_out, skip_special_tokens=True
                )
                target_atom_texts = extract_atomic_structure(target_texts)
                # print("Target Texts:", target_texts,target_atom_texts)

                gen_out = self.model.generate(
                    input_ids=labels,
                    max_new_tokens=2024,
                    use_cache=True,
                )
                # print("gen_out", gen_out)

                # Decode the generated outputs for analysis or debugging
                gen_texts = self.tokenizer.batch_decode(
                    gen_out, skip_special_tokens=True
                )
                gen_atom_texts = extract_atomic_structure(gen_texts)
                # print("Generated Texts:", gen_texts,gen_atom_texts)
                loss_fn = nn.L1Loss()
                target = torch.tensor(
                    [i.density for i in target_atom_texts],
                    device=labels.device,
                    dtype=torch.float,
                    requires_grad=False,
                )
                pred = torch.tensor(
                    [i.density for i in gen_atom_texts],
                    device=labels.device,
                    dtype=torch.float,
                    requires_grad=True,
                )

                # target = torch.tensor([i.density for i in target_atom_texts]).to(labels.device)
                # pred = torch.tensor([i.density for i in gen_atom_texts]).to(labels.device)
                loss = loss_fn(target, pred)
                print("target", target)
                print("pred", pred)
                print("loss", loss)
                return loss

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
