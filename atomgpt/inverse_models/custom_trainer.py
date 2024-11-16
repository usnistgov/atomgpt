from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from trl import SFTTrainer


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
                print("self.tokenizer", self.tokenizer)
                target_texts = self.tokenizer.batch_decode(
                    labels
                )  # , skip_special_tokens=True)
                target_inputs = self.tokenizer(
                    target_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                # Generate outputs
                outputs = self.model.generate(
                    input_ids=target_inputs["input_ids"],
                    max_new_tokens=1024,
                    use_cache=True,
                )

                # Decode the generated outputs for analysis or debugging
                generated_texts = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                print("Generated Texts:", generated_texts)
                import sys

                sys.exit()
                x = logits  # .view(-1, logits.size(-1))
                y = labels  # .view(-1)
                print("x", x, x.shape, logits.shape)
                print("y", y, y.shape, labels.shape)
                outputs = self.model.generate(
                    target, max_new_tokens=1024, use_cache=True
                )
                print("outputs", outputs)
                response = self.tokenizer.batch_decode(
                    labels
                )  # [0].split("# Output:")[1]
                # loss_fn = nn.L1Loss()
                # target = labels.float()
                # loss = loss_fn(logits.view(-1), target.view(-1))

        elif self.loss_type == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss
