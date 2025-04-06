from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
import torch
from atomgpt.inverse_models.utils import (
    text2atoms,
)
import subprocess


class ExampleTrainerCallbackNew(TrainerCallback):
    def __init__(
        self,
        some_tokenized_dataset,
        tokenizer,
        max_length=2048,
        batch_size=4,
        # max_samples=10,
        run_every="step",
        callback_samples=2,
    ):
        """
        Args:
            run_every: "epoch" or "step"
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.run_every = run_every

        # Pre-select a smaller sample dataset for speed
        self.sample_dataset = some_tokenized_dataset.select(
            range(min(callback_samples, len(some_tokenized_dataset)))
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.run_every == "step":
            self._run_generation(**kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.run_every == "epoch":
            self._run_generation(**kwargs)

    def _run_generation(self, **kwargs):
        model = kwargs["model"]
        model.eval()

        B = self.batch_size
        with torch.no_grad():
            for i in range(0, len(self.sample_dataset), B):
                batch = self.sample_dataset[i : i + B]

                prompts = []
                targets = []
                print("batch", batch, type(batch))

                # Generate predictions in batch
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=self.max_length,
                    do_sample=False,
                )

                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for j, k in zip(batch["output"], generated_texts):
                    print("Target", j)
                    print("Predicted", k)
                    print()
                # """
                for j, (prompt, gen_text, target) in enumerate(
                    zip(prompts, generated_texts, targets)
                ):
                    # Remove prompt from generated output
                    predicted_answer = (
                        gen_text[len(prompt) :].strip()
                        if gen_text.startswith(prompt)
                        else gen_text.strip()
                    )

                    # Optional: convert text to atom structure (if available)
                    target_atoms = text2atoms("\n" + target)
                    predicted_atoms = text2atoms(predicted_answer)

                    print(f"\n🔹 Sample {i + j}")
                    print(f"🔸 Prompt   :\n{prompt}")
                    print(f"🔸 Target   :\n{target_atoms}")
                    print(f"🔸 Predicted:\n{predicted_atoms}")
                # """


class PrintGPUUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        try:
            gpu_info = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
            print(gpu_info)
        except Exception as e:
            print("Error fetching GPU info:", e)


class ExampleTrainerCallback(TrainerCallback):
    def __init__(
        self,
        some_tokenized_dataset,
        tokenizer,
        max_length=2048,
        run_every="step",
        callback_samples=2,
    ):
        """
        Args:
            run_every: "epoch" or "step"
        """
        super().__init__()
        self.some_tokenized_dataset = some_tokenized_dataset
        self.tokenizer = tokenizer
        # self.max_new_tokens = max_new_tokens
        self.run_every = run_every
        self.max_length = max_length
        self.callback_samples = callback_samples

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.run_every == "step":
            self._run_generation(args, state, control, **kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.run_every == "epoch":
            self._run_generation(args, state, control, **kwargs)

    def _run_generation(self, args, state, control, **kwargs):
        # print("🧠 Generating predictions...")

        model = kwargs["model"]
        model.eval()

        sample_dataset = (self.some_tokenized_dataset).select(
            range(min(self.callback_samples, len(self.some_tokenized_dataset)))
        )

        with torch.no_grad():
            for idx, item in enumerate(sample_dataset):
                try:
                    # Decode full input to string
                    full_input = self.tokenizer.decode(
                        item["input_ids"], skip_special_tokens=True
                    )

                    # Extract prompt up to '### Output:'
                    if "### Output:" in full_input:
                        prompt = (
                            full_input.split("### Output:")[0] + "### Output:"
                        )
                    else:
                        prompt = full_input

                    # Re-tokenize the trimmed prompt
                    encoded_prompt = self.tokenizer(
                        prompt, return_tensors="pt"
                    ).to(model.device)

                    # Generate continuation from the prompt
                    generated_ids = model.generate(
                        input_ids=encoded_prompt["input_ids"],
                        attention_mask=encoded_prompt.get(
                            "attention_mask", None
                        ),
                        max_new_tokens=self.max_length,
                        # max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # deterministic output
                    )

                    # Decode full output
                    generated_text = self.tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    print(f"\n🔹 Sample {idx}")
                    print("generated_text", generated_text)
                    # Optional: remove prompt from generated text to isolate prediction
                    if generated_text.startswith(prompt):
                        predicted_answer = generated_text[
                            len(prompt) :
                        ].strip()
                    else:
                        predicted_answer = generated_text.strip()
                    predicted_answer = "\n" + predicted_answer
                    # Get human-written target
                    target_text = "\n" + (item.get("output", "<no target>"))
                    # print("target_text", target_text)
                    # print("predicted_answer", predicted_answer)
                    target_text = text2atoms(target_text)
                    print(f"🔸 Target   :\n{target_text}")
                    # print('target_text2',target_text)
                    predicted_answer = text2atoms(predicted_answer)
                    # print(f"🔸 Prompt   :\n{prompt}")
                    print(f"🔸 Predicted:\n{predicted_answer}")
                except Exception:
                    pass
