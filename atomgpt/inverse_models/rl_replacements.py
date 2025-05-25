__all__ = [
    "RL_EXTRA_ARGS",
    "RL_FUNCTIONS",
    "RL_PRE_ITEMS",
    "RL_CONFIG_CHANGES",
    "RL_METRICS_CHANGES",
]

import re
import torch
import inspect
from collections import defaultdict

# from atomgpt.inverse_models.rl_replacements import RL_REPLACEMENTS
RL_EXTRA_ARGS = defaultdict(list)
RL_FUNCTIONS = defaultdict(list)
RL_PRE_ITEMS = defaultdict(list)
RL_CONFIG_CHANGES = defaultdict(list)
RL_METRICS_CHANGES = defaultdict(list)

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}


__all__ = ["RL_REPLACEMENTS"]

import torch
import inspect
import os
import numpy as np
from typing import Union, Callable, Optional, List, Dict

RL_REPLACEMENTS = dict()

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,  # Disable Triton mm kernels
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}


# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@torch.compile(
    dynamic=True,
    fullgraph=True,
    options=torch_compile_options,
)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(
        logits, dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim=-1)
    per_token_logps = (
        selected_logits - logsumexp_values
    )  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps


pass
RL_REPLACEMENTS["selective_log_softmax"] = selective_log_softmax


# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    old_logits, new_logits, input_ids, mask, beta, advantages
):
    # All AtomGPT Zoo code licensed under LGPLv3
    old_logits = old_logits.to(torch.float32)
    new_logits = new_logits.to(torch.float32)
    input_ids = input_ids.unsqueeze(-1)

    # x_i - logsumexp(x_i)
    old_x = torch.gather(old_logits, dim=-1, index=input_ids).squeeze(-1)
    new_x = torch.gather(new_logits, dim=-1, index=input_ids).squeeze(-1)
    old = old_x - torch.logsumexp(old_logits, dim=-1)
    new = new_x - torch.logsumexp(new_logits, dim=-1)

    # Reverse KL
    kl_i = torch.exp(old - new) - (old - new) - 1.0
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    loss_i = -(loss_i - beta * kl_i)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # See https://github.com/huggingface/trl/pull/2881
    loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    loss = loss_per_reward.mean()
    # loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()
    pass
    return loss, completion_length, mean_kl


pass
RL_REPLACEMENTS["grpo_compute_loss"] = grpo_compute_loss
RL_REPLACEMENTS["grpo_compute_loss_slow"] = (
    f"@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)\n"
    f"{inspect.getsource(grpo_compute_loss)}"
)
RL_REPLACEMENTS["grpo_compute_loss_slow"] = RL_REPLACEMENTS[
    "grpo_compute_loss_slow"
].replace(
    "def grpo_compute_loss",
    "def grpo_compute_loss_slow",
)


# AtomGPT's memory efficient GRPO implementation
class AtomGPTEfficientGRPO(torch.autograd.Function):
    # All AtomGPT Zoo code licensed under LGPLv3
    @staticmethod
    def forward(
        ctx,
        _new_hidden_states,
        _old_hidden_states,
        lm_head,
        _input_ids,
        _mask,
        _advantages,
        beta,
        scaler=None,
        n_chunks=1,
    ):
        def compute_loss(
            new_hidden_states,
            old_hidden_states,
            input_ids,
            mask,
            advantages,
            scaling,
        ):
            new_logits = torch.matmul(new_hidden_states, lm_head.t())
            new_logits = new_logits[
                :, :-1, :
            ]  # exclude the last logit: it corresponds to the next token pred
            old_logits = torch.matmul(old_hidden_states, lm_head.t())
            old_logits = old_logits[
                :, :-1, :
            ]  # exclude the last logit: it corresponds to the next token pred
            loss, completion_length, mean_kl = grpo_compute_loss(
                old_logits,
                new_logits,
                input_ids,
                mask,
                beta,
                advantages,
            )
            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (
                loss.detach(),
                completion_length,
                mean_kl,
            )

        pass

        device = _new_hidden_states.device
        grad_inputs = torch.empty_like(_new_hidden_states)
        accumulated_loss = torch.zeros(1, device=device)
        accumulated_completion_length = torch.zeros(1, device=device)
        accumulated_mean_kl = torch.zeros(1, device=device)

        def accumulate_chunk(
            new_hidden_states_j,
            old_hidden_states_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
        ):
            (chunk_grad_input,), (
                chunk_loss,
                (
                    unscaled_loss,
                    chunk_completion_length,
                    chunk_mean_kl,
                ),
            ) = torch.func.grad_and_value(
                compute_loss,
                argnums=(0,),
                has_aux=True,
            )(
                new_hidden_states_j,
                old_hidden_states_j,
                input_ids_j,
                mask_j,
                advantages_j,
                scaling,
            )
            accumulated_loss.add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl.add_(chunk_mean_kl)
            return chunk_grad_input

        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph=True,
            options=torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs, chunks=n_chunks, dim=0)
        new_hidden_states = torch.chunk(
            _new_hidden_states, chunks=n_chunks, dim=0
        )
        old_hidden_states = torch.chunk(
            _old_hidden_states, chunks=n_chunks, dim=0
        )
        input_ids = torch.chunk(_input_ids, chunks=n_chunks, dim=0)
        mask = torch.chunk(_mask, chunks=n_chunks, dim=0)
        advantages = torch.chunk(_advantages, chunks=n_chunks, dim=0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)

        for (
            grad_inputs_j,
            new_hidden_states_j,
            old_hidden_states_j,
            input_ids_j,
            mask_j,
            advantages_j,
        ) in zip(
            grad_inputs_chunks,
            new_hidden_states,
            old_hidden_states,
            input_ids,
            mask,
            advantages,
        ):

            mark_dynamic(new_hidden_states_j)
            mark_dynamic(old_hidden_states_j)
            mark_dynamic(input_ids_j)
            mark_dynamic(mask_j)

            grad_inputs_j.copy_(
                accumulate_chunk(
                    new_hidden_states_j,
                    old_hidden_states_j,
                    input_ids_j,
                    mask_j,
                    advantages_j,
                    scaling,
                )
            )
        pass

        grad_inputs.div_(n_chunks)
        accumulated_loss.div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl.div_(n_chunks)
        ctx.save_for_backward(grad_inputs)

        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
        )

    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl):
        (grad_input,) = ctx.saved_tensors
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    pass


pass
RL_REPLACEMENTS["AtomGPTEfficientGRPO"] = AtomGPTEfficientGRPO


def grpo_accumulated_loss(
    trainer,
    input_ids,
    logits_to_keep,
    completion_mask,
    advantages,
    n_chunks=-1,
):
    # All AtomGPT Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1:
        n_chunks = bsz
    n_chunks = factors[
        min(np.searchsorted(factors, n_chunks), len(factors) - 1)
    ]

    mixed_dtype = (
        torch.float16
        if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
        else torch.bfloat16
    )
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type="cuda", dtype=mixed_dtype):
        with torch.inference_mode(), trainer.accelerator.unwrap_model(
            trainer.model, keep_fp32_wrapper=False
        ).disable_adapter():
            old_hidden_states = trainer.model(
                input_ids=input_ids, logits_to_keep=logits_to_keep + 1
            ).logits
        pass

        new_hidden_states = trainer.model(
            input_ids=input_ids, logits_to_keep=logits_to_keep + 1
        ).logits

        loss, completion_length, mean_kl = AtomGPTEfficientGRPO.apply(
            new_hidden_states,
            old_hidden_states,
            lm_head,
            completion_input_ids,
            completion_mask,
            advantages,
            trainer.beta,
            trainer.accelerator.scaler,
            n_chunks,
        )
        return loss, completion_length, mean_kl

        # Old non efficient code path
        new_logits = torch.matmul(new_hidden_states, lm_head.t())
        new_logits = new_logits[
            :, :-1, :
        ]  # exclude the last logit: it corresponds to the next token pred
        old_logits = torch.matmul(old_hidden_states, lm_head.t())
        old_logits = old_logits[
            :, :-1, :
        ]  # exclude the last logit: it corresponds to the next token pred
        loss, completion_length, mean_kl = grpo_compute_loss(
            old_logits,
            new_logits,
            completion_input_ids,
            completion_mask,
            trainer.beta,
            advantages,
        )
        return loss, completion_length, mean_kl
    pass


pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset

RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset


def sft_trainer_fix_untrained_tokens(call_args, extra_args):
    """
    if "model" in call_args and "train_dataset" in call_args:
        fix_tokenizer = \
        "IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\\n')\n"\
        "from atomgpt.inverse_models.tokenizer_utils import fix_untrained_tokens\n"\
        "from atomgpt.inverse_models.training_utils  import fix_zero_training_loss\n"\
        "if 'tokenizer' not in locals(): tokenizer = processing_class\n"\
        "fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n"\
        "fix_zero_training_loss(model, tokenizer, train_dataset)\n"
        return fix_tokenizer
    """
    return ""


pass
RL_EXTRA_ARGS["sft_trainer"].append(sft_trainer_fix_untrained_tokens)


# Remove DPO columns which might randomnly be tokenized
def dpo_trainer_fix_columns(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_dpo = (
            "if hasattr(train_dataset, 'column_names'):\n"
            "    column_names = set(train_dataset.column_names)\n"
            "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"
            "             'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"
            "             'prompt_input_ids', 'prompt_attention_mask']\n"
            "    if all(x in column_names for x in check):\n"
            "        train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"
            "    del check, column_names\n"
        )
        return fix_dpo
    return ""


pass
RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_columns)


# Fix tokenizer double BOS
def sft_trainer_prepare_dataset(function_name, function):
    if (
        function_name != "_prepare_non_packed_dataloader"
        and function_name != "_prepare_dataset"
    ):
        return function

    fast_sft_prepare_dataset = RL_REPLACEMENTS.get("sft_prepare_dataset", None)
    if fast_sft_prepare_dataset is not None:
        params = inspect.signature(fast_sft_prepare_dataset).parameters.keys()
        params = ".*?".join(params)
        matched = re.match(
            r"[\s]{0,}def _prepare_dataset\(.*?" + params + r".*?\)",
            function,
            flags=re.MULTILINE | re.DOTALL,
        )
        if matched:
            # Use fast version!
            function = inspect.getsource(fast_sft_prepare_dataset)
            function = function.split("\n")
            function = "\n".join(" " * 4 + x for x in function)
            function = function.replace(
                "def sft_prepare_dataset", "def _prepare_dataset"
            )
            return function
        pass
    pass

    check_text = (
        "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"
        "if 'formatting_func'    not in locals(): raise RuntimeError('AtomGPT: Please file a bug report - `formatting_func` does not exist!')\n"
        "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"
        "if 'dataset_text_field' not in locals(): raise RuntimeError('AtomGPT: Please file a bug report - `dataset_text_field` does not exist!')\n"
        "test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]\n"
        "chat_template = getattr(tokenizer, 'chat_template', None)\n"
        "chat_template = '' if chat_template is None else chat_template\n"
        "has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) "
        "if getattr(tokenizer, 'bos_token', None) is not None else False\n"
        "if 'add_special_tokens' not in locals() and has_bos_token_already:\n"
        "    from functools import partial\n"
        "    tokenizer_call = tokenizer.__call__\n"
        "    tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)\n"
        "    processing_class = tokenizer\n"
        "else:\n"
        "    tokenizer_call = None\n"
        "    add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)\n"
    )

    check_text = check_text.split("\n")
    check_text = "\n".join(" " * 8 + x for x in check_text)
    check_text = check_text.rstrip() + "\n"

    # .*? matches first match. .+? matches final match.
    replacer = re.findall(
        r"def " + function_name + r"\(.*?\).*?\:\n",
        function,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(replacer) != 0:
        replacer = replacer[0]
        function = function.replace(replacer, replacer + check_text)
    pass

    # Return tokenizer's original state
    return_state = (
        "if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call\n"
    )
    function = re.sub(
        r"\n([ ]{4,})(return .*?[\s]{0,})$",
        rf"\1{return_state}\1\2",
        function,
    )
    return function


pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_prepare_dataset)


# Ignore mean_token_accuracy since it needs logits
# We override it directly with our version
def sft_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss":
        return function

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        return outputs

    pass

    function = inspect.getsource(compute_loss)
    return function


pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)


# Autocast precision for GRPO
def grpo_trainer__prepare_inputs(function_name, function):
    if function_name != "_prepare_inputs":
        return function

    if "with torch.inference_mode()" not in function:
        return function

    # Add mixed precision training
    function = function.replace(
        "with torch.inference_mode():",
        "with torch.inference_mode(), "
        "torch.amp.autocast(device_type = 'cuda', "
        "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "
        "if not torch.is_autocast_enabled('cuda') else nullcontext())"
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):",
    )

    # Disable attaching a float32 conversion hook which upcasts logits to FP32
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    return function


pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Remove _move_model_to_vllm
def grpo_trainer__move_model_to_vllm(function_name, function):
    if function_name != "_move_model_to_vllm":
        return function

    def _move_model_to_vllm(self, *args, **kwargs):
        return None

    function = inspect.getsource(_move_model_to_vllm)
    return function


pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer__get_per_token_logps(function_name, function):
    if function_name != "_get_per_token_logps":
        return function

    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "0":
            return None  # AtomGPT efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = (
                torch.float16
                if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16")
                == "fp16"
                else torch.bfloat16
            )
            if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
                self._autocast_dtype = torch.float16
        with torch.amp.autocast(
            device_type="cuda", dtype=self._autocast_dtype
        ):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            return logits
            # return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
        pass

    pass

    function = inspect.getsource(_get_per_token_logps)
    return function


pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)

grpo_compute_loss = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
AtomGPTEfficientGRPO = RL_REPLACEMENTS["AtomGPTEfficientGRPO"]
grpo_accumulated_loss = RL_REPLACEMENTS["grpo_accumulated_loss"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(AtomGPTEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_accumulated_loss))
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)


# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss":
        return function

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError(
                "The GRPOTrainer does not support returning outputs"
            )
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = None
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        input_ids = input_ids[:, -logits_to_keep:]
        if per_token_logps is not None:
            loss, completion_length, mean_kl = grpo_compute_loss_slow(
                ref_per_token_logps,
                per_token_logps,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
            )
        else:
            loss, completion_length, mean_kl = grpo_accumulated_loss(
                self,
                _input_ids,
                logits_to_keep,
                completion_mask,
                advantages,
                n_chunks=self.args.unsloth_num_chunks,
            )

        # Log the metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(
                completion_length.item()
            )
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())
        return loss

    pass

    function = inspect.getsource(compute_loss)
    return function


pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_compute_loss)


# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L356
# TRL warns if batch size is not a multiple of num_generations -> fix this.
def grpo_trainer_fix_batch_size(RLTrainer_source, RLConfig_source):
    if "divisible by the number of generations" not in RLTrainer_source:
        return ""
    if "num_generations" not in RLConfig_source:
        return ""

    check_batch_size = (
        "div = per_device_train_batch_size // num_generations\n"
        "if div * num_generations != per_device_train_batch_size:\n"
        "    print('AtomGPT: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"
        "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"
        "    per_device_train_batch_size = num_generations\n"
    )
    return check_batch_size


pass
RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_trainer_fix_batch_size)


# Add other reward function names
def grpo_trainer_metrics(RLTrainer_source, RLConfig_source):
    if "reward_funcs" not in RLTrainer_source:
        return ""

    log_metrics = (
        "if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]\n"
        "else: _reward_funcs = reward_funcs\n"
        "for reward_func in _reward_funcs:\n"
        "    try:\n"
        "        reward_func_name = reward_func.__name__\n"
        "        other_metrics.append(f'rewards/{reward_func_name}')\n"
        "    except: pass\n"
    )
    return log_metrics


pass
RL_METRICS_CHANGES["grpo_trainer"].append(grpo_trainer_metrics)
