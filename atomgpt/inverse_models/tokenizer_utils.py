from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
import re
import os
from transformers.models.llama.modeling_llama import logger
from peft import PeftModelForCausalLM
import torch
import itertools
import collections
import numpy as np
import gc
import subprocess
import torch
import gc
import numpy as np
import itertools
import datasets
import re

"""
from unsloth_zoo.tokenizer_utils import (
    mean_of_trained_tokens,
    add_new_tokens,
    fix_untrained_tokens,
)
from unsloth_zoo.training_utils import (
    fix_zero_training_loss,
)
"""


__all__ = [
    "load_correct_tokenizer",
    "fix_sentencepiece_tokenizer",
    "check_tokenizer",
    "add_new_tokens",
    "fix_sentencepiece_gguf",
    "mean_of_trained_tokens",
    "add_new_tokens",
    "fix_untrained_tokens",
    "patch_tokenizer",
]


@torch.inference_mode
def mean_of_trained_tokens(model, eps=1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    # All AtomGPT Zoo code licensed under LGPLv3
    embedding_matrix = model.get_input_embeddings().weight.clone()
    lm_head_matrix = model.get_output_embeddings().weight.clone()

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    # if n_untrained != 0:
    #     print(
    #         f"AtomGPT: Not an error, but your model has {n_untrained} untrained tokens.\n"\
    #         "We shall set them to the mean of the other trained tokens."
    #     )
    # pass

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    # Remove bad tokens
    sum_embedding -= torch.sum(
        embedding_matrix[where_untrained], dtype=torch.float32, axis=0
    )
    sum_lm_head -= torch.sum(
        lm_head_matrix[where_untrained], dtype=torch.float32, axis=0
    )

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    return mean_embedding, mean_lm_head


pass


def add_new_tokens(
    model,
    tokenizer,
    new_tokens=[],
    method="mean",
    interpolation=0.5,
):
    """
    Smartly resizes the tokenizer and adds new tokens to the model.
    We also disregard untrained tokens by removing them from the mean calculation.
    """
    # All AtomGPT Zoo code licensed under LGPLv3
    assert isinstance(new_tokens, (list, tuple))
    assert len(new_tokens) > 0
    assert method == "mean" or method == "interpolation"
    assert interpolation >= 0 and interpolation <= 1

    # Check if tokens already exist
    overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
    if len(overlapping_tokens) != 0:
        print(
            f"AtomGPT: You're adding new_tokens = {new_tokens}\n"
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
    pass

    # Get mean of trained tokens
    # mean_embedding, mean_lm_head = fix_untrained_tokens(model)

    # Weirdly be careful reserved tokens can pop out
    mean_embedding, mean_lm_head = mean_of_trained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head = mean_lm_head.to(torch.float32)

    # Get old lengths
    old_input_embedding = model.get_input_embeddings().weight
    old_output_embedding = model.get_output_embeddings().weight
    old_input_length = old_input_embedding.shape[0]
    old_output_length = old_output_embedding.shape[0]
    old_config_size = model.config.vocab_size

    # Check for tied weights as well
    is_tied = (
        old_input_embedding.data_ptr() == old_output_embedding.data_ptr()
    ) or (model.config.tie_word_embeddings)

    # Add tokens!
    old_length = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    # Also resizes lm_head as well!
    model.resize_token_embeddings(len(tokenizer))

    # If we use interpolation, we interpolate between the mean embeddings and
    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight

    # Confirm sizes are correct
    if embedding_matrix.shape[0] != (old_input_length + len(new_tokens)):
        raise RuntimeError(
            "AtomGPT: Embedding matrix size did not get resized properly. Please file a bug report!"
        )
    if lm_head_matrix.shape[0] != (old_output_length + len(new_tokens)):
        raise RuntimeError(
            "AtomGPT: LM Head matrix size did not get resized properly. Please file a bug report!"
        )
    if model.config.vocab_size != (old_config_size + len(new_tokens)):
        raise RuntimeError(
            "AtomGPT: Model's config vocab_size did not get resized properly. Please file a bug report!"
        )
    pass

    if method == "interpolation":
        print(
            "AtomGPT: You are using interpolation to add new tokens.\n"
            f"We shall set new tokens = mean(embeddings)*{1-interpolation} + mean(new_tokens)*{interpolation}"
        )
        for j, token in enumerate(new_tokens):
            input_ids = tokenizer(token, add_special_tokens=False).input_ids
            mean_embedding_token = embedding_matrix[input_ids].mean(
                axis=0, dtype=torch.float32
            )
            mean_lm_head_token = lm_head_matrix[input_ids].mean(
                axis=0, dtype=torch.float32
            )

            # Interpolate
            mean_embedding_token = (
                mean_embedding * (1 - interpolation)
                + mean_embedding_token * interpolation
            )
            mean_lm_head_token = (
                mean_lm_head * (1 - interpolation)
                + mean_lm_head_token * interpolation
            )

            # Set the new vector
            with torch.no_grad():
                embedding_matrix[old_length + j] = mean_embedding_token
                lm_head_matrix[old_length + j] = mean_lm_head_token
        pass
    else:
        # Now set the new tokens to the mean!
        with torch.no_grad():
            embedding_matrix[old_length:] = mean_embedding
            lm_head_matrix[old_length:] = mean_lm_head
    pass

    # We set a flag to say we need to train embeddings
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    pass
    internal_model._need_to_train_embeddings = True

    # Fix up all vocab sizes
    current_model = model
    while hasattr(current_model, "model") and hasattr(current_model, "config"):
        if hasattr(current_model.config, "vocab_size"):
            current_model.config.update({"vocab_size": len(tokenizer)})
        current_model = current_model.model
    if hasattr(current_model, "model") and hasattr(current_model, "config"):
        if hasattr(current_model.config, "vocab_size"):
            current_model.config.update({"vocab_size": len(tokenizer)})
    pass

    # Must tie lm_head and embed_tokens if they are tied!
    # Otherwise error will occur on saving models ie use save_model
    if is_tied:
        model.tie_weights()

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return


pass


@torch.inference_mode
def fix_untrained_tokens(
    model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES=[], eps=1e-16
):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    print("tokenizer here", tokenizer)
    # All AtomGPT Zoo code licensed under LGPLv3
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer = (
        tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    )

    # Ignore some model checks for now
    if model.config._name_or_path in IGNORED_TOKENIZER_NAMES:
        return
    pass

    # Sometimes the sizes can be different like in vision models
    # Ie <image> is in input, but not in output
    min_size = min(embedding_matrix.shape[0], lm_head_matrix.shape[0])
    embedding_matrix = embedding_matrix[:min_size]
    lm_head_matrix = lm_head_matrix[:min_size]

    # Get untrained tokens
    indicator_untrained1 = torch.amax(embedding_matrix, axis=1) <= eps
    # Check lm_head as well

    # Does NOT work for Llama 3.1!!
    indicator_untrained2 = torch.amax(lm_head_matrix, axis=1) <= eps

    # We instead check for repeated vectors
    lm_head_where = torch.where(indicator_untrained1)[0]
    lm_head_bad = lm_head_matrix[lm_head_where.to(lm_head_matrix.device)]
    lm_head_bad = lm_head_bad.cpu().float().numpy().round(3)
    from collections import Counter

    counter = Counter()
    for row in lm_head_bad:
        counter[hash(row.data.tobytes())] += 1
    counter = Counter({k: c for k, c in counter.items() if c >= 2})

    lm_head_where = lm_head_where.cpu().numpy()
    final_bad_lm_head = []
    for j, row in enumerate(lm_head_bad):
        if hash(row.data.tobytes()) in counter:
            final_bad_lm_head.append(lm_head_where[j])
    indicator_untrained2 = indicator_untrained2 | torch.zeros_like(
        indicator_untrained2
    )
    indicator_untrained2[final_bad_lm_head] = True

    # Combine both checks
    indicator_untrained = indicator_untrained1.to(
        "cpu"
    ) & indicator_untrained2.to("cpu")

    # Remove pad token and other important token possibilities
    special_tokens = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    )
    for special_token in special_tokens:
        if hasattr(tokenizer, special_token + "_id"):
            token_id = eval(f"tokenizer.{special_token}_id")
            if (
                token_id is not None
                and token_id < indicator_untrained.shape[0]
            ):
                indicator_untrained[token_id] = False
        pass
    pass

    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # Get set and actual tokens
    where_untrained = where_untrained.tolist()
    if len(where_untrained) == 0:
        return

    # Remove untrained indices where it's longer

    where_untrained_set = frozenset(where_untrained)
    print("tokenizer tokenizer", tokenizer)
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    # Remove None items in actual_bad_tokens
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    # Check if tokenizer and training datasets have bad tokens
    if_bad_first = False
    if_bad_second = False
    # Check tokenizer's chat template for any untrained tokens
    if chat_template is not None:
        if_bad_first = any(x in chat_template for x in actual_bad_tokens)
    pass

    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check, since the code below assumes
        # an indexable dataset
        return

    # Check the first 250, last 250 input_ids
    size_dataset = len(train_dataset)
    size = min(size_dataset, 250)
    for j in range(size):
        input_ids = train_dataset[j]
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
            if_bad = any(item in where_untrained_set for item in input_ids)
            if if_bad:
                if_bad_second = True
                break
            pass
        pass
    pass

    # Check last 250
    if not if_bad_second:
        left = max(size_dataset - 250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                if_bad = any(item in where_untrained_set for item in input_ids)
                if if_bad:
                    if_bad_second = True
                    break
                pass
            pass
        pass
    pass

    # Check if bad tokens exists!
    if not if_bad_first and not if_bad_second:
        return

    # Check if lm_head / embed_token are trainable!
    bad_not_trainable = False
    if not embedding_matrix.requires_grad:
        bad_not_trainable = True
    if not lm_head_matrix.requires_grad:
        bad_not_trainable = True

    if bad_not_trainable:

        final_bad_items = []
        which_locations = []

        # Re-check the first 250, last 250 input_ids
        size_dataset = len(train_dataset)
        size = min(size_dataset, 250)
        for j in range(size):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)
                        which_locations.append(j)
            pass
        pass

        # Re-check last 250
        left = max(size_dataset - 250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)
                        which_locations.append(j)
            pass
        pass

        # If no bad tokens, possibly chat template itself has issues?
        if len(final_bad_items) == 0:
            # Recheck 2000 and last 2000 items
            size_dataset = len(train_dataset)
            size = min(size_dataset, 2000)
            for j in range(size):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)
                            which_locations.append(j)
                pass
            pass

            # Re-check last 2000
            left = max(size_dataset - 2000, 0)
            for j in range(left, size_dataset):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)
                            which_locations.append(j)
                pass
            pass
            # Most likely false signal!
            if len(final_bad_items) == 0:
                return
        pass

        token_ids = list(set(final_bad_items))
        tokens = tokenizer.decode(token_ids)
        raise ValueError(
            f"AtomGPT: Untrained tokens in rows [{list(set(which_locations))}] found.\n"
            f"The token ids are [{token_ids}] and tokens are [{tokens}].\n"
            f"The issue is the embed_tokens & lm_head not trainable, which will cause NaNs. "
            "Restart then add `embed_tokens` & `lm_head` to "
            '`FastLanguageModel.get_peft_model(target_modules = [..., "embed_tokens", "lm_head",]). `'
            "Are you using the `base` model? Instead, use the `instruct` version to silence this warning.",
        )
    pass

    # Count all the possible bad tokens
    final_counts = np.zeros(
        max(len(tokenizer), embedding_matrix.shape[0]), dtype=np.int64
    )

    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(
            itertools.chain.from_iterable(input_ids), dtype=np.int32
        )
        np.add.at(final_counts, counter, 1)

    pass
    train_dataset.map(mapping, batched=True, desc="Counting untrained tokens")

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    # Remove bad tokens
    sum_embedding -= torch.sum(
        embedding_matrix[where_untrained], dtype=torch.float32, axis=0
    )
    sum_lm_head -= torch.sum(
        lm_head_matrix[where_untrained], dtype=torch.float32, axis=0
    )

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    # Scale each to be equal to 1/max_frequency. Also set some to 0 if none seen
    scaling = final_counts[where_untrained] / max(final_counts.max(), 1)
    scaling = torch.tensor(scaling, device=mean_embedding.device).unsqueeze(1)
    mean_embedding = (
        mean_embedding.repeat(
            (
                n_untrained,
                1,
            )
        )
        * scaling
    )
    mean_lm_head = (
        mean_lm_head.repeat(
            (
                n_untrained,
                1,
            )
        )
        * scaling
    )
    where_null = scaling.ravel() == 0
    mean_embedding[where_null] = 0
    mean_lm_head[where_null] = 0

    # Set them to the mean
    print(
        "AtomGPT: Setting embed_tokens & lm_head untrained tokens to "
        "mean(trained) to counteract NaNs during training."
    )
    embedding_matrix[where_untrained] = mean_embedding.to(
        embedding_matrix.dtype
    )
    lm_head_matrix[where_untrained] = mean_lm_head.to(lm_head_matrix.dtype)

    # Clean up
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    pass
    return


pass


POSSIBLE_RESERVED_TOKENS = (
    "<|finetune_right_pad_id|>",  # Llama-3.1
    "<pad>",  # Mistral Nemo
    "<|vision_pad|>",  # Qwen 2.5
    "<|image_pad|>",  # Qwen 2.5
    "<|video_pad|>",  # Qwen 2.5
    "<|reserved",  # Llama-3
    "<|placeholder",  # Phi-3
    "[control",  # Mistral type models
    "|<EXTRA_TOKENS_",  # Molmo
    "<SPECIAL_",  # Pixtral
    "<unused",  # PaliGemma
)


@torch.inference_mode
def patch_tokenizer(model, tokenizer):
    """
    Phi3's pad_token isn't set. We set it to <|placeholder...
    Llama-3 is <|reserved...
    Llama-2 is <unk>
    Check if pad_token is not the same as eos_token otherwise the loss will ignore it!!
    Fixes https://github.com/unslothai/unsloth/issues/5
    """
    # All AtomGPT Zoo code licensed under LGPLv3
    joiner = "\1\0=+=\0\1"
    number_repetitions = 3 - 1  # Number of reserved tokens needed

    original_tokenizer = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    bad_pad_token = False
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None:
        # Check if pad_token is not the same as eos_token otherwise the loss will ignore it!!
        bad_pad_token = tokenizer.eos_token == tokenizer.pad_token
    elif hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        bad_pad_token = True
    else:
        bad_pad_token = False
    pass

    if bad_pad_token:
        # Find a better pad token
        added_tokens = [
            str(x) for x in tokenizer.added_tokens_decoder.values()
        ]
        all_added_tokens = joiner.join(added_tokens[::-1])
        all_added_tokens += joiner

        final_pad_token = None
        final_good_match = False

        for possible_reserved_token in POSSIBLE_RESERVED_TOKENS:
            possible_reserved_token = re.escape(possible_reserved_token)
            found = re.finditer(f"{possible_reserved_token}", all_added_tokens)
            first_match = None
            good_match = False
            for j, x in enumerate(found):
                if j == 0:
                    first_match = x
                if j >= number_repetitions:
                    good_match = True
                    break
                pass
            pass

            if first_match is None:
                continue

            # If it ends with |> or > etc, then set it as a good pad token!
            start = first_match.span(0)[0]
            possible_pad_token = first_match.group(0)
            end = all_added_tokens.find(joiner, start)
            first_match = all_added_tokens[start:end]

            if first_match is not None:
                good_match = possible_pad_token.endswith((">", "|>", "]", ")"))
            pass
            possible_pad_token = first_match

            # Replace current pad token if another exact match is found
            if not final_good_match and good_match:
                final_good_match = True
                final_pad_token = possible_pad_token
                break
            else:
                final_good_match = False
                final_pad_token = possible_pad_token
            pass
        pass
        possible_pad_token = final_pad_token

        # Try unk_token
        if possible_pad_token is None and hasattr(tokenizer, "unk_token"):
            possible_pad_token = tokenizer.unk_token
        pass

        # Check pad token's id must be less than vocab size
        if possible_pad_token is not None:
            check_pad_token = tokenizer(
                possible_pad_token, add_special_tokens=False
            ).input_ids
            if len(check_pad_token) != 1:
                possible_pad_token = None

            if (
                model is not None
                and hasattr(model.config, "vocab_size")
                and check_pad_token[0] >= model.config.vocab_size
            ):

                possible_pad_token = None
        pass

        if possible_pad_token is None:
            # Failure to find a good replacement!! We shall manually add one!
            new_pad_token = "<|PAD_TOKEN|>"
            while new_pad_token in tokenizer.get_vocab():
                new_pad_token = f"<{new_pad_token}>"
            pass
            possible_pad_token = new_pad_token
        pass

        name = model.config._name_or_path if model is not None else "Model"
        print(
            f"{name} does not have a padding token! Will use pad_token = {possible_pad_token}."
        )

        # Edit pad_token
        tokenizer.add_special_tokens({"pad_token": possible_pad_token})
        tokenizer.pad_token = possible_pad_token
        if model is not None:
            model.config.update({"pad_token_id": tokenizer.pad_token_id})
            if getattr(model, "generation_config") is not None:
                model.generation_config.update(
                    pad_token_id=tokenizer.pad_token_id
                )
    else:
        if model is not None:
            if model.config.pad_token_id is None:
                model.config.update({"pad_token_id": tokenizer.pad_token_id})
                if getattr(model, "generation_config") is not None:
                    model.generation_config.update(
                        pad_token_id=tokenizer.pad_token_id
                    )
        pass
    pass

    if model is not None:
        if getattr(model, "generation_config") is not None:
            if hasattr(model.config, "max_position_embeddings"):
                model.generation_config.update(
                    max_length=model.config.max_position_embeddings
                )
    pass

    return model, original_tokenizer


pass

IGNORED_TOKENIZER_CHECKING = frozenset(
    (
        "CodeLlamaTokenizerFast",
        "CodeLlamaTokenizer",
    )
)


IGNORED_TOKENIZER_NAMES = [
    # Qwen Coder did not train on tool calling. Math did!
    "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct",
]
IGNORED_TOKENIZER_NAMES = frozenset(
    [x.lower() for x in IGNORED_TOKENIZER_NAMES]
    + [x.lower() + "-bnb-4bit" for x in IGNORED_TOKENIZER_NAMES]
)
os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "\n".join(
    IGNORED_TOKENIZER_NAMES
)

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
del keynames


def try_fix_tokenizer(tokenizer, prepend=True):

    if hasattr(tokenizer, "_tokenizer"):
        converted_tokenizer = tokenizer._tokenizer
    else:
        converted_tokenizer = convert_slow_tokenizer(tokenizer)
    pass

    tokenizer_string = converted_tokenizer.to_str()

    # Llama does _apple. Sometimes this is wrong!!
    prepend_text = '{"type":"Prepend","prepend":"▁"},'
    if not prepend and prepend_text in tokenizer_string:
        tokenizer_string = tokenizer_string.replace(prepend_text, "", 1)
    pass

    dir_names = dir(tokenizer)
    # Get eos_token, bos_token etc
    token_names = [
        x for x in dir_names if x.endswith("_token") and x.count("_") == 1
    ]

    for token_name in token_names:
        token = getattr(tokenizer, token_name, None)
        if token is None:
            continue
        token_id = getattr(tokenizer, token_name + "_id", None)

        # Locate the token's id mapping in the string
        find_text = f'"id":{token_id},"content":"'
        start = tokenizer_string.find(find_text) + len(find_text)
        if start == -1:
            continue
        end = tokenizer_string.find('",', start)

        bad_token = tokenizer_string[start:end]
        # Check if token is the actual same one - if not, edit it
        if bad_token != token:
            bad_text = f'{find_text}{bad_token}",'
            good_text = f'{find_text}{token}",'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)

            # And replace vocab section
            bad_text = f'"{bad_token}":{token_id},'
            good_text = f'"{token}":{token_id},'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)
        pass
    pass

    fixed_tokenizer = converted_tokenizer.from_str(tokenizer_string)
    return fixed_tokenizer


pass


def get_sorted_dict(dictionary):
    sorted_keys = sorted(dictionary.values())
    inverted_dictionary = {value: key for key, value in dictionary.items()}

    sorted_dictionary = {}
    for key in sorted_keys:
        value = inverted_dictionary[key]
        sorted_dictionary[value] = key
    return sorted_dictionary


pass


def convert_to_fast_tokenizer(
    slow_tokenizer,
    temporary_location="_unsloth_sentencepiece_temp",
):
    is_fast = getattr(slow_tokenizer, "is_fast", False)
    if is_fast:
        return slow_tokenizer

    try:
        tokenizer_name = slow_tokenizer.__class__.__name__
        lowered_tokenizer_name = tokenizer_name.lower()
        if lowered_tokenizer_name.endswith("tokenizer"):
            class_name = lowered_tokenizer_name[: -len("tokenizer")]
            FastTokenizer = eval(
                f'__import__(f"transformers.models.{class_name}").{tokenizer_name}Fast'
            )
        else:
            FastTokenizer = PreTrainedTokenizerFast
    except:
        FastTokenizer = PreTrainedTokenizerFast
    pass

    # Get all arguments (bos_token, etc)
    docs = FastTokenizer.__doc__
    docs = docs[docs.find("Args:") :]
    args = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags=re.MULTILINE)
    args = [x for x in args if not x.endswith("_file")]

    # Also some missing maybe!
    docs = PreTrainedTokenizerFast.__doc__
    docs = docs[docs.find("Args:") :]
    args2 = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags=re.MULTILINE)
    args2 = [x for x in args2 if not x.endswith("_file")]
    args = list(set(args + args2))

    kwargs = {}
    for arg in args:
        kwargs[arg] = getattr(slow_tokenizer, arg, None)
    kwargs["tokenizer_object"] = try_fix_tokenizer(
        slow_tokenizer, prepend=True
    )
    fast_tokenizer = FastTokenizer(**kwargs)

    # Check if they're similar!
    sorted_slow_tokenizer = get_sorted_dict(slow_tokenizer.get_vocab())
    sorted_fast_tokenizer = get_sorted_dict(fast_tokenizer.get_vocab())

    check_vocab = sorted_slow_tokenizer == sorted_fast_tokenizer
    check_special = (
        slow_tokenizer.all_special_tokens == fast_tokenizer.all_special_tokens
    )

    # Failure so return slow_tokenizer
    if not check_vocab or not check_special:
        return slow_tokenizer

    # Now confirm if they match
    if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        # Maybe remove prepending of __apple?
        kwargs["tokenizer_object"] = try_fix_tokenizer(
            slow_tokenizer, prepend=False
        )
        fast_tokenizer = FastTokenizer(**kwargs)
        if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            # Failure :(
            return slow_tokenizer
        pass
    pass

    # Also tokenizer.model is missing!
    name = slow_tokenizer.name_or_path.replace("/", "_")
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass
    new_location = f"{temporary_location}/{name}"
    slow_tokenizer.save_pretrained(new_location)
    fast_tokenizer.save_pretrained(new_location)

    # Now load it!
    fast_tokenizer = AutoTokenizer.from_pretrained(new_location)
    if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        return fast_tokenizer
    return slow_tokenizer


pass


# Check Mistral chat template without BOS / EOS
mistral_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% if messages[1]['role'] == 'user' %}"
    "{{ '[INST] ' + messages[0]['content'] + ' ' + messages[1]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[2:] %}"
    "{% else %}"
    "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% endif %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)
pass

# Check Llama chat template without BOS / EOS
llama_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% if messages[1]['role'] == 'user' %}"
    "{{ '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' + messages[1]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[2:] %}"
    "{% else %}"
    "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% endif %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'].strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'].strip() + ' ' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)
pass


def assert_same_tokenization(slow_tokenizer, fast_tokenizer):
    # Get eos_token, bos_token etc
    if not hasattr(slow_tokenizer, "all_special_tokens"):
        return True
    dir_names = dir(slow_tokenizer)
    special_tokens = list(
        filter(
            None,
            (
                getattr(slow_tokenizer, x)
                for x in dir_names
                if x.endswith("_token") and x.count("_") == 1
            ),
        )
    )
    all_special_tokens = list(
        set(special_tokens + slow_tokenizer.all_special_tokens)
    )

    # Remove replacement char for false positive
    replacement_char = b"\xc3\xaf\xc2\xbf\xc2\xbd".decode("utf-8")
    all_special_tokens = [
        x for x in all_special_tokens if x != replacement_char
    ]

    # Check if chat template is enabled!
    check_chat_template1 = True
    check_chat_template2 = True
    check_chat_template3 = True

    """
    Weirdly Mistral tokenizers are actually correct??
    Ie below will actually load mistral v1 and v3 incorrectly!

    slow_chat_template = getattr(slow_tokenizer, "chat_template", None)
    fast_chat_template = getattr(fast_tokenizer, "chat_template", None)
    messages = [
        {"role": "user", "content": " What is 2+2? "},
        {"role": "assistant", "content": " It's 4. "},
    ]
    # Check the tokenizer's own chat template
    if slow_chat_template is not None and fast_chat_template is not None:
        check_chat_template1 = \
            slow_tokenizer.apply_chat_template(messages) == \
            fast_tokenizer.apply_chat_template(messages)
    pass

    # Check Mistral chat template without BOS / EOS
    slow_tokenizer.chat_template = mistral_template
    fast_tokenizer.chat_template = mistral_template
    check_chat_template2 = \
        slow_tokenizer.apply_chat_template(messages) == \
        fast_tokenizer.apply_chat_template(messages)
    pass

    # Check Llama chat template without BOS / EOS
    slow_tokenizer.chat_template = llama_template
    fast_tokenizer.chat_template = llama_template
    check_chat_template3 = \
        slow_tokenizer.apply_chat_template(messages) == \
        fast_tokenizer.apply_chat_template(messages)
    pass

    # Combine them all and revert chat templates
    slow_tokenizer.chat_template = slow_chat_template
    fast_tokenizer.chat_template = fast_chat_template
    """
    check_chat_template = (
        check_chat_template1 and check_chat_template2 and check_chat_template3
    )

    # Try special tokens
    try:
        string = (
            "\n".join(all_special_tokens)
            + "A quick brown fox jumps over the lazy dog!!\n\nHi</s>\n\n"
            + "".join(all_special_tokens)
        )
        check_special_tokens = (
            slow_tokenizer(string).input_ids
            == fast_tokenizer(string).input_ids
        )

        return check_chat_template and check_special_tokens
    except:
        # For eg see https://github.com/unslothai/unsloth/issues/292
        # Sometimes tokenizer has weird tokens, causing a combined tokenization to fail.
        # [TODO] We temporarily disable this for CodeLlama tokenizers
        if (
            slow_tokenizer.__repr__().split("(", 1)[0]
            in IGNORED_TOKENIZER_CHECKING
        ):
            return check_chat_template
        else:
            return False
    pass


pass


def fix_sentencepiece_tokenizer(
    old_tokenizer,
    new_tokenizer,
    token_mapping,
    temporary_location="_unsloth_sentencepiece_temp",
):
    # From https://github.com/google/sentencepiece/issues/121
    # We need to manually edit the sentencepiece tokenizer!
    from transformers.utils import sentencepiece_model_pb2

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass

    # Check if tokenizer.model exists
    if not os.path.isfile(f"{temporary_location}/tokenizer.model"):
        return new_tokenizer
    pass

    # First save the old tokenizer
    old_tokenizer.save_pretrained(temporary_location)

    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    tokenizer_file.ParseFromString(
        open(f"{temporary_location}/tokenizer.model", "rb").read()
    )

    # Now save the new tokenizer
    new_tokenizer.save_pretrained(temporary_location)

    # Now correct the old tokenizer's .model file
    for old_token, new_token in token_mapping.items():
        ids = old_tokenizer([old_token], add_special_tokens=False).input_ids
        ids = ids[0]
        if len(ids) != 1:
            # Skip this token!
            print(
                f"Skip mapping {old_token} to {new_token} since {new_token} is already in the tokenizer!"
            )
            continue
        pass
        ids = ids[0]
        # [TODO] Hack for Starling - try except
        try:
            tokenizer_piece = tokenizer_file.pieces[ids]
        except:
            continue
        assert tokenizer_piece.piece == old_token
        tokenizer_piece.piece = new_token
    pass

    # And now write it
    with open(f"{temporary_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())
    pass

    # And load it!
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        temporary_location,
        eos_token=new_tokenizer.eos_token,
        pad_token=new_tokenizer.pad_token,
    )
    return tokenizer


pass


def fix_sentencepiece_gguf(saved_location):
    """
    Fixes sentencepiece tokenizers which did not extend the vocabulary with
    user defined tokens.
    Inspiration from https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py
    """
    from copy import deepcopy
    from transformers.utils import sentencepiece_model_pb2
    import json
    from enum import IntEnum

    class SentencePieceTokenTypes(IntEnum):
        NORMAL = 1
        UNKNOWN = 2
        CONTROL = 3
        USER_DEFINED = 4
        UNUSED = 5
        BYTE = 6

    pass

    # Load tokenizer.model
    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    if not os.path.isfile(f"{saved_location}/tokenizer.model"):
        return
    tokenizer_file.ParseFromString(
        open(f"{saved_location}/tokenizer.model", "rb").read()
    )
    sentence_piece_size = len(tokenizer_file.pieces)

    # Load added_tokens_json
    if not os.path.isfile(f"{saved_location}/added_tokens.json"):
        return
    with open(
        f"{saved_location}/added_tokens.json", "r", encoding="utf-8"
    ) as file:
        added_tokens_json = json.load(file)
    pass
    if len(added_tokens_json) == 0:
        return

    added_tokens_json = dict(
        sorted(added_tokens_json.items(), key=lambda item: item[1])
    )
    new_size = sentence_piece_size + len(added_tokens_json)

    # Confirm added_tokens_json is correct
    added_tokens_ids = np.array(list(added_tokens_json.values()))
    diff = np.diff(added_tokens_ids)
    if diff.min() != 1 or diff.max() != 1:
        return
    if added_tokens_ids.min() != sentence_piece_size:
        return

    # Edit sentence piece tokens with added_tokens_json
    logger.warning(
        f"AtomGPT: Extending {saved_location}/tokenizer.model with added_tokens.json.\n"
        f"Originally tokenizer.model is of size ({sentence_piece_size}).\n"
        f"But we need to extend to sentencepiece vocab size ({new_size})."
    )
    new_tokens = deepcopy(tokenizer_file.pieces[-len(added_tokens_ids) :])
    for new_token, added_token in zip(new_tokens, added_tokens_json.keys()):
        new_token.piece = added_token.encode("utf-8")
        new_token.score = -1000.0
        new_token.type = SentencePieceTokenTypes.USER_DEFINED
    pass

    tokenizer_file.pieces.extend(new_tokens)

    with open(f"{saved_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())
    pass

    # Add padding tokens
    # actual_vocab_size = model.config.vocab_size
    # padding = actual_vocab_size - len(tokenizer_file.pieces)
    return


pass


def _load_correct_tokenizer(
    tokenizer_name,
    model_max_length=None,
    padding_side="right",
    token=None,
    trust_remote_code=False,
    cache_dir="huggingface_tokenizers_cache",
    fix_tokenizer=True,
):
    if IS_COLAB_ENVIRONMENT:
        cache_dir = cache_dir
    elif IS_KAGGLE_ENVIRONMENT:
        # /tmp of Kaggle seems has a 80GB limit!
        # Let's utilize them
        cache_dir = os.path.join(KAGGLE_TMP, cache_dir)
    else:
        cache_dir = None
    pass

    # Try loading the slow tokenizer. If it fails, then try Fast only
    # Mainly to solve Deepseek models with no tokenizer.model file
    slow_tokenizer = None
    try:
        slow_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length=model_max_length,
            padding_side=padding_side,
            token=token,
            trust_remote_code=trust_remote_code,
            # Cannot just use use_fast = False as per https://twitter.com/danielhanchen/status/1789659394302718373
            use_fast=False,
            legacy=False,
            from_slow=True,
            cache_dir=cache_dir,
        )
    except:
        slow_tokenizer = None
        # print(
        #     f"AtomGPT: {tokenizer_name} has no tokenizer.model file.\n"\
        #     "Just informing you about this - this is not a critical error."
        # )
    pass
    # Unsure why this occurs!
    if type(slow_tokenizer) is bool:
        slow_tokenizer = None

    fast_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length=model_max_length,
        padding_side=padding_side,
        token=token,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    if not fix_tokenizer or tokenizer_name in IGNORED_TOKENIZER_NAMES:
        return fast_tokenizer
    # Ignore Mistral ones - they're a bit weird to handle!
    elif "mistral" in tokenizer_name.lower():
        return fast_tokenizer
    # Ignore Phi-4 ones as well
    elif "phi-4" in tokenizer_name.lower():
        return fast_tokenizer
    elif slow_tokenizer is not None:
        if hasattr(fast_tokenizer, "add_bos_token") and hasattr(
            slow_tokenizer, "add_bos_token"
        ):
            fast_tokenizer.add_bos_token = slow_tokenizer.add_bos_token
        if hasattr(fast_tokenizer, "add_eos_token") and hasattr(
            slow_tokenizer, "add_eos_token"
        ):
            fast_tokenizer.add_eos_token = slow_tokenizer.add_eos_token

        # Confirm if slow and fast are equivalent!
        if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            return fast_tokenizer
        else:
            logger.warning(
                f"AtomGPT: Will load {tokenizer_name} as a legacy tokenizer."
            )
            return convert_to_fast_tokenizer(slow_tokenizer)
        pass
    else:
        return fast_tokenizer
    pass


pass


def load_correct_tokenizer(
    tokenizer_name,
    model_max_length=None,
    padding_side="right",
    token=None,
    trust_remote_code=False,
    cache_dir="huggingface_tokenizers_cache",
    fix_tokenizer=True,
):
    tokenizer = _load_correct_tokenizer(
        tokenizer_name=tokenizer_name,
        model_max_length=model_max_length,
        padding_side=padding_side,
        token=token,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        fix_tokenizer=fix_tokenizer,
    )

    ### 1. Fixup tokenizer's chat_template
    old_chat_template = getattr(tokenizer, "chat_template", None)

    # Ignore mistral type models since they don't have a add_generation_prompt
    if "mistral" in str(getattr(tokenizer, "name_or_path", "")).lower():
        chat_template = old_chat_template

    # Also check Llama-2 old style models
    elif (
        old_chat_template is not None
        and "[/INST]" in old_chat_template
        and "[INST]" in old_chat_template
        and "bos_token" in old_chat_template
        and "eos_token" in old_chat_template
    ):

        chat_template = old_chat_template

    else:
        chat_template = fix_chat_template(tokenizer)
        if old_chat_template is not None and chat_template is None:
            raise RuntimeError(
                "AtomGPT: Fixing chat template failed - please file a report immediately!"
            )
        pass
    pass

    tokenizer.chat_template = chat_template
    return tokenizer


pass


def _find_end_position(template, endfor, endif):
    where_endfor = template.find(endfor)
    where_endif = template.find(endif)
    if where_endfor == where_endif == -1:
        return None
    elif where_endfor > where_endif:
        return endfor
    else:
        return endif
    pass


pass


def _fix_chat_template(chat_template):
    endfor = "{% endfor %}"
    endif = "{% endif %}"
    chosen_end = _find_end_position(chat_template, endfor, endif)
    if chosen_end is None:
        endfor = "{%- endfor %}"
        endif = "{%- endif %}"
        chosen_end = _find_end_position(chat_template, endfor, endif)
    if chosen_end is None:
        return chat_template

    where = chat_template.find(chosen_end)

    after_endfor = chat_template[where + len(chosen_end) :]

    dash = "-" if chosen_end.startswith("{%-") else ""

    if (
        "{%" + dash + " if" not in after_endfor
        and "{%" + dash + " set " not in after_endfor
        and after_endfor.startswith("{{")
        and after_endfor.endswith("}}")
        and after_endfor.count("{{") == 1
        and after_endfor.count("}}") == 1
    ):

        after_endfor = (
            "{%" + dash + " if add_generation_prompt %}" + after_endfor + endif
        )

        chat_template = chat_template[: where + len(chosen_end)] + after_endfor
    pass
    return chat_template


pass


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return None

    ### 1. Check if add_generation_prompt works
    # Check for ShareGPT style first
    is_sharegpt = None
    try:
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        is_sharegpt = False
    except:
        try:
            messages = [
                {"from": "human", "value": "Who are you?"},
            ]
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            is_sharegpt = True
        except:
            is_sharegpt = None
        pass
    pass

    # Not ShareGPT or HF style - just return
    if is_sharegpt is None:
        return chat_template

    # Tokenize
    messages = [
        (
            {"role": "user", "content": "Who are you?"}
            if not is_sharegpt
            else {"from": "human", "value": "Who are you?"}
        )
    ]
    no = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    yes = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    if no == yes:
        # SAME?! That's not good! We check for add_generation_prompt
        if (
            "{% if add_generation_prompt %}" not in chat_template
            and "{%- if add_generation_prompt %}" not in chat_template
        ):
            # Try fixing it by adding it
            new_chat_template = _fix_chat_template(chat_template)
            if (
                "{% if add_generation_prompt %}" not in new_chat_template
                and "{%- if add_generation_prompt %}" not in new_chat_template
            ):
                raise RuntimeError(
                    f"AtomGPT: The tokenizer `{tokenizer.name_or_path}`\n"
                    "does not have a {% if add_generation_prompt %} for generation purposes.\n"
                    f"Please file a bug report to the maintainers of `{tokenizer.name_or_path}` - thanks!"
                )
            else:
                logger.warning_once(
                    "AtomGPT: We successfully patched the tokenizer to add a {% if add_generation_prompt %} to the chat_template.\n"
                    f"This is not a bug, but please notify the maintainers of `{tokenizer.name_or_path}` - thanks!"
                )
                chat_template = new_chat_template
            pass
        else:
            raise RuntimeError(
                f"AtomGPT: The tokenizer `{tokenizer.name_or_path}`\n"
                "has a {% if add_generation_prompt %} for generation purposes, but wasn't provided correctly.\n"
                "Please file a bug report immediately - thanks!"
            )
        pass
    pass
    return chat_template


pass


def check_tokenizer(
    model,
    tokenizer,
    model_name="unsloth/llama-2-7b-bnb-4bit",
    model_max_length=4096,
    padding_side="right",
    token=None,
    _reload=True,
):
    # Checks tokenizer for out of bounds ids.
    # Mainly a fix for https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
    # where <sep> had token id=32002.
    # See https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
    # Seems like the Fast tokenizer in Rust breaks things!

    # We ignore some of them!
    if tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
        return tokenizer
    pass

    max_embedding_size = model.model.embed_tokens.weight.shape[0]
    added_tokens_fast = tokenizer.added_tokens_decoder
    added_tokens_fast = {
        index: str(value) for index, value in added_tokens_fast.items()
    }
    sorted_keys = sorted(added_tokens_fast)
    added_tokens_fast = {key: added_tokens_fast[key] for key in sorted_keys}

    for j, index in enumerate(added_tokens_fast.keys()):
        if index >= max_embedding_size:
            bad_indices = list(added_tokens_fast.keys())[j:]
            bad_tokens = list(added_tokens_fast.values())[j:]
            if not _reload:
                # Try removing the token
                added_tokens = [
                    str(x) for x in tokenizer.added_tokens_decoder.values()
                ]
                special_tokens = tokenizer.special_tokens_map
                import itertools

                special_tokens = frozenset(
                    itertools.chain.from_iterable(
                        [x] if type(x) is str else x
                        for x in special_tokens.values()
                    )
                )
                can_be_removed1 = [
                    x for x in bad_tokens if x not in special_tokens
                ]
                can_be_removed2 = [
                    x
                    for x in can_be_removed1
                    if x in tokenizer._added_tokens_encoder.keys()
                ]

                # Check of extra tokens can in fact we removed!
                can_be_removed = (
                    len(can_be_removed1) == len(bad_tokens)
                ) and (len(can_be_removed2) == len(bad_tokens))

                # Check if sep_token or other generic types
                remove_generic = False
                try_mapper = []
                if not can_be_removed:
                    names = dir(tokenizer)
                    names = (
                        x
                        for x in names
                        if x.endswith("_token") and x.count("_") == 1
                    )
                    generic_tokens = [
                        (x, getattr(tokenizer, x, None)) for x in names
                    ]

                    try_removal = []
                    for token in bad_tokens:
                        for name_token, check_token in generic_tokens:
                            if check_token == token:
                                try_removal.append(token)
                                try_mapper.append(name_token)
                            pass
                        pass
                    pass

                    # Recheck!
                    can_be_removed = len(try_removal) == len(bad_tokens)
                    if can_be_removed:
                        remove_generic = True
                    can_be_removed1 = bad_tokens
                pass

                if can_be_removed:
                    # Yes it can be fixed!
                    for j, bad_token in enumerate(can_be_removed1):
                        remove_id = tokenizer._added_tokens_encoder[bad_token]
                        del tokenizer._added_tokens_decoder[remove_id]
                        del tokenizer._added_tokens_encoder[bad_token]

                        if remove_generic and (try_removal[j] == bad_token):
                            # Remove sep token for example
                            setattr(tokenizer, try_mapper[j], None)
                            setattr(tokenizer, try_mapper[j] + "_id", None)
                        pass
                    pass
                    # Confirm 1 more time!
                    if (
                        max(tokenizer.added_tokens_decoder.keys())
                        < max_embedding_size
                    ):
                        logger.warning_once(
                            f"AtomGPT loaded a broken tokenizer `{model_name}`, but managed to repair it!\n"
                            f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                            "We removed these bad tokens. If you think this is incorrect, fix your tokenizer first."
                        )
                        return convert_to_fast_tokenizer(tokenizer)
                    pass
                pass

                # :( Failure
                raise RuntimeError(
                    f"AtomGPT tried to load `{model_name}`, but cannot succeed.\n"
                    f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                    f"Fix your tokenizer since it'll perform out of bounds memory accesses."
                )
            pass

            if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
                cache_dir = "huggingface_tokenizers_cache"
            else:
                cache_dir = None
            pass

            # Sometimes slow tokenizer does not work like Deepseek
            try:
                # Try slow tokenizer which can fix things!
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_length=model_max_length,
                    padding_side=padding_side,
                    token=token,
                    # Cannot just use use_fast = False as per https://twitter.com/danielhanchen/status/1789659394302718373
                    use_fast=False,
                    legacy=False,
                    from_slow=True,
                    cache_dir=cache_dir,
                )
                return check_tokenizer(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_max_length=model_max_length,
                    padding_side=padding_side,
                    token=token,
                    _reload=False,
                )
                break
            except:
                # Tokenizer has out of bounds issues and we can't
                # load the slow tokenizer version :(
                logger.warning_once(
                    "AtomGPT: Tokenizer is most likely buggy, and AtomGPT failed to repair it.\n"
                    "It will still work, but beware of out of bounds memory accesses.\n"
                    "Please file an issue on the model owner's repo about this issue."
                )
                return tokenizer
            pass
        pass
    pass
    return convert_to_fast_tokenizer(tokenizer)


pass


import inspect
from inspect import getsource
import trl
import trl.trainer.sft_trainer
from trl.trainer.sft_trainer import *
from transformers.trainer import *

try:
    from trl.trainer.sft_trainer import neftune_post_forward_hook
except:

    def neftune_post_forward_hook(module, input, output):
        """
        Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
        torch.nn.Embedding layers. This method is slightly adapted from the original source code
        that can be found here: https://github.com/neelsjain/NEFTune

        Simply add it to your model as follows:
        ```python
        model = ...
        model.embed_tokens.neftune_noise_alpha = 0.1
        model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
        ```

        Args:
            module (`torch.nn.Module`):
                The embedding module where the hook is attached. Note that you need to set
                `module.neftune_noise_alpha` to the desired noise alpha value.
            input (`torch.Tensor`):
                The input tensor to the model.
            output (`torch.Tensor`):
                The output tensor of the model (i.e. the embeddings).
        """
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(
                -mag_norm, mag_norm
            )
        return output

    pass
pass


def patch_sft_trainer_tokenizer():
    """
    Patches the trainer with changes
    """
    try:
        sft_trainer = eval(f"trl.trainer.sft_trainer.SFTTrainer")
    except:
        return
    all_imports = dir(trl.trainer.sft_trainer)

    for (
        function_name,
        replacer,
    ) in (
        # ("_prepare_non_packed_dataloader", "def tokenize(element):",),
        (
            "_prepare_non_packed_dataloader",
            None,
        ),
        (
            "_prepare_dataset",
            None,
        ),
        # ("_prepare_packed_dataloader", "if dataset_text_field is not None",),
    ):
        if not hasattr(sft_trainer, function_name):
            continue

        function = getsource(eval(f"sft_trainer.{function_name}"))
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        check_text = (
            "\n"
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
            "    tokenizer = partial(tokenizer, add_special_tokens = False)\n"
            "    processing_class = tokenizer\n"
            "else:\n"
            "    add_special_tokens = False if has_bos_token_already else add_special_tokens\n\n"
        )

        check_text = check_text.split("\n")
        check_text = "\n".join(" " * where + x for x in check_text)
        check_text = check_text.rstrip() + "\n"

        if replacer is None:
            # .*? matches first match. .+? matches final match.
            replacer = re.findall(
                f"def {function_name}" + r"\(.*?\).*?\:\n",
                function,
                flags=re.MULTILINE | re.DOTALL,
            )
            if len(replacer) == 0:
                continue
            replacer = replacer[0]
            function = function.replace(replacer, replacer + check_text)
        else:
            function = function.replace(replacer, check_text + replacer)
        pass

        x = [x for x in all_imports if x in function]
        exec(f"from trl.trainer.sft_trainer import ({','.join(x)})", locals())
        exec(function, locals(), globals())
        exec(
            f"trl.trainer.sft_trainer.SFTTrainer.{function_name} = {function_name}",
            globals(),
        )
    pass

    # Patch train with fix_untrained_tokens
    for path_to_trainer in (
        "sft_trainer.SFTTrainer",
        "dpo_trainer.DPOTrainer",
        "kto_trainer.KTOTrainer",
    ):

        function_name, replacer = (
            "train",
            "if resume_from_checkpoint is False:",
        )
        function = getsource(
            eval(f"trl.trainer.{path_to_trainer}.{function_name}")
        )
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        check_text = (
            "\n"
            "import subprocess, re, gc, numpy as np\n"
            "a = np.array([0,])\n"
            "try:\n"
            "    a = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell = True)\n"
            "    a = re.findall(rb'([\\d]{1,})[\\s]{1,}M', a)\n"
            "    a = np.array([int(x.decode('utf-8'))/1024 for x in a])\n"
            "except:\n"
            "    if not torch.cuda.is_available():\n"
            "        raise RuntimeError('AtomGPT: We do not support AMD / Intel machines yet - it is a work in progress!')\n"
            "if ((a - PRE_CHECK) >= 1).sum() > 1:\n"
            "    raise RuntimeError('AtomGPT currently does not support multi GPU setups - but we are working on it!')\n"
            "for _ in range(3):\n"
            "    gc.collect()\n"
            "    torch.cuda.empty_cache()\n"
            "pass\n"
            "\n"
            "tokenizer = self.processing_class if hasattr(self, 'processing_class') else self.tokenizer\n"
            "fix_untrained_tokens(self.model, tokenizer, self.train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n\n"
            "fix_zero_training_loss(self.model, tokenizer, self.train_dataset)\n\n"
        )

        # Warn on gradient accumulation steps if it's used
        check_text += (
            "\n"
            "try:\n"
            "    gradient_accumulation_steps = self.args.gradient_accumulation_steps\n"
            "    if type(gradient_accumulation_steps) is int and gradient_accumulation_steps > 1:\n"
            "        from transformers import __version__ as transformers_version\n"
            "        from packaging.version import Version\n"
            "        if Version(transformers_version) <= Version('4.45.2'):\n"
            "            print('**** AtomGPT: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and AtomGPT!\\n'\\\n"
            "                  '`pip install --upgrade --no-cache-dir --no-deps unsloth transformers git+https://github.com/huggingface/trl.git`')\n"
            "except:\n"
            "    pass\n"
            "\n\n"
        )

        # Add NEFTune since it doesn't seem to work?? We need to manually inject it
        check_text += (
            "\n"
            "if hasattr(self, 'neftune_hook_handle'):\n"
            "    self.neftune_hook_handle.remove()\n"
            "    if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle\n"
            "\n"
            "if getattr(self, 'neftune_noise_alpha', None) is not None:\n"
            "    self.model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha\n"
            "    self.neftune_hook_handle = self.model.get_input_embeddings().register_forward_hook(neftune_post_forward_hook)\n"
            "pass\n"
            "\n"
        )

        # Also DPO weirdly tokenizes non numeric columns? Delete them!
        check_text += (
            "\n"
            "if hasattr(self.train_dataset, 'column_names'):\n"
            "    column_names = set(self.train_dataset.column_names)\n"
            "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"
            "        'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"
            "        'prompt_input_ids', 'prompt_attention_mask']\n"
            "    if all(x in column_names for x in check):\n"
            "        self.train_dataset = self.train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"
            "    del check, column_names\n"
            "\n"
        )

        check_text = check_text.split("\n")
        check_text = "\n".join(" " * where + x for x in check_text)

        function = function.replace(replacer, check_text + replacer)
        exec(function, globals())

        exec(
            f"trl.trainer.{path_to_trainer}.{function_name} = {function_name}",
            globals(),
        )
    pass


pass

# Finally patch TRL tokenizer things -> moved to RL
# patch_sft_trainer_tokenizer()
