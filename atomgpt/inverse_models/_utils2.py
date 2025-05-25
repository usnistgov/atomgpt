import torch
from typing import Union, Optional, List, Any, Callable, Tuple
from platform import system as platform_system

platform_system = platform_system()
import numpy as np
import contextlib
import re
import warnings, subprocess, re, inspect, psutil, os, math

# from unsloth_zoo.utils import Version

# from unsloth_zoo.tokenizer_utils import (
#    patch_tokenizer as _patch_tokenizer,
# )

from packaging.version import Version as TrueVersion
import torch


def Version(version):
    # All AtomGPT Zoo code licensed under LGPLv3
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack

        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"AtomGPT: Could not get version for `{version}`\n"
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass


pass


def is_main_process():
    is_initialized = torch.distributed.is_initialized()
    return (not is_initialized) or (
        is_initialized and torch.distributed.get_rank() == 0
    )


pass


def is_distributed():
    return torch.distributed.is_initialized()


pass


def distributed_function(n=1, function=None, *args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            object_list = function(*args, **kwargs)
            if n == 1:
                object_list = [object_list]
        else:
            object_list = [None for _ in range(n)]
        # broadcast_object_list auto blocks so no need for barrier
        torch.distributed.broadcast_object_list(
            object_list, src=0, device="cpu"
        )
        if n == 1:
            result = object_list[0]
    else:
        result = function(*args, **kwargs)
    return result


pass


def _get_dtype(dtype):
    __DTYPE_MAP = {
        "float32": torch.float32,
        torch.float32: torch.float32,
        "float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
    }
    if dtype is None or dtype == None:
        return None
    elif dtype in __DTYPE_MAP:
        return __DTYPE_MAP[dtype]
    else:
        print(f"AtomGPT: {dtype} is not recognized, so we'll default to None")
        return None
    pass


pass

# Ignore logging messages
