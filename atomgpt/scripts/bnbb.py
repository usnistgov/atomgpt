import bitsandbytes
import bitsandbytes as bnb
get_ptr = bnb.functional.get_ptr
import ctypes
import torch
cdequantize_blockwise_fp32      = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4  = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4  = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16

