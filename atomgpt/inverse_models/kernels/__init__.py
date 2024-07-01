from atomgpt.inverse_models.kernels.cross_entropy_loss import fast_cross_entropy_loss
from .rms_layernorm import fast_rms_layernorm
from .rope_embedding import fast_rope_embedding, inplace_rope_embedding
from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
from .geglu import (
	geglu_exact_forward_kernel,
	geglu_exact_backward_kernel,
	geglu_approx_forward_kernel,
	geglu_approx_backward_kernel,
)
from .fast_lora import (
	get_lora_parameters,
	apply_lora_mlp_swiglu,
	apply_lora_mlp_geglu_exact,
	apply_lora_mlp_geglu_approx,
	apply_lora_qkv,
	apply_lora_o,
)
from .utils import fast_dequantize, fast_gemv, QUANT_STATE, fast_linear_forward, matmul_lora
