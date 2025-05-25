from atomgpt.inverse_models.kernels.cross_entropy_loss import (
    fast_cross_entropy_loss,
    post_patch_loss_function,
    patch_loss_functions,
)
from atomgpt.inverse_models.kernels.rms_layernorm import (
    fast_rms_layernorm,
    patch_rms_layernorm,
    unpatch_rms_layernorm,
)
from atomgpt.inverse_models.kernels.layernorm import (
    fast_layernorm,
    patch_layernorm,
)
from atomgpt.inverse_models.kernels.rope_embedding import (
    fast_rope_embedding,
    inplace_rope_embedding,
)
from atomgpt.inverse_models.kernels.swiglu import (
    swiglu_fg_kernel,
    swiglu_DWf_DW_dfg_kernel,
)
from atomgpt.inverse_models.kernels.geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_approx_backward_kernel,
)
from atomgpt.inverse_models.kernels.fast_lora import (
    get_lora_parameters,
    get_lora_parameters_bias,
    apply_lora_mlp_swiglu,
    apply_lora_mlp_geglu_exact,
    apply_lora_mlp_geglu_approx,
    apply_lora_qkv,
    apply_lora_o,
    fast_lora_forward,
)
from atomgpt.inverse_models.kernels.utils import (
    fast_dequantize,
    fast_gemv,
    QUANT_STATE,
    fast_linear_forward,
    matmul_lora,
)

from atomgpt.inverse_models.kernels.flex_attention import (
    HAS_FLEX_ATTENTION,
    slow_attention_softcapping,
    slow_inference_attention_softcapping,
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
)

# import os
# if "UNSLOTH_ZOO_IS_PRESENT" not in os.environ:
#    try:
#        print("AtomGPT: Will patch your computer to enable 2x faster free finetuning.")
#    except:
#        print("AtomGPT: Will patch your computer to enable 2x faster free finetuning.")
#    pass
# pass
# del os
