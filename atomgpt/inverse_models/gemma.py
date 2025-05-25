from .llama import *
from ._utils import __version__

from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaModel,
    GemmaForCausalLM,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)

# For Pytorch 2.1.1
try:
    from transformers.models.gemma.modeling_gemma import (
        GemmaSdpaAttention,
        GemmaFlashAttention2,
    )
except:
    GemmaSdpaAttention = GemmaAttention
    GemmaFlashAttention2 = GemmaAttention
pass


torch_nn_functional_gelu = torch.nn.functional.gelu


def fast_geglu_inference(self, X):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    # mlp_size = self.config.intermediate_size
    # temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda")

    gate = fast_linear_forward(self.gate_proj, X)  # , out = temp[0])
    up = fast_linear_forward(self.up_proj, X)  # , out = temp[1])
    gate = torch_nn_functional_gelu(gate, approximate="tanh")
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out=up[:, :, :hd])
    return down


pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def GemmaDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
    *args,
    **kwargs,
):
    if use_cache and hasattr(
        self, "_flag_for_generation"
    ):  # past_key_value is not None:
        out_weight = torch.empty(
            self.input_layernorm.weight.shape,
            dtype=torch.float32,
            device="cuda",
        )

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(
            self.input_layernorm, hidden_states, out_weight
        )
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(
            self.post_attention_layernorm, hidden_states, out_weight
        )
        hidden_states = fast_geglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(
            self.input_layernorm, hidden_states, gemma=True
        )
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(
            self.post_attention_layernorm, hidden_states, gemma=True
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


pass


from math import sqrt as math_sqrt


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def GemmaModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask=None,
):
    out_weight = torch.empty_like(
        self.model.layers[0].input_layernorm.weight,
        dtype=torch.float32,
        device="cuda",
    )
    input_ids = input_ids[:, : self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    # 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
    # 2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
    hidden_states *= torch.tensor(
        math_sqrt(self.config.hidden_size), dtype=hidden_states.dtype
    )

    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            seq_len,
        )
    pass

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(
            decoder_layer.input_layernorm, hidden_states, out_weight
        )
        hidden_states, present_key_value = (
            LlamaAttention_fast_forward_inference(
                decoder_layer.self_attn,
                hidden_states=hidden_states,
                past_key_value=past_key_values[idx],
                position_ids=position_ids,
                attention_mask=attention_mask,
                do_prefill=not hasattr(
                    decoder_layer.self_attn, "paged_attention"
                ),
            )
        )
        hidden_states += residual

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(
            decoder_layer.post_attention_layernorm, hidden_states, out_weight
        )
        hidden_states = fast_geglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference_gemma(
        self.model.norm, hidden_states, out_weight
    )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=[],
        attentions=[],
    )


pass


# Follows line by line https://github.com/google-deepmind/gemma/blob/main/gemma/positional_embeddings.py#L45
# Formulates cos and sin differently from Llama!
class GemmaFixedRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, device=None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.max_seq_len_cached = seq_len

        # The difference is we do division explicitly instead of t * (1/x) ie we do t/x.
        freq_exponents = (2.0 / self.dim) * (
            torch.arange(
                self.dim // 2, dtype=torch.int64, device="cpu"
            ).float()
        )
        timescale = self.base**freq_exponents
        positions = torch.arange(
            self.max_seq_len_cached, device="cpu", dtype=torch.int64
        ).float()
        radians_new = positions[..., None] / timescale[None, None, :]
        radians_new = radians_new.squeeze(0)

        emb = torch.cat((radians_new, radians_new), dim=-1)
        # We must do RoPE in float32!
        cos = emb.cos().to(
            device=device, non_blocking=True
        )  # , dtype = dtype)
        sin = emb.sin().to(
            device=device, non_blocking=True
        )  # , dtype = dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype
            )

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    pass


pass


class FastGemmaModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        GemmaAttention.forward = LlamaAttention_fast_forward
        GemmaSdpaAttention.forward = LlamaAttention_fast_forward
        GemmaFlashAttention2.forward = LlamaAttention_fast_forward
        GemmaDecoderLayer.forward = GemmaDecoderLayer_fast_forward
        GemmaModel.forward = LlamaModel_fast_forward
        GemmaForCausalLM.forward = CausalLM_fast_forward(
            GemmaModel_fast_forward_inference
        )
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma.modeling_gemma

        transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding = (
            GemmaFixedRotaryEmbedding
        )
        return

    pass

    @staticmethod
    def post_patch(model):
        # Patch model for Gemma
        layers = model.model.layers

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(
            model.model.embed_tokens.weight
        )
        model.config.update({"unsloth_version": __version__})

        # We also do this for the lm_head
        lm_head = torch.nn.Linear(1, 1, bias=None)
        del lm_head.weight
        lm_head.weight = model.lm_head.weight
        lm_head.in_features = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        model.lm_head = lm_head

        # Gemma has tied weights! This means lm_head == embed_tokens
        if (
            model.model.embed_tokens.weight.data_ptr()
            != model.lm_head.weight.data_ptr()
        ):
            lm_head = torch.nn.Linear(1, 1, bias=None)
            del lm_head.weight
            lm_head.weight = model.model.embed_tokens.weight
            lm_head.in_features = lm_head.weight.shape[1]
            lm_head.out_features = lm_head.weight.shape[0]
            model.lm_head = lm_head
        pass

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = (
                        correct_dtype  # Cast to correct dtype
                    )
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
            # Downcast RoPE embedding to correct data type
            # RoPE must be done in float32 for Gemma
            # if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")) \
            #     and (module.cos_cached.dtype != correct_dtype):

            #     module.cos_cached = module.cos_cached.to(correct_dtype)
            #     module.sin_cached = module.sin_cached.to(correct_dtype)
            #     pass
            # pass
        pass

        # Add 1 to weight
        # return output * (1 + self.weight)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L89
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

        # Freeze all parameters except LoRA
        # We do this first since += 1 seems to not be liked by requires_grad = True
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        pass

        # Patch RMS Layernorm
        for name, module in model.named_modules():
            if isinstance(module, GemmaRMSNorm):
                # Must be in float32
                # https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L36
                # module = module.to(torch.float32)
                # Leave + 1 to Triton kernel itself
                # module.weight += 1.0 # return output * (1 + self.weight)
                if not hasattr(module, "variance_epsilon"):
                    module.variance_epsilon = (
                        module.eps
                    )  # Gemma doesn't use variance_epsilon
        pass

        # Clear deleted GPU items
        import gc

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model

    pass


pass
