"""
from llama_cpp import Llama
model_kwargs = {"n_ctx":2048,"n_gpu_layers":0,"n_threads":2}
llm = Llama(model_path="xyz-unsloth.Q8_0.gguf", **model_kwargs)
generation_kwargs = {
    "max_tokens":200, # Max number of new tokens to generate
    "stop":["<|endoftext|>", "</s>"], # Text sequences to stop generation on
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}
prompt = "The meaning of life is "
res = llm(prompt, **generation_kwargs) # Res is a dictionary
print(res["choices"][0]["text"])
"""
