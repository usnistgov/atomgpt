import transformers
import torch
import random
import numpy as np
import os


def set_seed():
    os.environ["WANDB_ANONYMOUS"] = "must"
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(random_seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
    torch.use_deterministic_algorithms(True)


class AtomGPTPredictorLMhead(torch.nn.Module):
    def __init__(
        self,
        model_name=None,
        n_out=1,
        latent_dim=1024,
        tokenizer="",
        low_cpu_mem_usage=True,
    ):

        super(AtomGPTPredictorLMhead, self).__init__()
        self.model_name = model_name
        self.n_out = n_out
        self.latent_dim = latent_dim
        if "t5" in model_name:
            model = transformers.T5ForConditionalGeneration.from_pretrained(
                model_name
            )
        else:
            # config = GPT2Config.from_pretrained("gpt2")
            # model = GPT2Model(config)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                # load_in_8bit=False,
                # torch_dtype=torch.float16,
                # load_in_8bit=True,
                # device_map="auto"
            )
        model.resize_token_embeddings(len(tokenizer))
        self.config = model.config
        model.lm_head = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, latent_dim),
            torch.nn.Linear(latent_dim, n_out),
        )
        self.model = model

    def forward(self, input_ids):
        # outputs = self.model(input_ids)
        if "t5" in model_name:
            outputs = self.model(input_ids, decoder_input_ids=input_ids)
        else:
            outputs = self.model(input_ids)
        return outputs
