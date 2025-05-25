# coverage run --omit="*scripts/*,*cache/*,*__pycache__/*" -m pytest -s -v atomgpt/tests/
#  coverage report -m -i
import os
import atomgpt
import torch

# print("config", config)


# """
def test_inverse():
    if torch.cuda.is_available():
        from atomgpt.inverse_models.inverse_models import main
        from atomgpt.inverse_models.loader import (
            FastLanguageModel,
            FastVisionModel,
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="knc6/atomgpt_mistral_tc_supercon"
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit"
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-1b-pt-unsloth-bnb-4bit"
        )
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name="knc6/microscopy_gpt_llama3.2_vision_11b"
        )
        try:
            config = os.path.join(
                atomgpt.__path__[0], "examples", "inverse_model", "config.json"
            )
            main(config)
        except Exception as exp:
            print("exp", exp)
            pass


test_inverse()
# """

# atomgpt/examples/inverse_model/config.json
# run_inverse()
# from atomgpt.inverse_models.mistral import FastMistralModel

# def test_inverse_model():
#   f = FastMistralModel.from_pretrained()
#   print(f)
