from atomgpt.inverse_models.inverse_models import run_atomgpt_inverse
import os
import atomgpt

config = os.path.join(
    atomgpt.__path__[0], "examples", "inverse_model", "config.json"
)
print("config", config)


def run_inverse():
    try:
        run_atomgpt_inverse(config)
    except Exception as exp:
        print("exp", exp)
        pass


# atomgpt/examples/inverse_model/config.json
# run_inverse()
# from atomgpt.inverse_models.mistral import FastMistralModel

# def test_inverse_model():
#   f = FastMistralModel.from_pretrained()
#   print(f)
