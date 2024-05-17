from atomgpt.inverse_models.mistral import FastMistralModel

def test_inverse_model():
   f = FastMistralModel.from_pretrained()
   print(f)
