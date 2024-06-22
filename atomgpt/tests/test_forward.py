import os
def test_forward():
  cmd='python atomgpt/forward_models/forward_models.py --config_name atomgpt/examples/forward_model/config.json'
  os.system(cmd)
