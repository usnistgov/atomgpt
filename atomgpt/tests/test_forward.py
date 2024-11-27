import os
from atomgpt.forward_models.forward_models import main


def test_forward():
    main("atomgpt/examples/forward_model/config.json")
    # cmd='python atomgpt/forward_models/forward_models.py --config_name atomgpt/examples/forward_model/config.json'
    # os.system(cmd)


#test_forward()
