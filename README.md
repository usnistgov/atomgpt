# AtomGPT: atomistic generative pre-trained transformer for forward and inverse materials design

Large language models (LLMs) such as generative pretrained transformers (GPTs) have shown potential for various commercial applications, but their applicability for materials design remains underexplored. In this Letter, AtomGPT is introduced as a model specifically developed for materials design based on transformer architectures, demonstrating capabilities for both atomistic property prediction and structure generation. This study shows that a combination of chemical and structural text descriptions can efficiently predict material properties with accuracy comparable to graph neural network models, including formation energies, electronic bandgaps from two different methods, and superconducting transition temperatures. Furthermore, AtomGPT can generate atomic structures for tasks such as designing new superconductors, with the predictions validated through density functional theory calculations. This work paves the way for leveraging LLMs in forward and inverse materials design, offering an efficient approach to the discovery and optimization of materials.

## Forward model example

python atomgpt/forward_models/forward_models.py --config_name atomgpt/examples/forward_model/config.json

## Inverse model example

python atomgpt/inverse_models/inverse_models.py --config_name atomgpt/examples/inverse_model/config.json

#python atomgpt/examples/inverse_model/run.py

# Google colab/Jupyter notebook

[![Open in Google Colab]](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg


(Documentation development is in progress...)
