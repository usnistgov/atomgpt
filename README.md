# AtomGPT: atomistic generative pre-trained transformer for forward and inverse materials design

Large language models (LLMs) such as [ChatGPT](https://openai.com/chatgpt/) have shown immense potential for various commercial applications, but their applicability for materials design remains underexplored. In this work, AtomGPT is introduced as a model specifically developed for materials design based on transformer architectures, demonstrating capabilities for both atomistic property prediction and structure generation tasks. This study shows that a combination of chemical and structural text descriptions can efficiently predict material properties with accuracy comparable to graph neural network models, including formation energies, electronic bandgaps from two different methods, and superconducting transition temperatures. Furthermore, AtomGPT can generate atomic structures for tasks such as designing new superconductors, with the predictions validated through density functional theory calculations. This work paves the way for leveraging LLMs in forward and inverse materials design, offering an efficient approach to the discovery and optimization of materials.


Both forward and inverse models take a config.json file as an input. Such a config file provides basic training parameters, and an `id_prop.csv` file path similar to the ALIGNN (https://github.com/usnistgov/alignn) model. See an example here: [id_prop.csv](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/examples/forward_model/id_prop.csv). 

## Forward model example (structure to property)

Forwards model are used for developing surrogate models for atomic structure to property predictions. It requires text input which can be either the raw POSCAR type files or a text description of the material. After that, we can use Google-T5/ OpenAI GPT2 etc. models with customizing langauage head for accomplishing such a task. The description of a material is generated with [ChemNLP/describer](https://github.com/usnistgov/jarvis/blob/master/jarvis/core/atoms.py#L1567) function. If you turn [`convert`](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/forward_models/forward_models.py#L277) to `False`, you can also train on bare POSCAR files.

```
python atomgpt/forward_models/forward_models.py --config_name atomgpt/examples/forward_model/config.json
```

## Inverse model example (property to structure)

Inverse models are used for generating materials given property and description such as chemical formula. Currently, we use Mistral model, but other models such as Gemma, Lllama etc. can also be easily used. After the structure generation, we can optimize the structure with ALIGNN-FF model (example [here](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb) and then subject to density functional theory calculations for a few selected candidates using JARVIS-DFT or similar workflow (tutorial for example [here](https://pages.nist.gov/jarvis/tutorials/). Note that currently, the inversely model training as well as conference requires GPUs.

```
python atomgpt/inverse_models/inverse_models.py --config_name atomgpt/examples/inverse_model/config.json
```

# Google colab/Jupyter notebook
Examples for running AtomGPT is given in the [notebook](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)
[![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

For other notebook example, see [here](https://github.com/JARVIS-Materials-Design/jarvis-tools-notebooks)

![AtomGPT layer schematic](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/data/schematic.jpeg)


# Referenes:

1. [AtomGPT: Atomistic Generative Pretrained Transformer for Forward and Inverse Materials Design](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.4c01126)
2. [ChemNLP: A Natural Language Processing based Library for Materials Chemistry Text Data](https://github.com/usnistgov/chemnlp)

<a name="contrib"></a>
How to contribute
-----------------

For detailed instructions, please see [Contribution instructions](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

<a name="corres"></a>
Correspondence
--------------------

Please report bugs as Github issues (https://github.com/usnistgov/atomgpt/issues) or email to kamal.choudhary@nist.gov.

<a name="fund"></a>
Funding support
--------------------

NIST-MGI (https://www.nist.gov/mgi) and CHIPS (https://www.nist.gov/chips)

Code of conduct
--------------------

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)
