# AtomGPT: atomistic generative pre-trained transformer for forward and inverse materials design

Large language models (LLMs) such as [ChatGPT](https://openai.com/chatgpt/) have shown immense potential for various commercial applications, but their applicability for materials design remains underexplored. In this work, AtomGPT is introduced as a model specifically developed for materials design based on transformer architectures, demonstrating capabilities for both atomistic property prediction and structure generation tasks. This study shows that a combination of chemical and structural text descriptions can efficiently predict material properties with accuracy comparable to graph neural network models, including formation energies, electronic bandgaps from two different methods, and superconducting transition temperatures. Furthermore, AtomGPT can generate atomic structures for tasks such as designing new superconductors, with the predictions validated through density functional theory calculations. This work paves the way for leveraging LLMs in forward and inverse materials design, offering an efficient approach to the discovery and optimization of materials.

![AtomGPT layer schematic](https://github.com/usnistgov/atomgpt/blob/main/atomgpt/data/schematic.jpeg)

Both forward and inverse models take a config.json file as an input. Such a config file provides basic training parameters, and an `id_prop.csv` file path similar to the ALIGNN (https://github.com/usnistgov/alignn) model. See an example here: [id_prop.csv](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/examples/forward_model/id_prop.csv). 

## Installation

First create a conda environment:
Install miniconda environment from https://conda.io/miniconda.html
Based on your system requirements, you'll get a file something like 'Miniconda3-latest-XYZ'.

Now,

```
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
```
Download 32/64 bit python 3.10 miniconda exe and install (for windows)

```
conda create --name my_atomgpt python=3.10
conda activate my_atomgpt
```

```
git clone https://github.com/usnistgov/atomgpt.git
cd atomgpt
pip install -q -r dev-requirements.txt
pip install -q -e .
```

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


## DiffractGPT model example (spectral property to structure)

Inverse models are also used for generating materials given spectra/multi value property such as X-ray diffraction and description such as chemical formula. 

```
python atomgpt/inverse_models/inverse_models.py --config_name atomgpt/examples/inverse_model_multi/config.json
```

More detailed examples/case-studies would be added here soon.

# Google colab/Jupyter notebook


| Notebooks                                                                                                                                      | Google&nbsp;Colab                                                                                                                                        | Descriptions                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Forward Model training](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_forward_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_forward_example.ipynb)                                 | Example of forward model training for exfoliation energy.                                                                                                                                                                                                                                                                       |
| [Inverse Model training](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)                                 | Example of installing AtomGPT, inverse model training for 5 sample materials, using the trained model for inference, relaxing structures with ALIGNN-FF, generating a database of atomic structures.                                                                                                                                                                                                                                                                       |
 [HuggingFace AtomGPT model inference](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example_huggingface.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example_huggingface.ipynb)                            | AtomGPT Structure Generation/Inference example with a model hosted on Huggingface.                                                                                                  | 
 [Inverse Model DiffractGPT inference](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/DiffractGPT_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/DiffractGPT_example.ipynb)                                 | Example of predicting crystal structure from X-ray diffraction data.                                                                                                                                                                                                                       |                                                                                                                                  |


[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg




For similar other notebook examples, see [JARVIS-Tools-Notebook Collection](https://github.com/JARVIS-Materials-Design/jarvis-tools-notebooks)

# HuggingFace link :hugs:

https://huggingface.co/knc6


# Referenes:

1. [AtomGPT: Atomistic Generative Pretrained Transformer for Forward and Inverse Materials Design](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.4c01126)
2. [DiffractGPT: Atomic Structure Determination from X-ray Diffraction Patterns using Generative Pre-trained Transformer](https://chemrxiv.org/engage/chemrxiv/article-details/673e6c3ff9980725cf1bdf22)
3. [ChemNLP: A Natural Language Processing based Library for Materials Chemistry Text Data](https://github.com/usnistgov/chemnlp)
4. [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard)
5. [NIST-JARVIS Infrastructure](https://jarvis.nist.gov/)
6. [Unsloth AI](https://github.com/unslothai/unsloth)
   


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
