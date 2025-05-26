# AtomGPT : Atomistic Generative Pre-trained Transformer for Forward and Inverse Materials Design

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Forward Property Prediction Model](#forward)
* [Inverse Property Prediction Model](#inverse)
* [DiffractGPT Model](#diffract)
* [MicroscopyGPT Model](#micro)
* [Notebooks](#notebooks)
* [HuggingFace](#hug)
* [References](#refs)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>
# Introduction

Large language models (LLMs), such as [ChatGPT](https://openai.com/chatgpt/), have demonstrated immense potential across various commercial applications; however, their applicability to materials design remains underexplored. In this work, we introduce AtomGPT, a transformer-based model specifically developed for materials design, capable of both atomistic property prediction and structure generation. AtomGPT can be used with both theoretical and experimental data, primarily taking input in `.csv` format. It has been successfully applied to the computational discovery of new superconductors and semiconductors, as well as to the acceleration of experimental workflows such as X-ray diffraction and microscopy.

<a name="install"></a>
# Installation

First create a conda environment:
Install miniforge https://github.com/conda-forge/miniforge

For example: 

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Based on your system requirements, you'll get a file something like 'Miniforge3-XYZ'.

```
bash Miniforge3-$(uname)-$(uname -m).sh
```

Now, make a conda environment:

```
conda create --name my_atomgpt python=3.10 -y
conda activate my_atomgpt
```

```
git clone https://github.com/usnistgov/atomgpt.git
cd atomgpt
pip install -e .
```

<a name="forward"></a>
# Forward model example (structure to scalar property)

Forwards model are used for developing surrogate models for atomic structure to property predictions. It requires text input which can be either the raw POSCAR type files or a text description of the material. After that, we can use Google-T5/ OpenAI GPT2 etc. models with customizing langauage head for accomplishing such a task. The description of a material is generated with [ChemNLP/describer](https://github.com/usnistgov/jarvis/blob/master/jarvis/core/atoms.py#L1567) function. If you turn [`convert`](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/forward_models/forward_models.py#L277) to `False`, you can also train on bare POSCAR files.

Both forward and inverse models take a config.json file as an input. Such a config file provides basic training parameters, and an `id_prop.csv` file path similar to the ALIGNN (https://github.com/usnistgov/alignn) model. See an example here: [id_prop.csv](https://github.com/usnistgov/atomgpt/blob/develop/atomgpt/examples/forward_model/id_prop.csv). 

For training:

```
python atomgpt/forward_models/forward_models.py --config_name atomgpt/examples/forward_model/config.json
```

or use `atomgpt_forward_train` global executable.

For inference:


```
python atomgpt/forward_models/forward_predict.py --output_dir out --pred_csv atomgpt/examples/forward_model/pred_list_forward.csv
```

or use `atomgpt_forward_predict` global executable.

<a name="inverse"></a>
# Inverse model example (scalar property to structure)

Inverse models are used for generating materials given property and description such as chemical formula. Currently, we use Mistral model, but other models such as Gemma, Lllama etc. can also be easily used. After the structure generation, we can optimize the structure with ALIGNN-FF model (example [here](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb) and then subject to density functional theory calculations for a few selected candidates using JARVIS-DFT or similar workflow (tutorial for example [here](https://pages.nist.gov/jarvis/tutorials/). Note that currently, the inversely model training as well as conference requires GPUs.

For training:

```
python atomgpt/inverse_models/inverse_models.py --config_name atomgpt/examples/inverse_model/config.json
```

or use `atomgpt_inverse_train` global executable.

For inference:

```
python atomgpt/inverse_models/inverse_predict.py --output_dir outputs/checkpoint-2/  --pred_csv "atomgpt/examples/inverse_model/pred_list_inverse.csv"
```


If you want to use the huggingface model:

```
python atomgpt/inverse_models/inverse_predict.py --model_name knc6/atomgpt_mistral_tc_supercon  --pred_csv "atomgpt/examples/inverse_model/pred_list_inverse.csv"
```

or if you want infer for just one compound

```
python atomgpt/inverse_models/inverse_predict.py --model_name knc6/atomgpt_mistral_tc_supercon  --formula FeTe  --config_path atomgpt/examples/inverse_model/config.json
```

Instead of python atomgpt/inverse_models/inverse_predict.py you can also use `atomgpt_inverse_predict` global executable.

<a name="diffract"></a>
# DiffractGPT model example (spectral property to structure)

Inverse models are also used for generating materials given spectra/multi value property such as X-ray diffraction and description such as chemical formula. 

For training:

```
python atomgpt/inverse_models/inverse_models.py --config_name atomgpt/examples/inverse_model_multi/config.json
```

For inference:

```
python atomgpt/inverse_models/inverse_predict.py --model_name knc6/diffractgpt_mistral_chemical_formula --formula LaB6 --dat_path atomgpt/examples/inverse_model_multi/my_data.dat --config_path atomgpt/examples/inverse_model_multi/config.json
```

For multiple files/formats

```
python atomgpt/inverse_models/inverse_predict.py --output_dir outputs_xrd/checkpoint-2 --pred_csv atomgpt/examples/inverse_model_multi/pred_list_inverse.csv --config_path atomgpt/examples/inverse_model_multi/config.json
```

<a name="micro"></a>
# MicorscopyGPT model example (image property to structure)

Example training:

```
python atomgpt/inverse_models/inverse_vision.py --max_samples 10
```

Example inference:

```
python atomgpt/inverse_models/inverse_vision_predict.py --image_path atomgpt/examples/inverse_model_vision/FeTe.png --formula FeTe
```

More detailed examples/case-studies would be added here soon.

<a name="notebooks"></a>
# Google colab/Jupyter notebook


| Notebooks                                                                                                                                      | Google&nbsp;Colab                                                                                                                                        | Descriptions                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Forward Model training](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_forward_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_forward_example.ipynb)                                 | Example of forward model training for exfoliation energy.                                                                                                                                                                                                                                                                       |
| [Inverse Model training](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)                                 | Example of installing AtomGPT, inverse model training for 5 sample materials, using the trained model for inference, relaxing structures with ALIGNN-FF, generating a database of atomic structures.                                                                                                                                                                                                                                                                       |
 [HuggingFace AtomGPT model inference](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example_huggingface.ipynb)                                                  | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example_huggingface.ipynb)                            | AtomGPT Structure Generation/Inference example with a model hosted on Huggingface.                                                                                                  | 
 [Inverse Model DiffractGPT inference](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/DiffractGPT_example.ipynb)                                                       | [![Open in Google Colab]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/DiffractGPT_example.ipynb)                                 | Example of predicting crystal structure from X-ray diffraction data.                                                                                                                                                                                                                       |                                                                                                                                  |


[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg




For similar other notebook examples, see [JARVIS-Tools-Notebook Collection](https://github.com/JARVIS-Materials-Design/jarvis-tools-notebooks)

<a name="hug"></a>
# HuggingFace link :hugs:

https://huggingface.co/knc6

<a name="refs"></a>
# Referenes:

1. [AtomGPT: Atomistic Generative Pretrained Transformer for Forward and Inverse Materials Design](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.4c01126)
2. [DiffractGPT: Atomic Structure Determination from X-ray Diffraction Patterns using Generative Pre-trained Transformer](https://pubs.acs.org/doi/10.1021/acs.jpclett.4c03137)
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
