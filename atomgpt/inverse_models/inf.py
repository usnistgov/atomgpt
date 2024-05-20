
from jarvis.db.jsonutils import loadjson
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
from tqdm import tqdm
from jarvis.io.vasp.inputs import Poscar

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#torch.cuda.is_available = lambda : False
alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # 
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/wrk/knc6/AtomGPT/SuperCon/atomgpt_bulk_gen_formation_energy_peratom/lora_model_m", # YOUR MODEL YOU USED FOR TRAINING
    #model_name = "lora_model_mo", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto"
    
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


def text2atoms(response):
        tmp_atoms_array = response.split('\n')

        lat_lengths = np.array(tmp_atoms_array[1].split(),dtype='float')
        lat_angles = np.array(tmp_atoms_array[2].split(),dtype='float')

        lat = Lattice.from_parameters(lat_lengths[0], lat_lengths[1], lat_lengths[2], lat_angles[0], lat_angles[1], lat_angles[2])
        elements=[]
        coords=[]
        for ii,i in enumerate(tmp_atoms_array):
            if ii>2 and ii<len(tmp_atoms_array)-1:
              tmp=(i.split())
              elements.append(tmp[0])
              coords.append([float(tmp[1]),float(tmp[2]),float(tmp[3])])
        atoms = Atoms(coords=coords,elements=elements,lattice_mat=lat.lattice(),cartesian=False)
        return atoms

def gen_atoms(prompt="", max_new_tokens = 512):
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", # instruction
                #"Below is a description of a superconductor material.", # instruction
                prompt, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")


        outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True)
      
        response = tokenizer.batch_decode(outputs)#[0].split('# Output:')[1]
        #response = tokenizer.batch_decode(outputs)[0].split('# Output:')[1]
        print('response',response)
        #atoms = text2atoms(response)

        #return atoms

#'42.\n\nThe answer to the ultimate question of life, the universe and everything is 42.\n\nThis is according to Douglas Adams’s Hitchhiker’s Guide to the Galaxy.\n\nI have been thinking about this a lot lately. I am not sure if it is because I am getting older or because I am more aware of my mortality, but I have been thinking about what I want to do with my life and how I can make a difference in the world.\n\nI have always wanted to be a writer. I love writing and I love reading. I love books. I love words. I love stories. I love the way that words can transport you to another place, another time, another world. I love the way that words can make you feel things that you never thought possible.\n\nI have always wanted to be a writer. But I have also always been afraid of being a writer. I have always been afraid of'

if __name__=="__main__":
 prompt_example = "The chemical formula is MgB2 The  Tc_supercon is 6.483. The spacegroup is 12. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
 prompt_example = "The chemical formula is FeBN The  Tc_supercon is 36.483. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
 prompt_example = "The chemical formula is MgB3 The formation_energy_peratom is -0.19. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
 prompt_example = "The chemical elements are Mg and B The formation_energy_peratom is 0.19. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
 prompt_example = "The chemical formula is MgB3 The formation_energy_peratom is 1.19. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
 prompt_example = "The meaning of life is"

 gen_mat = gen_atoms(prompt=prompt_example)
 print(gen_mat)
