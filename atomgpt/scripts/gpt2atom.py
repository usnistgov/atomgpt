from typing import *
import pandas as pd
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2TokenizerFast, TrainingArguments, Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
import torch,json
import numpy as np
torch.cuda.is_available = lambda : False
# Mondegreen fun
# 0. is the misheard version
# 1. is the real version
# regression task
dat = data('dft_3d')
dd=[]
prop = 'formation_energy_peratom'#'exfoliation_energy'
prop = 'dfpt_piezo_max_dielectric'
prop = 'exfoliation_energy'
for i in dat:
 if i[prop]!='na': #[0:10]
     atoms=i['atoms']
     lattice_mat = np.round(np.array(atoms['lattice_mat']),decimals=4)
     coords = np.round(np.array(atoms['coords']),decimals=4)
     atoms=Atoms(lattice_mat=lattice_mat,elements=atoms['elements'],coords=coords,cartesian=atoms['cartesian'],props=atoms['props'])
     i['atoms']=atoms.to_dict()
     dd.append(i)
    #dd=dd[0:10]
#dd=dd[10:22]
n_train=int(len(dd)*.8)
n_test=len(dd)-n_train
train_dd=dd[0:n_train]
test_dd=dd[-n_test:]

train_df = pd.DataFrame([
    {"text": "Money for nothin' and chips for free", "label": [0.]},
    {"text": "Money for nothin' and your chicks for free", "label": [1.]},

    {"text": "Every time you go away, you take a piece of meat with you", "label": [0.]},
    {"text": "Every time you go away take a piece of me with you", "label": [1.]},

    {"text": "Sue Lawley", "label": [0.]},
    {"text": "So lonely", "label": [1.]},

    {"text": "We built this city on sausage rolls", "label": [0.]},
    {"text": "We built this city on rock 'n' roll", "label": [1.]},

    {"text": "Saving his life from this warm sausage tea", "label": [0.]},
    {"text": "Spare him his life from this monstrosity", "label": [1.]},

    {"text": "See that girl, watch her scream, kicking the dancing queen", "label": [0.]},
    {"text": "See that girl, watch that scene, dig in the dancing queen", "label": [1.]},

    {"text": "Excuse me while I kiss this guy", "label": [0.]},
    {"text": "Excuse me while I kiss the sky", "label": [1.]},

    {"text": "Dancing queen, feel the beat from the tangerine", "label": [0.]},
    {"text": "Dancing queen, feel the beat from the tambourine", "label": [1.]},

    {"text": "Sweet dreams are made of cheese", "label": [0.]},
    {"text": "Sweet dreams are made of these", "label": [1.]},

    {"text": "Calling Jamaica", "label": [0.]},
    {"text": "Call me when you try to wake her", "label": [1.]},

    {"text": "Or should I just keep chasing penguins", "label": [0.]},
    {"text": "Or should I just keep chasing pavements", "label": [1.]},

    {"text": "All the lonely Starbucks lovers", "label": [0.]},
    {"text": "Got a long list of ex-lovers", "label": [1.]},

    {"text": "I can see clearly now, Lorraine is gone", "label": [0.]},
    {"text": "I can see clearly now, the rain is gone", "label": [1.]},

    {"text": "Gimme Gimme Gimme a man after midnight, take me to the doctors at the break of the day", "label": [0.]},
    {"text": "Gimme Gimme Gimme a man after midnight, take me through the darkness to the break of the day", "label": [1.]},

    {"text": "Poppadom Peach", "label": [0.]},
    {"text": "Papa don’t preach", "label": [1.]},

    {"text": "It doesn’t make a difference if we’re naked or not", "label": [0.]},
    {"text": "It doesn’t make a difference if we make it or not", "label": [1.]},

    {"text": "I'm farting carrots", "label": [0.]},
    {"text": "I'm 14 carat", "label": [1.]},

    {"text": "Then I saw her face, now I'm gonna leave her", "label": [0.]},
    {"text": "Then I saw her face, now I'm a believer", "label": [1.]},

    {"text": "I want to hold your ham", "label": [0.]},
    {"text": "I want to hold your hand", "label": [1.]},

    {"text": "Kicking your cat all over the place", "label": [0.]},
    {"text": "Kicking your can all over the place", "label": [1.]},
])


test_df = pd.DataFrame([
    {"text": "Blue seal in the sky with diamonds", "label": [0.]},
    {"text": "Lucy in the sky with diamonds", "label": [1.]},

    {"text": "Here we are now, in containers", "label": [0.]},
    {"text": "Here we are now, entertain us", "label": [1.]},

    {"text": "Let's pee in the corner, let's pee in the spotlight", "label": [0.]},
    {"text": "That's me in the corner, that's me in the spotlight", "label": [1.]},

    {"text": "I remove umbilicals", "label": [0.]},
    {"text": "I believe in miracles", "label": [1.]},

    {"text": "I like big butts in a can of limes", "label": [0.]},
    {"text": "I like big butts and I cannot lie", "label": [1.]},
])


mem=[]
for i in train_dd:
    info={}
    text=Poscar(Atoms.from_dict(i['atoms'])).to_string()
    #text=(Atoms.from_dict(i['atoms'])).composition.reduced_formula
    #text=json.dumps(i['atoms'])
    info['text']=text
    info['label']=[i[prop]] 
    mem.append(info)
train_df = pd.DataFrame(mem)

mem=[]
for i in test_dd:
    info={}
    text=Poscar(Atoms.from_dict(i['atoms'])).to_string()
    #text=(Atoms.from_dict(i['atoms'])).composition.reduced_formula
    #text=json.dumps(i['atoms'])
    info['text']=text
    info['label']=[i[prop]] 
    mem.append(info)
test_df = pd.DataFrame(mem)

config = GPT2Config.from_pretrained(
    "gpt2",
    pad_token_id=50256, # eos_token_id
    num_labels=1,
)
tokenizer = GPT2TokenizerFast.from_pretrained(
    config.model_type,
    padding=True,
    truncation=True,
    pad_token_id=config.pad_token_id,
    pad_token="<|endoftext|>", # eos_token
)
tokenizer.pad_token
model = GPT2ForSequenceClassification(config)

def tokenize(df: pd.DataFrame, tokenizer: GPT2TokenizerFast) -> List[Dict[str, Any]]:
    tokenized_df = pd.DataFrame(
        df.text.apply(tokenizer).tolist()
    )
    return (
        pd.merge(
            df,
            tokenized_df,
            left_index=True,
            right_index=True,
        )
        .drop(columns="text")
        .to_dict("records")
    )

train_ds = tokenize(train_df, tokenizer)
test_ds = tokenize(test_df, tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions

    return {
        "mae": mean_absolute_error(labels, predictions),
        #"mse": mean_squared_error(labels, predictions),
    }

training_args = TrainingArguments(
    report_to="none",
    evaluation_strategy="steps",
    max_steps=100,
    eval_steps=10,
    metric_for_best_model="mse",
    greater_is_better=False,
    # going to delete all of this
    output_dir="kaggle",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
