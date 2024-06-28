from typing import *
import pandas as pd
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2TokenizerFast, TrainingArguments, Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
import torch,json
import numpy as np
from describe import atoms_describer
from sklearn.model_selection import train_test_split
import ast
import os
# torch.cuda.is_available = lambda : False
import argparse
from multiprocessing import Pool
from tqdm import tqdm

# if output dir not exists, create it
if not os.path.exists('output'):
    os.mkdir('output')

parser = argparse.ArgumentParser()
parser.add_argument('--prop', type=str, required=True)
parser.add_argument('--modelname', type=str, default='gpt2')
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='dft_3d_2021')

args = parser.parse_args()

prop = args.prop
modelname = args.modelname
random_state = args.random_state
dataset_name = args.dataset_name

output_dir=f'output/{modelname}_{dataset_name}_{prop}'

print('imports done')
print('torch.cuda.is_available',torch.cuda.is_available())

#%%
def process_data(i):
    atoms = i['atoms']
    lattice_mat = np.round(np.array(atoms['lattice_mat']), decimals=4)
    coords = np.round(np.array(atoms['coords']), decimals=4)
    i['atoms'] = Atoms(lattice_mat=lattice_mat, elements=atoms['elements'], coords=coords, cartesian=atoms['cartesian'])
    i['atoms'] = json.dumps(atoms_describer(i['atoms']))
    return i

print('prop',prop,flush=True)

# prop = 'exfoliation_energy'
df_csv = f'{dataset_name}_described.csv'
if os.path.exists(df_csv):
    df = pd.read_csv(df_csv)[['atoms',prop]]
else:
    dat = data(dataset_name)

    pool = Pool()
    dd = []
    for result in tqdm(pool.imap(process_data, dat), total=len(dat)):
        dd.append(result)

    df = pd.DataFrame(dd)
    # df = df.set_index(df.columns[0])
    df = df.replace('na', '')
    df = df.replace('',None)
    df.to_csv(df_csv)

# replace all values of "na" with numpy nan
df = df.dropna(subset=[prop])

# random split into train and test
train_dd, test_dd = train_test_split(df, test_size=0.2, random_state=random_state)
train_ids, test_ids = train_dd.index, test_dd.index
n_train, n_test = len(train_dd), len(test_dd)
print(n_train, n_test)

# use the 'atoms' and 'prop' column to create a dataframe with 'text' and 'label' columns
print('MAD of test set',np.abs(df.loc[test_ids,prop]-df.loc[test_ids,prop].mean()).mean())

text = df['atoms']
label = df[prop].apply(lambda x: [x])

train_df = pd.DataFrame({'text':text.loc[train_ids],'label':label.loc[train_ids]})
test_df = pd.DataFrame({'text':text.loc[test_ids],'label':label.loc[test_ids]})

print('df created')


config = GPT2Config.from_pretrained(
    modelname,
    # 'gpt2-medium',
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


print('model loaded')


def tokenize(df: pd.DataFrame, tokenizer: GPT2TokenizerFast) -> List[Dict[str, Any]]:
    tokenized_df = pd.DataFrame(
        df['text'].apply(tokenizer).tolist()
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

print('tokenized')

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
    max_steps=1000,
    eval_steps=50,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=128,
    metric_for_best_model="mse",
    greater_is_better=False,
    learning_rate=5e-5,
    # going to delete all of this
    output_dir=output_dir,
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

print('trainer loaded')

trainer.train()
# save model
trainer.save_model(f'{output_dir}/final_{modelname}_{dataset_name}_{prop}')
# save scores
scores = trainer.evaluate()
with open(f'{output_dir}/scores_{modelname}_{dataset_name}_{prop}.json','w') as f:
    json.dump(scores,f)
