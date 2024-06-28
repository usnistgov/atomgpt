#mean_absolute_error: 64.72426134969325
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from chemnlp.utils.describe import atoms_describer
# Load JSON data
def load_data_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Preprocess data and convert text to embeddings
tag  = 'formation_energy_peratom'
def preprocess_data(dat,prop='',model='gpt2'):#, model_name):
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model)
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model)

    embeddings = []
    labels=[]
    print(model)
    for entry in dat:
     try:
        text=json.dumps(atoms_describer(atoms=Atoms.from_dict(entry['atoms']))) #Poscar(Atoms.from_dict(entry['atoms'])).to_string()
        #text = entry['text']
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        #print(output.keys(),output['past_key_values'])
        emb = output.last_hidden_state.mean(dim=1).numpy().flatten()
        #print('emb',emb,emb.shape)
        embeddings.append(emb)
        labels.append(entry[prop])
        #labels.append(entry['exfoliation_energy'])
        #embeddings.append(output.last_hidden_state.mean(dim=1).numpy())
        #embeddings.append(output.last_hidden_state.mean(dim=1).numpy())
     except Exception as exp:
           print ('exp',exp,text,len(text))
           pass

    embeddings = np.vstack(embeddings)
    #labels = np.array([entry['exfoliation_energy'] for entry in dat])
    return embeddings, labels

# Main function
def main():
    dat = data('dft_3d')
    dd=[]
    prop = 'formation_energy_peratom'#'exfoliation_energy'
    prop = 'exfoliation_energy'
    for i in dat:
     if i[prop]!='na': #[0:10]
         dd.append(i)
    #dd=dd[0:10]
    print('dd',len(dd))
    X, y = preprocess_data(dd,prop=prop)#, model_name)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit a linear regression model
    regression_model = RandomForestRegressor() #LinearRegression()
    regression_model.fit(X_train, y_train)

    # Predict using the test set
    y_pred = regression_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("mean_absolute_error:", mae)
    plt.plot(y_test, y_pred,'.')
    plt.savefig('plot.png')
    plt.close()
    #print("Mean Squared Error:", mse)

if __name__ == "__main__":
    main()
#info=[{"text":"Ram is a good boy","target":1},{"text":"Ravan is bad boy","target":0}]
#embeddings, labels = preprocess_data(info,"gpt2")
#print('embeddings',embeddings,embeddings.shape)
#print('labels',labels,labels.shape)
