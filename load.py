import transformers, torch, os, json, zipfile
from finetune import get_crystal_string
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from tqdm import tqdm

root_dir = "/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard"
benchmark_file = (
    "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae.csv.zip"
)
benchmark_file = "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip"
output_dir = "out_gpt2_dft_3d_formation_energy_peratom/best_model.pt"

method = benchmark_file.split("-")[0]
task = benchmark_file.split("-")[1]
prop = benchmark_file.split("-")[2]
dataset = benchmark_file.split("-")[3]
temp = dataset + "_" + prop + ".json.zip"
temp2 = dataset + "_" + prop + ".json"
fname = os.path.join(root_dir, "benchmarks", method, task, temp)
zp = zipfile.ZipFile(fname)
bench = json.loads(zp.read(temp2))

train_ids = list(bench["train"].keys())
val_ids = list(bench["val"].keys())
test_ids = list(bench["test"].keys())

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
dft_3d = data("dft_3d")
model_name = "gpt2"
max_length = 128
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model.lm_head = torch.nn.Linear(model.config.hidden_size, 1)
model.load_state_dict(torch.load(output_dir, map_location=device))
f = open("tmp.csv", "w")
f.write("id,target,predictions\n")
for ii, i in tqdm(enumerate(dft_3d), total=len(test_ids)):
    if i["jid"] in test_ids:
        atoms = Atoms.from_dict(i["atoms"])
        text = get_crystal_string(atoms)
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        outputs = model(**inputs)
        out = (
            outputs.logits.squeeze()
            .mean(dim=-1)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        line = i["jid"] + "," + str(i[prop]) + "," + str(out) + "\n"
        f.write(line)
        # print(i['jid'],out,i[prop])
f.close()
