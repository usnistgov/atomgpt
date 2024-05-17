from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from atomgpt.data.dataset import get_crystal_string


def test_desc():
 atoms=Atoms.from_dict(get_jid_data(jid='JVASP-1174',dataset='dft_3d')['atoms'])
 get_crystal_string(atoms)



def get_ff():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    latent_dim = 512

    samples = [
        {
            "text": "The volume is 60.@\nGa 0 0 0&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1]],
            "energy": 1,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1]],
            "energy": 1,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2\nAl 3 3 3&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1]],
            "energy": 2,
        },
        {
            "text": "The volume is 60.@\nGa 1 1 1 \nAs 2 2 2 \nAl 3 3 3 \nXe 4 4 4&",
            "stress": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "forces": [[1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1], [1.2, 1, 1]],
            "energy": 3,
        },
    ]
    train(
        tokenizer=tokenizer,
        latent_dim=512,
        train_array=samples,
        val_array=samples,
        test_array=samples,
        include_stress=True,
        batch_size=2,
        num_epochs=10,
        pretrained_model_name="gpt2",
    )

