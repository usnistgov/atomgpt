import torch
import os
from PIL import Image
import numpy as np
from jarvis.io.vasp.inputs import Poscar
from transformers import TrainerCallback, TrainerState, TrainerControl
import sys
from tqdm import tqdm
from atomgpt.inverse_models.loader import (
    FastVisionModel,
)  # FastLanguageModel for LLMs
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
import argparse

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--model_name",
    default="knc6/microscopy_gpt_llama3.2_vision_11b",
    help="Name/path of the model",
)
parser.add_argument(
    "--image_path",
    default="atomgpt/examples/inverse_model_vision/C.png",
    help="Path of STEM image",
)
parser.add_argument(
    "--formula",
    default="C",
    help="Chemical formula of system",
)


def text2atoms(response):
    response = response.split("assistant<|end_header_id|>\n")[1].split(
        ". The "
    )[0]
    tmp_atoms_array = response.strip("</s>").split("\n")
    # tmp_atoms_array= [element for element in tmp_atoms_array  if element != '']
    # print("tmp_atoms_array", tmp_atoms_array)
    lat_lengths = np.array(tmp_atoms_array[1].split(), dtype="float")
    lat_angles = np.array(tmp_atoms_array[2].split(), dtype="float")

    lat = Lattice.from_parameters(
        lat_lengths[0],
        lat_lengths[1],
        lat_lengths[2],
        lat_angles[0],
        lat_angles[1],
        lat_angles[2],
    )
    elements = []
    coords = []
    for ii, i in enumerate(tmp_atoms_array):
        if ii > 2 and ii < len(tmp_atoms_array):
            # if ii>2 and ii<len(tmp_atoms_array)-1:
            tmp = i.split()
            elements.append(tmp[0])
            coords.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])

    atoms = Atoms(
        coords=coords,
        elements=elements,
        lattice_mat=lat.lattice(),
        cartesian=False,
    )
    return atoms


def get_model(model_name="unsloth/Pixtral-12B-2409"):

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    try:
        model = FastVisionModel.get_peft_model(
            model,
            # We do NOT finetune vision & attention layers since Pixtral uses more memory!
            finetune_vision_layers=False,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=False,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=8,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=8,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

    except:
        pass

    return model, tokenizer


def relax_atoms(
    atoms=None,
    fmax=0.05,
    nsteps=150,
    constant_volume=False,
):
    from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
    from ase.optimize.fire import FIRE
    from ase.constraints import ExpCellFilter
    import time
    from jarvis.core.atoms import ase_to_atoms
    import torch._functorch.config

    torch._functorch.config.donated_buffer = False

    calculator = AlignnAtomwiseCalculator(path=default_path(), device="cpu")
    t1 = time.time()
    # if calculator is None:
    #  return atoms
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
    # TODO: Make it work with any other optimizer
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)
    en = ase_atoms.atoms.get_potential_energy()
    final_atoms = ase_to_atoms(ase_atoms.atoms)
    t2 = time.time()
    return final_atoms


def process_image_to_bw_256(pil_image, size=256):
    """
    Process a PIL image to be 256x256 and black and white (grayscale).

    Args:
        pil_image (PIL.Image): Input PIL image object

    Returns:
        PIL.Image: A 256x256 black and white version of the image
    """
    # Check if the input is a PIL Image
    if not isinstance(pil_image, Image.Image):
        raise TypeError("Input must be a PIL Image object")

    # Get original dimensions
    width, height = pil_image.size

    # Calculate scaling factor to maintain aspect ratio
    scaling_factor = min(size / width, size / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new black background image of size 256x256
    final_image = Image.new("RGB", (size, size), color="black")

    # Calculate position to paste the resized image (centered)
    paste_x = (size - new_width) // 2
    paste_y = (size - new_height) // 2

    # Paste the resized image onto the black background
    final_image.paste(resized_image, (paste_x, paste_y))

    # Convert to black and white (grayscale)
    bw_image = final_image.convert("L")

    return bw_image


def inference(
    model_name="knc6/microscopy_gpt_llama3.2_vision_11b",
    image_path="atomgpt/examples/inverse_model_vision/C.png",
    formula="C",
    # model_name, image_path="dft_2d_formula_based/JVASP-6070_1x1x1_001.jpg"
):
    model, tokenizer = get_model(model_name=model_name)
    model.eval()
    instruction = (
        "The chemical formula is "
        + str(formula)
        + ". "
        + "Generate atomic structure description with lattice lengths"
        + ", angles, coordinates, and atom types. Also predict the Miller index."
    )

    FastVisionModel.for_inference(model)  # Enable for training!
    print(f"\n🔍 Running evaluation on samples...")

    sample = {
        "id": "New",
        "messages": [
            {
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    {"type": "image", "text": None},
                ],
                "role": "user",
            },
            {
                "content": [
                    {
                        "type": "text",
                        "text": "\n3.34 3.34 25.76\n90 90 120\nTa 0.000 0.000 0.1...",
                    }
                ],
                "role": "assistant",
            },
        ],
    }

    image = Image.open(image_path)
    # print("size", image.size)
    image = process_image_to_bw_256(image)
    # print("size", image.size)

    # Format prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = tokenizer(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1554)

    generated = tokenizer.batch_decode(outputs)[0]

    # Target atoms
    # target_text = sample["messages"][1]["content"][0]["text"]

    # target_atoms = text2atoms("assistant<|end_header_id|>\n" + target_text)
    pred_atoms = text2atoms(generated)
    # print("target_atoms", target_atoms)
    # print()
    print("pred_atoms", pred_atoms)
    print()
    # rel = relax_atoms(atoms=pred_atoms)
    # print("rel_atoms", rel)
    # print()


if __name__ == "__main__":
    # run(model_name="formula_output_dir_dft_2d_unsloth/Llama-3.2-11B-Vision-Instruct/checkpoint-620")
    args = parser.parse_args(sys.argv[1:])
    inference(
        model_name=args.model_name,
        image_path=args.image_path,
        formula=args.formula,
    )
