from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
import numpy as np


def text2atoms(response):
    # print("response", response)
    response = response.strip("</s>")
    if response.startswith("\n"):
        subs = 0
    else:
        subs = 1
    tmp_atoms_array = response.split("\n")
    lat_lengths = np.array(tmp_atoms_array[1 - subs].split(), dtype="float")
    lat_angles = np.array(tmp_atoms_array[2 - subs].split(), dtype="float")
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
        if ii > 2 - subs and ii < len(tmp_atoms_array) - subs:

            tmp = i.split()
            if len(tmp) > 2:
                elements.append(tmp[0])
                coords.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
    atoms = Atoms(
        coords=coords,
        elements=elements,
        lattice_mat=lat.lattice(),
        cartesian=False,
    )
    return atoms


def text2atoms_old(response):
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


def gen_atoms(
    prompt="",
    model=None,
    tokenizer="",
    max_new_tokens=1024,
    alpaca_prompt="",
    instruction="",
):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,
                prompt,  # input
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )
    response = tokenizer.batch_decode(outputs)[0].split("# Output:")[1]
    atoms = None
    try:
        atoms = text2atoms(response)
    except Exception as exp:

        print(exp)
        pass
    return atoms


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    # crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str


def gen_atoms_batch(
    prompts=[], max_new_tokens=2048, model=None, tokenizer=None
):
    # Ensure CUDA is available for GPU usage
    if not prompts:
        raise ValueError("Prompts list cannot be empty.")

    # Tokenize all prompts at once
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Below is a description of a material.",  # instruction
                prompt,  # input prompt
                "",  # leave blank for generation
            )
            for prompt in prompts
        ],
        return_tensors="pt",
        padding=True,  # Pad inputs to the same length
        truncation=True,  # Ensure long prompts are truncated
    ).to("cuda")

    # Generate outputs in batch
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )

    # Decode and convert the generated texts into atoms
    responses = [
        tokenizer.decode(output, skip_special_tokens=True)
        .split("# Output:")[1]
        .strip("</s>")
        for output in outputs
    ]

    # Convert each response to atoms
    atoms_list = [text2atoms(response) for response in responses]
    return atoms_list


def get_figlet():
    x = """
         _                   _____ _____ _______ 
    /\  | |                 / ____|  __ \__   __|
   /  \ | |_ ___  _ __ ___ | |  __| |__) | | |   
  / /\ \| __/ _ \| '_ ` _ \| | |_ |  ___/  | |   
 / ____ \ || (_) | | | | | | |__| | |      | |   
/_/    \_\__\___/|_| |_| |_|\_____|_|      |_|   
   """
    return x
