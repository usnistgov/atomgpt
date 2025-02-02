from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


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


def relax_atoms(
    atoms=None,
    calculator=None,
    fmax=0.05,
    nsteps=150,
    constant_volume=False,
):
    from ase.optimize.fire import FIRE
    from ase.constraints import ExpCellFilter

    if calculator is None:
        return atoms

    t1 = time.time()
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


def gen_atoms(
    prompt="",
    model=None,
    tokenizer="",
    max_new_tokens=1024,
    alpaca_prompt="",
    instruction="Below is a description of a material.",
    device="cuda",
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
    ).to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )
    response = (
        tokenizer.batch_decode(outputs)[0].split("# Output:")[1].strip("</s>")
    )
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
    prompts=[],
    max_new_tokens=2048,
    model=None,
    tokenizer=None,
    alpaca_prompt="",
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


def main_spectra(
    spectra=[],
    formulas=[],
    model=None,
    tokenizer=None,
    calculator=None,
    device="cpu",
    max_new_tokens=500,
    intvl=0.3,
    thetas=[0, 90],
    filename=None,
    panels=[
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
    ],
    fmax=0.05,
    nsteps=150,
    alpaca_prompt="",
):
    the_grid = GridSpec(len(spectra), 4)
    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=(16, 4 * len(spectra)))
    count = 0
    for ii, cccc in enumerate(spectra):

        plt.subplot(the_grid[ii, 1])
        # plt.plot(cccc,label='Target')
        targ = spectra[ii][1]
        cccc = targ
        y_new_str = spectra[ii][0]
        # atoms1 = Atoms.from_dict(
        #     get_jid_data(jid=filename, dataset="dft_3d")["atoms"]
        # )
        # y_new_str,cccc = smooth_xrd(atoms=atoms1,intvl=0.3)

        title = "(" + panels[count] + ") " + "Input XRD"
        count += 1
        plt.title(title)

        plt.plot(cccc, c="red")
        plt.ylim([-0.02, 1])
        plt.xticks([0, 150, 300], [0, 45, 90])
        plt.xlabel(r"$2\theta$")
        plt.tight_layout()
        formula = formulas[ii]  # atoms1.composition.reduced_formula

        info = {}
        info["instruction"] = "Below is a description of a material."
        info["input"] = (
            "The chemical formula is "
            + formula
            # + " The  "
            + " The  "
            + "XRD"
            # "The chemical elements are "
            # + Composition.from_string(formula).search_string
            + " is "
            + y_new_str
            + "."
            + " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
        )
        # print(info)
        atoms = gen_atoms(
            prompt=info["input"],
            model=model,
            alpaca_prompt=alpaca_prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        print(atoms)
        plt.subplot(the_grid[ii, 2])
        y_new_str, cccc = smooth_xrd(atoms=atoms, intvl=intvl, thetas=thetas)
        # x, d_hkls1, y = XRD().simulate(atoms=optim)
        # y=np.array(y)/max(y)
        # plt.bar(x,y,label='DiffractGPT+ALIGNN-FF')
        # mae=round(stats.pearsonr(targ, cccc)[0], 2) #
        mae = round(mean_absolute_error(targ, cccc), 3)
        # plt.title('DiffractGPT'+str(mae))
        plt.plot(cccc, c="blue")
        plt.xticks([0, 150, 300], [0, 45, 90])
        plt.ylim([-0.02, 1])

        plt.xlabel(r"$2\theta$")
        plt.tight_layout()
        title = "(" + panels[count] + ") " + "DGPT" + " (" + str(mae) + ")"
        count += 1
        plt.title(title)
        # plt.legend()
        # from alignn.ff.ff import AlignnAtomwiseCalculator
        # calculator = AlignnAtomwiseCalculator()
        calculator = None
        optim = relax_atoms(
            atoms=atoms, calculator=calculator, fmax=fmax, nsteps=nsteps
        )

        # x, d_hkls1, y = XRD().simulate(atoms=atoms1)
        # y=np.array(y)/max(y)
        # plt.bar(x,y,label='Target')

        plt.subplot(the_grid[ii, 3])
        y_new_str, cccc = smooth_xrd(atoms=optim, intvl=intvl, thetas=thetas)
        # x, d_hkls1, y = XRD().simulate(atoms=optim)
        # y=np.array(y)/max(y)
        # plt.bar(x,y,label='DiffractGPT+ALIGNN-FF')
        # mae=round(stats.pearsonr(targ, cccc)[0], 2)
        mae = round(mean_absolute_error(targ, cccc), 3)
        # plt.title('DiffractGPT-A'+str(mae))
        plt.plot(cccc, c="green")
        plt.xticks([0, 150, 300], [0, 45, 90])
        plt.xlabel(r"$2\theta$")
        plt.ylim([-0.02, 1])

        title = "(" + panels[count] + ") " + "DGPT+AFF" + " (" + str(mae) + ")"
        count += 1
        plt.title(title)
        plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    # plt.xlim([0,90])
    # plt.legend()
