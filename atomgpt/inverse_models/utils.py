from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import numpy as np
from jarvis.core.atoms import Atoms

# from jarvis.analysis.diffraction.xrd import smooth_xrd
from sklearn.metrics import mean_absolute_error
from jarvis.analysis.diffraction.xrd import XRD
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def baseline_als(y, lam, p, niter=10):
    """ALS baseline correction to remove broad background trends."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def sharpen_peaks(y, sigma=0.5):
    """Sharpen peaks using a narrow Gaussian filter."""
    # Use a very small sigma to reduce peak broadening
    y_sharp = gaussian_filter1d(y, sigma=sigma, mode="constant")
    return y_sharp


"""
def recast_array(
    x_original=[], y_original=[], x_new=np.arange(0, 90, 1), tol=0.1
):
    x_original = np.array(x_original)
    # Initialize the new y array with NaNs or a default value
    y_new = np.zeros_like(x_new,  dtype=np.float64)
    #y_new = np.full_like(x_new, 0, dtype=np.float64)

    # Fill the corresponding bins
    for x_val, y_val in zip(x_original, y_original):
        closest_index = np.abs(
            x_new - x_val
        ).argmin()  # Find the closest x_new index
        y_new[closest_index] = y_val
    # y_new[y_new<tol]=0
    return x_new, y_new
"""


def recast_array(
    x_original=[], y_original=[], x_new=np.arange(0, 90, 1), tol=0.1
):
    x_original = np.array(x_original)
    # Initialize the new y array with NaNs or a default value
    y_new = np.full_like(x_new, 0, dtype=np.float64)

    # Fill the corresponding bins
    for x_val, y_val in zip(x_original, y_original):
        closest_index = np.abs(
            x_new - x_val
        ).argmin()  # Find the closest x_new index
        y_new[closest_index] = y_val
    # y_new[y_new<tol]=0
    return x_new, y_new


def smooth_xrd(atoms=None, thetas=[0, 90], intvl=0.3, sep=";"):
    # def smooth_xrd(atoms=None, thetas=[0, 90], intvl=0.5):
    a, b, c = XRD(thetas=thetas).simulate(atoms=atoms)
    a = np.array(a)
    c = np.array(c)
    c = c / np.max(c)
    a, c = recast_array(
        x_original=a,
        y_original=c,
        x_new=np.arange(thetas[0], thetas[1], intvl),
    )
    c = c / np.max(c)
    # c_str = "\n".join(["{0:.3f}".format(x) for x in c])
    c_str = sep.join(["{0:.2f}".format(x) for x in c])

    return c_str, c


def text2atomsNEW(response):
    """
    Converts text output into a JARVIS Atoms object.

    Expected format:
        First 3 lines: Lattice matrix rows (each with 3 float values)
        Remaining lines: Element x y z (fractional coordinates)
    """
    try:
        response = response.strip().strip("</s>").strip()
        if not response:
            raise ValueError("Empty response string")

        tmp_atoms_array = [
            line.strip() for line in response.split("\n") if line.strip()
        ]
        print("tmp_atoms_array", tmp_atoms_array)

        # Expect at least 3 lines for lattice and 1 for atoms
        if len(tmp_atoms_array) < 4:
            raise ValueError(
                "Not enough lines to define a 3x3 lattice and atomic structure."
            )

        # Parse 3 lines of 3 lattice values each
        lattice_mat = []
        for i in range(3):
            row = list(map(float, tmp_atoms_array[i].split()))
            if len(row) != 3:
                raise ValueError(
                    f"Lattice row {i+1} must have 3 float values, got: {row}"
                )
            lattice_mat.append(row)

        print("lattice_mat", lattice_mat)

        elements = []
        coords = []

        # Parse atomic lines
        for line in tmp_atoms_array[3:]:
            parts = line.split()
            if len(parts) >= 4:
                elements.append(parts[0])
                coords.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            else:
                print(f"⚠️ Skipping malformed atom line: {line}")

        if len(elements) == 0 or len(coords) == 0:
            raise ValueError("No valid atomic data found.")

        atoms = Atoms(
            coords=coords,
            elements=elements,
            lattice_mat=lattice_mat,
            cartesian=False,
        )
        print("atoms", atoms)
        return atoms

    except Exception as e:
        print(f"❌ Failed to parse atoms from response: {e}")
        raise


def text2atomsXXXXX(response):
    """
    Converts text output into a JARVIS Atoms object.

    Expected format:
        Line 1: (optional) blank
        Line 2: 9 lattice numbers (3x3 matrix)
        Remaining lines: Element x y z (fractional coordinates)
    """
    try:
        response = response.strip().strip("</s>").strip()
        if not response:
            raise ValueError("Empty response string")

        tmp_atoms_array = [
            line.strip() for line in response.split("\n") if line.strip()
        ]
        print("tmp_atoms_array", tmp_atoms_array)

        # Expect at least one line for lattice and one for atoms
        if len(tmp_atoms_array) < 2:
            raise ValueError("Not enough lines to define structure.")

        # Detect and parse lattice line
        lat_line = tmp_atoms_array[0]
        lat_values = lat_line.split()
        if len(lat_values) != 9:
            raise ValueError(
                "Lattice line must contain exactly 9 float values.", lat_values
            )

        lattice_mat = np.array(lat_values, dtype=float).reshape(3, 3).tolist()
        print("lattice_mat", lattice_mat)

        elements = []
        coords = []

        # Parse atomic lines
        for line in tmp_atoms_array[1:]:
            parts = line.split()
            if len(parts) >= 4:
                elements.append(parts[0])
                coords.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            else:
                print(f"⚠️ Skipping malformed atom line: {line}")

        if len(elements) == 0 or len(coords) == 0:
            raise ValueError("No valid atomic data found.")

        atoms = Atoms(
            coords=coords,
            elements=elements,
            lattice_mat=lattice_mat,
            cartesian=False,
        )
        print("atoms", atoms)
        return atoms

    except Exception as e:
        print(f"❌ Failed to parse atoms from response: {e}")
        raise


def text2atomsold(response):
    # print("response", response)
    response = response.strip("</s>")
    if response.startswith("\n"):
        subs = 0
    else:
        subs = 1
    tmp_atoms_array = response.split("\n")
    print("tmp_atoms_array", tmp_atoms_array)
    print("lat", tmp_atoms_array[1 - subs].split())
    lattice_mat = (
        np.array(tmp_atoms_array[1 - subs].split(), dtype="float")
        .reshape(3, 3)
        .tolist()
    )
    print("lat2", lattice_mat)
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
        lattice_mat=lattice_mat,
        cartesian=False,
    )
    return atoms


def text2atoms(response):
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
    response = tokenizer.batch_decode(outputs)[0]
    print("response", response)
    response = response.split("# Output:")[1].strip("</s>")
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


def get_crystal_string_tNEW(atoms):
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords
    lat_mat = np.array(atoms.lattice_mat)

    # Clean string formatting for lattice matrix
    lat_mat_str = "\n".join(
        " ".join(f"{0.0 if abs(x) < 1e-6 else x:.2f}" for x in row)
        for row in lat_mat
    )

    # Assemble full crystal structure string
    crystal_str = (
        lat_mat_str
        + "\n"
        + "\n".join(
            f"{t} " + " ".join(f"{x:.3f}" for x in c)
            for t, c in zip(atom_ids, frac_coords)
        )
    )

    return crystal_str


def get_crystal_string_tOSOSOSO(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords
    lat_mat = np.array(atoms.lattice_mat)
    # lat_mat = "\n".join(" ".join(f"{x:.2f}" for x in row) for row in lat_mat)
    lat_mat_str = "\n".join(
        " ".join(f"{0.0 if abs(x) < 1e-6 else x:.2f}" for x in row)
        for row in lat_mat
    )

    crystal_str = (
        lat_mat
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


def processed(
    x,
    y,
    x_range=[0, 90],
    intvl=0.1,
    sigma=0.05,
    recast=True,
    tol=0.1,
    background_subs=True,
):
    """Process the spectrum: background removal and peak sharpening."""
    y = np.array(y, dtype="float")
    if background_subs:

        # 1. Baseline correction
        background = baseline_als(y, lam=10000, p=0.01)
        y_corrected = y - background
    else:
        y_corrected = y

    # 2. Normalize the corrected spectrum
    y_corrected = y_corrected / np.max(y_corrected)

    # 3. Generate new x-axis values
    x_new = np.arange(x_range[0], x_range[1], intvl)

    # 4. Recast the spectrum onto the new grid
    if recast:
        x_new, y_corrected = recast_array(x, y_corrected, x_new, tol=tol)

    # 5. Sharpen the peaks using Gaussian filtering
    y_sharp = sharpen_peaks(y_corrected, sigma=sigma)

    # 6. Final normalization
    if np.max(y_sharp) > 0:
        y_sharp = y_sharp / np.max(y_sharp)

    return x_new, y_sharp


def main_spectra(
    spectra=[],
    formulas=[],
    model=None,
    tokenizer=None,
    calculator=None,
    device="cpu",
    max_new_tokens=1024,
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
    atoms_array = []
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
        print("formula", formula)
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
        # print(atoms)
        atoms_array.append(atoms)
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


def parse_formula(formula):
    def expand_group(element_dict, group, multiplier):
        """Expand nested groups with multipliers."""
        for elem, count in group.items():
            element_dict[elem] += count * multiplier

    def get_elements_and_count(group):
        """Extract elements and their counts from a string group."""
        elements = defaultdict(int)
        # Capture elements and their counts, ignoring charge notations
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", group)
        for element, count in matches:
            elements[element] += int(count) if count else 1
        return elements

    # Remove all charge descriptions (e.g., ^4+, ^3-, etc.)
    formula = re.sub(r"\^\d*[\+\-]?", "", formula)

    # Remove underscores which may indicate counts
    formula = formula.replace("_", "")

    # Pattern to capture element groups (e.g., (CO3)4)
    pattern = r"\(([^()]+)\)(\d+)"
    element_dict = defaultdict(int)

    # Extract all parenthesized groups and process them
    while re.search(pattern, formula):
        matches = re.findall(pattern, formula)
        for group, multiplier in matches:
            elements = get_elements_and_count(group)
            expand_group(element_dict, elements, int(multiplier))
            formula = formula.replace(f"({group}){multiplier}", "", 1)

    # Process remaining non-parenthesized elements
    remaining_elements = get_elements_and_count(formula)
    expand_group(element_dict, remaining_elements, 1)

    # Construct the compact formula
    compact_formula = "".join(
        f'{elem}{count if count > 1 else ""}'
        for elem, count in sorted(element_dict.items())
    )
    return compact_formula


def load_exp_file(filename="", intvl=0.3):
    # df = pd.read_csv(
    #     filename,
    #     skiprows=1,
    #     sep=" ",
    #     engine="python",
    # )

    df = pd.read_csv(
        filename,
        skiprows=1,
        sep=r"[ ,]+",
        names=["X", "Y"],
        comment="#",
        engine="python",
    )

    # df = pd.read_csv(filename, skiprows=1, sep=" ", names=["X", "Y"], comment="#")
    print("df")
    print(df)
    if ".txt" in filename:

        with open(filename, "r") as f:
            lines = f.read().splitlines()
        for i in lines:
            if "##IDEAL CHEMISTRY=" in i:
                formula = Composition.from_string(
                    i.split("##IDEAL CHEMISTRY=")[1]
                    .replace("_", "")
                    .replace("^", "")
                    .replace("+", "")
                ).reduced_formula

                tmp = (
                    i.split("##IDEAL CHEMISTRY=")[1]
                    .replace("_", "")
                    .split("&#")[0]
                )
                formula = parse_formula(tmp)
                print(formula, i)

    else:
        formula = filename.split(".dat")[0]
    x = df["X"].values
    y = df["Y"].values
    # if df["Z"].isnull()[0]:
    #     y = df["Y"].values
    # else:
    #     y = df["Z"].values
    y = np.array(y)
    y = y / np.max(y)
    x, y_corrected = processed(x=x, y=y, intvl=intvl)
    return formula, x, y_corrected


def post_patch_loss_function(model):
    current_model = model
    while hasattr(current_model, "model"):
        try:
            # model.loss_function starts as a dict to a loss fx
            # We invoke it to save it
            current_model.loss_function = current_model.loss_function()
        except:
            # Failed means we already invoked it, and we need args to the loss fx
            pass
        pass
        current_model = current_model.model
    pass
    try:
        current_model.loss_function = current_model.loss_function()
    except:
        pass
    return model


pass
