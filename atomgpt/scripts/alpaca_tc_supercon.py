from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson

dft_3d = data("dft_3d")


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


def make_alpaca_json(dataset=[], prop="Tc_supercon"):
    mem = []
    for i in dataset:
        if i[prop] != "na":
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["instruction"] = (
                "Below is a description of a superconductor material."
            )
            info["input"] = (
                "The chemical formula is "
                + atoms.composition.reduced_formula
                + " The  "
                + prop
                + " is "
                + str(round(i[prop], 3))
                + ". The spacegroup is "
                + i["spg_number"]
                + "."
                + " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


m = make_alpaca_json(dataset=dft_3d, prop="Tc_supercon")
dumpjson(data=m, filename="alpaca_Tc_supercon.json")
