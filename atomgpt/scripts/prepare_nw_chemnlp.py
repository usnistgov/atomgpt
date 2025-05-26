from jarvis.db.figshare import data
from jarvis.db.jsonutils import dumpjson
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
from collections import defaultdict
from tqdm import tqdm


def atoms_describer(
    atoms=[],
    xrd_peaks=5,
    xrd_round=1,
    cutoff=4,
    take_n_bonds=2,
    include_spg=True,
):
    """Describe an atomic structure."""
    if include_spg:
        spg = Spacegroup3D(atoms)
    theta, d_hkls, intens = XRD().simulate(atoms=(atoms))
    #     x = atoms.atomwise_angle_and_radial_distribution()
    #     bond_distances = {}
    #     for i, j in x[-1]["different_bond"].items():
    #         bond_distances[i.replace("_", "-")] = ", ".join(
    #             map(str, (sorted(list(set([round(jj, 2) for jj in j])))))
    #         )
    dists = defaultdict(list)
    elements = atoms.elements
    for i in atoms.get_all_neighbors(r=cutoff):
        for j in i:
            key = "-".join(sorted([elements[j[0]], elements[j[1]]]))
            dists[key].append(j[2])
    bond_distances = {}
    for i, j in dists.items():
        dist = sorted(set([round(k, 2) for k in j]))
        if len(dist) >= take_n_bonds:
            dist = dist[0:take_n_bonds]
        bond_distances[i] = ", ".join(map(str, dist))
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    info = {}
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": (fracs),
        "atomic_X": ", ".join(
            map(str, [Specie(s).X for s in atoms.uniq_species])
        ),
        "atomic_Z": ", ".join(
            map(str, [Specie(s).Z for s in atoms.uniq_species])
        ),
    }
    struct_info = {
        "lattice_parameters": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.abc])
        ),
        "lattice_angles": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.angles])
        ),
        # "spg_number": spg.space_group_number,
        # "spg_symbol": spg.space_group_symbol,
        "top_k_xrd_peaks": ", ".join(
            map(
                str,
                sorted(list(set([round(i, xrd_round) for i in theta])))[
                    0:xrd_peaks
                ],
            )
        ),
        "density": round(atoms.density, 3),
        # "crystal_system": spg.crystal_system,
        # "point_group": spg.point_group_symbol,
        # "wyckoff": ", ".join(list(set(spg._dataset["wyckoffs"]))),
        "bond_distances": bond_distances,
        # "natoms_primitive": spg.primitive_atoms.num_atoms,
        # "natoms_conventional": spg.conventional_standard_structure.num_atoms,
    }
    if include_spg:
        struct_info["spg_number"] = spg.space_group_number
        struct_info["spg_symbol"] = spg.space_group_symbol
        struct_info["crystal_system"] = spg.crystal_system
        struct_info["point_group"] = spg.point_group_symbol
        struct_info["wyckoff"] = ", ".join(list(set(spg._dataset["wyckoffs"])))
        struct_info["natoms_primitive"] = spg.primitive_atoms.num_atoms
        struct_info["natoms_conventional"] = (
            spg.conventional_standard_structure.num_atoms
        )
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    line = "The number of atoms are: " + str(
        atoms.num_atoms
    )  # +"., The elements are: "+",".join(atoms.elements)+". "
    for i, j in info.items():
        if not isinstance(j, dict):
            line += "The " + i + " is " + j + ". "
        else:
            # print("i",i)
            # print("j",j)
            for ii, jj in j.items():
                tmp = ""
                if isinstance(jj, dict):
                    for iii, jjj in jj.items():
                        tmp += iii + ": " + str(jjj) + " "
                else:
                    tmp = jj
                line += "The " + ii + " is " + str(tmp) + ". "
    line1 = line
    # print('bond_distances', struct_info['bond_distances'])
    tmp = ""
    p = struct_info["bond_distances"]
    for ii, (kk, vv) in enumerate(p.items()):
        if ii == len(p) - 1:
            punc = " Å."
        else:
            punc = " Å; "
        tmp += kk + ": " + vv + punc
    line2 = (
        chem_info["atomic_formula"]
        + " crystallizes in the "
        + struct_info["crystal_system"]
        + " "
        + str(struct_info["spg_symbol"])
        + " spacegroup, "
        + struct_info["point_group"]
        + " pointgroup with a prototype of "
        + str(chem_info["prototype"])
        +
        # " and a molecular weight of " +
        # str(chem_info['molecular_weight']) +
        ". The atomic fractions are: "
        + str(chem_info["atomic_fraction"]).replace("{", "").replace("}", "")
        + " with electronegaticities as "
        + str(chem_info["atomic_X"])
        + " and atomic numbers as "
        + str(chem_info["atomic_Z"])
        + ". The bond distances are: "
        + str(tmp)
        + "The lattice lengths are: "
        + struct_info["lattice_parameters"]
        + " Å, and the lattice angles are: "
        + struct_info["lattice_angles"]
        + "º with some of the top XRD peaks at "
        + struct_info["top_k_xrd_peaks"]
        + "º with "
        + "Wyckoff symbols "
        + struct_info["wyckoff"]
        + "."
    )
    info["desc_1"] = line1
    info["desc_2"] = line2
    return info


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "#\n"
        + " ".join([str(int(x)) for x in angles])
        + "@\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c]) + "&"
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    crystal_str = atoms_describer(atoms)["desc_2"] + "\n*\n" + crystal_str
    return crystal_str


mem = []
dft_3d = data("dft_3d")
for i in tqdm(dft_3d):
    info = {}
    atoms = Atoms.from_dict(i["atoms"])
    desc = get_crystal_string_t(atoms)
    info["id"] = i["jid"]
    info["desc"] = desc
    # print(desc)
    mem.append(info)
dumpjson(data=mem, filename="chemnlp_new_desc.json")
