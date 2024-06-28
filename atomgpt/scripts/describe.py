"""Module to describe an atomic structure."""
import json
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
import pprint
from collections import defaultdict


def atoms_describer(
    atoms=[], xrd_peaks=5, xrd_round=1, cutoff=4, take_n_bomds=2
):
    """Describe an atomic structure."""
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
        if len(dist) >= take_n_bomds:
            dist = dist[0:take_n_bomds]
        bond_distances[i] = ", ".join(map(str, dist))
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    info = {}
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": json.dumps(fracs),
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
        "spg_number": spg.space_group_number,
        "spg_symbol": spg.space_group_symbol,
        "top_k_xrd_peaks": ", ".join(
            map(
                str,
                sorted(list(set([round(i, xrd_round) for i in theta])))[
                    0:xrd_peaks
                ],
            )
        ),
        "density": round(atoms.density, 3),
        "crystal_system": spg.crystal_system,
        "point_group": spg.point_group_symbol,
        "wyckoff": ", ".join(list(set(spg._dataset["wyckoffs"]))),
        "bond_distances": bond_distances,
        "natoms_primitive": spg.primitive_atoms.num_atoms,
        "natoms_conventional": spg.conventional_standard_structure.num_atoms,
    }
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    line = ""
    for i, j in info.items():
        if not isinstance(j, dict):
            line += "The " + i + " is " + j + ". "
        else:
            for ii, jj in j.items():
                # tmp=''
                # if isinstance(jj,dict):
                #   for iii,jjj in jj.items():
                #        tmp=iii+": "+str(jjj)
                # else:
                #   tmp=jj
                line += "The " + ii + " is " + str(jj) + ". "
    return line


if __name__ == "__main__":
    atoms = Atoms.from_dict(
        get_jid_data(jid="JVASP-32", dataset="dft_3d")["atoms"]
    )
    info = atoms_describer(atoms=atoms)
    print(info)
    import sys

    sys.exit()
    # pprint.pprint(info)
    from pygments import highlight
    from pygments.formatters.terminal256 import Terminal256Formatter
    from pygments.lexers.web import JsonLexer

    raw_json = json.dumps(info, indent=4)
    colorful = highlight(
        raw_json,
        lexer=JsonLexer(),
        formatter=Terminal256Formatter(),
    )
    print(colorful)
