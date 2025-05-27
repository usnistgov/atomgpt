from robocrys import StructureCondenser, StructureDescriber
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
from tqdm import tqdm
from robocrys.cli import robocrystallographer
from jarvis.core.atoms import Atoms

import os


def get_robo(structure=None):
    description = robocrystallographer(structure)

    # structure = Structure.from_file("POSCAR")
    # other file formats also supported
    # alternatively, uncomment the lines below to use the MPRester object
    # to fetch structures from the Materials Project database
    # from pymatgen import MPRester
    # structure = MPRester(API_KEY=None).get_structure_by_material_id("mp-856")
    # condenser = StructureCondenser()
    # describer = StructureDescriber()
    # condensed_structure = condenser.condense_structure(structure)
    # description = describer.describe(condensed_structure)
    # description = describer.describe(structure)
    print(description)
    return description


# atoms=Atoms.from_poscar('POSCAR-PdH').pymatgen_converter()
# desc = get_robo(atoms)
dft_3d = data("dft_3d")
for i in tqdm(dft_3d):
    fname = i["jid"] + ".json"
    if not os.path.exists(fname):
        # if i['Tc_supercon']!='na':
        # try:
        print("fname", fname)
        info = {}
        atoms = Atoms.from_dict(i["atoms"]).pymatgen_converter()
        desc = get_robo(atoms)
        info["desc"] = desc
        info["jid"] = i["jid"]
        dumpjson(data=info, filename=fname)
    # except:
    #    print("Failed", i["jid"])
    #    pass
