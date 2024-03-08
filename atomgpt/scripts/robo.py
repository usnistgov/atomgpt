from pymatgen.core.structure import Structure
from robocrys import StructureCondenser, StructureDescriber

structure = Structure.from_file("POSCAR") # other file formats also supported

# alternatively, uncomment the lines below to use the MPRester object
# to fetch structures from the Materials Project database
# from pymatgen import MPRester
# structure = MPRester(API_KEY=None).get_structure_by_material_id("mp-856")

condenser = StructureCondenser()
describer = StructureDescriber()

condensed_structure = condenser.condense_structure(structure)
description = describer.describe(condensed_structure)
print(description)
