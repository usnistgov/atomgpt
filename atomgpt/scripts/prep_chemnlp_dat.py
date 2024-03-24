from tinnet159x import get_crystal_string_t
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import dumpjson
dft_3d=data('dft_3d')
x={}
for i in tqdm(dft_3d):
    atoms=Atoms.from_dict(i['atoms'])
    desc=get_crystal_string_t(atoms)
    x[i['jid']]=desc

dumpjson(data=x,filename='desc.json')
