import numpy as np
from atomgpt.inverse_models.utils import smooth_xrd
from jarvis.io.vasp.inputs import Poscar

pos = """LaB6
1.0
4.154998579020728 0.0 0.0
0.0 4.154998579020728 -0.0
0.0 0.0 4.154998579020728
La B
1 6
Cartesian
0.0 0.0 0.0
3.3248945674454897 2.0775 2.0775
0.8301054325545105 2.0775 2.0775
2.0775 2.0775 3.3248945674454897
2.0775 2.0775 0.8301054325545105
2.0775 0.8301054325545105 2.0775
2.0775 3.3248945674454897 2.0775
"""
atoms = Poscar.from_string(pos).atoms
y_new_str, cccc = smooth_xrd(atoms=atoms, intvl=0.3, thetas=[0, 90])
# print(y_new_str)
two_theta = np.arange(0, 90, 0.3)
for i, j in zip(two_theta, cccc):
    print(i, ",", j)
