from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import os
import json
import zipfile
from jarvis.db.figshare import data
from tqdm import tqdm
import torch
import csv
import numpy as np
import random
def atoms_describer(
    atoms=[],
    xrd_peaks=5,
    xrd_round=1,
    cutoff=4,
    take_n_bomds=2,
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
        struct_info[
            "natoms_conventional"
        ] = spg.conventional_standard_structure.num_atoms
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
    return line


def get_crystal_string(atoms):
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
    # extra=str(atoms.num_atoms)+"\n"+atoms.composition.reduced_formula
    # crystal_str+=" "+extra
    # extra+="\n"+crystal_str
    # return extra
    # extra=atoms.composition.reduced_formula
    # crystal_str+="\n"+extra+"\n"+atoms.spacegroup()+"\n"
    crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str


def get_robo(structure=None):
    from robocrys import StructureCondenser, StructureDescriber

    # structure = Structure.from_file("POSCAR")
    # other file formats also supported
    # alternatively, uncomment the lines below to use the MPRester object
    # to fetch structures from the Materials Project database
    # from pymatgen import MPRester
    # structure = MPRester(API_KEY=None).get_structure_by_material_id("mp-856")
    condenser = StructureCondenser()
    describer = StructureDescriber()
    # condensed_structure = condenser.condense_structure(structure)
    # description = describer.describe(condensed_structure)
    description = describer.describe(structure)
    print(description)
    return description


class AtomGPTDataset(Dataset):
    def __init__(
        self, texts=[], targets=[], ids=[], tokenizer="", max_length=512
    ):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not ids:
            ids = ["text-" + str(i) for i in range(len(texts))]
        self.ids = ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # torch.tensor(inputs*10,dtype=inputs.dtype)
        return (
            inputs,
            self.ids[idx],
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def data_from_benchmark_file(
    benchmark_file="", leaderboard_dir="", tokenizer="", max_length=512, batch_size=8
):
    print("benchmark_file", benchmark_file)
    method = benchmark_file.split("-")[0]
    task = benchmark_file.split("-")[1]
    prop = benchmark_file.split("-")[2]
    dataset = benchmark_file.split("-")[3]
    temp = dataset + "_" + prop + ".json.zip"
    temp2 = dataset + "_" + prop + ".json"
    fname = os.path.join(leaderboard_dir, "benchmarks", method, task, temp)
    zp = zipfile.ZipFile(fname)
    bench = json.loads(zp.read(temp2))
    dft_3d = data(dataset)
    id_tag = "jid"
    if "jid" in dft_3d[0]:
        id_tag = "jid"
    else:
        id_tag = "id"

    train_ids = list(bench["train"].keys())
    test_ids = list(bench["test"].keys())
    if "val" in bench:
        val_ids = list(bench["val"].keys())
    else:
        val_ids = test_ids
    print("total", len(dft_3d))
    print("test_ids", len(test_ids))
    print("val_ids", len(val_ids))
    print("train_ids", len(train_ids))

    train_texts = []
    train_targets = []
    train_ids_temp = []
    val_texts = []
    val_targets = []
    val_ids_temp = []
    test_texts = []
    test_targets = []
    test_ids_temp = []

    for i in tqdm(dft_3d):
        if i[prop] != "na":
            atoms = Atoms.from_dict(i["atoms"])
            tmp = get_crystal_string(atoms)
            if i[id_tag] in train_ids:
                train_texts.append(tmp)
                train_targets.append(i[prop])
                train_ids_temp.append(i[id_tag])
            elif i[id_tag] in test_ids:
                test_texts.append(tmp)
                test_targets.append(i[prop])
                test_ids_temp.append(i[id_tag])
            elif i[id_tag] in val_ids:
                val_texts.append(tmp)
                val_targets.append(i[prop])
                val_ids_temp.append(i[id_tag])
    print("total", len(dft_3d))
    print("test_ids", len(test_ids))
    print("val_ids", len(val_ids))
    print("train_ids", len(train_ids))

    print("test_texts:", len(test_texts))
    print("test_texts examples:", test_texts[0])
    train_dataset = AtomGPTDataset(
        texts=train_texts,
        targets=train_targets,
        ids=train_ids_temp,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = AtomGPTDataset(
        texts=val_texts,
        targets=val_targets,
        tokenizer=tokenizer,
        ids=val_ids_temp,
        max_length=max_length,
    )
    test_dataset = AtomGPTDataset(
        texts=test_texts,
        targets=test_targets,
        tokenizer=tokenizer,
        ids=test_ids_temp,
        max_length=max_length,
    )

    # val_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # val_dataloader = test_dataloader
    return train_dataloader, val_dataloader, test_dataloader


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = (
        ids[-(n_val + n_test) : -n_test]
        if n_test > 0
        else ids[-(n_val + n_test) :]
    )  # noqa:E203
    id_test = ids[-n_test:] if n_test > 0 else []
    return id_train, id_val, id_test


def data_from_id_prop(
    id_prop_path="DataDir",
    tokenizer="",
    max_length=512,
    file_format="poscar",
    split_seed=42,
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    keep_data_order=False,
    batch_size=5
):
    id_prop_dat = os.path.join(id_prop_path, "id_prop.csv")
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        dat = [row for row in reader]
    texts = []
    props = []
    for i in dat:
        file_name = i[0]
        file_path = os.path.join(id_prop_path, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )
        desc = get_crystal_string(atoms)
        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        texts.append(desc)
        props.append(tmp)
    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(texts),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
    )
    train_texts = []
    train_targets = []
    train_ids_temp = []
    val_texts = []
    val_targets = []
    val_ids_temp = []
    test_texts = []
    test_targets = []
    test_ids_temp = []
    for i in id_train:
        train_texts.append(texts[i])
        train_targets.append(props[i])
        train_ids_temp.append(i)
    for i in id_val:
        val_texts.append(texts[i])
        val_targets.append(props[i])
        val_ids_temp.append(i)
    for i in id_test:
        test_texts.append(texts[i])
        test_targets.append(props[i])
        test_ids_temp.append(i)
    train_dataset = AtomGPTDataset(
        texts=train_texts,
        targets=train_targets,
        ids=train_ids_temp,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = AtomGPTDataset(
        texts=val_texts,
        targets=val_targets,
        tokenizer=tokenizer,
        ids=val_ids_temp,
        max_length=max_length,
    )
    test_dataset = AtomGPTDataset(
        texts=test_texts,
        targets=test_targets,
        tokenizer=tokenizer,
        ids=test_ids_temp,
        max_length=max_length,
    )

    print("test_ids", len(test_ids_temp))
    print("val_ids", len(val_ids_temp))
    print("train_ids", len(train_ids_temp))
    # val_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # val_dataloader = test_dataloader
    return train_dataloader, val_dataloader, test_dataloader
