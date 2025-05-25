"""Module to simulate STEM images using convoltuin approximation."""

# Adapted from https://github.com/jacobjma/fourier-scale-calibration
import numpy as np
from scipy.interpolate import interp1d
import os
import json
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms, get_supercell_dims, crop_square
from jarvis.core.lattice import get_2d_lattice
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from skimage.transform import warp, AffineTransform
from jarvis.analysis.structure.spacegroup import (
    symmetrically_distinct_miller_indices,
)
from jarvis.analysis.defects.surface import wulff_normals, Surface
import os
import json
from datasets import load_dataset  # Optional if using Hugging Face
from jarvis.core.utils import gaussian
from jarvis.core.utils import lorentzian2 as lorentzian
from jarvis.core.atoms import Atoms  # , get_supercell_dims, crop_square
from typing import List


class STEMConv(object):
    """Module to simulate STEM images using convoltuin approximation."""

    def __init__(
        self,
        atoms=None,
        output_size=[50, 50],
        power_factor=1.7,
        gaussian_width=0.5,
        lorentzian_width=0.5,
        intensity_ratio=0.5,
        nbins=100,
        tol=0.5,
        crop=False,
    ):
        """
        Intitialize the class.
        """
        self.atoms = atoms
        self.output_size = output_size
        self.power_factor = power_factor
        self.gaussian_width = gaussian_width
        self.lorentzian_width = lorentzian_width
        self.intensity_ratio = intensity_ratio
        self.nbins = nbins
        self.tol = tol
        self.crop = crop

    def superpose_deltas(self, positions, array):
        """Superpose deltas."""
        z = 0
        shape = array.shape[-2:]
        rounded = np.floor(positions).astype(np.int32)
        rows, cols = rounded[:, 0], rounded[:, 1]

        array[z, rows, cols] += (1 - (positions[:, 0] - rows)) * (
            1 - (positions[:, 1] - cols)
        )
        array[z, (rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (
            1 - (positions[:, 1] - cols)
        )
        array[z, rows, (cols + 1) % shape[1]] += (
            1 - (positions[:, 0] - rows)
        ) * (positions[:, 1] - cols)
        array[z, (rows + 1) % shape[0], (cols + 1) % shape[1]] += (
            rows - positions[:, 0]
        ) * (cols - positions[:, 1])
        return array

    def simulate_surface(
        self,
        atoms: Atoms,
        px_scale: float = 0.2,
        eps: float = 0.6,
        rot: float = 0,
        shift: List = [0, 0],
    ):
        """Simulate a STEM image.

        atoms: jarvis.core.Atoms material slab
        px_scale: pixel size in angstroms/px
        eps: tolerance factor (angstroms)
        for rendering atoms outside the field of view
        rot: rotation about the image center (degrees)
        shift: rigid translation of field of view [dx, dy] (angstroms)

        """
        shift = np.squeeze(shift)
        output_px = np.squeeze(self.output_size)  # px

        # field of view size in angstroms
        view_size = px_scale * (output_px - 1)

        # construct a supercell grid big enough to fill the field of view
        cell_extent = atoms.lattice.abc[0:2]  # np.diag(atoms.lattice_mat)[:2]

        cells = ((view_size // cell_extent) + 1).astype(int)
        # print ('cells',cells)
        atoms = atoms.make_supercell_matrix((3 * cells[0], 3 * cells[1], 1))

        # Set up real-space grid (in angstroms)
        # construct the probe array with the output target size
        # fftshift, pad, un-fftshift
        x = np.fft.fftfreq(output_px[0]) * output_px[0] * px_scale
        y = np.fft.fftfreq(output_px[1]) * output_px[1] * px_scale
        r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)

        # construct the probe profile centered
        # at (0,0) on the periodic spatial grid
        x = np.linspace(0, 4 * self.lorentzian_width, self.nbins)
        profile = gaussian(
            x, self.gaussian_width
        ) + self.intensity_ratio * lorentzian(x, self.lorentzian_width)
        profile /= profile.max()
        f = interp1d(x, profile, fill_value=0, bounds_error=False)
        intensity = f(r)

        # shift the probe profile to the center
        # apply zero-padding, and shift back to the origin
        margin = int(np.ceil(5 / px_scale))  # int like 20
        intensity = np.fft.fftshift(intensity)
        intensity = np.pad(intensity, (margin, margin))
        intensity = np.fft.fftshift(intensity)

        # project atomic coordinates onto the image
        # center them as well
        centroid = np.mean(atoms.cart_coords[:, :2], axis=0)

        # center atom positions around (0,0)
        pos = atoms.cart_coords[:, :2] - centroid

        # apply field of view rotation
        # (actually rotate the lattice coordinates)
        if rot != 0:
            rot = np.radians(rot)
            R = np.array(
                [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
            )
            pos = pos @ R

        # shift to center of image
        pos += view_size / 2

        # apply rigid translation of atoms wrt image field of view
        pos += shift

        # select only atoms in field of view
        in_view = (
            (pos[:, 0] > -eps)
            & (pos[:, 0] < view_size[0] + eps)
            & (pos[:, 1] > -eps)
            & (pos[:, 1] < view_size[1] + eps)
        )
        # pos = pos[in_view]
        numbers = np.array(atoms.atomic_numbers)
        # numbers = numbers[in_view]

        atom_px = pos / px_scale  # AA / (AA/px) -> px

        # atom_px = atom_px + margin

        render = in_view
        # render = (
        #     (pos[:, 0] > 0)
        #     & (pos[:, 0] < view_size[0])
        #     & (pos[:, 1] > 0)
        #     & (pos[:, 1] < view_size[1])
        # )

        numbers_render = numbers[render]
        # # shift atomic positions to offset zero padding
        atom_px_render = atom_px[render] + margin

        # initialize arrays with zero padding
        array = np.zeros((1,) + intensity.shape)  # adding extra 1
        mask = np.zeros((1,) + intensity.shape)
        # print(f"intensity: {array.shape}")
        for number in np.unique(np.array(atoms.atomic_numbers)):

            temp = np.zeros((1,) + intensity.shape)
            temp = self.superpose_deltas(
                atom_px_render[numbers_render == number], temp
            )
            array += temp * number**self.power_factor
            temp = np.where(temp > 0, number, temp)
            mask += temp[0]

        # FFT convolution of beam profile and atom position delta functions
        array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real

        # crop the FFT padding and fix atom coordinates relative to
        # the image field of view
        sel = slice(margin, -margin)
        array = array[0, sel, sel]
        mask = mask[0, sel, sel]
        # atom_px = atom_px - margin

        atom_px = pos[in_view] / px_scale
        numbers = numbers[in_view]

        return array, mask, atom_px, numbers


random.seed(42)


class STEMDatasetGenerator:
    def __init__(
        self,
        dataset_name="",
        output_folder="",
        supercell_scales=[1],  # , 2, 3],
        id_tag="jid",
        miller_indices=[],
        power_factor=1.7,
        output_size=[256, 256],
        # px_scale=0.2,
        px_scale=0.05,
        max_miller_index=1,
        cell_size=50,
        ids=[],
        save_images=True,
    ):
        self.id_tag = id_tag
        self.save_images = save_images
        self.dataset_name = dataset_name
        self.output_folder = output_folder
        self.supercell_scales = supercell_scales
        self.id_tag = id_tag
        self.miller_indices = miller_indices
        self.power_factor = power_factor
        self.output_size = output_size
        self.px_scale = px_scale
        self.max_miller_index = max_miller_index
        self.cell_size = cell_size
        os.makedirs(self.output_folder, exist_ok=True)
        # os.makedirs(self.output_folder, exist_ok=True)
        self.ids = ids
        # print("self.miller_indices",self.miller_indices)

    def get_crystal_string_t(self, atoms):
        # atoms = atoms.center(vacuum=1)
        # atoms = atoms.center(vacuum=12)
        lengths = atoms.lattice.abc  # Lattice lengths
        angles = atoms.lattice.angles  # Lattice angles
        atom_ids = atoms.elements  # Atom types
        frac_coords = atoms.frac_coords  # Fractional coordinates

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
        return crystal_str

    def add_artifacts(self, image, artifact_type="noise", **kwargs):
        if artifact_type == "noise":
            noise_type = kwargs.get("noise_type", "gaussian")
            var = kwargs.get("var", 0.01)
            if noise_type == "gaussian":
                image = random_noise(image, mode="gaussian", var=var)
            elif noise_type == "poisson":
                image = random_noise(image, mode="poisson")
            image = (255 * image).astype("uint8")
        elif artifact_type == "blur":
            sigma = kwargs.get("sigma", 1)
            image = gaussian_filter(image, sigma=sigma)
        elif artifact_type == "distortion":
            scale = kwargs.get("scale", 1.02)
            transform = AffineTransform(scale=(scale, scale))
            image = warp(image, transform, mode="wrap")
            image = (255 * image).astype("uint8")
        elif artifact_type == "stripe":
            stripe_intensity = kwargs.get("stripe_intensity", 0.2)
            stripe_period = kwargs.get("stripe_period", 10)
            for i in range(0, image.shape[0], stripe_period):
                image[i, :] = np.clip(
                    image[i, :] + stripe_intensity * 255, 0, 255
                )
            image = image.astype("uint8")
        return image

    def load_dataset(self):
        if os.path.exists(self.dataset_name):
            with open(self.dataset_name, "r") as f:
                return json.load(f)
        else:
            return data(self.dataset_name)

    def generate_dataset(self, input_dataset):
        dataset = []

        for material in tqdm(input_dataset, total=len(input_dataset)):
            try:
                atoms = Atoms.from_dict(material["atoms"])
                cvn_atoms = atoms.get_conventional_atoms
                # print("self.miller_indices",self.miller_indices)
                if len(self.miller_indices) == 0:

                    self.miller_indices = (
                        symmetrically_distinct_miller_indices(
                            cvn_atoms=cvn_atoms,
                            max_index=self.max_miller_index,
                        )
                    )

                for miller_index in self.miller_indices:
                    # print('miller_index',miller_index)
                    surf = (
                        Surface(
                            atoms=cvn_atoms,
                            indices=miller_index,
                            layers=1,
                            vacuum=18,
                        )
                        .make_surface()
                        .center_around_origin()
                    )
                    class_lat = get_2d_lattice(surf.to_dict())[0]

                    material_id = material.get(self.id_tag, "unknown_id")
                    self.ids.append(material_id)
                    # print(f"Processing material: {material_id}")
                    for scale_factor in self.supercell_scales:
                        dims = get_supercell_dims(
                            atoms=surf,
                            enforce_c_size=scale_factor
                            * self.cell_size
                            * 3,  # something bigger to crop from
                        )
                        dims[2] = 1
                        supercell = surf.make_supercell_matrix(dims)
                        cropped_atoms = crop_square(
                            supercell, csize=self.cell_size * scale_factor
                        )

                        simulated_image = STEMConv(
                            power_factor=self.power_factor,
                            output_size=self.output_size,
                        ).simulate_surface(
                            cropped_atoms, px_scale=self.px_scale
                        )[
                            0
                        ]

                        array = simulated_image
                        normalized_array = (
                            255
                            * (array - np.min(array))
                            / (np.max(array) - np.min(array))
                        ).astype("uint8")

                        noisy_image = self.add_artifacts(
                            normalized_array, artifact_type="noise", var=0.02
                        )
                        blurred_image = self.add_artifacts(
                            noisy_image, artifact_type="blur", sigma=1
                        )
                        final_image = blurred_image
                        """
                        distorted_image = self.add_artifacts(
                            blurred_image,
                            artifact_type="distortion",
                            scale=1.02,
                        )
                        final_image = self.add_artifacts(
                            distorted_image,
                            artifact_type="stripe",
                            stripe_intensity=0.1,
                            stripe_period=20,
                        )
                        """

                        pil_image = Image.fromarray(final_image).convert("L")

                        image_filename = f"{material_id}_{scale_factor}x{scale_factor}x{scale_factor}_{miller_index[0]}{miller_index[1]}{miller_index[2]}.jpg"
                        image_path = os.path.join(
                            self.output_folder, image_filename
                        )
                        if self.save_images:
                            pil_image.save(image_path)

                        poscar_string = self.get_crystal_string_t(atoms)

                        question = (
                            "The chemical formula is "
                            + atoms.composition.reduced_formula
                            # "The chemical elements are "
                            # + str(
                            #    atoms.composition.search_string.replace(
                            #        "-", " ,"
                            #    )
                            # )
                            + ". Generate atomic structure description with lattice lengths, angles, coordinates, and atom types. Also predict the Miller index."
                        )
                        explanation = (
                            f"\n{poscar_string}"
                            + ". The miller index is ("
                            + str(" ".join(map(str, miller_index)))
                            + "). "
                        )

                        dataset.append(
                            {
                                "id": image_filename.split(".jpg")[0],
                                "messages": [
                                    {
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": question,
                                                # "index": None,
                                            },
                                            {
                                                "type": "image",
                                                "text": None,
                                                # "index": 0,
                                            },
                                        ],
                                        "role": "user",
                                    },
                                    {
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": explanation,
                                                # "index": None,
                                            }
                                        ],
                                        "role": "assistant",
                                    },
                                ],
                                "images": [pil_image],
                            }
                        )
                        # print(dataset[-1])
            except Exception as exp:
                print("Exception:", exp, material)
                pass

        return dataset

    def dump_dataset(self, train_dataset, test_dataset):
        train_path = os.path.join(
            self.output_folder, self.dataset_name + "_train_dataset.json"
        )
        test_path = os.path.join(
            self.output_folder, self.dataset_name + "_test_dataset.json"
        )
        with open(train_path, "w") as f:
            # json.dump([i["id"] for i in train_dataset], f)
            # dat=[i.pop("images") for i in train_dataset]
            dat = [
                {k: v for k, v in i.items() if k != "images"}
                for i in train_dataset
            ]
            # print("dat",dat)
            json.dump(dat, f)
        with open(test_path, "w") as f:
            # json.dump([i["id"] for i in test_dataset], f)
            # dat=[i.pop("images") for i in test_dataset]
            dat = [
                {k: v for k, v in i.items() if k != "images"}
                for i in test_dataset
            ]
            json.dump(dat, f)
            # json.dump(test_dataset, f, indent=2)
        print(f"Train dataset saved to {train_path}")
        print(f"Test dataset saved to {test_path}")


# Usage example


def generate_dataset_new(
    dataset_name="dft_2d",
    output_folder="pi_image_dataset_with_artifacts",
    train_ratio=0.9,
    miller_indices=[],
    max_samples=None,
):
    # Predefined miller indices
    if dataset_name == "dft_2d":
        miller_indices = [[0, 0, 1]]
    if dataset_name == "dft_3d":
        miller_indices = [[1, 1, 0], [1, 1, 1], [0, 0, 1]]

    train_path = os.path.join(
        output_folder, dataset_name + "_train_dataset.json"
    )
    test_path = os.path.join(
        output_folder, dataset_name + "_test_dataset.json"
    )

    # ✅ Check if both JSON files exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading cached dataset from {train_path} and {test_path}")
        with open(train_path, "r") as f:
            train_dataset = json.load(f)
        with open(test_path, "r") as f:
            test_dataset = json.load(f)
        return train_dataset, test_dataset

    # 🔄 Generate fresh data otherwise
    generator = STEMDatasetGenerator(
        dataset_name=dataset_name,
        output_folder=output_folder,
        miller_indices=miller_indices,
    )

    print("Loading dataset...")
    db = generator.load_dataset()

    if max_samples is not None:
        db = db[:max_samples]

    random.shuffle(db)
    total_size = len(db)
    n_train = int(total_size * train_ratio)

    print(f"n_train: {n_train}, n_test: {total_size - n_train}")
    print("Generating training dataset...")
    train_dataset = generator.generate_dataset(db[:n_train])

    print("Generating testing dataset...")
    test_dataset = generator.generate_dataset(db[n_train:])

    print("Saving datasets...")
    generator.dump_dataset(train_dataset, test_dataset)

    return train_dataset, test_dataset


def generate_dataset(
    dataset_name="dft_2d",
    output_folder="pi_image_dataset_with_artifacts",
    train_ratio=0.9,
    miller_indices=[],
    max_samples=None,
    id_tag="jid",
):
    # if dataset_name == "dft_2d":
    #    miller_indices = [[0, 0, 1]]
    if dataset_name == "dft_3d":
        miller_indices = [[1, 1, 0], [1, 1, 1], [0, 0, 1]]
    else:
        miller_indices = [[0, 0, 1]]
    generator = STEMDatasetGenerator(
        dataset_name=dataset_name,
        id_tag=id_tag,
        output_folder=output_folder,
        miller_indices=miller_indices,
    )
    print("Loading dataset...")
    db = generator.load_dataset()  # [0:10]
    if max_samples is not None:
        db = db[:max_samples]
    random.shuffle(db)

    total_size = len(db)
    n_train = int(total_size * train_ratio)
    n_test = total_size - n_train

    print(f"n_train: {n_train}, n_test: {n_test}")

    print("Generating training dataset...")
    train_dataset = generator.generate_dataset(db[:n_train])

    print("Generating testing dataset...")
    test_dataset = generator.generate_dataset(db[n_train:])

    print("Saving datasets...")
    generator.dump_dataset(train_dataset, test_dataset)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = generate_dataset(dataset_name="dft_3d")
