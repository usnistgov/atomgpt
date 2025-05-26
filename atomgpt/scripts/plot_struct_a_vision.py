# conda activate /lab/mml/kipp/677/jarvis/Software/microgpt310
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from scipy import stats
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pymatgen.analysis.structure_matcher import StructureMatcher
from jarvis.core.lattice import get_2d_lattice

the_grid = GridSpec(2, 3)
plt.rcParams.update({"font.size": 18})
fig = plt.figure(figsize=(14, 8))
# plt.figure(figsize=(16,14))


def emd_distance(p, q, bins=None):
    """
    Compute Earth Mover's Distance (Wasserstein distance) between two 1D counts.

    Args:
        p (np.ndarray): First count or histogram (can be values or weights).
        q (np.ndarray): Second count or histogram.
        bins (np.ndarray or list, optional): Positions of each bin (same length as p and q).
                                             If None, assumes equally spaced bins [0, 1, ..., N-1].

    Returns:
        float: Earth Mover's Distance (EMD)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize if they are not already
    p /= np.sum(p)
    q /= np.sum(q)

    # Use default positions if not specified
    if bins is None:
        bins = np.arange(len(p))

    return wasserstein_distance(bins, bins, u_weights=p, v_weights=q)


def kl_divergence(p, q):
    """
    Compute KL divergence between two counts (p || q)
    Args:
        p (np.array): First count (usually the true distribution)
        q (np.array): Second count (usually the predicted distribution)
    Returns:
        float: KL divergence
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize to make sure they are probability counts
    p /= np.sum(p)
    q /= np.sum(q)
    return stats.entropy(p, q)
    # Use scipy's rel_entr (returns elementwise p * log(p/q))
    # return np.sum(rel_entr(p, q))


d = loadjson(
    "dft_2d_formula_based_dft_2d_formula_output_dir_dft_2d_unsloth/Llama-3.2-11B-Vision-Instructold/checkpoint-620/test_predictions.json"
)
d = loadjson(
    "/lab/mml/kipp/677/jarvis/Software/atomgpt310/atomgpt/atomgpt/inverse_models/c2db_formula_based_c2db_unsloth/Llama-3.2-11B-Vision-Instruct/test_predictions.json"
)

x_a = [i["target"]["abc"][0] for i in d]
y_a = [i["predicted"]["abc"][0] for i in d]

plt.subplot(the_grid[0, 0])
weights_x = np.ones_like(x_a) / len(x_a) * 100
weights_y = np.ones_like(y_a) / len(y_a) * 100
plt.hist(
    x_a,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="target",
    alpha=0.6,
    color="tab:blue",
)
plt.hist(
    y_a,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="predicted",
    color="plum",
    alpha=0.6,
)
plt.xlabel("a ($\AA$)")
plt.title("(a)")
plt.ylabel("Materials dist.")
plt.legend()

x_b = [i["target"]["abc"][1] for i in d]
y_b = [i["predicted"]["abc"][1] for i in d]


plt.subplot(the_grid[0, 1])
weights_x = np.ones_like(x_b) / len(x_b) * 100
weights_y = np.ones_like(y_b) / len(y_b) * 100
plt.hist(
    x_b,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="target_b",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_b,
    bins=np.arange(2, 7, 0.1),
    weights=weights_y,
    label="predicted_b",
    color="plum",
    alpha=0.6,
)
plt.xlabel("b ($\AA$)")
plt.title("(b)")


plt.subplot(the_grid[0, 2])
x_gamma = [i["target"]["angles"][2] for i in d]
y_gamma = [i["predicted"]["angles"][2] for i in d]
weights_x = np.ones_like(x_gamma) / len(x_gamma) * 100
weights_y = np.ones_like(y_gamma) / len(y_gamma) * 100
plt.hist(
    x_gamma,
    bins=np.arange(30, 150, 10),
    weights=weights_x,
    label="target_gamma",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_gamma,
    bins=np.arange(30, 150, 10),
    weights=weights_y,
    label="predicted_gamma",
    color="plum",
    alpha=0.6,
)
plt.xlabel("$\gamma$ ($^\circ$)")

plt.title("(c)")


x_c = [i["target"]["abc"][2] for i in d]
y_c = [i["predicted"]["abc"][2] for i in d]
# plt.subplot(the_grid[1, 2])
# plt.hist(x_c,bins=np.arange(10,30,.1),label='target_c', color='tab:blue', alpha=0.6)
# plt.hist(y_c,bins=np.arange(10,30,.1),label='predicted_c', color='plum', alpha=0.6)
# plt.xlabel('c')
# plt.ylabel('Materials count')


x_alpha = [i["target"]["angles"][0] for i in d]
y_alpha = [i["predicted"]["angles"][0] for i in d]
# plt.subplot(the_grid[1, 0])
# plt.hist(x_alpha,bins=np.arange(0,180,10),label='target_alpha', color='tab:blue', alpha=0.6)
# plt.hist(y_alpha,bins=np.arange(0,180,10),label='predicted_alpha', color='plum', alpha=0.6)
# plt.xlabel('alpha')
# plt.ylabel('Materials count')


x_beta = [i["target"]["angles"][1] for i in d]
y_beta = [i["predicted"]["angles"][1] for i in d]
# plt.subplot(the_grid[1, 1])
# plt.hist(x_beta,bins=np.arange(0,180,10),label='target_beta', color='tab:blue', alpha=0.6)
# plt.hist(y_beta,bins=np.arange(0,180,10),label='predicted_beta', color='plum', alpha=0.6)
# plt.xlabel('beta')
# plt.ylabel('Materials count')
# plt.ylabel('Materials count')
print("a", mean_absolute_error(x_a, y_a))
print("b", mean_absolute_error(x_b, y_b))
print("c", mean_absolute_error(x_c, y_c))
print("alpha", mean_absolute_error(x_alpha, y_alpha))
print("beta", mean_absolute_error(x_beta, y_beta))
print("gamma", mean_absolute_error(x_gamma, y_gamma))
# plt.plot(x_a,x_a)
# plt.plot(x_a,y_a,'.')
# plt.savefig('a.png')
# plt.close()
comp = []
spg = []
samps_spg = []
samps_comp = []
x_spg = []
y_spg = []
x_Z = []
y_Z = []
x_lat = []
y_lat = []
for i in d:
    a1 = Atoms.from_dict(i["target"])
    a2 = Atoms.from_dict(i["predicted"])
    comp1 = a1.composition.reduced_formula
    comp2 = a2.composition.reduced_formula
    x_Z.append(a1.composition.weight)
    y_Z.append(a2.composition.weight)
    lat_1 = get_2d_lattice(atoms=i["target"])[1]
    lat_2 = get_2d_lattice(atoms=i["predicted"])[1]

    x_lat.append(lat_1)
    y_lat.append(lat_2)
    # x_Z.extend(a1.atomic_numbers)
    # y_Z.extend(a2.atomic_numbers)
    if comp1 == comp2:
        comp.append(i)
    else:
        print("different comp", comp1, comp2)
    samps_comp.append(i)
    try:
        spg1 = a1.get_spacegroup[0]
        spg2 = a2.get_spacegroup[0]
        x_spg.append(spg1)
        y_spg.append(spg2)
        if spg1 == spg2:
            spg.append(i)
        samps_spg.append(i)
        if spg1 == spg2 and comp1 == comp2:
            matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
            rms_dist = matcher.get_rms_anonymous(
                a1.pymatgen_converter(), a2.pymatgen_converter()
            )
            print(a1)
            print(a2)
            print(rms_dist)
            print(i["id"], spg1, spg2)
        # print('target',a1,a1.density,a1.spacegroup())
        # print('predicted',a2,a2.density,a2.spacegroup())
    except:
        pass
print("comp", len(comp), len(samps_comp), "spg", len(spg), len(samps_spg))
print("KLD a", kl_divergence(x_a, y_a))
print("KLD b", kl_divergence(x_b, y_b))
print("KLD c", kl_divergence(x_c, y_c))
print("KLD alpha", kl_divergence(x_alpha, y_alpha))
print("KLD beta", kl_divergence(x_beta, y_beta))
print("KLD gamma", kl_divergence(x_gamma, y_gamma))
print("KLD lat", kl_divergence(x_lat, y_lat))
print("KLD spg", kl_divergence(x_spg, y_spg))
print("EMD a", emd_distance(x_a, y_a))
print("EMD b", emd_distance(x_b, y_b))
print("EMD gamma", emd_distance(x_gamma, y_gamma))
print("EMD spg", emd_distance(x_spg, y_spg))
print("EMD lat", emd_distance(x_lat, y_lat))
print("Min Max a", min(x_a), max(x_a))
print("Min Max b", min(x_b), max(x_b))
print("Min Max c", min(x_c), max(x_c))
print("Min Max gamma", min(x_gamma), max(x_gamma))


plt.subplot(the_grid[1, 2])
print("xZ_", min(x_Z), max(x_Z))
weights_x = np.ones_like(x_Z) / len(x_Z) * 100
weights_y = np.ones_like(y_Z) / len(y_Z) * 100
plt.hist(
    x_Z,
    bins=np.arange(15, 2000, 100),
    weights=weights_x,
    label="target_wt",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_Z,
    bins=np.arange(15, 2000, 100),
    weights=weights_y,
    label="predicted_wt",
    color="plum",
    alpha=0.6,
)
plt.xlabel("Weight (AMU)")
plt.title("(f)")


plt.subplot(the_grid[1, 0])
weights_x = np.ones_like(x_spg) / len(x_spg) * 100
weights_y = np.ones_like(y_spg) / len(y_spg) * 100
plt.hist(
    x_spg,
    bins=np.arange(50, 220, 10),
    weights=weights_x,
    label="target_spg",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_spg,
    bins=np.arange(50, 220, 10),
    weights=weights_y,
    label="predicted_spg",
    color="plum",
    alpha=0.6,
)
plt.ylabel("Materials dist.")
plt.xlabel("Spacegroup number")
plt.title("(d)")


plt.subplot(the_grid[1, 1])
weights_x = np.ones_like(x_lat) / len(x_lat) * 100
weights_y = np.ones_like(y_lat) / len(y_lat) * 100
plt.hist(
    x_lat,
    bins=np.arange(0, 5, 0.5),
    weights=weights_x,
    label="target_lat",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_lat,
    bins=np.arange(0, 5, 0.5),
    weights=weights_y,
    label="predicted_lat",
    color="plum",
    alpha=0.6,
)
plt.xlabel("Bravais lattice")
plt.title("(e)")

plt.tight_layout()
plt.savefig("distribution.pdf")
plt.close()


print("Min Max spg", min(x_spg), max(x_spg))
x_Z = np.array(x_Z)
y_Z = np.array(y_Z)
mask = np.isfinite(x_Z) & np.isfinite(y_Z)
x_Z = x_Z[mask]
y_Z = y_Z[mask]
print("Min Max weight", min(x_Z), max(x_Z))
print("KLD weight", kl_divergence(x_Z, y_Z))
print("EMD weight", emd_distance(x_Z, y_Z))


# Load predictions
# d = loadjson('dft_2d_formula_based_dft_2d_formula_output_dir_dft_2d_unsloth/Llama-3.2-11B-Vision-Instructold/checkpoint-620/test_predictions.json')

# Prepare data arrays
labels = ["a", "b", "c", "alpha", "beta", "gamma"]
targets = [
    np.array(
        [
            i["target"]["abc"][j] if j < 3 else i["target"]["angles"][j - 3]
            for i in d
        ]
    )
    for j in range(6)
]
preds = [
    np.array(
        [
            (
                i["predicted"]["abc"][j]
                if j < 3
                else i["predicted"]["angles"][j - 3]
            )
            for i in d
        ]
    )
    for j in range(6)
]
bins = [
    np.arange(2, 7, 0.1),
    np.arange(2, 7, 0.1),
    np.arange(10, 30, 0.1),
    np.arange(0, 180, 10),
] * 2

# Plot setup
plt.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, figure=fig)

for idx, label in enumerate(labels):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.hist(
        targets[idx],
        bins=bins[idx],
        label=f"Target {label}",
        alpha=0.6,
        color="tab:blue",
    )
    ax.hist(
        preds[idx],
        bins=bins[idx],
        label=f"Predicted {label}",
        alpha=0.6,
        color="plum",
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Material Count")
    ax.set_title(f"Distribution of {label}")
    ax.legend()

fig.tight_layout()
plt.savefig("pretty_distribution_plot.png")
plt.close()
