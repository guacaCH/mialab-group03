"""Flip maps: WHERE in the brain does a feature group change the segmentation?

The ablation (cheap_experiments.py ablation) says HOW MUCH the Dice changes when a feature group is removed. A Dice
difference of 0.01 can hide two very different stories: a few voxels everywhere, or a whole band along one border.
A flip map shows the place. For every test voxel inside the brain mask we compare two forests:

    reference  : the 35-feature forest ("all35", = extended_plus)
    without g  : the same forest without the feature group g (retrained without the columns: --mode drop), or the
                 reference forest with the columns of g shuffled within the subject (--mode permute)

and mark the voxels whose predicted label changes:

    harm  = reference correct, without g wrong      (g was NEEDED for this voxel)
    help  = reference wrong,   without g correct    (g CONFUSED the forest at this voxel)
    other = both wrong, but with a different label   (counted, not mapped)

All test subjects are registered to the MNI atlas, so the voxel index is the same anatomical position in every
subject. Averaging the 0/1 harm maps over the 10 subjects (and the seeds) gives, per voxel, the FRACTION of
subjects x seeds in which removing g made the voxel wrong: 0.5 = in half of the cases. Overlaid on the atlas T1 this
is the picture "where does the forest need the location / the prior / T2 ...".

Concrete example: hippocampus voxel at the border to the amygdala. With all35 it is labelled Hippocampus (correct).
Without the prior distances the forest calls it Amygdala (wrong) -> harm = 1 at this voxel for this subject. If that
happens in 7 of 10 subjects, the harm map shows 0.7 there.

No forest is trained twice: in drop mode the predictions are taken from results/cheap/<experiment>/predictions/
when cheap_experiments.py saved them there (it does for every forest it trains), otherwise the forest is trained
here and its predictions are kept in results/flipmaps/<mode>/predictions/.

Usage (needs the feature matrices of "python extract_features.py --features extended_plus --seeds 0 1 2"):
    python flip_maps.py                                # drop mode, seeds 0 1 2, all groups of GROUPS
    python flip_maps.py --seeds 0 --groups location prior     # quick look
    python flip_maps.py --mode permute                 # shuffle instead of retrain (explains the trained forest)
    python flip_maps.py --report_only                  # only redo the tables and plots from the saved maps
    python flip_maps.py --borders                      # H4: flip rate by distance to the structure borders

Writes to results/flipmaps/<mode>/:
    <group>_harm.nii.gz, <group>_help.nii.gz   mean flip maps in atlas space (open in 3D Slicer on the atlas T1)
    flip_summary_per_subject.csv, flip_summary.csv
                         per group: flipped voxels per subject, split by ground-truth structure, harm/help, and the
                         share of flips on a label border (compared with the share of border voxels in the brain)
    flipmaps_<view>.png  one row per group, harm (red) and help (blue) on the atlas T1, the same slice for all groups
    flipmaps_<group>.png all three views of one group
"""
import argparse
import glob
import os
import sys
import timeit

import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from scipy import ndimage

# pymia uses np.in1d, which newer numpy versions removed; np.isin does the same
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import mialab.filtering.feature_pool as fpool  # noqa: E402
import mialab.utilities.pipeline_utilities as putil  # noqa: E402
import run_experiment as rexp  # noqa: E402

# the forest of all experiments (50 trees, depth 20, sqrt features per split), never changed
FOREST = dict(rexp.DEFAULT_CONFIG['forest'])
LABEL_NAMES = rexp.LABEL_NAMES                       # 0 Background, 1 WhiteMatter, 2 GreyMatter, 3 Hippocampus, ...
IMAGE_FAMILIES = ('intensity', 'gradient', 'smoothing', 'log', 'localstats', 'combination', 'symmetry')
LOCATION_FAMILIES = ('spatial', 'distance', 'prior', 'priordistance')

# the feature groups that get a flip map: name -> rule that picks the columns (family tag, modality or name)
# image / location = the two main groups of the hypothesis; the rest are the families with the largest ablation
# effects (prior, priordistance, T2) and spatial (the coordinates, for comparison with the prior).
GROUPS = {
    'location': lambda c: c['family'] in LOCATION_FAMILIES,
    'image': lambda c: c['family'] in IMAGE_FAMILIES,
    'prior': lambda c: c['family'] == 'prior',
    'priordistance': lambda c: c['family'] == 'priordistance',
    'spatial': lambda c: c['family'] == 'spatial',
    'T2': lambda c: c['family'] in IMAGE_FAMILIES and (c['modality'] == 'T2' or 'T2' in c['name']),
}
# the same forests in cheap_experiments.py (drop mode): group -> (experiment, run name). "Without location" is the
# run image_only of the groups experiment, and so on. The reference all35 is in the ablation experiment.
CHEAP_RUNS = {'all35': ('ablation', 'all35'), 'location': ('groups', 'image_only'),
              'image': ('groups', 'location_only'), 'prior': ('ablation', 'no_prior'),
              'priordistance': ('ablation', 'no_priordistance'), 'spatial': ('ablation', 'no_spatial'),
              'T2': ('ablation', 'no_T2')}


# =================================================================================================================
# Data: the cached feature matrices of extract_features.py (the same files as cheap_experiments.py)
# =================================================================================================================
def train_path(features_dir: str, seed: int) -> str:
    """Seed 0 -> train.npz, seed s -> train_seed<s>.npz (the naming of extract_features.py --seeds)."""
    return os.path.join(features_dir, 'train.npz' if seed == 0 else f'train_seed{seed}.npz')


def load_data(features_dir: str, seeds: list) -> dict:
    """Loads the training voxels of every seed and the brain voxels of every test subject.

    Returns a dict with
        names: the column names (matrix order); columns: the 35 used columns (control_random left out)
        tags:  column name -> its feature-pool tags (family, modality, name)
        trains: seed -> (X, y);  tests: subject -> dict(X, y, brain_index)
    """
    trains, names = {}, None
    for seed in seeds:
        path = train_path(features_dir, seed)
        if not os.path.exists(path):
            raise SystemExit(f'{path} missing: run "python extract_features.py --features extended_plus '
                             f'--seeds {" ".join(map(str, seeds))}" first')
        d = np.load(path)
        names = [str(n) for n in d['feature_names']]
        trains[seed] = (d['X'].astype(np.float32), d['y'])
    tests = {}
    for path in sorted(glob.glob(os.path.join(features_dir, 'test_*.npz'))):
        d = np.load(path)
        tests[os.path.basename(path)[5:-4]] = {'X': d['X'].astype(np.float32), 'y': d['y'].astype(np.uint8),
                                               'brain_index': d['brain_index']}
    pool = {c['name']: c for c in fpool.all_columns()}
    tags = {n: dict(pool.get(n, {'family': None, 'modality': None}), name=n) for n in names}
    columns = [n for n in names if tags[n]['family'] in IMAGE_FAMILIES + LOCATION_FAMILIES]
    print(f'{len(trains)} seeds, {len(tests)} test subjects, {len(columns)} features')
    return {'names': names, 'columns': columns, 'tags': tags, 'trains': trains, 'tests': tests}


# =================================================================================================================
# Predictions: one forest per run and seed, labels of the brain voxels of every test subject
# =================================================================================================================
def fit_forest(data: dict, columns: list, seed: int):
    """The fixed forest on the training voxels of `seed` (random_state = seed, like run_experiment.py)."""
    x, y = data['trains'][seed]
    idx = [data['names'].index(c) for c in columns]
    return sk_ensemble.RandomForestClassifier(random_state=seed, **FOREST).fit(x[:, idx], y)


def predict(data: dict, forest, columns: list, seed: int, permute: list = None) -> dict:
    """Predicts every test subject; returns subject -> uint8 labels of the brain voxels.

    permute: columns to shuffle TOGETHER within each subject before predicting (permute mode). The same row order
             for all of them keeps their relations, but breaks their link to the voxel's position and label.
    """
    idx = [data['names'].index(c) for c in columns]
    predictions = {}
    for i, (subject, test) in enumerate(data['tests'].items()):
        x = test['X'][:, idx]
        if permute:
            cols = [columns.index(c) for c in permute]
            order = np.random.default_rng([seed, i]).permutation(len(x))   # own generator: reproducible
            x[:, cols] = x[order][:, cols]
        predictions[subject] = forest.predict(x).astype(np.uint8)
    return predictions


def cached_predictions(out_dir: str, cheap_dir: str, run: str, seed: int, compute) -> dict:
    """The predictions of one run and seed, from the first place that has them:

    1. results/cheap/<experiment>/predictions/<name>_seed<s>.npz (only for runs that cheap_experiments.py has)
    2. results/flipmaps/<mode>/predictions/<run>_seed<s>.npz (computed earlier by this script)
    3. compute() now, and save to 2. (so a stopped script resumes without training again)
    """
    if cheap_dir and run in CHEAP_RUNS:
        experiment, name = CHEAP_RUNS[run]
        path = os.path.join(cheap_dir, experiment, 'predictions', f'{name}_seed{seed}.npz')
        if os.path.exists(path):
            return dict(np.load(path))
    path = os.path.join(out_dir, 'predictions', f'{run}_seed{seed}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = timeit.default_timer()
    predictions = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **predictions)
    print(f'  {run} seed {seed}: {timeit.default_timer() - t0:.0f} s')
    return predictions


# =================================================================================================================
# Flip maps and the summary table
# =================================================================================================================
def border_mask(labels: np.ndarray) -> np.ndarray:
    """True where a voxel has a 6-neighbour with another label (the voxel lies on a structure border).

    grey dilation / erosion with a 3x3x3 cross: max / min label of the 6 neighbours; differs from the voxel itself
    exactly when a neighbour has another label.
    """
    cross = ndimage.generate_binary_structure(3, 1)
    return (ndimage.grey_dilation(labels, footprint=cross) != labels) | \
           (ndimage.grey_erosion(labels, footprint=cross) != labels)


def build_maps(data: dict, out_dir: str, cheap_dir: str, groups: list, seeds: list, mode: str):
    """Gets the reference and the group predictions, accumulates the harm / help maps and the summary rows."""
    shape = putil.atlas_t1.GetSize()[::-1]                 # numpy order z, y, x
    n_voxels = int(np.prod(shape))
    harm = {g: np.zeros(n_voxels, np.float32) for g in groups}
    help_ = {g: np.zeros(n_voxels, np.float32) for g in groups}
    rows = []
    cheap = cheap_dir if mode == 'drop' else None          # cheap_experiments.py only has retrained (drop) runs

    # ground truth image and its borders per subject (the same for every run)
    borders = {}
    for subject, test in data['tests'].items():
        gt = np.zeros(n_voxels, np.uint8)
        gt[test['brain_index']] = test['y']
        borders[subject] = border_mask(gt.reshape(shape)).ravel()[test['brain_index']]

    for seed in seeds:
        reference_forest = {}                               # fitted at most once per seed, only if needed

        def reference():
            if 'f' not in reference_forest:
                reference_forest['f'] = fit_forest(data, data['columns'], seed)
            return reference_forest['f']

        ref = cached_predictions(out_dir, cheap, 'all35', seed, lambda: predict(data, reference(),
                                                                                data['columns'], seed))
        for group in groups:
            members = [c for c in data['columns'] if GROUPS[group](data['tags'][c])]
            if mode == 'drop':
                kept = [c for c in data['columns'] if c not in members]
                compute = lambda: predict(data, fit_forest(data, kept, seed), kept, seed)          # noqa: E731
            else:
                compute = lambda: predict(data, reference(), data['columns'], seed, permute=members)  # noqa: E731
            other = cached_predictions(out_dir, cheap, group, seed, compute)

            for subject, test in data['tests'].items():
                y, a, b = test['y'], ref[subject], other[subject]
                is_harm = (a == y) & (b != y)
                is_help = (a != y) & (b == y)
                harm[group][test['brain_index'][is_harm]] += 1
                help_[group][test['brain_index'][is_help]] += 1
                border = borders[subject]
                row = {'group': group, 'n_columns': len(members), 'seed': seed, 'subject': subject,
                       'brain_voxels': len(y), 'harm': int(is_harm.sum()), 'help': int(is_help.sum()),
                       'other': int(((a != y) & (b != y) & (a != b)).sum()),
                       'harm_on_border': int((is_harm & border).sum()), 'help_on_border': int((is_help & border).sum()),
                       'border_share_of_brain': float(border.mean())}
                for label, name in LABEL_NAMES.items():           # flips split by the TRUE structure of the voxel
                    row[f'harm_{name}'] = int((is_harm & (y == label)).sum())
                    row[f'help_{name}'] = int((is_help & (y == label)).sum())
                rows.append(row)

    n_cases = len(seeds) * len(data['tests'])                     # subjects x seeds that went into each map
    for group in groups:
        for kind, maps in (('harm', harm), ('help', help_)):
            image = sitk.GetImageFromArray((maps[group] / n_cases).reshape(shape))
            image.CopyInformation(putil.atlas_t1)
            sitk.WriteImage(image, os.path.join(out_dir, f'{group}_{kind}.nii.gz'))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'flip_summary_per_subject.csv'), index=False)


def summarize(out_dir: str) -> pd.DataFrame:
    """Mean per group over subjects and seeds, as printed and saved in flip_summary.csv."""
    rows = pd.read_csv(os.path.join(out_dir, 'flip_summary_per_subject.csv'))
    table = rows.drop(columns=['subject', 'seed']).groupby('group', sort=False).mean()
    out = pd.DataFrame({'columns': table.n_columns.astype(int),
                        'harm_voxels': table.harm.round(0), 'help_voxels': table.help.round(0),
                        'net (harm-help)': (table.harm - table.help).round(0),
                        'harm_on_border_%': (100 * table.harm_on_border / table.harm).round(1),
                        'border_share_of_brain_%': (100 * table.border_share_of_brain).round(1)})
    for name in LABEL_NAMES.values():
        out[f'net_{name}'] = (table[f'harm_{name}'] - table[f'help_{name}']).round(0)
    out.to_csv(os.path.join(out_dir, 'flip_summary.csv'))
    print('\nFlipped voxels per test subject (mean over subjects and seeds); net = harm - help, by true structure')
    print(out.to_string())
    return out


# =================================================================================================================
# Plots: harm (red) and help (blue) on the atlas T1, the same slices for all groups
# =================================================================================================================
def slices(data: dict) -> dict:
    """Slice positions (z, y, x index) through the centre of hippocampus + amygdala of the test ground truths,
    in one hemisphere for the sagittal view, so all three views cut the small structures."""
    shape = putil.atlas_t1.GetSize()[::-1]
    count = np.zeros(int(np.prod(shape)), np.float32)
    for test in data['tests'].values():
        count[test['brain_index'][np.isin(test['y'], (3, 4))]] += 1
    z, y, x = np.nonzero(count.reshape(shape))
    one_side = x < shape[2] / 2
    return {'axial': int(np.median(z)), 'coronal': int(np.median(y)), 'sagittal': int(np.median(x[one_side]))}


def cut(volume: np.ndarray, view: str, position: int) -> np.ndarray:
    """One 2-D slice in display orientation (superior / anterior up with origin='lower')."""
    if view == 'axial':
        return volume[position]                    # (y, x)
    if view == 'coronal':
        return volume[:, position]                 # (z, x)
    return volume[:, :, position]                  # (z, y)


def overlay(ax, t1, harm, help_, vmax, title=''):
    """Grey atlas T1, harm in red and help in blue; opacity grows with the flip fraction."""
    import matplotlib.colors as mcolors
    ax.imshow(t1, cmap='gray', origin='lower')
    for values, color in ((harm, '#d62728'), (help_, '#1f77b4')):
        rgba = np.zeros(values.shape + (4,))
        rgba[..., :3] = mcolors.to_rgb(color)
        rgba[..., 3] = np.clip(values / vmax, 0, 1)            # 0 = invisible, vmax and above = opaque
        ax.imshow(rgba, origin='lower')
    ax.set_title(title, fontsize=9)
    ax.axis('off')


# readable names of the groups for the plot titles
GROUP_TITLES = {'location': 'location features', 'image': 'image features', 'prior': 'label prior',
                'priordistance': 'prior distances', 'spatial': 'coordinates', 'T2': 'all T2 features'}
LEGEND = ('red = correct with all 35 features, wrong without them (the features were NEEDED here)\n'
          'blue = wrong with all 35 features, correct without them (the features MISLED the forest here)\n'
          'colour strength = share of the 30 cases (10 test subjects x 3 seeds) in which this happens; '
          'transparent = never, full colour = {:.0%} or more')


def plot(out_dir: str, groups: list, positions: dict, vmax: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t1 = sitk.GetArrayFromImage(putil.atlas_t1)
    maps = {g: {k: sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out_dir, f'{g}_{k}.nii.gz')))
                for k in ('harm', 'help')} for g in groups}
    views = list(positions)
    # number of removed features per group, for the titles (from the summary of build_maps)
    summary = pd.read_csv(os.path.join(out_dir, 'flip_summary.csv'), index_col=0)
    title = {g: f'remove {GROUP_TITLES.get(g, g)} ({int(summary.loc[g, "columns"])})' for g in groups}

    # one figure per view: groups side by side, the same slice -> the groups are directly comparable
    for view in views:
        fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 3.2))
        for ax, g in zip(np.atleast_1d(axes), groups):
            overlay(ax, cut(t1, view, positions[view]), cut(maps[g]['harm'], view, positions[view]),
                    cut(maps[g]['help'], view, positions[view]), vmax, title[g])
        fig.suptitle(f'What changes when a feature group is removed and the forest retrained ({view} slice)',
                     fontsize=11)
        fig.text(0.5, 0.01, LEGEND.format(vmax), ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(os.path.join(out_dir, f'flipmaps_{view}.png'), dpi=200, facecolor='white')
        plt.close(fig)

    # one figure per group: all three views
    for g in groups:
        fig, axes = plt.subplots(1, len(views), figsize=(3.2 * len(views), 3.4))
        for ax, view in zip(axes, views):
            overlay(ax, cut(t1, view, positions[view]), cut(maps[g]['harm'], view, positions[view]),
                    cut(maps[g]['help'], view, positions[view]), vmax, f'{view} {positions[view]}')
        fig.suptitle(f'{title[g][0].upper() + title[g][1:]} and retrain: where the segmentation changes',
                     fontsize=11)
        fig.text(0.5, 0.01, LEGEND.format(vmax), ha='center', va='bottom', fontsize=7)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(os.path.join(out_dir, f'flipmaps_{g}.png'), dpi=200, facecolor='white')
        plt.close(fig)
    print('plots written to', out_dir)


# =================================================================================================================
# H4: flip rate as a function of the distance to the nearest structure border
# =================================================================================================================
# distance bins in mm (1 voxel = 1 mm in atlas space): upper edges and labels. 0 = the voxel itself lies on a border
# (a 6-neighbour has another label); 1 = next to it (also the diagonal neighbours, sqrt(2) and sqrt(3) mm), ...
DISTANCE_BINS = [(0.5, '0'), (1.8, '1'), (2.5, '2'), (4.5, '3-4'), (np.inf, '≥5')]


def border_distance(labels: np.ndarray, spacing: tuple) -> np.ndarray:
    """Distance (mm) of every voxel to the nearest border voxel of the ground truth (0 on the border).

    distance_transform_edt measures, for every non-zero voxel, the distance to the nearest zero voxel; the border
    voxels are made the zeros. sampling = the voxel spacing in numpy order (z, y, x), so the result is in mm.
    """
    return ndimage.distance_transform_edt(~border_mask(labels), sampling=spacing)


def border_profile(data: dict, out_dir: str, cheap_dir: str, groups: list, seeds: list, mode: str) -> pd.DataFrame:
    """Flip rate per distance bin: flipped voxels / all brain voxels in the bin, per group (only saved predictions).

    Why a RATE: 60 % of the brain voxels lie on a border (the cortex is thin), so "most flips are on a border" is true
    for any random set of voxels. Dividing by the number of voxels in the bin removes that base rate: a group that
    acts at borders has a much higher rate at 0 mm than at >= 5 mm, a group that acts inside regions has a flat or
    rising profile.
    Example: 1,000 harm flips among 500,000 border voxels = 0.2 %; 50 among 100,000 voxels at >= 5 mm = 0.05 %:
    removing the group hurts 4x more on the border than deep inside.
    Also written: the error rate of the reference all35 per bin (where the forest is wrong anyway).
    """
    shape = putil.atlas_t1.GetSize()[::-1]
    spacing = putil.atlas_t1.GetSpacing()[::-1]                    # numpy order z, y, x
    edges = [e for e, _ in DISTANCE_BINS]
    names = [n for _, n in DISTANCE_BINS]
    rows = []
    no_compute = lambda: (_ for _ in ()).throw(SystemExit('predictions missing: run flip_maps.py first'))  # noqa
    for subject, test in data['tests'].items():
        gt = np.zeros(int(np.prod(shape)), np.uint8)
        gt[test['brain_index']] = test['y']
        distance = border_distance(gt.reshape(shape), spacing).ravel()[test['brain_index']]
        bin_of = np.searchsorted(edges, distance, side='left')     # index of the distance bin of every brain voxel
        voxels = np.bincount(bin_of, minlength=len(edges))
        y = test['y']
        for seed in seeds:
            ref = cached_predictions(out_dir, cheap_dir if mode == 'drop' else None, 'all35', seed, no_compute)[subject]
            for b, name in enumerate(names):
                rows.append({'group': 'all35 errors', 'subject': subject, 'seed': seed, 'bin': name,
                             'voxels': voxels[b], 'count': int(np.sum((ref != y) & (bin_of == b)))})
            for group in groups:
                other = cached_predictions(out_dir, cheap_dir if mode == 'drop' else None, group, seed,
                                           no_compute)[subject]
                harm = (ref == y) & (other != y)
                help_ = (ref != y) & (other == y)
                for b, name in enumerate(names):
                    in_bin = bin_of == b
                    rows.append({'group': group, 'subject': subject, 'seed': seed, 'bin': name, 'voxels': voxels[b],
                                 'harm': int(np.sum(harm & in_bin)), 'help': int(np.sum(help_ & in_bin))})
    table = pd.DataFrame(rows)
    # rate per bin: pooled over subjects and seeds (sum of flips / sum of voxels)
    sums = table.groupby(['group', 'bin'], sort=False)[['voxels', 'harm', 'help', 'count']].sum()
    profile = pd.DataFrame({'voxels_per_case': sums.voxels / (len(seeds) * len(data['tests'])),
                            'harm_rate_%': 100 * sums.harm / sums.voxels, 'help_rate_%': 100 * sums.help / sums.voxels,
                            'error_rate_%': 100 * sums['count'] / sums.voxels}).reset_index()
    profile.to_csv(os.path.join(out_dir, 'border_profile.csv'), index=False)
    print('\nFlip rate (%) per distance to the nearest structure border (mm)')
    print(profile.pivot(index='group', columns='bin', values='harm_rate_%')[names].round(3).to_string())
    return profile


def plot_border_profile(profile: pd.DataFrame, out_dir: str, groups: list):
    """Two panels: rate of 'needed' flips (harm) and of 'misled' flips (help) per distance bin, one line per group."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for _, n in DISTANCE_BINS]
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # categorical slots 1-6, fixed order
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, (column, title) in zip(axes, (('harm_rate_%', 'correct -> wrong when removed (features were needed)'),
                                          ('help_rate_%', 'wrong -> correct when removed (features misled)'))):
        for group, color in zip(groups, colors):
            g = profile[profile.group == group].set_index('bin').reindex(names)
            ax.plot(x, g[column], marker='o', ms=6, lw=2, color=color, label=f'remove {GROUP_TITLES.get(group, group)}')
        ref = profile[profile.group == 'all35 errors'].set_index('bin').reindex(names)
        ax.plot(x, ref['error_rate_%'], ls='--', lw=1.5, color='#5d6878', label='all 35 features: error rate')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel('distance to the nearest structure border (mm)')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', which='both', color='#eceef2', lw=0.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('share of the brain voxels in the bin (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle('Where do the feature groups act? Flip rate by distance to the structure borders', fontsize=11,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(out_dir, 'border_profile.png'), dpi=200, facecolor='white')
    plt.close(fig)
    print('plot:', os.path.join(out_dir, 'border_profile.png'))


def main(args):
    putil.load_atlas_images(args.data_atlas_dir)              # atlas geometry of all test images
    out_dir = os.path.join(args.out_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    data = load_data(args.features_dir, args.seeds)
    if args.borders:
        # H4 only: needs the saved predictions of an earlier run, trains nothing
        plot_border_profile(border_profile(data, out_dir, args.cheap_dir, args.groups, args.seeds, args.mode),
                            out_dir, args.groups)
        return
    if not args.report_only:
        build_maps(data, out_dir, args.cheap_dir, args.groups, args.seeds, args.mode)
    summarize(out_dir)
    plot(out_dir, args.groups, slices(data), args.vmax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxel-wise flip maps per feature group',
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--mode', choices=('drop', 'permute'), default='drop',
                        help='drop = retrain without the group (like the ablation), permute = shuffle it')
    parser.add_argument('--groups', nargs='+', choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--vmax', type=float, default=0.5, help='flip fraction drawn fully opaque')
    parser.add_argument('--report_only', action='store_true')
    parser.add_argument('--borders', action='store_true',
                        help='H4: flip rate by distance to the nearest structure border (saved predictions only)')
    parser.add_argument('--features_dir', default=os.path.join(SCRIPT_DIR, 'features_extended_plus'))
    parser.add_argument('--data_atlas_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'atlas'))
    parser.add_argument('--cheap_dir', default=os.path.join(SCRIPT_DIR, 'results', 'cheap'),
                        help='results of cheap_experiments.py, whose saved predictions are reused')
    parser.add_argument('--out_dir', default=os.path.join(SCRIPT_DIR, 'results', 'flipmaps'))
    main(parser.parse_args())
