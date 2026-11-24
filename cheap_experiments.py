"""Cheap experiments: every experiment that only changes WHICH feature columns the forest sees.

All of them reuse the feature matrices that extract_features.py writes once, so no feature is computed again.
A forest then takes about 1-2 minutes (fit + predict 10 test subjects) instead of ~20 minutes per seed with
run_experiment.py. See "What to run next" and "Which evaluations to add" in the plan doc.

Step 0, once (about 20-30 min, about 2.7 GB in features_extended_plus/):
    python extract_features.py --features extended_plus --seeds 0 1 2
(one training file per seed: train.npz for seed 0, train_seed1.npz, train_seed2.npz; the test files are shared.
 If features_extended_plus/ already exists, only add the seeds: ... --seeds 1 2 --train_only)

The experiments (each one is a subcommand, writes to results/cheap/<subcommand>/ and prints its report at the end):
    python cheap_experiments.py check                 # all 35 columns = extended_plus; the 3 seeds give the noise floor
    python cheap_experiments.py subsets               # exp. 1: random image subsets at fixed k, without / with location
    python cheap_experiments.py noise                 # exp. 2: all 35 features + 35 / 100 pure-noise columns
    python cheap_experiments.py groups                # exp. 3: T1 only, image only, location only, image + location
    python cheap_experiments.py ablation              # exp. 4: extended_plus without one family at a time (and without T2)
    python cheap_experiments.py permutation           # grouped permutation importance of the 35-feature forest
    python cheap_experiments.py mi                    # mutual information of every feature with every structure

    python cheap_experiments.py report <subcommand>   # only the report (tables + plot) of what is already finished

Every subcommand can be stopped and started again: finished forests are skipped.

Seeds: every forest experiment runs on seeds 0 1 2 (--seeds). Seed s means the training voxels of seed s
(train_seed<s>.npz) AND the forest random_state s, exactly like results/<experiment>/seed_<s>/ of run_experiment.py.
So 3 seeds take 3 times as long as one; --seeds 0 gives a quick first look.

Output of the forest experiments (check, subsets, noise, groups, ablation), in results/cheap/<subcommand>/:
    runs.csv       one row per forest: name, seed, the feature columns, timings
    metrics.csv    one row per forest x subject x label x metric (DICE, HDRFDST95), like run_experiment.py
    predictions/   <name>_seed<s>.npz: the predicted labels of the brain voxels of every forest (reused by flip_maps.py)
    effects.csv    (noise, groups, ablation) the difference to the reference run per structure and metric, with a
                   95 % two-level bootstrap confidence interval (subjects AND seeds), the mean of every seed, and the
                   flags relevant / negligible / seeds_agree
    *.png          the plot of the report

How the effects are judged (see "Which evaluations to add" in the plan doc):
    * mean_diff = run - reference, per test subject first (seeds averaged), then the mean over the subjects.
    * ci_low, ci_high: 95 % bootstrap confidence interval of mean_diff (10,000 resamples of the subjects AND the
      seeds, so both the choice of test patients and the training randomness are in it, see bootstrap_ci).
    * negligible = the whole interval lies inside +- the minimal relevant difference (equivalence: "redundant given
      the other features"); seeds_agree = every seed alone gives the same sign.
    * relevant = the whole interval lies beyond the minimal relevant difference (--min_dice 0.01, --min_hd 1 mm).
      With 10 subjects this replaces the Wilcoxon p-value, which can't go below 0.002 and ignores the size.
    * The noise floor (check): how much the mean Dice of the SAME columns moves between the seeds. Every seed has its
      own training voxels and its own forest, so this is the full seed noise of run_experiment.py.

Notes:
    * Only voxels inside the brain mask are predicted (outside it everything is background after skull stripping);
      the check run shows that this gives the Dice of run_experiment.py within about 0.001.
    * Random draws (subsets, noise columns, permutations) use their own generators with fixed seeds, never
      np.random, so every result can be reproduced exactly.
"""
import argparse
import glob
import json
import os
import sys
import timeit

import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble

# pymia uses np.in1d, which newer numpy versions removed; np.isin does the same (no effect on older numpy)
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import mialab.filtering.feature_pool as fpool  # noqa: E402
import mialab.utilities.pipeline_utilities as putil  # noqa: E402
import run_experiment as rexp  # noqa: E402
import extract_features as fx  # noqa: E402  (only for train_path: where the training file of a seed is)

# The two feature groups (see the pipeline diagram).
# image: what the voxel looks like in the MRI; location: where the voxel is (* = built from the training labels)
IMAGE_FAMILIES = ('intensity', 'gradient', 'smoothing', 'log', 'localstats', 'combination', 'symmetry')
LOCATION_FAMILIES = ('spatial', 'distance', 'prior', 'priordistance')

# the forest of every experiment: the runner's defaults (50 trees, depth 20, sqrt(n) features per split)
FOREST = dict(rexp.DEFAULT_CONFIG['forest'])

LABELS = ['WhiteMatter', 'GreyMatter', 'Thalamus', 'Hippocampus', 'Amygdala']
LABEL_IDS = {name: i for i, name in rexp.LABEL_NAMES.items()}      # 'WhiteMatter' -> 1, ...
METRICS = {'DICE': 'Dice', 'HDRFDST95': 'HD95 (mm)'}
LOWER_IS_BETTER = {'HDRFDST95'}


# =================================================================================================================
# Data: the cached feature matrices of extract_features.py
# =================================================================================================================
class Data:
    """The training matrix, the test matrices and the ground truth images, loaded once per call of the script.

    Attributes:
        names (list of str): the column names, in matrix order
        family (dict): column name -> family tag of the feature pool ('intensity', 'prior', ...)
        modality (dict): column name -> 'T1', 'T2' or None
        groups (dict): 'image' / 'location' -> list of column names
        trains (dict): seed -> {X (training voxels x features, float32), y (labels)}; one training set per seed
        tests (dict): subject -> {X, y, brain_index}; brain_index = position of each row in the flattened image
        ground_truths (dict): subject -> ground truth image in atlas space (needed for HD95)
    """

    def __init__(self, features_dir: str, data_atlas_dir: str, seeds: list):
        putil.load_atlas_images(data_atlas_dir)          # all images are in atlas space: the atlas gives the geometry
        # one training set per seed (about 20 MB each, so all of them fit in memory)
        self.trains = {}
        for seed in seeds:
            path = fx.train_path(features_dir, seed)
            if not os.path.exists(path):
                raise SystemExit(f'{path} not found: run "python extract_features.py --features extended_plus '
                                 f'--seeds {" ".join(map(str, seeds))}" first')
            data = np.load(path)
            names = [str(n) for n in data['feature_names']]
            if self.trains and names != self.names:
                raise SystemExit(f'{path} has other columns than the other seeds: run extract_features.py again')
            self.names = names
            self.trains[seed] = {'X': data['X'].astype(np.float32), 'y': data['y']}

        self.tests = {}
        for path in sorted(glob.glob(os.path.join(features_dir, 'test_*.npz'))):
            data = np.load(path)
            if [str(n) for n in data['feature_names']] != self.names:
                raise SystemExit(f'{path} has other columns than the training files: run extract_features.py again')
            subject = os.path.basename(path)[len('test_'):-len('.npz')]
            # float32 halves the memory; the forest computes in float32 anyway
            self.tests[subject] = {'X': data['X'].astype(np.float32), 'y': data['y'].astype(np.uint8),
                                   'brain_index': data['brain_index']}
        if not self.tests:
            raise SystemExit(f'no test_<subject>.npz in {features_dir}')

        tags = {c['name']: c for c in fpool.all_columns()}
        self.family = {n: tags[n]['family'] if n in tags else None for n in self.names}
        self.modality = {n: tags[n]['modality'] if n in tags else None for n in self.names}
        # 'control_random' (the noise control of extract_features.py) is in neither group and never used
        self.groups = {'image': [n for n in self.names if self.family[n] in IMAGE_FAMILIES],
                       'location': [n for n in self.names if self.family[n] in LOCATION_FAMILIES]}
        self.ground_truths = {s: to_image(t['y'], t['brain_index']) for s, t in self.tests.items()}
        print(f'seeds {list(self.trains)}: ' + ', '.join(f'{len(t["y"])}' for t in self.trains.values())
              + f' training voxels, {len(self.tests)} test subjects, '
              f'{len(self.groups["image"])} image + {len(self.groups["location"])} location features')

    @property
    def train(self) -> dict:
        """The training set of the first seed, for the analyses with a single forest (permutation, mi)."""
        return next(iter(self.trains.values()))

    @property
    def all_features(self) -> list:
        """The 35 features of extended_plus (image + location), in matrix order."""
        return [n for n in self.names if n in self.groups['image'] or n in self.groups['location']]

    def columns_of_set(self, feature_set: str) -> list:
        """The columns of a named set of the feature pool, e.g. 'extended' (29 features)."""
        wanted = {c['name'] for c in fpool.select_columns(feature_set)}
        return [n for n in self.names if n in wanted]


def to_image(labels_in_brain: np.ndarray, brain_index: np.ndarray) -> sitk.Image:
    """Writes the labels of the brain voxels back into a full image in atlas space (background elsewhere)."""
    reference = putil.atlas_t1
    array = np.zeros(reference.GetNumberOfPixels(), dtype=np.uint8)
    array[brain_index] = labels_in_brain
    image = sitk.GetImageFromArray(array.reshape(reference.GetSize()[::-1]))  # numpy order is z, y, x
    image.CopyInformation(reference)
    return image


def noise_columns(n_rows: int, n_noise: int, key: int) -> np.ndarray:
    """n_noise columns of N(0, 1) noise; the same key always gives the same noise (own generator, not np.random)."""
    return np.random.default_rng(key).standard_normal((n_rows, n_noise), dtype=np.float32)


# =================================================================================================================
# One forest: fit on some columns, predict the test subjects, evaluate Dice and HD95
# =================================================================================================================
def run_forest(data: Data, columns: list, seed: int, n_noise: int = 0, save_path: str = None) -> tuple:
    """Trains the fixed forest on the given columns (+ n_noise noise columns) and evaluates it on all test subjects.

    save_path: if given, the predicted labels of the brain voxels of every test subject are saved there
               (npz, one uint8 array per subject, ~3 MB compressed per forest). Voxel-level analyses such as
               flip_maps.py then reuse them instead of training the forest again.

    Returns:
        rows (list of dict): subject, label, metric, value
        timing (dict): seconds for fitting and predicting
    """
    idx = [data.names.index(c) for c in columns]                  # column positions of the chosen features
    train = data.trains[seed]                                     # the training voxels of this seed
    x_train = train['X'][:, idx]
    if n_noise:
        x_train = np.hstack([x_train, noise_columns(len(x_train), n_noise, key=0)])

    forest = sk_ensemble.RandomForestClassifier(random_state=seed, **FOREST)
    t0 = timeit.default_timer()
    forest.fit(x_train, train['y'])
    t_fit = timeit.default_timer() - t0
    del x_train

    evaluator = putil.init_evaluator()                            # Dice + HD95 for the 5 structures
    predictions = {}
    t0 = timeit.default_timer()
    for i, (subject, test) in enumerate(data.tests.items(), 1):
        x_test = test['X'][:, idx]
        if n_noise:
            x_test = np.hstack([x_test, noise_columns(len(x_test), n_noise, key=i)])  # other noise per subject
        prediction = forest.predict(x_test).astype(np.uint8)
        evaluator.evaluate(to_image(prediction, test['brain_index']), data.ground_truths[subject], subject)
        predictions[subject] = prediction
        del x_test
    t_predict = timeit.default_timer() - t0

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **predictions)
    rows = [{'subject': r.id_, 'label': r.label, 'metric': r.metric, 'value': r.value} for r in evaluator.results]
    return rows, {'fit_s': round(t_fit, 1), 'predict_s': round(t_predict, 1)}


def prediction_path(out_dir: str, name: str, seed: int) -> str:
    """Where the predicted labels of one forest are kept: results/cheap/<experiment>/predictions/<name>_seed<s>.npz"""
    return os.path.join(out_dir, 'predictions', f'{name}_seed{seed}.npz')


def run_all(data: Data, runs: list, out_dir: str):
    """Runs every planned forest that is not finished yet; appends its results to runs.csv and metrics.csv.

    A run is a dict with 'name', 'seed', 'columns' and optionally 'n_noise' and any extra tags (e.g. 'k', 'draw').
    """
    os.makedirs(out_dir, exist_ok=True)
    runs_path, metrics_path = os.path.join(out_dir, 'runs.csv'), os.path.join(out_dir, 'metrics.csv')
    done = set()
    if os.path.exists(runs_path):
        finished = pd.read_csv(runs_path)
        done = set(zip(finished['name'], finished['seed']))
    todo = [r for r in runs if (r['name'], r['seed']) not in done]
    print(f'{len(runs)} forests planned, {len(runs) - len(todo)} already finished, {len(todo)} to run')

    for i, run in enumerate(todo, 1):
        n_features = len(run['columns']) + run.get('n_noise', 0)
        print(f'[{i}/{len(todo)}] {run["name"]} (seed {run["seed"]}): {n_features} features')
        # predictions/<name>_seed<s>.npz: kept for voxel-level analyses (flip_maps.py), see run_forest
        rows, timing = run_forest(data, run['columns'], run['seed'], run.get('n_noise', 0),
                                  save_path=prediction_path(out_dir, run['name'], run['seed']))

        tags = {k: v for k, v in run.items() if k != 'columns'}
        tags['n_features'] = n_features
        metrics = pd.DataFrame(rows)
        for key, value in reversed(list(tags.items())):          # identify the forest in every row
            metrics.insert(0, key, value)
        # append, so that a stopped script keeps everything finished so far
        metrics.to_csv(metrics_path, mode='a', index=False, header=not os.path.exists(metrics_path))
        pd.DataFrame([dict(tags, columns=json.dumps(run['columns']), **timing)]) \
            .to_csv(runs_path, mode='a', index=False, header=not os.path.exists(runs_path))
        dice = metrics[metrics.metric == 'DICE'].groupby('label')['value'].mean()
        print('    mean Dice: ' + ', '.join(f'{label} {dice[label]:.3f}' for label in LABELS if label in dice)
              + f'  ({timing["fit_s"]} s fit, {timing["predict_s"]} s predict)')


def load_metrics(out_dir: str) -> pd.DataFrame:
    path = os.path.join(out_dir, 'metrics.csv')
    if not os.path.exists(path):
        raise SystemExit(f'no results in {out_dir} yet: run the experiment first')
    return pd.read_csv(path)


# =================================================================================================================
# Statistics: effect size with a bootstrap confidence interval (replaces "p = 0.002")
# =================================================================================================================
def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple:
    """Mean of the paired differences and its 95 % TWO-LEVEL bootstrap confidence interval.

    diff: table of paired differences, shape (n_subjects, n_seeds): run minus reference for the same subject and
          the same seed (same seed = same training voxels for both runs, so the pairing removes part of the noise).
          A 1-D array is treated as one seed (then this is the plain subject bootstrap).

    Why two levels: the result varies for two reasons, (1) which patients are in the test set and (2) the
    randomness of training (which voxels were sampled, the bootstrap rows and split features of every tree).
    Resampling only the subjects (the old version: seeds averaged first) shows (1) only, so the interval was too
    narrow. Here every resample draws n_subjects subjects AND n_seeds seeds with replacement and takes the mean of
    that sub-table, so both sources of variation are in the interval.

    Example: diff has 10 subjects x 3 seeds. One resample might take subjects (3, 3, 7, 0, ...) and seeds (1, 1, 2):
    the mean of those 10 x 3 cells is one bootstrap mean. Repeated 10,000 times, the middle 95 % of the means is the
    interval. Caveat: with only 3 seeds the seed level is rough (only 10 different seed draws exist); 5 are better.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.ndim == 1:
        diff = diff[:, None]
    n_subjects, n_seeds = diff.shape
    rng = np.random.default_rng(seed)                                    # own generator, reproducible
    subjects = rng.integers(0, n_subjects, size=(n_boot, n_subjects))   # resampled subject indices per resample
    seeds = rng.integers(0, n_seeds, size=(n_boot, n_seeds))            # resampled seed indices per resample
    # diff[subjects]: (n_boot, n_subjects, n_seeds); then keep the resampled seeds (columns) of every resample
    picked = np.take_along_axis(diff[subjects], np.repeat(seeds[:, None, :], n_subjects, axis=1), axis=2)
    means = picked.mean(axis=(1, 2))                                     # the mean difference of every resample
    low, high = np.percentile(means, [2.5, 97.5])                        # the middle 95 % of those means
    return float(diff.mean()), float(low), float(high)


def effects(metrics: pd.DataFrame, reference: str, min_dice: float, min_hd: float) -> pd.DataFrame:
    """The difference of every run to the reference run, per structure and metric, with its bootstrap CI.

    ci_low, ci_high: two-level bootstrap over subjects AND seeds (see bootstrap_ci).
    seed_diffs:      the mean difference of every seed on its own (mean over the subjects), e.g. "+0.016 +0.028 +0.019"
    seeds_agree:     all seeds give the same sign (a simple check to show next to the interval)
    relevant:        the whole interval lies beyond the minimal relevant difference, in either direction
                     (a real effect, larger than what matters)
    negligible:      the whole interval lies INSIDE +- the minimal relevant difference (equivalence: as good as the
                     reference; for an ablation: the family is redundant GIVEN the other features, not useless)
    better:          the mean difference points in the good direction (higher Dice, lower HD95)
    """
    # one value per run x structure x metric x subject x seed
    values = metrics.groupby(['name', 'label', 'metric', 'subject', 'seed'])['value'].mean()
    if reference not in values.index.get_level_values('name'):
        raise SystemExit(f'the reference run "{reference}" is not finished yet')
    base = values.xs(reference, level='name')
    rows = []
    # runs in the order they were planned (= first appearance in metrics.csv), not alphabetically
    for name in [n for n in dict.fromkeys(metrics.name) if n != reference]:
        # paired by subject AND seed: the same seed means the same training voxels for both runs
        diff = (values.xs(name, level='name') - base).dropna()
        for (label, metric), g in diff.groupby(level=['label', 'metric']):
            table = g.droplevel(['label', 'metric']).unstack('seed').dropna(axis=1)   # subjects x seeds
            mean, low, high = bootstrap_ci(table.values)
            per_seed = table.mean(axis=0)                                # mean over the subjects, one per seed
            threshold = min_hd if metric in LOWER_IS_BETTER else min_dice
            rows.append({'run': name, 'reference': reference, 'label': label, 'metric': metric,
                         'n_subjects': table.shape[0], 'n_seeds': table.shape[1],
                         'mean_diff': mean, 'ci_low': low, 'ci_high': high,
                         'seed_diffs': ' '.join(f'{v:+.4f}' for v in per_seed),
                         'seeds_agree': bool((per_seed > 0).all() or (per_seed < 0).all()),
                         'relevant': bool(low > threshold or high < -threshold),
                         'negligible': bool(low > -threshold and high < threshold),
                         'better': bool(mean < 0 if metric in LOWER_IS_BETTER else mean > 0)})
    return pd.DataFrame(rows)


def print_effects(table: pd.DataFrame, metric: str = 'DICE'):
    """One line per run: mean difference [CI] per structure.

    Marks: * = relevant (real effect beyond the threshold), = = negligible (whole CI inside +-threshold),
           ? = the seeds disagree in sign (the mean is not stable over training runs).
    """
    view = table[table.metric == metric].copy()
    mark = pd.Series(np.where(view.relevant, ' *', np.where(view.negligible, ' =', '')), index=view.index) + \
        pd.Series(np.where(view.seeds_agree, '', '?'), index=view.index)
    view['effect'] = (view.mean_diff.map('{:+.3f}'.format) + ' [' + view.ci_low.map('{:+.3f}'.format) + ', '
                      + view.ci_high.map('{:+.3f}'.format) + ']' + mark)
    pivot = view.pivot(index='run', columns='label', values='effect')[[c for c in LABELS if c in set(view.label)]]
    print(f'\n{METRICS[metric]} difference to "{table.reference.iloc[0]}", mean [95 % CI over subjects AND seeds], '
          f'* = relevant, = = negligible, ? = seeds disagree in sign')
    print(pivot.to_string())


# =================================================================================================================
# Plots
# =================================================================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def heatmap(table: pd.DataFrame, path: str, title: str, fmt: str = '{:.3f}', diverging: bool = False,
            cbar_label: str = ''):
    """A labelled heatmap of a table (rows x columns); diverging = centred on 0 (red = worse, blue = better)."""
    plt = _plt()
    values = table.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3.5, 0.42 * table.shape[0] + 1.6))
    if diverging:
        limit = np.nanmax(np.abs(values)) or 1
        image = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    else:
        image = ax.imshow(values, cmap='Blues', aspect='auto')
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.tick_params(length=0)
    for (r, c), v in np.ndenumerate(values):
        if not np.isnan(v):
            strong = abs(v) > 0.6 * (np.nanmax(np.abs(values)) or 1)
            ax.text(c, r, fmt.format(v), ha='center', va='center', fontsize=8, color='white' if strong else 'black')
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    ax.set_title(title, fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('plot:', path)


# readable row names for the effect plots (run name -> label on the slide); unknown names are shown as they are
RUN_TITLES = {
    'no_T2': 'remove all T2 features', 'no_combination': 'remove T1 - T2', 'no_distance': 'remove brain-border distance',
    'no_gradient': 'remove gradients', 'no_intensity': 'remove intensities', 'no_localstats': 'remove local statistics',
    'no_log': 'remove LoG filters', 'no_prior': 'remove label prior', 'no_priordistance': 'remove prior distances',
    'no_smoothing': 'remove Gaussian smoothing', 'no_spatial': 'remove coordinates', 'no_symmetry': 'remove symmetry',
    'extended+29noise': '+ 29 noise columns', 'extended+100noise': '+ 100 noise columns',
    'all35+35noise': '+ 35 noise columns', 'all35+100noise': '+ 100 noise columns',
    'all35': 'all 35 features', 'extended': 'the 29 extended features',
}
# verdict -> colour (the meaning of each colour is in the legend of the plot)
VERDICTS = {'worse': ('clearly worse', '#d0453a'), 'better': ('clearly better', '#2a78d6'),
            'small': ('real, about the threshold', '#e3a33b'),
            'negligible': ('negligible', '#4a5463'), 'unclear': ('unclear', '#b3bcc8')}


def verdict(row, metric: str) -> str:
    """The verdict of one effect, from its 95 % CI [low, high] and the threshold t (minimal relevant difference):

    worse / better: the whole CI is beyond t (a real effect that is large enough to matter)
    negligible:     the whole CI is inside [-t, t] (any effect is too small to matter)
    small:          the CI excludes 0 but crosses t (the effect is real, its size is around t)
                    e.g. +0.014 [+0.008, +0.020] with t = 0.01: surely positive, maybe just under 0.01
    unclear:        the CI includes 0 and crosses t (we can't tell)
    """
    if row.relevant:
        good = row.mean_diff < 0 if metric in LOWER_IS_BETTER else row.mean_diff > 0
        return 'better' if good else 'worse'
    if row.negligible:
        return 'negligible'
    return 'small' if row.ci_low > 0 or row.ci_high < 0 else 'unclear'


def effect_plot(table: pd.DataFrame, path: str, title: str, metric: str = 'DICE', threshold: float = 0.01,
                drop_negligible: bool = False):
    """One panel per structure: difference to the reference with its 95 % CI (subjects and seeds) and every seed.

    For SMALL effects (ablation, noise); large effects belong in an absolute plot (groups_plot).
    drop_negligible: leave out the runs that are negligible in EVERY structure (a note under the plot names them),
    so the slide shows only the runs that tell something.
    Reading: bar = 95 % CI, big dot = mean, small open dots = the single seeds, grey band = +- threshold
    (differences inside it are too small to matter). The colour is the verdict (see VERDICTS); all panels share the
    x axis, so the structures are comparable.
    """
    plt = _plt()
    from matplotlib.lines import Line2D
    view = table[table.metric == metric].copy()
    view['verdict'] = [verdict(r, metric) for r in view.itertuples()]
    runs = list(dict.fromkeys(view.run))
    dropped = []
    if drop_negligible:
        dropped = [r for r in runs if (view[view.run == r].verdict == 'negligible').all()]
        runs = [r for r in runs if r not in dropped]
    labels = [label for label in LABELS if label in set(view.label)]
    shown = view[view.run.isin(runs)]
    limit = 1.12 * max(np.abs(shown[['ci_low', 'ci_high']].values).max(), 1.5 * threshold)

    fig, axes = plt.subplots(1, len(labels), figsize=(2.7 * len(labels) + 2.4, 0.6 * len(runs) + 2.7),
                             sharey=True, sharex=True)
    for ax, label in zip(np.atleast_1d(axes), labels):
        g = view[view.label == label].set_index('run').reindex(runs)
        ax.axvspan(-threshold, threshold, color='#e9ecf1', zorder=0)    # differences too small to matter
        ax.axvline(0, color='#8b95a5', lw=0.9, zorder=1)
        for yi, row in enumerate(g.itertuples()):
            color = VERDICTS[row.verdict][1]
            ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=3, solid_capstyle='round', zorder=2)
            ax.scatter(row.mean_diff, yi, color=color, s=60, zorder=3, edgecolors='white', linewidths=1.2)
            seeds = [float(v) for v in row.seed_diffs.split()]
            ax.scatter(seeds, [yi + 0.28] * len(seeds), facecolors='none', edgecolors='#5d6878', s=18, lw=0.8,
                       zorder=4)                                       # the seeds just below the bar
        ax.set_xlim(-limit, limit)
        ax.set_title(label, fontsize=11)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=8.5)
        ax.grid(axis='x', color='#f0f2f5', lw=0.6, zorder=0)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(np.arange(len(runs)))
    first.set_yticklabels([RUN_TITLES.get(r, r) for r in runs], fontsize=10)
    first.set_ylim(len(runs) - 0.4, -0.6)                              # first run on top, room for the seed dots

    unit = 'Dice' if metric == 'DICE' else 'HD95 (mm)'
    reference = RUN_TITLES.get(view.reference.iloc[0], view.reference.iloc[0])
    height = fig.get_figheight()                                     # bottom rows placed in inches, not fractions
    fig.supxlabel(f'{unit} difference to {reference}', fontsize=10, y=0.6 / height, va='bottom')
    handles = [Line2D([], [], color=c, lw=3, marker='o', markersize=7, markeredgecolor='white', label=n)
               for n, c in VERDICTS.values()]
    handles += [Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor='#5d6878',
                       markersize=5, label='single seeds'),
                plt.Rectangle((0, 0), 1, 1, color='#e9ecf1', label=f'±{threshold:g}: too small to matter')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.26 / height))
    note = f'bar = 95 % confidence interval over test subjects and seeds'
    if dropped:
        note += ';  not shown (too small to matter in every structure): ' + \
                ', '.join(RUN_TITLES.get(r, r).replace('remove ', '') for r in dropped)
    fig.text(0.5, 0.08 / height, note, fontsize=8, color='#5d6878', ha='center')
    fig.suptitle(title, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.85 / height, 1, 1))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print('plot:', path)


def mean_table(metrics: pd.DataFrame, metric: str = 'DICE') -> pd.DataFrame:
    """Mean over subjects and seeds: rows = runs (in run order), columns = structures."""
    view = metrics[metrics.metric == metric]
    table = view.groupby(['name', 'label'])['value'].mean().unstack('label')
    return table.reindex(list(dict.fromkeys(view.name)))[[c for c in LABELS if c in table.columns]]


# =================================================================================================================
# The experiments: each has a plan (the list of forests) and a report
# =================================================================================================================
REFERENCE = 'all35'   # the 35 features of extended_plus


def plan_check(data: Data, args) -> list:
    """All 35 columns: must reproduce extended_plus. With several seeds, the spread is the noise floor."""
    return [{'name': REFERENCE, 'seed': s, 'columns': data.all_features} for s in args.seeds]


def report_check(out_dir: str, args):
    metrics = load_metrics(out_dir)
    dice = metrics[metrics.metric == 'DICE']
    mine = dice.groupby('label')['value'].mean()
    per_seed = dice.groupby(['seed', 'label'])['value'].mean().unstack('label')
    table = pd.DataFrame({'mean Dice': mine})
    if len(per_seed) > 1:
        # how much the mean over the subjects moves between seeds = the noise floor of these experiments
        table['noise floor (std over seeds)'] = per_seed.std()
    # the same seed must give the same training voxels and forest as run_experiment.py: compare seed by seed
    for seed in per_seed.index:
        reference = os.path.join(SCRIPT_DIR, 'results', 'extended_plus', f'seed_{seed}', 'metrics.csv')
        if os.path.exists(reference):
            ref = pd.read_csv(reference)
            ref = ref[(ref.metric == 'DICE') & (~ref.postprocessed)].groupby('label')['value'].mean()
            table[f'difference to run_experiment.py (seed {seed})'] = per_seed.loc[seed] - ref
    print('\nCheck: all 35 features (should be within ~0.005 of run_experiment.py)')
    print(table.reindex(LABELS).to_string(float_format=lambda v: f'{v:.4f}'))
    table.to_csv(os.path.join(out_dir, 'check.csv'))


def plan_subsets(data: Data, args) -> list:
    """Exp. 1: for every k and draw one random subset of the image features, without and with all location features.

    The two arms use the SAME subset, so they differ only by the location features.
    """
    image = data.groups['image']
    runs = []
    for k in args.k:
        if k > len(image):
            raise SystemExit(f'k = {k} is larger than the {len(image)} image features')
        for draw in range(1 if k == len(image) else args.draws):      # all image features: only one subset
            rng = np.random.default_rng(1000 + draw * 100 + k)        # own generator, reproducible
            subset = sorted(rng.choice(image, size=k, replace=False).tolist(), key=image.index)
            for arm in ('image', 'image+location'):
                columns = subset + (data.groups['location'] if arm == 'image+location' else [])
                for seed in args.seeds:
                    runs.append({'name': f'{arm}_k{k:02d}_d{draw}', 'seed': seed, 'arm': arm, 'k': k,
                                 'draw': draw, 'columns': columns})
    return runs


def report_subsets(out_dir: str, args):
    metrics = load_metrics(out_dir)
    dice = metrics[metrics.metric == 'DICE']
    per_run = dice.groupby(['arm', 'k', 'draw', 'label'])['value'].mean().reset_index()   # mean over subjects
    per_k = per_run.groupby(['arm', 'label', 'k'])['value'].agg(['mean', 'min', 'max', 'count']).reset_index()

    rows = []
    for (arm, label), g in per_k.groupby(['arm', 'label']):
        g = g.sort_values('k')
        upper = g[g.k >= g.k.median()]                                 # the upper half of the k range
        rows.append({'arm': arm, 'label': label, f'k={g.k.iloc[0]}': g['mean'].iloc[0],
                     f'k={g.k.iloc[-1]}': g['mean'].iloc[-1],
                     'rise': g['mean'].iloc[-1] - g['mean'].iloc[0],          # smallest to largest k
                     f'rise_from_k={upper.k.iloc[0]}': upper['mean'].iloc[-1] - upper['mean'].iloc[0],  # plateau?
                     'spread': (g['max'] - g['min'])[g['count'] > 1].mean()})  # typical spread between draws
    table = pd.DataFrame(rows)
    # decision rule of the plan: amount matters only if the rise beats the relevant difference AND the spread
    table['amount_matters'] = (table.rise > args.min_dice) & (table.rise > table.spread)
    print('\nMean Dice over the test subjects; spread = max - min over the draws at the same k')
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    table.to_csv(os.path.join(out_dir, 'decision_table.csv'), index=False)

    plt = _plt()
    colors = {'image': '#2a78d6', 'image+location': '#eb6834'}
    labels = [label for label in LABELS if label in set(per_k.label)]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.6), sharex=True)
    for ax, label in zip(np.atleast_1d(axes), labels):
        for arm, g in per_k[per_k.label == label].groupby('arm'):
            g = g.sort_values('k')
            ax.fill_between(g.k, g['min'], g['max'], color=colors[arm], alpha=0.18, lw=0)
            ax.plot(g.k, g['mean'], marker='o', ms=5, lw=2, color=colors[arm], label=arm)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('image features (k)')
        ax.grid(axis='y', color='#e3e6ea')
        ax.spines[['top', 'right']].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('Dice (mean over test subjects)')
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Random subsets of the image features: line = mean over draws, band = min..max', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dice_vs_k.png'), dpi=200)
    plt.close(fig)
    print('plot:', os.path.join(out_dir, 'dice_vs_k.png'))


def plan_noise(data: Data, args) -> list:
    """Exp. 2: all 35 features without and with m pure-noise columns: the count rises, the information does not.

    The reference is all35, the same as in every other experiment (until 2026-11-24 it was 'extended' with 29
    features; those results are kept in results/cheap/noise_extended29/). Default m = 35 (doubles the feature count)
    and 100 (almost 4x).
    """
    runs = [{'name': REFERENCE, 'seed': s, 'n_noise': 0, 'columns': data.all_features} for s in args.seeds]
    for m in args.n_noise:
        runs += [{'name': f'{REFERENCE}+{m}noise', 'seed': s, 'n_noise': m, 'columns': data.all_features}
                 for s in args.seeds]
    return runs


def report_effects(out_dir: str, args, reference: str, title: str, plot: bool = True,
                   drop_negligible: bool = False):
    """For noise, groups and ablation: the effect of every run against the reference, table + plot.

    plot=False: only the tables (for experiments whose effects are too large for a difference plot, see groups).
    """
    metrics = load_metrics(out_dir)
    print(f'\nMean Dice over the test subjects')
    print(mean_table(metrics).to_string(float_format=lambda v: f'{v:.3f}'))
    table = effects(metrics, reference, args.min_dice, args.min_hd)
    table.to_csv(os.path.join(out_dir, 'effects.csv'), index=False)
    print_effects(table, 'DICE')
    print_effects(table, 'HDRFDST95')
    if plot:
        effect_plot(table, os.path.join(out_dir, 'effects_dice.png'), title, 'DICE', args.min_dice,
                    drop_negligible)
        effect_plot(table, os.path.join(out_dir, 'effects_hd95.png'), title, 'HDRFDST95', args.min_hd,
                    drop_negligible)
    return metrics, table


def plan_groups(data: Data, args) -> list:
    """Exp. 3: which structures need location information, and which are separated by image information?"""
    t1 = [n for n in data.names if n == 'T1_intensity']
    sets = {'T1_intensity': t1, 'image_only': data.groups['image'], 'location_only': data.groups['location'],
            REFERENCE: data.all_features}
    return [{'name': name, 'seed': s, 'columns': columns} for name, columns in sets.items() for s in args.seeds]


# the feature sets of the groups experiment: run name -> (label on the plot, colour = kind of information)
# blue = image information only, orange = location information only, green = both. Colours of the validated
# default categorical palette (slots 1-3); T1 only is a lighter blue because it is a subset of "image".
GROUP_SETS = {'T1_intensity': ('T1 intensity only (1 feature)', '#9cc3ef'),
              'image_only': ('image only (23)', '#2a78d6'),
              'location_only': ('location only (12)', '#eb6834'),
              REFERENCE: ('image + location (35)', '#1baf7a')}


def groups_plot(metrics: pd.DataFrame, path: str, metric: str = 'DICE'):
    """The slide picture of the groups experiment: absolute Dice (or HD95) per structure and feature set.

    The effects here are large (0.05 to 0.8 Dice), so the absolute value is the honest picture; a difference plot
    squeezes the seeds and the relevance band to invisible dots. One panel per structure, one bar per feature set
    (bar = mean over subjects and seeds, value printed on top), small dots = the 10 test subjects (seed-averaged),
    so the spread between patients is visible. Same y axis in every panel: the structures are comparable.
    """
    plt = _plt()
    view = metrics[metrics.metric == metric]
    per_subject = view.groupby(['name', 'label', 'subject'])['value'].mean()   # seeds averaged per subject
    sets = [n for n in GROUP_SETS if n in set(view.name)]
    labels = [label for label in LABELS if label in set(view.label)]
    fig, axes = plt.subplots(1, len(labels), figsize=(2.7 * len(labels) + 0.6, 3.9), sharey=True)
    rng = np.random.default_rng(0)                                   # horizontal jitter of the subject dots
    for ax, label in zip(np.atleast_1d(axes), labels):
        for x, name in enumerate(sets):
            values = per_subject.loc[(name, label)].values
            mean = values.mean()
            ax.bar(x, mean, width=0.72, color=GROUP_SETS[name][1], zorder=2)
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=9, color='#2b2f36', alpha=0.55,
                       lw=0, zorder=3)
            text = f'{mean:.2f}' if metric == 'DICE' else f'{mean:.1f}'
            ax.text(x, max(mean, values.max()) + (0.02 if metric == 'DICE' else 0.4), text, ha='center',
                    va='bottom', fontsize=8.5, color='#2b2f36')
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([''] * len(sets))
        ax.grid(axis='y', color='#e6e8ec', lw=0.8, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    first = np.atleast_1d(axes)[0]
    if metric == 'DICE':
        first.set_ylim(0, 1.0)
    first.set_ylabel('Dice' if metric == 'DICE' else 'HD95 (mm)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_SETS[n][1]) for n in sets]
    fig.legend(handles, [GROUP_SETS[n][0] for n in sets], loc='lower center', ncol=len(sets),
               frameon=False, fontsize=9.5)
    fig.suptitle('Which kind of information does each structure need?  bar = mean, dots = the 10 test subjects',
                 fontsize=11.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print('plot:', path)


def report_groups(out_dir: str, args):
    # tables only: the differences are far too large for the effect plot (see groups_plot)
    metrics, _ = report_effects(out_dir, args, REFERENCE, 'Image vs location features', plot=False)
    for name in ('effects_dice.png', 'effects_hd95.png'):            # the old difference plots would mislead
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    groups_plot(metrics, os.path.join(out_dir, 'groups_dice.png'), 'DICE')
    groups_plot(metrics, os.path.join(out_dir, 'groups_hd95.png'), 'HDRFDST95')


def ablation_sets(data: Data) -> dict:
    """extended_plus without one family at a time, and without all T2 columns."""
    sets = {}
    for family in dict.fromkeys(data.family[n] for n in data.all_features):
        sets[f'no_{family}'] = [n for n in data.all_features if data.family[n] != family]
    sets['no_T2'] = [n for n in data.all_features if data.modality[n] != 'T2' and 'T2' not in n]
    return sets


def plan_ablation(data: Data, args) -> list:
    """Exp. 4: does a family carry information that nothing else in the pool has? Retrain without it."""
    runs = [{'name': REFERENCE, 'seed': s, 'columns': data.all_features} for s in args.seeds]
    for name, columns in ablation_sets(data).items():
        runs += [{'name': name, 'seed': s, 'columns': columns} for s in args.seeds]
    return runs


def report_ablation(out_dir: str, args):
    _, table = report_effects(out_dir, args, REFERENCE,
                              'Remove one feature family from the 35 features and retrain',
                              drop_negligible=True)
    dice = table[table.metric == 'DICE'].pivot(index='run', columns='label', values='mean_diff')
    heatmap(dice[[c for c in LABELS if c in dice.columns]], os.path.join(out_dir, 'ablation_dice.png'),
            'Dice change when a family is removed', fmt='{:+.3f}', diverging=True,
            cbar_label='Dice difference')


# =================================================================================================================
# Grouped permutation importance: how much does the trained 35-feature forest depend on a family?
# =================================================================================================================
def dice_per_label(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Dice per structure on a set of voxels (fast, numpy only)."""
    result = {}
    for label in LABELS:
        p, t = prediction == LABEL_IDS[label], truth == LABEL_IDS[label]
        denominator = p.sum() + t.sum()
        result[label] = 2 * np.logical_and(p, t).sum() / denominator if denominator else np.nan
    return result


def permutation(data: Data, args, out_dir: str):
    """Shuffles all columns of one group TOGETHER within each test subject and measures the Dice drop.

    The same row order for every column of a group keeps the relations inside the group, but breaks the link
    between the group and the labels. A random fraction of each subject's brain voxels (--fraction) keeps it fast;
    voxel Dice on a random sample estimates the full Dice well.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'permutation.csv')
    columns = data.all_features
    idx = [data.names.index(c) for c in columns]
    groups = {f'{family}': [i for i, c in enumerate(columns) if data.family[c] == family]
              for family in dict.fromkeys(data.family[c] for c in columns)}
    groups['T2 (all)'] = [i for i, c in enumerate(columns) if data.modality[c] == 'T2' or 'T2' in c]
    groups['image (all)'] = [i for i, c in enumerate(columns) if c in data.groups['image']]
    groups['location (all)'] = [i for i, c in enumerate(columns) if c in data.groups['location']]

    done = set()
    if os.path.exists(path):
        done = set(zip(*[pd.read_csv(path)[k] for k in ('group', 'repeat')]))
    todo = [(g, r) for g in ['none'] + list(groups) for r in range(1 if g == 'none' else args.repeats)
            if (g, r) not in done]
    if not todo:
        print('all permutations finished')
        return

    # permutation importance explains ONE forest: the one of the first seed (its voxels and random_state)
    seed = next(iter(data.trains))
    print(f'training the 35-feature forest (seed {seed})')
    forest = sk_ensemble.RandomForestClassifier(random_state=seed, **FOREST)
    forest.fit(data.train['X'][:, idx], data.train['y'])

    rng = np.random.default_rng(7)                       # the same voxel sample for every group and repeat
    sample = {s: np.sort(rng.choice(len(t['y']), int(args.fraction * len(t['y'])), replace=False))
              for s, t in data.tests.items()}
    for n, (group, repeat) in enumerate(todo, 1):
        t0 = timeit.default_timer()
        rows = []
        for i, (subject, test) in enumerate(data.tests.items()):
            x = test['X'][sample[subject]][:, idx]
            if group != 'none':
                # own generator per group, repeat and subject: reproducible and independent
                order = np.random.default_rng([list(groups).index(group), repeat, i]).permutation(len(x))
                cols = groups[group]
                x[:, cols] = x[order][:, cols]
            dice = dice_per_label(forest.predict(x), test['y'][sample[subject]])
            rows += [{'group': group, 'repeat': repeat, 'subject': subject, 'label': k, 'dice': v}
                     for k, v in dice.items()]
        pd.DataFrame(rows).to_csv(path, mode='a', index=False, header=not os.path.exists(path))
        print(f'[{n}/{len(todo)}] {group}, repeat {repeat} ({timeit.default_timer() - t0:.0f} s)')


def report_permutation(out_dir: str, args):
    path = os.path.join(out_dir, 'permutation.csv')
    if not os.path.exists(path):
        raise SystemExit('no permutation results yet')
    table = pd.read_csv(path)
    base = table[table.group == 'none'].groupby(['subject', 'label'])['dice'].mean()
    perm = table[table.group != 'none'].groupby(['group', 'subject', 'label'])['dice'].mean()   # repeats averaged
    drop = (base - perm).groupby(['group', 'label']).mean().unstack('label')    # mean over the subjects
    drop = drop[[c for c in LABELS if c in drop.columns]].sort_values('Amygdala' if 'Amygdala' in drop else drop.columns[0],
                                                                      ascending=False)
    print('\nDice drop when a group is shuffled (35-feature forest, mean over the test subjects)')
    print(drop.to_string(float_format=lambda v: f'{v:.3f}'))
    drop.to_csv(os.path.join(out_dir, 'permutation_drop.csv'))
    heatmap(-drop, os.path.join(out_dir, 'permutation_dice.png'),
            'Dice change when a group is shuffled (red = the forest depends on it)', fmt='{:+.3f}', diverging=True,
            cbar_label='Dice difference')

    # compared with the ablation (retrained without the family): used vs replaceable vs essential
    ablation_path = os.path.join(os.path.dirname(out_dir), 'ablation', 'effects.csv')
    if os.path.exists(ablation_path):
        ablation = pd.read_csv(ablation_path)
        ablation = ablation[ablation.metric == 'DICE'].assign(family=lambda d: d.run.str.replace('no_', '', regex=False))
        ablation = ablation.set_index(['family', 'label'])['mean_diff']
        rows = []
        for (family, label), value in drop.stack().items():
            if (family, label) not in ablation.index:
                continue
            retrained = -ablation[(family, label)]          # Dice drop after retraining without the family
            used, needed = value > args.min_dice, retrained > args.min_dice
            rows.append({'family': family, 'label': label, 'permutation_drop': value, 'ablation_drop': retrained,
                         'meaning': 'essential' if used and needed else 'used but replaceable' if used
                         else 'needed only when retrained' if needed else 'not needed'})
        compare = pd.DataFrame(rows)
        compare.to_csv(os.path.join(out_dir, 'permutation_vs_ablation.csv'), index=False)
        print('\nPermutation vs ablation (Dice drop; relevant above', args.min_dice, ')')
        print(compare.pivot(index='family', columns='label', values='meaning')[
            [c for c in LABELS if c in set(compare.label)]].to_string())


# =================================================================================================================
# Mutual information: how much does a feature tell about a structure ON ITS OWN (no model)?
# =================================================================================================================
def mutual_information(data: Data, args, out_dir: str):
    from sklearn.feature_selection import mutual_info_classif
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    y = data.train['y']
    # a balanced sample: the same number of voxels per label (small structures are rare)
    sample = np.concatenate([rng.choice(np.flatnonzero(y == label), min(args.n_per_label, int(np.sum(y == label))),
                                        replace=False) for label in np.unique(y)])
    columns = data.all_features
    x = data.train['X'][sample][:, [data.names.index(c) for c in columns]]
    table = {}
    for label in LABELS:
        target = (y[sample] == LABEL_IDS[label]).astype(int)     # 1 = this structure, 0 = everything else
        # k-nearest-neighbour MI estimate for continuous features; the result is in nats
        mi = mutual_info_classif(x, target, n_neighbors=3, random_state=0)
        table[label] = mi / np.log(2)                            # nats -> bits; at most H(1/6) = 0.65 bits here (1 of 6 balanced labels)
        print(f'MI computed for {label}')
    table = pd.DataFrame(table, index=columns)
    table.insert(0, 'group', ['image' if c in data.groups['image'] else 'location' for c in columns])
    table.insert(1, 'family', [data.family[c] for c in columns])
    table.to_csv(os.path.join(out_dir, 'mutual_information.csv'), index_label='feature')
    report_mi(out_dir, args)


def report_mi(out_dir: str, args):
    table = pd.read_csv(os.path.join(out_dir, 'mutual_information.csv'), index_col='feature')
    table = table.sort_values(['group', 'family'])
    values = table[[c for c in LABELS if c in table.columns]]
    print('\nMutual information (bits) between each feature and each structure (one-vs-rest)')
    print(table.to_string(float_format=lambda v: f'{v:.3f}'))
    heatmap(values.rename(index=lambda n: f'{n}  [{table.group[n][0].upper()}]'),
            os.path.join(out_dir, 'mutual_information.png'),
            'Mutual information with each structure (bits; [I] image, [L] location)', cbar_label='bits')


# =================================================================================================================
# Command line
# =================================================================================================================
EXPERIMENTS = {
    # name: (plan, report)
    'check': (plan_check, report_check),
    'subsets': (plan_subsets, report_subsets),
    'noise': (plan_noise, lambda o, a: report_effects(o, a, REFERENCE, 'Add pure-noise columns to all 35 features')),
    'groups': (plan_groups, report_groups),
    'ablation': (plan_ablation, report_ablation),
}
OTHER = {'permutation': (permutation, report_permutation), 'mi': (mutual_information, report_mi)}


def main(args):
    out_dir = os.path.join(args.out_dir, args.experiment)
    if args.experiment == 'report':
        name = args.what
        out_dir = os.path.join(args.out_dir, name)
        (EXPERIMENTS.get(name) or OTHER[name])[1](out_dir, args)
        return
    # permutation and mi analyse one forest / one sample: only the first seed is loaded for them
    seeds = args.seeds if args.experiment in EXPERIMENTS else args.seeds[:1]
    data = Data(args.features_dir, args.data_atlas_dir, seeds)
    if args.experiment in EXPERIMENTS:
        plan, report = EXPERIMENTS[args.experiment]
        run_all(data, plan(data, args), out_dir)
        report(out_dir, args)
    else:
        compute, report = OTHER[args.experiment]
        compute(data, args, out_dir)
        if args.experiment != 'mi':
            report(out_dir, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cheap experiments on the cached feature matrices',
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('experiment', choices=list(EXPERIMENTS) + list(OTHER) + ['report'])
    parser.add_argument('what', nargs='?', choices=list(EXPERIMENTS) + list(OTHER),
                        help='with "report": which experiment')
    parser.add_argument('--features_dir', default=os.path.join(SCRIPT_DIR, 'features_extended_plus'),
                        help='output of "python extract_features.py --features extended_plus"')
    parser.add_argument('--data_atlas_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'atlas'))
    parser.add_argument('--out_dir', default=os.path.join(SCRIPT_DIR, 'results', 'cheap'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='seeds: training voxels (train_seed<s>.npz) and forest random_state')
    parser.add_argument('--min_dice', type=float, default=0.01, help='minimal relevant Dice difference')
    parser.add_argument('--min_hd', type=float, default=1.0, help='minimal relevant HD95 difference (mm)')
    # subsets
    parser.add_argument('--k', type=int, nargs='+', default=[3, 6, 12, 18, 23], help='subsets: image features')
    parser.add_argument('--draws', type=int, default=5, help='subsets: random subsets per k')
    # noise
    parser.add_argument('--n_noise', type=int, nargs='+', default=[35, 100], help='noise: numbers of noise columns')
    # permutation
    parser.add_argument('--repeats', type=int, default=3, help='permutation: shuffles per group')
    parser.add_argument('--fraction', type=float, default=0.2, help='permutation: fraction of brain voxels used')
    # mutual information
    parser.add_argument('--n_per_label', type=int, default=2000, help='mi: training voxels per label')
    arguments = parser.parse_args()
    if arguments.experiment == 'report' and not arguments.what:
        parser.error('report needs the experiment, e.g. "report groups"')
    main(arguments)
