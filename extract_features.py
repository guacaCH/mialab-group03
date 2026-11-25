"""Computes the features of all subjects once and saves them as numpy files for the analysis notebook.

Run it once (and again after adding a feature):
    python extract_features.py
For the robustness experiment, the test subjects with Gaussian noise added to the pre-processed T1w and T2w images:
    python extract_features.py --noise 0.1 0.2 0.5
or blurred (sigma in mm) or with a bias field (+-10 / +-20 %), see _add_noise / _bias_field in pipeline_utilities.py:
    python extract_features.py --features extended_plus --noise 0.1 0.3 0.5 --blur 1 2 --bias 0.1 0.2
With a larger feature set of mialab/filtering/feature_pool.py, written to its own directory so the basic features
stay as they are (also works with --noise):
    python extract_features.py --features extended        # multi-scale, local statistics, T1-T2, label prior
    python extract_features.py --features extended_plus   # + brain-border distance, prior distance, symmetry
Both add the random noise control. --extended is short for --features extended.
For several seeds of the training voxels (the same sampling as run_experiment.py --seeds 0 1 2):
    python extract_features.py --features extended_plus --seeds 0 1 2
    python extract_features.py --features extended_plus --seeds 1 2 --train_only   # add seeds to an existing cache
The feature images are taken from (and stored in) the same feature cache as run_experiment.py, so every further
seed only costs the loading and the voxel sampling, not the feature computation.

Writes to features/ (features_<set>/ for the larger sets, not committed to git):
    train.npz             X (sampled training voxels x features), y (labels), subject, feature_names (seed 0)
    train_seed<s>.npz     the same for the other seeds (--seeds), see train_path()
    test_<subject>.npz    X, y of all voxels inside the brain mask, brain_index (their position in the image)
    noise_<std>/test_<subject>.npz   the same with noise (with --noise)
"""
import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk

try:
    import mialab.data.structure as structure
    import mialab.filtering.feature_pool as fpool
    import mialab.filtering.preprocessing as fltr_prep
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.filtering.feature_pool as fpool
    import mialab.filtering.preprocessing as fltr_prep
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

PRE_PROCESS_PARAMS = {'registration_pre': True,
                      'normalization_pre': True,
                      'skullstrip_pre': True}

# the feature sets (see feature_pool.FEATURE_SETS) and their output directories; the notebook chooses the columns.
# The larger sets add the random control. Their features don't use np.random, so the same training voxels are sampled
FEATURE_SETS = {'baseline_t2': (['baseline_t2'], 'features'),
                'extended': (['extended', 'control_random'], 'features_extended'),
                'extended_plus': (['extended_plus', 'control_random'], 'features_extended_plus')}

DEFAULT_SEEDS = [0]  # the seed fixes the random choice of training voxels (np.random.seed, as in run_experiment.py)


def train_path(output_dir: str, seed: int) -> str:
    """The training file of one seed. Seed 0 keeps the old name train.npz, so older scripts still find it."""
    return os.path.join(output_dir, 'train.npz' if seed == 0 else f'train_seed{seed}.npz')


def crawl(data_dir: str) -> dict:
    return futil.FileSystemDataCrawler(data_dir, LOADING_KEYS, futil.BrainImageFilePathGenerator(),
                                       futil.DataDirectoryFilter()).data


def registered_ground_truths(data: dict) -> dict:
    """Registers the ground truths to the atlas, for the label prior feature."""
    ground_truths = {}
    for id_, paths in data.items():
        transform = sitk.ReadTransform(paths[structure.BrainImageTypes.RegistrationTransform])
        ground_truths[id_] = fltr_prep.ImageRegistration().execute(
            sitk.ReadImage(paths[structure.BrainImageTypes.GroundTruth]),
            fltr_prep.ImageRegistrationParameters(putil.atlas_t1, transform, True))
    return ground_truths


def save_test_subject(img: structure.BrainImage, path: str):
    """Saves the features of the voxels inside the brain mask (outside, everything is background anyway)."""
    brain = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.BrainMask]).ravel() > 0
    x, y = img.feature_matrix
    y = y.ravel()
    if np.any(y[~brain] > 0):
        print(f'warning: subject {img.id_} has {np.sum(y[~brain] > 0)} labelled voxels outside the brain mask')
    np.savez(path, X=x[brain], y=y[brain], brain_index=np.flatnonzero(brain), feature_names=img.feature_names)


def main(output_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, noise: list, features: str,
         seeds: list = DEFAULT_SEEDS, train_only: bool = False, feature_cache_dir: str = None,
         blur: list = (), bias: list = ()):
    putil.load_atlas_images(data_atlas_dir)
    os.makedirs(output_dir, exist_ok=True)

    # feature_cache_dir: the cached feature images of run_experiment.py (None = compute everything). With the cache,
    # a further seed only loads the feature images again and samples other voxels.
    params = dict(PRE_PROCESS_PARAMS, features=FEATURE_SETS[features][0], feature_cache_dir=feature_cache_dir)
    if any(column['family'] in fpool.PRIOR_FAMILIES for column in fpool.select_columns(params['features'])):
        # the label prior is built from the training subjects only, so no test label is used
        fpool.LABEL_PRIOR.clear()
        fpool.LABEL_PRIOR.update(fpool.build_label_prior(registered_ground_truths(crawl(data_train_dir))))

    # perturbations of the TEST images only (H3): (output folder, pre-processing parameter, value)
    perturbations = [(f'noise_{v}', 'noise_std', v) for v in noise] + \
        [(f'blur_{v}', 'blur_sigma', v) for v in blur] + [(f'bias_{v}', 'bias_level', v) for v in bias]

    if not perturbations:
        # training subjects: only the randomly sampled training voxels, once per seed.
        # np.random.seed(seed) right before pre_process_batch, exactly like run_experiment.py, so seed s gives the
        # same training voxels as results/<experiment>/seed_<s>/ (no feature uses np.random, the control has its own)
        for seed in seeds:
            print(f'training voxels of seed {seed}')
            np.random.seed(seed)
            images = putil.pre_process_batch(crawl(data_train_dir), dict(params, training=True),
                                             multi_process=False)
            np.savez(train_path(output_dir, seed),
                     X=np.concatenate([img.feature_matrix[0] for img in images]),
                     y=np.concatenate([img.feature_matrix[1] for img in images]).ravel(),
                     subject=np.concatenate([[img.id_] * len(img.feature_matrix[1]) for img in images]),
                     feature_names=images[0].feature_names)
            del images

    if train_only:
        # the test features don't depend on the seed: adding seeds needs no new test files
        print('features written to', output_dir)
        return

    # test subjects: all voxels inside the brain mask, one file per subject to keep the memory low
    # (no perturbation: the clean test files in output_dir itself)
    for folder, key, value in perturbations or [('', None, 0)]:
        out = os.path.join(output_dir, folder) if folder else output_dir
        os.makedirs(out, exist_ok=True)
        extra = {key: value} if key else {}
        for id_, paths in crawl(data_test_dir).items():
            img = putil.pre_process(id_, paths, **dict(params, training=False, **extra))
            save_test_subject(img, os.path.join(out, f'test_{id_}.npz'))

    print('features written to', output_dir)


if __name__ == '__main__':
    script_dir = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description='Feature extraction for the analysis notebook')
    parser.add_argument('--output_dir', default=None, help='default: features/, features_<set>/ for the larger sets')
    parser.add_argument('--data_atlas_dir', default=os.path.normpath(os.path.join(script_dir, '../data/atlas')))
    parser.add_argument('--data_train_dir', default=os.path.normpath(os.path.join(script_dir, '../data/train/')))
    parser.add_argument('--data_test_dir', default=os.path.normpath(os.path.join(script_dir, '../data/test/')))
    parser.add_argument('--noise', type=float, nargs='*', default=[],
                        help='noise standard deviations; only the test subjects are written')
    parser.add_argument('--blur', type=float, nargs='*', default=[],
                        help='Gaussian blur sigmas in mm (test subjects only), e.g. 1 2')
    parser.add_argument('--bias', type=float, nargs='*', default=[],
                        help='bias field strengths (test subjects only), e.g. 0.1 0.2 = +-10 %% / +-20 %%')
    parser.add_argument('--features', choices=FEATURE_SETS, default='baseline_t2', help='the feature set')
    parser.add_argument('--extended', action='store_true', help='short for --features extended')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='seeds of the training voxel sampling, e.g. 0 1 2 (one training file per seed)')
    parser.add_argument('--train_only', action='store_true',
                        help='only the training files, e.g. to add seeds to an existing cache')
    parser.add_argument('--feature_cache', default=os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                                                                 'feature_cache'),
                        help='feature cache directory (shared with run_experiment.py), "none" to disable it')
    args = parser.parse_args()
    features = 'extended' if args.extended else args.features
    output_dir = args.output_dir or os.path.normpath(os.path.join(script_dir, FEATURE_SETS[features][1]))
    main(output_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.noise, features,
         args.seeds, args.train_only, None if args.feature_cache.lower() == 'none' else args.feature_cache,
         args.blur, args.bias)
