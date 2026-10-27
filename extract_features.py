"""Computes the features of all subjects once and saves them as numpy files for the analysis notebook.

Run it once (and again after adding a feature):
    python extract_features.py
For the robustness experiment, the test subjects with Gaussian noise added to the pre-processed T1w and T2w images:
    python extract_features.py --noise 0.1 0.2 0.5
With the extended features (multi-scale filters, local statistics, T1-T2, label prior, a random control), written to
features_extended/ so the basic features stay as they are (also works with --noise):
    python extract_features.py --extended

Writes to features/ (not committed to git):
    train.npz             X (sampled training voxels x features), y (labels), subject, feature_names
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
    import mialab.filtering.preprocessing as fltr_prep
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.filtering.preprocessing as fltr_prep
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

# all features are computed; the notebook chooses which columns to use
PRE_PROCESS_PARAMS = {'registration_pre': True,
                      'normalization_pre': True,
                      'skullstrip_pre': True,
                      'coordinates_feature': True,
                      'intensity_feature': True,
                      'gradient_intensity_feature': True,
                      't2_intensity_feature': True,
                      't2_gradient_intensity_feature': True}

# the extended features. They don't use np.random, so the same training voxels are sampled as without them
EXTENDED_PARAMS = {'multiscale_feature': True,
                   'local_statistics_feature': True,
                   't1_t2_difference_feature': True,
                   'random_feature': True}

SEED = 0  # fixes the random choice of training voxels


def crawl(data_dir: str) -> dict:
    return futil.FileSystemDataCrawler(data_dir, LOADING_KEYS, futil.BrainImageFilePathGenerator(),
                                       futil.DataDirectoryFilter()).data


def label_counts(data: dict) -> np.ndarray:
    """Counts per atlas voxel how many subjects have each label 1-5, for the label prior feature."""
    counts = None
    for paths in data.values():
        transform = sitk.ReadTransform(paths[structure.BrainImageTypes.RegistrationTransform])
        ground_truth = fltr_prep.ImageRegistration().execute(
            sitk.ReadImage(paths[structure.BrainImageTypes.GroundTruth]),
            fltr_prep.ImageRegistrationParameters(putil.atlas_t1, transform, True))
        ground_truth = sitk.GetArrayFromImage(ground_truth)
        one_hot = np.stack([ground_truth == label for label in range(1, 6)]).astype(np.uint8)
        counts = one_hot if counts is None else counts + one_hot
    return counts


def save_test_subject(img: structure.BrainImage, path: str):
    """Saves the features of the voxels inside the brain mask (outside, everything is background anyway)."""
    brain = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.BrainMask]).ravel() > 0
    x, y = img.feature_matrix
    y = y.ravel()
    if np.any(y[~brain] > 0):
        print(f'warning: subject {img.id_} has {np.sum(y[~brain] > 0)} labelled voxels outside the brain mask')
    np.savez(path, X=x[brain], y=y[brain], brain_index=np.flatnonzero(brain), feature_names=img.feature_names)


def main(output_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, noise: list, extended: bool):
    putil.load_atlas_images(data_atlas_dir)
    os.makedirs(output_dir, exist_ok=True)

    params = dict(PRE_PROCESS_PARAMS)
    if extended:
        # the label prior is built from the training subjects only, so no test label is used
        train_data = crawl(data_train_dir)
        params.update(EXTENDED_PARAMS, label_counts=label_counts(train_data), label_count_subjects=len(train_data))

    if not noise:
        # training subjects: only the randomly sampled training voxels
        np.random.seed(SEED)
        images = putil.pre_process_batch(crawl(data_train_dir), dict(params, training=True),
                                         multi_process=False)
        np.savez(os.path.join(output_dir, 'train.npz'),
                 X=np.concatenate([img.feature_matrix[0] for img in images]),
                 y=np.concatenate([img.feature_matrix[1] for img in images]).ravel(),
                 subject=np.concatenate([[img.id_] * len(img.feature_matrix[1]) for img in images]),
                 feature_names=images[0].feature_names)
        del images

    # test subjects: all voxels inside the brain mask, one file per subject to keep the memory low
    for noise_std in noise or [0]:
        out = output_dir if noise_std == 0 else os.path.join(output_dir, f'noise_{noise_std}')
        os.makedirs(out, exist_ok=True)
        for id_, paths in crawl(data_test_dir).items():
            img = putil.pre_process(id_, paths, **dict(params, training=False, noise_std=noise_std))
            save_test_subject(img, os.path.join(out, f'test_{id_}.npz'))

    print('features written to', output_dir)


if __name__ == '__main__':
    script_dir = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description='Feature extraction for the analysis notebook')
    parser.add_argument('--output_dir', default=None, help='default: features/ (features_extended/ with --extended)')
    parser.add_argument('--data_atlas_dir', default=os.path.normpath(os.path.join(script_dir, '../data/atlas')))
    parser.add_argument('--data_train_dir', default=os.path.normpath(os.path.join(script_dir, '../data/train/')))
    parser.add_argument('--data_test_dir', default=os.path.normpath(os.path.join(script_dir, '../data/test/')))
    parser.add_argument('--noise', type=float, nargs='*', default=[],
                        help='noise standard deviations; only the test subjects are written')
    parser.add_argument('--extended', action='store_true', help='also compute the extended features')
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.normpath(
        os.path.join(script_dir, './features_extended' if args.extended else './features'))
    main(output_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.noise, args.extended)
