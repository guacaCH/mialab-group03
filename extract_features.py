"""Computes the features of all subjects once and saves them as numpy files for the analysis notebook.

Run it once (and again after adding a feature):
    python extract_features.py
For the robustness experiment, the test subjects with Gaussian noise added to the pre-processed T1w and T2w images:
    python extract_features.py --noise 0.1 0.2 0.5

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
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
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

SEED = 0  # fixes the random choice of training voxels


def crawl(data_dir: str) -> dict:
    return futil.FileSystemDataCrawler(data_dir, LOADING_KEYS, futil.BrainImageFilePathGenerator(),
                                       futil.DataDirectoryFilter()).data


def save_test_subject(img: structure.BrainImage, path: str):
    """Saves the features of the voxels inside the brain mask (outside, everything is background anyway)."""
    brain = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.BrainMask]).ravel() > 0
    x, y = img.feature_matrix
    y = y.ravel()
    if np.any(y[~brain] > 0):
        print(f'warning: subject {img.id_} has {np.sum(y[~brain] > 0)} labelled voxels outside the brain mask')
    np.savez(path, X=x[brain], y=y[brain], brain_index=np.flatnonzero(brain), feature_names=img.feature_names)


def main(output_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, noise: list):
    putil.load_atlas_images(data_atlas_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not noise:
        # training subjects: only the randomly sampled training voxels
        np.random.seed(SEED)
        images = putil.pre_process_batch(crawl(data_train_dir), dict(PRE_PROCESS_PARAMS, training=True),
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
            img = putil.pre_process(id_, paths, **dict(PRE_PROCESS_PARAMS, training=False, noise_std=noise_std))
            save_test_subject(img, os.path.join(out, f'test_{id_}.npz'))

    print('features written to', output_dir)


if __name__ == '__main__':
    script_dir = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description='Feature extraction for the analysis notebook')
    parser.add_argument('--output_dir', default=os.path.normpath(os.path.join(script_dir, './features')))
    parser.add_argument('--data_atlas_dir', default=os.path.normpath(os.path.join(script_dir, '../data/atlas')))
    parser.add_argument('--data_train_dir', default=os.path.normpath(os.path.join(script_dir, '../data/train/')))
    parser.add_argument('--data_test_dir', default=os.path.normpath(os.path.join(script_dir, '../data/test/')))
    parser.add_argument('--noise', type=float, nargs='*', default=[],
                        help='noise standard deviations; only the test subjects are written')
    args = parser.parse_args()
    main(args.output_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.noise)
