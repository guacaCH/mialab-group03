"""Runs one experiment (one config file, one or more seeds) and stores its results in a fixed folder structure.

Usage:
    python run_experiment.py experiments/baseline.json --seeds 0 1 2

Every run writes to results/<experiment>/seed_<seed>/ (see notes in experiment_runner.md):
    config.json              everything needed to reproduce the run
    metrics.csv              one row per subject x label x metric (tidy format)
    summary.csv              mean / std / median / min / max over the test subjects
    features.csv             name, family, modality and scale of every feature column
    feature_importance.csv   random forest (Gini) importance per feature
    timing.json              run times of the pipeline steps
    segmentations/           only with "save_segmentations": true (.mha) or "slicer" (.seg.nrrd, not committed to git)
With "save_segmentations": "slicer", the ground truth of every test subject is also written once per experiment
to results/<experiment>/ground_truth/<subject>_GT.seg.nrrd.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import timeit

import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn
import sklearn.ensemble as sk_ensemble
import pymia.data.conversion as conversion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mialab.data.structure as structure  # noqa: E402
import mialab.filtering.feature_pool as fpool  # noqa: E402
import mialab.utilities.file_access_utilities as futil  # noqa: E402
import mialab.utilities.pipeline_utilities as putil  # noqa: E402

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]

LABEL_NAMES = {0: 'Background', 1: 'WhiteMatter', 2: 'GreyMatter', 3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}

# fixed colour (r, g, b) per label, so a structure looks the same in every subject and experiment in 3D Slicer
LABEL_COLORS = {1: (0.95, 0.95, 0.85), 2: (0.55, 0.55, 0.75), 3: (0.86, 0.71, 0.27),
                4: (0.90, 0.33, 0.30), 5: (0.33, 0.68, 0.45)}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    'description': '',
    # passed unchanged to putil.pre_process()
    'pre_process': {'skullstrip_pre': True,
                    'normalization_pre': True,
                    'registration_pre': True,
                    # feature specification, see mialab/filtering/feature_pool.py
                    'features': 'baseline',
                    'exclude_features': []},
    # passed unchanged to sklearn's RandomForestClassifier (random_state is set from the seed)
    'forest': {'n_estimators': 50,
               'max_depth': 20,
               'max_features': 'sqrt',
               'class_weight': None,
               'n_jobs': -1},
    # passed to putil.post_process(); empty dict = no post-processing
    'post_process': {},
    'save_segmentations': False,
}


# --- helpers -------------------------------------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        user = json.load(f)
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if any(flag in user.get('pre_process', {}) for flag in fpool.LEGACY_FLAGS):
        del config['pre_process']['features']  # an old-style config: its boolean flags select the features
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run('status', '--porcelain', '--untracked-files=no')
    return {'commit': run('rev-parse', 'HEAD'),
            'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'uncommitted_changes': bool(status) if status is not None else None}


def crawl(data_dir: str) -> dict:
    return futil.FileSystemDataCrawler(data_dir, LOADING_KEYS, futil.BrainImageFilePathGenerator(),
                                       futil.DataDirectoryFilter()).data


def write_slicer_segmentation(label_image: sitk.Image, path: str):
    """Writes a label image as a 3D Slicer segmentation (.seg.nrrd): one named, coloured segment per label.

    The file is a normal NRRD image; 3D Slicer reads the segments from the extra header fields
    (see the .seg.nrrd format in the 3D Slicer documentation). Labels that don't occur get an empty segment.
    """
    image = sitk.Cast(label_image, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(image)  # index order z, y, x
    image.SetMetaData('Segmentation_MasterRepresentation', 'Binary labelmap')
    image.SetMetaData('Segmentation_ContainedRepresentationNames', 'Binary labelmap|')
    image.SetMetaData('Segmentation_ReferenceImageExtentOffset', '0 0 0')
    for i, label in enumerate(sorted(LABEL_COLORS)):
        z, y, x = np.nonzero(arr == label)
        extent = f'{x.min()} {x.max()} {y.min()} {y.max()} {z.min()} {z.max()}' if len(x) else '0 -1 0 -1 0 -1'
        fields = {'ID': f'Segment_{label}', 'Name': LABEL_NAMES[label], 'NameAutoGenerated': '0',
                  'Color': ' '.join(str(c) for c in LABEL_COLORS[label]), 'ColorAutoGenerated': '0',
                  'LabelValue': str(label), 'Layer': '0', 'Extent': extent, 'Tags': '|'}
        for key, value in fields.items():
            image.SetMetaData(f'Segment{i}_{key}', value)
    sitk.WriteImage(image, path, True)


def evaluator_rows(evaluator, experiment: str, seed: int, postprocessed: bool) -> list:
    return [{'experiment': experiment, 'seed': seed, 'subject': r.id_, 'label': r.label, 'metric': r.metric,
             'value': r.value, 'postprocessed': postprocessed} for r in evaluator.results]


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (metrics.groupby(['postprocessed', 'label', 'metric'])['value']
            .agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index())


# --- main ----------------------------------------------------------------------------------------------------------
def main(config_path: str, seeds: list, results_root: str, data_atlas_dir: str, data_train_dir: str,
         data_test_dir: str, overwrite: bool, feature_cache_dir: str):
    experiment = os.path.splitext(os.path.basename(config_path))[0]
    config = load_config(config_path)
    # fail before the slow part if the feature specification is invalid
    columns = fpool.select_columns(*fpool.feature_spec_from_kwargs(config['pre_process']))
    print(f'{len(columns)} features:', ', '.join(c['name'] for c in columns))
    # the cache location depends on the machine, so it is not part of the config
    pre_process_params = dict(config['pre_process'], feature_cache_dir=feature_cache_dir)

    run_dirs = {s: os.path.join(results_root, experiment, f'seed_{s}') for s in seeds}
    todo = [s for s in seeds if overwrite or not os.path.exists(os.path.join(run_dirs[s], 'metrics.csv'))]
    for s in seeds:
        if s not in todo:
            print(f'skipping seed {s}: {run_dirs[s]} already exists (use --overwrite to redo)')
    if not todo:
        return

    putil.load_atlas_images(data_atlas_dir)
    train_data = crawl(data_train_dir)
    test_data = crawl(data_test_dir)

    # 1) training: one forest per seed. The seed fixes the random voxel sampling and the forest.
    forests, info = {}, {}
    for seed in todo:
        print('-' * 5, f'Experiment {experiment}, seed {seed}: training')
        np.random.seed(seed)
        t0 = timeit.default_timer()
        # pre_process consumes the path dicts, so give it copies
        images = putil.pre_process_batch({k: dict(v) for k, v in train_data.items()},
                                         dict(pre_process_params, training=True), multi_process=False)
        t_pre = timeit.default_timer() - t0

        x_train = np.concatenate([img.feature_matrix[0] for img in images])
        y_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
        feature_names = images[0].feature_names
        assert all(img.feature_names == feature_names for img in images), 'subjects have different features'
        del images

        forest = sk_ensemble.RandomForestClassifier(random_state=seed, **config['forest'])
        t0 = timeit.default_timer()
        forest.fit(x_train, y_train)
        t_fit = timeit.default_timer() - t0

        labels, counts = np.unique(y_train, return_counts=True)
        forests[seed] = forest
        info[seed] = {'feature_names': feature_names,
                      'n_features': x_train.shape[1],
                      'n_train_subjects': len(train_data),
                      'n_train_voxels': int(x_train.shape[0]),
                      'train_voxels_per_label': {LABEL_NAMES.get(int(l), str(l)): int(c)
                                                 for l, c in zip(labels, counts)},
                      'timing': {'preprocess_train_s': t_pre, 'fit_s': t_fit, 'predict_s': 0.0,
                                 'preprocess_test_s': 0.0}}
        del x_train, y_train

    # 2) testing: test features don't depend on the seed, so each test subject is processed once
    #    and predicted by every forest. One subject at a time keeps memory low.
    evaluators = {s: putil.init_evaluator() for s in todo}
    evaluators_pp = {s: putil.init_evaluator() for s in todo}
    tmp_dirs = {s: run_dirs[s] + '.tmp' for s in todo}
    for s in todo:
        shutil.rmtree(tmp_dirs[s], ignore_errors=True)
        os.makedirs(tmp_dirs[s])

    for subject, paths in test_data.items():
        print('-' * 5, f'Experiment {experiment}: testing {subject}')
        t0 = timeit.default_timer()
        img = putil.pre_process(subject, dict(paths), **dict(pre_process_params, training=False))
        t_pre = timeit.default_timer() - t0
        ground_truth = img.images[structure.BrainImageTypes.GroundTruth]
        if config['save_segmentations'] == 'slicer':
            gt_dir = os.path.join(results_root, experiment, 'ground_truth')
            os.makedirs(gt_dir, exist_ok=True)
            write_slicer_segmentation(ground_truth, os.path.join(gt_dir, f'{subject}_GT.seg.nrrd'))

        for seed in todo:
            info[seed]['timing']['preprocess_test_s'] += t_pre
            t0 = timeit.default_timer()
            probabilities = forests[seed].predict_proba(img.feature_matrix[0])
            predictions = forests[seed].classes_[np.argmax(probabilities, axis=1)]
            info[seed]['timing']['predict_s'] += timeit.default_timer() - t0

            seg = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8), img.image_properties)
            prob = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)
            evaluators[seed].evaluate(seg, ground_truth, subject)

            seg_pp = None
            if config['post_process']:
                seg_pp = putil.post_process(img, seg, prob, **config['post_process'])
                evaluators_pp[seed].evaluate(seg_pp, ground_truth, subject)

            if config['save_segmentations']:
                seg_dir = os.path.join(tmp_dirs[seed], 'segmentations')
                os.makedirs(seg_dir, exist_ok=True)
                outputs = {'SEG': seg, 'SEG-PP': seg_pp}
                for name, image in outputs.items():
                    if image is None:
                        continue
                    if config['save_segmentations'] == 'slicer':
                        write_slicer_segmentation(image, os.path.join(seg_dir, f'{subject}_{name}.seg.nrrd'))
                    else:
                        sitk.WriteImage(image, os.path.join(seg_dir, f'{subject}_{name}.mha'), True)
        del img

    # 3) write the results of each seed
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                'SimpleITK': sitk.Version_VersionString()}
    for seed in todo:
        out = tmp_dirs[seed]
        rows = evaluator_rows(evaluators[seed], experiment, seed, False)
        rows += evaluator_rows(evaluators_pp[seed], experiment, seed, True)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(os.path.join(out, 'metrics.csv'), index=False)
        summarize(metrics).to_csv(os.path.join(out, 'summary.csv'), index=False)

        pd.DataFrame({'feature': info[seed]['feature_names'],
                      'importance': forests[seed].feature_importances_}) \
            .sort_values('importance', ascending=False) \
            .to_csv(os.path.join(out, 'feature_importance.csv'), index=False)

        # name and tags (family, modality, scale) of every feature column, in matrix order
        tags = {c['name']: c for c in fpool.all_columns()}
        pd.DataFrame([{k: tags[n][k] for k in ('name', 'feature', 'family', 'modality', 'scale')}
                      for n in info[seed]['feature_names']]) \
            .to_csv(os.path.join(out, 'features.csv'), index=False)

        with open(os.path.join(out, 'timing.json'), 'w') as f:
            json.dump({k: round(v, 2) for k, v in info[seed]['timing'].items()}, f, indent=2)

        run_info = {k: v for k, v in info[seed].items() if k != 'timing'}
        with open(os.path.join(out, 'config.json'), 'w') as f:
            json.dump({'experiment': experiment,
                       'seed': seed,
                       'config_file': os.path.relpath(config_path, SCRIPT_DIR),
                       'config': config,
                       **run_info,
                       'n_test_subjects': len(test_data),
                       'test_subjects': sorted(test_data.keys()),
                       'git': git_info(),
                       'versions': versions,
                       'machine': platform.node(),
                       'finished_at': datetime.datetime.now().isoformat(timespec='seconds')}, f, indent=2)

        # only a complete run gets the final folder name, so crashed runs never look finished
        shutil.rmtree(run_dirs[seed], ignore_errors=True)
        os.replace(out, run_dirs[seed])
        print(f'written {run_dirs[seed]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run one experiment config for one or more seeds.')
    parser.add_argument('config', help='experiment config, e.g. experiments/baseline.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--results_dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--data_atlas_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'atlas'))
    parser.add_argument('--data_train_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'train'))
    parser.add_argument('--data_test_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'test'))
    parser.add_argument('--overwrite', action='store_true', help='redo seeds that already have results')
    parser.add_argument('--feature_cache', default=os.path.join(SCRIPT_DIR, 'feature_cache'),
                        help='directory of the feature cache, "none" to disable it')
    args = parser.parse_args()
    main(args.config, args.seeds, args.results_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir,
         args.overwrite, None if args.feature_cache.lower() == 'none' else args.feature_cache)
