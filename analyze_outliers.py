"""Where do the HD95 differences come from? Stray false-positive voxels far from the structure, with and without a feature.

Usage:
    python analyze_outliers.py                                             # extended_plus_no_distance vs extended_plus
    python analyze_outliers.py --without extended --with extended_plus --seed 0

The two runs must exist (run_experiment.py). Their forests are trained again from the saved training voxels
(analyze_features.py, train.npz) with the same settings and seed, and the test subjects are predicted again, so the
Dice and HD95 values reproduce the run's metrics.csv (checked and printed). Per test subject and structure it measures:
    hd95_pred_to_gt   95th percentile of the distance from the predicted surface to the true surface: large when the
                      prediction has voxels far away from the structure (false positives)
    hd95_gt_to_pred   the other direction: large when parts of the true structure are missed
                      (the reported HD95 is the maximum of the two)
    fp, fp_far5/10    false-positive voxels, and those more than 5 / 10 mm away from the true structure
    stray_voxels      predicted voxels outside the two largest connected components (left and right structure)
    dice_clean, hd95_clean   Dice and HD95 after removing those stray components (keep the 2 largest)
and the value of the location features (prior and prior distance) at the far false positives.

Writes to results/_report/outliers_<without>_vs_<with>/:
    per_subject.csv, summary.csv, plots (hd95_direction, fp_distance, cleanup, feature_at_fp, example_<subject>)
and results/<with>/analysis/seed_<seed>/test_sample.npz (a balanced sample of test voxels with all features of the
larger set, used by analyze_information_gain.py --permutation).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from scipy import ndimage  # noqa: E402
import pymia.evaluation.metric as pymia_metric  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_features as af  # noqa: E402
import mialab.data.structure as structure  # noqa: E402
import mialab.filtering.feature_pool as fpool  # noqa: E402
import mialab.utilities.pipeline_utilities as putil  # noqa: E402
import run_experiment as rexp  # noqa: E402

SCRIPT_DIR = rexp.SCRIPT_DIR
LABELS = {3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}
LOCATION_FEATURES = {3: ('prior_hippocampus', 'priordistance_hippocampus'),
                     4: ('prior_amygdala', 'priordistance_amygdala'),
                     5: ('prior_thalamus', 'priordistance_thalamus')}
COLORS = {'without': '#c0504d', 'with': '#4f81bd'}
STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


# --- metrics --------------------------------------------------------------------------------------------------------
def directed_hd(pred: np.ndarray, ref: np.ndarray, spacing, percentile=95.0) -> tuple:
    """(symmetric, pred->gt, gt->pred) percentile Hausdorff distance, exactly as pymia computes it."""
    if not pred.any() or not ref.any():
        return np.inf, np.inf, np.inf
    d = pymia_metric.Distances(pred.astype(np.uint8), ref.astype(np.uint8), spacing)

    def perc(distances, areas):
        cum = np.cumsum(areas) / np.sum(areas)
        return distances[min(np.searchsorted(cum, percentile / 100.0), len(distances) - 1)]

    p2g = perc(d.distances_pred_to_gt, d.surfel_areas_pred)
    g2p = perc(d.distances_gt_to_pred, d.surfel_areas_gt)
    return max(p2g, g2p), p2g, g2p


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    s = pred.sum() + ref.sum()
    return 2.0 * np.logical_and(pred, ref).sum() / s if s else np.nan


def keep_largest(mask: np.ndarray, n: int = 2) -> np.ndarray:
    """Keeps the n largest 26-connected components (left and right structure)."""
    labels, count = ndimage.label(mask, structure=STRUCTURE_26)
    if count <= n:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(labels, keep)


# --- main -----------------------------------------------------------------------------------------------------------
def main(args):
    runs = {k: af.load_run(args.results_dir, e, args.seed) for k, e in (('without', args.without),
                                                                        ('with', args.with_))}
    forests, columns = {}, {}
    # the larger feature set is computed for the test subjects; each forest uses its own columns
    big = max(runs, key=lambda k: len(runs[k]['feature_names']))
    big_names = runs[big]['feature_names']
    for key, run in runs.items():
        seed_dir = os.path.join(args.results_dir, run['experiment'], 'analysis', f'seed_{args.seed}')
        x, y = af.training_data(run, seed_dir, args)
        forests[key] = af.fit_forest(run, x, y, args.results_dir)
        missing = set(run['feature_names']) - set(big_names)
        if missing:
            raise SystemExit(f'{run["experiment"]} has features the other run does not have: {sorted(missing)}')
        columns[key] = [big_names.index(n) for n in run['feature_names']]
        del x, y

    pre_process = dict(runs[big]['config']['pre_process'],
                       feature_cache_dir=None if args.feature_cache == 'none' else args.feature_cache)
    putil.load_atlas_images(args.data_atlas_dir)
    train_data = rexp.crawl(args.data_train_dir)
    fpool.LABEL_PRIOR.clear()
    fpool.LABEL_PRIOR.update(fpool.build_label_prior(rexp.registered_ground_truths(train_data)))
    test_data = rexp.crawl(args.data_test_dir)

    saved = {k: pd.read_csv(os.path.join(args.results_dir, r['experiment'], f'seed_{args.seed}', 'metrics.csv'))
             for k, r in runs.items()}
    loc_index = {n: big_names.index(n) for pair in LOCATION_FEATURES.values() for n in pair if n in big_names}

    rows, values, examples, sample_x, sample_y = [], [], {}, [], []
    rng = np.random.default_rng(args.seed)
    for subject, paths in sorted(test_data.items()):
        print(f'----- {subject}')
        img = putil.pre_process(subject, dict(paths), **dict(pre_process, training=False))
        x_all = img.feature_matrix[0]
        gt_img = img.images[structure.BrainImageTypes.GroundTruth]
        gt = sitk.GetArrayFromImage(gt_img)
        spacing = gt_img.GetSpacing()[::-1]
        shape = gt.shape

        # balanced sample of test voxels for the grouped permutation importance
        y_flat = gt.ravel()
        for label in np.unique(y_flat):
            idx = np.flatnonzero(y_flat == label)
            pick = rng.choice(idx, min(args.sample_per_label, len(idx)), replace=False)
            sample_x.append(x_all[pick].astype(np.float32))
            sample_y.append(y_flat[pick])

        preds = {}
        for key in runs:
            proba = forests[key].predict_proba(x_all[:, columns[key]])
            preds[key] = forests[key].classes_[np.argmax(proba, axis=1)].reshape(shape)
            del proba

        for label, name in LABELS.items():
            ref = gt == label
            dist_to_ref = ndimage.distance_transform_edt(~ref, sampling=spacing)  # mm to the nearest true voxel
            for key, pred_all in preds.items():
                pred = pred_all == label
                hd, p2g, g2p = directed_hd(pred, ref, spacing)
                fp = pred & ~ref
                clean = keep_largest(pred)
                hd_c, _, _ = directed_hd(clean, ref, spacing)
                _, n_comp = ndimage.label(pred, structure=STRUCTURE_26)
                check = saved[key].query('subject == @subject_int and label == @name and not postprocessed',
                                         local_dict={'subject_int': int(subject), 'name': name})
                rows.append({'subject': subject, 'label': name, 'model': key,
                             'experiment': runs[key]['experiment'],
                             'dice': dice(pred, ref), 'hd95': hd, 'hd95_pred_to_gt': p2g, 'hd95_gt_to_pred': g2p,
                             'dice_saved': check.query('metric == "DICE"')['value'].item(),
                             'hd95_saved': check.query('metric == "HDRFDST95"')['value'].item(),
                             'n_pred': int(pred.sum()), 'n_gt': int(ref.sum()), 'fp': int(fp.sum()),
                             'fn': int((ref & ~pred).sum()),
                             'fp_far5': int((fp & (dist_to_ref > 5)).sum()),
                             'fp_far10': int((fp & (dist_to_ref > 10)).sum()),
                             'components': n_comp, 'stray_voxels': int(pred.sum() - clean.sum()),
                             'dice_clean': dice(clean, ref), 'hd95_clean': hd_c})
                # feature values of the location features at the far false positives and at the true voxels
                far = (fp & (dist_to_ref > 5)).ravel()
                for group, mask in ((f'far FP ({key})', far), ('true structure', ref.ravel() if key == 'with'
                                                                else None)):
                    if mask is None or not mask.any():
                        continue
                    idx = np.flatnonzero(mask)
                    if len(idx) > 3000:
                        idx = rng.choice(idx, 3000, replace=False)
                    entry = {'label': name, 'group': group, 'subject': subject,
                             'dist_mm': dist_to_ref.ravel()[idx]}
                    for feat in LOCATION_FEATURES[label]:
                        if feat in loc_index:
                            entry[feat] = x_all[idx, loc_index[feat]]
                    values.append(pd.DataFrame(entry))
                if label == 3:
                    examples.setdefault(subject, {})[key] = pred_all == label
            if label == 3:
                examples[subject]['gt'] = ref
                examples[subject]['dist'] = dist_to_ref
                t1 = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T1w])
                examples[subject]['t1'] = t1
        del img, x_all, preds

    table = pd.DataFrame(rows)
    out_dir = os.path.join(args.results_dir, '_report', f'outliers_{args.without}_vs_{args.with_}')
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'per_subject.csv'), index=False)
    feature_values = pd.concat(values, ignore_index=True)
    feature_values.to_csv(os.path.join(out_dir, 'feature_values.csv'), index=False)

    # reproduction check: the retrained forests must give the run's numbers
    dd = (table['dice'] - table['dice_saved']).abs().max()
    dh = (table['hd95'] - table['hd95_saved']).abs().max()
    print(f'\nreproduction of the runs: max |Dice difference| {dd:.2e}, max |HD95 difference| {dh:.2e} mm')

    numeric = ['dice', 'hd95', 'hd95_pred_to_gt', 'hd95_gt_to_pred', 'fp', 'fp_far5', 'fp_far10', 'fn',
               'components', 'stray_voxels', 'dice_clean', 'hd95_clean']
    summary = table.groupby(['label', 'model'], sort=False)[numeric].mean()
    summary.to_csv(os.path.join(out_dir, 'summary.csv'))
    af.print_table(f'mean over the {table["subject"].nunique()} test subjects (seed {args.seed})', summary)

    sample = os.path.join(args.results_dir, runs[big]['experiment'], 'analysis', f'seed_{args.seed}',
                          'test_sample.npz')
    np.savez(sample, X=np.concatenate(sample_x), y=np.concatenate(sample_y), feature_names=big_names)
    print('test voxel sample:', sample)

    if args.plot:
        names = {k: runs[k]['experiment'] for k in runs}
        plot_direction(table, names, out_dir)
        plot_fp_distance(table, feature_values, names, out_dir)
        plot_cleanup(table, names, out_dir)
        plot_feature_at_fp(feature_values, names, out_dir)
        diff = table[table['label'] == 'Hippocampus'].pivot(index='subject', columns='model', values='hd95')
        subject = (diff['without'] - diff['with']).idxmax()
        plot_example(subject, examples[subject], names, table, out_dir)


# --- plots ----------------------------------------------------------------------------------------------------------
def plot_direction(table, names, out_dir):
    """Per structure: HD95 split into its two directions, paired per subject."""
    fig, axes = plt.subplots(1, len(LABELS), figsize=(4.2 * len(LABELS), 4), sharey=True)
    for ax, label in zip(axes, LABELS.values()):
        t = table[table['label'] == label]
        positions = {'hd95_pred_to_gt': (0, 1), 'hd95_gt_to_pred': (3, 4)}
        for metric, (p0, p1) in positions.items():
            a = t[t['model'] == 'without'].set_index('subject')[metric]
            b = t[t['model'] == 'with'].set_index('subject')[metric].reindex(a.index)
            for va, vb in zip(a, b):
                ax.plot([p0, p1], [va, vb], color='0.75', lw=0.8, zorder=1)
            ax.scatter([p0] * len(a), a, color=COLORS['without'], s=18, zorder=2)
            ax.scatter([p1] * len(b), b, color=COLORS['with'], s=18, zorder=2)
            for p, v in ((p0, a.mean()), (p1, b.mean())):
                ax.plot([p - 0.25, p + 0.25], [v, v], color='k', lw=1.5)
                ax.text(p + 0.28, v, f'{v:.1f}', fontsize=7, va='center')
        ax.set_xticks([0.5, 3.5], ['prediction → truth\n(extra voxels far away)',
                                   'truth → prediction\n(parts of the structure missed)'], fontsize=8)
        ax.set_xlim(-0.7, 4.9)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('95th percentile surface distance (mm)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=COLORS[k], label=names[k]) for k in ('without', 'with')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('HD95 = max of the two directions. Which direction changes?  (one line = one test subject, '
                 'bar = mean)', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    af.save_figure(fig, os.path.join(out_dir, 'hd95_direction.png'), tight=False)


def plot_fp_distance(table, values, names, out_dir):
    """How far from the true structure are the false-positive voxels? (all subjects pooled)"""
    fig, axes = plt.subplots(1, len(LABELS), figsize=(4.2 * len(LABELS), 3.6), sharey=True)
    for ax, label in zip(axes, LABELS.values()):
        t = table[table['label'] == label].groupby('model')[['fp', 'fp_far5', 'fp_far10']].sum()
        cats = ['fp', 'fp_far5', 'fp_far10']
        xs = np.arange(len(cats))
        for i, key in enumerate(('without', 'with')):
            ax.bar(xs + (i - 0.5) * 0.38, t.loc[key, cats], width=0.38, color=COLORS[key], label=names[key])
            for xx, v in zip(xs + (i - 0.5) * 0.38, t.loc[key, cats]):
                ax.text(xx, v, f'{int(v)}', ha='center', va='bottom', fontsize=7)
        ax.set_yscale('log')
        ax.set_xticks(xs, ['all false\npositives', '> 5 mm from\nthe structure', '> 10 mm from\nthe structure'],
                      fontsize=8)
        ax.set_title(label)
    axes[0].set_ylabel('false-positive voxels, 10 subjects (log)')
    axes[0].legend(fontsize=7)
    fig.suptitle('False-positive voxels by distance to the true structure', fontsize=10)
    af.save_figure(fig, os.path.join(out_dir, 'fp_distance.png'))


def plot_cleanup(table, names, out_dir):
    """If stray far-away voxels cause the HD95 gap, deleting the stray components should close it."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    m = table.groupby(['label', 'model'])[['hd95', 'hd95_clean', 'dice', 'dice_clean']].mean()
    xs = np.arange(len(LABELS))
    width = 0.2
    bars = [('without', 'hd95', 'raw'), ('without', 'hd95_clean', 'stray voxels removed'),
            ('with', 'hd95', 'raw'), ('with', 'hd95_clean', 'stray voxels removed')]
    for ax, metric, ylabel in ((axes[0], 'hd95', 'HD95 (mm), mean over subjects'),
                               (axes[1], 'dice', 'Dice, mean over subjects')):
        for i, (key, col, kind) in enumerate(bars):
            col = col.replace('hd95', metric)
            v = [m.loc[(lab, key), col] for lab in LABELS.values()]
            ax.bar(xs + (i - 1.5) * width, v, width, color=COLORS[key], alpha=1.0 if kind == 'raw' else 0.45,
                   hatch='' if kind == 'raw' else '//', edgecolor='w', label=f'{names[key]}, {kind}')
            for xx, vv in zip(xs + (i - 1.5) * width, v):
                ax.text(xx, vv, f'{vv:.1f}' if metric == 'hd95' else f'{vv:.3f}', ha='center', va='bottom',
                        fontsize=6 if metric == 'hd95' else 5)
        ax.set_xticks(xs, list(LABELS.values()))
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(0.5, 0.86)
    axes[0].legend(fontsize=7)
    fig.suptitle('Removing everything but the two largest connected components (left + right structure)',
                 fontsize=10)
    af.save_figure(fig, os.path.join(out_dir, 'cleanup.png'))


def plot_feature_at_fp(values, names, out_dir):
    """Value of the prior and of the prior distance at the far false positives vs. inside the true structure."""
    fig, axes = plt.subplots(2, len(LABELS), figsize=(4.2 * len(LABELS), 6.2))
    groups = {'true structure': '#9bbb59', f'far FP (without)': COLORS['without'], f'far FP (with)': COLORS['with']}
    labels_txt = {'true structure': 'inside the true structure',
                  'far FP (without)': f'far false positives, {names["without"]}',
                  'far FP (with)': f'far false positives, {names["with"]}'}
    for col, (label, name) in enumerate(LABELS.items()):
        prior, pdist = LOCATION_FEATURES[label]
        v = values[values['label'] == name]
        for row, feat in enumerate((prior, pdist)):
            ax = axes[row, col]
            data = [v[v['group'] == g][feat].dropna() for g in groups]
            if all(len(d) == 0 for d in data):
                ax.set_axis_off()
                continue
            lo = min(d.min() for d in data if len(d))
            hi = max(d.max() for d in data if len(d))
            bins = np.linspace(lo, hi, 40)
            for (g, color), d in zip(groups.items(), data):
                if len(d):
                    ax.hist(d, bins=bins, color=color, alpha=0.55, density=True,
                            label=f'{labels_txt[g]} (n={len(d)})')
            ax.set_xlabel(feat + (' (mm, < 0 = inside the typical region)' if 'distance' in feat else
                                  ' (fraction of training subjects)'), fontsize=8)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel('density')
    axes[0, 0].legend(fontsize=6.5, loc='upper center')
    fig.suptitle('Location features at the far (> 5 mm) false-positive voxels. The prior is 0 there, the same as '
                 'everywhere outside its region;\nthe prior distance still tells how far away the voxel is.',
                 fontsize=9)
    af.save_figure(fig, os.path.join(out_dir, 'feature_at_fp.png'))


def plot_example(subject, ex, names, table, out_dir):
    """Projections of the predicted hippocampus of one subject: near voxels vs. stray voxels > 5 mm away."""
    t1 = ex['t1']
    brain = t1 != 0
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    # numpy order (z, y, x): axis 2 = sagittal projection (left-right), axis 0 = axial projection (top-down)
    for row, key in enumerate(('without', 'with')):
        pred = ex[key]
        far = pred & (ex['dist'] > 5)
        near = pred & ~far
        stats = table[(table['subject'] == subject) & (table['label'] == 'Hippocampus') & (table['model'] == key)]
        for col, (axis, view) in enumerate(((2, 'sagittal (projected left-right)'),
                                            (0, 'axial (projected top-down)'))):
            ax = axes[row, col]
            bg = brain.any(axis=axis)
            ax.imshow(np.where(bg, 0.85, 1.0), cmap='gray', vmin=0, vmax=1, origin='lower')
            ax.contour(ex['gt'].any(axis=axis), levels=[0.5], colors='#2e7d32', linewidths=1.2)
            for mask, color in ((near, '#f0a030'), (far, '#d62728')):
                proj = mask.any(axis=axis)
                ax.imshow(np.ma.masked_where(~proj, proj), cmap=matplotlib.colors.ListedColormap([color]),
                          origin='lower', alpha=0.9, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(view, fontsize=9)
            if col == 0:
                r = stats.iloc[0]
                ax.set_ylabel(f'{names[key]}\nDice {r["dice"]:.3f}, HD95 {r["hd95"]:.1f} mm\n'
                              f'{int(r["fp_far5"])} voxels > 5 mm away', fontsize=8)
    handles = [plt.Line2D([], [], color='#2e7d32', label='true hippocampus (outline)'),
               plt.Line2D([], [], marker='s', ls='', color='#f0a030', label='predicted, within 5 mm'),
               plt.Line2D([], [], marker='s', ls='', color='#d62728', label='predicted, > 5 mm away (stray)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle(f'Test subject {subject}: predicted hippocampus, all slices projected onto one image', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    af.save_figure(fig, os.path.join(out_dir, f'example_{subject}.png'), tight=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='False positives far from the structure, with and without features.')
    parser.add_argument('--without', default='extended_plus_no_distance', help='run without the feature(s)')
    parser.add_argument('--with', dest='with_', default='extended_plus', help='run with the feature(s)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--sample_per_label', type=int, default=500,
                        help='test voxels per label and subject saved for the permutation importance')
    parser.add_argument('--results_dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--data_atlas_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'atlas'))
    parser.add_argument('--data_train_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'train'))
    parser.add_argument('--data_test_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'test'))
    parser.add_argument('--feature_cache', default=os.path.join(SCRIPT_DIR, 'feature_cache'))
    main(parser.parse_args())
