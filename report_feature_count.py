"""Table and plots of the feature-count report (mialab_notes/reports/feature_count_hypothesis.html) for the slides
and the written report: Dice and HD95 per structure for each feature set, on a white background.

Usage (after aggregate_results.py):
    python report_feature_count.py
    python report_feature_count.py --experiments baseline prior_only extended_plus
    # comparison with extended_plus without the label prior and the prior distances, written as *_with_no_prior.*
    python report_feature_count.py --experiments baseline baseline_t2 prior_only extended_plus_reduced_r09 \
        extended_plus_no_prior extended extended_plus_no_distance extended_plus --suffix _with_no_prior

Writes to results/_report/:
    feature_count_table.csv     one row per experiment: number of features, mean Dice / HD95 per structure, run time
    dice_per_structure.png      Dice per structure, experiments ordered by the number of features
    hd95_per_structure.png      HD95 per structure
    dice_hd95_per_structure.png both side by side
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# experiments of the report, in order of the number of features: short name for the x axis, added information
EXPERIMENTS = {
    'baseline': ('baseline', 'coordinates, T1 intensity and gradient'),
    'baseline_t2': ('+T2', '+ T2 intensity and gradient'),
    'prior_only': ('+prior', '+ label prior (5 structures)'),
    'extended_plus_reduced_r09': ('reduced', 'extended_plus minus 10 correlated features'),
    'extended_plus_no_prior': ('no prior', 'extended_plus without the label prior and the prior distances'),
    'extended': ('extended', '+ multi-scale, local statistics, T1-T2'),
    'extended_plus_no_distance': ('no dist.', 'extended_plus without the 4 distance features'),
    'extended_plus': ('ext_plus', '+ distances, prior distances, symmetry'),
}
# left out unless asked for with --experiments (comparison plots, see the usage above)
NOT_DEFAULT = ('extended_plus_no_prior',)
LABELS = {'WhiteMatter': ('White matter', '#8a6d1f'), 'GreyMatter': ('Grey matter', '#5b5fa8'),
          'Thalamus': ('Thalamus', '#2f7a3e'), 'Hippocampus': ('Hippocampus', '#c0562a'),
          'Amygdala': ('Amygdala', '#b0336c')}
METRICS = {'DICE': ('dice', 'Dice (higher is better)'), 'HDRFDST95': ('hd95', 'HD95 in mm (lower is better)')}


def test_time(results_dir: str, experiment: str) -> float:
    """Mean test pre-processing time over the seeds (s)."""
    times = []
    for path in glob.glob(os.path.join(results_dir, experiment, 'seed_*', 'timing.json')):
        with open(path) as f:
            times.append(json.load(f)['preprocess_test_s'])
    return sum(times) / len(times) if times else float('nan')


def build_table(summary: pd.DataFrame, experiments: list, results_dir: str) -> pd.DataFrame:
    summary = summary[~summary['postprocessed']]
    rows = []
    for experiment in experiments:
        part = summary[summary['experiment'] == experiment]
        row = {'experiment': experiment, 'n_features': int(part['n_features'].iloc[0]),
               'added_information': EXPERIMENTS.get(experiment, ('', ''))[1]}
        for metric, (prefix, _) in METRICS.items():
            means = part[part['metric'] == metric].set_index('label')['mean']
            row[f'{prefix}_mean'] = means.reindex(list(LABELS)).mean()
            for label in LABELS:
                row[f'{prefix}_{label}'] = means.get(label)
        row['test_preprocessing_s'] = test_time(results_dir, experiment)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('n_features', kind='stable').reset_index(drop=True)


def draw(ax, table: pd.DataFrame, prefix: str, y_label: str, step: tuple):
    x = range(len(table))
    if step:
        # the step that holds the gain: from the 7-feature to the 12-feature set
        ax.axvspan(step[0], step[1], color='#1f6f78', alpha=0.09, linewidth=0)
    for label, (name, color) in LABELS.items():
        ax.plot(x, table[f'{prefix}_{label}'], marker='o', markersize=5, linewidth=2, color=color, label=name)
    ax.set_xticks(list(x), [f'{EXPERIMENTS.get(e, (e,))[0]}\n{n} feat.'
                            for e, n in zip(table['experiment'], table['n_features'])], fontsize=9)
    ax.set_ylabel(y_label)
    ax.grid(axis='y', color='#dfe3e8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def main(results_dir: str, experiments: list, suffix: str = ''):
    summary = pd.read_csv(os.path.join(results_dir, '_aggregate', 'summary_table.csv'))
    available = list(dict.fromkeys(summary['experiment']))
    experiments = experiments or [e for e in EXPERIMENTS if e in available and e not in NOT_DEFAULT]
    missing = [e for e in experiments if e not in available]
    if missing:
        raise SystemExit(f'not in summary_table.csv (run aggregate_results.py first): {", ".join(missing)}')

    out_dir = os.path.join(results_dir, '_report')
    os.makedirs(out_dir, exist_ok=True)
    table = build_table(summary, experiments, results_dir)
    table.round(4).to_csv(os.path.join(out_dir, f'feature_count_table{suffix}.csv'), index=False)
    print(table.round(3).to_string(index=False))

    order = list(table['experiment'])
    step = (order.index('baseline_t2'), order.index('prior_only')) \
        if {'baseline_t2', 'prior_only'} <= set(order) else None
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                         'font.size': 10})
    width = max(6.5, 0.95 * len(table) + 1.5)
    for metric, (prefix, y_label) in METRICS.items():
        fig, ax = plt.subplots(figsize=(width, 4.5))
        draw(ax, table, prefix, y_label, step)
        ax.set_title(f'{"Dice" if prefix == "dice" else "HD95"} per structure')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{prefix}_per_structure{suffix}.png'), dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 4.8))
    for ax, (metric, (prefix, y_label)) in zip(axes, METRICS.items()):
        draw(ax, table, prefix, y_label, step)
        ax.set_title(f'{"Dice" if prefix == "dice" else "HD95"} per structure')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, f'dice_hd95_per_structure{suffix}.png'), dpi=200)
    plt.close(fig)
    print('written to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table and plots of the feature-count report.')
    parser.add_argument('--results_dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='default: all experiments of the report that have results, except NOT_DEFAULT')
    parser.add_argument('--suffix', default='', help='appended to the output file names, e.g. _with_no_prior')
    args = parser.parse_args()
    main(args.results_dir, args.experiments, args.suffix)
