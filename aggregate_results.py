"""Collects all runs in results/ into one table, compares experiments with a baseline and draws boxplots.

Usage:
    python aggregate_results.py --baseline baseline
    python aggregate_results.py --baseline baseline --experiments baseline t2 noise_padded --plot

Writes to results/_aggregate/:
    all_metrics.csv     every row of every metrics.csv
    summary_table.csv   per experiment x label x metric: mean, std over subjects, std over seeds
    wilcoxon.csv        paired Wilcoxon test of each experiment against the baseline
    boxplot_<metric>.png  (with --plot)
"""
import argparse
import glob
import json
import os

import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_ORDER = ['WhiteMatter', 'GreyMatter', 'Thalamus', 'Hippocampus', 'Amygdala']
LOWER_IS_BETTER = {'HDRFDST95'}


def load_all(results_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*', 'seed_*', 'metrics.csv'))):
        df = pd.read_csv(path)
        with open(os.path.join(os.path.dirname(path), 'config.json')) as f:
            df['n_features'] = json.load(f)['n_features']
        frames.append(df)
    if not frames:
        raise SystemExit(f'no results found in {results_dir}')
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ['experiment', 'postprocessed', 'label', 'metric']
    # average the seeds per subject first, so every subject counts once
    per_subject = df.groupby(keys + ['subject'])['value'].mean().reset_index()
    over_subjects = per_subject.groupby(keys)['value'].agg(mean='mean', std_subjects='std',
                                                           n_subjects='count').reset_index()
    # how much the mean moves between seeds (random sampling / forest)
    per_seed = df.groupby(keys + ['seed'])['value'].mean().reset_index()
    over_seeds = per_seed.groupby(keys)['value'].agg(std_seeds='std', n_seeds='count').reset_index()
    n_features = df.groupby('experiment')['n_features'].first().reset_index()
    return over_subjects.merge(over_seeds, on=keys).merge(n_features, on='experiment')


def wilcoxon_vs_baseline(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    keys = ['postprocessed', 'label', 'metric']
    per_subject = df.groupby(['experiment'] + keys + ['subject'])['value'].mean().reset_index()
    base = per_subject[per_subject.experiment == baseline]
    rows = []
    for exp in sorted(set(per_subject.experiment) - {baseline}):
        other = per_subject[per_subject.experiment == exp]
        paired = base.merge(other, on=keys + ['subject'], suffixes=('_base', '_exp'))
        for (pp, label, metric), g in paired.groupby(keys):
            diff = g.value_exp - g.value_base
            p = stats.wilcoxon(g.value_exp, g.value_base).pvalue if (diff != 0).any() and len(g) > 1 else 1.0
            better = diff.median() < 0 if metric in LOWER_IS_BETTER else diff.median() > 0
            rows.append({'experiment': exp, 'baseline': baseline, 'postprocessed': pp, 'label': label,
                         'metric': metric, 'n_subjects': len(g), 'median_diff': diff.median(),
                         'mean_diff': diff.mean(), 'p_value': p, 'better': bool(better),
                         'significant_0.05': p < 0.05})
    return pd.DataFrame(rows)


def boxplots(df: pd.DataFrame, out_dir: str, experiments: list):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    per_subject = df[~df.postprocessed].groupby(['experiment', 'label', 'metric', 'subject'])['value'] \
        .mean().reset_index()
    for metric, g in per_subject.groupby('metric'):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.boxplot(data=g, x='label', y='value', hue='experiment', order=LABEL_ORDER, hue_order=experiments, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(f'{metric} (mm)' if metric in LOWER_IS_BETTER else metric)
        ax.legend(title='', frameon=False, ncol=min(len(experiments), 4))
        sns.despine()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'boxplot_{metric}.png'), dpi=200)
        plt.close(fig)


def main(results_dir: str, baseline: str, experiments: list, plot: bool):
    df = load_all(results_dir)
    if experiments:
        df = df[df.experiment.isin(experiments)]
    experiments = experiments or sorted(df.experiment.unique())

    out_dir = os.path.join(results_dir, '_aggregate')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'all_metrics.csv'), index=False)

    table = summary_table(df)
    table.to_csv(os.path.join(out_dir, 'summary_table.csv'), index=False)
    view = table[~table.postprocessed].copy()
    view['mean ± std'] = view['mean'].map('{:.3f}'.format) + ' ± ' + view['std_subjects'].map('{:.3f}'.format)
    print(view.pivot_table(index=['metric', 'label'], columns='experiment', values='mean ± std', aggfunc='first')
          .to_string())

    if baseline:
        tests = wilcoxon_vs_baseline(df, baseline)
        tests.to_csv(os.path.join(out_dir, 'wilcoxon.csv'), index=False)
        if not tests.empty:
            print(f'\nWilcoxon vs {baseline} (no post-processing):')
            print(tests[~tests.postprocessed][['experiment', 'label', 'metric', 'median_diff', 'p_value', 'better']]
                  .to_string(index=False))

    if plot:
        boxplots(df, out_dir, experiments)
    print(f'\nwritten to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate experiment results.')
    parser.add_argument('--results_dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--baseline', default=None, help='experiment name to compare against')
    parser.add_argument('--experiments', nargs='*', default=None, help='only these experiments')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args.results_dir, args.baseline, args.experiments, args.plot)
