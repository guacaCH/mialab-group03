"""Feature importance on brain voxels only (Marc's correction of the importance analyses).

The training voxels of the pipeline are sampled over the whole image, so ~31 % of them lie outside the brain mask
(all background, T1 = T2 = 0, distance to the brain border < 0). They make "inside vs. outside the head" the easiest
split and inflate the importance of the features that encode it. The pipeline itself stays as it is (Dice changes
< 0.002 without these rows); only the importance analyses drop them.

For each run: load the run's training voxels (train.npz), drop the rows outside the brain, train the forest again with
the run's settings and seed, then compute
  * information gain per feature (same method as analyze_information_gain.py),
  * SHAP per label on a balanced sample of 2000 brain voxels (same method as analyze_features.py shap; main run only),
  * Spearman correlation pairs |r| >= 0.9 on the brain voxels (main run only).

Usage (from the repository root):
    .venv/bin/python "Final Presentation/brain_only_importance.py"

Writes to results/<experiment>/analysis/seed_0/brain_only/.
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn.ensemble as ske

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import analyze_features as af  # noqa: E402
import analyze_information_gain as aig  # noqa: E402

RES = os.path.join(ROOT, 'results')
MAIN = 'extended_plus'
RUNS = ['baseline', 'baseline_t2', 'prior_only', 'extended', 'extended_plus_no_distance', 'extended_plus',
        'extended_plus_reduced_r09']
SEED = 0


def inside_brain(x: np.ndarray, names: list) -> np.ndarray:
    """Rows inside the brain mask. Outside, the skull-stripped images are exactly 0 (and the distance is < 0)."""
    if 'distance_brain_border' in names:
        return x[:, names.index('distance_brain_border')] >= 0
    zero = x[:, names.index('T1_intensity')] == 0
    if 'T2_intensity' in names:
        zero &= x[:, names.index('T2_intensity')] == 0
    return ~zero


def information_gain(forest, names, families) -> tuple:
    nodes = aig.forest_gains(forest)
    nodes['name'] = np.array(names)[nodes['feature']]
    per_tree = nodes.pivot_table(index='tree', columns='name', values='info_gain', aggfunc='sum') \
        .reindex(columns=names).fillna(0.0)
    t = pd.DataFrame(index=pd.Index(names, name='feature'))
    t['family'] = families.reindex(names).values
    t['info_gain_bits'] = per_tree.mean()
    t['info_gain_std'] = per_tree.std()
    t['share'] = t['info_gain_bits'] / t['info_gain_bits'].sum()
    t['gini_importance'] = forest.feature_importances_
    t = t.sort_values('info_gain_bits', ascending=False)
    t['rank'] = np.arange(1, len(t) + 1)
    t['cumulative_share'] = t['share'].cumsum()
    roots = nodes[nodes['depth'] == 0]['name'].value_counts()
    return t, aig.root_entropy(forest), roots


def main():
    import shap
    totals = []
    for exp in RUNS:
        run = af.load_run(RES, exp, SEED)
        seed_dir = os.path.join(RES, exp, 'analysis', f'seed_{SEED}')
        data = np.load(os.path.join(seed_dir, 'train.npz'))
        x, y, names = data['X'], data['y'], list(data['feature_names'])
        keep = inside_brain(x, names)
        xb, yb = x[keep], y[keep]
        out = os.path.join(seed_dir, 'brain_only')
        os.makedirs(out, exist_ok=True)
        families = af.feature_families(RES, exp, SEED)

        forest = ske.RandomForestClassifier(random_state=run['seed'], **run['config']['forest']).fit(xb, yb)
        table, h_root, roots = information_gain(forest, names, families)
        table.to_csv(os.path.join(out, 'infogain_features.csv'))
        totals.append({'experiment': exp, 'n_features': len(names), 'rows_all': len(y), 'rows_brain': int(keep.sum()),
                       'outside_share': 1 - keep.mean(), 'root_entropy_bits': h_root,
                       'total_info_gain_bits': table['info_gain_bits'].sum(),
                       'n_for_80pct': int((table['cumulative_share'] < 0.8).sum() + 1),
                       'n_for_90pct': int((table['cumulative_share'] < 0.9).sum() + 1),
                       'top_root_feature': roots.index[0], 'top_feature': table.index[0]})
        print(f'{exp}: {keep.sum()} of {len(y)} rows inside the brain; root splits: {dict(roots.head(5))}')

        if exp != MAIN:
            continue
        gini_rank = table['gini_importance'].rank(ascending=False).astype(int)
        print('Gini rank on brain rows:', {f: int(gini_rank[f]) for f in ['T1_intensity', 'distance_brain_border']})

        index = af.balanced_sample(yb, 2000, SEED)
        values = shap.TreeExplainer(forest).shap_values(xb[index], check_additivity=False)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        labels = [af.LABEL_NAMES.get(int(c), str(c)) for c in forest.classes_]
        per_label = pd.DataFrame(np.abs(values).mean(axis=0), index=names, columns=labels)
        per_label = per_label.assign(overall=per_label.mean(axis=1)).sort_values('overall', ascending=False)
        per_label['rank'] = np.arange(1, len(per_label) + 1)
        per_label.to_csv(os.path.join(out, 'shap_importance.csv'), index_label='feature')

        corr = pd.DataFrame(xb, columns=names).corr(method='spearman')
        pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename('r').reset_index()
        pairs = pairs[pairs['r'].abs() >= 0.9].sort_values('r', key=abs, ascending=False)
        pairs.columns = ['feature_a', 'feature_b', 'r']
        pairs.to_csv(os.path.join(out, 'correlation_pairs.csv'), index=False)

    totals = pd.DataFrame(totals)
    totals.to_csv(os.path.join(RES, '_report', 'infogain_totals_brain_only.csv'), index=False)
    print(totals.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
