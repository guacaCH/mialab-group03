"""Information gain of the random forest: per feature, per feature family and per tree depth.

Usage:
    python analyze_information_gain.py --experiment extended_plus --plot
    python analyze_information_gain.py --compare baseline baseline_t2 prior_only extended extended_plus_no_distance \
        extended_plus --plot
    python analyze_information_gain.py --experiment extended_plus --permutation --plot   # needs test_sample.npz

How it works (see mialab_notes/information_gain.md):
    Every split node of every tree sends its voxels left or right. Its information gain is
        IG = H(parent) - n_left/n * H(left) - n_right/n * H(right),   H = Shannon entropy of the labels in bits.
    Weighted by the fraction of voxels that reach the node and summed over all nodes that split on a feature, this is
    the information that feature contributes to one tree. The forest value is the mean over the trees. The forest is
    grown with the Gini criterion; the same sum with the Gini impurity instead of the entropy is exactly sklearn's
    feature_importances_ (checked below), so information gain is the entropy version of the forest importance.
    The forest is trained again from the run's training voxels (results/<experiment>/analysis/seed_<seed>/train.npz,
    computed by analyze_features.py if missing), with the run's settings and seed.

Writes to results/<experiment>/analysis/seed_<seed>/infogain/:
    infogain_features.csv   per feature: information gain (bits, mean/std over trees), share, Gini importance,
                            number of splits, mean split depth, family
    infogain_family.csv     per family: summed information gain, share, number of features
    infogain_depth.csv      family x tree depth: information gain gained at that depth
    infogain_features.png, infogain_family.png, infogain_depth.png, infogain_cumulative.png, tree_top.png
    permutation_family.csv / .png (with --permutation): drop in accuracy when a whole family is shuffled on test voxels
With --compare, results/_report/infogain_by_family.csv / .png: the families' share of the information in each set.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_features as af  # noqa: E402

SCRIPT_DIR = af.SCRIPT_DIR
LABEL_NAMES = af.LABEL_NAMES

# fixed order (= order in the feature pool), so a family has the same colour in every plot and experiment
FAMILY_ORDER = ['spatial', 'intensity', 'gradient', 'smoothing', 'log', 'localstats', 'combination', 'prior',
                'distance', 'priordistance', 'symmetry', 'control']
FAMILY_COLORS = af.family_colors(pd.Series(FAMILY_ORDER))


# --- information gain of one tree ---------------------------------------------------------------------------------
def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row of class proportions."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)


def gini(p: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(p ** 2, axis=1)


def node_gains(tree) -> pd.DataFrame:
    """One row per split node: feature, depth, weight (fraction of the tree's voxels), entropy and Gini gain."""
    t = tree.tree_
    value = t.value[:, 0, :]
    p = value / value.sum(axis=1, keepdims=True)  # class proportions (sklearn may store counts or fractions)
    w = t.weighted_n_node_samples
    h, g = entropy(p), gini(p)

    depth = np.zeros(t.node_count, dtype=int)
    for node in range(t.node_count):  # children always have a higher index than their parent
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                depth[child] = depth[node] + 1

    split = np.flatnonzero(t.children_left != -1)
    left, right = t.children_left[split], t.children_right[split]
    n = w[split]
    ig = h[split] - (w[left] * h[left] + w[right] * h[right]) / n
    gg = g[split] - (w[left] * g[left] + w[right] * g[right]) / n
    weight = n / w[0]
    return pd.DataFrame({'node': split, 'feature': t.feature[split], 'threshold': t.threshold[split],
                         'depth': depth[split], 'weight': weight, 'n': n,
                         'info_gain': weight * ig, 'gini_gain': weight * gg,
                         'node_info_gain': ig, 'entropy': h[split],
                         'majority': np.argmax(p[split], axis=1)})


def forest_gains(forest) -> pd.DataFrame:
    """node_gains of every tree, with the tree index."""
    return pd.concat([node_gains(tree).assign(tree=i) for i, tree in enumerate(forest.estimators_)],
                     ignore_index=True)


def root_entropy(forest) -> float:
    """Mean entropy (bits) of the labels at the roots (= of the bootstrap samples): the information to be resolved."""
    values = []
    for tree in forest.estimators_:
        v = tree.tree_.value[0, 0, :]
        values.append(entropy((v / v.sum())[None])[0])
    return float(np.mean(values))


# --- analysis of one experiment -------------------------------------------------------------------------------------
def analyze(experiment: str, args, plot: bool) -> dict:
    run = af.load_run(args.results_dir, experiment, args.seed)
    seed_dir = os.path.join(args.results_dir, experiment, 'analysis', f'seed_{args.seed}')
    x, y = af.training_data(run, seed_dir, args)
    forest = af.fit_forest(run, x, y, args.results_dir)
    names = np.array(run['feature_names'])
    families = af.feature_families(args.results_dir, experiment, args.seed).reindex(names)
    nodes = forest_gains(forest)
    nodes['name'] = names[nodes['feature']]
    nodes['family'] = families.values[nodes['feature']]
    n_trees = len(forest.estimators_)

    # per feature: summed over the nodes of a tree, then mean / std over the trees
    per_tree = nodes.pivot_table(index='tree', columns='name', values='info_gain', aggfunc='sum') \
        .reindex(columns=names).fillna(0.0)
    gini_tree = nodes.pivot_table(index='tree', columns='name', values='gini_gain', aggfunc='sum') \
        .reindex(columns=names).fillna(0.0)
    # sklearn's importance = per-tree Gini gain normalised to 1, then averaged over the trees
    gini_check = (gini_tree.div(gini_tree.sum(axis=1), axis=0)).mean().values
    if not np.allclose(gini_check, forest.feature_importances_, atol=1e-8):
        print('note: the Gini gain does not reproduce sklearn\'s feature_importances_')

    table = pd.DataFrame(index=pd.Index(names, name='feature'))
    table['family'] = families.values
    table['info_gain_bits'] = per_tree.mean()
    table['info_gain_std'] = per_tree.std()
    table['share'] = table['info_gain_bits'] / table['info_gain_bits'].sum()
    table['gini_importance'] = forest.feature_importances_
    grouped = nodes.groupby('name')
    table['splits_per_tree'] = grouped.size().reindex(names).fillna(0) / n_trees
    table['mean_split_depth'] = grouped[['depth', 'info_gain']].apply(lambda d: np.average(d['depth'], weights=d['info_gain'])
                                              if d['info_gain'].sum() > 0 else np.nan).reindex(names)
    table = table.sort_values('info_gain_bits', ascending=False)
    table['rank'] = np.arange(1, len(table) + 1)
    table['cumulative_share'] = table['share'].cumsum()

    family = table.groupby('family').agg(info_gain_bits=('info_gain_bits', 'sum'), share=('share', 'sum'),
                                         n_features=('share', 'size'))
    family['bits_per_feature'] = family['info_gain_bits'] / family['n_features']
    family = family.sort_values('info_gain_bits', ascending=False)

    depth = nodes.pivot_table(index='depth', columns='family', values='info_gain', aggfunc='sum').fillna(0.0) / n_trees
    depth = depth.reindex(columns=[f for f in FAMILY_ORDER if f in depth.columns])

    h_root = root_entropy(forest)
    out_dir = os.path.join(seed_dir, 'infogain')
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'infogain_features.csv'))
    family.to_csv(os.path.join(out_dir, 'infogain_family.csv'))
    depth.to_csv(os.path.join(out_dir, 'infogain_depth.csv'))
    af.print_table(f'information gain per feature ({experiment}, seed {args.seed}; bits, mean over {n_trees} trees)',
                   table[['family', 'info_gain_bits', 'share', 'cumulative_share', 'gini_importance',
                          'splits_per_tree', 'mean_split_depth']])
    af.print_table('information gain per family', family)
    print(f'\nlabel entropy at the root: {h_root:.3f} bits; resolved by the splits: '
          f'{table["info_gain_bits"].sum():.3f} bits ({table["info_gain_bits"].sum() / h_root:.1%})')

    if plot:
        plot_features(table, experiment, out_dir)
        plot_family(family, experiment, h_root, out_dir)
        plot_depth(depth, experiment, out_dir)
        plot_cumulative(table, experiment, out_dir)
        plot_tree_top(forest, names, nodes, experiment, out_dir)
    return {'table': table, 'family': family, 'root_entropy': h_root, 'forest': forest, 'run': run}


# --- plots ----------------------------------------------------------------------------------------------------------
def family_legend(ax, families, **kwargs):
    present = [f for f in FAMILY_ORDER if f in set(families)]
    ax.legend(handles=[Patch(color=FAMILY_COLORS[f], label=f) for f in present], fontsize=7, **kwargs)


def plot_features(table, experiment, out_dir):
    fig, ax = plt.subplots(figsize=(7, 0.25 * len(table) + 1.5))
    t = table[::-1]
    ax.barh(t.index, t['info_gain_bits'], xerr=t['info_gain_std'], color=[FAMILY_COLORS[f] for f in t['family']],
            error_kw={'elinewidth': 0.6, 'ecolor': '0.4'})
    family_legend(ax, table['family'], loc='lower right')
    ax.set_xlabel('information gain per tree (bits), mean ± std over the trees')
    ax.set_title(f'{experiment}: information gain per feature')
    ax.tick_params(axis='y', labelsize=7)
    af.save_figure(fig, os.path.join(out_dir, 'infogain_features.png'))


def plot_family(family, experiment, h_root, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.35 * len(family) + 1.8), sharey=True)
    f = family[::-1]
    colors = [FAMILY_COLORS[n] for n in f.index]
    axes[0].barh(f.index, f['info_gain_bits'], color=colors)
    for i, (bits, share, n) in enumerate(zip(f['info_gain_bits'], f['share'], f['n_features'])):
        axes[0].text(bits, i, f' {share:.0%}  ({int(n)} feat.)', va='center', fontsize=7)
    axes[0].set_xlabel('information gain of the family (bits per tree)')
    axes[0].set_xlim(0, f['info_gain_bits'].max() * 1.35)
    axes[0].set_title('total per family')
    axes[1].barh(f.index, f['bits_per_feature'], color=colors)
    axes[1].set_xlabel('bits per feature in the family')
    axes[1].set_title('per feature (total / number of features)')
    fig.suptitle(f'{experiment}: information gain grouped by feature family '
                 f'(label entropy at the root: {h_root:.2f} bits)', fontsize=10)
    af.save_figure(fig, os.path.join(out_dir, 'infogain_family.png'))


def plot_depth(depth, experiment, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bottom = np.zeros(len(depth))
    share = depth.div(depth.sum(axis=1), axis=0)
    bottom_share = np.zeros(len(depth))
    for family in depth.columns:
        axes[0].bar(depth.index, depth[family], bottom=bottom, color=FAMILY_COLORS[family], label=family, width=0.85)
        axes[1].bar(depth.index, share[family], bottom=bottom_share, color=FAMILY_COLORS[family], width=0.85)
        bottom += depth[family].values
        bottom_share += share[family].values
    axes[0].set_xlabel('tree depth of the split (0 = root)')
    axes[0].set_ylabel('information gain (bits per tree)')
    axes[0].set_title('where in the tree the information is gained')
    axes[1].set_xlabel('tree depth of the split (0 = root)')
    axes[1].set_ylabel('share of the information gained at that depth')
    axes[1].set_title('which family splits at which depth')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(depth.index[::2])
    family_legend(axes[0], depth.columns, loc='upper right', ncol=2)
    fig.suptitle(experiment, fontsize=10)
    af.save_figure(fig, os.path.join(out_dir, 'infogain_depth.png'))


def plot_cumulative(table, experiment, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    k = np.arange(1, len(table) + 1)
    ax.bar(k, table['share'], color=[FAMILY_COLORS[f] for f in table['family']])
    ax2 = ax.twinx()
    ax2.plot(k, table['cumulative_share'], color='k', marker='o', markersize=3, lw=1)
    for level in (0.8, 0.9):
        n = int(np.searchsorted(table['cumulative_share'].values, level) + 1)
        ax2.axhline(level, color='0.6', lw=0.6, ls='--')
        ax2.annotate(f'{level:.0%} with {n} features', (n, level), xytext=(n + 1.5, level - 0.09), fontsize=8,
                     arrowprops={'arrowstyle': '-', 'color': '0.5', 'lw': 0.6})
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel('cumulative share')
    ax.set_xlabel('features, ranked by information gain')
    ax.set_ylabel('share of the information gain')
    family_legend(ax, table['family'], loc='center right', ncol=2)
    ax.set_title(f'{experiment}: how many features carry the information')
    af.save_figure(fig, os.path.join(out_dir, 'infogain_cumulative.png'))


def plot_tree_top(forest, names, nodes, experiment, out_dir, levels: int = 3):
    """The first levels of the first tree: split feature, threshold, voxels, majority label and information gain."""
    tree = forest.estimators_[0].tree_
    info = nodes[nodes['tree'] == 0].set_index('node')
    classes = [LABEL_NAMES.get(int(c), str(c)) for c in forest.classes_]
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_axis_off()
    width = 2 ** levels

    def draw(node, depth, x):
        if depth > levels:
            return
        value = tree.value[node, 0]
        p = value / value.sum()
        share = tree.weighted_n_node_samples[node] / tree.weighted_n_node_samples[0]
        text = f'{share:.0%} of voxels, H = {max(entropy(p[None])[0], 0):.2f} bits\nmost: {classes[np.argmax(p)]} ' \
               f'({p.max():.0%})'
        is_split = tree.children_left[node] != -1
        if is_split and depth < levels:
            row = info.loc[node]
            family = row['family']
            text = f'{names[tree.feature[node]]} ≤ {tree.threshold[node]:.2f}\n' + text + \
                   f'\nIG = {row["node_info_gain"]:.2f} bits'
            color = FAMILY_COLORS[family]
        else:
            color = (0.93, 0.93, 0.93)
        y = 1 - depth / levels
        ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
                bbox={'boxstyle': 'round,pad=0.35', 'fc': (*color[:3], 0.35), 'ec': color})
        if is_split and depth < levels:
            dx = width / 2 ** (depth + 2)
            for child, cx, edge in ((tree.children_left[node], x - dx, 'yes'), (tree.children_right[node], x + dx,
                                                                                  'no')):
                ax.annotate('', (cx, 1 - (depth + 1) / levels + 0.07), (x, y - 0.07),
                            arrowprops={'arrowstyle': '->', 'color': '0.5', 'lw': 0.7})
                ax.text((x + cx) / 2, y - 0.5 / levels, edge, fontsize=6, color='0.4', ha='center')
                draw(child, depth + 1, cx)

    draw(0, 0, width / 2)
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.12, 1.12)
    ax.set_title(f'{experiment}: first {levels} levels of tree 0 (box colour = feature family; '
                 f'IG = information gain of that split)', fontsize=9)
    af.save_figure(fig, os.path.join(out_dir, 'tree_top.png'))


# --- grouped permutation importance (family) on test voxels ----------------------------------------------------------
def permutation_family(result: dict, experiment: str, args, plot: bool):
    """Shuffle all features of a family together on held-out test voxels and measure how much worse the forest gets.

    Unlike information gain (measured on the training voxels while the tree is grown), this measures what the trained
    forest really loses on new subjects. Shuffling the whole family at once also removes the effect of correlated
    features standing in for each other (e.g. the T1 smoothing scales).
    """
    from sklearn.metrics import f1_score
    seed_dir = os.path.join(args.results_dir, experiment, 'analysis', f'seed_{args.seed}')
    path = os.path.join(seed_dir, 'test_sample.npz')
    if not os.path.exists(path):
        raise SystemExit(f'{path} is missing: run analyze_outliers.py first (it saves a sample of test voxels)')
    data = np.load(path)
    names = list(result['run']['feature_names'])
    columns = [list(data['feature_names']).index(n) for n in names]
    x, y = data['X'][:, columns], data['y']
    forest = result['forest']
    families = result['table']['family'].reindex(names)
    labels = forest.classes_

    def scores(xx):
        pred = forest.predict(xx)
        f1 = f1_score(y, pred, labels=labels, average=None)
        return np.concatenate([[np.mean(pred == y)], f1])

    columns_out = ['accuracy'] + [LABEL_NAMES[int(c)] for c in labels]
    base = scores(x)
    rng = np.random.default_rng(args.seed)
    rows = {}
    for family in [f for f in FAMILY_ORDER if f in set(families)]:
        idx = [i for i, n in enumerate(names) if families[n] == family]
        drops = []
        for _ in range(args.repeats):
            xp = x.copy()
            perm = rng.permutation(len(xp))
            xp[:, idx] = xp[perm][:, idx]  # the same permutation for all columns: keeps the family's joint values
            drops.append(base - scores(xp))
        rows[family] = np.mean(drops, axis=0)
    table = pd.DataFrame(rows, index=columns_out).T
    table.index.name = 'family'
    table.insert(0, 'n_features', [int(np.sum(families == f)) for f in table.index])
    table = table.sort_values('accuracy', ascending=False)
    out_dir = os.path.join(seed_dir, 'infogain')
    table.to_csv(os.path.join(out_dir, 'permutation_family.csv'))
    af.print_table(f'drop in accuracy / F1 per label when a family is shuffled ({len(y)} test voxels, '
                   f'{args.repeats} repeats); baseline accuracy {base[0]:.3f}', table)

    if plot:
        values = table.drop(columns=['n_features'])
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(values) + 1.8))
        im = ax.imshow(values.values, cmap='Reds', aspect='auto', vmin=0)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(values.shape[0]), [f'{f} ({n})' for f, n in zip(values.index, table['n_features'])],
                      fontsize=8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values.values[i, j]
                ax.text(j, i, f'{v:.2f}' if abs(v) >= 0.005 else '·', ha='center', va='center', fontsize=7,
                        color='w' if v > 0.6 * values.values.max() else 'k')
        fig.colorbar(im, ax=ax, label='drop when the family is shuffled')
        ax.set_title(f'{experiment}: grouped (family) permutation importance on test voxels\n'
                     f'columns: overall accuracy, then F1 per label', fontsize=9)
        af.save_figure(fig, os.path.join(out_dir, 'permutation_family.png'))
    return table


# --- comparison of experiments --------------------------------------------------------------------------------------
def compare(experiments: list, args):
    summary = pd.read_csv(os.path.join(args.results_dir, '_aggregate', 'summary_table.csv')) \
        if os.path.exists(os.path.join(args.results_dir, '_aggregate', 'summary_table.csv')) else None
    rows, n_features, roots = {}, {}, {}
    for experiment in experiments:
        result = analyze(experiment, args, plot=args.plot)
        rows[experiment] = result['family']['info_gain_bits']
        n_features[experiment] = len(result['table'])
        roots[experiment] = result['root_entropy']
    table = pd.DataFrame(rows).T.fillna(0.0)
    table = table[[f for f in FAMILY_ORDER if f in table.columns]]
    table.insert(0, 'n_features', pd.Series(n_features))
    table['total_bits'] = table.drop(columns='n_features').sum(axis=1)
    table['root_entropy'] = pd.Series(roots)
    out_dir = os.path.join(args.results_dir, '_report')
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'infogain_by_family.csv'), index_label='experiment')
    af.print_table('information gain per family and experiment (bits per tree)', table)

    if args.plot:
        families = [c for c in FAMILY_ORDER if c in table.columns]
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(table) + 1.6))
        labels = [f'{e}  ({n})' for e, n in zip(table.index, table['n_features'])]
        left = np.zeros(len(table))
        for family in families:
            share = table[family] / table['total_bits']
            ax.barh(labels, share, left=left, color=FAMILY_COLORS[family], label=family, height=0.7)
            for i, (l, s) in enumerate(zip(left, share)):
                if s > 0.06:
                    ax.text(l + s / 2, i, f'{s:.0%}', ha='center', va='center', fontsize=7)
            left += share.values
        dice = mean_dice(summary, table.index)
        for i, (e, d) in enumerate(dice.items()):
            ax.text(1.01, i, f'Dice {d:.3f}  |  {table.loc[e, "total_bits"]:.2f} bits', va='center', fontsize=8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel('share of the information gain (bits per tree, summed over the splits)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6, fontsize=7)
        ax.set_title('How the forest splits the same information between feature families\n'
                     '(experiment and number of features; right: mean Dice on the test set, total information gain)',
                     fontsize=9)
        af.save_figure(fig, os.path.join(out_dir, 'infogain_by_family.png'))


def mean_dice(summary, experiments) -> pd.Series:
    if summary is None or 'experiment' not in summary:
        return pd.Series(np.nan, index=experiments)
    report = os.path.join(SCRIPT_DIR, 'results', '_report', 'feature_count_table.csv')
    if os.path.exists(report):
        t = pd.read_csv(report).set_index('experiment')['dice_mean']
        return t.reindex(experiments)
    return pd.Series(np.nan, index=experiments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Information gain of the random forest per feature and family.')
    parser.add_argument('--experiment', help='name of the experiment in results/, e.g. extended_plus')
    parser.add_argument('--compare', nargs='+', help='several experiments: family shares side by side')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--permutation', action='store_true',
                        help='also grouped permutation importance per family on test voxels (needs test_sample.npz '
                             'from analyze_outliers.py)')
    parser.add_argument('--repeats', type=int, default=5, help='permutation repeats')
    parser.add_argument('--results_dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--data_atlas_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'atlas'))
    parser.add_argument('--data_train_dir', default=os.path.join(SCRIPT_DIR, '..', 'data', 'train'))
    parser.add_argument('--feature_cache', default=os.path.join(SCRIPT_DIR, 'feature_cache'))
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args)
    elif args.experiment:
        res = analyze(args.experiment, args, plot=args.plot)
        if args.permutation:
            permutation_family(res, args.experiment, args, plot=args.plot)
    else:
        parser.error('give --experiment or --compare')
