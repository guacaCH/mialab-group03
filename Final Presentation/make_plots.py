"""Slide plots for the final presentation, made from the finished runs in results/ (nothing is retrained).

Usage (from the repository root):
    .venv/bin/python "Final Presentation/make_plots.py"

Plots 07 and the B_* backups are copies of existing figures from results/ (see talking_points.md).

Writes to "Final Presentation/plots/". Every plot: large fonts, axis labels with units, legend outside the axes.

Importance plots (02, 04) use brain voxels only: run brain_only_importance.py first. It drops the ~31 % of the
training voxels that lie outside the brain mask and retrains the forest (see Marc's analysis). The versions made on
all training voxels are in archive/plots_all_training_rows/.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(HERE, 'plots')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 13, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 100, 'savefig.dpi': 200})

STRUCTURES = {'WhiteMatter': ('White matter', '#8a6d1f'), 'GreyMatter': ('Grey matter', '#5b5fa8'),
              'Thalamus': ('Thalamus', '#2f7a3e'), 'Hippocampus': ('Hippocampus', '#c0562a'),
              'Amygdala': ('Amygdala', '#b0336c')}
# one fixed colour per feature family, used in every plot
FAMILIES = {'intensity': ('intensity', '#ff7f0e'), 'gradient': ('gradient', '#2ca02c'),
            'smoothing': ('smoothing', '#d62728'), 'localstats': ('local statistics', '#8c564b'),
            'log': ('Laplacian of Gaussian', '#9467bd'), 'combination': ('T1 − T2', '#e377c2'),
            'symmetry': ('hemisphere symmetry', '#aec7e8'), 'spatial': ('coordinates', '#1f77b4'),
            'distance': ('distance to brain border', '#bcbd22'), 'prior': ('label prior (atlas)', '#7f7f7f'),
            'priordistance': ('distance to typical region', '#17becf')}
# the main ladder: every step adds features
LADDER = {'baseline': 'baseline', 'baseline_t2': '+ T2', 'prior_only': '+ prior',
          'extended': 'extended', 'extended_plus': 'extended+'}
WITHOUT, WITH = '#c0504d', '#4f81bd'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('written', name)


def n_features(exp):
    return len(pd.read_csv(os.path.join(RES, exp, 'seed_0', 'features.csv')))


def subject_means():
    """Metric per experiment, subject, label, metric: mean over the 3 seeds (no post-processing)."""
    m = pd.read_csv(os.path.join(RES, '_aggregate', 'all_metrics.csv'))
    m = m[~m['postprocessed']]
    return m.groupby(['experiment', 'subject', 'label', 'metric'])['value'].mean()


# --- 1. Dice and HD95 along the feature ladder ----------------------------------------------------------------------
def plot_ladder(sm):
    exps = list(LADDER)
    xt = [f'{LADDER[e]}\n{n_features(e)} feat.' for e in exps]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric, ylabel in zip(axes, ['DICE', 'HDRFDST95'], ['Dice (higher is better)', 'HD95 in mm (lower is better)']):
        for lab, (name, col) in STRUCTURES.items():
            y = [sm.loc[e, :, lab, metric].mean() for e in exps]
            ax.plot(range(len(exps)), y, 'o-', color=col, lw=2.5, ms=7, label=name)
        ax.set_xticks(range(len(exps)), xt)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.axvspan(1.5, 2.5, color='#2f7a3e', alpha=0.08)
    axes[0].set_title('Dice per structure')
    axes[1].set_title('HD95 per structure')
    axes[1].annotate('biggest gain:\nlabel prior', xy=(2, 11), ha='center', fontsize=12, color='#2f7a3e')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle('More features: big jump with the prior (12 features), then a plateau', y=1.02, fontsize=17)
    save(fig, '01_dice_hd95_feature_ladder.png')


# --- 2. cumulative information gain ---------------------------------------------------------------------------------
def plot_cumulative(exp='extended_plus'):
    d = pd.read_csv(os.path.join(RES, exp, 'analysis', 'seed_0', 'brain_only', 'infogain_features.csv'))
    d = d.sort_values('rank')
    x = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, d['share'] * 100, color=[FAMILIES[f][1] for f in d['family']], edgecolor='none')
    ax.set_xlabel('features, ranked by information gain (1 = most)')
    ax.set_ylabel('share of the information gain per feature (%)')
    ax.set_xlim(0.3, len(d) + 0.7)
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(x, d['cumulative_share'] * 100, 'k.-', lw=2, ms=8)
    ax2.set_ylabel('cumulative share (%)')
    ax2.set_ylim(0, 105)
    for level in (80, 90):
        k = int((d['cumulative_share'] * 100 < level).sum() + 1)
        ax2.axhline(level, color='grey', ls='--', lw=1)
        ax2.axvline(k, ymax=level / 105, color='grey', ls='--', lw=1)
        ax2.text(k + 0.4, level - 6, f'{level} % with {k} of {len(d)} features', fontsize=13)
    present = [f for f in FAMILIES if f in set(d['family'])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f][1]) for f in present]
    ax.legend(handles, [FAMILIES[f][0] for f in present], title='feature family', loc='upper left',
              bbox_to_anchor=(1.08, 1.0), frameon=False)
    ax.set_title(f'Information gain of the random forest, {exp} ({len(d)} features), brain voxels only')
    save(fig, '02_infogain_cumulative_extended_plus.png')


# --- 3. compute cost ------------------------------------------------------------------------------------------------
def plot_cost(sm):
    exps = ['baseline', 'baseline_t2', 'prior_only', 'extended_plus_reduced_r09', 'extended', 'extended_plus']
    names = {**LADDER, 'extended_plus_reduced_r09': 'reduced'}
    rows = []
    for e in exps:
        t = [json.load(open(os.path.join(RES, e, f'seed_{s}', 'timing.json'))) for s in range(3)]
        rows.append({'exp': e, 'n': n_features(e), 'extract': np.mean([x['preprocess_test_s'] for x in t]),
                     'fit': np.mean([x['fit_s'] for x in t]),
                     'dice': sm.loc[e, :, :, 'DICE'].groupby('label').mean().mean()})
    r = pd.DataFrame(rows).sort_values('n')
    xt = [f'{names[e]} ({n})' for e, n in zip(r['exp'], r['n'])]
    x = np.arange(len(r))
    # memory: each feature is one float32 value per brain voxel (~1.77 million voxels per subject)
    mem = r['n'] * 1.77e6 * 4 / 1e6
    panels = [(r['extract'], 'time (s)', 'Feature extraction\n(10 test subjects)', '{:.0f}', '#4f81bd'),
              (mem, 'size (MB)', 'Feature matrix\nper test subject', '{:.0f}', '#4f81bd'),
              (r['dice'], 'mean Dice over 5 structures', 'Segmentation quality\n(mean Dice)', '{:.3f}', '#2f7a3e')]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (vals, ylabel, title, fmt, col) in zip(axes, panels):
        ax.bar(x, vals, color=col, width=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x, xt, rotation=35, ha='right', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, fmt.format(v), ha='center', va='bottom', fontsize=12)
    axes[2].set_ylim(0.6, 0.75)
    fig.supxlabel('feature set (number of features)', y=-0.08, fontsize=15)
    fig.suptitle('Cost grows with every feature, the mean Dice stops growing after 12 features', y=1.03,
                 fontsize=17)
    fig.text(0.5, -0.15, 'times: mean over 3 seeds on one laptop; training the forest itself stays small '
             f'({r["fit"].min():.1f} s → {r["fit"].max():.1f} s); size = brain voxels × features × 4 bytes',
             ha='center', fontsize=11, color='grey')
    save(fig, '03_compute_cost.png')


# --- 4. SHAP by family and structure --------------------------------------------------------------------------------
def plot_shap_family(exp='extended_plus'):
    s = pd.read_csv(os.path.join(RES, exp, 'analysis', 'seed_0', 'brain_only', 'shap_importance.csv')).set_index('feature')
    fam = pd.read_csv(os.path.join(RES, exp, 'analysis', 'seed_0', 'feature_summary.csv')).set_index('feature')['family']
    g = s[list(STRUCTURES)].groupby(fam).sum()
    share = g / g.sum() * 100                                    # % of the total |SHAP| of each structure
    order = ['intensity', 'smoothing', 'combination', 'gradient', 'localstats', 'log', 'symmetry', 'distance',
             'spatial', 'prior', 'priordistance']
    share = share.loc[order]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    left = np.zeros(len(STRUCTURES))
    y = np.arange(len(STRUCTURES))[::-1]
    for f in order:
        v = share.loc[f].values
        ax.barh(y, v, left=left, color=FAMILIES[f][1], label=FAMILIES[f][0], height=0.7)
        for yi, l, w in zip(y, left, v):
            if w >= 6:
                ax.text(l + w / 2, yi, f'{w:.0f}%', ha='center', va='center', fontsize=12)
        left += v
    ax.set_yticks(y, [STRUCTURES[k][0] for k in STRUCTURES])
    ax.set_xlabel('share of mean |SHAP value| of each structure (%), summed per feature family')
    ax.set_xlim(0, 100)
    # bracket the two groups: image appearance vs. location
    ax.legend(title='feature family', loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title(f'SHAP: which kind of feature each structure relies on ({exp}, brain voxels only)')
    save(fig, '04_shap_family_per_structure.png')
    return share


# --- 5. dropping the correlated features ----------------------------------------------------------------------------
def plot_correlation_drop(sm):
    a, b = 'extended_plus', 'extended_plus_reduced_r09'
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, metric, ylabel in zip(axes, ['DICE', 'HDRFDST95'], ['Dice (higher is better)', 'HD95 in mm (lower is better)']):
        x = np.arange(len(STRUCTURES))
        for i, (e, col, name) in enumerate([(a, '#4f81bd', 'all 35 features'),
                                            (b, '#f0a030', '25 features (10 correlated ones dropped)')]):
            means = [sm.loc[e, :, lab, metric].mean() for lab in STRUCTURES]
            stds = [sm.loc[e, :, lab, metric].std() for lab in STRUCTURES]
            ax.bar(x + (i - 0.5) * 0.38, means, 0.38, yerr=stds, capsize=3, color=col, label=name)
        for xi, lab in zip(x, STRUCTURES):
            va, vb = sm.loc[a, :, lab, metric], sm.loc[b, :, lab, metric]
            p = wilcoxon(va.values, vb.loc[va.index].values).pvalue
            top = max(va.mean() + va.std(), vb.mean() + vb.std())
            ax.text(xi, top * 1.02, f'p={p:.3f}' if p >= 0.001 else 'p<0.001', ha='center', fontsize=11,
                    color='black' if p < 0.05 else 'grey')
        ax.set_xticks(x, [STRUCTURES[k][0].replace(' ', '\n') for k in STRUCTURES])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0.5, 0.92)
    axes[0].set_title('Dice: every change ≤ 0.01')
    axes[1].set_title('HD95: amygdala worse (4.0 → 5.7 mm)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.suptitle('Dropping features with |Spearman r| ≥ 0.9 (incl. priordistance_amygdala and _thalamus)', y=1.02,
                 fontsize=17)
    fig.text(0.5, -0.13, 'mean ± std over the 10 test subjects (each averaged over 3 seeds); '
             'paired Wilcoxon test, black = p < 0.05', ha='center', fontsize=11, color='grey')
    fig.text(0.5, -0.24, 'Correlations were computed on all training voxels, incl. the 31 % outside the brain.\n'
             'Inside the brain both dropped prior distances correlate less than 0.9 (r = 0.83 and −0.72) '
             'and would have been kept.', ha='center', fontsize=12, color='#b0336c')
    save(fig, '05_correlation_drop.png')


# --- 6. far false positives without the distance features -----------------------------------------------------------
def plot_fp_distance():
    d = os.path.join(RES, '_report', 'outliers_extended_plus_no_distance_vs_extended_plus')
    s = pd.read_csv(os.path.join(d, 'summary.csv')).set_index(['label', 'model'])
    labels = ['Hippocampus', 'Amygdala', 'Thalamus']
    cats = [('fp', 'all false\npositives'), ('fp_far5', '> 5 mm from\nthe structure'),
            ('fp_far10', '> 10 mm from\nthe structure')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, lab in zip(axes, labels):
        x = np.arange(len(cats))
        for i, (model, col, name) in enumerate([('without', WITHOUT, 'without distance features (31 features)'),
                                                ('with', WITH, 'with distance features (35 features)')]):
            v = np.array([s.loc[(lab, model), c] * 10 for c, _ in cats])   # summary = mean per subject -> 10 subjects
            bars = ax.bar(x + (i - 0.5) * 0.38, np.maximum(v, 0.5), 0.38, color=col, label=name)
            for bar, val in zip(bars, v):
                txt = f'{val / 1000:.0f}k' if val >= 10000 else f'{val:.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0.5) * 1.15, txt, ha='center', fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(0.5, 3e5)
        ax.set_xticks(x, [c[1] for c in cats], fontsize=12)
        hd = (s.loc[(lab, 'without'), 'hd95'], s.loc[(lab, 'with'), 'hd95'])
        ax.set_title(f'{lab}\nHD95 {hd[0]:.1f} → {hd[1]:.1f} mm')
    axes[0].set_ylabel('false-positive voxels,\nsum over 10 test subjects (log scale)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle('Without the distance features, stray voxels far from the structure survive and drive HD95',
                 y=1.04, fontsize=17)
    save(fig, '06_fp_distance.png')


# --- 7. flow chart of the talk --------------------------------------------------------------------------------------
def plot_flowchart():
    from matplotlib.patches import FancyBboxPatch
    steps = [
        ('Intro\n \n ', 'Speaker 1 · 1:00', 'Task and pipeline\nHypothesis 6 →\nrefined hypothesis\nFeature pool:\n5 → 35 features', None, '#e8e8e8'),
        ('Part 1\nAmount vs.\ninformation', 'Speaker 1 · 2:45',
         'Big gain only with the\nlabel prior, then a\nplateau at Dice 0.72\nTotal information is fixed:\n80 % in 15 of 35 features\nCost grows with\nevery feature',
         'plots 01, 02, 03', '#dbe7f5'),
        ('Part 2\nWhich features,\nfor which structure', 'Speaker 2 · 2:45',
         'SHAP per structure\nWM/GM: intensity-type\nfeatures\nSmall structures: atlas\nprior + distance to region\nBoruta: all 35 features\nbeat random noise',
         'plot 04 (+ Boruta backup)', '#e3f0dc'),
        ('Part 3\nRedundant vs.\nunique information', 'Speaker 3 · 2:45',
         'Drop 10 correlated features:\nsame Dice, half the cost …\n… but amygdala HD95\n4.0 → 5.7 mm (correlation\nfrom outside-brain voxels)\nDistance features remove\nstray voxels far away\n→ HD95 halves',
         'plots 05, 06, 07', '#f7e6d5'),
        ('Conclusion\n \n ', 'Speaker 3 · 0:45', 'More features help only\nwhile they add new\ninformation; a small,\nnon-redundant set\nis enough\nLimits, next steps', None, '#e8e8e8'),
    ]
    fig, ax = plt.subplots(figsize=(20, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis('off')
    w, gap = 17.4, 2.6
    for i, (title, who, body, plots, col) in enumerate(steps):
        x0 = 0.5 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 4), w, 35, boxstyle='round,pad=0.4,rounding_size=1.2', fc=col,
                                    ec='#555555', lw=1.5))
        ax.text(x0 + w / 2, 37.5, title, ha='center', va='top', fontsize=14, weight='bold')
        ax.text(x0 + w / 2, 29.5, who, ha='center', va='top', fontsize=12, color='#444444')
        ax.text(x0 + w / 2, 26.5, body, ha='center', va='top', fontsize=12.5, linespacing=1.4)
        if plots:
            ax.text(x0 + w / 2, 6, plots, ha='center', va='bottom', fontsize=11.5, style='italic', color='#333366')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x0 + w + gap - 0.2, 20), xytext=(x0 + w + 0.6, 20),
                        arrowprops=dict(arrowstyle='-|>', lw=2.5, color='#555555'))
    ax.text(50, 43.5, 'Final presentation: "A larger amount of features improves the segmentation performance" '
            '(10 min)', ha='center', va='top', fontsize=16, weight='bold')
    ax.text(50, 0.5, 'Backup slides for questions: feature overview · information gain vs. correlation · '
            'Boruta · correlation heatmap · no-prior run', ha='center', fontsize=12, color='#555555')
    save(fig, '00_flow_chart.png')


if __name__ == '__main__':
    sm = subject_means()
    plot_ladder(sm)
    plot_cumulative()
    plot_cost(sm)
    print(plot_shap_family().round(1))
    plot_correlation_drop(sm)
    plot_fp_distance()
    plot_flowchart()
