# Feature analysis: which features matter, and which can be left out

[`analyze_features.py`](../analyze_features.py) (repo root) analyzes the features of an experiment that was already run with [`run_experiment.py`](../run_experiment.py) (see [experiment_runner.md](experiment_runner/experiment_runner.md)). It answers three questions:

| Question | Command | Method |
|---|---|---|
| Which features does the forest use? | `importance`, `shap` | forest (Gini) importance over all seeds; SHAP values per label |
| Is a feature better than chance? | `boruta` | Boruta: compares every feature with shuffled copies ("shadow features") |
| Which features are redundant? | `correlation` | correlation matrix, groups of strongly correlated features, a suggestion which to drop |

Every command prints its numbers in the terminal and writes them as `.csv`. With `--plot` it also saves simple plots (`.png`). `summary` puts everything computed so far into one table.

---

## 1. Running it

First run the experiment (once):

```
python run_experiment.py experiments/extended_plus.json --seeds 0 1 2
```

Then the analysis, everything at once:

```
python analyze_features.py all --experiment extended_plus --plot
```

or one step at a time:

```
python analyze_features.py importance  --experiment extended_plus --plot
python analyze_features.py shap        --experiment extended_plus --plot
python analyze_features.py boruta      --experiment extended_plus --plot
python analyze_features.py correlation --experiment extended_plus --plot --write_config
python analyze_features.py summary     --experiment extended_plus
```

Leave out `--plot` if you only want the numbers.

Options:

| Option | Used by | Default | Meaning |
|---|---|---|---|
| `--experiment` | all | (required) | the folder name in `results/`, i.e. the config file name |
| `--seed` | shap, boruta, correlation, summary | `0` | which run to analyze (`importance` always uses all finished seeds) |
| `--plot` | all except summary | off | also save `.png` plots |
| `--n_samples` | shap, boruta, correlation | shap 2000, boruta 10000, correlation all | number of training voxels, the same number per label |
| `--top` | shap | all | number of features shown in the beeswarm plots |
| `--max_iter` | boruta | `50` | Boruta iterations |
| `--method` | correlation | `spearman` | `spearman` (any monotonic relation) or `pearson` (linear only) |
| `--threshold` | correlation | `0.9` | from which \|r\| two features count as redundant |
| `--write_config` | correlation | off | write an experiment config without the redundant features |
| `--results_dir`, `--data_atlas_dir`, `--data_train_dir`, `--feature_cache` | | as in `run_experiment.py` | |

Run times on a laptop (`extended_plus`, 35 features): the first command that needs the training voxels takes ~3 min (see §2), then `importance` <1 s, `shap` ~2 min, `boruta` ~10 s, `correlation` ~3 s.

## 2. Where the data comes from

- `importance` only reads the `feature_importance.csv` files of the finished runs, so it computes nothing.
- `shap`, `boruta` and `correlation` need the **training voxels** of the run. The runner doesn't save them, so the first of these commands computes them again, the same way as `run_experiment.py`: same config, same seed, and the label prior built from the training subjects only. They are saved to `results/<experiment>/analysis/seed_<seed>/train.npz` (~20 MB), so later commands start immediately. If the features of the run change, the file is recomputed automatically.
- `shap` also trains the forest of the run again (same settings and seed, ~5 s). It checks that the importances are identical to the run's `feature_importance.csv`, and prints a note if they are not (e.g. after a code change).
- The samples are **balanced over the labels** (the same number of voxels per label). Without that, a random sample would contain hardly any amygdala voxels.

## 3. The output, file by file

```
results/<experiment>/analysis/
├── importance_gini.csv / .png
└── seed_<seed>/
    ├── train.npz
    ├── shap_importance.csv, shap_bar.png, shap_beeswarm_<label>.png
    ├── boruta.csv / .png
    ├── correlation_matrix.csv, correlation_pairs.csv, correlation_groups.csv, correlation_heatmap.png
    └── feature_summary.csv
```

| File | Content |
|---|---|
| `importance_gini.csv` | per feature: `mean`, `std` over seeds, `n_seeds`, `rank` |
| `importance_gini.png` | bar per feature with std as error bar, coloured by feature family |
| `shap_importance.csv` | per feature: mean \|SHAP\| for every label, `overall` (mean over labels), `rank` |
| `shap_bar.png` | mean \|SHAP\| stacked by label: shows **for which structure** a feature is used |
| `shap_beeswarm_<label>.png` | one dot per voxel: x = effect on the probability of this label, colour = feature value (red high, blue low) |
| `boruta.csv` / `.png` | per feature: `decision` (confirmed / tentative / rejected) and `rank` (1 = confirmed) |
| `correlation_matrix.csv` | the full feature × feature correlation matrix |
| `correlation_pairs.csv` | all pairs with \|r\| ≥ threshold, strongest first |
| `correlation_groups.csv` | features in a group of mutually correlated features, with `importance` and `action` (keep / drop) |
| `correlation_heatmap.png` | the matrix, rows and columns ordered by the clustering, so correlated features form blocks |
| `feature_summary.csv` | one row per feature: family, gini + rank, shap + rank, boruta + rank, correlation group + action |

## 4. How to read the results

**Gini vs. SHAP.** The forest's own importance (Gini) is computed on the training data and favours features with many different values. SHAP measures how much each feature actually changes the predicted probabilities. When the two disagree, trust SHAP. Example from `extended_plus`: `T2_symmetry_absdiff` is rank 7 by Gini but rank 15 by SHAP. The SHAP bar plot also shows *for which label* a feature matters, e.g. `priordistance_thalamus` is used almost only for the thalamus, and `prior_amygdala` mostly for the amygdala. A feature with a low overall rank can still be essential for one small structure.

**SHAP beeswarm (`shap_beeswarm_<label>.png`).** There is one plot per label, and each answers: *how does each feature push a voxel towards or away from this label?*

- **One row per feature**, sorted from top (most important for this label) to bottom. The order differs between labels.
- **One dot per voxel** of the sample (2000 by default) in every row. Where many voxels have similar values, the dots pile up into a thick band.
- **x position = SHAP value**: how much this feature changed the predicted probability of this label for this voxel, compared with the average voxel. +0.2 means "this feature raised the probability of hippocampus by 0.2 (20 percentage points)", −0.1 means it lowered it by 0.1. The grey line at 0 means "no effect".
- **Colour = the feature's own value** for that voxel: red = high, blue = low, purple = in between. It is relative to that feature's range, so "high" for `T1_intensity` and for `prior_WM` are different numbers.

How to read the colour and position together:

| What you see | Meaning |
|---|---|
| red dots on the **right**, blue on the left | high values → **more** likely this label (positive relation) |
| red dots on the **left**, blue on the right | high values → **less** likely this label (negative relation) |
| **blue** dots on the right | low values → more likely this label |
| red and blue cleanly separated | the feature acts like a clear threshold: a good, easy-to-use feature |
| colours mixed on both sides | the effect depends on other features (interactions); the value alone doesn't tell the direction |
| all dots in a narrow band around 0 | the feature hardly matters for this label |
| a long tail on one side only | the feature matters strongly for a few voxels and not at all for the rest |

Examples from `shap_beeswarm_Hippocampus.png` (`extended_plus`):
- `prior_hippocampus`: **red on the right** (up to +0.2). Voxels where many training subjects had hippocampus are likely hippocampus. This is the expected direction.
- `priordistance_hippocampus`: **blue on the right**, with a long tail up to +0.27. The distance is negative inside the typical hippocampus region, so a *low* value means "inside the region", and that pushes towards hippocampus. Blue on the right is correct here; it depends on the sign convention of the feature.
- `prior_amygdala`: **many red dots on the left** (down to −0.15). Where the amygdala usually is, the voxel is *not* hippocampus. The feature helps by ruling out the neighbouring structure.
- `prior_GM`: red on the left. Typical cortex positions are not hippocampus.
- The `log_laplacian` rows at the bottom: all dots near 0, so these features are not used for the hippocampus.

So "many red dots on the left" means: **when this feature is high, the forest is less likely to predict this label**. That is not bad; a feature that excludes a label is as useful as one that confirms it. What makes a feature useful is how far the dots spread from 0, not to which side.

**Correlation heatmap (`correlation_heatmap.png`).** It shows the correlation of every feature with every other feature: the same feature names on both axes, one coloured square per pair.

- **Colour = correlation r**: dark red = +1 (they rise together), dark blue = −1 (one rises when the other falls), white = 0 (unrelated). The diagonal is always dark red (each feature with itself).
- **Rows and columns are reordered** by the clustering, so features that correlate sit next to each other and form **coloured blocks** along the diagonal. The matrix is symmetric: the upper and lower triangles show the same values.
- **The tree (dendrogram)** on the top and the left shows the clustering. Two features or groups are joined by a horizontal bar, at a height of 1 − |r| between their *least* similar members. A join close to the heatmap means strongly correlated; a join far away means unrelated. The groups in `correlation_groups.csv` are the branches that join below 1 − threshold (0.1 for `--threshold 0.9`).

How to read it:

| What you see | Meaning |
|---|---|
| dark red block on the diagonal | a group of features that carry nearly the same information: candidates to drop all but one |
| dark **blue** squares | strong negative correlation: just as redundant as dark red (the sign doesn't matter for the forest) |
| a row/column that is white everywhere | a feature unrelated to all others: unique information, never dropped (whether it is *useful* is shown by SHAP) |
| a light block (pale red/blue) | related, but each feature still adds something of its own |

Examples from `extended_plus`:
- The big block in the upper left: `T1_intensity`, `T1_smoothing_gaussian_s1/s2`, `T1_localstats_mean_r2` and `combination_T1_minus_T2` are dark red with each other. Smoothing an image a little and averaging over 5×5×5 voxels give nearly the original image, so they are redundant.
- In the same area, the T1 features and the T2 features are **dark blue** with each other: after normalization, tissue that is bright in T1 is dark in T2. T2 therefore adds little that T1 doesn't already have *at the same voxel*.
- The block of `T1/T2_gradient_magnitude`, `T1/T2_localstats_std_r2` and `T1/T2_symmetry_absdiff`: all of them are high at tissue borders. `prior_GM` sits next to it in a lighter red (r ≈ 0.6): the cortex is full of borders, but the prior still adds information of its own.
- `spatial_coord_x` is white everywhere: left/right position is unrelated to everything else (the brain is nearly symmetric), and SHAP shows it is also not useful.
- The `log_laplacian` features at `s1` and `s2` form their own small blocks, with blue between T1 and T2 of the same scale.

Use the heatmap together with the SHAP plots. In a dark block, the feature with the highest SHAP value is the one to keep, which is what `correlation_groups.csv` does automatically.

**Boruta.** A feature is *confirmed* when it beats the best shuffled feature significantly often. With many voxels almost every feature with any signal is confirmed (in `extended_plus` all 35 were). Boruta then tells you that nothing is pure noise, but not which features are dispensable. To make it stricter, use fewer voxels (`--n_samples 2000`), or add the random control column to the experiment (`"features": ["extended_plus", "control_random"]`) and check which features rank below it.

**Correlation: which features can be left out.**
1. All features are clustered with distance `1 - |r|` (complete linkage). Every pair of features in a group has \|r\| ≥ threshold, so each group carries roughly the same information.
2. In each group the most important feature is kept. Importance is taken from SHAP if `shap_importance.csv` exists, otherwise from Gini (so run `shap` first; `all` does this). The others are marked `drop`.
3. Features that are not in any group don't correlate strongly with any other feature and are always kept.

Spearman is the default because many features are related monotonically but not linearly (e.g. an image and its smoothed version). A negative correlation is just as redundant as a positive one (e.g. `T1_localstats_mean_r2` and `T2_localstats_mean_r2`, r = −0.92).

**Correlated ≠ useless.** Two features with r = 0.9 still differ in 10–20 % of their variance, and the difference can be exactly what separates two small structures. Example: `priordistance_thalamus` (SHAP rank 3) is dropped at 0.9 because it correlates with `distance_brain_border` (r = −0.90), although it is the main thalamus feature. So the list is a **suggestion to test, not a result**. `--write_config` writes the config to test it, and the command prints the two lines that run and compare it:

```
python analyze_features.py correlation --experiment extended_plus --threshold 0.9 --write_config
python run_experiment.py experiments/extended_plus_reduced_r09.json --seeds 0 1 2
python aggregate_results.py --baseline extended_plus --experiments extended_plus extended_plus_reduced_r09 --plot
```

If Dice and HD95 don't get worse (Wilcoxon test in `results/_aggregate/wilcoxon.csv`, per label!), the dropped features were really redundant. Try a stricter threshold (e.g. `--threshold 0.95`, fewer features dropped) if a label gets worse. Look at the small structures (amygdala, hippocampus) first: they lose most when a feature is dropped.

## 5. Notes

- All analyses use the **training** voxels. That is fine for comparing features with each other, and it keeps the test set out of feature selection: choosing features on test data and evaluating on the same data would be optimistic.
- SHAP and the correlation depend on the seed only through the sampled voxels; the differences between seeds are small. Use `--seed 1` to check.
- `results/` is not committed to git; the configs written by `--write_config` (in `experiments/`) are.
