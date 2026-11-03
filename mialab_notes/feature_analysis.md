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

**Beeswarm.** In `shap_beeswarm_Hippocampus.png`, for instance, red dots on the right mean "a high value of this feature makes the voxel more likely to be hippocampus". A clean red/blue split means the feature works like a clear threshold.

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
