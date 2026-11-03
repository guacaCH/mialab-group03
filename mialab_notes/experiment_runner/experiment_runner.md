# Experiment runner: how it works and what it writes

Two scripts for to-do tasks 2.3 and 2.4, to copy into the repo root (next to `pipeline.py`):

| File | Purpose |
|---|---|
| [run_experiment.py](run_experiment.py) | Runs one experiment (one config file) for one or more seeds, writes one result folder per seed |
| [aggregate_results.py](aggregate_results.py) | Reads all result folders, makes the comparison table, Wilcoxon tests and boxplots |
| [experiments/](experiments/) | Example configs: `baseline.json`, `no_gradient.json`, `baseline_t2.json` |
| [example_results/](example_results/) | Real output of a small test run (see below) |

Both scripts need the Phase 1 preprocessing fixes ([phase1_code.md](../phase1_code.md)) and, since 20.10, the Phase 2 feature pool ([phase2_code.md](../phase2_code.md)). They use only packages already in our `requirements.txt` (pandas, scipy, seaborn).

**Tested:** both scripts were run on a copy of the repo with the Phase 1 and Phase 2 code, 2 training subjects and 3 test subjects, configs `baseline`, `no_gradient` and `baseline_t2` (seed 0, 10 trees for the test), plus seeds 0 and 1 in an earlier test that also checked that re-running an existing seed is skipped. All files below were produced as described. The numbers in `example_results/` come from that tiny run (2 training subjects), so **don't interpret them**; they only show the format.

---

## 1. The idea

**One experiment = one JSON config file.** The file name is the experiment name. Each config only lists what differs from the defaults, so you can see at a glance what an experiment changes:

```json
{
  "description": "Baseline without the T1 gradient feature",
  "pre_process": {"features": "baseline", "exclude_features": ["family=gradient"]},
  "forest": {"n_estimators": 50, "max_depth": 20},
  "post_process": {"simple_post": true}
}
```

Config sections:

| Section | What it controls | Default |
|---|---|---|
| `description` | free text, shows up in `config.json` | `""` |
| `pre_process` | passed **unchanged** to `putil.pre_process()`: registration/skull stripping/normalization flags and the feature selection `features` / `exclude_features` (syntax in [phase2_code.md §2](../phase2_code.md#2-selecting-features)). Configs with the old booleans (`coordinates_feature` etc.) still work. | all preprocessing on; `"features": "baseline"` |
| `forest` | passed unchanged to scikit-learn's `RandomForestClassifier`. `random_state` is always set from the seed. | 50 trees, depth 20, `max_features="sqrt"`, no class weights, all CPU cores |
| `post_process` | passed to `putil.post_process()`. Empty = no post-processing. | `{}` |
| `save_segmentations` | also save the predicted segmentations: `true` writes `.mha`, `"slicer"` writes `.seg.nrrd` for 3D Slicer (see below) plus the ground truth | `false` |

Freeze the forest settings in `baseline.json` once task 1.7 is done, and copy the same `forest` block into every feature experiment.

**Viewing segmentations in 3D Slicer.** With `"save_segmentations": "slicer"`, every prediction is written as `<subject>_SEG.seg.nrrd` (and `_SEG-PP` with post-processing), and the ground truth once per experiment as `ground_truth/<subject>_GT.seg.nrrd`. Drag a file into Slicer and it opens as a segmentation with five named segments (WhiteMatter, GreyMatter, Hippocampus, Amygdala, Thalamus) in the same colours every time. The segmentations are in atlas space (after registration), so load the atlas T1 (`../data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz`) as the background image, not the subject's original T1. Each file is about 0.4 MB.

**One seed = one repetition.** The seed fixes (a) the random choice of training voxels and (b) the randomness inside the forest. Running seeds 0, 1, 2 tells you how much results change by chance alone. If a feature set beats the baseline by less than that, it's noise.

## 2. Running it

```
python run_experiment.py experiments/baseline.json --seeds 0 1 2
python run_experiment.py experiments/no_gradient.json --seeds 0 1 2
python aggregate_results.py --baseline baseline --plot
```

Options of `run_experiment.py`: `--results_dir` (default `results/`), `--data_atlas_dir`, `--data_train_dir`, `--data_test_dir` (default `../data/...`, one level above the repo, as in the provided pipeline), `--overwrite`, `--feature_cache` (default `feature_cache/`, `none` to disable; see [phase2_code.md §4](../phase2_code.md#4-the-feature-cache-22)).

Before anything slow starts, the runner checks the feature selection and prints it, e.g. `4 features: spatial_coord_x, spatial_coord_y, spatial_coord_z, T1_intensity`. A typo in a feature name stops it right there.

What happens inside, in order:
1. **Training, once per seed:** set the seed → preprocess the 20 training subjects and sample training voxels → train the forest.
2. **Testing, once per test subject:** preprocess the subject once (test features don't depend on the seed) → predict with every seed's forest → evaluate Dice and HD95 (and post-processing if configured). Subjects are handled one by one so memory stays low even with large feature sets later.
3. **Writing:** each seed's files go into `seed_<n>.tmp/` and the folder is renamed to `seed_<n>/` only when everything is written. A crashed run therefore never looks finished.

**Already finished seeds are skipped.** You can re-run the same command after a crash or add seeds later (`--seeds 0 1 2 3 4` only runs 3 and 4). Use `--overwrite` after changing the code.

## 3. The output, file by file

```
results/
├── baseline/
│   ├── seed_0/
│   │   ├── config.json
│   │   ├── metrics.csv
│   │   ├── summary.csv
│   │   ├── features.csv
│   │   ├── feature_importance.csv
│   │   ├── timing.json
│   │   └── segmentations/          (only with "save_segmentations": true or "slicer")
│   ├── seed_1/ ...
│   └── ground_truth/               (only with "save_segmentations": "slicer")
├── no_gradient/
│   └── ...
└── _aggregate/                     (written by aggregate_results.py)
    ├── all_metrics.csv
    ├── summary_table.csv
    ├── wilcoxon.csv
    ├── boxplot_DICE.png
    └── boxplot_HDRFDST95.png
```

### `config.json`: everything needed to reproduce the run

| Field | Meaning |
|---|---|
| `experiment`, `seed`, `config_file` | which experiment, which repetition, which config file |
| `config` | the **full** config after defaults were filled in, so it stays correct even if defaults change later |
| `feature_names` | the name of every feature column, in order, e.g. `spatial_coord_x`, `T1_intensity`, `T1_gradient_magnitude`. Needed later for importance, SHAP and Boruta. |
| `n_features` | number of features: the x-axis of the "performance vs. number of features" plot |
| `n_train_subjects`, `n_train_voxels`, `train_voxels_per_label` | how many training voxels of each structure were actually sampled, so you can check the sampling |
| `n_test_subjects`, `test_subjects` | which subjects were evaluated |
| `git` | commit hash, branch and whether there were uncommitted changes. Tells you which code produced the result. Commit before running experiments you want to keep. |
| `versions`, `machine`, `finished_at` | Python and package versions, computer name, time |

### `metrics.csv`: the raw result, one row per subject × structure × metric

```
experiment,seed,subject,label,metric,value,postprocessed
no_gradient,0,117122,WhiteMatter,DICE,0.806,False
no_gradient,0,117122,WhiteMatter,HDRFDST95,3.162,False
...
no_gradient,0,117122,WhiteMatter,DICE,0.806,True
```

This is the "tidy" or "long" format: every value is one row, every column one property. With 10 test subjects, 5 structures, 2 metrics, that's 100 rows (200 with post-processing, marked by `postprocessed=True`). The format is what makes aggregation easy: concatenating many runs gives one big table that pandas and seaborn can group and plot directly. It replaces the provided `results.csv` (which has one column per metric and `-PP` glued onto subject names).

`DICE` is 0..1 (higher = better), `HDRFDST95` is the 95th-percentile Hausdorff distance in mm (lower = better).

### `summary.csv`: quick look at this run

```
postprocessed,label,metric,mean,std,median,min,max,count
False,Amygdala,DICE,0.398,0.078,0.396,0.321,0.477,3
```

Mean, std, median, min, max over the test subjects for each structure and metric (`count` = number of subjects). Same information as the provided `results_summary.csv`, plus median/min/max, one row per combination.

### `features.csv`: name and tags of every feature column

```
name,feature,family,modality,scale
spatial_coord_x,spatial_coord,spatial,none,none
T1_intensity,T1_intensity,intensity,T1,none
T1_gradient_magnitude,T1_gradient_magnitude,gradient,T1,none
```

One row per column of the feature matrix, in matrix order. `feature` is the computation the column comes from (the three coordinates share `spatial_coord`). Merge it with `feature_importance.csv` (or SHAP values) on the name to sum importance per family, modality or scale.

### `feature_importance.csv`: which features the forest used

```
feature,importance
T1_intensity,0.578
spatial_coord_y,0.163
...
```

The forest's built-in (Gini / "mean decrease in impurity") importance, sorted, summing to 1. It's free to compute, so it's stored for every run. **Use it only as a first impression**: it is biased toward features with many distinct values and splits importance between correlated features. The real analysis uses SHAP, Boruta and permutation importance ([feature_methods.md](../feature_methods.md)).

### `timing.json`: cost of the run

```json
{"preprocess_train_s": 24.8, "fit_s": 0.2, "predict_s": 27.6, "preprocess_test_s": 13.1}
```

Seconds for: preprocessing and feature extraction of the training subjects, fitting the forest, predicting all test subjects, preprocessing the test subjects (shared by all seeds of one call). Bigger feature sets cost more time; this is the "cost" side of "more features".

### `segmentations/` (optional)

`<subject>_SEG.mha` (and `_SEG-PP.mha` with post-processing), viewable in ITK-SNAP. Each is several MB, so only enable it for runs you want to look at or show.

## 4. The aggregate output (`aggregate_results.py`)

```
python aggregate_results.py --baseline baseline --plot
python aggregate_results.py --baseline baseline --experiments baseline t2 full_pool --plot
```

### `all_metrics.csv`
All `metrics.csv` files stacked, plus `n_features`. The input for any custom plot.

### `summary_table.csv`: the main comparison table

One row per experiment × post-processing × structure × metric:

| Column | Meaning |
|---|---|
| `mean` | mean over test subjects (each subject first averaged over seeds, so every subject counts once) |
| `std_subjects` | spread between subjects: how differently the method works on different brains |
| `std_seeds` | spread of the mean between seeds: how much the result changes by chance |
| `n_subjects`, `n_seeds`, `n_features` | for context |

The console also prints it as a pivot, e.g.:

```
experiment                   baseline     no_gradient
metric    label
DICE      Amygdala      0.415 ± 0.066   0.410 ± 0.075
          GreyMatter    0.694 ± 0.012   0.702 ± 0.011
...
```

**How to read it:** a difference between two experiments is only interesting if it is clearly larger than `std_seeds`, and it is only *significant* if the Wilcoxon test says so.

### `wilcoxon.csv`: is the difference real?

For every experiment vs. the `--baseline`, per structure and metric:

| Column | Meaning |
|---|---|
| `median_diff`, `mean_diff` | experiment minus baseline, paired by subject |
| `p_value` | paired Wilcoxon signed-rank test (see feature_methods.md §4) |
| `better` | whether the median difference goes in the good direction (Dice up, HD95 down) |
| `significant_0.05` | p < 0.05 |

With 10 test subjects, the smallest possible p-value is about 0.002 (all 10 differences in the same direction). With the 3 test subjects of the example run it can't go below 0.25, which is why nothing there is significant. When reporting many tests (5 structures × 2 metrics × several experiments), mention that some will be "significant" by chance; a Bonferroni or Holm correction is the honest option.

### `boxplot_DICE.png`, `boxplot_HDRFDST95.png`
One box per structure and experiment, over the test subjects. The first slide-ready figure for the midterm.

## 5. Git rules for results

- **Commit:** `experiments/*.json` and the small files in `results/` (`config.json`, `metrics.csv`, `summary.csv`, `features.csv`, `feature_importance.csv`, `timing.json`). They're a few KB per run, and this way everyone has all results after a `git pull`.
- **Don't commit:** segmentations, the feature cache and the old pipeline output. Add to `.gitignore`:
  ```
  feature_cache/
  results/**/segmentations/
  results/**/*.tmp/
  results/_aggregate/
  mia-result/
  *.mha
  ```
  `_aggregate/` can be regenerated by anyone at any time, so it doesn't need to be committed.
- **Avoid conflicts:** two people should not run the same experiment name. Give experiments descriptive names (`t2_intensity`, `pyradiomics_glcm_r1`) and say in the group chat who runs what.
