# Information gain of the random forest, and where the HD95 outliers come from

Two scripts, both run on finished experiments (`run_experiment.py`):

| Script | Question | Output |
|---|---|---|
| [`analyze_information_gain.py`](../analyze_information_gain.py) | How much information does each feature / feature family contribute to the trees, and at which depth? | `results/<experiment>/analysis/seed_0/infogain/`, `results/_report/infogain_by_family.*` |
| [`analyze_outliers.py`](../analyze_outliers.py) | Do the distance features really remove stray false positives far from the structure? | `results/_report/outliers_<without>_vs_<with>/` |

```
python analyze_information_gain.py --experiment extended_plus --plot
python analyze_information_gain.py --compare baseline baseline_t2 prior_only extended extended_plus_no_distance extended_plus extended_plus_reduced_r09 --plot
python analyze_outliers.py --plot                                  # extended_plus_no_distance vs extended_plus, ~5 min
python analyze_information_gain.py --experiment extended_plus --permutation --plot   # after analyze_outliers.py
```

Both scripts retrain the forest of the run from its saved training voxels (`train.npz`, same settings and seed) and check that it is the same forest: the Gini version of the information gain reproduces sklearn's `feature_importances_` exactly, and the re-predicted test segmentations reproduce Dice and HD95 of the run's `metrics.csv`.

---

## 1. How a decision tree uses information gain

A tree is grown from the root down. At every node it has a set of training voxels with their labels and looks for the one question "feature ≤ threshold?" that separates the labels best. "Best" is measured by how much the **impurity** of the labels drops:

- **Entropy** H = −Σ p_k log₂ p_k (in bits). 0 bits = all voxels have the same label; with 6 equally frequent labels it is log₂ 6 = 2.58 bits. Our balanced-ish training sample starts at ≈ 2.1 bits at the root.
- **Information gain** of a split = H(parent) − (n_left/n · H(left) + n_right/n · H(right)): how many bits of uncertainty about the label the question removes for the voxels at that node.
- The forest in this project uses the **Gini impurity** (1 − Σ p_k², sklearn's default) instead of the entropy. Both pick almost always the same splits; Gini is just cheaper to compute. Our script measures the entropy-based information gain of the splits the Gini forest chose.

At each node, a random forest only looks at a random subset of √n features (`max_features='sqrt'`: 5–6 of 35) and each tree sees a bootstrap sample of the voxels. That makes the trees different, and it means a weak feature still gets used whenever it happens to be the best of its random subset.

**From splits to feature importance.** Weight each split's gain by the fraction of the tree's voxels that reach the node, sum over all splits that use a feature, average over the trees: that is the feature's information gain (bits per tree). With the Gini impurity instead of the entropy and normalised to 1, it is exactly sklearn's `feature_importances_` (mean decrease in impurity, MDI), so "Gini importance" and "information gain" are the same idea with a different impurity measure.

All splits of a tree together can remove at most the entropy at the root. With depth 20 the leaves are almost pure, so every forest removes nearly the same ≈ 2 bits, **whatever the number of features**. Adding features cannot increase this total; it can only change *which* features the forest uses to get there. This is the core of the argument against "more features = better" (§3).

**Caveats.**
- The gain is measured on the **training** voxels while the tree is grown. Like the Gini importance it favours features with many distinct values (continuous intensities over the prior, which is 0 almost everywhere), and it says nothing about generalisation. The grouped permutation importance (§4) is measured on test voxels.
- Correlated features share the gain: if `T1_intensity` and `T1_smoothing_gaussian_s1` carry the same information, each is chosen in about half of the splits and each looks half as important. Summing over a **family** removes this split-up.

## 2. Figures for one experiment (`results/extended_plus/analysis/seed_0/infogain/`)

| File | What it shows | `extended_plus`, seed 0 |
|---|---|---|
| `tree_top.png` | The first 3 levels of one tree: question, voxels reaching the node, entropy, majority label, gain of that split. Good slide to explain how a tree works. | Root: `distance_brain_border ≤ −0.5` sends 31 % of the voxels to a pure background leaf (IG 0.73 bits). Then `T2_localstats_mean_r2`, then `priordistance_thalamus` (→ 97 % thalamus) and `priordistance_amygdala` (→ 55 % hippocampus). |
| `infogain_features.png` / `.csv` | Information gain per feature (bits per tree, ± std over the 50 trees), coloured by family | Top 5 are within 7–8 % each: T1−T2, T1 smoothing s1, brain-border distance, T1 gradient, T1 intensity. |
| `infogain_family.png` / `infogain_family.csv` | **Grouped by family** ("group feature importance"): left the family total, right per feature in the family | smoothing 14 %, priordistance 14 %, prior 13 %, gradient 12 % … log 4 %, spatial 2 %. Per feature, the single features (T1−T2, distance) and the 3 prior distances are worth most; the 6 LoG features together are worth less than one gradient. |
| `infogain_depth.png` / `.csv` | Information gained at each tree depth, stacked by family (left absolute, right share) | Depth 0–1: intensity, gradient, smoothing, distance, symmetry (tissue vs. background vs. WM/GM). Depth 2–6: prior and prior distance (where are the small structures?). Depth > 10: LoG, smoothing, coordinates, with almost no gain left (fine-tuning a few voxels). |
| `infogain_cumulative.png` | Features ranked by gain, cumulative share | 16 of 35 features give 80 %, 21 give 90 %. |
| `permutation_family.png` / `.csv` | **Grouped permutation importance on test voxels** (§4) | priordistance −0.21 accuracy, prior −0.08, distance and gradient −0.03, all other families ≤ 0.004. |

## 3. Comparison of the feature sets (`results/_report/infogain_by_family.png`)

One bar per experiment, split into the share of each family; on the right the mean test Dice and the total information the splits remove.

| Experiment (features) | Total IG (bits per tree) | Mean Dice |
|---|---|---|
| baseline (5) | 2.05 | 0.669 |
| baseline_t2 (7) | 2.05 | 0.666 |
| prior_only (12) | 2.04 | 0.722 |
| extended (29) | 2.04 | 0.729 |
| extended_plus_no_distance (31) | 2.03 | 0.730 |
| extended_plus (35) | 2.03 | 0.722 |
| extended_plus_reduced_r09 (25) | 2.03 | 0.722 |

Talking points:
- **The total is the same for 5 and for 35 features.** Every tree has to resolve the same ≈ 2.1 bits of label uncertainty at the root and gets ~97 % of it on the training voxels. Adding features does not add information to resolve; the new features take a share from the old ones.
- **What matters is whether the share moves to a feature that generalises.** From `baseline_t2` to `prior_only`, the prior takes 34 % and the spatial coordinates drop from 27 % to 7 %: both encode *location*, but the prior encodes it per structure and generalises much better (Dice +0.056). From `prior_only` to `extended_plus`, intensity-derived families (smoothing, local statistics, LoG, T1−T2) take ~35 % of the share, and Dice doesn't change: they re-describe the intensity the forest already had.
- Removing 10 correlated features (`reduced_r09`) just moves their share to the partners they correlate with (without `T1_intensity` and `T1_smoothing_gaussian_s1`, the share of T1−T2 rises from 8 % to 14 % and the LoG features from 4 % to 9 %), Dice unchanged.

## 4. Information gain vs. grouped permutation importance

Information gain is measured on the training voxels while the tree grows. **Grouped permutation importance** asks the trained forest on new data: shuffle all features of one family together (the same permutation for all its columns) on 30 000 test voxels (500 per label and subject, saved by `analyze_outliers.py`), and measure how much accuracy and per-label F1 drop.

The two disagree, and that's informative:

| Family | share of training IG | test accuracy drop when shuffled |
|---|---|---|
| smoothing (6) | 14 % | 0.001 |
| priordistance (3) | 14 % | **0.213** (hippocampus F1 −0.34, amygdala −0.28) |
| prior (5) | 13 % | 0.082 |
| gradient (2) | 12 % | 0.025 |
| intensity (2) | 9 % | 0.004 |
| localstats (4), log (6), combination (1), symmetry (2) | 4–8 % each | ≤ 0.002 |

The smoothing family is used in many splits, but when it is shuffled the forest falls back on the raw intensity and the local mean, which carry the same information: it is **redundant**, not useless. The prior distances carry information no other family has, so shuffling them hurts. This is the link to the correlation analysis: a family is only as important as the information *nobody else* has. (Caveat: shuffling one family at a time still lets correlated families in *other* groups stand in, so the drop is a lower bound on how much the forest uses the family.)

---

## 5. The distance features and the HD95 outliers (`results/_report/outliers_extended_plus_no_distance_vs_extended_plus/`)

The claim in the feature-count report was: *without the distance features, hippocampus HD95 is twice as high because of scattered false-positive voxels far from the hippocampus; the distance-to-hippocampus feature removes them, at a small Dice cost.* `analyze_outliers.py` tests this directly. It retrains both forests (`extended_plus_no_distance`, 31 features, and `extended_plus`, 35 features, seed 0), predicts the 10 test subjects again (reproduces the run's Dice to 2·10⁻⁵ and HD95 exactly) and measures each structure.

**Why stray voxels drive HD95.** HD95 is the maximum of two directed distances, each the 95th percentile over the *surface* of one segmentation:
- prediction → truth: for every point on the predicted surface, the distance to the true surface. Large when the prediction has **extra voxels far away** (false positives).
- truth → prediction: large when **parts of the true structure are missed** (false negatives).

A single stray voxel is all surface (6 faces), while a big blob has little surface per voxel. So 2–3 % of the predicted voxels scattered as small islands can make up more than 5 % of the predicted surface and set the 95th percentile.

**Results** (mean over the 10 test subjects, seed 0; without → with the 4 distance features):

| | Hippocampus | Amygdala | Thalamus |
|---|---|---|---|
| HD95 (mm) | 10.3 → **4.1** | 5.3 → 4.1 | 3.1 → 3.2 |
| HD95 prediction → truth | 10.3 → **4.1** | 5.3 → 4.1 | 3.0 → 2.9 |
| HD95 truth → prediction | 2.3 → 2.5 | 2.5 → 2.8 | 2.8 → 3.1 |
| false positives > 10 mm away (voxels) | 299 → **0.5** | 40 → 0.4 | 8 → 0 |
| connected components | 71 → 15 | 23 → 4 | 26 → 6 |
| voxels outside the 2 largest components | 462 → 27 | 64 → 4 | 31 → 4 |
| Dice | 0.648 → 0.638 | 0.636 → 0.609 | 0.808 → 0.804 |
| predicted voxels (true: 8 925 / 3 330 / 17 748) | 17 584 → 18 269 | 6 298 → 6 894 | 25 268 → 25 593 |

Evidence, one figure each:

1. **`hd95_direction.png`: only the prediction → truth direction changes.** The hippocampus HD95 drop is entirely in the direction that measures extra voxels (10.3 → 4.1 mm, in all 10 subjects); the direction that measures missed parts stays at ~2.4 mm. So the high HD95 comes from extra voxels, not from a worse boundary.
2. **`fp_distance.png`: the far false positives disappear.** False positives more than 10 mm from the true hippocampus: ~3 000 voxels over the 10 subjects without the distance features, 5 with them.
3. **`example_122620.png`: where they are.** All slices of one subject projected onto one image. Without the distance features there are hippocampus voxels in a blob above the hippocampus, near the midline and down in the brainstem (red); with them, everything predicted lies along the true hippocampus.
4. **`cleanup.png`: removing the stray components closes the gap.** Keep only the 2 largest connected components (left and right hippocampus) of the prediction *without* distance features: HD95 drops from 10.3 to 4.1 mm, the same as with the distance features (4.1). So the stray components are the whole HD95 difference. And Dice even *rises* to 0.660, better than both raw models.
5. **`feature_at_fp.png`: why the distance feature can remove them and the prior cannot.** At the far false positives, `prior_hippocampus` is 0, the same value as at every voxel outside the typical hippocampus region, 1 cm or 6 cm away. The forest therefore can't tell "just outside" from "far away" with the prior. `priordistance_hippocampus` keeps growing with the distance: inside the true hippocampus it is around 0 (−3 to +3 mm), at the far false positives 8–20 mm, so one split separates them.

**Where the Dice cost comes from.** With the distance features, the predicted structures get *larger* (hippocampus +685 voxels, amygdala +596): the far false positives disappear, but more voxels right next to the border are labelled hippocampus/amygdala. Both models already predict about twice the true volume (17 600 vs 8 900 hippocampus voxels): Dice is limited by over-segmentation at the border, and the distance features make that slightly worse. That's the trade: HD95 better (no far outliers), Dice slightly worse (thicker border).

**What to say in the presentation:**
- Dice and HD95 measure different errors. Dice counts voxels, so 460 stray voxels out of 17 600 barely change it. HD95 measures the worst 5 % of the surface, so the same voxels double it.
- The distance features don't make the classifier better at the boundary; they remove anatomically impossible predictions far away. A simple post-processing step (largest connected components) does the same here, without the feature and without the Dice cost.
- The over-segmentation (prediction ≈ 2× true volume) is a candidate for further work: the training sample is balanced over the labels, so the forest sees hippocampus much more often than it occurs in a brain.

**Limits.** One seed (0); the run means over 3 seeds give 8.9 vs 4.2 mm for hippocampus HD95, the same picture. "Far" uses a fixed 5 / 10 mm threshold to the nearest true voxel. Keeping the 2 largest components assumes both hemispheres' structures are each one connected piece, which holds for these three structures.
