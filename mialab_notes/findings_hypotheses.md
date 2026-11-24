# Findings so far and what they say about H1-H4

State of 24.11.2026. All numbers come from `cheap_experiments.py` (reference: the 35 features of `extended_plus`,
"all35") and `flip_maps.py`, on the 10 test subjects. The sub-hypotheses H1-H4 are the ones in
[groupwork_requirements.md](groupwork_requirements.md#refined-hypothesis-draft).

## How to read the numbers

- **Seeds:** check, groups, noise, ablation and the flip maps use **5 seeds** (0-4). Subsets use 3 seeds × 5 random
  draws per k. Seed s = the training voxels of seed s and the forest `random_state` s.
- **Interval:** 95 % two-level bootstrap. It resamples the test subjects *and* the seeds, so it contains the
  patient-to-patient variation and the training randomness (`bootstrap_ci` in `cheap_experiments.py`).
- **Threshold:** a Dice difference below 0.01 is treated as too small to matter. That gives five verdicts:

| verdict | rule (95 % CI) | meaning |
|---|---|---|
| clearly worse / better | the whole CI is beyond ±0.01 | real and large enough to matter |
| real, about the threshold | the CI excludes 0 but crosses ±0.01 | surely an effect, about 0.01 in size |
| negligible | the whole CI is inside ±0.01 | too small to matter (an equivalence statement) |
| unclear | the CI includes 0 and crosses ±0.01 | can't tell |

## What changed from 3 to 5 seeds

- **The intervals didn't get narrower** (median width ratio 5/3 = 0.99). Their width comes mostly from the
  differences between patients, which more seeds can't reduce. What 5 seeds bought is a *reliable* estimate of the
  seed part: with 3 seeds there were only 10 different seed resamples, now there are 126.
- **No clear result changed.** Every "clearly worse/better" and every groups result stayed the same, with all 5
  seeds agreeing.
- **Three "unclear" ablation results became "negligible":** Gaussian smoothing → Amygdala (−0.007), local
  statistics → Hippocampus (−0.007), T2 → Hippocampus (−0.007). Only **3 of 12 families** now matter anywhere:
  label prior, prior distances, T2.
- **The other borderline results became "real, about the threshold":** all 5 seeds agree in sign and the CI
  excludes 0. They are prior → Amygdala/Hippocampus (−0.009), T2 → Amygdala (−0.009), prior distances →
  Hippocampus (+0.014), and +100 noise → Amygdala (+0.007).

## Main results (5 seeds, Dice)

**Groups** (absolute Dice, plot `results/cheap/groups/groups_dice.png`)

| | WM | GM | Thal | Hippo | Amyg |
|---|---|---|---|---|---|
| T1 intensity only (1) | 0.79 | 0.67 | 0.02 | 0.01 | 0.00 |
| image only (23) | 0.81 | 0.70 | 0.36 | 0.28 | 0.34 |
| location only (12) | 0.68 | 0.59 | 0.75 | 0.55 | 0.57 |
| image + location (35) | 0.84 | 0.73 | 0.79 | 0.63 | 0.62 |

**Subsets** (random image subsets of size k, 3 seeds × 5 draws, `results/cheap/subsets/dice_vs_k.png`)

| | rise from k = 3 to 23, image only | rise from k = 3 to 23, + all location |
|---|---|---|
| WM | +0.037 | +0.007 |
| GM | +0.046 | +0.026 |
| Thalamus | +0.255 | +0.015 |
| Hippocampus | +0.210 | +0.031 |
| Amygdala | +0.290 | +0.021 |

**Ablation** (remove one family, retrain; `results/cheap/ablation/effects_dice.png`)

- Label prior: WM **−0.017 clearly worse**; Hippocampus and Amygdala −0.009 (real, about the threshold).
- Prior distances: Amygdala **+0.019 clearly better**; Hippocampus +0.014 (real, about the threshold). Removing
  them *improves* the small structures.
- T2: Amygdala −0.009 (real, about the threshold).
- The other 9 families are negligible in every structure.

**Noise** (add pure-noise columns to all35; `results/cheap/noise/effects_dice.png`)

- +35 noise columns (2× the features): negligible in every structure.
- +100 noise columns (about 4×): GM **−0.015 clearly worse**; Amygdala +0.007 (real, about the threshold).

**Flip maps** (`results/flipmaps/drop/`, 5 seeds)

- Without the image features, about 300k voxels per subject become wrong, spread along the cortical GM/WM borders.
- Without the location features, about 114k become wrong, concentrated in the deep structures and the
  brainstem / ventral diencephalon.
- Without the label prior, the same brainstem block appears. This tissue is unlabelled background *inside* the
  brain mask, and without the prior the forest calls it WM or GM. That is where the WM −0.017 comes from.

**Over-segmentation** (`results/flipmaps/drop/predictions`, 3 seeds)

- The small structures are predicted about twice as large as they are. With all35, per subject:
  - Hippocampus: 8,670 correct voxels, 9,768 false positives, 254 missed.
  - Amygdala: 3,118 correct voxels, 3,678 false positives, 212 missed.
- The likely cause is the training sampling, which oversamples them about 11× (see
  [feature_methods.md](feature_methods.md#training-voxel-sampling)).
- Removing the prior distances cuts about 800 Hippocampus and 400 Amygdala false positives, which explains why it
  helps.

## What this says about H1-H4

**H1 (amount vs. information): supported.**
- The feature *count* is not what matters:
  - 12 location features beat 23 image features by 0.2-0.4 Dice on the small structures (groups).
  - Doubling the count with noise changes nothing; quadrupling it hurts GM (noise).
  - 9 of 12 families can be removed without a relevant loss (ablation).
- More features help only while they add missing *information*. Without location, Dice rises steeply with the
  number of image features (+0.21 to +0.29 for the small structures). Once location is there, the rise is ≤ 0.03
  (subsets).
- More features can even hurt: removing the 3 prior-distance features improves the Amygdala by 0.019.
- Not tested yet: Boruta subset vs. the full pool with 5 seeds (`analyze_features.py` has Boruta on one seed).

**H2 (structure dependence): supported, the clearest result.**
- WM and GM are separated by image information: image only reaches 0.81 / 0.70, location only 0.68 / 0.59.
- Thalamus, Hippocampus and Amygdala depend on location: T1 only gives about 0, image only 0.28-0.36,
  location only 0.55-0.75.
- The anatomical reason: these structures have almost the same T1/T2 contrast as the surrounding grey matter, so
  only *where* a voxel is tells them apart.
- The flip maps show the same picture in image space.

**H3 (what the information depends on): partly supported, the weakest part so far.**
- Modality: T2 matters only for the Amygdala (−0.009). Everywhere else T1 and T2 are redundant.
- Scale (smoothing, LoG, local statistics): each family alone is negligible. The multi-scale image features are
  interchangeable.
- Missing: the image perturbation experiment (noise, blur, bias field on the test images,
  `extract_features.py --noise`). That is the direct test of H3.

**H4 (where features act): qualitatively supported, the measure needs work.**
- The flip maps show image features acting along tissue borders (cortex) and location features acting in whole
  regions (deep structures, brainstem).
- The current border measure (6-neighbour label change) marks 60 % of all brain voxels as border, because the
  cortex is thin. So "61 % of the image flips are on a border" says nothing yet.
- Better: bin the flips by distance to the nearest label border (0, 1, 2, ≥ 3 mm) and compare with the base rate
  per bin.
- Already informative: only 26 % of the location flips are on a border, so location acts mainly *inside*
  regions.

## Suggested slides

1. **"More features ≠ better":** the groups bar chart. 12 location features beat 23 image features on the small
   structures; T1 alone is about 0.
2. **"Count doesn't help, information does":** the subsets curve (image vs. image + location) and the noise
   result (2× noise: no change).
3. **"Most families are redundant, one hurts":** the trimmed ablation plot (prior, prior distances, T2), with
   the footnote naming the 9 negligible families.
4. **"Where":** flip maps, coronal view. The image features act at the cortex borders, the location features in
   the deep structures, and the prior handles the brainstem block.
5. **"Why prior distances hurt":** over-segmentation numbers (about 2× volume) and the sampling rates. This is a
   limitation and also an explanation.
6. **Methods box:** 5 seeds, two-level bootstrap, ±0.01 relevance threshold, the five verdicts.

## Open points

- The image perturbation experiment (H3).
- Distance-to-border bins for the flip maps (H4).
- Optional: correct the class probabilities for the sampling rates, and check how much the over-segmentation and
  the prior-distance effect shrink.
- Boruta with 5 seeds, if H1 needs the "small all-relevant subset" claim.
