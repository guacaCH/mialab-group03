# Methods explained (simplified plan, 21.10)

This explains every method in our plan ([groupwork_requirements.md](groupwork_requirements.md)) from the ground up, and where it is in our code. It assumes no prior experience with features or random forests.

---

## 1. The big picture: how our pipeline segments a brain

Segmentation here means: **for every voxel (3D pixel) of the brain, decide which tissue it belongs to**: background, white matter (WM), grey matter (GM), hippocampus, amygdala or thalamus.

It is a classic machine learning problem:

1. **Describe each voxel with numbers.** These numbers are the *features*, e.g. its T1 intensity, its T2 intensity, how quickly the intensity changes around it, and where it sits in the brain.
2. **Train a classifier** on voxels of the 20 training subjects, where the correct label (ground truth) is known.
3. **Predict** the label of every voxel of the 10 test subjects from their features.
4. **Evaluate** the prediction against the ground truth (Dice, HD95).

So each voxel becomes one row of a table:

| voxel | ATLAS_COORD_0 | ATLAS_COORD_1 | ATLAS_COORD_2 | T1w_INTENSITY | T1w_GRADIENT_INTENSITY | ... | label |
|---|---|---|---|---|---|---|---|
| #1 | 12.0 | -30.0 | 8.0 | 1.21 | 0.14 | ... | WM |
| #2 | 13.0 | -30.0 | 8.0 | 0.35 | 0.55 | ... | GM |

The classifier only sees these numbers. **If the features don't contain the information that distinguishes two tissues, no classifier can separate them.** That's why our hypothesis asks whether *more* features help.

**In our code:**
- `extract_features.py` builds these tables once and saves them: `features/train.npz` (the table `X` and the labels `y` of the training voxels) and `features/test_<subject>.npz` (all brain voxels of one test subject).
- `analysis.ipynb` does steps 2–4 with plain scikit-learn.

### Pre-processing (before features are computed)

All of this is in `mialab/` and already done (Phase 1):
- **Registration:** every subject is resampled into the space of the MNI atlas with the provided affine transform. Only then do coordinates mean the same thing in every subject.
- **Z-score normalization:** `(intensity − mean) / std`, with mean and std computed inside the brain mask. After this, intensities are comparable between subjects. A value of 1.0 means "one standard deviation brighter than the average brain voxel".
- **Skull stripping:** voxels outside the brain mask are set to 0.

---

## 2. What is a feature?

A **feature** is any number you can compute for a voxel. Ours, from simple to more complex:

| Feature (name in the code) | What it measures | Why it could help |
|---|---|---|
| `ATLAS_COORD_0/1/2` | position (x, y, z in mm) in atlas space | small structures (amygdala, hippocampus, thalamus) are always in roughly the same place |
| `T1w_INTENSITY`, `T2w_INTENSITY` | the grey value of the voxel | WM is bright and GM darker in T1. In T2 the contrast is reversed, and CSF is very bright |
| `T1w_GRADIENT_INTENSITY`, `T2w_GRADIENT_INTENSITY` | how strongly the intensity changes around the voxel | high at tissue boundaries, low inside a tissue |
| `T1w_GAUSS_S1/S2/S4` (task 2.2) | intensity after Gaussian smoothing with σ = 1, 2, 4 mm | the average of the surroundings: less noise, more context, but blurred boundaries |
| `T1w_LOG_S1/S2/S4` (task 2.2) | Laplacian of Gaussian: second derivative after smoothing | responds to blobs and edges of a size around σ |
| local mean / std (optional, task 2.3) | mean and standard deviation in a small cube around the voxel | is the surrounding uniform or mixed? |

**Scale** is the size of the neighbourhood a feature looks at (σ or cube size). Small scale captures fine detail and noise. Large scale captures context but blurs boundaries. Comparing σ = 1, 2 and 4 mm is our handle on "what does the information depend on".

**How a feature is added** (pattern in `FeatureExtractor.execute`, `mialab/utilities/pipeline_utilities.py`): compute an image with SimpleITK and store it in `self.img.feature_images` under a name. Every voxel of that image becomes one value in the feature's column. Example for the optional local statistics:

```python
if self.local_statistics_feature:
    for image_type, modality in ((structure.BrainImageTypes.T1w, 'T1w'), (structure.BrainImageTypes.T2w, 'T2w')):
        image = self.img.images[image_type]
        self.img.feature_images[f'{modality}_MEAN_R1'] = sitk.Mean(image, [1, 1, 1])   # 3x3x3 cube
        self.img.feature_images[f'{modality}_STD_R1'] = sitk.Noise(image, [1, 1, 1])   # local standard deviation
```

---

## 3. The classifier: random forest

### A decision tree

A **decision tree** is a sequence of yes/no questions on the features:

```
Is T1w_INTENSITY > 0.8?
├─ yes → Is ATLAS_COORD_2 > 10?
│        ├─ yes → WM
│        └─ no  → Thalamus
└─ no  → Is T2w_INTENSITY > 1.5?
         ├─ yes → Background (CSF)
         └─ no  → GM
```

**Training** builds the tree automatically. At each node it tries many features and thresholds and picks the question that best separates the classes (the groups become as "pure" as possible, measured by the Gini impurity). It stops when a group is pure or the maximum depth is reached. Each final box (*leaf*) predicts the most common label of its training voxels.

A single tree is unstable: small changes in the data give a very different tree, and a deep tree memorizes noise (*overfitting*).

### A random forest

A **random forest** combines many trees:
1. **Bagging:** each tree is trained on a random sample of the training voxels.
2. **Random features:** at each split, a tree may only choose among a random subset of the features (`max_features='sqrt'`). This makes the trees different from each other.
3. **Voting:** every tree predicts a label, and the forest takes the majority.

Averaging many different, imperfect trees cancels out their individual errors. Random forests work well with many features and need little tuning.

| Setting (`RandomForestClassifier`) | Meaning | Provided pipeline | Notebook default |
|---|---|---|---|
| `n_estimators` | number of trees | 1 | 50 |
| `max_depth` | maximum number of questions per path | 5 | 20 |
| `max_features` | features considered per split | all | `'sqrt'` |
| `random_state` | seed, makes the result reproducible | none | the seed |

We choose one setting (task 1.1) and **keep it fixed**, so that differences between experiments come only from the features.

### Training voxels

A brain has millions of voxels, so we train on a random **sample**. If every structure were sampled with the same percentage, the amygdala (about 2'500 voxels, 150× fewer than GM) would hardly appear in the sample, and the forest would never learn it. The pipeline therefore samples a higher percentage of the small structures: `[0.0003, 0.004, 0.003, 0.04, 0.04, 0.02]` for background, WM, GM, hippocampus, amygdala and thalamus (`_generate_feature_matrix` in `pipeline_utilities.py`). That gives about 6'500 voxels per subject, about 130'000 in total. The sample is fixed by `SEED` in `extract_features.py`, so every experiment uses exactly the same training voxels.

---

## 4. Evaluation and statistics

- **Dice coefficient:** overlap between prediction A and ground truth B, `2·|A∩B| / (|A|+|B|)`, from 0 (no overlap) to 1 (perfect). Harsh on small structures: a few wrong voxels cost the amygdala much more Dice than WM. Function `dice()` in the notebook.
- **HD95** (`HDRFDST95`): for every boundary point of one segmentation, the distance to the closest boundary point of the other; the 95th percentile, in mm. Catches stray blobs far away that Dice barely notices. Lower is better. It is `inf` if a structure isn't predicted at all. In the notebook: `evaluate(..., with_hd95=True)`, which uses the pipeline's evaluator.
- **Mean ± std over the 10 test subjects**, per structure: `summary()` in the notebook.
- **Seeds:** training with seeds 0, 1, 2 shows how much a result changes by chance alone. A difference smaller than that is noise.
- **Paired Wilcoxon signed-rank test** (`wilcoxon()` in the notebook): to claim "feature set B is better than A", compare their Dice on the *same* 10 subjects. The test asks whether the differences are consistently in one direction. With 10 subjects, the smallest possible p-value is about 0.002 (all 10 differences in the same direction). p < 0.05 is the usual threshold. With 5 structures and several comparisons, some results will be "significant" by chance, so say so in the presentation.

---

## 5. SHAP: how much does the forest use each feature?

**Idea:** SHAP (SHapley Additive exPlanations) splits a single prediction into one contribution per feature. For one voxel and the class *amygdala*:

```
P(amygdala) = base value + SHAP(ATLAS_COORD_2) + SHAP(T2w_INTENSITY) + SHAP(T1w_GRADIENT_INTENSITY) + ...
   0.82     =    0.05    +       +0.45         +        +0.25        +            +0.07             + ...
```

- **Base value:** the average predicted probability.
- **Positive SHAP value:** this feature pushed the prediction *towards* amygdala for this voxel; negative: away from it.
- The contributions always add up exactly to the prediction.

The idea comes from game theory: the features are players who produce the prediction together, and each one gets its fair share, averaged over all orders in which the players could join. For tree models `shap.TreeExplainer` computes this exactly.

**Importance per structure:** compute SHAP values for a sample of voxels (100 per label in the notebook), and average the **absolute** values per feature and class. The result says how much the forest relies on each feature to recognise each structure. The notebook (§3) draws this as a bar chart with one bar per structure.

**Watch out for correlated features:** if two features carry almost the same information (e.g. `T1w_GAUSS_S1` and `T1w_INTENSITY`), the forest uses them interchangeably and SHAP splits the importance between them. Both then look less important than the information really is. That's why we also sum SHAP over feature groups and run the leave-one-group-out test.

---

## 6. Experiments that test the explanations

### Leave one group out (notebook §4)

Remove a whole group of features (e.g. all coordinates), **retrain** the forest without them, and measure Dice. Repeat for each group.

SHAP tells us what the forest *uses*. Leaving a group out tells us whether that information is *irreplaceable*: after retraining, the forest can fall back on other features. Two outcomes to look out for:
- **High SHAP but no drop when removed:** the group's information is also contained in other features (redundant).
- **Big drop:** only this group carries that information.

### Random-feature control (task 4.2)

**What it is for.** The main test of "does more help?" is the comparison with growing sets of *real* features. The random-feature control is a check next to it, not a replacement. The hypothesis says "a larger *amount* of features". So if the real feature sets improve Dice, someone could ask: is it the information in the new features, or simply that the forest has more columns to choose its splits from? Random columns add amount but no information about the label. If adding them doesn't help, the improvement from the real features must come from what they carry.

**What makes a feature useful.** It is not enough that a feature is uncorrelated with the others: random numbers are uncorrelated with everything, including the label, and are useless. A feature helps if it carries *new information about the label*, i.e. something the existing features don't already tell the forest. Two consequences:
- **Redundant features add little.** `T1w_GAUSS_S1` is almost the same as `T1w_INTENSITY`; the forest gains little from having both.
- **Many weak features can even hurt slightly.** With `max_features='sqrt'`, each split only looks at a random subset of the features (e.g. 4 out of 19). The more weak or redundant columns there are, the less often a split gets to see the strong ones. With a limited number of training voxels, the trees may also pick splits on noise that happen to look good on the training data.

That's why we expect Dice to rise at first and then level off as more features are added, not to rise forever.

**How:** add as many random columns as there are real features and train again:

```python
rng = np.random.default_rng(0)
n = len(FEATURE_NAMES)
X_train_rand = np.hstack([X_train, rng.normal(size=(len(X_train), n))])
forest = RandomForestClassifier(random_state=0, **FOREST, n_jobs=-1).fit(X_train_rand, y_train)
for subject, data in test.items():
    X_test_rand = np.hstack([data['X'], rng.normal(size=(len(data['X']), n))])
    prediction = forest.predict(X_test_rand)
    ...  # dice() per label, as in evaluate()
```

**Expected result:** no improvement, maybe slightly worse, because some splits are wasted on noise. Together with the real-feature comparison this supports "information, not amount".

### Image noise (notebook §5)

`python extract_features.py --noise 0.1 0.2 0.5` adds Gaussian noise to the pre-processed T1 and T2 test images before the features are computed. The noise is in units of the brain's intensity standard deviation, so 0.2 means "20% of the normal spread of brain intensities". The forests stay trained on clean images.

What to look at:
- Which feature set loses most Dice? Features computed from single voxels (intensity, gradient) should suffer more than smoothed ones (`GAUSS_S2`, `GAUSS_S4`), and coordinates don't change at all.
- Which structures suffer most? Small structures with weak contrast should depend on coordinates and stay more stable.
- Optional: recompute SHAP on noisy voxels. Does the forest shift towards coordinates and smoothed features?

This links each feature type to an image property (noise), which is the "what influences the information" part of the lecturers' question.

---

## 7. Visuals for the presentation

| Figure | Shows | Question |
|---|---|---|
| Dice vs. number of features (one line per structure, random-feature control as an extra point) | whether more helps and where it stops | 1 |
| Dice boxplots per structure for the main feature sets | the spread over the 10 subjects, with significance | 1 |
| SHAP importance per structure, summed per feature group | which information each structure relies on | 2 |
| Leave-one-group-out table or bar chart | used vs. irreplaceable | 2 |
| Dice vs. noise level, one line per feature set | which features break under noise | 3 |
| Segmentation example: ground truth vs. baseline vs. best, same slice | visual intuition | context |

---

## 8. Glossary

- **Voxel:** 3D pixel (here 1 mm³).
- **Feature map:** an image in which each voxel holds the value of one feature.
- **Ground truth:** the reference segmentation we compare against.
- **Overfitting:** the model memorizes the training data and does worse on new data.
- **Redundant features:** features that carry the same information.
- **Ablation:** removing a component to measure its contribution (here: leave one group out).
- **Seed:** a number that fixes the randomness, so the result is reproducible.
- **σ (sigma):** the width of the Gaussian smoothing filter, in mm.
