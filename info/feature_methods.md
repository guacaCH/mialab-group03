# Feature methods explained

This document explains, from the ground up, every method in the *Experimental plan* of [groupwork_requirements.md](groupwork_requirements.md). It assumes no prior experience with features or random forests. Read it top to bottom once; later use the headings as a reference.

---

## 1. The big picture: how our pipeline segments a brain

Segmentation here means: **for every voxel (3D pixel) of the brain, decide which tissue it belongs to**: background, white matter (WM), grey matter (GM), hippocampus, amygdala or thalamus.

The pipeline does this as a classic machine learning problem:

1. **Describe each voxel with numbers.** These numbers are the *features*. For one voxel we might store: its T1 intensity, its T2 intensity, how quickly the intensity changes around it, and where it sits in the brain.
2. **Train a classifier** on voxels from the 20 training subjects, where we know the correct label (the ground truth from FreeSurfer).
3. **Predict** the label of every voxel in the 10 test subjects from their features.
4. **Evaluate** the predicted segmentation against the ground truth (Dice, HD95).

So each voxel becomes one row in a table:

| voxel | coord_x | coord_y | coord_z | T1_intensity | T1_gradient | ... | label |
|---|---|---|---|---|---|---|---|
| #1 | 0.31 | 0.52 | 0.40 | 812 | 14.2 | ... | WM |
| #2 | 0.30 | 0.52 | 0.40 | 640 | 55.0 | ... | GM |

The classifier only ever sees these numbers. **If the features don't contain the information that distinguishes two tissues, no classifier can separate them.** That is why feature extraction matters, and why our hypothesis asks whether *more* features help.

---

## 2. What is a feature?

A **feature** is any measurable number you can compute for a voxel. Examples, from simple to complex:

- **Intensity:** the grey value of the voxel in the T1 or T2 image. WM is bright in T1 and dark in T2; GM is the opposite; that's why T2 should help.
- **Gradient magnitude:** how strongly intensity changes around the voxel. High at tissue boundaries, low inside a homogeneous tissue.
- **Atlas coordinates:** the voxel's position after aligning the brain to a standard template (MNI atlas). The amygdala is always in roughly the same place, so position is a strong hint.
- **Neighborhood statistics (first-order texture):** take a small cube around the voxel (e.g. 3×3×3 or 5×5×5) and compute the mean, variance, skewness, percentiles, entropy... of the intensities inside it. Tells you whether the surrounding is uniform or mixed.
- **Higher-order texture (GLCM etc.):** describes how intensities are *arranged* in the neighborhood, not just their distribution (see §5).
- **Filtered images:** smooth the image with a Gaussian of width σ, or apply a Laplacian of Gaussian (LoG, a blob/edge detector), and use the filtered value as a feature.

### Family, modality and scale

For our analysis we tag every feature with three properties, so we can later ask questions like "do texture features matter more than intensity features?" or "does T2 add anything?":

- **Family:** *what kind* of property the feature measures.
  - `spatial`: atlas coordinates
  - `intensity`: raw intensity, smoothed intensity
  - `gradient`: gradient magnitude, LoG (edge/blob detectors)
  - `first-order`: neighborhood statistics (mean, variance, percentiles...)
  - `texture`: GLCM, GLRLM, GLSZM... (spatial arrangement of intensities)
- **Modality:** which image it was computed from: `T1`, `T2`, or `none` (coordinates).
- **Scale:** the size of the neighborhood or filter used: e.g. kernel radius 1 (3×3×3 cube), 2 (5×5×5), or Gaussian σ = 1, 2, 4 mm. Small scale captures fine detail (and noise); large scale captures context (and blurs boundaries).

Example name: `T2_firstorder_variance_r2` = family *first-order*, modality *T2*, scale *radius 2*.

---

## 3. The classifier: random forest from scratch

### A decision tree

A **decision tree** is a sequence of yes/no questions on the features:

```
Is T1_intensity > 700?
├─ yes → Is coord_z > 0.6?
│        ├─ yes → WM
│        └─ no  → Thalamus
└─ no  → Is T2_intensity > 400?
         ├─ yes → GM
         └─ no  → Background
```

**Training** builds this tree automatically. At each node, the algorithm tries many features and thresholds and picks the one question that best separates the classes, measured by how "pure" the resulting groups become (Gini impurity or information gain, see the classification slides). It keeps splitting until the groups are pure or too small. The final boxes are *leaves*; each leaf predicts the most common label among the training voxels that ended up there.

A single tree is unstable: small changes in the training data give a very different tree, and a deep tree memorizes noise (overfitting).

### A random forest

A **random forest** = many trees whose votes are combined:

1. **Bagging:** each tree is trained on a random sample (with replacement) of the training voxels.
2. **Random feature subsets:** at each split, a tree may only choose among a random subset of the features (`max_features`). This makes trees different from each other.
3. **Voting:** for a new voxel, every tree predicts a label; the forest outputs the majority (or the average of class probabilities via `predict_proba`).

Averaging many different, imperfect trees cancels out their individual errors. Random forests work well with many features, need little tuning and don't need feature normalization, which is why they're used here.

**Important settings** (scikit-learn `RandomForestClassifier`):

| Parameter | Meaning | Current value | Sensible start |
|---|---|---|---|
| `n_estimators` | number of trees | 1 | 50–100 |
| `max_depth` | maximum questions per path | 5 | 15–25, or None |
| `max_features` | features considered per split | all | `"sqrt"` (default) |
| `class_weight` | up-weight rare classes | none | `"balanced"` worth trying |
| `random_state` | seed for reproducibility | none | fixed, e.g. 42 |
| `n_jobs` | CPU cores | 1 | -1 (all cores) |

The current setting (1 tree, depth 5) is far too weak; with it, adding features can't show much effect. We pick one reasonable setting once and **keep it fixed** for all experiments, so that differences come from the features alone.

### Training voxel sampling

A brain has millions of voxels; we can't train on all of them, so we train on a random **sample** of voxels. *How many voxels of each structure* end up in that sample decides how much the forest "cares" about each structure.

#### Step 1: count how many voxels each structure has

Real numbers from our 20 training subjects (1 mm³ voxels, image 181×217×181, average per subject, computed from `labels_native.nii.gz`):

| Label | Voxels per subject | Share of the brain mask |
|---|---|---|
| Background, whole image | 6'313'435 | – |
| Background **inside the brain mask** (mostly CSF) | 560'032 | 41.3 % |
| White matter | 374'585 | 27.6 % |
| Grey matter | 398'132 | 29.4 % |
| Hippocampus | 6'766 | 0.50 % |
| Amygdala | 2'488 | **0.18 %** |
| Thalamus | 13'729 | 1.0 % |

How to compute it yourself: load the label image, count each label value.

```python
import SimpleITK as sitk, numpy as np
labels = sitk.GetArrayFromImage(sitk.ReadImage("labels_native.nii.gz"))
counts = np.bincount(labels.ravel(), minlength=6)   # counts[k] = number of voxels with label k
```

The amygdala is about **150× smaller** than grey matter.

#### Step 2: why proportional sampling drowns small structures

If you sample the same percentage of every structure, the sample keeps these proportions. Out of 10'000 sampled voxels, only ~18 would be amygdala. A decision tree splits to make groups as pure as possible *overall*; misclassifying 18 amygdala voxels as grey matter barely changes impurity, so the trees never bother to build branches for the amygdala. The forest then simply predicts "grey matter" there. Result: amygdala Dice close to 0.

#### Step 3: choose a sampling rule

Decide a total budget **B** of training voxels per subject (e.g. 10'000; total training set = 20 × B = 200'000 rows) and how to split it between the K = 6 labels. A convenient family of rules uses an exponent **α**:

```
N_c = B · n_c^α / Σ_k n_k^α          (N_c = voxels to sample for label c, n_c = voxels available)
p_c = N_c / n_c                      (fraction of label c to sample = the percentage the pipeline expects)
```

- **α = 1, proportional:** sample keeps the natural proportions. Small structures drown.
- **α = 0, balanced:** every label gets B / K voxels, regardless of size.
- **α = 0.5, square-root compromise:** small structures are boosted, but large ones still get more samples than small ones.

Worked example with B = 10'000, background restricted to the brain mask:

| Label | n_c | α = 1 (proportional) | α = 0.5 (sqrt) | α = 0 (balanced) |
|---|---|---|---|---|
| Background (in mask) | 560'032 | 4'131 (p = 0.74 %) | 3'340 (0.60 %) | 1'667 (0.30 %) |
| White matter | 374'585 | 2'763 (0.74 %) | 2'731 (0.73 %) | 1'667 (0.44 %) |
| Grey matter | 398'132 | 2'937 (0.74 %) | 2'816 (0.71 %) | 1'667 (0.42 %) |
| Hippocampus | 6'766 | 50 (0.74 %) | 367 (5.4 %) | 1'667 (24.6 %) |
| Amygdala | 2'488 | **18** (0.74 %) | 223 (9.0 %) | 1'667 (**67 %**) |
| Thalamus | 13'729 | 101 (0.74 %) | 523 (3.8 %) | 1'667 (12.1 %) |

**For comparison, the current pipeline** uses the percentages `[0.0003, 0.004, 0.003, 0.04, 0.04, 0.02]` (`pipeline_utilities.py`, `_generate_feature_matrix`) on the *whole* image. That gives ~5'200 voxels per subject: background 36 %, WM 29 %, GM 23 %, hippocampus 5.2 %, amygdala 1.9 %, thalamus 5.2 %. So someone already hand-tuned a compromise between proportional and balanced, roughly similar to α = 0.5.

Two practical caps:
- **p_c can't exceed 1.** If B is large, a small label may not have enough voxels (amygdala has only ~2'500); then take all of them.
- **Restrict background to the brain mask** (`background_mask=` in `get_mask`). Otherwise most "background" samples are air outside the head, which is trivially easy and wastes budget.

#### Step 4: the trade-off (why not always balanced?)

Sampling changes what the forest believes about how *common* each class is (its prior). With balanced sampling, the forest thinks amygdala is as common as grey matter, so in uncertain voxels it predicts amygdala too often → **false positives**, scattered amygdala blobs, worse HD95. Proportional sampling causes the opposite: **false negatives**. The best choice is somewhere in between and must be measured: try α ∈ {0, 0.5, 1} (or the current percentages) with the frozen forest, compare Dice and HD95 per structure, pick one and keep it fixed (to-do task 1.7).

Alternatives that achieve a similar effect without changing the sample:
- `RandomForestClassifier(class_weight="balanced")`: keeps proportional sampling but makes errors on rare classes count more in the split criterion.
- Correct the prior after prediction: divide each class probability by its sampling share and multiply by its true share, then take the argmax.

Because the sampling is random, we fix the seed and repeat experiments a few times to see the variability. **For our study, sampling must stay identical across all feature experiments**, otherwise differences in feature importance could be sampling artefacts.

---

## 4. Evaluation metrics

- **Dice coefficient:** overlap between prediction and ground truth, `2·|A∩B| / (|A|+|B|)`, from 0 (no overlap) to 1 (perfect). Easy to read, but harsh on small structures: a 1-voxel error costs much more Dice on the amygdala than on WM.
- **HD95 (95th percentile Hausdorff distance):** for every boundary point of one segmentation, the distance to the nearest boundary point of the other; take the 95th percentile (in mm). Measures how far off the boundaries are, and catches stray false-positive blobs far away that Dice barely notices. Lower is better.

We report both, per structure, as mean ± standard deviation over the 10 test subjects.

**Paired Wilcoxon signed-rank test:** to claim "feature set B is better than A", we compare their Dice on the *same* 10 subjects pairwise. The Wilcoxon test asks whether the differences are consistently positive, without assuming a normal distribution. p < 0.05 is the usual threshold; with 10 subjects and 5 structures, mention that many tests were made.

---

## 5. The tools

### 5.1 PyRadiomics: building the large feature pool

**What it is:** an open-source Python library that computes hundreds of standardized image features ("radiomics"). Normally it computes one value per region (e.g. per tumour). In **voxel-based mode** (`voxelBased=True`) it slides a small cube (radius set by `kernelRadius`) over the image and computes the features for every voxel, producing one *feature map* (an image) per feature, which is exactly what our voxel classifier needs.

**Feature classes it provides:**

| Class | What it measures, in plain words |
|---|---|
| `firstorder` | statistics of intensities in the cube: mean, variance, skewness, entropy, percentiles (like our existing neighborhood features) |
| `glcm` (Gray Level Co-occurrence Matrix) | how often intensity *i* sits next to intensity *j*. Captures smoothness vs. graininess: e.g. *contrast*, *homogeneity*, *correlation* |
| `glrlm` (Run Length) | how long runs of equal intensity are in a direction: long runs = smooth/striped structure |
| `glszm` (Size Zone) | size of connected zones of equal intensity: large zones = homogeneous tissue |
| `gldm` (Dependence) | how many neighbors have a similar intensity |
| `ngtdm` (Neighbouring Gray Tone Difference) | difference between a voxel and the average of its neighbors: "coarseness", "busyness" |

**Image filters:** PyRadiomics can compute all of the above on filtered versions of the image too: **LoG** at several σ (blob/edge structure at different scales) and **wavelet** decompositions (high/low frequency content in each direction). This multiplies the number of features, which is how you get to hundreds.

**Practical notes:** texture features need intensities to be *discretized* into bins first (`binWidth` or `binCount`). Voxel-based extraction over a whole brain is slow, so restrict it to the brain mask, choose a few classes and kernel radii, run it once per subject and save the feature maps to disk.

**What it gives our study:** the "larger amount of features" in our hypothesis, with clear semantics per feature (family, modality, scale), so that we can later explain *which kind* of information helps.

### 5.2 Boruta: which features are actually relevant?

**Problem:** with hundreds of features, many are useless or redundant. Random forest importance scores alone can't tell you where "useful" stops, because every feature gets some score above zero just by chance.

**How Boruta works:**

1. For every real feature, create a **shadow feature**: a copy whose values are randomly shuffled across voxels. Shuffling destroys any relation to the label, so shadows are pure noise with the same distribution as the original.
2. Train a random forest on real + shadow features and compute importance for all of them.
3. A real feature scores a "hit" if its importance beats the **best shadow feature**.
4. Repeat many times. Features that consistently beat the best shadow are **confirmed** (relevant); features that consistently lose are **rejected**; the rest stay **tentative**.

Boruta is an **all-relevant** method: it keeps every feature that carries *any* information, including redundant ones, instead of the smallest sufficient set. Python package: `BorutaPy` (works with scikit-learn's random forest).

**Boruta per structure:** run Boruta not only on the full 6-class problem, but once per structure as a **one-vs-rest** problem ("amygdala vs. everything else", "WM vs. everything else", ...). The confirmed sets then tell you which features matter *for that particular structure*. Comparing the sets across structures answers H2: e.g. amygdala might confirm mostly coordinates and T2 texture, WM mostly T1 intensity.

**Why it is our built-in noise control:** if "more features" helped regardless of content, shadow (noise) features would get confirmed too. They never are, by construction, which directly supports "information, not amount".

### 5.3 SHAP: why did the forest predict this label for this voxel?

**What it is:** SHAP (SHapley Additive exPlanations) explains a single prediction by splitting it into contributions from each feature. The idea comes from game theory (Shapley values): treat features as players cooperating to produce the prediction, and fairly divide the "payout" among them by averaging each feature's contribution over all possible orders of adding features.

For one voxel and one class (say *amygdala*):

```
P(amygdala) = base value + SHAP(coord_z) + SHAP(T2_intensity) + SHAP(T1_glcm_contrast) + ...
   0.82     =    0.05    +     +0.40     +       +0.25        +        +0.08         + ...
```

- **Base value:** the average predicted probability over the training data.
- **Positive SHAP value:** this feature pushed the prediction *toward* amygdala for this voxel.
- **Negative value:** pushed it away.
- The values always add up exactly to the prediction.

For tree models, `shap.TreeExplainer` computes these exactly and fairly fast. With 6 classes you get one set of SHAP values per class.

**SHAP per structure (global importance):** take a sample of voxels, compute SHAP values for the class of interest (e.g. thalamus), and average the **absolute** SHAP value of each feature. This ranks features by how much they influence thalamus predictions on average. The **top features of a structure** are simply the highest-ranked features in that list (e.g. the top 10 for amygdala). Do this per structure and compare.

**Other SHAP plots you will use:**

- **Summary (beeswarm) plot:** every dot is one voxel; x-position = SHAP value, color = feature value (red high, blue low). Shows both importance and direction ("high T2 intensity pushes toward GM").
- **Dependence plot:** feature value on x, its SHAP value on y. Shows *how* the feature is used, e.g. a threshold effect.
- **Interaction values:** how much two features act together beyond their individual effects.

**Practical note:** SHAP on a 100-tree forest over millions of voxels is too slow. Use a stratified sample (a few thousand voxels per class) for the global analysis, and one or two slices for attribution maps.

### 5.4 SHAP attribution maps

Because every voxel gets SHAP values, we can put them back into image space: for a chosen class and feature (or feature family), create an image where each voxel's value is its SHAP value. Overlay it on the MRI slice with a color map (red = pushes toward the class, blue = pushes away).

**What it shows:** *where* in the brain a feature does its work. For example, a map for "texture features → GM" might light up only along the GM/WM boundary, while "coordinates → amygdala" lights up as a blob around the amygdala. That is direct visual evidence for H4 ("texture/gradient matters at boundaries") and very convincing on a slide.

Summing SHAP values over all features of one family gives a **family attribution map** ("how much did texture help here?"), which is easier to read than per-feature maps.

### 5.5 Perturbations: testing the explanations

SHAP and Boruta tell you what the model *uses*. Perturbation experiments check whether that really matters for the result, by deliberately breaking something and measuring the damage.

#### (a) Permutation importance

Take the trained forest and the test data. Shuffle one feature's column (break its link to the label), predict again, and measure how much Dice drops. Large drop = the model depends on that feature. No retraining needed. scikit-learn: `sklearn.inspection.permutation_importance`.

**Problem with correlated features:** if `T1_mean_r1` and `T1_mean_r2` are almost identical, shuffling one does nothing (the model uses the other), so both look unimportant even though the information is crucial. SHAP has a similar issue: importance gets split between them.

#### (b) Grouped permutation importance

Solution: first group features that are strongly correlated with each other, then shuffle a **whole group at once**.

How to form groups: compute the Spearman correlation between all features, turn it into a distance (`1 − |correlation|`), run hierarchical clustering, cut the tree at a threshold (e.g. correlation > 0.8 → same cluster). Alternatively, group by *family*, *modality* or *scale* directly.

Result: "shuffling all T2 texture features drops amygdala Dice by 0.12" is a clean, robust statement.

#### (c) Drop-one-family ablation

**Ablation** = remove a component and see what breaks. Here: remove one whole family (e.g. all texture features), **retrain** the forest without them, and evaluate. Repeat for each family.

Difference from permutation: permutation tests a model that was trained *with* the feature; ablation retrains, so the model can compensate using other features. If dropping texture costs nothing after retraining, then texture information is *redundant* with something else (e.g. gradients). Comparing both tells you whether a family is *used* vs. *irreplaceable*.

#### (d) Image perturbations

Instead of manipulating feature columns, degrade the **input images** before feature extraction, then recompute features and predict:

- **Gaussian noise** added to T1 or T2 → simulates a noisier scan
- **Blurring** (Gaussian smoothing) → simulates lower resolution / partial volume
- **Intensity scaling / shift** → simulates scanner differences
- **Synthetic bias field** (a smooth multiplicative brightness gradient) → the classic MRI artifact

Then measure: (1) how much Dice/HD95 drop, (2) how feature importance (SHAP) shifts. Expected example: adding noise should hurt fine-scale texture features most and shift the model toward coordinates and large-scale features.

**What it answers:** H3, "what influences the information a feature carries". It links each feature family to a physical image property (noise level, resolution, contrast), which is the kind of *why* explanation the lecturers asked for.

---

## 6. The benchmark controls

- **Feature ladder:** baseline (5 features) → + T2 → + neighborhood first-order → + multi-scale filters → + PyRadiomics texture. Performance plotted against the number of features. This is the "is more better?" curve.
- **Noise-padded set:** baseline + as many pure random features as the full pool has. If performance doesn't improve, the *count* of features isn't what helps.
- **Boruta subset:** only the confirmed features. If it matches the full pool, the extra features were not needed.
- **Top-k per structure:** retrain using only the k best features per structure (from SHAP). Shows how few features are really needed for each structure.

---

## 7. What the presentation visuals show

| Visual | What it shows | Supports |
|---|---|---|
| **Performance vs. number of features** (line plot, one line per structure, error bars over repetitions; noise-padded and Boruta sets as extra markers) | Whether more features help and where it plateaus; that noise doesn't help; that the Boruta subset is as good | H1 |
| **Dice/HD95 boxplots** per structure for the main feature sets | Spread over the 10 test subjects, significance stars | H1 |
| **Importance by family, per structure** (stacked or grouped bar chart from SHAP or grouped permutation) | Which kind of information each structure relies on | H2 |
| **Boruta confirmed features per structure** (heatmap: structures × features or families) | Overlap and differences of relevant feature sets | H1, H2 |
| **SHAP beeswarm** for one small and one large structure | Direction and size of the top features' effects | H2 |
| **SHAP attribution maps** overlaid on an MRI slice | Where each feature family acts (boundaries vs. interior) | H4 |
| **Perturbation plot** (noise level on x, Dice on y, one line per feature set; plus importance shift) | Which features break under which image degradation, i.e. what information they carry | H3 |
| **Correlation cluster heatmap** | How redundant the feature pool is | H1 |
| **Segmentation examples** (ground truth vs. baseline vs. best, same slice) | Visual intuition for the improvement | context |

---

## 8. Glossary

- **Voxel:** 3D pixel.
- **Feature map:** an image where each voxel holds one feature's value.
- **Ground truth:** the reference segmentation we compare against (here FreeSurfer labels).
- **Training/test split:** 20 subjects to train, 10 never seen during training to evaluate.
- **Overfitting:** the model memorizes training data and does worse on new data.
- **Curse of dimensionality:** with many features and limited samples, the space becomes so large that data is sparse and models find accidental patterns.
- **Redundant feature:** carries the same information as another feature.
- **One-vs-rest:** turning a multi-class problem into "this class vs. all others".
- **Ablation:** removing a component to measure its contribution.
- **Seed:** a number that fixes the randomness so results are reproducible.
