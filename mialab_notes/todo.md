# Groupwork to-do list (Hypothesis 6: features)

A step-by-step guide from today to the final presentation. Each task says **what** to do, **why**, and **how** (files, functions, packages), plus a "done when" check. Methods are explained in [feature_methods.md](feature_methods.md); the overall plan is in [groupwork_requirements.md](groupwork_requirements.md).

Owner tags are a suggestion for an equal split (equal workload is required to pass):
- **[A] Pipeline & features**: feature extraction, PyRadiomics, caching
- **[B] Experiments & selection**: RF setup, experiment runner, benchmark, Boruta, ablations, statistics
- **[C] Explanation & visuals**: evaluation/plots, SHAP, perturbations, figures
- **[all]**: everyone

---

## Phase 0: Admin (by Sun 18.10, 23:59)

- [ ] **0.1 [all] Fork and clone.** One person forks `github.com/ubern-mialab/MIALab`; the others clone that fork. Add the original as upstream: `git remote add upstream https://github.com/ubern-mialab/MIALab.git`. Use personal branches, merge into `main` when something works.
- [ ] **0.2 [all] Keep data out of git.** Add `MIALab_2020_Dataset/`, `mia-result/` and any feature cache folder to `.gitignore`. *Why:* the slides say "DO NOT COMMIT DATA" and the images are ~650 MB.
- [ ] **0.3 [all] Post on Slack:** member names, fork link, hypothesis 6 (official wording: "A larger amount of features improves the segmentation performance").
- [ ] **0.4 [all] Shared environment.** Same Python version (3.10 or 3.11 is safe), install `requirements.txt` plus `pyradiomics`, `shap`, `boruta`, `scipy`, `pandas`, `seaborn`. Run `test/test_install.py`.
  - *Done when:* everyone can run `python pipeline.py` to the end and gets `mia-result/<date>/results.csv`.

## Phase 1: Make the baseline pipeline correct (week of 19.10)

The provided pipeline still contains exercise `# todo`s. Several of them silently break our study, so they come first.

- [ ] **1.1 [A] Apply the registration** (`mialab/filtering/preprocessing.py`, `ImageRegistration.execute`). Use the provided affine transform with `sitk.Resample(image, atlas, transform, interpolator)`; use `sitk.sitkNearestNeighbor` for the ground truth and brain mask, `sitk.sitkLinear` for T1/T2. Enable `registration_pre: True` in `pipeline.py`.
  - *Why:* without registration, the **atlas coordinate features are meaningless** (each subject is in its own position), and coordinates are probably our most important feature for small structures.
  - *Done when:* a registered T1 overlaid on the atlas in ITK-SNAP lines up.
- [ ] **1.2 [A] Skull stripping** (`SkullStripping.execute`): `sitk.Mask(image, mask)`.
- [ ] **1.3 [A] Z-score normalization** (`ImageNormalization.execute`): `(x - mean) / std` computed inside the brain mask; enable `normalization_pre: True`.
  - *Why:* intensity-based features must be comparable across subjects, otherwise the forest learns scanner differences.
- [ ] **1.4 [C] Add HD95** in `init_evaluator()` (`pipeline_utilities.py:293`): `metric.HausdorffDistance(percentile=95, metric='HDRFDST95')`.
- [ ] **1.5 [C] Results plotting** (`plot_results.py`): load `results.csv`, boxplot Dice and HD95 per label.
- [ ] **1.6 [B] Make runs reproducible.** Set seeds (`np.random.seed`, `random_state` in the forest), since training voxels are sampled randomly.
- [ ] **1.7 [B] Choose and freeze the random forest.** Try a few settings on the *current* features only, e.g. `n_estimators` ∈ {50, 100}, `max_depth` ∈ {15, 25, None}, `class_weight` ∈ {None, "balanced"}. Pick one with good Dice and acceptable run time and **never change it again**.
  - *Why:* the current 1-tree, depth-5 forest is so weak that it hides any effect of features. Freezing it means later differences come only from features.
  - *Watch out:* don't tune on the test set too much; ideally use a few training subjects as validation (e.g. 15 train / 5 validation).
  - *Done when:* a baseline table of Dice and HD95 per structure (mean ± std) is saved. **This is the reference for everything else.**

## Phase 2: Experiment infrastructure (week of 19.10 → 26.10)

- [ ] **2.1 [B] Configurable feature sets.** Refactor `FeatureExtractor` (`pipeline_utilities.py`) so a feature set is selected by name or a list of feature IDs, not by three booleans. Each feature gets a name tagged with family, modality, scale (e.g. `T2_firstorder_variance_r2`). Store the names alongside the feature matrix.
- [ ] **2.2 [A] Feature cache.** Compute feature maps once per subject and save them (e.g. `.npy` or `.nii.gz` in a git-ignored `feature_cache/`). Experiments then load columns instead of recomputing.
  - *Why:* PyRadiomics voxel-based extraction is slow; you don't want to recompute it for every experiment.
- [ ] **2.3 [B] Experiment runner.** A script `run_experiment.py --features <set> --seed <n> --out <dir>` that trains, predicts, evaluates and writes `results.csv` + the config used. Run all experiments through it.
- [ ] **2.4 [C] Aggregation script.** Collect all `results.csv` into one table (experiment, seed, subject, label, Dice, HD95). Compute mean ± std and the **paired Wilcoxon test** vs. baseline (`scipy.stats.wilcoxon`).
  - *Done when:* one command produces a summary table for any set of experiments.

## Phase 3: Build the feature pool (26.10 → 05.11)

Add features in groups. After each group, run a quick experiment to make sure it works and helps (or not).

- [ ] **3.1 [A] T2 features**: T2 intensity and T2 gradient magnitude (the `# todo: add T2w features`). *Expected:* clear improvement for GM/WM, since T2 has opposite contrast.
- [ ] **3.2 [A] Neighborhood first-order features**: enable the existing `NeighborhoodFeatureExtractor` (16 statistics) for T1 and T2, kernel 3×3×3 and 5×5×5. It's slow in pure Python; consider replacing with `scipy.ndimage` filters (`uniform_filter` for mean, variance via mean of squares, `generic_filter` / percentile filters).
- [ ] **3.3 [A] Multi-scale filters**: Gaussian-smoothed intensity and LoG at σ = 1, 2, 4 mm for T1 and T2 (`sitk.SmoothingRecursiveGaussian`, `sitk.LaplacianRecursiveGaussian`). Cheap and informative about **scale**.
- [ ] **3.4 [A] PyRadiomics voxel-based texture**: start with one subject and one slab to measure run time. Settings to begin with: `voxelBased=True`, `kernelRadius` 1–2, classes `glcm`, `glszm`, `ngtdm`, `binWidth` chosen for the normalized intensities, mask = brain mask. Only expand if run time allows.
  - *Done when:* every feature exists as a cached map for all 30 subjects, with a table listing feature name, family, modality, scale.
- [ ] **3.5 [B] Redundancy check**: Spearman correlation matrix on a voxel sample, hierarchical clustering into correlated groups (`scipy.cluster.hierarchy`). Save the cluster assignment; it's needed for grouped permutation importance.

## Phase 4: Midterm presentation (prepare 05.11 → 10.11, present 11.11)

Max. 5 minutes, graded as a bonus only, but feedback is valuable.

- [ ] **4.1 [B] Run the first benchmark**: baseline vs. +T2 vs. +first-order vs. +multi-scale (+ PyRadiomics if ready), 3 seeds each.
- [ ] **4.2 [C] Plots**: Dice/HD95 boxplots per structure; performance vs. number of features.
- [ ] **4.3 [all] Slides** (~5): refined hypothesis and sub-questions H1–H4, experimental design (feature pool + Boruta/SHAP/perturbation plan), first results, challenges (run time, redundancy), next steps. **Each person presents a part.**
- [ ] **4.4 [all] Ask the TAs** for feedback on the plan, especially whether the H1–H4 framing answers their "why" expectation.

## Phase 5: Main experiments (12.11 → 25.11)

- [ ] **5.1 [B] Full benchmark with controls**, 3–5 seeds each:
  - feature ladder (baseline → full pool)
  - full pool + noise padding (as many random features as the pool has)
  - Boruta subset (from 5.2)
  - top-k per structure (from 5.3), k = 5, 10, 20
- [ ] **5.2 [B] Boruta**: on a stratified voxel sample (e.g. 2'000–5'000 voxels per class) with `BorutaPy(RandomForestClassifier(...), n_estimators='auto', max_iter=100)`. Once on the 6-class problem and once per structure (one-vs-rest). Save confirmed / tentative / rejected per structure.
- [ ] **5.3 [C] SHAP global importance**: `shap.TreeExplainer(forest)` on a stratified sample of test voxels. Per structure: mean |SHAP| per feature → top features; sum per family/modality/scale. Beeswarm plot for one small (amygdala) and one large (GM) structure.
- [ ] **5.4 [B] Grouped permutation importance**: per correlated cluster and per family, shuffle the whole group in the test features, measure Dice drop per structure. Repeat 5×.
- [ ] **5.5 [B] Drop-one-family ablation**: retrain without each family (spatial, intensity, gradient, first-order, texture) and without each modality (T1, T2). Compare against permutation results: used vs. irreplaceable.
- [ ] **5.6 [C] Image perturbations**: on the test images apply Gaussian noise (several levels), blur (σ = 0.5, 1, 2 mm), intensity scaling, synthetic bias field; recompute features; evaluate the frozen model trained on the full pool and on the baseline set. Also recompute SHAP family importance under noise.
- [ ] **5.7 [C] SHAP attribution maps**: for 1–2 axial slices of one test subject, compute SHAP per voxel, sum per family, write back into image space, overlay on T1 for amygdala, hippocampus, GM. Look specifically at boundary vs. interior voxels (H4): e.g. compare mean |SHAP| of texture features on voxels within 1–2 mm of a tissue boundary vs. the rest.

## Phase 6: Interpret (26.11 → 01.12)

- [ ] **6.1 [all] Answer each sub-hypothesis explicitly** with the evidence:
  - **H1** (amount vs. information): does performance plateau? does noise padding help? does Boruta's subset match the full pool?
  - **H2** (structure dependence): which families dominate per structure (SHAP, Boruta per structure)?
  - **H3** (what the information depends on): which features lose importance under which perturbation? how do scale and modality matter?
  - **H4** (where features act): do attribution maps concentrate on boundaries?
- [ ] **6.2 [all] Look for anatomical/clinical explanations** (the official guiding question asks for them): e.g. amygdala and hippocampus have little T1 contrast to surrounding GM, so position (coordinates) must carry the information; T2 helps CSF/GM separation; partial volume at boundaries explains why texture/gradients act there.
- [ ] **6.3 [all] Conclude**: confirm, reject or refine hypothesis 6. A nuanced answer ("more features help only up to X, and only for Y, because Z") is the goal. Negative results are fine if clearly explained.
- [ ] **6.4 [B] Write down all hyperparameters and settings** in one table (for an appendix slide and reproducibility).

*Note: written exam on 02.12. Plan study time in this week too.*

## Phase 7: Final presentation (03.12 → 09.12 or 16.12)

10 minutes + 2 minutes of questions, 60% of the grade, judged mostly on **how** it is presented.

- [ ] **7.1 [C] Final figures**, consistent colors per structure and per feature family across all plots. See the visuals table in [feature_methods.md](feature_methods.md#7-what-the-presentation-visuals-show).
- [ ] **7.2 [all] Storyline** (~10 slides): problem and pipeline (1) → hypothesis and H1–H4 (1) → design (1–2) → is more better? (H1, 1–2) → which features and why (H2, H4 with attribution maps, 2) → what the information depends on (H3, perturbations, 1) → conclusion and limitations (1).
- [ ] **7.3 [all] Split speaking time equally** and make responsibilities visible (rubric: load-balancing).
- [ ] **7.4 [all] Rehearse at least twice with a timer**; prepare answers for likely questions (Why random forest? Why these features? Is 10 test subjects enough? Are correlated features hurting SHAP?).

---

## Rules of thumb for the whole project

- Change **one thing at a time**; keep the forest, sampling and seeds fixed.
- Every experiment writes its config next to its results. No result without a config.
- Report Dice **and** HD95, per structure, mean ± std, plus significance.
- Keep a short lab log (date, what was run, what was learned). It makes the presentation easy to write.
- Commit code often to personal branches, merge working code into `main`; never commit data or results images.
