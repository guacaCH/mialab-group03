# To-do list (simplified plan, 21.10)

The plan behind this list is in [groupwork_requirements.md](groupwork_requirements.md); the methods are explained in [feature_methods.md](feature_methods.md).

Owner tags are a suggestion for an equal split (required to pass), and each one gives a visible part for the presentation:
- **[A] Features:** adds the new features in `mialab/`, runs `extract_features.py`
- **[B] Experiments:** forest choice, feature-set comparison, random-feature control, statistics
- **[C] Explanation:** SHAP, leave-one-group-out, noise experiment, figures
- **[all]:** everyone

---

## Phase 0: Set up the simplified project (this week)

- [ ] **0.1 [all] Decide where the code lives.** Either a new branch `simple` in `mialab-group03` (recommended: keeps history and GitHub), or a fresh repository. Copy the contents of `mialab_simple/` there. Don't commit `features/` (it is in `.gitignore`); the data lives outside the project folder.
- [ ] **0.2 [all] Data.** The data goes one level above the project folder, as in the provided pipeline: `..\data\atlas`, `..\data\train`, `..\data\test`. All scripts and the notebook use these paths by default.
- [ ] **0.3 [all] Environment.** In the project folder:
  ```
  uv venv --python 3.9
  .venv\Scripts\activate
  uv pip install -r requirements.txt
  ```
  Then in VS Code select `.venv` as the Python interpreter and as the notebook kernel.
- [ ] **0.4 [all] Compute the features:** `python extract_features.py` (about 20 seconds per subject on our test machine, longer on a laptop). It writes `features/train.npz` and one `features/test_<subject>.npz` per test subject (about 70 MB each).
- [ ] **0.5 [all] Run `analysis.ipynb` once from top to bottom.** Everyone should be able to do this before the next meeting. *Done when:* every cell runs and prints a table.

## Phase 1: Baseline (until 28.10)

- [ ] **1.1 [B] Choose the random forest** (notebook §1). Try a few settings on the provided features, e.g. 50 or 100 trees, depth 15, 25 or unlimited (`None`). Pick one with good Dice and acceptable run time, write it into `FOREST`, and **don't change it afterwards**. Then differences between experiments come only from the features.
  - *Why:* the provided forest (1 tree, depth 5) never predicts the amygdala at all (Dice 0.00 in our run on 21.10).
- [ ] **1.2 [B] Baseline table:** Dice and HD95 per structure (mean ± std over the 10 test subjects) for the provided features with the chosen forest (notebook §6). *This is the reference for everything else.*

## Phase 2: More features (until 05.11)

- [ ] **2.1 [A] T2 features.** *Already in the code* (`T2w_INTENSITY`, `T2w_GRADIENT_INTENSITY`). Run notebook §2 to compare "baseline" with "baseline + T2".
- [ ] **2.2 [A] Multi-scale filters** for T1 and T2: Gaussian smoothing and Laplacian of Gaussian (LoG) at σ = 1, 2, 4 mm. That adds 12 features. In `FeatureExtractor` (`mialab/utilities/pipeline_utilities.py`), add a flag in `__init__`:
  ```python
  self.multiscale_feature = kwargs.get('multiscale_feature', False)
  ```
  and this block in `execute`, next to the other features:
  ```python
  if self.multiscale_feature:
      for image_type, modality in ((structure.BrainImageTypes.T1w, 'T1w'), (structure.BrainImageTypes.T2w, 'T2w')):
          image = self.img.images[image_type]
          for sigma in (1, 2, 4):
              self.img.feature_images[f'{modality}_GAUSS_S{sigma}'] = sitk.SmoothingRecursiveGaussian(image, sigma)
              self.img.feature_images[f'{modality}_LOG_S{sigma}'] = sitk.LaplacianRecursiveGaussian(image, sigma)
  ```
  Then set `'multiscale_feature': True` in `PRE_PROCESS_PARAMS` (`extract_features.py`) and run `python extract_features.py` again.
- [ ] **2.3 [A] Optional: neighbourhood statistics** (local mean and standard deviation in a 3×3×3 or 5×5×5 cube), same pattern as 2.2. See feature_methods.md §2.
- [ ] **2.4 [B] Feature sets in the notebook.** Add the new sets to `FEATURE_SETS`, e.g. `'+ multi-scale'`, and to `GROUPS`, e.g. `'smoothing'` and `'LoG'`.

## Phase 3: Midterm presentation (prepare 05.11 → 10.11, present 11.11)

Max. 5 minutes. It only counts as a bonus, but the feedback is valuable.

- [ ] **3.1 [B] First comparison:** baseline vs. + T2 vs. + multi-scale, 3 seeds each (notebook §2), boxplot of Dice per structure.
- [ ] **3.2 [all] Slides** (~5): hypothesis and the three questions, the approach (pipeline → features → notebook), first results, what comes next. **Each person presents a part.**
- [ ] **3.3 [all] Ask the tutors** whether the three questions answer their "why" expectation well enough.

## Phase 4: Main experiments (12.11 → 25.11)

- [ ] **4.1 [B] Does more help?** Dice per structure against the number of features: baseline (5) → + T2 (7) → + multi-scale (19). Plot one line per structure (mean over 3 seeds).
- [ ] **4.2 [B] Random-feature control** (a check next to 4.1, not a replacement). Add as many columns of random numbers as the full set has and train again. If Dice doesn't improve, the improvement in 4.1 comes from the information in the real features, not from their number. Explanation and code in feature_methods.md §6.
- [ ] **4.3 [C] SHAP per structure** (notebook §3): which features does the forest use for which structure? Sum per group too (coordinates, T1, T2, smoothing, LoG).
- [ ] **4.4 [C] Leave one group out** (notebook §4): retrain without each group. Compare with SHAP: "used" (high SHAP) vs. "irreplaceable" (Dice drops when removed).
- [ ] **4.5 [C] Noise** (notebook §5): `python extract_features.py --noise 0.1 0.2 0.5`, then compare how the baseline and the full set lose Dice per structure.
- [ ] **4.6 [B] Statistics:** paired Wilcoxon test against the baseline (`wilcoxon()` in the notebook) for the main comparisons.

## Phase 5: Interpret (26.11 → 01.12)

- [ ] **5.1 [all] Answer the three questions** from groupwork_requirements.md, each with one figure as evidence.
- [ ] **5.2 [all] Anatomical explanations.** Examples to check against our results:
  - Amygdala and hippocampus have little contrast to the surrounding grey matter, so position (coordinates) should matter most.
  - T2 should help to separate grey matter from CSF.
  - Smoothing at a large σ gives context but blurs the boundaries of small structures.
- [ ] **5.3 [B] One table with all settings** (forest, sampling, seeds, features) for an appendix slide.

*Note: written exam on 02.12. Plan study time in this week too.*

## Phase 6: Final presentation (03.12 → 09.12 or 16.12)

10 minutes + 2 minutes of questions, 60% of the grade, judged mostly on **how** it is presented.

- [ ] **6.1 [all] Optional, if time allows: PyRadiomics.** A few texture features on a small region, to answer the lecturers' question "are PyRadiomics features useful?" at least qualitatively. Skip it if time is short; "simple filters were enough to answer our questions" is a valid point.
- [ ] **6.2 [C] Final figures** with consistent colours per structure and per feature group.
- [ ] **6.3 [all] Storyline** (~10 slides): problem and pipeline → hypothesis and three questions → setup → does more help? → which features and why → what the information depends on (noise, scale) → conclusion and limitations.
- [ ] **6.4 [all] Equal speaking time**, and responsibilities visible on the slides.
- [ ] **6.5 [all] Rehearse twice with a timer**, and prepare answers for likely questions: Why a random forest? Why these features? Are 10 test subjects enough? Why no PyRadiomics or Boruta?

---

## Rules of thumb

- Change **one thing at a time**; keep the forest, the sampling and the seeds fixed.
- Report Dice (and HD95 for the final models) per structure, mean ± std.
- Keep a short lab log in the notebook (a markdown cell per experiment: what, why, result). It makes the presentation easy to write.
- Don't commit data or `features/`.
