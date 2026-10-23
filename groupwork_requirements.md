# MIA Lab groupwork: what we have to do (simplified plan, 21.10)

This replaces the earlier, larger plan. After the team meeting on 21.10 we cut the code and the scope down to what we can understand and explain with the knowledge from *Introduction to Image Analysis*. The research question stays the same.

## The task in one paragraph

Automated brain tissue segmentation of T1w/T2w MRI (30 healthy HCP subjects, 20 train / 10 test) into background, white matter, grey matter, hippocampus, amygdala and thalamus. A working but deliberately weak pipeline is provided (registration → pre-processing → feature extraction → random forest → post-processing → evaluation). Each group picks one component, formulates a hypothesis about it, adapts the pipeline and designs experiments that confirm or reject the hypothesis. The result is **presented**, not written up.

## Grading

- **60% final presentation**, graded more on *how* we present than on the results:
  - thoughtful experiment design with respect to the hypothesis
  - clarity of what we found (positive *or* negative results are fine)
  - load balancing: everyone has a visible part
  - creativity
- **40% written multiple-choice exam** on the course content (02.12.2026).
- The midterm presentation only counts as a bonus.
- To pass, the workload must be shared equally.

## Dates

| Date | What |
|---|---|
| 28.10 / 05.11 | Project feedback sessions: bring questions |
| **11.11.2026** | **Midterm presentation, max 5 min** (approach, progress, challenges, insights) |
| 18.11 / 26.11 | Project feedback sessions |
| 02.12.2026 | Written exam (40%) |
| **09.12 or 16.12.2026** | **Final presentation, 10 min + 2 min questions** (60%) |

## Our hypothesis

**Hypothesis 6 (feature extraction): "A larger amount of features improves the segmentation performance."**

Guiding questions from the lecturers:
- Are PyRadiomics features useful?
- What are the important features? Is there a reason or even a clinical explanation?
- Do we need to select features, or is it better to use all of them?

What the lecturers expect (heard on 14.10): not just a benchmark of feature sets, but an explanation of **why** some features matter more and **what influences the information they carry**.

## Our plan in three questions

1. **Does more help, and where does it stop?** Train the same forest on growing sets of *real* features (baseline → + T2 → + multi-scale) and plot Dice against the number of features. This is the main experiment. We expect Dice to rise and then level off: a feature only helps if it tells the forest something about the label that the existing features don't already say, and later features increasingly repeat earlier ones (e.g. T1 smoothed at σ = 1 mm is almost the raw T1).
   - **Control:** add the same number of columns of *random* numbers. They add amount without information. If they don't help, an improvement in the main experiment comes from what the features carry, not from having more columns. The control doesn't answer the question on its own; it rules out one alternative explanation.
2. **Which features matter for which structure, and why?** SHAP shows how much the forest uses each feature per structure. Leaving out one feature group shows whether its information can be replaced by the others. Then we look for an anatomical explanation. For example, the amygdala has almost no contrast to the surrounding grey matter, so its position (atlas coordinates) should matter most.
3. **What does the information depend on?** Two simple handles:
   - **Image noise:** test the trained forests on noisier images and see which feature sets break.
   - **Scale:** compare smoothing filters at σ = 1, 2 and 4 mm and see which scale each structure prefers.

A nuanced answer is the goal, e.g. "more features help up to X, mostly for structure Y, because Z".

## How the code is organised (and why)

| Part | What it does | Who needs to understand it |
|---|---|---|
| `mialab/` (provided, with our Phase 1 fixes) | registration, z-score normalization, skull stripping, feature computation | the person adding features |
| `extract_features.py` | runs the pre-processing and feature computation once for all subjects and saves the results as numpy files in `features/` | everyone runs it, nobody needs to change it |
| `analysis.ipynb` | everything else, in plain numpy / scikit-learn: train forests, Dice/HD95, comparisons, SHAP, leave-one-group-out, noise | everyone |
| `pipeline.py` (provided) | the original end-to-end run with post-processing, kept as a reference | nobody needs it for the study |

**A feature set is just a list of column names** in the notebook (`X[:, columns(names)]`). Trying another set takes one line and no recomputation. This replaces the experiment runner, config files and feature cache from the earlier plan.

## What we deliberately left out, and what we say if asked

| Left out | Why | Our answer if asked |
|---|---|---|
| PyRadiomics voxel-based texture | slow (minutes per subject), and needs a cache and extra code | "We used simple, well-understood filters (smoothing, Laplacian of Gaussian at 3 scales). PyRadiomics is optional if time allows (todo.md, task 6.1)." |
| Boruta | a second importance method on top of SHAP | "Boruta compares each feature with shuffled copies of itself. Our random-feature control asks a similar question in a simpler way, and SHAP plus leaving groups out shows which real features matter." |
| Grouped permutation importance, SHAP attribution maps | more code and more figures than we can explain in 10 minutes | "Leaving out feature groups and retraining is the clearer test." |
| Experiment runner, configs, feature cache | too much infrastructure for our level | "All experiments are in one notebook; features are computed once." |
