# MIA Lab groupwork: what we have to do

Sources read: `8_ImAnLab/Context_MIA_Lab.txt`, all PDFs in `8_ImAnLab/` (01_01 Introduction, 02 Git/tools/code introduction with the hypothesis proposals, 03_01 Registration, 01_02 Pipeline, 01_03 Python Assessment, 01_04 Python & MATLAB, 03_02 Pre/Post, 04_01 Feature Extraction, 04_02 Classification, Group8_anonymized example report), speaker notes of `04_01_feature_extraction_erika.pptx`, and the pipeline code in `mialab-group03/`.

## The task in one paragraph
Automated brain tissue segmentation of T1w/T2w MRI (30 healthy HCP subjects, 20 train / 10 test) into background, white matter, grey matter, hippocampus, amygdala and thalamus. A working but deliberately weak pipeline is provided (registration → pre-processing → feature extraction → random forest → post-processing → evaluation). Each group picks one pipeline component, formulates a hypothesis about it, adapts the pipeline and designs experiments that confirm or reject the hypothesis. The result is presented, not written up for grading.

## Grading (01_01 Introduction, slide 11)
- **60% final presentation**, graded more on *how* you present than on the actual results.
- **40% written multiple-choice exam** on course content (02.12.2026).
- Midterm presentation is graded internally and counts only as a bonus. Optional assignments help round up a fractional grade.
- Criteria named on the rubric:
  - Thoughtful design of experiments with respect to the hypothesis
  - Presentation of results: load-balancing of responsibilities within the team, clarity of what you found (positive *or* negative results are fine)
  - Creativity ("don't be afraid of exploring, thinking out of the box")

## Dates and deadlines
| Date | What |
|---|---|
| 14.10.2026 (today) | Groups confirmed and GitHub repository link handed in (Introduction slides) |
| **Sun 18.10.2026, 23:59** | **Post on Slack: group member names, forked repo link, chosen hypothesis** (02 slides, homework) |
| 21.10.2026 | Lecture: visualization, scientific writing & LaTeX |
| 28.10 / 05.11 | Project feedback sessions |
| **11.11.2026** | **Midterm presentation, max 5 min** (approach, progress, challenges, insights) |
| 18.11 / 26.11 | Project feedback sessions |
| 02.12.2026 | Written multiple-choice exam (40%) |
| **09.12 or 16.12.2026** | **Final presentation, max 10 min + 2 min questions** (60%) |

## Our group
Chosen topic: **Hypothesis 6 (Feature extraction): "A larger amount of features improves the segmentation performance"** (official wording, 02 slides p.22). One group per hypothesis; you may adapt the hypothesis later.

Guiding questions the lecturers attach to it:
- Are PyRadiomics features useful?
- What are the important/crucial features? Is there a reason or even a clinical explanation?
- Do we need to select some features, or is it better to employ all available features?

General rules from the 02 slides:
- "Focus on analysis (performance is secondary)": demonstrate the hypothesis with experiments, critically analyze and interpret, create new experiments based on findings, iterate.
- To pass, the workload must be shared equally.
- One person forks `ubern-mialab/MIALab`, the others clone the fork; add the original as `upstream`.
- **Do not commit data** (the local copy contains `MIALab_2020_Dataset/`, so check it is in `.gitignore`).
- Don't use deep learning unless very experienced (stated under hypothesis 4, applies in spirit).
- Slack is mandatory; all official communication happens there.

## Current pipeline state relevant to us (code in `mialab-group03/`)
- Features today (`mialab/utilities/pipeline_utilities.py`, `FeatureExtractor`): 3 atlas coordinates, T1w intensity, T1w gradient magnitude = **5 features**. T2w features are a `todo` (code warns "No features from T2-weighted image extracted").
- Already implemented but unused: `NeighborhoodFeatureExtractor` with 16 first-order texture statistics per voxel neighborhood (mean, variance, skewness, kurtosis, entropy, energy, percentiles, ...) in `mialab/filtering/feature_extraction.py`.
- Classifier (`pipeline.py:76`): `RandomForestClassifier(n_estimators=1, max_depth=5)`, i.e. a single shallow tree. This must be fixed to something reasonable and held constant, otherwise any result reflects the weak classifier more than the features.
- Training voxels are randomly subsampled per label (`RandomizedTrainingMaskGenerator`), so results vary between runs unless the seed is fixed.
- Pre-processing is not implemented yet: registration, skull stripping and z-score normalization are exercise `# todo`s that return the unprocessed image (`mialab/filtering/preprocessing.py`). Without registration the atlas coordinate features are meaningless, so this must be fixed first.
- Evaluation: Dice only so far; slides recommend several metrics, Dice and Hausdorff (HD95) are the standard pair.
- Slides suggest feature families: intensity (T1, T2), gradients, first-order texture, GLCM texture, radiomics (PyRadiomics), atlas/spatial coordinates, hemisphere symmetry, template matching, deep learning features. They stress feature *selection* (irrelevant/redundant/noisy features, curse of dimensionality), which is the natural counter-argument to our hypothesis.

## What a strong result looks like (from the Group 8 example report)
Group 8 tested "spatial regularization (post-processing) improves segmentation". Their structure is a good template:
1. A clear hypothesis and a fixed baseline vs. improved pipeline on the 10 held-out test subjects.
2. Per-structure results for Dice **and** HD95, reported as mean ± std, with boxplots across subjects.
3. **Ablations** that isolate each component, and an order-dependence study.
4. A nuanced conclusion: the hypothesis holds, but the effect is scale-dependent (big gains on small subcortical structures, plateau/trade-off on GM/WM).
5. An appendix with all hyperparameters (reproducibility).

## Lecturer expectation (overheard 14.10, relayed by Marc)
This matches the official guiding questions above ("What are the crucial features? Is there a reason or even clinical explanation?").
A plain benchmark of feature sets is **not** enough. The lecturers expect us to look into **why** certain features have a stronger impact on the segmentation metrics and **what influences the information they carry**. Suggested tools: **PyRadiomics, SHAP, Boruta, perturbations**.

So the feature-ladder benchmark is only the starting point; the main contribution has to be explanatory.

## Refined hypothesis (draft)
> Adding features improves segmentation only as long as the new features carry non-redundant, structure-specific information. Beyond that point performance plateaus, and the gain can be traced to a small set of features whose relevance depends on the target structure and on the image properties they encode.

Testable sub-questions:
- **H1 (amount vs. information):** a larger feature set beats the baseline, but a small all-relevant subset (Boruta) reaches the same Dice/HD95, and padding with the same number of noise features does not help.
- **H2 (structure dependence):** different structures rely on different feature families, e.g. atlas coordinates dominate for low-contrast subcortical structures (amygdala, hippocampus), while intensity/T2 and gradient features drive the GM/WM boundary.
- **H3 (what the information depends on):** a feature's importance changes with scale (neighborhood size, filter sigma), modality (T1 vs T2), redundancy with other features, and image quality. Perturbing the input images (noise, blur, intensity shift/bias field) should reduce exactly the features that encode that property.
- **H4 (where features act):** texture/gradient features matter mostly at tissue boundaries and partial-volume regions, not inside homogeneous tissue.

## How the suggested tools fit
| Tool | What it gives us | Question |
|---|---|---|
| **PyRadiomics** | Large, standardized feature pool: first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM, plus LoG/wavelet filtered versions. Use voxel-based mode (`voxelBased=True`) with a small kernel radius to get per-voxel feature maps. | Build the "many features" pool with known semantics per family |
| **Boruta** (`BorutaPy`) | All-relevant selection against shuffled "shadow" copies of every feature, i.e. a built-in noise control. Run per structure (one-vs-rest) as well as globally. | H1, H2 |
| **SHAP** (`TreeExplainer` on the RF) | Per-class, per-voxel attributions; global importance per structure; dependence plots (how a feature value maps to a class); interaction values. SHAP values can be written back into image space as **attribution maps**. | H2, H4 |
| **Perturbations** | (a) Permutation importance per feature and per *correlated feature cluster* (avoids importance being split across redundant features). (b) Drop-one-family ablation with retraining. (c) Image-level perturbations (Gaussian noise, blur, intensity scaling, synthetic bias field) applied before feature extraction, then measure which features lose importance and how Dice/HD95 drop. | H1, H3 |

## Experimental plan
1. **Fixed, fair baseline:** stronger RF (e.g. 50-100 trees, deeper), fixed seeds, same training-voxel sampling, several repetitions. Evaluate Dice + HD95 per structure on the 10 test subjects, mean ± std, paired Wilcoxon tests.
2. **Feature pool:** coordinates, T1/T2 intensity and gradient, multi-scale Gaussian/LoG, existing neighborhood first-order stats, PyRadiomics voxel-based texture for T1 and T2. Tag every feature with family, modality and scale.
3. **Short benchmark (context only):** baseline vs. +T2 vs. full pool vs. full pool + equal number of noise features vs. Boruta subset. Plot performance vs. number of features.
4. **Explain:** Boruta per structure, SHAP per structure, grouped permutation importance on correlation clusters (hierarchical clustering on Spearman correlation).
5. **Verify the explanations causally:** retrain with only the top features per structure, drop-one-family ablations, image perturbations; check that the effects match what SHAP/Boruta predicted.
6. **Visualize:** SHAP attribution maps overlaid on MRI slices, per-structure importance bar charts by feature family, performance-vs-features curve. These are the presentation's key visuals.

## Practical risks
- PyRadiomics voxel-based extraction on whole brain volumes is slow. Mitigate with small kernels, a few chosen feature classes, the brain mask, multiprocessing, caching feature maps to disk once, and possibly computing test-time features on a subset of slices.
- SHAP on a big forest is slow too: compute it on a stratified voxel sample (a few thousand per class).
- Correlated features distort single-feature importance (both SHAP and permutation), so report importance per feature family/cluster as well.
- Class imbalance: keep the per-label sampling fixed so that importance differences are not sampling artefacts.

## Suggested to-do list
1. By Sun 18.10 23:59: post names, fork link and hypothesis on Slack. Make sure `MIALab_2020_Dataset/` is git-ignored.
2. Get the pipeline running for everyone; fix seeds; choose the fixed RF configuration.
3. Add HD95 to evaluation and a results-aggregation script (per subject, per label, mean ± std, boxplots, Wilcoxon).
4. Implement T2 features, then the feature pool with PyRadiomics; cache feature maps.
5. Run the short benchmark (step 3 above), including the noise and Boruta arms.
6. SHAP, grouped permutation importance, ablations, image perturbations.
7. Midterm (5 min, 11.11): refined hypothesis, design, baseline vs. +T2 vs. pool, first Boruta/SHAP findings.
8. Possible split (visible in the presentation): one person owns the feature pool and pipeline, one owns selection and ablations (Boruta, permutation, drop-family), one owns explanation and visuals (SHAP maps, perturbation study).
