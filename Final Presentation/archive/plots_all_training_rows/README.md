# Old plots: importances on all training voxels (2026-11-24)

These are the first versions of the presentation plots that change when the importances use brain voxels only. Keep them for comparison.

**Why they were replaced.** The training voxels are sampled over the whole image, so 41 372 of the 132 880 training rows (31 %) lie outside the brain mask. All of them are background, with T1 = T2 = 0 and distance to the brain border < 0. "Inside vs. outside the head" then becomes the easiest split, which inflates the features that encode it: distance to the brain border, gradients and symmetry. Marc found this; the new plots follow his recommendation. The pipeline stays as it is, and only the importance analyses drop these rows ([`../../brain_only_importance.py`](../../brain_only_importance.py)).

| Plot | Old (here, all rows) | New (`../../plots/`, brain rows only) |
|---|---|---|
| `00_flow_chart` | "80 % in 16 of 35 features" | "80 % in 15 of 35 features"; Part 3 notes that the correlation came from voxels outside the brain |
| `02_infogain_cumulative` | top: T1−T2, T1 smoothing, distance to brain border; 80 % / 90 % with 16 / 21 features | top: T1 intensity, T1−T2, prior distances; 80 % / 90 % with 15 / 22. Distance to brain border drops to rank 24 |
| `04_shap_family_per_structure` | WM 45 % intensity-type; thalamus / hippocampus / amygdala 62 / 59 / 54 % location | WM 49 % intensity-type; thalamus / hippocampus / amygdala 70 / 68 / 63 % location. Gradient ≈ 1 %, symmetry < 1 % |
| `05_correlation_drop` | same data | same data; footnote: inside the brain the two dropped prior distances correlate with r = 0.83 and −0.72, so they would have been kept |

Not affected, so not archived: `01`, `03`, `06` and `07`. They use test metrics and timings, and the test voxels are always brain voxels.

`make_plots_all_training_rows.py` is the script that made these plots. It expects to sit in `Final Presentation/`, so copy it there before running it.
