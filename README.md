# MIA Lab group 03: does a larger amount of features improve the segmentation?

Brain tissue segmentation with a random forest, based on the provided MIALab pipeline. Plan and methods: [groupwork_requirements.md](groupwork_requirements.md), [feature_methods.md](feature_methods.md), [todo.md](todo.md).

## Setup (once)

1. The data goes one level above this folder, as in the provided pipeline: `..\data\atlas`, `..\data\train`, `..\data\test`.
2. Create the environment (Python 3.9):
   ```
   uv venv --python 3.9
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
3. In VS Code, select `.venv` as the interpreter and as the notebook kernel.

## Run

1. `python extract_features.py`: pre-processing and features of all subjects, saved to `features/`. Run it again after adding a feature.
2. Open `analysis.ipynb` and run it from top to bottom: all experiments and plots.

`pipeline.py` is the original end-to-end pipeline (with post-processing), kept as a reference.

## Files

| File | Content |
|---|---|
| `mialab/` | the provided pipeline code with our changes (registration, z-score normalization, skull stripping, HD95, T2 features, feature names) |
| `extract_features.py` | computes the features once and saves them |
| `analysis.ipynb` | experiments: forest choice, feature sets, SHAP, leave-one-group-out, noise |
| `pipeline.py` | original pipeline (reference) |
