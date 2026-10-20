# Phase 2 code: tasks 2.1 to 2.4

Code for [todo.md](todo.md) Phase 2, built on your current repo (the version with your Phase 1 changes).

| Task | What you get |
|---|---|
| 2.1 Configurable feature sets | A **feature pool**: every feature is one object with tags (family, modality, scale) and a name built from them. Experiments select features by set name, feature name, glob pattern or tag, instead of three booleans. The names travel with the feature matrix (`img.feature_names`). |
| 2.2 Feature cache | Slow features are saved per subject in `feature_cache/` and loaded on the next run. |
| 2.3 / 2.4 Runner and aggregation | The updated `run_experiment.py` from [experiment_runner/](experiment_runner/), now without the name workaround, plus a new `features.csv` per run. `aggregate_results.py` is unchanged. |

**Tested** on a copy of your repo (2 training subjects, 3 test subjects):
- With the provided features, the new `FeatureExtractor` gives **exactly the same** feature matrix and labels as your current one, for training (same seed) and test, and also with multiprocessing.
- Experiments `baseline` (5 features), `no_gradient` (4), `baseline_t2` (7) and an old-style config with the booleans all ran through the runner, wrote `features.csv`, and were aggregated.
- `pipeline.py` runs unchanged with its booleans.
- The cache was tested with a deliberately slow test feature: the second run loaded it from disk, the result was identical, and a different pre-processing setting got its own cache folder.

---

## 0. First, three small things in your Phase 1 code

1. **Remove the debug line** in `pre_process` (`pipeline_utilities.py`, just before `# execute pipeline on the T2w image`):
   ```python
   sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], 'check_T1.nii.gz')
   ```
   It writes a file into the working directory for every subject, and with multiprocessing several processes write the same file at once.
2. **Notes live in the repo now:** `mialab_notes/` inside `mialab-group03` is the one shared copy, committed so everyone gets it with a pull. Claude only writes inside that folder; check `git status` before committing.
3. Optional, from phase1_code.md: `class ImageNormalization(pymia_fltr.Filter, abc.ABC)` with `@abc.abstractmethod` on `normalize`, plus the `ValueError` when the mask is missing. Without `abc.ABC`, forgetting `normalize` in a new subclass only fails at run time. Without the mask check, a missing mask fails with a less clear error message.

---

## 1. The idea (2.1)

**Feature vs. column.** A *feature* is one computation that produces one feature image, e.g. the T1 gradient magnitude, or the atlas coordinates. A feature image can have several components: the coordinates have 3 (x, y, z), and a PyRadiomics GLCM run will have ~24, one per texture measure. Each component becomes one **column** of the feature matrix, with its own name.

**Tags and names.** Every feature has:

| Tag | Meaning | Examples |
|---|---|---|
| `family` | what kind of information | `spatial`, `intensity`, `gradient`, `firstorder`, `texture` |
| `modality` | input image | `T1`, `T2`, none (coordinates) |
| `measure` | what within the family (optional) | `coord`, `magnitude`, `gaussian`, `glcm` |
| `scale` | neighborhood size (optional) | `s2` (σ = 2 mm), `r1` (kernel radius 1) |

The column name joins them: `modality_family_measure_component_scale`, leaving out what's empty. Today's pool:

| Column | family | modality |
|---|---|---|
| `spatial_coord_x`, `_y`, `_z` | spatial | none |
| `T1_intensity` | intensity | T1 |
| `T1_gradient_magnitude` | gradient | T1 |
| `T2_intensity` | intensity | T2 |
| `T2_gradient_magnitude` | gradient | T2 |

These are the names you'll see in `feature_importance.csv`, SHAP plots and Boruta output. The tags are what you'll sum SHAP values over ("how important is the texture family for the amygdala?"), so every run writes them to `features.csv`.

**Why this design.**
- **The pipeline's style is kept.** The computation itself is still a pymia `Filter` (`AtlasCoordinates`, and later `NeighborhoodFeatureExtractor` or a PyRadiomics filter). A `Feature` only says which image goes in, which filter to apply, and how to tag the result.
- **One place to add features.** New features are one `register(...)` line in `feature_pool.py`. `FeatureExtractor`, the runner and the analysis code don't change. That keeps feature branches from conflicting with each other.
- **Selection works on columns.** Phase 5 needs column subsets: the Boruta subset, the top-k features per structure, drop-one-family ablations. A feature is computed if at least one of its columns is selected, and only the selected columns go into the matrix.
- **Old configs keep working.** `pipeline.py` with `coordinates_feature` etc. still works unchanged. Giving both `features` and the old flags is an error, so a config can't silently mean something else.

---

## 2. Selecting features

In `pre_process` parameters (or the `pre_process` section of an experiment JSON):

```json
"pre_process": {"features": "baseline", "exclude_features": ["family=gradient"]}
```

`features` and `exclude_features` take a string or a list. Each item can be:

| Item | Selects | Example |
|---|---|---|
| set name from `FEATURE_SETS` | the set's content | `"baseline"`, `"baseline_t2"`, `"all"` |
| feature name | all columns of that feature | `"spatial_coord"` (3 columns) |
| column name | one column | `"spatial_coord_z"` |
| glob pattern on names | matching columns | `"T2_*"`, `"*_r1"` |
| tag selector `tag=value` | columns with that tag (globs allowed) | `"family=texture"`, `"modality=T2"`, `"scale=s*"` |

Columns are always in pool order, whatever order you list them in, so the same selection always gives the same matrix. An item that matches nothing raises an error listing the known features (typos don't pass silently).

Examples for later experiments:

| Experiment | `features` | `exclude_features` |
|---|---|---|
| provided pipeline | `"baseline"` | |
| + T2 (task 3.1) | `"baseline_t2"` | |
| full pool | `"all"` | |
| drop one family (5.5) | `"all"` | `["family=texture"]` |
| drop one modality (5.5) | `"all"` | `["modality=T2"]` |
| only one scale | `"all"` | `["scale=s4", "scale=r2"]` |
| Boruta subset (5.2) | list of column names from the Boruta output | |

---

## 3. The code

### 3.1 New file `mialab/filtering/feature_pool.py`

Copy as is (also available as [phase2_code/feature_pool.py](phase2_code/feature_pool.py)).

```python
"""The feature pool: every feature the pipeline can compute, with tags, selection and an on-disk cache.

A :class:`Feature` describes *how* one feature image is computed (which input image, which filter) and *what* it
is (family, modality, scale). Each component of the feature image becomes one column of the feature matrix.
Column names are built from the tags, e.g. ``T1_gradient_magnitude`` or ``T2_texture_glcm_Contrast_r1``.

Experiments select columns with a feature specification (see :func:`select_columns`), e.g.::

    'baseline'                              a named set from FEATURE_SETS
    ['baseline', 'T2_intensity']            sets, feature names and column names can be mixed
    ['all'], exclude=['family=gradient']    tag selectors: family=..., modality=..., scale=...
    ['T1_*']                                glob patterns on feature and column names
"""
import fnmatch
import hashlib
import json
import os
import typing as t

import pymia.filtering.filter as fltr
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat

MODALITIES = {'T1': structure.BrainImageTypes.T1w,
              'T2': structure.BrainImageTypes.T2w}

TAGS = ('family', 'modality', 'scale')


class Feature:
    """Represents one entry of the feature pool."""

    def __init__(self, family: str, filter_: fltr.Filter = None, modality: str = None, measure: str = None,
                 scale: str = None, components: t.List[str] = None,
                 params: t.Callable[[structure.BrainImage], fltr.FilterParams] = None, cache: bool = False):
        """Initializes a new instance of the Feature class.

        Args:
            family (str): The feature family, e.g. 'spatial', 'intensity', 'gradient', 'firstorder', 'texture'.
            filter_ (fltr.Filter): The filter computing the feature image from the input image.
                None means the input image itself is the feature.
            modality (str): The input image, 'T1' or 'T2'. None for features that only need the image geometry
                (e.g. atlas coordinates); they get the T1w image.
            measure (str): What is measured within the family, e.g. 'magnitude' for the gradient.
            scale (str): The scale, e.g. 's2' for sigma = 2 mm or 'r1' for a kernel radius of 1 voxel.
            components (list of str): Names of the components if the filter returns a vector image,
                e.g. ['x', 'y', 'z']. None for scalar images.
            params: A function returning the filter parameters for an image, e.g.
                ``lambda img: MyParams(img.images[structure.BrainImageTypes.BrainMask])``. None if not needed.
            cache (bool): Whether to store the feature image in the feature cache. Use it for slow features only;
                cheap ones (intensity, gradient, Gaussian) are faster to recompute than to load.
        """
        if modality is not None and modality not in MODALITIES:
            raise ValueError(f'unknown modality {modality}, use one of {list(MODALITIES)}')
        self.family = family
        self.filter = filter_
        self.modality = modality
        self.measure = measure
        self.scale = scale
        self.components = components
        self.params = params
        self.cache = cache

    @property
    def name(self) -> str:
        """The unique name of the feature, e.g. 'T1_gradient_magnitude'."""
        return _join(self.modality, self.family, self.measure, self.scale)

    @property
    def columns(self) -> t.List[dict]:
        """One entry per column of the feature matrix, with the column name and its tags."""
        components = self.components or [None]
        return [{'name': _join(self.modality, self.family, self.measure, component, self.scale),
                 'feature': self.name, 'component': index,
                 'family': self.family, 'modality': self.modality or 'none', 'scale': self.scale or 'none'}
                for index, component in enumerate(components)]

    def compute(self, img: structure.BrainImage) -> sitk.Image:
        """Computes the feature image of a pre-processed brain image.

        Args:
            img (structure.BrainImage): The (registered, pre-processed) brain image.

        Returns:
            sitk.Image: The feature image (as many components as the feature has columns).
        """
        image = img.images[MODALITIES[self.modality or 'T1']]
        if self.filter is None:
            feature_image = sitk.Cast(image, sitk.sitkFloat32)
        else:
            params = self.params(img) if self.params is not None else None
            feature_image = self.filter.execute(image, params)

        if feature_image.GetNumberOfComponentsPerPixel() != len(self.columns):
            raise ValueError(f'feature {self.name}: filter returned {feature_image.GetNumberOfComponentsPerPixel()} '
                             f'components, but {len(self.columns)} were declared')
        return feature_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'Feature:\n name: {self.name}\n filter: {self.filter}\n'.format(self=self)


def _join(*parts) -> str:
    return '_'.join(str(p) for p in parts if p is not None)


# --- the pool ------------------------------------------------------------------------------------------------------
FEATURES: t.Dict[str, Feature] = {}  # feature name -> Feature, in the order the columns appear in the feature matrix


def register(feature: Feature):
    """Adds a feature to the pool."""
    if feature.name in FEATURES:
        raise ValueError(f'feature {feature.name} is already registered')
    FEATURES[feature.name] = feature


register(Feature('spatial', fltr_feat.AtlasCoordinates(), measure='coord', components=['x', 'y', 'z']))
for _modality in MODALITIES:
    register(Feature('intensity', modality=_modality))
    register(Feature('gradient', fltr_feat.SimpleITKFunction(sitk.GradientMagnitude), modality=_modality,
                     measure='magnitude'))

# Phase 3: add new features here, e.g.
# for _modality in MODALITIES:
#     for _sigma in (1, 2, 4):
#         register(Feature('intensity', fltr_feat.SimpleITKFunction(sitk.SmoothingRecursiveGaussian, sigma=_sigma),
#                          modality=_modality, measure='gaussian', scale=f's{_sigma}'))

# named feature specifications; a set can use other sets
FEATURE_SETS = {
    'baseline': ['spatial_coord', 'T1_intensity', 'T1_gradient_magnitude'],  # the features of the provided pipeline
    'baseline_t2': ['baseline', 'T2_intensity', 'T2_gradient_magnitude'],
    'all': ['*'],
}

# the boolean flags of the provided pipeline, still accepted by FeatureExtractor
LEGACY_FLAGS = {'coordinates_feature': 'spatial_coord',
                'intensity_feature': 'T1_intensity',
                'gradient_intensity_feature': 'T1_gradient_magnitude'}


# --- selection -----------------------------------------------------------------------------------------------------
def all_columns() -> t.List[dict]:
    """All columns of the pool, in pool order."""
    return [column for feature in FEATURES.values() for column in feature.columns]


def _matches(item: str, column: dict) -> bool:
    if '=' in item:
        tag, value = item.split('=', 1)
        if tag not in TAGS:
            raise ValueError(f'unknown tag "{tag}" in "{item}", use one of {TAGS}')
        return fnmatch.fnmatchcase(column[tag], value)
    return fnmatch.fnmatchcase(column['name'], item) or fnmatch.fnmatchcase(column['feature'], item)


def _expand(spec, seen=()) -> t.List[str]:
    """Replaces set names by their content (recursively)."""
    items = [spec] if isinstance(spec, str) else list(spec)
    expanded = []
    for item in items:
        if item in FEATURE_SETS:
            if item in seen:
                raise ValueError(f'feature set {item} contains itself')
            expanded += _expand(FEATURE_SETS[item], seen + (item,))
        else:
            expanded.append(item)
    return expanded


def select_columns(spec, exclude=None) -> t.List[dict]:
    """Selects columns of the feature pool.

    Args:
        spec (str or list of str): Set names, feature names, column names, glob patterns or tag selectors
            ('family=texture', 'modality=T2', 'scale=r1').
        exclude (str or list of str): Same syntax; matching columns are removed afterwards.

    Returns:
        list of dict: The selected columns (see :attr:`Feature.columns`), always in pool order.

    Raises:
        ValueError: If an item matches no column or nothing is selected.
    """
    columns = all_columns()
    include = _expand(spec)
    exclude = _expand(exclude) if exclude else []
    for item in include + exclude:
        if not any(_matches(item, column) for column in columns):
            raise ValueError(f'"{item}" matches no feature. Known features: {list(FEATURES)}, '
                             f'sets: {list(FEATURE_SETS)}')
    selected = [c for c in columns if any(_matches(i, c) for i in include)
                and not any(_matches(e, c) for e in exclude)]
    if not selected:
        raise ValueError(f'no feature selected by {spec} (exclude={exclude})')
    return selected


def feature_spec_from_kwargs(kwargs: dict):
    """Gets the feature specification from the pre-processing parameters.

    Either the new ``features`` (and optional ``exclude_features``) or the old boolean flags
    (``coordinates_feature``, ``intensity_feature``, ``gradient_intensity_feature``) can be used, not both.
    """
    used_flags = [flag for flag in LEGACY_FLAGS if flag in kwargs]
    if 'features' in kwargs:
        if used_flags:
            raise ValueError(f'use either "features" or the old flags {used_flags}, not both')
        return kwargs['features'], kwargs.get('exclude_features')
    return [LEGACY_FLAGS[flag] for flag in used_flags if kwargs[flag]], kwargs.get('exclude_features')


# --- cache ---------------------------------------------------------------------------------------------------------
CACHE_VERSION = 1  # increase to invalidate all caches, e.g. after a change in registration or normalization code

# pre-processing parameters that change the images the features are computed from
CACHE_RELEVANT_PARAMS = ('registration_pre', 'skullstrip_pre', 'normalization_pre')


class FeatureCache:
    """Stores feature images on disk, so slow features are computed only once per subject.

    Layout: ``<directory>/<pre-processing signature>/<subject>/<feature name>.mha``. The signature is a short hash
    of the pre-processing parameters, so different pre-processing settings never share cached features.
    ``<directory>/<signature>/preprocessing.json`` lists the parameters behind a signature.

    The cache does not notice changes in the code of a feature: delete the feature's files (or the whole folder)
    after changing how it is computed.
    """

    def __init__(self, directory: str, pre_process_params: dict):
        """Initializes a new instance of the FeatureCache class.

        Args:
            directory (str): The cache root directory, e.g. 'feature_cache'.
            pre_process_params (dict): The pre-processing parameters (only CACHE_RELEVANT_PARAMS are used).
        """
        relevant = {key: pre_process_params.get(key, False) for key in CACHE_RELEVANT_PARAMS}
        relevant['cache_version'] = CACHE_VERSION
        self.description = json.dumps(relevant, sort_keys=True)
        self.signature = hashlib.sha1(self.description.encode()).hexdigest()[:10]
        self.directory = os.path.join(directory, self.signature)

    def path(self, subject: str, feature: Feature) -> str:
        return os.path.join(self.directory, subject, feature.name + '.mha')

    def load(self, subject: str, feature: Feature) -> t.Optional[sitk.Image]:
        """Loads a cached feature image, or returns None if it is not cached yet."""
        path = self.path(subject, feature)
        return sitk.ReadImage(path) if os.path.exists(path) else None

    def save(self, subject: str, feature: Feature, image: sitk.Image):
        """Saves a feature image (written to a temporary file first, so a crash never leaves a broken file)."""
        path = self.path(subject, feature)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        info_file = os.path.join(self.directory, 'preprocessing.json')
        if not os.path.exists(info_file):
            with open(info_file, 'w') as f:
                f.write(self.description)
        tmp_path = path[:-len('.mha')] + '.tmp.mha'
        sitk.WriteImage(image, tmp_path, True)
        os.replace(tmp_path, path)
```

### 3.2 `mialab/filtering/feature_extraction.py`: add `SimpleITKFunction`

Paste this class after `class AtlasCoordinates` (before `def first_order_texture_features_function`). It turns any SimpleITK function into a pymia filter, so most Phase 3 filters need no class of their own.

```python
class SimpleITKFunction(fltr.Filter):
    """Wraps a SimpleITK function (e.g. ``sitk.GradientMagnitude``) as a filter, so it can be used in the feature pool.

    Examples:
        >>> SimpleITKFunction(sitk.GradientMagnitude)
        >>> SimpleITKFunction(sitk.SmoothingRecursiveGaussian, sigma=2.0)
    """

    def __init__(self, function_, **kwargs):
        """Initializes a new instance of the SimpleITKFunction class.

        Args:
            function_: A SimpleITK function taking an image as first argument and returning an image.
            kwargs: Further arguments passed to the function.
        """
        super().__init__()
        self.function = function_
        self.kwargs = kwargs

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes the function on an image (as float32, so integer inputs give float outputs)."""
        return self.function(sitk.Cast(image, sitk.sitkFloat32), **self.kwargs)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SimpleITKFunction:\n function: {self.function.__name__}\n kwargs: {self.kwargs}\n' \
            .format(self=self)
```

### 3.3 `mialab/utilities/pipeline_utilities.py`

**Imports:** remove `import enum` and `import warnings` (no longer used), and add the pool:

```python
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.feature_pool as fpool
import mialab.filtering.postprocessing as fltr_postp
```

**Delete the class `FeatureImageTypes`** (the pool replaces it; nothing else uses it).

**Replace `class FeatureExtractor`** from its first line down to just before `@staticmethod def _image_as_numpy_array` (which stays unchanged):

```python
class FeatureExtractor:
    """Represents a feature extractor.

    Computes the selected columns of the feature pool (see :mod:`mialab.filtering.feature_pool`) and builds
    the feature matrix. The column names are stored in ``img.feature_names``.
    """

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
            kwargs: The pre-processing parameters. Used here:
                ``training`` (bool): sample training voxels (True) or use all voxels (False).
                ``features``, ``exclude_features``: the feature specification, see ``feature_pool.select_columns``.
                ``coordinates_feature``, ``intensity_feature``, ``gradient_intensity_feature``: the old flags,
                still accepted if ``features`` is not given.
                ``feature_cache_dir`` (str): directory of the feature cache; None (default) disables it.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        spec, exclude = fpool.feature_spec_from_kwargs(kwargs)
        self.columns = fpool.select_columns(spec, exclude)
        cache_dir = kwargs.get('feature_cache_dir', None)
        self.cache = fpool.FeatureCache(cache_dir, kwargs) if cache_dir else None

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # compute every feature that has at least one selected column, in pool order
        for name in dict.fromkeys(column['feature'] for column in self.columns):
            self.img.feature_images[name] = self._compute(fpool.FEATURES[name])

        self._generate_feature_matrix()

        return self.img

    def _compute(self, feature: fpool.Feature) -> sitk.Image:
        """Computes a feature image, or loads it from the cache if the feature is cacheable and cached."""
        if self.cache is None or not feature.cache:
            return feature.compute(self.img)
        image = self.cache.load(self.img.id_, feature)
        if image is None:
            image = feature.compute(self.img)
            self.cache.save(self.img.id_, feature, image)
        return image

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features: per feature image, keep only the selected components
        data = []
        for name, image in self.img.feature_images.items():
            components = [column['component'] for column in self.columns if column['feature'] == name]
            data.append(self._image_as_numpy_array(image, mask)[:, components])
        data = np.concatenate(data, axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))
        self.img.feature_names = [column['name'] for column in self.columns]
```

What changed:
- `__init__` turns the parameters into the list of selected columns (old flags still accepted) and opens the cache if `feature_cache_dir` is given.
- `execute` computes each needed feature once, in pool order. The `# todo: add T2w features` warning is gone, since T2 features are now in the pool.
- `_compute` adds the cache: load if cached, otherwise compute and save. Only features marked `cache=True` use it.
- `_generate_feature_matrix`: the sampling part is untouched. The matrix keeps only the selected components of each feature image, and the column names go to `img.feature_names`.

### 3.4 `mialab/data/structure.py`

Add one line at the end of `BrainImage.__init__`:

```python
        self.feature_names = []  # the name of each column of the feature matrix
```

### 3.5 `mialab/utilities/multi_processor.py`

With `multi_process=True` the images are converted to a picklable copy and back, which would drop the names. Three lines:

```python
# in PicklableBrainImage.__init__, after self.feature_matrix = None ...
        self.feature_names = []

# in BrainImageToPicklableBridge.convert, after pickable_brain_image.feature_matrix = ...
        pickable_brain_image.feature_names = brain_image.feature_names

# in PicklableToBrainImageBridge.convert, after brain_image.feature_matrix = ...
        brain_image.feature_names = picklable_brain_image.feature_names
```

### 3.6 `pipeline.py`

Nothing to change: it uses the old flags, which give exactly the baseline. If you want the new form there too, replace the three flags with `'features': 'baseline'`.

---

## 4. The feature cache (2.2)

**When it's used.** Only for features registered with `cache=True`, and only if `feature_cache_dir` is set (the runner sets it to `feature_cache/` next to the script; `--feature_cache none` disables it; `pipeline.py` doesn't use it). Today's features are all `cache=False`. Measured on one subject: the gradient takes 0.14 s and a σ = 2 Gaussian 0.6 s to compute, while loading a cached map takes 0.1–0.2 s. Recomputing is about as fast and needs no disk space. Use `cache=True` for PyRadiomics and the neighborhood features (Phase 3), which take minutes per subject.

**Layout.**
```
feature_cache/
└── e90191a3e4/                      one folder per pre-processing setting
    ├── preprocessing.json           {"registration_pre": true, "skullstrip_pre": true, "normalization_pre": true, ...}
    └── 117122/
        └── T1_texture_glcm_r1.mha   one file per subject and feature
```

The folder name is a hash of the pre-processing settings that change the images (`registration_pre`, `skullstrip_pre`, `normalization_pre`). Features computed with z-score normalization therefore never get mixed with those from histogram matching. `preprocessing.json` tells you which folder is which.

**Correctness.**
- Features are computed from the pre-processed images before the training voxels are sampled. They don't depend on the seed or on training vs. test, so caching them is safe.
- A file is written under a temporary name and renamed when complete, so a crash never leaves a half-written file that would be loaded later.
- **The cache does not notice code changes.** If you change how a cached feature is computed, delete its files (e.g. all `T1_texture_glcm_r1.mha`) or the whole `feature_cache/`. If you change the registration or normalization *code*, increase `CACHE_VERSION` in `feature_pool.py`.

**Disk space.** One map is the size of the atlas grid (197 × 233 × 189 voxels, float32 ≈ 35 MB raw). Compression helps a lot when everything outside the brain is 0, which is the case for masked PyRadiomics output (≈ 7 MB for the gradient). For 30 subjects, plan roughly 0.2–1 GB per cached feature. A GLCM run with ~24 components counts as 24 features. `feature_cache/` must be in `.gitignore`.

---

## 5. Experiment runner (2.3) and aggregation (2.4)

Copy into the **repo root** (next to `pipeline.py`) from [experiment_runner/](experiment_runner/):
- `run_experiment.py` (updated today)
- `aggregate_results.py` (unchanged)
- the folder `experiments/` with `baseline.json`, `no_gradient.json`, `baseline_t2.json`

Changes in `run_experiment.py`:
- The monkeypatch that recorded feature names is gone; the names now come from `img.feature_names`.
- The default `pre_process` config is `"features": "baseline"`. Old-style configs that use the three booleans still work: they replace the default selection.
- The feature selection is checked and printed before the slow part starts, e.g. `4 features: spatial_coord_x, spatial_coord_y, spatial_coord_z, T1_intensity`.
- New option `--feature_cache DIR` (default `feature_cache/`, `none` to disable).
- New output file per run, `features.csv`, with one row per column in matrix order:
  ```
  name,feature,family,modality,scale
  spatial_coord_x,spatial_coord,spatial,none,none
  T1_intensity,T1_intensity,intensity,T1,none
  T1_gradient_magnitude,T1_gradient_magnitude,gradient,T1,none
  ```
  Merge it with `feature_importance.csv` (and later SHAP values) on `name` to sum importance per family or modality.

[experiment_runner.md](experiment_runner/experiment_runner.md) describes all output files and is updated accordingly.

---

## 6. `.gitignore`

Add:

```
# pipeline output and caches
feature_cache/
mia-result/
results/**/segmentations/
results/**/*.tmp/
results/_aggregate/
*.mha
```

`results/<experiment>/seed_<n>/` with its small CSV/JSON files stays committed, so everyone has all results after a pull.

---

## 7. Check that it works

```
python run_experiment.py experiments/baseline.json --seeds 0
python run_experiment.py experiments/no_gradient.json --seeds 0
python aggregate_results.py --baseline baseline --plot
```

Expected: the runner prints `5 features: ...` and `4 features: ...` and writes `results/baseline/seed_0/` and `results/no_gradient/seed_0/` with `features.csv`; the aggregation prints a table with both experiments. A full run (20 training, 10 test subjects) takes a while, mostly pre-processing.

A quick check of the selection syntax without running anything:

```
python -c "import mialab.filtering.feature_pool as p; print([c['name'] for c in p.select_columns('all', ['modality=T1'])])"
```

prints `['spatial_coord_x', 'spatial_coord_y', 'spatial_coord_z', 'T2_intensity', 'T2_gradient_magnitude']`.

---

## 8. Adding a feature (preview of Phase 3)

Example for task 3.3, multi-scale Gaussian and Laplacian of Gaussian for T1 and T2. These are the only lines needed, in `feature_pool.py` below the existing `register` calls:

```python
for _modality in MODALITIES:
    for _sigma in (1, 2, 4):
        register(Feature('intensity', fltr_feat.SimpleITKFunction(sitk.SmoothingRecursiveGaussian, sigma=_sigma),
                         modality=_modality, measure='gaussian', scale=f's{_sigma}'))
        register(Feature('gradient', fltr_feat.SimpleITKFunction(sitk.LaplacianRecursiveGaussian, sigma=_sigma),
                         modality=_modality, measure='log', scale=f's{_sigma}'))
```

This adds `T1_intensity_gaussian_s1` … `T2_gradient_log_s4` (12 columns), selectable with patterns like `"*_gaussian_*"` or tags like `"scale=s2"`. A feature set for them goes into `FEATURE_SETS`, e.g. `'multiscale': ['baseline_t2', '*_gaussian_*', '*_log_*']`.

For a filter that needs more than the image (e.g. PyRadiomics needs the brain mask), pass `params`:

```python
register(Feature('texture', MyRadiomicsFilter('glcm', kernel_radius=1), modality='T1', measure='glcm', scale='r1',
                 components=MyRadiomicsFilter.GLCM_NAMES,
                 params=lambda img: MyRadiomicsParams(img.images[structure.BrainImageTypes.BrainMask]),
                 cache=True))
```

`components` lists the names of the output components; `Feature.compute` checks that the filter returns that many, so a mismatch fails immediately instead of shifting column names.

---

## 9. What's next

Remaining Phase 1 tasks, now that the runner exists:
- **1.5 plots:** done by `aggregate_results.py --plot` (boxplots per structure for Dice and HD95).
- **1.6 seeds:** done by `run_experiment.py --seeds 0 1 2`.
- **1.7 freeze the forest:** a few configs with the baseline features and different forests, 3 seeds each, then pick the smallest forest close to the best one. For example `experiments/rf_100_d25.json`:
  ```json
  {"description": "forest tuning", "forest": {"n_estimators": 100, "max_depth": 25}}
  ```
  Worth comparing: `n_estimators` 50 / 100, `max_depth` 15 / 25 / null, and `class_weight` null / `"balanced"`. Then copy the winning `forest` block into `baseline.json` and every later experiment.

After that, Phase 3 is one feature branch per feature group (T2, neighborhood, multi-scale, PyRadiomics), each adding `register(...)` lines and a feature set. Merge them one at a time; since they all only append to the pool, conflicts are small.

One thing to keep in mind for Phase 3: at test time the matrix covers every voxel of the atlas grid (8.7 million rows), so with 100 features one test subject needs about 3.5 GB of memory. If that becomes a problem, the fix is to predict only inside the brain mask (1.8 million voxels) and set everything outside to background. The ground truth has no labels outside the mask (checked on a test subject).
