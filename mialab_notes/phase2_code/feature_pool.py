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
