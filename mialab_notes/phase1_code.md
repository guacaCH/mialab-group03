# Phase 1 code: tasks 1.1 to 1.4

Replacement code for the four `# todo`s from [todo.md](todo.md) Phase 1. Most blocks replace the **body of one method** in the repo (1.3 replaces a whole class); copy them in by hand.

**Tested:** these exact changes were run on a copy of the repo, training on 2 subjects (100307, 100408) and testing on 117122, with 10 trees and depth 15 for the test run only. The pipeline ran end to end and reported Dice and HD95 for all five structures (e.g. WM Dice 0.81 / HD95 4.1 mm, amygdala Dice 0.37 / HD95 18.4 mm). The registration direction was checked separately: resampling subject 100307's brain mask with the provided transform overlaps the atlas brain mask with Dice 0.95, while the inverse transform gives only 0.65, so the transform is used as given.

`pipeline.py` already sets `registration_pre`, `skullstrip_pre` and `normalization_pre` to `True` (lines 61-63); nothing to change there.

---

## File 1: `mialab/filtering/preprocessing.py`

### Imports (top of file)

Add `abc` and numpy:

```python
import abc
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
```

### 1.3 Normalization: exchangeable normalization filters

**Replace the whole `ImageNormalization` class** (from `class ImageNormalization` down to just before `class SkullStrippingParameters`) with the code below.

**Design.** It follows the pipeline's existing pattern: every processing step is a `pymia` `Filter` with its image-specific inputs in a `FilterParams` object (like `SkullStripping` + `SkullStrippingParameters`).
- `ImageNormalization` becomes an **abstract base class**. It does the shared work: numpy conversion, selecting the brain voxels from the mask, and handling the background. Each subclass only implements `normalize(values, brain_values, params)`: compute the statistics from the brain voxels, then apply the mapping to the whole image.
- **Subclasses:**
  - `ZScoreNormalization`
  - `MinMaxNormalization`
  - `PercentileNormalization(lower, upper)`: robust min-max with clipping
  - `HistogramMatching(landmarks)`: maps the brain's intensity percentiles onto the atlas brain's percentiles, with linear interpolation in between (piecewise-linear landmark matching after Nyúl et al.)
- `NormalizationParameters(img_mask, skull_stripped, reference, reference_mask)` carries the brain mask, whether the image was skull-stripped, and the reference image with its mask (for histogram matching).
- `create_normalization()` is a small factory. It turns the config value into a filter object, so `pre_process` doesn't need to know which classes exist.
- **Adding a method** means writing one subclass with `normalize()` and adding it to `NORMALIZATIONS`. Nothing else changes, which keeps feature branches from conflicting.

**Brain mask and skull stripping:**
- **The brain mask is required.** The filter raises an error without one. The statistics always come from the brain voxels only. Without the mask, the zeros of a skull-stripped background, or the skull of a non-stripped image, would distort the mean and std.
- **With skull stripping** (`skull_stripped=True`): everything outside the mask is set to 0 after normalization, so the background stays 0.
- **Without skull stripping** (`skull_stripped=False`): the mapping is applied to the whole image and nothing is reset. Air and skull keep their place on the normalized scale instead of being set to 0, which after z-score would mean "average brain intensity".
- **The brain is normalized identically in both modes**, so an experiment with skull stripping on vs. off compares only the effect of skull stripping.
- **Histogram matching** uses the brain of the atlas (the atlas mask) as reference, so the skull in the atlas doesn't affect it.

```python
class NormalizationParameters(pymia_fltr.FilterParams):
    """Normalization parameters."""

    def __init__(self, img_mask: sitk.Image, skull_stripped: bool, reference: sitk.Image = None,
                 reference_mask: sitk.Image = None):
        """Initializes a new instance of the NormalizationParameters class.

        Args:
            img_mask (sitk.Image): The brain mask on the same grid as the image. The normalization statistics are
                computed from the voxels inside the mask only.
            skull_stripped (bool): Whether the image was skull-stripped. If True, all voxels outside the mask are
                set to 0 after normalization. If False, the normalization is applied to the whole image.
            reference (sitk.Image): A reference image (e.g. the atlas); only needed by HistogramMatching.
            reference_mask (sitk.Image): The brain mask of the reference image (e.g. the atlas mask).
        """
        self.img_mask = img_mask
        self.skull_stripped = skull_stripped
        self.reference = reference
        self.reference_mask = reference_mask


class ImageNormalization(pymia_fltr.Filter, abc.ABC):
    """Base class of all intensity normalization filters.

    The base class handles the numpy conversion, the brain mask and the background. Subclasses only implement
    :meth:`normalize`: compute the statistics from the brain voxels and apply the mapping to all voxels.
    """

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: NormalizationParameters = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (NormalizationParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        if params is None or params.img_mask is None:
            raise ValueError(f'{self.__class__.__name__} needs a brain mask (NormalizationParameters.img_mask)')

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        brain = sitk.GetArrayFromImage(params.img_mask) > 0

        img_out_arr = self.normalize(img_arr, img_arr[brain], params).astype(np.float32)
        if params.skull_stripped:
            img_out_arr[~brain] = 0  # keep the background of a skull-stripped image at 0

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)
        return img_out

    @abc.abstractmethod
    def normalize(self, values: np.ndarray, brain_values: np.ndarray,
                  params: NormalizationParameters) -> np.ndarray:
        """Normalizes intensities.

        Args:
            values (np.ndarray): The intensities to normalize (the whole image).
            brain_values (np.ndarray): 1-D array with the intensities of the brain voxels, used for the statistics.
            params (NormalizationParameters): The parameters (e.g. for a reference image).

        Returns:
            np.ndarray: The normalized intensities, same shape as values.
        """
        raise NotImplementedError()

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return '{self.__class__.__name__}:\n'.format(self=self)


class ZScoreNormalization(ImageNormalization):
    """Z-score normalization: zero mean and unit standard deviation inside the brain."""

    def normalize(self, values, brain_values, params):
        return (values - brain_values.mean()) / brain_values.std()


class MinMaxNormalization(ImageNormalization):
    """Min-max normalization: maps the brain intensities to [0, 1]."""

    def normalize(self, values, brain_values, params):
        return (values - brain_values.min()) / (brain_values.max() - brain_values.min())


class PercentileNormalization(ImageNormalization):
    """Robust min-max normalization: maps the lower/upper brain percentiles to [0, 1] and clips outliers."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        """Initializes a new instance of the PercentileNormalization class.

        Args:
            lower (float): The percentile mapped to 0.
            upper (float): The percentile mapped to 1.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper

    def normalize(self, values, brain_values, params):
        low, high = np.percentile(brain_values, [self.lower, self.upper])
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    def __str__(self):
        return 'PercentileNormalization:\n lower: {self.lower}\n upper: {self.upper}\n'.format(self=self)


class HistogramMatching(ImageNormalization):
    """Histogram matching to a reference image (piecewise-linear landmark matching, Nyul et al.).

    The percentiles of the brain intensities are mapped onto the same percentiles of the reference brain
    (e.g. the skull-stripped atlas); intensities in between are interpolated linearly.
    """

    def __init__(self, landmarks: int = 11):
        """Initializes a new instance of the HistogramMatching class.

        Args:
            landmarks (int): The number of percentiles from 1 to 99 that are matched.
        """
        super().__init__()
        self.landmarks = landmarks

    def normalize(self, values, brain_values, params):
        if params.reference is None:
            raise ValueError('HistogramMatching needs a reference image (NormalizationParameters.reference)')
        reference = sitk.GetArrayFromImage(params.reference).astype(np.float32)
        if params.reference_mask is not None:
            reference = reference[sitk.GetArrayFromImage(params.reference_mask) > 0]
        else:
            reference = reference[reference > 0]

        percentiles = np.concatenate([[0], np.linspace(1, 99, self.landmarks), [100]])
        source_landmarks = np.percentile(brain_values, percentiles)
        reference_landmarks = np.percentile(reference, percentiles)
        # values outside the brain range (e.g. skull without skull-stripping) are clamped to the reference range
        return np.interp(values, source_landmarks, reference_landmarks)

    def __str__(self):
        return 'HistogramMatching:\n landmarks: {self.landmarks}\n'.format(self=self)


NORMALIZATIONS = {'zscore': ZScoreNormalization,
                  'minmax': MinMaxNormalization,
                  'percentile': PercentileNormalization,
                  'histogram_matching': HistogramMatching}


def create_normalization(method) -> ImageNormalization:
    """Creates a normalization filter from a config value.

    Args:
        method: True (= 'zscore'), a name from NORMALIZATIONS, or a dict like
            {'method': 'percentile', 'lower': 0.5, 'upper': 99.5} whose other keys are passed to the constructor.

    Returns:
        ImageNormalization: The normalization filter.
    """
    if method is True:
        method = 'zscore'
    kwargs = {}
    if isinstance(method, dict):
        kwargs = {k: v for k, v in method.items() if k != 'method'}
        method = method['method']
    if method not in NORMALIZATIONS:
        raise ValueError(f'unknown normalization "{method}", choose from {sorted(NORMALIZATIONS)}')
    return NORMALIZATIONS[method](**kwargs)
```

#### And in `mialab/utilities/pipeline_utilities.py`

**1. Load the atlas mask.** At the top of the file, next to `atlas_t1` and `atlas_t2`:

```python
atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()
atlas_mask = sitk.Image()
```

and in `load_atlas_images`:

```python
    global atlas_t1
    global atlas_t2
    global atlas_mask
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    atlas_mask = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
```

**2. Use the new filters in `pre_process`.** Replace the normalization step of the T1 pipeline

```python
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageNormalization())
```

with

```python
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.create_normalization(kwargs['normalization_pre']))
        pipeline_t1.set_param(fltr_prep.NormalizationParameters(img.images[structure.BrainImageTypes.BrainMask],
                                                                 skull_stripped=kwargs.get('skullstrip_pre', False),
                                                                 reference=atlas_t1, reference_mask=atlas_mask),
                              len(pipeline_t1.filters) - 1)
```

and the same for the T2 pipeline, with `pipeline_t2` and `reference=atlas_t2`. The atlas mask is the same for T1 and T2.

#### Choosing the method

`normalization_pre` in `pipeline.py` (or in the `pre_process` section of an experiment config) now accepts:

| Value | Filter |
|---|---|
| `True` or `"zscore"` | `ZScoreNormalization` (default, existing configs keep working) |
| `"minmax"` | `MinMaxNormalization` |
| `"percentile"` or `{"method": "percentile", "lower": 0.5, "upper": 99.5}` | `PercentileNormalization` |
| `"histogram_matching"` or `{"method": "histogram_matching", "landmarks": 11}` | `HistogramMatching` against the atlas brain |
| `False` | no normalization |

**Tested** on test subject 117122 with registration on, each method with skull stripping on and off (T1 shown, T2 behaves the same):

| Method | Brain voxels (both modes) | Outside the brain, skull stripping on | Outside the brain, skull stripping off |
|---|---|---|---|
| z-score | mean 0.000, std 1.000 | all 0 | from -3.9 to 10.0 (air and skull on the brain's scale) |
| min-max | in [0, 1], mean 0.35 | all 0 | from -0.02 to 1.29 |
| percentile | in [0, 1], mean 0.60 | all 0 | clipped to [0, 1]; dark air becomes 0 |
| histogram matching | median 67.65, equal to the atlas brain median 67.65 | all 0 | from 10.6 to 96.4 (clamped to the atlas brain's range) |

The brain values are identical with and without skull stripping. Calling the filter without a brain mask raises a clear error.

**Remaining limitation:** without skull stripping, percentile normalization and histogram matching clamp the bright skull and fat to the brain's maximum, because their mapping is defined only on the brain's intensity range. The classifier only has to recognize these voxels as "not brain", so this should not matter.

### 1.2 Skull stripping: `SkullStripping.execute`

Replace the method body with:

```python
        mask = params.img_mask  # the brain mask

        # set all voxels outside the brain mask to 0
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        return sitk.Mask(image, mask, outsideValue=0)
```

`sitk.Mask` needs an integer mask; the provided brain mask is stored as float, hence the cast. The mask is registered before T1/T2 are processed (see `pre_process` in `pipeline_utilities.py`), so image and mask are on the same grid.

### 1.1 Registration: `ImageRegistration.execute`

Replace the method body with:

```python
        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # apply the provided affine transformation: resample the image onto the atlas grid.
        # labels and masks must keep their integer values, so they use nearest-neighbour interpolation;
        # intensity images use linear interpolation
        interpolator = sitk.sitkNearestNeighbor if is_ground_truth else sitk.sitkLinear
        return sitk.Resample(image, atlas, transform, interpolator, 0.0, image.GetPixelID())
```

What it does: `sitk.Resample` creates a new image with the atlas' size, spacing and orientation (197×233×189 instead of the native 181×217×181) and fills each voxel by looking up the corresponding position in the subject image through the affine transform. `is_ground_truth` is `True` for the labels and the brain mask (that's how `pre_process` calls it), so they get nearest-neighbour interpolation: linear interpolation would invent label values like 2.5 between grey matter (2) and hippocampus (3).

---

## File 2: `mialab/utilities/pipeline_utilities.py`

### 1.4 HD95: `init_evaluator`

Replace

```python
    metrics = [metric.DiceCoefficient()]
    # todo: add hausdorff distance, 95th percentile (see metric.HausdorffDistance)
    warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')
```

with

```python
    metrics = [metric.DiceCoefficient(),
               metric.HausdorffDistance(percentile=95, metric='HDRFDST95')]
```

`results.csv` then has an extra `HDRFDST95` column (in mm), and the console summary shows mean and std for both metrics.

---

## After pasting

1. Run `python pipeline.py`. The warnings "No registration / skull-stripping / normalization implemented" should be gone.
2. Optional sanity check: save a registered T1 (e.g. add `sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], 'check_T1.nii.gz')` temporarily in `pre_process`) and open it together with the atlas in ITK-SNAP; they should line up.
3. The forest is still 1 tree with depth 5, so results will be poor until task 1.7.
