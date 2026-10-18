# Phase 1 code: tasks 1.1 to 1.4

Replacement code for the four `# todo`s from [todo.md](todo.md) Phase 1. Each block replaces the **body of one method** in the repo; copy them in by hand.

**Tested:** these exact changes were run on a copy of the repo, training on 2 subjects (100307, 100408) and testing on 117122, with 10 trees and depth 15 for the test run only. The pipeline ran end to end and reported Dice and HD95 for all five structures (e.g. WM Dice 0.81 / HD95 4.1 mm, amygdala Dice 0.37 / HD95 18.4 mm). The registration direction was checked separately: resampling subject 100307's brain mask with the provided transform overlaps the atlas brain mask with Dice 0.95, while the inverse transform gives only 0.65, so the transform is used as given.

`pipeline.py` already sets `registration_pre`, `skullstrip_pre` and `normalization_pre` to `True` (lines 61-63); nothing to change there.

---

## File 1: `mialab/filtering/preprocessing.py`

### Imports (top of file)

Add numpy:

```python
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
```

### 1.3 Normalization: `ImageNormalization.execute`

Replace the whole method body with:

```python
        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

        # z-score normalization using only the brain voxels (after skull-stripping, the background is 0)
        brain = img_arr != 0
        if not brain.any():
            brain = np.ones_like(img_arr, dtype=bool)
        mean = img_arr[brain].mean()
        std = img_arr[brain].std()
        img_arr[brain] = (img_arr[brain] - mean) / std

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out
```

Why only brain voxels: after skull stripping most of the image is 0. Including those zeros would make the mean and standard deviation depend on head size and field of view, not on tissue intensities. The background stays 0.

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
