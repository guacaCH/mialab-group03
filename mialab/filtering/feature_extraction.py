"""The feature extraction module contains classes for feature extraction."""
import sys

import numpy as np
import pymia.filtering.filter as fltr
import SimpleITK as sitk


class AtlasCoordinates(fltr.Filter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # create matrix with homogenous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # generate transformation matrix
        tmp_mat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmp_mat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'AtlasCoordinates:\n' \
            .format(self=self)

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

class ImageParams(fltr.FilterParams):
    """Filter parameters that carry one extra image, e.g. the second modality for ImageDifference."""

    def __init__(self, image: sitk.Image):
        """Initalizes a new instance of the ImageParams class.
        Args:
            image (sitk.Image): The extra image.
        """
        self.image = image


class ImageDifference(fltr.Filter):
    """Subtracts a second image from the input image, voxel by voxel: ``image - params.image``.
    
    Used for the T1 - T2 feature. After z-score normalization moth images have mean 0 and std 1 inside the brain.
    The difference combines the two contrasts into a single number (WM: high T1 - low T2 = large positive,
    GM: lower T1 - higher T2 = negative or arround 0, CSF: low T1 - high T2 = large negative).
    """
    def execute(self, image: sitk.Image, params: ImageParams = None) -> sitk.Image:
        """Computes difference image - params.image, both cast to float32."""        
        return sitk.Cast(image, sitk.sitkFloat32) - sitk.Cast(params.image, sitk.sitkFloat32)

    def __str__(self):
        return 'ImageDifference:\n'

class LabelPriorParams(fltr.FilterParams):
    """Parameters of the LabelPrior filter."""

    def __init__(self, counts: np.ndarray, subject_ids: set, subject_id: str, ground_truth: sitk.Image):
        """Initializes a new instance of the LabelPriorParams class.
        
        Args:
            counts (np.ndarray): Shape (n_labels, z, y, x). counts[i] is the number of training subjects
            that have label i + 1 at a voxel, in atlas space (seee feature_pool.build_label_prior).
            subject_ids (set): The ids of the training subjects counted in ``counts``.
            subject_id (str): The id of the subject the prior is computed for.
            ground_truth (sitk.Image): The registered ground truth of that subject (only used if the subject is one
            of the counted subjects, to remove it from its own prior)
        """
        self.counts = counts
        self.subject_ids =  subject_ids
        self.subject_id = subject_id
        self.ground_truth = ground_truth

class LabelPrior(fltr.Filter):
    """Probabilistic atlas for every voxel and label, the fraction of training subjects with that label at that voxel.
    
    Example: a value of 0.80 for a voxel in the hippocampus component means that this voxel has the hippocampus label
    in 16 of 20 training subjects (after registration to the atlas). This feature shows where a structure usually is
    and helps the forest to classify low contrast structures (hippocampus, amygdala).
    
    Leave one out: If a subject is one of the training subjects, it own label is removed from the counts. 
    Otherwise a training voxel would see its own label in the feature and the forest would trust the prior too much, 
    giving worse generalizability and worse test results than the training would suggest.
    """

    def execute(self, image: sitk.Image, params: LabelPriorParams = None) -> sitk.Image:
        """Returns a vector image with one component per label (values between 0 and 1).

        Args:
            image (sitk.Image): Any image in atlas space; only its geometry is used.
            params (LabelPriorParams): The counts and the subject.
        """

        counts = params.counts.astype(np.float32)
        n_subjects = len(params.subject_ids)
        if params.subject_id in params.subject_ids:
            ground_truth = sitk.GetArrayFromImage(params.ground_truth)
            counts -= np.stack([ground_truth == label for label in range(1, len(counts) +1)])
            n_subjects -= 1
        prior = np.moveaxis(counts / n_subjects, 0, -1) # (z, y, x, n_labels): one vector per voxel
        prior_image = sitk.GetImageFromArray(prior, isVector=True)
        prior_image.CopyInformation(image)
        return prior_image

    def __str__(self):
        return 'LabelPrior:\n'

class SeedParams(fltr.FilterParams):
    """Filter parameters with a random seed."""

    def __init__(self, seed: int):
        self.seed = seed

class RandomNoiseImage(fltr.Filter):
    """A feature of pure noise (standard normal values), as a control.

    A real feature that the forest uses less than this one carries no useful information. The noise has its own
    random generator (seeded per subject), so it does not change ``np.random``, which samples the training voxels:
    with or without this feature, exactly the same training voxels are used.
    """

    def  execute(self, image: sitk.Image, params: SeedParams = None) -> sitk.Image:
        """Returns an image with the geometry of ``image`` filled with N(0,1) noise."""
        rng = np.random.default_rng(params.seed)
        shape = image.GetSize()[::-1]   # np order [z, y, x]
        noise = sitk.GetImageFromArray(rng.standard_normal(shape).astype(np.float32))
        noise.CopyInformation(image)
        return noise

    def __str__(self):
        return 'RandomNoiseImage:\n'
    
    
# --- Phase 3b: further features (the 3 chosen from "More feature ideas" in the feature pool doc) -----------------------------
# All filters below take the pre-processed image in atlas space (1 mm voxels) and return an image with the same
# geometry. Where a mask is needed it comes in through ImageParams (the registered brain mask).


class SignedDistanceToMask(fltr.Filter):
    """Signed distance (mm) of every voxel to the border of a mask: positive inside, negative outside.

    Used with the brain mask: cortex (GM) lies within a few mm of the brain surface, WM and the thalamus lie deep
    inside. Unlike the atlas coordinates, this distance is measured in the subject's own brain, so it still works
    when the subject's brain is a bit smaller or larger than the atlas.
    """

    def execute(self, image: sitk.Image, params: ImageParams = None) -> sitk.Image:
        """Args: image: any image (only its geometry is used); params.image: the mask (non-zero = inside)."""
        mask = sitk.Cast(params.image > 0, sitk.sitkUInt8)
        distance = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True, squaredDistance=False,
                                                useImageSpacing=True)
        distance = sitk.Cast(distance, sitk.sitkFloat32)
        distance.CopyInformation(image)
        return distance

    def __str__(self):
        return 'SignedDistanceToMask:\n'


class PriorDistance(fltr.Filter):
    """Signed distance (mm) to the region where a label's prior is above ``threshold`` (negative inside).

    Example: the amygdala lies directly in front of the hippocampus. A voxel 3 mm in front of the typical
    hippocampus region has hippocampus distance +3 and amygdala distance around 0 or negative, which separates the
    two even where their intensities are the same. Uses the (leave-one-out) prior of LabelPrior.
    """

    def __init__(self, labels=(3, 4, 5), threshold: float = 0.5):
        """Args: labels: the labels (1 = WM ... 5 = thalamus) to compute a distance for; threshold: prior level."""
        super().__init__()
        self.labels = labels
        self.threshold = threshold

    def execute(self, image: sitk.Image, params: LabelPriorParams = None) -> sitk.Image:
        """Returns a vector image with one distance component per label in ``self.labels``."""
        prior = LabelPrior().execute(image, params)
        components = []
        for label in self.labels:
            region = sitk.Cast(sitk.VectorIndexSelectionCast(prior, label - 1, sitk.sitkFloat32) > self.threshold,
                               sitk.sitkUInt8)
            distance = sitk.SignedMaurerDistanceMap(region, insideIsPositive=False, squaredDistance=False,
                                                    useImageSpacing=True)
            components.append(sitk.Cast(distance, sitk.sitkFloat32))
        return sitk.Compose(components)

    def __str__(self):
        return 'PriorDistance:\n labels: {self.labels}\n threshold: {self.threshold}\n'.format(self=self)


class HemisphereAsymmetry(fltr.Filter):
    """Absolute difference between a voxel and its mirror voxel in the other hemisphere: |I(x) - I(-x)|.

    The images are registered to the symmetric MNI atlas (mni_icbm152_..._sym), whose midline is the plane x = 0
    in the middle of the image. Flipping the image along x therefore maps each voxel onto its mirror position.
    Healthy brains are nearly symmetric, so the value is small almost everywhere; it is large where the
    registration is off or where the two hemispheres really differ. Expect low importance (itself a finding).
    """

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Returns |image - image flipped along x|."""
        # check that the midline is in the middle of the image, otherwise flipping is not a mirror at x = 0
        center = [(size - 1) / 2 for size in image.GetSize()]
        center_x = image.TransformContinuousIndexToPhysicalPoint(center)[0]
        if abs(center_x) > image.GetSpacing()[0]:
            raise ValueError(f'image centre is at x = {center_x:.1f} mm, not on the midline x = 0')
        image = sitk.Cast(image, sitk.sitkFloat32)
        mirrored = sitk.Flip(image, [True, False, False])
        mirrored.CopyInformation(image)
        return sitk.Abs(image - mirrored)

    def __str__(self):
        return 'HemisphereAsymmetry:\n'


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features:

            - mean
            - variance
            - sigma
            - skewness
            - kurtosis
            - entropy
            - energy
            - snr
            - min
            - max
            - range
            - percentile10th
            - percentile25th
            - percentile50th
            - percentile75th
            - percentile90th
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    num_values = len(values)
    p = values / (np.sum(values) + eps)
    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(num_values * (num_values - 1)) / (num_values - 2) * np.sum((values - mean) ** 3) /
                     (num_values*std**3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (num_values * std ** 4 + eps),  # kurtosis
                     np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p**2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90)
                     ])


class NeighborhoodFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.neighborhood_radius = 3
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if np.isscalar(function_output):
            img_out = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        elif not isinstance(function_output, np.ndarray):
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.ndim > 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.shape[0] <= 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray with at least two elements')
        else:
            img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])

        img_out_arr = sitk.GetArrayFromImage(img_out)
        img_arr = sitk.GetArrayFromImage(image)
        z, y, x = img_arr.shape

        z_offset = self.kernel[2]
        y_offset = self.kernel[1]
        x_offset = self.kernel[0]
        pad = ((0, z_offset), (0, y_offset), (0, x_offset))
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')

        for xx in range(x):
            for yy in range(y):
                for zz in range(z):

                    val = self.function(img_arr_padded[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
                    img_out_arr[zz, yy, xx] = val

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'NeighborhoodFeatureExtractor:\n' \
            .format(self=self)


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator.

    A training mask is an image with intensity values 0 and 1, where 1 represents masked.
    Such a mask can be used to sample voxels for training.
    """

    @staticmethod
    def get_mask(ground_truth: sitk.Image,
                 ground_truth_labels: list,
                 label_percentages: list,
                 background_mask: sitk.Image = None) -> sitk.Image:
        """Gets a training mask.

        Args:
            ground_truth (sitk.Image): The ground truth image.
            ground_truth_labels (list of int): The ground truth labels,
                where 0=background, 1=label1, 2=label2, ..., e.g. [0, 1]
            label_percentages (list of float): The percentage of voxels of a corresponding label to extract as mask,
                e.g. [0.2, 0.2].
            background_mask (sitk.Image): A mask, where intensity 0 indicates voxels to exclude independent of the
            label.

        Returns:
            sitk.Image: The training mask.
        """

        # initialize mask
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)  # convert to float because of np.nan
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x = indices[no][0]
                y = indices[no][1]
                z = indices[no][2]
                mask_array[x, y, z] = 1  # this is a masked item

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetOrigin(ground_truth.GetOrigin())
        mask.SetDirection(ground_truth.GetDirection())
        mask.SetSpacing(ground_truth.GetSpacing())

        return mask
