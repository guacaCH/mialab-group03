"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import abc
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

class NormalizationParameters(pymia_fltr.FilterParams):
    """Normalization parameters"""

    def __init__(self, img_mask: sitk.Image, skull_stripped: bool, reference: sitk.Image = None,
                 reference_mask: sitk.Image = None):        
        """Initializes a new instance of the NormalizationParameters class
        
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

class ImageNormalization(pymia_fltr.Filter):
    """Base class of all normalization filters.
    
    The base class handles the numpy conversion, the brain mask and the background.  
    Subclasses only implement the specific normalization method of computing the statistics from the brain voxels 
    and apply the mapping to all voxels.
    """

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: NormalizationParameters = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        brain = sitk.GetArrayFromImage(params.img_mask) > 0

        img_out_arr = self.normalize(img_arr, img_arr[brain], params).astype(np.float32)
        if params.skull_stripped:
            img_out_arr[~brain] = 0     # keep the background of a skull stripped image at 0
       
        img_out = sitk.GetImageFromArray(img_out_arr) # normalize the image using numpy
        img_out.CopyInformation(image)
        return img_out

    @abc.abstractmethod
    def normalize(self, values: np.ndarray, brain_values: np.ndarray,
                  params: NormalizationParameters) -> np.ndarray:
        """Normalize intensities
        
        Args:
            values (np.ndarray): The intensities to normalize (the whole image)
            brain_values (np.ndarray): 1D array with the intensities of the brain voxels, used for the statistics.
            params (NormalizationParameters): The parameters (e.g. for the reference image).
            
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
        """Initalizes a new instance of the PercentileNormalization class
        
        Args:
            lower (float): The percentile mapped to 0.
            upper (float): The percentile mapped to 1.
        """
        super.__init__()
        self.lower = lower
        self.upper = upper

    def normalize(self, values, brain_values, params):
        low, high = np.percentile(brain_values, [self.lower, self.upper])
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    def __str__(self):
        return 'PercentileNormalization:\n lower: {self.lower}\n upper: {self.upper}\n'.format(self=self)

class HistogramMatching(ImageNormalization):
    """Histogram matching to a reference image (picewise-linear landmark matching)
    
    The percentiles of the brain intensities are mapped onto the same percentiles of the reference image
    (e.g. the skull-stripped atlas); linear interpolation for intensities in between
    """
    def __init__(self, landmarks: int = 11):
        """ Initializes a new instance of the HistagromMatching class.
        Args:
            landmarks (int): The number of percentiles from 1 to 99 that are matched.
        """
        super().__init__()
        self.landmarks = landmarks
    def normalize(self, values, brain_values, params):
        if params.reference is None:
            raise ValueError('HistoggramMatching need a reference image (NormalizationParameter.reference)')
        reference = sitk.GetArrayFromImage(params.reference).astype(np.float32)
        if params.reference_mask is not None:
            reference = reference[sitk.GetArrayFromImage(params.reference_mask) > 0]
        else:
            reference = reference[reference > 0]

        percentiles = np.concatenate([[0], np.linspace(1, 99, self.landmarks), [100]])
        source_landmarks = np.percentile(brain_values, percentiles)
        reference_landmarks = np.percentile(reference, percentiles)
        # values outside the brain range (e.g. skull without skull-stripping) are clamped to the range of the reference 
        return np.interp(values, source_landmarks, reference_landmarks)

    def __str__(self):
        return 'HistogramMatching:\n landmarks: {self.landmarks}\n'.format(self=self)

NORMALIZATIONS = {'zscore': ZScoreNormalization,
                  'minmax': MinMaxNormalization,
                  'percentile': PercentileNormalization,
                  'histogram_matching': HistogramMatching}

def create_normalization(method) -> ImageNormalization:
    """Creates a normalization filter from a config value
    
    Args:
        method: True (= 'zscore'), a name from NORMALIZATIONS, or a dict like
        {'method': 'percentile', 'lower': 0.5, 'upper': 99.5} (keys passed to constructor)
        
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

class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask

        # set all voxels outside the brain mask to 0
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        return sitk.Mask(image, mask, outsideValue=0)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """
        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        # apply the provided affine transformation from image to atlas space: resample the image
        # onto the atlas grid. labels and masks use nearest-neighbour interpolation to ensure
        # that they keep their integer values. intensity images use linear interpolation
        interpolator = sitk.sitkNearestNeighbor if is_ground_truth else sitk.sitkLinear
        return sitk.Resample(image, atlas, transform, interpolator, 0.0, image.GetPixelID())

     
    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
