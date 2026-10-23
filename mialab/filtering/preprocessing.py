"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalizationParameters(pymia_fltr.FilterParams):
    """Image normalization parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the ImageNormalizationParameters

        Args:
            img_mask (sitk.Image): The brain mask image. The mean and standard deviation are computed inside it.
        """
        self.img_mask = img_mask


class ImageNormalization(pymia_fltr.Filter):
    """Represents a z-score normalization filter: (intensity - mean) / std, with mean and std of the brain."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageNormalizationParameters = None) -> sitk.Image:
        """Executes a z-score normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (ImageNormalizationParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        brain = sitk.GetArrayFromImage(params.img_mask) > 0

        # statistics from the brain only, so the background (air, skull) does not influence them
        mean = img_arr[brain].mean()
        std = img_arr[brain].std()
        img_arr = (img_arr - mean) / std

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


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
