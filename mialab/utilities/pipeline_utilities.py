"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings

import numpy as np
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.feature_pool as fpool
import mialab.filtering.postprocessing as fltr_postp

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()
atlas_mask = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    global atlas_mask
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    atlas_mask = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))

    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5


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
        # no cache for perturbed images (noise, blur, bias field): the cache does not know the perturbation
        use_cache = cache_dir and not is_perturbed(kwargs)
        self.cache = fpool.FeatureCache(cache_dir, kwargs) if use_cache else None

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

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    # construct pipeline for brain mask registration
    # we need to perform this before the T1w and T2w pipeline because the registered mask is used for skull-stripping
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                      len(pipeline_brain_mask.filters) - 1)

    # execute pipeline on the brain mask image
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for T1w image pre-processing
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.create_normalization(kwargs['normalization_pre']))
        pipeline_t1.set_param(fltr_prep.NormalizationParameters(img.images[structure.BrainImageTypes.BrainMask],
                                                                skull_stripped=kwargs.get('skullstrip_pre', False),
                                                                reference=atlas_t1, reference_mask=atlas_mask),
                            len(pipeline_t1.filters) - 1)

    # execute pipeline on the T1w image (bias field: on the raw image, before; noise and blur: on the result, after)
    t1 = _bias_field(img.images[structure.BrainImageTypes.T1w], **kwargs)
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(t1)
    img.images[structure.BrainImageTypes.T1w] = _add_noise(img.images[structure.BrainImageTypes.T1w],
                                                           img.images[structure.BrainImageTypes.BrainMask],
                                                           noise_seed=42, **kwargs)

    # construct pipeline for T2w image pre-processing
    pipeline_t2 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t2.add_filter(fltr_prep.create_normalization(kwargs['normalization_pre']))
        pipeline_t2.set_param(fltr_prep.NormalizationParameters(img.images[structure.BrainImageTypes.BrainMask],
                                                                        skull_stripped=kwargs.get('skullstrip_pre', False),
                                                                        reference=atlas_t2, reference_mask=atlas_mask),
                            len(pipeline_t2.filters) - 1)
    

    # execute pipeline on the T2w image
    t2 = _bias_field(img.images[structure.BrainImageTypes.T2w], **kwargs)
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(t2)
    # another noise seed than T1: the noise of the two images must be independent, as in a real scan
    img.images[structure.BrainImageTypes.T2w] = _add_noise(img.images[structure.BrainImageTypes.T2w],
                                                           img.images[structure.BrainImageTypes.BrainMask],
                                                           noise_seed=43, **kwargs)

    # construct pipeline for ground truth image pre-processing
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                              len(pipeline_gt.filters) - 1)

    # execute pipeline on the ground truth image
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    return img


# =====================================================================================================================
# Image perturbations for the robustness experiment (H3: what does the information of a feature depend on?)
# Only the TEST images are perturbed (extract_features.py --noise / --blur / --bias); the forest is trained on clean
# images. Each perturbation destroys one image property:
#   noise_std  (after normalization): Gaussian noise, in units of the brain's intensity std -> fine detail, edges
#   blur_sigma (after normalization): Gaussian smoothing, sigma in mm                       -> small scales, edges
#   bias_level (on the RAW image):    smooth multiplicative intensity drift of +-bias_level  -> absolute intensity
# =====================================================================================================================
def is_perturbed(kwargs: dict) -> bool:
    """True if any perturbation is switched on (then the feature cache must not be used)."""
    return any(kwargs.get(key, 0) for key in ('noise_std', 'blur_sigma', 'bias_level'))


def _add_noise(image: sitk.Image, brain_mask: sitk.Image, noise_seed: int = 42, **kwargs) -> sitk.Image:
    """Optional perturbations on the pre-processed (registered, skull-stripped, z-scored) image: noise, then blur.

    noise_std:  standard deviation of additive Gaussian noise. After z-score normalization the brain has std 1, so
                0.2 = noise with 20 % of the brain's intensity spread. noise_seed makes it reproducible (T1 and T2 use
                different seeds, so their noise is independent).
    blur_sigma: sigma (mm) of a Gaussian blur, e.g. 1 or 2. Simulates a lower resolution / more partial volume.
    After either, the image is masked again, so the background outside the brain stays 0 like in the clean images.
    """
    noise_std = kwargs.get('noise_std', 0)
    blur_sigma = kwargs.get('blur_sigma', 0)
    if noise_std <= 0 and blur_sigma <= 0:
        return image
    if noise_std > 0:
        image = sitk.AdditiveGaussianNoise(image, noise_std, 0.0, noise_seed)
    if blur_sigma > 0:
        image = sitk.SmoothingRecursiveGaussian(image, blur_sigma)
    if kwargs.get('skullstrip_pre', False):
        image = sitk.Mask(image, sitk.Cast(brain_mask > 0, sitk.sitkUInt8))
    return image


def _bias_field(image: sitk.Image, **kwargs) -> sitk.Image:
    """Optional perturbation on the RAW image (before registration and normalization): a smooth bias field.

    MRI bias fields are slow, multiplicative intensity drifts across the head (coil sensitivity). We simulate the
    simplest one: a linear ramp along the diagonal of the image (left-right + front-back + bottom-top), scaled so
    that it goes from -1 to +1 across the head, and multiply the image by 1 + bias_level * ramp.
    Example: bias_level = 0.2 -> one corner of the head is 20 % darker, the opposite one 20 % brighter, the middle
    unchanged. The z-score normalization later removes the global scale but NOT this spatial drift, exactly like a
    real bias field. Features that use absolute intensity are hit; local contrast (gradients, local std, LoG) hardly
    changes, because the field is almost constant within a few mm.
    """
    level = kwargs.get('bias_level', 0)
    if level <= 0:
        return image
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n, dtype=np.float32) for n in array.shape], indexing='ij')
    ramp = (x + y + z) / 3
    positive = array[array > 0]
    head = array > np.percentile(positive, 5) if positive.size else np.ones(array.shape, bool)
    low, high = ramp[head].min(), ramp[head].max()
    ramp = 2 * (ramp - low) / (high - low) - 1                       # -1 .. +1 across the head
    biased = sitk.GetImageFromArray(array * (1 + level * ramp))
    biased.CopyInformation(image)
    return biased


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator.

    Returns:
        eval.Evaluator: An evaluator.
    """

    # initialize metrics
    metrics = [metric.DiceCoefficient(),
               metric.HausdorffDistance(percentile=95, metric='HDRFDST95')]
    

    # define the labels to evaluate
    labels = {1: 'WhiteMatter',
              2: 'GreyMatter',
              3: 'Hippocampus',
              4: 'Amygdala',
              5: 'Thalamus'
              }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, multi_process: bool = True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[structure.BrainImage]: A list of images.
    """
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())
    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        images = [pre_process(id_, path, **pre_process_params) for id_, path in params_list]
    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process: bool = True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]
    return pp_images
