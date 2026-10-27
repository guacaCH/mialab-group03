"""This module contains utility classes and functions."""
import enum
import os
import typing as t
<<<<<<< Updated upstream
import warnings
=======
import zlib
>>>>>>> Stashed changes

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
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)
<<<<<<< Updated upstream
=======
        self.t2_intensity_feature = kwargs.get('t2_intensity_feature', False)
        self.t2_gradient_intensity_feature = kwargs.get('t2_gradient_intensity_feature', False)
        # extended features (extract_features.py --extended)
        self.multiscale_feature = kwargs.get('multiscale_feature', False)
        self.local_statistics_feature = kwargs.get('local_statistics_feature', False)
        self.t1_t2_difference_feature = kwargs.get('t1_t2_difference_feature', False)
        self.random_feature = kwargs.get('random_feature', False)
        # label prior: label_counts[i] counts how many training subjects have label i + 1 at a voxel (atlas space)
        self.label_counts = kwargs.get('label_counts', None)
        self.label_count_subjects = kwargs.get('label_count_subjects', 0)
>>>>>>> Stashed changes

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # todo: add T2w features
        warnings.warn('No features from T2-weighted image extracted.')

        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])

        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]

        if self.gradient_intensity_feature:
            # compute gradient magnitude images
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])

<<<<<<< Updated upstream
=======
        if self.t2_intensity_feature:
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]

        if self.t2_gradient_intensity_feature:
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])

        # to add a feature: add a flag in __init__ and an if-block like the ones above. The key in feature_images
        # becomes the feature's name, so it can be a FeatureImageTypes entry or a string like 'T1w_GAUSS_S2'.

        modalities = ((structure.BrainImageTypes.T1w, 'T1w'), (structure.BrainImageTypes.T2w, 'T2w'))

        if self.multiscale_feature:
            # Gaussian smoothing (the surroundings) and Laplacian of Gaussian (blobs and edges) at 3 scales (sigma in mm)
            for image_type, modality in modalities:
                image = self.img.images[image_type]
                for sigma in (1, 2, 4):
                    self.img.feature_images[f'{modality}_GAUSS_S{sigma}'] = \
                        sitk.SmoothingRecursiveGaussian(image, sigma)
                    self.img.feature_images[f'{modality}_LOG_S{sigma}'] = \
                        sitk.LaplacianRecursiveGaussian(image, sigma)

        if self.local_statistics_feature:
            # mean and standard deviation in a 5x5x5 cube around the voxel: uniform tissue or a boundary?
            for image_type, modality in modalities:
                image = self.img.images[image_type]
                self.img.feature_images[f'{modality}_LOCAL_MEAN'] = sitk.Mean(image, [2, 2, 2])
                self.img.feature_images[f'{modality}_LOCAL_STD'] = sitk.Noise(image, [2, 2, 2])

        if self.t1_t2_difference_feature:
            # both images are z-scored, so a ratio would divide by values close to 0. The difference combines the two
            # contrasts in the same way (WM: high T1, low T2. GM: lower T1, higher T2)
            self.img.feature_images['T1w_MINUS_T2w'] = \
                self.img.images[structure.BrainImageTypes.T1w] - self.img.images[structure.BrainImageTypes.T2w]

        if self.label_counts is not None:
            # probabilistic atlas: the fraction of training subjects with this label at this voxel. A training subject
            # is left out of its own prior, otherwise the feature would contain its own label
            counts = self.label_counts.astype(np.float32)
            n = self.label_count_subjects
            if self.training:
                ground_truth = sitk.GetArrayFromImage(self.img.images[structure.BrainImageTypes.GroundTruth])
                counts -= np.stack([ground_truth == label for label in range(1, len(counts) + 1)])
                n -= 1
            for name, prior in zip(('WM', 'GM', 'HIPPOCAMPUS', 'AMYGDALA', 'THALAMUS'), counts / n):
                image = sitk.GetImageFromArray(prior)
                image.CopyInformation(self.img.images[structure.BrainImageTypes.T1w])
                self.img.feature_images[f'PRIOR_{name}'] = image

        if self.random_feature:
            # pure noise as a control: a real feature less important than this one is useless. A separate random
            # generator per subject, so the training voxels sampled with np.random stay the same
            rng = np.random.default_rng(zlib.crc32(self.img.id_.encode()))
            shape = sitk.GetArrayFromImage(self.img.images[structure.BrainImageTypes.T1w]).shape
            image = sitk.GetImageFromArray(rng.standard_normal(shape).astype(np.float32))
            image.CopyInformation(self.img.images[structure.BrainImageTypes.T1w])
            self.img.feature_images['RANDOM_NOISE'] = image

>>>>>>> Stashed changes
        self._generate_feature_matrix()

        return self.img

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

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

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

    # execute pipeline on the T1w image
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

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
    
    sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], 'check_T1.nii.gz')

    # execute pipeline on the T2w image
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

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
