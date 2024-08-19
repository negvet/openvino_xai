# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Mapping, Tuple


import numpy as np
import openvino.runtime as ov
from scipy.optimize import Bounds, direct

from openvino_xai.common.parameters import Task
from openvino_xai.common.utils import (
    IdentityPreprocessFN,
    infer_size_from_image,
    logger,
    scaling,
    sigmoid,
)
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod, Preset


class AISEBase(BlackBoxXAIMethod, ABC):
    """
    AISE explains models in black-box mode using
    AISE: Adaptive Input Sampling for Explanation of Black-box Models
    (TODO (negvet): add link to the paper.)

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Post-processing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[Mapping], np.ndarray]
    :param preprocess_fn: Pre-processing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param device_name: Device type name.
    :type device_name: str
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[Mapping], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(model=model, preprocess_fn=preprocess_fn, device_name=device_name)
        self.postprocess_fn = postprocess_fn

        self.data_preprocessed = None
        self.target: int | None = None
        self.num_iterations_per_kernel: int | None = None
        self.kernel_widths: List[float] | np.ndarray | None = None
        self._current_kernel_width: float | None = None
        self.solver_epsilon = 0.1
        self.locally_biased = False
        self.kernel_params_hist: Dict = collections.defaultdict(list)
        self.pred_score_hist: Dict = collections.defaultdict(list)
        self.input_size: Tuple[int, int] | None = None
        self._mask_generator: GaussianPerturbationMask | None = None
        self.bounds = None
        self.preservation = True
        self.deletion = True

        if prepare_model:
            self.prepare_model()

    def _run_synchronous_explanation(self) -> np.ndarray:
        for kernel_width in self.kernel_widths:
            self._current_kernel_width = kernel_width
            self._run_optimization()
        return self._kernel_density_estimation()

    def _run_optimization(self):
        """Run DIRECT optimizer by default."""
        _ = direct(
            func=self._objective_function,
            bounds=self.bounds,
            eps=self.solver_epsilon,
            maxfun=self.num_iterations_per_kernel,
            locally_biased=self.locally_biased,
        )

    def _objective_function(self, args) -> float:
        """
        Objective function to optimize (to find a global minimum).
        Hybrid (dual) paradigm supporte two sub-objectives:
            - preservation
            - deletion
        """
        mh, mw = args
        kernel_params = (mh, mw, self._current_kernel_width)
        self.kernel_params_hist[self._current_kernel_width].append(kernel_params)

        kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
        kernel_mask = np.clip(kernel_mask, 0, 1)

        pred_loss_preserve = 0.0
        if self.preservation:
            data_perturbed_preserve = self.data_preprocessed * kernel_mask
            pred_loss_preserve = self._get_loss(data_perturbed_preserve)

        pred_loss_delete = 0.0
        if self.deletion:
            data_perturbed_delete = self.data_preprocessed * (1 - kernel_mask)
            pred_loss_delete = self._get_loss(data_perturbed_delete)

        loss = pred_loss_preserve - pred_loss_delete

        self.pred_score_hist[self._current_kernel_width].append(pred_loss_preserve - pred_loss_delete)

        loss *= -1  # Objective: minimize
        return loss

    @abstractmethod
    def _get_loss(self, data_perturbed: np.array) -> float:
        pass

    def _kernel_density_estimation(self) -> np.ndarray:
        """Aggregate the result per kernel with KDE."""
        saliency_map_per_kernel = np.zeros((len(self.kernel_widths), self.input_size[0], self.input_size[1]))
        for kernel_index, kernel_width in enumerate(self.kernel_widths):
            kernel_masks_weighted = np.zeros(self.input_size)
            for i in range(self.num_iterations_per_kernel):
                kernel_params = self.kernel_params_hist[kernel_width][i]
                kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
                score = self.pred_score_hist[kernel_width][i]
                kernel_masks_weighted += kernel_mask * score
            kernel_masks_weighted_max = kernel_masks_weighted.max()
            if kernel_masks_weighted_max > 0:
                kernel_masks_weighted = kernel_masks_weighted / kernel_masks_weighted_max
            saliency_map_per_kernel[kernel_index] = kernel_masks_weighted

        saliency_map = saliency_map_per_kernel.sum(axis=0)
        saliency_map /= saliency_map.max()
        return saliency_map


class AISEClassification(AISEBase):
    """
    AISE for classification models.

    postprocess_fn expected to return one container with scores. Without batch dim.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Post-processing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[OVDict], np.ndarray]
    :param preprocess_fn: Pre-processing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param device_name: Device type name.
    :type device_name: str
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(
            model=model,
            postprocess_fn=postprocess_fn,
            preprocess_fn=preprocess_fn,
            device_name=device_name,
            prepare_model=prepare_model,
        )
        self.bounds = Bounds([0.0, 0.0], [1.0, 1.0])

    def generate_saliency_map(  # type: ignore
        self,
        data: np.ndarray,
        target_indices: List[int] | None,
        preset: Preset = Preset.BALANCE,
        num_iterations_per_kernel: int | None = None,
        kernel_widths: List[float] | np.ndarray | None = None,
        solver_epsilon: float = 0.1,
        locally_biased: bool = False,
        scale_output: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Generates inference result of the AISE algorithm.
        Optimized for per class saliency map generation. Not effcient for large number of classes.

        :param data: Input image.
        :type data: np.ndarray
        :param target_indices: List of target indices to explain.
        :type target_indices: List[int]
        :param preset: Speed-Quality preset, defines predefined configurations that manage the speed-quality tradeoff.
        :type preset: Preset
        :param num_iterations_per_kernel: Number of iterations per kernel, defines compute budget.
        :type num_iterations_per_kernel: int
        :param kernel_widths: Kernel bandwidths.
        :type kernel_widths: List[float] | np.ndarray
        :param solver_epsilon: Solver epsilon of DIRECT optimizer.
        :type solver_epsilon: float
        :param locally_biased: Locally biased flag of DIRECT optimizer.
        :type locally_biased: bool
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        self.data_preprocessed = self.preprocess_fn(data)

        if target_indices is None:
            num_classes = self.get_num_classes(self.data_preprocessed)
            if num_classes > 10:
                logger.info(f"num_classes = {num_classes}, which might take significant time to process.")
            target_indices = list(range(num_classes))

        self.num_iterations_per_kernel, self.kernel_widths = self._preset_parameters(
            preset,
            num_iterations_per_kernel,
            kernel_widths,
        )

        self.solver_epsilon = solver_epsilon
        self.locally_biased = locally_biased

        self.input_size = infer_size_from_image(self.data_preprocessed)
        self._mask_generator = GaussianPerturbationMask(self.input_size)

        saliency_maps = {}
        for target in target_indices:
            self.kernel_params_hist = collections.defaultdict(list)
            self.pred_score_hist = collections.defaultdict(list)

            self.target = target
            saliency_map_per_target = self._run_synchronous_explanation()
            if scale_output:
                saliency_map_per_target = scaling(saliency_map_per_target)
            saliency_maps[target] = saliency_map_per_target
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_iterations_per_kernel: int | None,
        kernel_widths: List[float] | np.ndarray | None,
    ) -> Tuple[int, np.ndarray]:
        if preset == Preset.SPEED:
            iterations = 25
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.BALANCE:
            iterations = 50
            widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.QUALITY:
            iterations = 85
            widths = np.linspace(0.075, 0.25, 4)
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_iterations_per_kernel is None:
            num_iterations_per_kernel = iterations
        if kernel_widths is None:
            kernel_widths = widths
        return num_iterations_per_kernel, kernel_widths

    def _get_loss(self, data_perturbed: np.array) -> float:
        """Get loss for perturbed input."""
        x = self.model_forward(data_perturbed, preprocess=False)
        x = self.postprocess_fn(x)
        if np.max(x) > 1 or np.min(x) < 0:
            x = sigmoid(x)
        pred_scores = x.squeeze()  # type: ignore
        return pred_scores[self.target]


class AISEDetection(AISEBase):
    """
    AISE for detection models.

    postprocess_fn expected to return three containers: boxes (format: [x1, y1, x2, y2]), scores, labels. Without batch dim.
    
    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Post-processing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[OVDict], np.ndarray]
    :param preprocess_fn: Pre-processing function, identity function by default
        (assume input images are already preprocessed by user).
    :type preprocess_fn: Callable[[np.ndarray], np.ndarray]
    :param device_name: Device type name.
    :type device_name: str
    :param prepare_model: Loading (compiling) the model prior to inference.
    :type prepare_model: bool
    """

    def __init__(
        self,
        model: ov.Model,
        postprocess_fn: Callable[[OVDict], np.ndarray],
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
        prepare_model: bool = True,
    ):
        super().__init__(
            model=model,
            postprocess_fn=postprocess_fn,
            preprocess_fn=preprocess_fn,
            device_name=device_name,
            prepare_model=prepare_model,
        )
        self.deletion = False

    def generate_saliency_map(  # type: ignore
        self,
        data: np.ndarray,
        target_indices: List[int] | None,
        preset: Preset = Preset.BALANCE,
        num_iterations_per_kernel: int | None = None,
        divisors: List[float] | np.ndarray | None = None,
        solver_epsilon: float = 0.05,
        locally_biased: bool = False,
        scale_output: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Generates inference result of the AISE algorithm.
        Optimized for per class saliency map generation. Not effcient for large number of classes.

        :param data: Input image.
        :type data: np.ndarray
        :param target_indices: List of target indices to explain.
        :type target_indices: List[int]
        :param preset: Speed-Quality preset, defines predefined configurations that manage the speed-quality tradeoff.
        :type preset: Preset
        :param num_iterations_per_kernel: Number of iterations per kernel, defines compute budget.
        :type num_iterations_per_kernel: int
        :param divisors: List of dividors, used to derive kernel widths in an adaptive manner.
        :type divisors: List[float] | np.ndarray
        :param solver_epsilon: Solver epsilon of DIRECT optimizer.
        :type solver_epsilon: float
        :param locally_biased: Locally biased flag of DIRECT optimizer.
        :type locally_biased: bool
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        # TODO (negvet): support custom bboxes (not predicted ones)

        self.data_preprocessed = self.preprocess_fn(data)
        forward_output = self.model_forward(self.data_preprocessed, preprocess=False)

        # postprocess_fn expected to return three containers: boxes (x1, y1, x2, y2), scores, labels, without batch dim.
        boxes, scores, labels = self.postprocess_fn(forward_output)

        if target_indices is None:
            num_boxes = len(boxes)
            if num_boxes > 10:
                logger.info(f"num_boxes = {num_boxes}, which might take significant time to process.")
            target_indices = list(range(num_boxes))

        self.num_iterations_per_kernel, self.divisors = self._preset_parameters(
            preset,
            num_iterations_per_kernel,
            divisors,
        )

        self.solver_epsilon = solver_epsilon
        self.locally_biased = locally_biased

        self.input_size = infer_size_from_image(self.data_preprocessed)
        original_size = infer_size_from_image(data)
        self._mask_generator = GaussianPerturbationMask(self.input_size)

        saliency_maps = {}
        for target in target_indices:
            self.kernel_params_hist = collections.defaultdict(list)
            self.pred_score_hist = collections.defaultdict(list)

            self.target_box = boxes[target]
            self.target_label = labels[target]

            self._process_box()
            saliency_map_per_target = self._run_synchronous_explanation()
            if scale_output:
                saliency_map_per_target = scaling(saliency_map_per_target)
            saliency_maps[target] = saliency_map_per_target

            self._update_metadata(boxes, scores, labels, target, original_size)
        return saliency_maps

    @staticmethod
    def _preset_parameters(
        preset: Preset,
        num_iterations_per_kernel: int | None,
        divisors: List[float] | np.ndarray | None,
    ) -> Tuple[int, np.ndarray]:
        if preset == Preset.SPEED:
            iterations = 50
            divs = np.linspace(7, 1, 3)
        elif preset == Preset.BALANCE:
            iterations = 100
            divs = np.linspace(7, 1, 3)
        elif preset == Preset.QUALITY:
            iterations = 150
            divs = np.linspace(8, 1, 5)
        else:
            raise ValueError(f"Preset {preset} is not supported.")

        if num_iterations_per_kernel is None:
            num_iterations_per_kernel = iterations
        if divisors is None:
            divisors = divs
        return num_iterations_per_kernel, divisors

    def _process_box(self, padding_coef: float = 0.5) -> None:
        target_box_scaled = [
            self.target_box[0] / self.input_size[1],  # x1
            self.target_box[1] / self.input_size[0],  # y1
            self.target_box[2] / self.input_size[1],  # x2
            self.target_box[3] / self.input_size[0],  # y2
        ]
        box_width = target_box_scaled[2] - target_box_scaled[0]
        box_height = target_box_scaled[3] - target_box_scaled[1]
        self._min_box_size = min(box_width, box_height)
        self.kernel_widths = [self._min_box_size / div for div in self.divisors]

        x_from = max(target_box_scaled[0] - box_width * padding_coef, 0.0)
        x_to = min(target_box_scaled[2] + box_width * padding_coef, 1.0)
        y_from = max(target_box_scaled[1] - box_height * padding_coef, 0.0)
        y_to = min(target_box_scaled[3] + box_height * padding_coef, 1.0)
        self.bounds = Bounds([x_from, y_from], [x_to, y_to])

    def _get_loss(self, data_perturbed: np.array) -> float:
        """Get loss for perturbed input."""
        forward_output = self.model_forward(data_perturbed, preprocess=False)
        boxes, pred_scores, labels = self.postprocess_fn(forward_output)

        loss = 0
        for box, pred_score, label in zip(boxes, pred_scores, labels):
            if label == self.target_label:
                loss = max(loss, self._iou(self.target_box, box) * pred_score)
        return loss

    @staticmethod
    def _iou(box1: np.ndarray | List[float], box2: np.ndarray | List[float]) -> float:
        box1 = np.asarray(box1)
        box2 = np.asarray(box2)
        tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
        br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
        intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
        area1 = np.prod(box1[2:] - box1[:2])
        area2 = np.prod(box2[2:] - box2[:2])
        return intersection / (area1 + area2 - intersection)

    def _update_metadata(
        self,
        boxes: np.ndarray | List,
        scores: np.ndarray | List[float],
        labels: np.ndarray | List[int],
        target: int,
        original_size: Tuple[int, int],
    ) -> None:
        x1, y1, x2, y2 = boxes[target]
        width_scale = original_size[1] / self.input_size[1]
        height_scale = original_size[0] / self.input_size[0]
        x1, x2 = x1 * width_scale, x2 * width_scale
        y1, y2 = y1 * height_scale, y2 * height_scale
        self.metadata[Task.DETECTION][target] = [x1, y1, x2, y2], scores[target], labels[target]


class GaussianPerturbationMask:
    """
    Perturbation mask generator.
    """

    def __init__(self, input_size: Tuple[int, int]):
        h = np.linspace(0, 1, input_size[0])
        w = np.linspace(0, 1, input_size[1])
        self.h, self.w = np.meshgrid(w, h)

    def _get_2d_gaussian(self, gauss_params: Tuple[float, float, float]) -> np.ndarray:
        mh, mw, sigma = gauss_params
        A = 1 / (2 * math.pi * sigma * sigma)
        B = (self.h - mh) ** 2 / (2 * sigma**2)
        C = (self.w - mw) ** 2 / (2 * sigma**2)
        return A * np.exp(-(B + C))

    def generate_kernel_mask(self, gauss_param: Tuple[float, float, float], scale: float = 1.0):
        """
        Generates 2D gaussian mask.
        """
        gaussian = self._get_2d_gaussian(gauss_param)
        return (gaussian / gaussian.max()) * scale
