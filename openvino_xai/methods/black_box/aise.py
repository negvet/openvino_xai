# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Callable, List

import math
from scipy.optimize import direct, Bounds
import numpy as np
import openvino.runtime as ov
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from openvino_xai.common.utils import IdentityPreprocessFN, infer_size_from_image, scaling, sigmoid
from openvino_xai.methods.black_box.base import BlackBoxXAIMethod, Preset


class GaussianPerturbationMask:
    def __init__(self, input_size):
        h = np.linspace(0, 1, input_size[0])
        w = np.linspace(0, 1, input_size[1])
        self.h, self.w = np.meshgrid(w, h)

    def _gaussian_2d(self, gauss_params):
        mh, mw, sigma = gauss_params
        A = 1 / (2 * math.pi * sigma * sigma)
        B = (self.h - mh) ** 2 / (2 * sigma ** 2)
        C = (self.w - mw) ** 2 / (2 * sigma ** 2)
        return A * np.exp(-(B + C))

    def generate_kernel_mask(self, gauss_param, scale=1.0):
        kernel = self._gaussian_2d(gauss_param)
        return (kernel / kernel.max()) * scale


class AISE(BlackBoxXAIMethod):
    """AISE explains classification models in black-box mode using 
    AISE: Adaptive Input Sampling for Explanation of Black-box Models.

    :param model: OpenVINO model.
    :type model: ov.Model
    :param postprocess_fn: Preprocessing function that extract scores from IR model output.
    :type postprocess_fn: Callable[[OVDict], np.ndarray]
    :param preprocess_fn: Preprocessing function, identity function by default
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
        super().__init__(model=model, preprocess_fn=preprocess_fn, device_name=device_name)
        self.postprocess_fn = postprocess_fn

        # TODO: resete state ever run
        self.data_preprocessed = None
        self.target = None
        self.num_iterations_per_kernel = None
        self.kernel_width = None
        self.seed = None

        self._kernel_params_hist = collections.defaultdict(list)
        self._pred_score_hist = collections.defaultdict(list)
        self.input_size = None
        self._mask_generator = None
        
        if prepare_model:
            self.prepare_model()

    def generate_saliency_map(
        self,
        data: np.ndarray,
        explain_target_indices: List[int] | None = None,
        preset: Preset = Preset.BALANCE,
        num_iterations_per_kernel: int | None = None,
        kernel_widths: List[float] | np.ndarray | None = None,
        seed: int = 0,
        scale_output: bool = True,
    ):
        """
        Generates inference result of the RISE algorithm.

        :param data: Input image.
        :type data: np.ndarray
        :param explain_target_indices: List of target indices to explain.
        :type explain_target_indices: List[int]
        :param num_iterations_per_kernel: TBD.
        :type num_iterations_per_kernel: int
        :param seed: TBD.
        :type seed: int
        :param scale_output: Whether to scale output or not.
        :type scale_output: bool
        """
        # TODO: support multiple explain_target_indices
        if len(explain_target_indices) > 1:
            raise ValueError

        self.data_preprocessed = self.preprocess_fn(data)
        self.target = explain_target_indices[0]
        self._preset_parameters(preset, num_iterations_per_kernel, kernel_widths)
        self.seed = seed

        self.input_size = infer_size_from_image(self.data_preprocessed)
        self._mask_generator = GaussianPerturbationMask(self.input_size)
        
        saliency_maps = self._run_synchronous_explanation()

        if scale_output:
            saliency_maps = scaling(saliency_maps)
        return {explain_target_indices[0]: saliency_maps}

    def _preset_parameters(
            self, 
            preset: Preset, 
            num_iterations_per_kernel: int | None, 
            kernel_widths: List[float] | np.ndarray | None,
        ) -> None:
        if preset == Preset.SPEED:
            self.num_iterations_per_kernel = 25
            self.kernel_widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.BALANCE:
            self.num_iterations_per_kernel = 50
            self.kernel_widths = np.linspace(0.1, 0.25, 3)
        elif preset == Preset.QUALITY:
            self.num_iterations_per_kernel = 85
            self.kernel_widths = np.linspace(0.075, 0.25, 4)
        else:
            raise ValueError(f"Preset {preset} is not supported.")
        
        if num_iterations_per_kernel is not None:
            self.num_iterations_per_kernel = num_iterations_per_kernel
        if num_iterations_per_kernel is not None:
            self.kernel_widths = kernel_widths

    def _run_synchronous_explanation(self) -> np.ndarray:
        for kernel_width in self.kernel_widths:
            self._kernel_width = kernel_width
            self._run_optimization()

        print('DIRECT:', 'num_iterations_per_kernel', self.num_iterations_per_kernel, 'kernel_widths', self.kernel_widths)

        saliency_map = self._kernel_density_estimation()

        assert all(np.isclose(saliency_map[100][:5], np.array([0.14402176, 0.14759968, 0.15121181, 0.15485552, 0.15852812])))

        return saliency_map

    def _run_optimization(self):
        # DIRECT params
        # TODO: move to constructor
        solver_epsilon = 0.1
        locally_biased = False
        _ = direct(func=self._objective_function,
                    bounds=Bounds([0.0, 0.0], [1.0, 1.0]),
                    eps=solver_epsilon,
                    maxfun=self.num_iterations_per_kernel,
                    locally_biased=locally_biased,
                    )

    def _objective_function(self, args) -> float:
        """
        Objective function to optimize (to find a global minimum).
        Hybrid (dual) paradigm adopted with two sub-objectives:
            - preservation
            - deletion
        """
        mh, mw = args
        kernel_params = (mh, mw, self._kernel_width)
        self._kernel_params_hist[self._kernel_width].append(kernel_params)

        kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
        kernel_mask = np.clip(kernel_mask, 0, 1)

        data_perturbed_preserve = self.data_preprocessed * kernel_mask
        pred_score_preserve = self._get_score(data_perturbed_preserve)

        data_perturbed_delete = self.data_preprocessed * (1 - kernel_mask)
        pred_score_del = self._get_score(data_perturbed_delete)

        loss = (pred_score_preserve - pred_score_del)

        self._pred_score_hist[self._kernel_width].append(pred_score_preserve)
        # self._pred_score_hist[self._kernel_width].append(pred_score_preserve - pred_score_del)  # TODO: ?, max(0, ..)
        # self._pred_score_hist[self._kernel_width].append(max(0, pred_score_preserve - pred_score_del))  # TODO: ?, max(0, ..)

        loss *= -1  # minimizing
        return loss

    def _get_score(self, data_perturbed: np.array) -> float:
        """Get model prediction score for perturbed input."""
        x = self.model_forward(data_perturbed, preprocess=False)
        x = self.postprocess_fn(x)
        if np.max(x) > 1 or np.min(x) < 0:
            x = sigmoid(x)
        pred_scores = x.squeeze()
        return pred_scores[self.target]

    def _kernel_density_estimation(self) -> np.ndarray:
        """Aggregate the result per kernel with KDE."""
        saliency_map_per_kernel = np.zeros((len(self.kernel_widths), self.input_size[0], self.input_size[1]))
        for kernel_index, kernel_width in enumerate(self.kernel_widths):
            kernel_masks_weighted = np.zeros(self.input_size)
            for i in range(self.num_iterations_per_kernel):
                kernel_params = self._kernel_params_hist[kernel_width][i]
                kernel_mask = self._mask_generator.generate_kernel_mask(kernel_params)
                score = self._pred_score_hist[kernel_width][i]
                kernel_masks_weighted += kernel_mask * score
            kernel_masks_weighted = kernel_masks_weighted / kernel_masks_weighted.max()
            saliency_map_per_kernel[kernel_index] = kernel_masks_weighted

        saliency_map = saliency_map_per_kernel.sum(axis=0)  # TODO: softmax?
        saliency_map /= saliency_map.max()
        return saliency_map
