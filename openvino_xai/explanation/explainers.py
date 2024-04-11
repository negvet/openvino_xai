# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_xai
from openvino_xai.algorithms.black_box.black_box_methods import RISE
from openvino_xai.common.parameters import TaskType
from openvino_xai.common.utils import has_xai
from openvino_xai.common.utils import logger, SALIENCY_MAP_OUTPUT_NAME
from openvino_xai.explanation.explanation_parameters import ExplanationParameters
from openvino_xai.explanation.explanation_result import ExplanationResult
from openvino_xai.explanation.post_process import PostProcessor
from openvino_xai.explanation.explanation_parameters import ExplainMode

import openvino.runtime as ov


class Explainer:
    def __init__(
            self,
            model,
            task_type,
            preprocess_fn,
            postprocess_fn=None,
            explain_mode=ExplainMode.AUTO,
            insertion_parameters=None,
    ):
        self.model = model
        self.compiled_model = None
        self.task_type = task_type

        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        self.insertion_parameters = insertion_parameters

        self.explain_mode = explain_mode

        self._set_explain_mode()

        self._load_model()

    def _set_explain_mode(self):
        if self.explain_mode == ExplainMode.WHITEBOX:
            if has_xai(self.model):
                logger.info("Model already has XAI - using white-box mode.")
            else:
                self._insert_xai()
                logger.info("Explaining the model in the white-box mode.")
        elif self.explain_mode == ExplainMode.BLACKBOX:
            if self.postprocess_fn is None:
                raise ValueError("Postprocess function has to be provided for the black-box mode.")
            logger.info("Explaining the model in the black-box mode.")
        elif self.explain_mode == ExplainMode.AUTO:
            if has_xai(self.model):
                logger.info("Model already has XAI - using white-box mode.")
                self.explain_mode = ExplainMode.WHITEBOX
            else:
                try:
                    self._insert_xai()
                    self.explain_mode = ExplainMode.WHITEBOX
                    logger.info("Explaining the model in the white-box mode.")
                except Exception as e:
                    print(e)
                    logger.info("Failed to insert XAI into the model - use black-box mode.")
                    if self.postprocess_fn is None:
                        raise ValueError("Postprocess function has to be provided for the black-box mode.")
                    self.explain_mode = ExplainMode.BLACKBOX
                    logger.info("Explaining the model in the black-box mode.")
        else:
            raise ValueError(f"Not supported explain mode {self.explain_mode}.")

    def _insert_xai(self):
        logger.info("Model does not have XAI - trying to insert XAI to use white-box mode.")
        # Do we need to keep the original model?
        self.model = openvino_xai.insert_xai(self.model, self.task_type, self.insertion_parameters)

    def _load_model(self):
        self.compiled_model = ov.Core().compile_model(self.model, "CPU")

    def __call__(self, data, explanation_parameters=None, **kwargs):
        """Explainer call that generates processed explanation result."""
        if explanation_parameters is None:
            explanation_parameters = ExplanationParameters()

        if self.explain_mode == ExplainMode.WHITEBOX:
            saliency_map = self._generate_saliency_map_white_box(data)
        else:
            saliency_map = self._generate_saliency_map_black_box(data, explanation_parameters, **kwargs)

        explanation_result = ExplanationResult(
            saliency_map,
            explanation_parameters.target_explain_group,
            explanation_parameters.target_explain_indices,
            explanation_parameters.target_explain_names,
        )
        explanation_result = PostProcessor(
            explanation_result,
            data,
            explanation_parameters.post_processing_parameters,
        ).postprocess()
        return explanation_result

    def model_forward(self, x) -> ov.utils.data_helpers.wrappers.OVDict:
        """Forward pass of the compiled model"""
        x = self.preprocess_fn(x)
        return self.compiled_model(x)

    def _generate_saliency_map_white_box(self, data):
        model_output = self.model_forward(data)
        return model_output[SALIENCY_MAP_OUTPUT_NAME]

    def _generate_saliency_map_black_box(self, data, explanation_parameters, **kwargs):
        if self.task_type == TaskType.CLASSIFICATION:
            saliency_map = RISE.run(
                self.compiled_model,
                self.preprocess_fn,
                self.postprocess_fn,
                data,
                explanation_parameters,
                **kwargs,
            )
            return saliency_map
        raise ValueError(f"Task type {self.task_type} is not supported in the black-box mode.")
