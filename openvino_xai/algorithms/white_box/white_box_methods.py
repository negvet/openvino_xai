# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod
from typing import Optional, Tuple, Union, List

import numpy as np

import openvino
from openvino.runtime import opset10 as opset

from openvino_xai.explanation.explanation_parameters import TargetExplainGroup
from openvino_xai.insertion.insertion_parameters import ModelType
from openvino_xai.insertion.parse_model import IRParserCls, IRParser


class WhiteBoxXAIMethodBase(ABC):
    """Base class for methods that generates XAI branch of the model."""

    def __init__(self, model: openvino.runtime.Model, embed_normalization: bool = True):
        self._model_ori = model
        self.embed_normalization = embed_normalization

    @property
    def model_ori(self):
        return self._model_ori

    @property
    def model_ori_params(self):
        return self._model_ori.get_parameters()

    @abstractmethod
    def generate_xai_branch(self):
        """Implements specific XAI algorithm"""

    @staticmethod
    def _propagate_dynamic_batch_dimension(model: openvino.runtime.Model):
        # TODO: support models with multiple inputs.
        assert len(model.inputs) == 1, "Support only for models with a single input."
        if not model.input(0).partial_shape[0].is_dynamic:
            partial_shape = model.input(0).partial_shape
            partial_shape[0] = -1  # make batch dimensions to be dynamic
            model.reshape(partial_shape)

    @staticmethod
    def _normalize_saliency_maps(saliency_maps: openvino.runtime.Node, per_class: bool) -> openvino.runtime.Node:
        """Normalize saliency maps to [0, 255] range, per-map."""
        # TODO: unify for per-class and for per-image
        if per_class:
            # Normalization for per-class saliency maps
            _, num_classes, h, w = saliency_maps.get_output_partial_shape(0)
            num_classes, h, w = num_classes.get_length(), h.get_length(), w.get_length()
            saliency_maps = opset.reshape(saliency_maps, (num_classes, h * w), False)
            max_val = opset.unsqueeze(opset.reduce_max(saliency_maps.output(0), [1]), 1)
            min_val = opset.unsqueeze(opset.reduce_min(saliency_maps.output(0), [1]), 1)
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(opset.subtract(max_val.output(0), min_val.output(0)),
                                    opset.constant(1e-12, dtype=np.float32))
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            saliency_maps = opset.reshape(saliency_maps, (1, num_classes, h, w), False)
            return saliency_maps
        else:
            # Normalization for per-image saliency map
            max_val = opset.reduce_max(saliency_maps.output(0), [0, 1, 2])
            min_val = opset.reduce_min(saliency_maps.output(0), [0, 1, 2])
            numerator = opset.subtract(saliency_maps.output(0), min_val.output(0))
            denominator = opset.add(opset.subtract(max_val.output(0), min_val.output(0)),
                                    opset.constant(1e-12, dtype=np.float32))
            saliency_maps = opset.divide(numerator, denominator)
            saliency_maps = opset.multiply(saliency_maps.output(0), opset.constant(255, dtype=np.float32))
            return saliency_maps


class ActivationMapXAIMethod(WhiteBoxXAIMethodBase):
    """Implements ActivationMap"""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = False
        self.model_type = ModelType.CNN
        self.supported_target_explain_groups = [TargetExplainGroup.IMAGE]
        self.default_target_explain_group = TargetExplainGroup.IMAGE
        self._target_layer = target_layer

    def generate_xai_branch(self) -> openvino.runtime.Node:
        target_node_ori = IRParserCls.get_target_node(self._model_ori, self.model_type, self._target_layer)
        saliency_maps = opset.reduce_mean(target_node_ori.output(0), 1)
        if self.embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps


class FeatureMapPerturbationBase(WhiteBoxXAIMethodBase):
    """Base class for Recipro-CAM methods.

    :param model: OpenVino model.
    :type model: openvino.runtime.Model
    :parameter target_layer: Target layer (node) name after which the XAI branch will be inserted.
    :type target_layer: str
    :param embed_normalization: Whether to normalize output or not.
    :type embed_normalization: bool
    """

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = True
        self.supported_target_explain_groups = [
            TargetExplainGroup.ALL,
            TargetExplainGroup.PREDICTIONS,
            TargetExplainGroup.CUSTOM,
        ]
        self.default_target_explain_group = TargetExplainGroup.PREDICTIONS
        self._target_layer = target_layer

    def generate_xai_branch(self):
        """Generates XAI branch."""
        model_clone = self._model_ori.clone()
        self._propagate_dynamic_batch_dimension(model_clone)

        saliency_maps = self._get_saliency_map(
            model_clone
        )

        if self.embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps

    @abstractmethod
    def _get_saliency_map(self, model_clone: openvino.runtime.Model):
        raise NotImplementedError


class ReciproCAMXAIMethod(FeatureMapPerturbationBase):
    """Implements Recipro-CAM for CNN models"""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
    ):
        super().__init__(model, target_layer, embed_normalization)
        self.model_type = ModelType.CNN

    def _get_saliency_map(self, model_clone: openvino.runtime.Model):
        target_node_ori = IRParserCls.get_target_node(self._model_ori, self.model_type, self._target_layer)
        target_node_name = self._target_layer or target_node_ori.get_friendly_name()
        post_target_node_clone = IRParserCls.get_post_target_node(model_clone, self.model_type, target_node_name)

        logit_node = IRParserCls.get_logit_node(self._model_ori, search_softmax=True)
        logit_node_clone_model = IRParserCls.get_logit_node(model_clone, search_softmax=True)

        if not logit_node_clone_model.output(0).partial_shape[0].is_dynamic:
            raise ValueError("Batch shape of the output should be dynamic, but it is static. "
                             "Make sure that the dynamic inputs can propagate through the model graph.")

        _, c, h, w = target_node_ori.get_output_partial_shape(0)
        c, h, w = c.get_length(), h.get_length(), w.get_length()
        if not self._is_valid_layout(c, h, w):
            raise ValueError(f"ReciproCAM supports only NCHW layout, but got NHWC, with shape: [N, {c}, {h}, {w}]")

        feature_map_repeated = opset.tile(target_node_ori.output(0), (h * w, 1, 1, 1))
        mosaic_feature_map_mask = np.zeros((h * w, c, h, w), dtype=np.float32)
        tmp = np.arange(h * w)
        spacial_order = np.reshape(tmp, (h, w))
        for i in range(h):
            for j in range(w):
                k = spacial_order[i, j]
                mosaic_feature_map_mask[k, :, i, j] = np.ones((c))
        mosaic_feature_map_mask = opset.constant(mosaic_feature_map_mask)
        mosaic_feature_map = opset.multiply(feature_map_repeated, mosaic_feature_map_mask)

        for node in post_target_node_clone:
            node.input(0).replace_source_output(mosaic_feature_map.output(0))

        mosaic_prediction = logit_node_clone_model

        tmp = opset.transpose(mosaic_prediction.output(0), (1, 0))
        _, num_classes = logit_node.get_output_partial_shape(0)
        saliency_maps = opset.reshape(tmp, (1, num_classes.get_length(), h, w), False)
        return saliency_maps

    @staticmethod
    def _is_valid_layout(c: int, h: int, w: int):
        return h < c and w < c


class ViTReciproCAMXAIMethod(FeatureMapPerturbationBase):
    """Implements ViTRecipro-CAM for transformer models.

    :param use_gaussian: Whether to use Gaussian for mask generation or not.
    :type use_gaussian: bool
    :param cls_token: Whether to use cls token for mosaic prediction or not.
    :type cls_token: bool
    :param final_norm: Whether the model has normalization after the last transformer block.
    :type final_norm: bool
    :param k: Count of the transformer block (from head) before which XAI branch will be inserted, 1-indexed.
    :type k: int
    """

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: Optional[str] = None,
            embed_normalization: bool = True,
            use_gaussian: bool = True,
            cls_token: bool = True,
            final_norm: bool = True,
            k: int = 1,
    ):
        super().__init__(model, target_layer, embed_normalization)
        self.model_type = ModelType.TRANSFORMER
        self._use_gaussian = use_gaussian
        self._cls_token = cls_token

        # Count of target "Add" node (between the blocks), from the output, 1-indexed
        self._k = k * 2 + int(final_norm)

    def _get_saliency_map(self, model_clone: openvino.runtime.Model):
        #      Add       -> add node before the target transformer blocks
        #   ↓       ↓
        #  skip   block  -> skip connection to the next block and target block itself
        #   ↓       ↓
        #      Add       -> add node after the target transformer blocks

        # Get target Add node in-between the transformer blocks
        target_node_ori = IRParserCls.get_target_node(self._model_ori, self.model_type, self._target_layer, self._k)
        target_node_name = self._target_layer or target_node_ori.get_friendly_name()

        # Get post-add nodes and check them
        post_target_node_clone = IRParserCls.get_post_target_node(model_clone, self.model_type, target_node_name)
        self._post_add_node_check(post_target_node_clone)

        # Get logit nodes. Check them and retrieve info
        logit_node = IRParserCls.get_logit_node(self._model_ori, search_softmax=False)
        logit_node_clone = IRParserCls.get_logit_node(model_clone, search_softmax=False)
        if not logit_node_clone.output(0).partial_shape[0].is_dynamic:
            raise ValueError("Batch shape of the output should be dynamic, but it is static. "
                             "Make sure that the dynamic inputs can propagate through the model graph.")

        _, num_classes = logit_node.get_output_partial_shape(0)
        dim, h, w, num_aux_tokens = self._get_internal_size(target_node_ori)

        # Depth first search till the end of the LayerNorm (traverse of the block branch)
        post_target_node_ori = IRParserCls.get_post_target_node(self._model_ori, self.model_type, target_node_name)
        norm_node_ori = self._get_non_add_node_from_two_nodes(post_target_node_ori)
        while norm_node_ori.get_type_name() != "Add":
            if len(norm_node_ori.outputs()) > 1:
                raise ValueError
            inputs = norm_node_ori.output(0).get_target_inputs()
            if len(inputs) > 1:
                raise ValueError
            norm_node_ori = next(iter(inputs)).get_node()

        # Mosaic feature map after the LayerNorm
        post_target_node_clone_norm = IRParserCls.get_post_target_node(
            model_clone, self.model_type, norm_node_ori.get_friendly_name()
        )
        mosaic_feature_map_norm = self._get_mosaic_feature_map(norm_node_ori, dim, h, w, num_aux_tokens)
        for node in post_target_node_clone_norm:
            node.input(0).replace_source_output(mosaic_feature_map_norm.output(0))

        # Mosaic feature map after the Add node
        mosaic_feature_map = self._get_mosaic_feature_map(target_node_ori, dim, h, w, num_aux_tokens)
        add_node_clone = self._get_add_node_from_two_nodes(post_target_node_clone)
        add_node_clone.input(0).replace_source_output(mosaic_feature_map.output(0))

        # Transform mosaic predictions into the saliency map
        mosaic_prediction = logit_node_clone
        tmp = opset.transpose(mosaic_prediction.output(0), (1, 0))
        saliency_maps = opset.reshape(tmp, (1, num_classes.get_length(), h, w), False)
        return saliency_maps

    def _get_internal_size(self, target_node):
        _, token_number, dim = target_node.get_output_partial_shape(0)
        if token_number.is_dynamic or dim.is_dynamic:
            first_conv_node = IRParserCls.get_first_conv_node(self._model_ori)
            _, dim, h, w = first_conv_node.get_output_partial_shape(0)
            dim, h, w = dim.get_length(), h.get_length(), w.get_length()

            first_concat_node = IRParserCls.get_first_concat_node(self._model_ori)
            num_aux_tokens = len(first_concat_node.inputs()) - 1
        else:
            token_number, dim = token_number.get_length(), dim.get_length()
            h = w = int((token_number - 1) ** 0.5)
            num_aux_tokens = token_number - (h * w)
        return dim, h, w, num_aux_tokens

    def _get_add_node_from_two_nodes(self, node_list):
        self._post_add_node_check(node_list)

        node1, node2 = node_list
        if node1.get_type_name() == "Add":
            return node1
        return node2

    def _get_non_add_node_from_two_nodes(self, node_list):
        self._post_add_node_check(node_list)

        node1, node2 = node_list
        if node1.get_type_name() != "Add":
            return node1
        return node2

    @staticmethod
    def _post_add_node_check(node_list):
        if len(node_list) != 2:
            raise ValueError(f"Only two outputs of the between block Add node supported, "
                             f"but got {len(node_list)}.")
        node1, node2 = node_list
        if not (node1.get_type_name() == "Add") != (node2.get_type_name() == "Add"):
            raise ValueError(f"One (and only one) of the nodes has to be Add type. "
                             f"But got {node1.get_type_name()} and {node2.get_type_name()}.")

    def _get_mosaic_feature_map(self, target_node_ori, dim, h, w, num_aux_tokens):
        if self._use_gaussian:
            if self._cls_token:
                cls_token = opset.slice(target_node_ori, np.array([0]), np.array([num_aux_tokens]), np.array([1]), np.array([1]))
            else:
                cls_token = opset.constant(np.zeros((1, 1, dim)), dtype=np.float32)
            cls_token = opset.tile(cls_token.output(0), (h * w, 1, 1))

            target_node_ori_wo_cls_token = opset.slice(
                target_node_ori, np.array([1]), np.array([h * w + 1]), np.array([1]), np.array([1])
            )
            feature_map_spacial = opset.reshape(target_node_ori_wo_cls_token, (1, h, w, dim), False)
            feature_map_spacial_repeated = opset.tile(feature_map_spacial.output(0), (h * w, 1, 1, 1))

            tmp = np.arange(h * w)
            spacial_order = np.reshape(tmp, (h, w))
            gaussian = np.array(
                [[1 / 16.0, 1 / 8.0, 1 / 16.0], [1 / 8.0, 1 / 4.0, 1 / 8.0], [1 / 16.0, 1 / 8.0, 1 / 16.0]]
            )
            mosaic_feature_map_mask_padded = np.zeros((h * w, h + 2, w + 2), dtype=np.float32)
            for i in range(h):
                for j in range(w):
                    k = spacial_order[i, j]
                    i_pad = i + 1
                    j_pad = j + 1
                    mosaic_feature_map_mask_padded[k, i_pad - 1: i_pad + 2, j_pad - 1: j_pad + 2] = gaussian
            mosaic_feature_map_mask = mosaic_feature_map_mask_padded[:, 1:-1, 1:-1]
            mosaic_feature_map_mask = np.expand_dims(mosaic_feature_map_mask, 3)
            mosaic_feature_map_mask = opset.constant(mosaic_feature_map_mask)
            mosaic_feature_map_mask = opset.tile(mosaic_feature_map_mask.output(0), (1, 1, 1, dim))

            mosaic_fm_wo_cls_token = opset.multiply(feature_map_spacial_repeated, mosaic_feature_map_mask)
            mosaic_fm_wo_cls_token = opset.reshape(mosaic_fm_wo_cls_token, (h * w, h * w, dim), False)
            mosaic_feature_map = opset.concat([cls_token, mosaic_fm_wo_cls_token], 1)
        else:
            mosaic_feature_map_mask_wo_cls_token = np.zeros((h * w, h * w), dtype=np.float32)
            for i in range(h * w):
                mosaic_feature_map_mask_wo_cls_token[i, i] = 1
            if self._cls_token:
                cls_token_mask = np.ones((h * w, 1), dtype=np.float32)
            else:
                cls_token_mask = np.zeros((h * w, 1), dtype=np.float32)
            mosaic_feature_map_mask = np.hstack((cls_token_mask, mosaic_feature_map_mask_wo_cls_token))

            mosaic_feature_map_mask = np.expand_dims(mosaic_feature_map_mask, 2)

            mosaic_feature_map_mask = opset.constant(mosaic_feature_map_mask)
            mosaic_feature_map_mask = opset.tile(mosaic_feature_map_mask.output(0), (1, 1, dim))  # e.g. 784x785x768
            feature_map_repeated = opset.tile(target_node_ori.output(0), (h * w, 1, 1))  # e.g. 784x785x768
            mosaic_feature_map = opset.multiply(feature_map_repeated, mosaic_feature_map_mask)
        return mosaic_feature_map


class DetClassProbabilityMapXAIMethod(WhiteBoxXAIMethodBase):
    """Implements DetClassProbabilityMap, used for single-stage detectors, e.g. SSD, YOLOX or ATSS."""

    def __init__(
            self,
            model: openvino.runtime.Model,
            target_layer: List[str],
            num_anchors: List[int],
            saliency_map_size: Union[Tuple[int, int], List[int]] = (23, 23),
            embed_normalization: bool = True,
    ):
        super().__init__(model, embed_normalization)
        self.per_class = True
        self.supported_target_explain_groups = [
            TargetExplainGroup.ALL,
            TargetExplainGroup.PREDICTIONS,
            TargetExplainGroup.CUSTOM,
        ]
        self.default_target_explain_group = TargetExplainGroup.ALL
        self._target_layer = target_layer
        self._num_anchors = num_anchors  # Either num_anchors or num_classes has to be provided to process cls head output
        self._saliency_map_size = saliency_map_size  # Not always can be obtained from model -> defined externally

    def generate_xai_branch(self) -> openvino.runtime.Node:
        cls_head_output_nodes = []
        for op in self._model_ori.get_ordered_ops():
            if op.get_friendly_name() in self._target_layer:
                cls_head_output_nodes.append(op)
        if len(cls_head_output_nodes) != len(self._target_layer):
            raise ValueError("Not all target layers were found.")

        # TODO: better handle num_classes num_anchors availability
        _, num_channels, _, _ = cls_head_output_nodes[-1].get_output_partial_shape(0)
        num_cls_out_channels = num_channels.get_length() // self._num_anchors[-1]

        # Handle anchors
        for scale_idx in range(len(cls_head_output_nodes)):
            cls_scores_per_scale = cls_head_output_nodes[scale_idx]
            _, _, h, w = cls_scores_per_scale.get_output_partial_shape(0)
            cls_scores_anchor_grouped = opset.reshape(
                cls_scores_per_scale,
                (1, self._num_anchors[scale_idx], num_cls_out_channels, h.get_length(), w.get_length()),
                False,
            )
            cls_scores_out = opset.reduce_max(cls_scores_anchor_grouped, 1)
            cls_head_output_nodes[scale_idx] = cls_scores_out

        # Handle scales
        for scale_idx in range(len(cls_head_output_nodes)):
            cls_head_output_nodes[scale_idx] = opset.interpolate(
                cls_head_output_nodes[scale_idx].output(0),
                output_shape=np.array([1, num_cls_out_channels, *self._saliency_map_size]),
                scales=np.array([1, 1, 1, 1], dtype=np.float32),
                mode="linear",
                shape_calculation_mode="sizes"
            )

        saliency_maps = opset.reduce_mean(opset.concat(cls_head_output_nodes, 0), 0, keep_dims=True)
        saliency_maps = opset.softmax(saliency_maps.output(0), 1)

        if self.embed_normalization:
            saliency_maps = self._normalize_saliency_maps(saliency_maps, self.per_class)
        return saliency_maps