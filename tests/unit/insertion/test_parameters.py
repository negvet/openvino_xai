# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: rm this import. AttributeError: module 'openvino.model_api' has no attribute 'models'
from openvino.model_api.models import ClassificationModel

from openvino_xai.common.parameters import XAIMethodType
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters
from openvino_xai.insertion.insertion_parameters import DetectionInsertionParameters


def test_classification_insertion_parameters():
    cls_insertion_params = ClassificationInsertionParameters()
    assert cls_insertion_params.target_layer is None
    assert cls_insertion_params.embed_normalization
    assert cls_insertion_params.explain_method_type == XAIMethodType.RECIPROCAM


def test_detection_insertion_parameters():
    det_insertion_params = DetectionInsertionParameters(["target_layer_name"], [5, 5, 5])
    assert det_insertion_params.target_layer == ["target_layer_name"]
    assert det_insertion_params.num_anchors == [5, 5, 5]
    assert det_insertion_params.saliency_map_size == (23, 23)
    assert det_insertion_params.embed_normalization
    assert det_insertion_params.explain_method_type == XAIMethodType.DETCLASSPROBABILITYMAP
