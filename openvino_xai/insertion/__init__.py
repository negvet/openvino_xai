# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Interface for inserting XAI branch into OV IR.
"""
from openvino_xai.insertion.insert_xai_into_model import insert_xai
from openvino_xai.insertion.insertion_parameters import (
    ClassificationInsertionParameters,
    DetectionInsertionParameters,
    InsertionParameters,
)

__all__ = [
    "insert_xai",
    "InsertionParameters",
    "ClassificationInsertionParameters",
    "DetectionInsertionParameters",
]
