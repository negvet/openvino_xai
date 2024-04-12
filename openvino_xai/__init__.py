# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Openvino-XAI library for explaining OpenVINO™ IR models.
"""


from .insertion import insert_xai

__all__ = [
    "insert_xai",
]
