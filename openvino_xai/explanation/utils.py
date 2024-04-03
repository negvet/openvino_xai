# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, List

import numpy as np

from openvino_xai.explanation.explanation_parameters import TargetExplainGroup


def select_target_indices(
    target_explain_group: TargetExplainGroup,
    explain_target_indices: Optional[Union[List[int], np.ndarray]] = None,
    total_num_targets: Optional[int] = None,
) -> Union[List[int], np.ndarray]:
    """
    Selects target indices.

    :param target_explain_group: Target explain group.
    :type target_explain_group: TargetExplainGroup
    :param explain_target_indices: Target explain indices.
    :type explain_target_indices: Optional[Union[list, np.ndarray]]
    :param total_num_targets: Total number of targets.
    :type total_num_targets: Optional[int]
    """

    if target_explain_group == TargetExplainGroup.CUSTOM:
        if explain_target_indices is None:
            raise ValueError(f"Explain targets has to be provided for {target_explain_group}.")
        if not total_num_targets:
            raise ValueError("total_num_targets has to be provided.")
        if not all(0 <= target_index <= (total_num_targets - 1) for target_index in explain_target_indices):
            raise ValueError(f"All targets explanation indices have to be in range 0..{total_num_targets - 1}.")
        return explain_target_indices

    raise ValueError(f"Unsupported target_explain_group: {target_explain_group}")
