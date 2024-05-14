# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from openvino_xai.common.utils import logger
from openvino_xai.explanation.explanation_parameters import (
    SaliencyMapLayout,
    TargetExplainGroup,
)
from openvino_xai.explanation.utils import get_explain_target_indices


class ExplanationResult:
    """
    ExplanationResult selects target saliency maps, holds it and its layout.

    :param saliency_map: Raw saliency map.
    :param target_explain_group: Defines targets to explain: all, only predictions, custom list, per-image.
    :type target_explain_group: TargetExplainGroup
    :param target_explain_labels: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type target_explain_labels: List[int | str] | None
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    """

    def __init__(
        self,
        saliency_map: np.ndarray,
        target_explain_group: TargetExplainGroup,
        target_explain_labels: List[int | str] | None = None,
        label_names: List[str] | None = None,
    ):
        self._check_saliency_map(saliency_map)
        self._saliency_map = self._format_sal_map_as_dict(saliency_map)

        if "per_image_map" in self._saliency_map:
            self.layout = SaliencyMapLayout.ONE_MAP_PER_IMAGE_GRAY
            if target_explain_group != TargetExplainGroup.IMAGE:
                logger.warning(
                    f"Setting target_explain_group to TargetExplainGroup.IMAGE, {target_explain_group} "
                    f"is not supported when only single (global) saliency map per image is available."
                )
            self.target_explain_group = TargetExplainGroup.IMAGE
        else:
            if target_explain_group == TargetExplainGroup.IMAGE:
                raise ValueError(
                    "TargetExplainGroup.IMAGE supports only single (global) saliency map per image. "
                    "But multiple saliency maps are available."
                )
            self.layout = SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
            self.target_explain_group = target_explain_group

        if self.target_explain_group == TargetExplainGroup.CUSTOM:
            self._saliency_map = self._select_target_saliency_maps(target_explain_labels, label_names)

        self.label_names = label_names

    @property
    def saliency_map(self) -> Dict[int | str, np.ndarray]:
        """Saliency map as a dict {map_id: np.ndarray}."""
        return self._saliency_map

    @saliency_map.setter
    def saliency_map(self, saliency_map: Dict[int | str, np.ndarray]):
        self._saliency_map = saliency_map

    @property
    def sal_map_shape(self):
        idx = next(iter(self._saliency_map))
        sal_map_shape = self._saliency_map[idx].shape
        return sal_map_shape

    @staticmethod
    def _check_saliency_map(saliency_map: np.ndarray):
        if saliency_map is None:
            raise RuntimeError("Saliency map is None.")
        if not isinstance(saliency_map, np.ndarray):
            raise ValueError(f"Raw saliency_map has to be np.ndarray, but got {type(saliency_map)}.")
        if saliency_map.size == 0:
            raise RuntimeError("Saliency map is zero size array.")
        if saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for saliency maps should be 1.")

    @staticmethod
    def _format_sal_map_as_dict(raw_saliency_map: np.ndarray) -> Dict[int | str, np.ndarray]:
        """Returns dict with saliency maps in format {target_id: class_saliency_map}."""
        dict_sal_map: Dict[int | str, np.ndarray]
        if raw_saliency_map.ndim == 3:
            # Per-image saliency map
            dict_sal_map = {"per_image_map": raw_saliency_map[0]}
        elif raw_saliency_map.ndim == 4:
            # Per-target saliency map
            dict_sal_map = {}
            for index, sal_map in enumerate(raw_saliency_map[0]):
                dict_sal_map[index] = sal_map
        else:
            raise ValueError(
                f"Raw saliency map has to be tree or four dimensional tensor, " f"but got {raw_saliency_map.ndim}."
            )
        return dict_sal_map

    def _select_target_saliency_maps(
        self,
        target_explain_labels: List[int | str] | None = None,
        label_names: List[str] | None = None,
    ) -> Dict[int | str, np.ndarray]:
        assert self.layout == SaliencyMapLayout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        explain_target_indices = self._select_target_indices(
            self.target_explain_group,
            target_explain_labels,
            label_names,
            len(self._saliency_map),
        )
        saliency_maps_selected = {i: self._saliency_map[i] for i in explain_target_indices}
        return saliency_maps_selected

    @staticmethod
    def _select_target_indices(
        target_explain_group: TargetExplainGroup,
        target_explain_labels: List[int | str] | None = None,
        label_names: List[str] | None = None,
        total_num_targets: int | None = None,
    ) -> List[int] | np.ndarray:
        explain_target_indices = get_explain_target_indices(target_explain_labels, label_names)

        if target_explain_labels is None:
            raise ValueError(f"Explain labels has to be provided for {target_explain_group}.")
        if not total_num_targets:
            raise ValueError("total_num_targets has to be provided.")
        if not all(0 <= target_index <= (total_num_targets - 1) for target_index in explain_target_indices):
            raise ValueError(f"All targets explanation indices have to be in range 0..{total_num_targets - 1}.")
        return explain_target_indices

    def save(self, dir_path: Path | str, name: str | None = None) -> None:
        """Dumps saliency map."""
        # TODO: add unit test
        os.makedirs(dir_path, exist_ok=True)
        save_name = f"{name}_" if name else ""
        for i, (cls_idx, map_to_save) in enumerate(self._saliency_map.items()):
            if isinstance(cls_idx, str):
                target_name = cls_idx
            else:
                if self.label_names:
                    target_name = self.label_names[cls_idx]
                else:
                    target_name = str(cls_idx)
            cv2.imwrite(os.path.join(dir_path, f"{save_name}target_{target_name}.jpg"), img=map_to_save)