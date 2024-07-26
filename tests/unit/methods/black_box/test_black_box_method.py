# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.aise import AISE
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.methods.black_box.rise import RISE
from tests.intg.test_classification import DEFAULT_CLS_MODEL


class InputSampling:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn()

    def get_model(self, fxt_data_root):
        retrieve_otx_model(fxt_data_root, DEFAULT_CLS_MODEL)
        model_path = fxt_data_root / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        return ov.Core().read_model(model_path)


class TestAISE(InputSampling):
    @pytest.mark.parametrize("explain_target_indices", [[0], [0, 1]])
    def test_run(self, explain_target_indices, fxt_data_root: Path):
        model = self.get_model(fxt_data_root)

        aise_method = AISE(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            explain_target_indices=explain_target_indices,
            preset=Preset.SPEED,
            num_iterations_per_kernel=10,
            kernel_widths=[0.1],
        )
        assert aise_method.num_iterations_per_kernel == 10
        assert aise_method.kernel_widths == [0.1]

        assert isinstance(saliency_map, dict)
        assert len(saliency_map) == len(explain_target_indices)
        for target in explain_target_indices:
            assert target in saliency_map

        ref_target = 0
        assert saliency_map[ref_target].dtype == np.uint8
        assert saliency_map[ref_target].shape == (224, 224)
        assert (saliency_map[ref_target] >= 0).all() and (saliency_map[ref_target] <= 255).all()

        actual_sal_vals = saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = np.array([68, 69, 69, 69, 70, 70, 71, 71, 72, 72], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)


class TestRISE(InputSampling):
    @pytest.mark.parametrize("explain_target_indices", [[0], None])
    def test_run(self, explain_target_indices, fxt_data_root: Path):
        model = self.get_model(fxt_data_root)

        rise_method = RISE(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = rise_method.generate_saliency_map(
            self.image,
            explain_target_indices,
            num_masks=5,
        )

        assert saliency_map.dtype == np.uint8
        assert saliency_map.shape == (1, 20, 224, 224)
        assert (saliency_map >= 0).all() and (saliency_map <= 255).all()

        actual_sal_vals = saliency_map[0][0][0, :10].astype(np.int16)
        ref_sal_vals = np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
