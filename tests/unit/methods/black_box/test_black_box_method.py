# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.aise import AISEClassification, AISEDetection
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.methods.black_box.rise import RISE
from tests.intg.test_classification import DEFAULT_CLS_MODEL
from tests.intg.test_detection import DEFAULT_DET_MODEL


class InputSampling:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn()

    def get_cls_model(self, fxt_data_root):
        retrieve_otx_model(fxt_data_root, DEFAULT_CLS_MODEL)
        model_path = fxt_data_root / "otx_models" / (DEFAULT_CLS_MODEL + ".xml")
        return ov.Core().read_model(model_path)

    def get_det_model(self, fxt_data_root):
        detection_model = "det_yolox_bccd"
        retrieve_otx_model(fxt_data_root, detection_model)
        model_path = fxt_data_root / "otx_models" / (detection_model + ".xml")
        return ov.Core().read_model(model_path)

    def _generate_with_preset(self, method, preset):
        _ = method.generate_saliency_map(
            data=self.image,
            target_indices=[1],
            preset=preset,
        )

    @staticmethod
    def preprocess_det_fn(x: np.ndarray) -> np.ndarray:
        x = cv2.resize(src=x, dsize=(416, 416))  # OTX YOLOX
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, 0)
        return x

    @staticmethod
    def postprocess_det_fn(x) -> np.ndarray:
        """Returns boxes, scores, labels."""
        # return x["boxes"][:, :4], x["boxes"][:, 4], x["labels"]
        return x["boxes"][0][:, :4], x["boxes"][0][:, 4], x["labels"][0]

class TestAISEClassification(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], [0, 1]])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)

        aise_method = AISEClassification(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            target_indices=target_indices,
            preset=Preset.SPEED,
            num_iterations_per_kernel=10,
            kernel_widths=[0.1],
        )
        assert aise_method.num_iterations_per_kernel == 10
        assert aise_method.kernel_widths == [0.1]

        assert isinstance(saliency_map, dict)
        assert len(saliency_map) == len(target_indices)
        for target in target_indices:
            assert target in saliency_map

        ref_target = 0
        assert saliency_map[ref_target].dtype == np.uint8
        assert saliency_map[ref_target].shape == (224, 224)
        assert (saliency_map[ref_target] >= 0).all() and (saliency_map[ref_target] <= 255).all()

        actual_sal_vals = saliency_map[0][0, :10].astype(np.int16)
        ref_sal_vals = np.array([68, 69, 69, 69, 70, 70, 71, 71, 72, 72], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_preset(self, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)
        method = AISEClassification(model, self.postprocess_fn, self.preprocess_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic

        assert time_speed < time_balance < time_quality


class TestAISEDetection(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], [0, 1]])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_det_model(fxt_data_root)

        aise_method = AISEDetection(model, self.postprocess_det_fn, self.preprocess_det_fn)
        saliency_map = aise_method.generate_saliency_map(
            data=self.image,
            target_indices=target_indices,
            preset=Preset.SPEED,
            num_iterations_per_kernel=10,
            divisors=[5],
        )
        assert aise_method.num_iterations_per_kernel == 10
        assert aise_method.divisors == [5]

        assert isinstance(saliency_map, dict)
        assert len(saliency_map) == len(target_indices)
        for target in target_indices:
            assert target in saliency_map

        ref_target = 0
        assert saliency_map[ref_target].dtype == np.uint8
        assert saliency_map[ref_target].shape == (416, 416)
        assert (saliency_map[ref_target] >= 0).all() and (saliency_map[ref_target] <= 255).all()

        tmp = saliency_map[0]

        actual_sal_vals = saliency_map[0][150, 240:250].astype(np.int16)
        ref_sal_vals = np.array([152, 168, 184, 199, 213, 225, 235, 243, 247, 249], dtype=np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_preset(self, fxt_data_root: Path):
        model = self.get_det_model(fxt_data_root)
        method = AISEDetection(model, self.postprocess_det_fn, self.preprocess_det_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic

        assert time_speed < time_balance < time_quality


class TestRISE(InputSampling):
    @pytest.mark.parametrize("target_indices", [[0], None])
    def test_run(self, target_indices, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)

        rise_method = RISE(model, self.postprocess_fn, self.preprocess_fn)
        saliency_map = rise_method.generate_saliency_map(
            self.image,
            target_indices,
            num_masks=5,
        )

        if target_indices == [0]:
            assert isinstance(saliency_map, dict)
            assert saliency_map[0].dtype == np.uint8
            assert saliency_map[0].shape == (224, 224)
            assert (saliency_map[0] >= 0).all() and (saliency_map[0] <= 255).all()

            actual_sal_vals = saliency_map[0][0, :10].astype(np.int16)
            ref_sal_vals = np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
        else:
            isinstance(saliency_map, np.ndarray)
            assert saliency_map.dtype == np.uint8
            assert saliency_map.shape == (1, 20, 224, 224)
            assert (saliency_map >= 0).all() and (saliency_map <= 255).all()

            actual_sal_vals = saliency_map[0][0][0, :10].astype(np.int16)
            ref_sal_vals = np.array([246, 241, 236, 231, 226, 221, 216, 211, 205, 197], dtype=np.uint8)
            assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    def test_preset(self, fxt_data_root: Path):
        model = self.get_cls_model(fxt_data_root)
        method = RISE(model, self.postprocess_fn, self.preprocess_fn)

        tic = time.time()
        self._generate_with_preset(method, Preset.SPEED)
        toc = time.time()
        time_speed = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.BALANCE)
        toc = time.time()
        time_balance = toc - tic

        tic = time.time()
        self._generate_with_preset(method, Preset.QUALITY)
        toc = time.time()
        time_quality = toc - tic

        assert time_speed < time_balance < time_quality
