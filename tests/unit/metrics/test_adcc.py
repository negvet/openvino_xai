import json
from typing import Callable, List, Mapping

import cv2
import numpy as np
import openvino as ov
import pytest

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.explanation import Explanation
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
    sigmoid,
)
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.metrics.adcc import ADCC
from openvino_xai.metrics.insertion_deletion_auc import InsertionDeletionAUC
from openvino_xai.metrics.pointing_game import PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

MODEL_NAME = "mlc_mobilenetv3_large_voc"


class TestADCC:
    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        self.model = ov.Core().read_model(model_path)
        self.explainer = Explainer(
            model=self.model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )
        self.adcc = ADCC(self.model, self.preprocess_fn, self.postprocess_fn, self.explainer)

    def test_adcc_init_wo_explainer(self):
        adcc_wo_explainer = ADCC(self.model, self.preprocess_fn, self.postprocess_fn)
        assert isinstance(adcc_wo_explainer.explainer, Explainer)

    def test_adcc(self):
        input_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        saliency_map = np.random.rand(224, 224)

        complexity_score = self.adcc.complexity(saliency_map)
        assert complexity_score >= 0.2

        model_output = self.adcc.model_predict(input_image)
        class_idx = np.argmax(model_output)

        average_drop_score = self.adcc.average_drop(saliency_map, class_idx, input_image, model_output)
        assert average_drop_score >= 0.2

        coherency_score = self.adcc.coherency(saliency_map, class_idx, input_image)
        assert coherency_score >= 0.2

        adcc_score = self.adcc(saliency_map, class_idx, input_image)["adcc"]
        assert adcc_score >= 0.4

    def test_evaluate(self):
        input_images = [np.random.rand(224, 224, 3) for _ in range(5)]
        explanations = [
            Explanation({0: np.random.rand(224, 224), 1: np.random.rand(224, 224)}, targets=[0, 1]) for _ in range(5)
        ]

        adcc_score = self.adcc.evaluate(explanations, input_images)["adcc"]

        assert isinstance(adcc_score, float)
        assert 0 <= adcc_score <= 1
