# OpenVINO-XAI (OVXAI) usage

OpenVINO-XAI provides a suite of eXplainable AI (XAI) algorithms for explanation of OpenVINO™ Intermediate Representation (IR).
Model explanation helps to identify the parts of the input that are responsible for the model's prediction, 
which is useful for analyzing model's performance.

Current tutorial is primarily for classification CNNs.

OpenVINO-XAI API documentation can be found [here](https://curly-couscous-ovjvm29.pages.github.io/).

Content:
- Explainer as interface to XAI algorithms
- Basic usage (auto mode)
- Advanced usage (white-box)
- Advanced usage (black-box)


## Explainer as interface to XAI algorithms
```python
explainer = Explainer(
    model,
    task_type=TaskType.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)
explanation = explainer(data, explanation_parameters)
```


## Basic usage (auto mode)

Under the hood: will try to run white-box, if fails => will run black-box.

```python
import cv2
import numpy as np
import openvino.runtime as ov

from openvino_xai.common.parameters import TaskType
from openvino_xai.explanation.explainers import Explainer
from openvino_xai.explanation.explanation_parameters import ExplanationParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Explainer object will prepare and load the model once in the beginning
explainer = Explainer(
    model,
    task_type=TaskType.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
explanation_parameters = ExplanationParameters(
    target_explain_indices=[11, 14],  # indices of classes to explain
)
explanation = explainer(image, explanation_parameters)

# Saving saliency maps
explanation.save("output_path", "name")
```


## Advanced usage (white-box mode)

```python
import cv2
import numpy as np
import openvino.runtime as ov

from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.explanation.explainers import Explainer
from openvino_xai.explanation.explanation_parameters import ExplainMode, ExplanationParameters, TargetExplainGroup, PostProcessParameters
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Optional - create insertion parameters
insertion_parameters = ClassificationInsertionParameters(
    # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
    embed_normalization=True,  # True by default.  If set to True, saliency map normalization is embedded in the model
    explain_method_type=XAIMethodType.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
)

# Explainer object will prepare and load the model once in the beginning
explainer = Explainer(
    model,
    task_type=TaskType.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
    explain_mode=ExplainMode.WHITEBOX,
    insertion_parameters=insertion_parameters,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
          'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
explanation_parameters = ExplanationParameters(
    target_explain_group=TargetExplainGroup.CUSTOM,
    target_explain_indices=[11, 14],  # indices of classes to explain
    target_explain_names=voc_labels,
    post_processing_parameters=PostProcessParameters(overlay=True),  # by default, saliency map overlay over image
)
explanation = explainer(image, explanation_parameters)

# Saving saliency maps
explanation.save("output_path", "name")
```


## Advanced usage (black-box mode)

```python
import cv2
import numpy as np
import openvino.runtime as ov

from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.explanation.explainers import Explainer
from openvino_xai.explanation.explanation_parameters import ExplainMode, ExplanationParameters, TargetExplainGroup, PostProcessParameters
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    # Implementing own pre-process function based on model's implementation
    x = cv2.resize(src=x, dsize=(224, 224))
    x = np.expand_dims(x, 0)
    return x

def postprocess_fn(x):
    # Implementing own post-process function based on model's implementation
    return x["logits"]

# Creating model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Explainer object will prepare and load the model once in the beginning
explainer = Explainer(
    model,
    task_type=TaskType.CLASSIFICATION,
    preprocess_fn=preprocess_fn,
    postprocess_fn=postprocess_fn,
    explain_mode=ExplainMode.BLACKBOX,
)

# Generate and process saliency maps (as many as required, sequentially)
image = cv2.imread("path/to/image.jpg")
explanation_parameters = ExplanationParameters(
    target_explain_indices=[11, 14],  # indices of classes to explain
)
explanation = explainer(
    image, 
    explanation_parameters,
    num_masks=1000,  # kwargs of the RISE algo
)

# Saving saliency maps
explanation.save("output_path", "name")
```
