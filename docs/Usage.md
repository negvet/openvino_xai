# OpenVINO-XAI (OVXAI) usage

OVXAI objective is to explain the model.
Model explanation helps to identify the parts of the input that are responsible for the model's prediction, 
which is useful for analyzing model's performance.

Current tutorial is primarily for classification CNNs.

OpenVINO-XAI API documentation can be found [here](https://curly-couscous-ovjvm29.pages.github.io/).

Content:
- Usage in white-box mode
  - Insertion
  - Explanation
- Usage in black-box mode
  - Explanation
- Example scripts

## Usage in white-box mode

White-box regime is a two-step process that includes OV model update and further inference of the updated model.

Updated model has additional XAI branch inserted. XAI branch generates saliency maps during model inference. 
Saliency maps extend the list of model outputs, i.e. saliency maps are generated along with the original model outputs. 
Depending on the white-box algorithm, computational overhead of inserted XAI branch may vary, 
but it is usually relatively modest.

Insertion performed in such a way that both the original model and the updated model are inferable 
using the same inference pipeline.

### Insertion: insert XAI branch into the model

The use case implies that the user has an OV model. `insert_xai()` API is used for insertion.

```python
import openvino.runtime as ov
import openvino_xai as ovxai
from openvino_xai.common.parameters import TaskType, XAIMethodType
from openvino_xai.insertion.insertion_parameters import ClassificationInsertionParameters


# Creating original model
model = ov.Core().read_model("path/to/model.xml")  # type: ov.Model

# Optional - create insertion parameters
insertion_parameters = ClassificationInsertionParameters(
    # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
    embed_normalization=True,  # True by default.  If set to True, saliency map normalization is embedded in the model
    explain_method_type=XAIMethodType.RECIPROCAM,  # ReciproCAM is the default XAI method for CNNs
)

# Inserting XAI branch into the model graph
model_xai = ovxai.insert_xai(
    model=model,
    task_type=TaskType.CLASSIFICATION,
    insertion_parameters=insertion_parameters,
)  # type: ov.Model

# ***** Downstream task: user's code that infers model_xai and picks 'saliency_map' output *****
```

### Explanation: generate explanation

The use case implies that the user has model inference pipeline: compiled model, preprocessing, postprocessing, etc.
Inference pipeline is used to infer the model and retrieve inference result. 
Saliency maps are retrieved from the inference result.

#### Get raw saliency maps

The use case uses the original model inference pipeline without modification.

```python
import cv2
import numpy as np
import openvino.runtime as ov


# Original user's code that defines a model_inferrer (it can be a function, class, or just a script)
def model_inferrer(image: np.ndarray) -> OVDict:
    image_processed = cv2.resize(src=image, dsize=(224, 224))
    compiled_model = ov.Core().compile_model(model_xai, "CPU")
    result = compiled_model([image_processed])
    return result

# Execute inference call
result = model_inferrer(cv2.imread("path/to/image.jpg"))

# Get model predictions
logits = result["logits"]  # "logits" is an original model prediction w/o modification

# Get raw saliency map
raw_saliency_map = result["saliency_map"]  # "saliency_map" is an additional model output added during insertion
raw_saliency_map: np.ndarray  # e.g. 20x7x7 uint8 array for 20 classes, where 7x7 is the spacial size of the last conv layer
```

#### Get processed saliency maps (optional)

The use case implies that the user has a model_inferrer or a model wrapper, 
which is a callable object and incorporates an inference pipeline.

Current use case lightly modifies original model inference pipeline, 
which is consumed by `explain()` API to generate explanation results (see example below).
Explain API is capable of selecting target classes to explain and applying postprocessing. 
Processed saliency maps can be dumped to the disk.

```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as ovxai
from openvino_xai.explanation.explanation_parameters import (
    ExplainMode, 
    PostProcessParameters,
    TargetExplainGroup, 
    ExplanationParameters,
)
from openvino_xai.explanation.utils import InferenceResult


# User's code that defines a callable model_inferrer with InferenceResult output
def model_inferrer(image: np.ndarray) -> InferenceResult:
    image_processed = cv2.resize(src=image, dsize=(224, 224))
    compiled_model = ov.Core().compile_model(model_xai, "CPU")
    result = compiled_model([image_processed])
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    logits = result["logits"]
    logits = softmax(logits)
    raw_saliency_map = result["saliency_map"]  # "saliency_map" is an additional output added during insertion

    # Create InferenceResult object
    inference_result = InferenceResult(prediction=logits, saliency_map=raw_saliency_map)
    return inference_result

# Optional - create explanation parameters
explanation_parameters = ExplanationParameters(
    explain_mode=ExplainMode.WHITEBOX,  # by default, run white-box XAI (expect XAI branch is inserted into the model)
    target_explain_group=TargetExplainGroup.PREDICTIONS,  # by default, explain only predicted classes
    post_processing_parameters=PostProcessParameters(overlay=True),  # by default, saliency map overlay over image
)

# Generate processed saliency map via .explain(model_inferrer, image) call
explanation = ovxai.explain(
    model_inferrer=model_inferrer,
    data=cv2.imread("path/to/image.jpg"),
    explanation_parameters=explanation_parameters,
)  # type: ExplanationResult
# explanation.saliency_map  # type Dict[int: np.ndarray]  # key - class id, value - processed saliency map e.g. 3x354x500

explanation.save("output_path_wb", "image_name")
```

`ExplanationParameters` by default consider only predicted classes as target classes to explain. 
Assuming that the model predicted two classes with indices 11 and 14, 
 the corresponding processed saliency maps will be saved 
to `/output_path/image_name_target11.jpg` and `/output_path/image_name_target14.jpg`.




## Usage in black-box mode

Black-box regime is a one-step process - 
the model inference pipeline (with the original model, no prior insertion performed) is provided to the explain API.
Black-box approaches are based on the perturbation of the input data and measurement of the model's output change.
The process is repeated many times, which requires hundreds or thousands of forward passes 
and introduces significant computational overhead.

### Explanation: generate explanation

Model inferrer requirements are similar to [white box explanation](Usage.md#get-processed-saliency-maps-(optional)).
See below for an example.

```python
import cv2
import numpy as np
import openvino.runtime as ov


# Create original model
model: ov.Model
model = ov.Core().read_model("path/to/model.xml")

# Compile original model (no XAI branch inserted)
compiled_model = ov.Core().compile_model(model, "CPU")

# User's code that defines a callable model_inferrer with InferenceResult output
def model_inferrer(image: np.ndarray) -> InferenceResult:
    image_processed = cv2.resize(src=image, dsize=(224, 224))
    compiled_model = ov.Core().compile_model(model_xai, "CPU")
    result = compiled_model([image_processed])
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    logits = rsult["logits"]
    logits = softmax(logits)

    # Create InferenceResult object, w/o saliency map.
    # Saliency map can be available in InferenceResult, but will be ignored when explain_mode=ExplainMode.BLACKBOX
    inference_result = InferenceResult(prediction=logits, saliency_map=None)
    return inference_result

# Generate explanation
explanation_parameters = ExplanationParameters(
    explain_mode=ExplainMode.BLACKBOX,  # Black-box XAI method will be used under .explain() call
)
explanation = ovxai.explain(
    model_inferrer=model_inferrer,
    data=cv2.imread("path/to/image.jpg"),
    explanation_parameters=explanation_parameters,
)  # type: ExplanationResult
# explanation.saliency_map  # type Dict[int: np.ndarray]  # key - class id, value - processed saliency map e.g. 3x354x500

explanation.save("output_path_bb", "image_name")
```


## Example scripts
More usage scenarios are available in [examples](./../examples).

```python
# Retrieve models by running tests
# Models are downloaded and stored in .data/otx_models
pytest tests/test_classification.py

# Run a bunch of classification examples
# All outputs will be stored in the corresponding output directory
python examples/run_classification.py .data/otx_models/mlc_mobilenetv3_large_voc.xml \
tests/assets/cheetah_person.jpg --output output
```







## Updated usage proposal (following request from Mark and Minje)

Insert API, should be preserved:
```python
model_xai = ovxai.insert(model)
```

Explain API:
```python
explanation = ovxai.explain(model, data)  # Not possible, inference code required
```

To be able to output explanation in one step (from the user perspective), explain api should have access to: 
the model, model load code, model inference code (at least), etc.

Requirements for the model inferrer:
```python
from abc import ABC, abstractmethod


class ModelInferrer(ABC):
    @abstractmethod
    def __call__(
            self, x: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> InferenceResult:
        """Implements forward pass."""
        raise NotImplementedError


class WhiteBoxModelInferrer(ModelInferrer, ABC):
    def get_model(self) -> ov.Model:
        """Get the model."""
        raise NotImplementedError
    
    @abstractmethod
    def set_model(self, model: ov.Model):
        """Update the model."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        """Loads the model: compile, create infer queue, etc."""
        raise NotImplementedError


class BlackBoxModelInferrer(ModelInferrer, ABC):
    pass
```

Usage example:
```python
import cv2
import numpy as np
import openvino.runtime as ov

import openvino_xai as ovxai
from openvino_xai.explanation.utils import InferenceResult


class CustomModelInferrer(WhiteBoxModelInferrer):
    def __init___(self, model: ov.Model):
        self.model = model

    def get_model(self):
        return self.model
        
    def set_model(self, model: ov.Model):
        self.model = model
        
    def load_model(self):
        self.compiled_model = ov.Core().compile_model(self.model, "CPU")
        self.async_queue = AsyncInferQueue(self.compiled_model)  # for example, optional
        
    def __call__(
            self, x: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> InferenceResult:
        # ***** Start of user's forward pass code *****
        # Forward pass code:
        # - preprocessing
        # - inference
        # - postprocessing
        # ***** End of user's forward pass code *****
        
        prediction = processed_logits
        raw_saliency_map = raw_model_output["saliency_map"]
        
        # Create InferenceResult object
        inference_result = InferenceResult(prediction=prediction, saliency_map=raw_saliency_map)
        return inference_result


model = ov.Core().read_model("path/to/model.xml") 
model_inferrer = CustomModelInferrer(model)

explanation = ovxai.explain(
    model_inferrer=model_inferrer,
    data=cv2.imread("path/to/image.jpg"),
)
```

Pros:
- simple explain call
- Can supports auto-explain mode

Cons:
- Significant effort for the preparation of the model inference code
