# XAI method User Guide

## Method overviews

At the moment, the following XAI methods are supported:

| Method                 | Using model internals | Per-target support | Single-shot | #Model inferences |
|------------------------|-----------------------|--------------------|-------------|-------------------|
| White-Box              |                       |                    |             |                   |
| Activation Map         | Yes                   | No                 | Yes         | 1                 |
| Recipro-CAM            | Yes                   | Yes (class)        | Yes*        | 1*                |
| ViT Recipro-CAM        | Yes                   | Yes (class)        | Yes*        | 1*                |
| DetClassProbabilityMap | Yes**                 | Yes (class)        | Yes         | 1                 |
| Black-Box              |                       |                    |             |                   |
| RISE                   | No                    | Yes (class)        | No          | 1000-10000        |
| AISEClassification     | No                    | Yes (class)        | No          | 120-500           |
| AISEDetection          | No                    | Yes (bbox)         | No          | 60-250            |

\* Recipro-CAM re-infers part of the graph (usually neck + head or last transformer block) H*W times, where HxW – feature map size of the target layer.

** DetClassProbabilityMap requires explicit definition of the target layers.
The rest of the white-box methods support automatic detection of the target layer.

Target layer is the part of the model graph where XAI branch will be inserted (applicable for white-box methods).

All supported methods are gradient-free, which suits deployment framework settings (e.g. OpenVINO™), where the model is in optimized or compiled representation.

## White-Box methods

When to use?
- When model architecture follows standard CNN-based or ViT-based design (OV-XAI support 1000+ CNN and ViT models TODO:LINK).
- When speed matters. White-box methods is fast - it takes ~one model inference to generate saliency map.
- When it is required to obtain saliency map together with model prediction at the inference server environment. White-box methods update model graph, so that the XAI branch and saliency map output added to the model. Therefore, with a minor compute overhead, it is possible to generate both model predictions and saliency maps.

All white-box methods require access to model internal state. To generate saliency map, supported white-box methods potentially change and process internal model activations in a way that fosters compute efficiency.

### Activation Map

Suitable for:
- Binary classification problems (e.g. inspecting model reasoning when predicting a positive class).
- Visualization of the global (class-agnostic) model attention (e.g. inspecting which input pixels are the most salient for the model).

Activation Map is the most basic and naive approach. It takes the outputs of the model’s feature extractor (backbone) and averages it in the channel dimension. The results highly rely on the backbone and ignore neck and head computations. Basically, it gives a relatively good and fast result, which highlight the most activated features from the backbone perspective.

Below saliency map was obtained for [ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k) from [timm](https://huggingface.co/timm):

![OpenVINO XAI Architecture](_static/map_samples/ActivationMap_resnet18.a1_in1k_activation_map.jpg)

### Recipro-CAM (ViT Recipro-CAM for ViT models)

Suitable for:
- Almost all CNN-based architectures.
- Many ViT-based architectures.

[Recipro-CAM](../../openvino_xai/methods/white_box/recipro_cam.py) involves spatially masking of the extracted feature maps to exploit the correlation between activation maps and model predictions for target classes. It is perturbation-based method which perturbs internal model activations.

Assume 7x7 feature map which is extracted by the CNN backbone. One location of the feature map is preserved (e.g. at index [0, 0]), while the rest feature map values is masked out with e.g. zeros (perturbation is the same across channel dimension). Perturbed feature map inferred through the model head. The the model prediction scores are used as saliency scores for index [0, 0]. This is repeated for all 49 spatial location. The final saliency map obtained after resizing and scaling. See [paper](https://arxiv.org/abs/2209.14074) for more details.

`Recipro-CAM` is an efficient XAI method.
The main weak point is that saliency for each pixel in the feature map space is estimated in isolation, without taking into account joint contribution of different pixels/features.

`Recipro-CAM` is the default method for the classification task. [ViT Recipro-CAM](../../openvino_xai/methods/white_box/recipro_cam.py) is a modification of Recipro-CAM for ViT-based models.

Below saliency map was obtained for [ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k) from [timm](https://huggingface.co/timm) and "cheetah" class:

![OpenVINO XAI Architecture](_static/map_samples/ReciproCAM_resnet18.a1_in1k_293.jpg)


### DetClassProbabilityMap

Suitable for:
- Single-stage object detection models.
- When it is enough to estimate saliency maps per-class.

[DetClassProbabilityMap](../../openvino_xai/methods/white_box/det_class_probability_map.py) takes the raw classification head output and uses class probability maps to calculate regions of interest for each class. So, it creates different salience maps for each class. This algorithm is implemented for single-stage detectors only and required explicit list of target layers.

The main limitation of this method is that, due to the training loss design of most single-stage detectors, activation values drift towards the center of the object while propagating through the network. Many object detectors, while being designed to precisely estimate location of the objects, might mess up spatial location of object features in the latent space.

Below saliency map was obtained for `YOLOX` trained in-house on PASCAL VOC dataset:

![OpenVINO XAI Architecture](_static/map_samples/DetClassProbabilityMap.jpg)

## Black-Box methods

When to use?
- When custom models are used and/or white-box methods fail (e.g. Swin-based transformers).
- When more advanced model explanation is required. See more details below (e.g. in the RISE overview).
- When spacial location of the features is messed up in the latent space (e.g. some single-stage object detection models).

All black-box methods are perturbation-based - they perturb the input and register the change in the output.
Usually, for high quality saliency map, hundreds or thousands of model inferences required. That is the reason for them to be compute-inefficient. On the other hand, black box methods are model-agnostic.

Given that the quality of the saliency maps usually correlates with the number of available inferences, we propose the following presets for the black-box methods: `Preset.SPEED`, `Preset.BALANCE`, `Preset.QUALITY` (`Preset.BALANCE` is used by default).
Apart from that, methods parameters can be defined directly via Explainer or Method API.

### RISE

Suitable for:
- All classification models which can generate per-class prediction scores.
- More flexibility and more advanced use cases (e.g. control of granularity of the saliency map).

[RISE](../../openvino_xai/methods/black_box/rise.py) probes a model by sub-sampling the input image via random masks and records its response to each of them.
RISE creates random masks from down-scaled space (e.g. 7×7 grid) and adds random translation shifts for the pixel-level explanation with further up-sampling. Weighted sum of all sampled masks used to generate the fine-grained saliency map.
Since it approximates the true saliency map with Monte Carlo sampling, it requires multiple thousands of forward passes to generate a fine-grained saliency map. RISE is a non-deterministic method.
See [paper](https://arxiv.org/abs/1806.07421v3) for more details.

`RISE` generates saliency maps for all classes at once, although indices of target classes can be provided (which might bring some performance boost).

`RISE` has two main hyperparameter: `num_cells` (define the resolution of the grid for the masks) and `num_masks` (defines number of inferences).
Number of cells defines granularity of the saliency map (usually, the higher the granularity - the better). Higher number of cells require more masks to converge. Going from `Preset.SPEED` to `Preset.QUALITY`, the number of masks (compute budget) increases.

Below saliency map was obtained for [ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k) from [timm](https://huggingface.co/timm) and "cheetah" class (default parameters).

![OpenVINO XAI Architecture](_static/map_samples/RISE_resnet18.a1_in1k_293.jpg)

It is possible to see, that some grass-related pixels from the left cheetah also contribute to the cheetah prediction, which might indicates that model learned cheetah features in combination with grass (which makes sense).

### AISEClassification

Suitable for:
- All classification models which can generate per-class prediction scores.
- Cases when speed matters.

`AISE` formulates saliency map generation as a kernel density estimation (KDE) problem, and adaptively sample input masks using a derivative-free optimizer to maximize mask saliency score. KDE requires a proper kernel width, which is not known. A set of pre-defined kernel widths is used simultaneously, and the result is them aggregated. This adaptive sampling mechanism improves the efficiency of input mask generation and thus increases convergence speed. AISE is designed to be task-agnostic and can be applied to a wide range of classification and object detection architectures.
`AISE` is optimized for generating saliency map for a specific class (or a few classes). To specify target classes, use targets argument.

[AISEClassification](../../openvino_xai/methods/black_box/aise/classification.py) is designed for classification models.

Below saliency map was obtained for [ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k) from [timm](https://huggingface.co/timm) and "cheetah" class:

![OpenVINO XAI Architecture](_static/map_samples/AISE_resnet18.a1_in1k_293.jpg)

### AISEDetection

Suitable for:
- All detection models which can generate bounding boxes, labels and scores.
- When speed matters.
- When it is required to get per-box saliency map.

[AISEDetection](../../openvino_xai/methods/black_box/aise/detection.py) is designed for detection models and support per-bounding box saliency maps.

Below saliency map was obtained for `YOLOX` trained in-house on PASCAL VOC dataset (with default parameters, `Preset.BALANCE):

![OpenVINO XAI Architecture](_static/map_samples/AISEDetection.jpg)
