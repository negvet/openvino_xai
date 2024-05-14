<!-- markdownlint-disable -->
TODO (Galina): enable

# OpenVINO-XAI: Classification explanation

This notebook shows an example how to use OpenVINO-XAI.

**OpenVINO-XAI** library is a tool that provides a suite of Explainable AI (XAI) algorithms for explanation of
[OpenVINO™](https://github.com/openvinotoolkit/openvino) Intermediate Representation (IR) models.

It depicts a heatmap with areas of interest where neural network (classification or detection) focuses before making a desicion. 

Example: Saliency map for `person` class for EfficientV2 classification model:

![Saliency Map Example](../../docs/images/saliency_map_person.png)

## Notebook Contents

The tutorial consists of the following steps:

- Update IR model with XAI branch to receive saliency maps
- Create CustomInferrer to infer model and receive outputs
- Explain model 
- Adding ImageNet label names to add them in saliency maps
- Saliency map examples in different usecases and its interpretations
    - True Positive High confidence
    - True Positive Low confidence   
    - False Positive High confidence  
    - Two mixed predictions


## Installation Instructions

```python
# Create virtual env
!python3 -m venv .ovxai

# Activate virtual env
!source .ovxai/bin/activate

# Package installation
%pip install -q ../..

```