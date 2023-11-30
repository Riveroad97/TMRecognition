# TMRecognition
TMRecognition is a model for Tympanomastoidectomy Surgical Phase Recognition using deep learning.

## Overview

<img src="images/recognition_overview.jpg" alt="TMRecognition" width="100%" height="100%"/>

## Training

For teacher network
```python
python train.py -c configs/train/ent_6class_labeled.json
```
For student network
```python
python train.py -c configs/train/ent_6class_soft_pseudo.json
```
