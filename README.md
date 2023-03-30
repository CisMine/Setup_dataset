<p align="center">
 <h1 align="center">Setup dataset from scratch </h1>
</p>

## Introduction
Yes, it is true that there are many frameworks available for setting up datasets, such as `ImageFolder`, which is commonly used in computer vision tasks. However, in the real world, the data you may encounter `may not always be in a format` that can be directly used with these frameworks. This would require you to build your `own dataset and preprocess your data accordingly` 


## How to use my code
With my code, you can:
* Use your dataset in your path folder
* You can customize your preprocessing steps based on the specific needs of your task

## Requirements:
* Python
* Torch
* PIL
* Torchvision

## Explain Code
first you import these:
`import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize`
