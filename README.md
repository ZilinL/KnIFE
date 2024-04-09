# KnIFE：<ins>Kn</ins>owledge Distillation-based Phase <ins>I</ins>nvariant <ins>F</ins>eature <ins>E</ins>xtraction

## Introduction

The repository contains the implementation of "Domain Generalization for Zero-calibration BCIs with Knowledge Distillation-based Phase Invariant Feature Extraction", which is currently under review.

This is a demo of the proposed Knife, and more complete code will come soon.

alg/algs/Knife.py contains the core code of the proposed method, which includes the realization of knowledge distillation framework, Correlation alignment, and spectrum transfer.
The graphical abstract is shown below:

![GA](https://github.com/ZilinL/KnIFE/assets/10232596/5509b800-2ae4-47cc-ab61-00a4d9d19d94)

## Environments
```python
pip install -r requirements.txt
```

## Run the code
train_OpenBMI.py: A demo to run Knife on OpenBMI dataset.

## Datasets
1. [BCI competition IV-2a](https://www.bbci.de/competition/iv/#dataset2a)
2. [BCI competition IV-2b](https://www.bbci.de/competition/iv/#dataset2b)
3. [OpenBMI](http://gigadb.org/dataset/view/id/100542)

Please request data from the above link.

An example dataset used for train_OpenBMI.py: [OpenBMI_GoogleDrive](https://drive.google.com/drive/folders/1BtFluXOPe8Dk2Yee7zICE9gG7NM8lNwW?usp=sharing)

Put the downloaded OpenBMI data into the data/OpenBMI/filterdMat/.

Note: This dataset is for demo purposes only. For further use of the data, please request for authorization from the original source.

## Acknowledge
Greant thanks to [deepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG). We extend our method based on this toolkit and have compared and validated our method on it.

To be continued...
