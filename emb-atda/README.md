## Emb-Atda Evaluation
Emb-atda is a "unsupervised" domain adaptation algorithm based on "[Asymmetric Tri-training for Unsupervised Domain Adaptation (atda)](https://arxiv.org/pdf/1702.08400.pdf)". Atda algorithm was modified such that it could be ported to an embedded device or a microcontroller.

## Requirements / Preparation
### Software
 - python 3.7.3
 - tensorflow 1.14.0
 - keras 2.2.4
### Data
```bash
$ cd data
```
Download the BSDS500 dataset:
```bash
$ curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
```
Create the mnistm dataset with:
```bash
$ python3 create_mnistm.py
```

## Run emb-atda on MNIST &rarr; MNISTM
In order to test the emb-atda algorithm run:
```bash
$ cd ..
$ python3 mnist2mnistm.py Load noPlot
```

**OR**
```bash
$ cd ..
$ python3 mnist2mnistm.py Train Plot
```
to retrain the neural network on source domain data and enable plots
