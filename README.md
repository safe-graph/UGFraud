<p align="center">
    <br>
    <a href="https://image.flaticon.com/icons/svg/1671/1671517.svg">
        <img src="https://github.com/safe-graph/UGFraud/blob/master/UGFraud_logo.png" width="500"/>
    </a>
    <br>
<p>
<!-- <p align="center">
    <a href="https://travis-ci.org/github/safe-graph/DGFraud">
        <img alt="PRs Welcome" src="https://travis-ci.org/safe-graph/DGFraud.svg?branch=master">
    </a>
    <a href="https://github.com/safe-graph/DGFraud/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/safe-graph/DGFraud">
    </a>
    <a href="https://github.com/safe-graph/DGFraud/archive/master.zip">
        <img alt="Downloads" src="https://img.shields.io/github/downloads/safe-graph/DGFraud/total">
    </a>
    <a href="https://github.com/safe-graph/DGFraud/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/safe-graph/DGFraud?include_prereleases">
    </a>
</p> -->

<h3 align="center">
<p>An Unsupervised Graph-based Toolbox for Fraud Detection
</h3>

**Introduction:** 



We welcome contributions on adding new fraud detectors and extending the features of the toolbox. Some of the planned features are listed in [TODO list](#todo-list). 

If you use the toolbox in your project, please cite the [paper](https://arxiv.org/abs/2006.06069) below and the [algorithms](#implemented-models) you used :
```bibtex
@inproceedings{dou2020robust,
  title={Robust Spammer Detection by Nash Reinforcement Learning},
  author={Dou, Yingtong and Ma, Guixiang and Yu, Philip S and Xie, Sihong},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```

**Useful Resources**
- [Graph-based Fraud Detection Paper List](https://github.com/safe-graph/graph-fraud-detection-papers) 
- [Awesome Fraud Detection Papers](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)
- [Attack and Defense Papers on Graph Data](https://github.com/safe-graph/graph-adversarial-learning-literature)
- [PyOD: A Python Toolbox for Scalable Outlier Detection (Anomaly Detection)](https://github.com/yzhao062/pyod)
- [PyODD: An End-to-end Outlier Detection System](https://github.com/datamllab/pyodds)
- [DGL: Deep Graph Library](https://github.com/dmlc/dgl)
- [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/)

**Table of Contents**
- [Installation](#installation)
- [User Guide](#user-guide)
- [Implemented Models](#implemented-models)
- [Model Comparison](#model-comparison)
- [TODO List](#todo-list)
- [How to Contribute](#how-to-contribute)


## Installation
```bash
git clone https://github.com/safe-graph/DGFraud.git
cd DGFraud
python setup.py install
```
### Requirements
```bash
* python 3.6, 3.7
* tensorflow>=1.14.0,<2.0
* numpy>=1.16.4
* scipy>=1.2.0
```
### Dataset
#### Yelp dataset
The demo data is not the intact data (`rating` and `date` information are missing). The rating information is only used in ZooBP demo. If you need the intact date to play demo. Please download intact data from [Yelp Spam Review Dataset](http://odds.cs.stonybrook.edu/yelpchi-dataset/). The `.gz` file includes:
- `user_id`: 38063 number of users
- `product_id`: 201 number of products
- `rating`: from 1.0 (low) to 5.0 (high)
- `label`: -1 is not spam, 1 is spam
- `date`: creation time


## User Guide

### Running the example code
You can find the implemented models in `Demo` directory. For example, you can run fBox using:
```bash
python eval_fBox.py 
```

### Running on your datasets
Have a look at the load_data_dblp() function in utils/utils.py for an example.

In order to use your own data, you have to provide:
* adjacency matrices or adjlists (for GAS);
* a feature matrix
* a label matrix
then split feature matrix and label matrix into testing data and training data.

You can specify a dataset as follows:
```bash
python xx_main.py --dataset your_dataset 
```
or by editing xx_main.py

### The structure of code
The repository is organized as follows:
- `algorithms/` contains the implemented models and the corresponding example code;
- `base_models/` contains the basic models (GCN);
- `dataset/` contains the necessary dataset files;
- `utils/` contains:
    * loading and splitting the data (`data_loader.py`);
    * contains various utilities (`utils.py`).


## Implemented Models

| Model  | Paper  | Venue  | Reference  |
|-------|--------|--------|--------|
| **SpEagle** | [Collective Opinion Spam Detection: Bridging Review Networks and Metadata](https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf)  | KDD 2015  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/speagle.txt) |
| **GANG** | [GANG: Detecting Fraudulent Users in Online Social Networks via Guilt-by-Association on Directed Graph](https://ieeexplore.ieee.org/document/8215519)  | ICDM 2017  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gang.txt)|
| **fBox** | [Spotting Suspicious Link Behavior with fBox: An Adversarial Perspective](https://arxiv.org/pdf/1410.3915.pdf)  | ICDM 2014 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fbox.txt) |
| **Fraudar** | [FRAUDAR: Bounding Graph Fraud in the Face of Camouflage](https://bhooi.github.io/papers/fraudar_kdd16.pdf)  | KDD 2016 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fraudar.txt) |
| **ZooBP** | [ZooBP: Belief Propagation for Heterogeneous Networks](http://www.vldb.org/pvldb/vol10/p625-eswaran.pdf)  | VLDB 2017 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/zoobp.txt)  |
| **SVD** | [Singular value decomposition and least squares solutions](https://link.springer.com/content/pdf/10.1007/978-3-662-39778-7_10.pdf)  | - |[BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/svd.txt) |
| **Prior** | Evaluating suspicioueness based on prior information  | - |  - |


## Model Comparison
| Model  | Application  | Graph Type  |
|-------|--------|--------|
| **SpEagle** | Review Spam | Tripartite  |
| **GANG** | Social Sybil  | Bipartite |
| **fBox** | Social Fraudster  | Bipartite | 
| **Fraudar** |  Social Fraudster | Bipartite |
| **ZooBP** | E-commerce Fraud | Tripartite | 
| **SVD** | Dimension Reduction  | Bipartite |
<!--| **HACUD** |  |  |   |-->
<!--| **GraphConsis** | Opinion Fraud  | Homogeneous   | GraphSAGE |-->

## TODO List
- Homogeneous graph implementation


## How to Contribute
You are welcomed to contribute to this open-source toolbox. Currently, you can create issues or send email to [ytongdou@gmail.com](mailto:ytongdou@gmail.com) for enquiry.
