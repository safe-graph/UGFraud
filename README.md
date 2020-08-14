<p align="center">
    <br>
    <a href="https://image.flaticon.com/icons/svg/1671/1671517.svg">
        <img src="https://github.com/safe-graph/UGFraud/blob/master/UGFraud_logo.png" width="400"/>
    </a>
    <br>
<p>
<p align="center">
    <a href="https://travis-ci.org/github/safe-graph/UGFraud">
        <img alt="Building" src="https://travis-ci.org/safe-graph/UGFraud.svg?branch=master">
    </a>
    <a href="https://github.com/safe-graph/UGFraud/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/safe-graph/UGFraud">
    </a>
    <a href="https://pepy.tech/project/ugfraud">
        <img alt="Downloads" src="https://pepy.tech/badge/ugfraud">
    </a>
    <a href="https://pypi.org/project/UGFraud/">
        <img alt="Pypi version" src="https://img.shields.io/pypi/v/ugfraud">
    </a>
</p>

<h3 align="center">
<p>An Unsupervised Graph-based Toolbox for Fraud Detection
</h3>

**Introduction:** 
UGFraud is an unsupervised graph-based fraud detection toolbox that integrates several state-of-the-art graph-based fraud detection algorithms. It can be applied to bipartite graphs (e.g., user-product graph), and it can estimate the suspiciousness of both nodes and edges. The implemented models can be found [here](#implemented-models).

The toolbox incorporates the Markov Random Field (MRF)-based algorithm, dense-block detection-based algorithm, and SVD-based algorithm. For MRF-based algorithms, the users only need the graph structure and the prior suspicious score of the nodes as the input. For other algorithms, the graph structure is the only input.

Meanwhile, we have a [deep graph-based fraud detection toolbox](https://github.com/safe-graph/DGFraud) which implements state-of-the-art graph neural network-based fraud detectors.

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
- [DGFraud: A Deep Graph-based Fraud Detection Toolbox](https://github.com/safe-graph/DGFraud)
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
You can install UGFraud from `pypi`:

```bash
pip install UGFraud
```

or download and install from `github`:

```bash
git clone https://github.com/safe-graph/UGFraud.git
cd UGFraud
python setup.py install
```

### Dataset
The demo data is not the intact data (`rating` and `date` information are missing). The rating information is only used in ZooBP demo. If you need the intact date to play demo, please email [bdscsafegraph@gmail.com](mailto:bdscsafegraph@gmail.com) to download the intact data from [Yelp Spam Review Dataset](http://odds.cs.stonybrook.edu/yelpchi-dataset/). The `metadata.gz` file in `/UGFraud/Yelp_Data/YelpChi` includes:
- `user_id`: 38063 number of users
- `product_id`: 201 number of products
- `rating`: from 1.0 (low) to 5.0 (high)
- `label`: -1 is not spam, 1 is spam
- `date`: data creation time


## User Guide

### Running the example code
You can find the implemented models in `/UGFraud/Demo` directory. For example, you can run fBox using:
```bash
python eval_fBox.py 
```

### Running on your datasets
Have a look at the `/UGFraud/Demo/data_to_network_graph.py` to convert your own data into a graph ([networkx graph](https://networkx.github.io/documentation/stable/tutorial.html#creating-a-graph).

In order to use your own data, you have to provide the following information at least:
* a dict of dict:
```
'user_id':{
        'product_id':
                {
                'label': 1
                }
```
* a dict of prior

You can use `dict_to networkx(graph_dict)` function from `/Utils/helper.py` file to convert your graph_dict into a networkx graph.
For more detial, please see `data_to_network_graph.py`.

### The structure of code
The `/UGFraud` repository is organized as follows:
- `Demo/` contains the implemented models and the corresponding example code;
- `Detector/` contains the basic models;
- `Yelp_Data/` contains the necessary dataset files;
- `Utils/` contains the every help functions.


## Implemented Models

| Model  | Paper  | Venue  | Reference  |
|-------|--------|--------|--------|
| **SpEagle** | [Collective Opinion Spam Detection: Bridging Review Networks and Metadata](https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf)  | KDD 2015  | [BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/speagle.txt) |
| **GANG** | [GANG: Detecting Fraudulent Users in Online Social Networks via Guilt-by-Association on Directed Graph](https://ieeexplore.ieee.org/document/8215519)  | ICDM 2017  | [BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/gang.txt)|
| **fBox** | [Spotting Suspicious Link Behavior with fBox: An Adversarial Perspective](https://arxiv.org/pdf/1410.3915.pdf)  | ICDM 2014 | [BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/fbox.txt) |
| **Fraudar** | [FRAUDAR: Bounding Graph Fraud in the Face of Camouflage](https://bhooi.github.io/papers/fraudar_kdd16.pdf)  | KDD 2016 | [BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/fraudar.txt) |
| **ZooBP** | [ZooBP: Belief Propagation for Heterogeneous Networks](http://www.vldb.org/pvldb/vol10/p625-eswaran.pdf)  | VLDB 2017 | [BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/zoobp.txt)  |
| **SVD** | [Singular value decomposition and least squares solutions](https://link.springer.com/content/pdf/10.1007/978-3-662-39778-7_10.pdf)  | - |[BibTex](https://github.com/safe-graph/UGFraud/blob/master/reference/svd.txt) |
| **Prior** | Evaluating suspicioueness based on prior information  | - |  - |


## Model Comparison
| Model  | Application  | Graph Type  | Model Type  |
|-------|--------|--------|-------|
| **SpEagle** | Review Spam | Tripartite  | MRF  |
| **GANG** | Social Sybil  | Bipartite |  MRF    |
| **fBox** | Social Fraudster  | Bipartite |  SVD |
| **Fraudar** |  Social Fraudster | Bipartite | Dense-block  |
| **ZooBP** | E-commerce Fraud | Tripartite | MRF   |
| **SVD** | Dimension Reduction  | Bipartite |  SVD  |


## TODO List
- Homogeneous graph implementation


## How to Contribute
You are welcomed to contribute to this open-source toolbox. Currently, you can create issues or send email to [bdscsafegraph@gmail.com](mailto:bdscsafegraph@gmail.com) for inquiry.
