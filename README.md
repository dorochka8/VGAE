# Variational GRAPH AutoEncoder

This repository contains an implementation of the Variational Graph Auto-Encoder (VGAE) as described in the paper by T. Kipf and M. Welling, ["Variational Graph Auto-Encoders"](https://arxiv.org/pdf/1611.07308.pdf), published on arXiv in 2016. The VGAE is an approach for unsupervised learning on graph-structured data, demonstrating superior performance on the link prediction task.

## Overview
The implementation focuses on the Cora dataset, adhering to the default settings specified in the original paper. This includes:

- Training for 200 epochs.
- Utilizing the `torch.optim.Adam` optimizer with a learning rate of 0.02.
- A two-layer Graph Convolutional Network (GCN) for the encoder, with additional layers for computing `mu` (the mean) and `log_std` (the log standard deviation).
- Hidden layer size of 32 and a latent space size of 16.
- Employing the reparametrization trick for latent space embedding generation.
- A simple inner product decoder for reconstructing the graph.

## Installation
To set up the project for local development and testing, follow this:
```
git clone https://github.com/dorochka8/VGAE.git
```

## Usage 
To train the model with the default settings on the Cora dataset, run: 
```
python3 main.py
```

You can modify the training parameters within the `config.py` script to experiment with different configurations.

## Results
The results are provided for 10 independent training loops, each running for 200 epochs. Below is a table showcasing the loss, ROC AUC (Receiver Operating Characteristic Area Under the Curve), and AP (Average Precision) for 10 independent training loops:

Loss   | ROC AUC  | AP   | 
:---:  |   :---:  | :---:|
0.256  |  0.906   | 0.900|

The mean ROC AUC across the training loops is 0.906, indicating a high true positive rate relative to the false positive rate. The mean Average Precision (AP), which measures the precision-recall tradeoff, is 0.9, demonstrating the model's effectiveness in link prediction within the graph.

<p float="left">
  <img 
    src="https://github.com/dorochka8/VGAE/assets/97133490/bc9a6ccc-7250-4146-95cd-1fd7586f2eee" 
    width=30% 
    height=45%
    />
  <img 
    src="https://github.com/dorochka8/VGAE/assets/97133490/e19e5204-a435-41e0-bb65-37a812f46fc9" 
    width=30% 
    height=45%
    /> 
  <img 
    src="https://github.com/dorochka8/VGAE/assets/97133490/0b0eae03-7de8-415a-958a-21c1ad1ccfac"
    width=30% 
    height=45%
    />
</p>

These results underscore the VGAE model's robustness and efficiency in handling graph-structured data, closely aligning with the benchmarks set by the [original paper](https://arxiv.org/pdf/1611.07308.pdf). The consistent performance across multiple runs highlights the model's stability and reliability for unsupervised learning tasks on graph data.
