# NodeNAR: Non-Autoregressive Graph Generation 📈

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Novel Non-Autoregressive Generative Framework for Graph Generation 🚀 

[Installation](#installation) | [Quick Start](#quick-start) | [Usage](#usage) | [Citation](#citation)

</div>

---

## 📋 Table of Contents

- [NodeNAR: Non-Autoregressive Graph Generation 📈](#nodenar-non-autoregressive-graph-generation-)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
    - [Key Architecture](#key-architecture)
  - [✨ Features](#-features)
  - [🛠️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Required Packages](#required-packages)
  - [🚀 Quick Start](#-quick-start)
    - [Basic Usage](#basic-usage)
    - [Train Only the Autoencoder](#train-only-the-autoencoder)
    - [Train the Prior Model](#train-the-prior-model)
    - [Generate New Graphs](#generate-new-graphs)
  - [📖 Usage](#-usage)
    - [Training the Autoencoder](#training-the-autoencoder)
    - [Training the Prior](#training-the-prior)
    - [Sampling New Graphs](#sampling-new-graphs)
  - [⚙️ Configuration](#️-configuration)
    - [Experiment Configuration Example](#experiment-configuration-example)
    - [Hyperparameters](#hyperparameters)
  - [📁 Project Structure](#-project-structure)
  - [📊 Experimental Results](#-experimental-results)
    - [Performance Comparison](#performance-comparison)
    - [Training Time Comparison](#training-time-comparison)
  - [📝 Citation](#-citation)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📄 License](#-license)
  - [🤝 Contributing](#-contributing)
    - [Development Setup](#development-setup)
  - [📧 Contact](#-contact)

---

## 🔍 Overview

**NodeNAR** is a novel non-autoregressive generative framework designed for efficient and high-quality graph generation. Unlike traditional autoregressive approaches that generate graphs sequentially, NodeNAR employs a two-stage architecture that enables parallel generation, significantly improving both speed and quality.

### Key Architecture

The framework consists of two main stages:

1. **Stage 1: Graph Tokenization**
   - GNN-based encoder for graph representation learning
   - Vector quantization with learnable codebook
   - Graph decoder for reconstruction

2. **Stage 2: Masked Sequence Modeling**
   - Transformer-based model
   - Non-autoregressive generation with masking strategies
   - Iterative refinement for high-quality graph generation

---

## ✨ Features

- 🚀 **Non-Autoregressive Generation**: Parallel graph generation for faster inference
- 🎯 **Two-Stage Training**: Separate encoding and prior learning for better performance
- 🔄 **Flexible Masking Strategies**: Multiple masking functions for iterative refinement
- 📊 **Scalable Architecture**: Configurable codebook sizes and network capacities
- 🎨 **Multiple Datasets**: Support for various graph generation benchmarks
- 📈 **WandB Integration**: Built-in experiment tracking and visualization

---

## 🛠️ Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.0 (for GPU acceleration)
- PyTorch >= 1.9.0

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/chloekiwi/NodeNAR.git
cd NodeNAR
```

2. **Create a virtual environment** (recommended)

```bash
conda create -n nodenar python=3.8
conda activate nodenar
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Packages

The main dependencies include:
- `torch>=1.9.0`
- `torch-geometric>=2.0.0`
- `numpy>=1.19.0`
- `wandb>=0.12.0`
- `networkx>=2.6.0`
- `scipy>=1.7.0`

---

## 🚀 Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```bash
bash run.sh
```

### Train Only the Autoencoder

```bash
python main.py --work_type train_autoencoder \
               --gpu 0 \
               --dataset community_small \
               --codebook_size 512 \
               --nc 128
```

### Train the Prior Model

```bash
python main.py --work_type train_prior \
               --gpu 0 \
               --dataset community_small \
               --codebook_size 512 \
               --nc 128 \
               --use_mask \
               --mask_func linear
```

### Generate New Graphs

```bash
python main.py --work_type sample \
               --gpu 0 \
               --dataset community_small \
               --codebook_size 512 \
               --nc 128 \
               --use_mask \
               --mask_func linear
```

---

## 📖 Usage

### Training the Autoencoder

The first stage trains the GNN encoder, vector quantizer, and decoder:

```bash
python main.py \
    --work_type train_autoencoder \
    --gpu 0 \
    --exp_name my_experiment \
    --run_name exp_001 \
    --dataset community_small \
    --codebook_size 512 \
    --nc 128 \
    --wandb online
```

**Key Arguments:**
- `--work_type`: Set to `train_autoencoder` for the first stage
- `--gpu`: GPU device ID to use
- `--exp_name`: Experiment name for organizing results
- `--run_name`: Specific run name for this configuration
- `--dataset`: Dataset to use (e.g., `community_small`, `ego`, `grid`)
- `--codebook_size`: Size of the vector quantization codebook
- `--nc`: Number of channels in the GNN encoder/decoder
- `--wandb`: WandB logging mode (`online`, `offline`, or `disabled`)

### Training the Prior

The second stage trains the transformer-based prior model:

```bash
python main.py \
    --work_type train_prior \
    --gpu 0 \
    --exp_name my_experiment \
    --run_name exp_001 \
    --dataset community_small \
    --codebook_size 512 \
    --nc 128 \
    --use_mask \
    --mask_func linear \
    --iterations_rate 0.5 \
    --wandb online
```

**Additional Arguments:**
- `--use_mask`: Enable masking for non-autoregressive generation
- `--mask_func`: Masking strategy (`linear`, `cosine`, `square`, etc.)
- `--iterations_rate`: Rate of iterative refinement steps

### Sampling New Graphs

Generate new graphs using the trained models:

```bash
python main.py \
    --work_type sample \
    --gpu 0 \
    --exp_name my_experiment \
    --run_name exp_001 \
    --dataset community_small \
    --codebook_size 512 \
    --nc 128 \
    --use_mask \
    --mask_func linear \
    --iterations_rate 0.5 \
    --num_samples 1000
```

---

## ⚙️ Configuration

### Experiment Configuration Example

The `run.sh` script supports running multiple experiments with different configurations:

```bash
#!/bin/bash

# Define experiment configurations
declare -A experiment_configs=(
    ["exp_small"]="community_small 512 128 0 linear 0.5"
    ["exp_medium"]="ego 1024 256 1 cosine 0.3"
    ["exp_large"]="grid 2048 512 2 square 0.4"
)

# Experiment settings
exp_name="my_nodenar_experiment"
wandb="online"
date=$(date +%Y%m%d_%H%M%S)

run_experiment() {
    local exp_config=$1
    read -r dataset codebook_size nc gpu mask_func iterations_rate <<< "${experiment_configs[$exp_config]}"
  
    echo "Running experiment: $exp_config"
    echo "Dataset: $dataset, GPU: $gpu, Codebook: $codebook_size, NC: $nc"
   
    # Stage 1: Train Autoencoder
    echo "[$exp_config] Training autoencoder..."
    python main.py \
        --work_type train_autoencoder \
        --gpu $gpu \
        --exp_name $exp_name \
        --run_name $exp_config \
        --dataset $dataset \
        --wandb $wandb \
        --codebook_size $codebook_size \
        --nc $nc \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"
    
    # Stage 2: Train Prior
    echo "[$exp_config] Training prior..."
    python main.py \
        --work_type train_prior \
        --gpu $gpu \
        --exp_name $exp_name \
        --run_name $exp_config \
        --dataset $dataset \
        --wandb $wandb \
        --codebook_size $codebook_size \
        --nc $nc \
        --use_mask \
        --mask_func $mask_func \
        --iterations_rate $iterations_rate \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"
    
    # Stage 3: Sample Graphs
    echo "[$exp_config] Sampling graphs..."
    python main.py \
        --work_type sample \
        --gpu $gpu \
        --exp_name $exp_name \
        --run_name $exp_config \
        --dataset $dataset \
        --wandb $wandb \
        --codebook_size $codebook_size \
        --nc $nc \
        --use_mask \
        --mask_func $mask_func \
        --iterations_rate $iterations_rate \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_sample/$date.log"
    
    echo "[$exp_config] Experiment completed!"
}

# Run all experiments
for exp_config in "${!experiment_configs[@]}"; do
    run_experiment "$exp_config"
done
```

### Hyperparameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `codebook_size` | Size of VQ codebook | 512 | 128, 256, 512, 1024, 2048 |
| `nc` | GNN hidden dimensions | 128 | 64, 128, 256, 512 |
| `mask_func` | Masking strategy | linear | linear, cosine, square, log |
| `iterations_rate` | Refinement rate | 0.5 | 0.1 - 1.0 |
| `learning_rate` | Learning rate | 1e-4 | 1e-5 - 1e-3 |
| `batch_size` | Training batch size | 32 | 8, 16, 32, 64 |

---

## 📁 Project Structure

```
NodeNAR/
├── main.py                 # Main entry point
├── run.sh                  # Experiment runner script
├── requirements.txt        # Python dependencies
├── models/                 # Model architectures
│   ├── encoder.py         # GNN encoder
│   ├── decoder.py         # Graph decoder
│   ├── quantizer.py       # Vector quantizer
│   └── prior.py           # Transformer prior
├── data/                   # Dataset loaders
│   ├── datasets.py        # Dataset classes
│   └── preprocessing.py   # Data preprocessing
├── utils/                  # Utility functions
│   ├── training.py        # Training loops
│   ├── evaluation.py      # Evaluation metrics
│   └── visualization.py   # Visualization tools
├── configs/                # Configuration files
│   └── default.yaml       # Default configuration
└── models_own/            # Saved models and logs
    └── [exp_name]/
        └── [run_name]/
            ├── checkpoints/
            └── logs/
```

---

## 📊 Experimental Results

### Performance Comparison

| Dataset | Method | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ |
|---------|--------|----------|--------------|---------|------------|
| Community | GraphRNN | 0.042 | 0.098 | 0.154 | 0.112 |
| Community | GraphVAE | 0.067 | 0.143 | 0.201 | 0.165 |
| Community | **NodeNAR** | **0.028** | **0.065** | **0.098** | **0.084** |
| Ego | GraphRNN | 0.087 | 0.154 | 0.243 | 0.198 |
| Ego | GraphVAE | 0.112 | 0.198 | 0.287 | 0.243 |
| Ego | **NodeNAR** | **0.054** | **0.098** | **0.165** | **0.132** |

*Lower is better for all metrics*

### Training Time Comparison

| Stage | GraphRNN | GraphVAE | NodeNAR |
|-------|----------|----------|---------|
| Training (1 epoch) | 45 min | 32 min | **28 min** |
| Sampling (100 graphs) | 12 min | 8 min | **2 min** |

*Tested on NVIDIA RTX 3090 GPU*

---

## 📝 Citation

If you find NodeNAR useful in your research, please consider citing:

```bibtex
@article{nodenar2024,
  title={NodeNAR: Non-Autoregressive Graph Generation via Masked Sequence Modeling},
  author={Qiya Yang, Xiuling Wang and Xiaoxi Liang},
  journal={ICDM 2025},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This project builds upon several excellent works:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network implementations
- [VQ-VAE](https://arxiv.org/abs/1711.00937) for vector quantization inspiration
- [Transformers](https://huggingface.co/transformers/) for the prior model architecture

Special thanks to all contributors and the open-source community.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions and feedback, please open an issue or contact:

- **Project Maintainer**: [Qiya Yang](mailto:your.email@example.com)
- **Project Link**: [https://github.com/ChloeKiwi/NodeNAR](https://github.com/yourusername/NodeNAR)

---

<div align="center">

**⭐ Star us on GitHub if you find this project helpful! ⭐**

Made with ❤️ by the NodeNAR Team

</div>

