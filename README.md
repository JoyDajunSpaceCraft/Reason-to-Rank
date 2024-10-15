# Reason-to-Rank: Learning to Rank through Reasoning-Based Knowledge Distillation

This repository contains the code and datasets for our paper **"Reason-to-Rank: Learning to Rank through Reasoning-Based Knowledge Distillation"**. The goal of this project is to enhance transparency and performance in document reranking by incorporating both pointwise and comparative reasoning into the ranking process. We utilize large language models (LLMs) as teacher models to generate high-quality explanations for ranking decisions, which are then distilled into smaller, more efficient student models.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Ranking documents based on their relevance to a given query is a critical task in information retrieval. Traditional reranking models often improve ranking accuracy but lack transparency in explaining why certain documents are ranked higher than others. **Reason-to-Rank (R2R)** addresses this limitation by generating two types of reasoning:
1. **Pointwise reasoning**: Explains how each document addresses the query.
2. **Comparative reasoning**: Justifies the relevance of one document over another.

Our framework leverages large language models (LLMs) as teacher models to generate reasoning, and distills this knowledge into smaller, efficient student models that can perform reranking and reasoning generation tasks. Experiments across multiple datasets show that R2R improves ranking accuracy while offering clear explanations for ranking decisions.

## Features

- **Dual Reasoning**: Incorporates both pointwise and comparative reasoning for more transparent document reranking.
- **Knowledge Distillation**: Uses LLMs as teacher models and distills their knowledge into smaller student models.
- **Open Source**: The code and datasets are available for researchers to further explore and improve reranking tasks.

## Installation

To install and set up the project locally, follow these steps:

1. **Install the required dependencies**:
Create a virtual environment and install the dependencies:
```
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your datasets in the `data/` directory. The datasets should follow the format described in the [Datasets](#datasets) section.

### 2. Training

To train a student model with Reason-to-Rank, run the following command:

```bash
python train.py --config config.yaml
```

## Datasets

### Prepare dataset

The option for the datatype is three `combine`, `pointwise` and `compare` reflect to the different reasoning types in the paper.

```
python prepare_reason_rank/multi_rank.py --datatype combine
```

### BM25 dataset prepare 

We also add the BM25 generated result in the place of /data

