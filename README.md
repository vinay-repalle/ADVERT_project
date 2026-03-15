# ADVRET

Adversarial Robustness Evaluation and Testing Platform

## Overview

ADVRET is a PyTorch-based system designed to evaluate how vulnerable deep learning image classification models are to adversarial attacks. It supports multiple state-of-the-art attacks including FGSM, PGD, DeepFool, and Carlini-Wagner, with both single-attack testing and comprehensive benchmarking capabilities.

## Features

* Pretrained ResNet18 model loading
* Four adversarial attacks: FGSM, PGD, DeepFool, Carlini-Wagner
* Dual operating modes: Single attack testing and multi-attack benchmarking
* Batch processing of image folders
* Perturbation visualization
* Attack success rate and confidence drop metrics
* Automated benchmark table generation
* Saving original, adversarial, and perturbation images

## Project Structure

```
attacks/          # Attack implementations
dataset/          # Image loading utilities
models/           # Model loading
utils/            # Helper functions and labels
evaluation/       # Metrics and benchmarking
outputs/          # Generated results
main.py           # Main application
requirements.txt  # Dependencies
PROJECT_DOCUMENTATION.md  # Detailed documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Choose between single attack mode or benchmark mode to evaluate model robustness.

## Documentation

For detailed information about the project, implementation details, and future plans, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).