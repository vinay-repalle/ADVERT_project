# ADVRET: Adversarial Robustness Evaluation and Testing Platform

## Abstract

ADVRET is a comprehensive PyTorch-based framework designed for evaluating the adversarial robustness of deep learning image classification models. The platform implements multiple state-of-the-art adversarial attacks and provides benchmarking capabilities to compare their effectiveness. By supporting both single-attack testing and multi-attack benchmarking modes, ADVRET serves as a valuable tool for researchers and practitioners in adversarial machine learning.

## Project Overview

### Core Components

- **Model Loading**: Supports pretrained torchvision models (ResNet18 / MobileNetV2 / VGG16) and optional custom PyTorch model upload (.pt/.pth) via the web UI
- **Image Processing**: Handles batch processing of image folders with automatic preprocessing
- **Attack Implementations**: Four different adversarial attack algorithms
- **Evaluation Metrics**: Comprehensive metrics including success rates and confidence drops
- **Visualization**: Generates static benchmark graphs (saved to `outputs/`)
- **Benchmarking**: Automated comparison across multiple attacks (single attack mode or all-attacks mode)
- **Web Interface**: Flask + Jinja templates for running ADVRET from the browser

### Implemented Attacks

1. **FGSM (Fast Gradient Sign Method)**: Single-step gradient-based attack
   - Uses epsilon parameter for perturbation magnitude
   - Fast and computationally efficient

2. **PGD (Projected Gradient Descent)**: Iterative gradient-based attack
   - Multiple iterations with step size control
   - More powerful than FGSM with higher success rates

3. **DeepFool**: Minimal perturbation attack
   - Computes minimal L2 perturbation to cross decision boundary
   - Iterative algorithm targeting top predicted classes

4. **Carlini-Wagner (C&W)**: Optimization-based attack
   - Uses Adam optimizer to minimize L2 distance + classification loss
   - Considered one of the strongest adversarial attacks

## Technical Implementation

### Architecture

```
ADVRET/
├── main.py                 # Main entry point with mode selection
├── app.py                  # Flask web app entry point
├── attacks/                # Attack implementations
│   ├── fgsm_attack.py
│   ├── pgd_attack.py
│   ├── deepfool_attack.py
│   └── cw_attack.py
├── dataset/
│   └── image_loader.py     # Image preprocessing pipeline
├── models/
│   └── model_loader.py     # Model loading utilities
├── utils/
│   ├── helpers.py
│   ├── imagenet_labels.py  # ImageNet class labels
│   └── predict.py          # Prediction utilities
├── evaluation/
│   └── metrics.py          # Benchmarking and metrics
├── templates/              # Web UI templates (Jinja)
│   ├── base.html
│   ├── index.html          # Landing page + form
│   ├── results.html        # Results page
│   └── partials/
├── static/
│   └── js/
│       └── progress.js     # Simple progress modal logic (no polling)
├── uploads/                # Temporary uploads (ignored by git)
├── outputs/                # Generated images and results
└── requirements.txt        # Python dependencies
```

### Key Features

- **Dual Operating Modes**:
  - Single Attack Mode: Test individual attacks with detailed output
  - Benchmark Mode: Compare all attacks across multiple images

- **Comprehensive Metrics**:
  - Attack Success Rate: Percentage of successful misclassifications
  - Average Confidence Drop: Mean reduction in model confidence
  - Per-attack statistics tracking

- **Image Processing Pipeline**:
  - Automatic image loading and preprocessing
  - Support for common image formats (JPG, PNG)
  - Batch processing with GPU acceleration

- **Visualization Capabilities**:
  - Graphical attack benchmarking:
    - Success rate bar chart
    - Confidence drop bar chart
    - Combined success vs confidence comparison chart
    - Robustness curve (accuracy vs epsilon)
  - Generated outputs saved into the `outputs/` directory (typically ignored by git)

### Web UI Details (Flask)

The web interface provides a minimal, synchronous workflow:

- **Landing page (`GET /`)**:
  - Shows project info and instructions
  - Form inputs:
    - pretrained model selection (ResNet18 / MobileNetV2 / VGG16)
    - optional custom model upload (`.pt` / `.pth`)
    - **folder upload** for images (`webkitdirectory`), accepting `.jpg` / `.jpeg` / `.png`
    - attack mode (single or benchmark)
  - Shows a centered progress modal when “Run” is clicked (approximate % only; no polling)

- **Run endpoint (`POST /run`)**:
  - Processes images synchronously
  - Computes metrics and generates graphs
  - Renders `results.html` with:
    - results table
    - static graphs served from `/outputs/...`
    - summary + defense suggestions
    - report download button

### Temporary uploads & cleanup policy

To keep the project lightweight and safe:

- User uploads are stored under `uploads/` and treated as **temporary**.
- Old uploads are deleted/overwritten when:
  - the landing page is refreshed, or
  - the user uploads new files
- Uploaded artifacts are never intended to be committed to git.

## Usage Examples

### Single Attack Mode
```
Select Mode:
1 - Run Single Attack
2 - Run All Attacks (Benchmark Mode)
> 1

Select Attack:
1 - FGSM
2 - PGD
3 - DeepFool
4 - Carlini-Wagner
> 2
```

### Benchmark Mode
```
Select Mode:
1 - Run Single Attack
2 - Run All Attacks (Benchmark Mode)
> 2

[Processing multiple attacks on all images...]

==============================
ADVRET Attack Benchmark
==============================

## Attack            Success Rate    Avg Confidence Drop
FGSM               33.33%          0.45
PGD                66.67%          0.61
DeepFool           50.00%          0.52
Carlini-Wagner     83.33%          0.74

Images Tested: 12
```

## Implementation Details

### Attack Algorithms

Each attack implementation follows a consistent interface:
- Takes model, image, and label as inputs
- Returns adversarial image tensor
- Handles gradient computations appropriately
- Ensures output images remain in valid [0,1] range

### Metrics Calculation

- **Success Detection**: Compares original vs adversarial predictions
- **Confidence Tracking**: Measures softmax confidence scores
- **Aggregate Statistics**: Maintains running averages across all test images

### Performance Considerations

- GPU acceleration support via PyTorch
- Memory-efficient batch processing
- Optimized gradient computations
- Modular design for easy extension

## Dependencies

- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- NumPy >= 1.24.0
- Pillow >= 9.0.0

## Results and Analysis

The benchmarking capabilities reveal important insights:

- **Attack Strength Hierarchy**: C&W typically achieves highest success rates
- **Computational Trade-offs**: FGSM fastest, C&W most computationally intensive
- **Confidence Impact**: Stronger attacks cause larger confidence drops
- **Model Vulnerability**: Comprehensive assessment across attack types

## Conclusion

ADVRET provides a robust, extensible framework for adversarial robustness evaluation. Its modular architecture and comprehensive feature set make it suitable for both research and practical applications in adversarial machine learning.

## Future Work

### Planned Enhancements

1. **Additional Attacks**:
   - Boundary Attack
   - AutoAttack ensemble
   - Universal perturbations

2. **Extended Metrics**:
   - Perturbation distances (L2, L∞)
   - Attack transferability analysis
   - Time complexity measurements

3. **Model Support**:
   - Custom model architectures
   - Multiple pretrained models
   - Fine-tuned model evaluation

4. **Advanced Features**:
   - Targeted attacks
   - Defense mechanism evaluation
   - Dataset-level robustness assessment

5. **User Interface**:
   - Web-based interface
   - Configuration file support
   - Automated reporting

### Research Directions

- Correlation analysis between attack success and image characteristics
- Robustness improvement strategies
- Real-world adversarial example detection
- Cross-domain transferability studies

---

*ADVRET is developed as an educational and research tool for understanding adversarial vulnerabilities in deep learning systems.*