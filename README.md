# ADVRET

Adversarial Robustness Evaluation and Testing Platform

## Overview

ADVRET is a PyTorch-based system designed to evaluate how vulnerable deep learning image classification models are to adversarial attacks. It supports multiple state-of-the-art attacks including FGSM, PGD, DeepFool, and Carlini-Wagner, with both single-attack testing and comprehensive benchmarking capabilities.

## Features

* Pretrained model loading (ResNet18 / MobileNetV2 / VGG16)
* Four adversarial attacks: FGSM, PGD, DeepFool, Carlini-Wagner
* Dual operating modes: Single attack testing and multi-attack benchmarking
* Batch processing of image folders
* Attack success rate and confidence drop metrics
* Automated benchmark table generation
* Static benchmark graphs (success rate, confidence drop, comparison, robustness curve)
* Web UI (Flask) + CLI mode

## Project Structure

```
attacks/          # Attack implementations
dataset/          # Image loading utilities
models/           # Model loading
utils/            # Helper functions and labels
evaluation/       # Metrics and benchmarking
templates/        # Web UI templates (Jinja)
static/           # Web UI static assets (JS)
uploads/          # Temporary user uploads (ignored by git)
outputs/          # Generated results
app.py            # Flask web app
main.py           # Main application
requirements.txt  # Dependencies
PROJECT_DOCUMENTATION.md  # Detailed documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web UI (recommended)

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` (local) or the Render service URL (deployment).

In the UI you can:
- Select a pretrained model (ResNet18 / MobileNetV2 / VGG16) or upload a custom `.pt` / `.pth`
- Upload a **folder** of images (`.jpg` / `.jpeg` / `.png`)
- Run single-attack mode or benchmark mode
- View results on a dedicated results page (static graphs + summary + defense suggestions)
- Download a zipped report

**Progress indicator:** the UI shows a simple centered progress modal with an approximate percentage (no polling / no extra backend routes).

**Temporary uploads:** uploaded images/models are treated as temporary (see “Uploads & Git safety” below).

### CLI mode

```bash
python main.py
```

Choose between single attack mode or benchmark mode to evaluate model robustness.

## Uploads & Git safety

- User uploads are stored under `uploads/` and treated as **temporary**.
- Old uploads are automatically removed when:
  - the landing page is refreshed, or
  - the user uploads new files
- `uploads/` is ignored by git, and model file extensions (`*.pt`, `*.pth`) are also ignored.
- `outputs/` is ignored by git (generated graphs/report files).

## Deployment (Render)

### Run locally

```bash
python app.py
```

### Push to GitHub

```bash
git init
git add .
git commit -m "deployment ready"
git branch -M main
git remote add origin <repo>
git push -u origin main
```

### Deploy on Render

1. Go to Render
2. Create new **Web Service**
3. Connect the GitHub repo
4. Build command:

```bash
pip install -r requirements.txt
```

5. Start command:

```bash
python app.py
```

## Documentation

For detailed information about the project, implementation details, and future plans, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
