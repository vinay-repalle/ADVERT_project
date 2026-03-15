# ADVRET

Adversarial Robustness Evaluation and Testing Platform

## Overview

ADVRET is a PyTorch-based system designed to evaluate how vulnerable deep learning image classification models are to adversarial attacks. It supports FGSM and PGD attacks and visualizes the generated perturbations.

## Features

* Pretrained ResNet18 model loading
* FGSM adversarial attack
* PGD adversarial attack
* Batch processing of image folders
* Perturbation visualization
* Attack success rate reporting
* Saving original, adversarial, and perturbation images

## Project Structure

attacks/
dataset/
models/
utils/
evaluation/
outputs/
main.py
requirements.txt

## Installation

pip install -r requirements.txt

## Usage

python main.py