# cw_attack.py
# This file contains the implementation of the Carlini & Wagner (C&W) adversarial attack.

import torch
import torch.nn as nn
import torch.optim as optim


def cw_attack(model, image, label, c=1, kappa=0, iterations=100, learning_rate=0.01):
    """
    Generate an adversarial example using the Carlini & Wagner (C&W) attack.

    Parameters:
    - model: pretrained neural network
    - image: input image tensor (batch size 1)
    - label: correct class label tensor
    - c: regularization parameter
    - kappa: confidence parameter
    - iterations: number of optimization iterations
    - learning_rate: learning rate for Adam optimizer

    Returns:
    - adversarial_image: the adversarial image
    """
    # Clone the original image
    original_image = image.clone().detach()

    # Create perturbation variable
    perturbation = torch.zeros_like(image, requires_grad=True)

    # Adam optimizer
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    # True label
    true_label = label.item()

    for _ in range(iterations):
        # Generate adversarial image
        adversarial_image = original_image + perturbation
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # Forward pass
        logits = model(adversarial_image)

        # Compute L2 distance
        l2_distance = torch.sum((adversarial_image - original_image) ** 2)

        # Compute classification loss
        # f(x') = max(max_{j≠t} Z_j - Z_t, -kappa)
        z_true = logits[0, true_label]
        z_other = torch.cat([logits[0, :true_label], logits[0, true_label+1:]])
        max_other = torch.max(z_other)
        classification_loss = torch.max(max_other - z_true, torch.tensor(-kappa, device=logits.device))

        # Total loss
        loss = l2_distance + c * classification_loss

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final adversarial image
    adversarial_image = original_image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image