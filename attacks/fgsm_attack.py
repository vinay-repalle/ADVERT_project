# fgsm_attack.py
# This file contains the implementation of the Fast Gradient Sign Method (FGSM) adversarial attack.

import torch
import torch.nn.functional as F


def fgsm_attack(model, image, label, epsilon):
    """
    Generate an adversarial example using the Fast Gradient Sign Method (FGSM).

    Parameters:
    - model: pretrained neural network
    - image: input image tensor
    - label: correct class label tensor
    - epsilon: attack strength

    Returns:
    - perturbed_image: the adversarial image
    """
    # Enable gradient calculation for the input image
    image.requires_grad = True

    # Perform a forward pass through the model
    output = model(image)

    # Compute the cross entropy loss between the model output and the true label
    loss = F.cross_entropy(output, label)

    # Clear existing gradients in the model
    model.zero_grad()

    # Perform backpropagation to compute gradients
    loss.backward()

    # Extract the gradient of the image
    gradient = image.grad.data

    # Create the perturbation using the sign of the gradient
    perturbation = epsilon * torch.sign(gradient)

    # Generate the adversarial image
    perturbed_image = image + perturbation

    # Clamp the pixel values to remain between 0 and 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image
