# pgd_attack.py
# This file contains the implementation of the Projected Gradient Descent (PGD) adversarial attack.

import torch
import torch.nn.functional as F


def pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, iterations=10):
    """Generate an adversarial example using Projected Gradient Descent (PGD).

    Parameters:
    - model: pretrained neural network
    - image: input image tensor
    - label: correct class label tensor
    - epsilon: maximum perturbation magnitude
    - alpha: step size for each iteration
    - iterations: number of gradient update steps

    Returns:
    - perturbed_image: the adversarial image
    """
    # Clone the original image and add small random noise within the epsilon-ball.
    perturbed_image = image.clone().detach()
    perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    for _ in range(iterations):
        # Enable gradient tracking for the perturbed image
        perturbed_image.requires_grad = True

        # Forward pass
        output = model(perturbed_image)

        # Compute loss
        loss = F.cross_entropy(output, label)

        # Zero model gradients
        model.zero_grad()

        # Backpropagate to get gradients w.r.t. the input
        loss.backward()

        # Update the image using the sign of the gradient
        gradient = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * torch.sign(gradient)

        # Project the perturbation back into the epsilon-ball around the original image
        perturbation = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = image + perturbation

        # Clamp to valid pixel range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Detach to avoid accumulating gradients
        perturbed_image = perturbed_image.detach()

    return perturbed_image
