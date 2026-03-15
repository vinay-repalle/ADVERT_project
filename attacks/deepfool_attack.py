# deepfool_attack.py
# This file contains the implementation of the DeepFool adversarial attack.

import torch


def deepfool_attack(model, image, max_iter=50, overshoot=0.02):
    """
    Generate an adversarial example using the DeepFool attack.

    Parameters:
    - model: pretrained neural network
    - image: input image tensor (batch size 1)
    - max_iter: maximum number of iterations
    - overshoot: overshoot parameter

    Returns:
    - perturbed_image: the adversarial image
    """
    # Clone the input image
    perturbed_image = image.clone().detach()

    # Get original prediction
    with torch.no_grad():
        orig_logits = model(image)
        orig_pred = orig_logits.argmax(dim=1).item()

    for iteration in range(max_iter):
        # Enable gradients
        perturbed_image.requires_grad = True

        # Forward pass
        logits = model(perturbed_image)

        # Check if prediction changed
        pred = logits.argmax(dim=1).item()
        if pred != orig_pred:
            break

        # Get the logits for the top two classes
        sorted_logits, indices = logits.sort(dim=1, descending=True)
        k1 = indices[0, 0].item()
        k2 = indices[0, 1].item()

        # Compute the difference in logits
        f_k1 = logits[0, k1]
        f_k2 = logits[0, k2]

        # Compute gradient of (f_k1 - f_k2)
        loss = f_k1 - f_k2
        model.zero_grad()
        loss.backward()

        # Get the gradient
        grad = perturbed_image.grad.data[0]  # Remove batch dim

        # Compute the perturbation
        w_norm_sq = torch.sum(grad ** 2)
        if w_norm_sq == 0:
            break

        r = ((f_k1 - f_k2).item() / w_norm_sq.item()) * grad

        # Apply perturbation
        perturbed_image = perturbed_image.detach() + r.unsqueeze(0)

        # Clamp to valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Apply overshoot
    perturbation = perturbed_image - image
    perturbed_image = image + (1 + overshoot) * perturbation

    # Final clamp
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image