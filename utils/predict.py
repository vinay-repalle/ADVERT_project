# utils/predict.py
# This module handles model inference and returns the predicted class and confidence score for an input image.

import torch
import torch.nn.functional as F


def predict(model, image):
    """
    Perform inference on the input image using the given model.

    Parameters:
    - model: pretrained neural network
    - image: preprocessed image tensor (with batch dimension, shape [1, C, H, W])

    Returns:
    - predicted_class: integer representing the predicted class
    - confidence_score: float representing the confidence score for the predicted class
    """
    # Disable gradient computation for inference
    with torch.no_grad():
        # Pass the image through the model to get the output logits
        output = model(image)

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(output, dim=1)

        # Find the predicted class using argmax (for the first item in the batch)
        predicted_class = torch.argmax(probabilities, dim=1)[0].item()

        # Extract the confidence score for the predicted class
        confidence_score = probabilities[0, predicted_class].item()

    # Return the predicted class and confidence score
    return predicted_class, confidence_score