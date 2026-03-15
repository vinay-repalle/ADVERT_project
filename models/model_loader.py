# model_loader.py
# This file will handle loading pretrained deep learning models (e.g., ResNet18).
import torch
import torchvision.models as models

def load_model():
    """
    Loads a pretrained ResNet18 model for image classification.
    
    Returns:
        model: The pretrained PyTorch model set to evaluation mode.
        device: The device (GPU or CPU) the model is loaded onto.
    """
    # 3. Automatically detect GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Use torchvision to load a pretrained ResNet18 model
    # We use the modern 'weights' parameter to load default pretrained weights
    print("Loading pretrained ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Move the model to the detected device
    model = model.to(device)

    # 2. Set the model to evaluation mode
    model.eval()

    # 4. Return the model and device
    return model, device
