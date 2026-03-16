# model_loader.py
# This module handles loading pretrained deep learning models (ResNet18, MobileNetV2, VGG16).

import torch
import torchvision.models as models

# Supported pretrained model configurations
_MODEL_CONFIGS = {
    "1": {
        "name": "ResNet18",
        "constructor": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
    },
    "2": {
        "name": "MobileNetV2",
        "constructor": models.mobilenet_v2,
        "weights": models.MobileNet_V2_Weights.DEFAULT,
    },
    "3": {
        "name": "VGG16",
        "constructor": models.vgg16,
        "weights": models.VGG16_Weights.DEFAULT,
    },
}


def load_model(choice: str, device: torch.device):
    """Load a pretrained model based on user selection.

    Args:
        choice: The selected model option (e.g., "1" for ResNet18).
        device: The device (CPU/GPU) to move the model onto.

    Returns:
        Tuple[torch.nn.Module, torch.device]: The loaded model (in eval mode) and the device.
    """

    # Normalize choice to string
    choice_str = str(choice).strip()

    # Default to ResNet18 if invalid
    if choice_str not in _MODEL_CONFIGS:
        print(f"Invalid model choice '{choice}'. Defaulting to ResNet18.")
        choice_str = "1"

    cfg = _MODEL_CONFIGS[choice_str]
    model_name = cfg["name"]

    print(f"Loading pretrained {model_name} model...")
    model = cfg["constructor"](weights=cfg["weights"])

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    return model, device
