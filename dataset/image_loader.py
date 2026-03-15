import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path, device):
    """
    Loads and preprocesses an image for a pretrained PyTorch model.
    """
    # 1. Load the image using the Pillow (PIL) library and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        # 2. Resize the image to 224x224 pixels
        transforms.Resize((224, 224)),
        # 3. Convert the image into a PyTorch tensor
        transforms.ToTensor(),
        # 4. Normalize the tensor using ImageNet normalization values
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply the preprocessing transforms to the image
    tensor = preprocess(image)
    
    # Add a batch dimension using unsqueeze(0)
    # This changes shape from [C, H, W] to [1, C, H, W]
    tensor = tensor.unsqueeze(0)
    
    # Move the tensor to the provided device (CPU or GPU)
    tensor = tensor.to(device)
    
    return tensor
