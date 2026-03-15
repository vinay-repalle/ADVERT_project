# main.py
# Entry point of the ADVRET onsole adversarial testing system

import os
import torch
import torchvision.utils as tv_utils
from models.model_loader import load_model
from dataset.image_loader import load_image
from utils.predict import predict
from attacks.fgsm_attack import fgsm_attack
from attacks.pgd_attack import pgd_attack
from utils.imagenet_labels import IMAGENET_CLASSES


def main():
    # Load model and device
    model, device = load_model()

    # Ask user for image folder
    image_folder = input("Enter the path to a folder of images: ")

    # Collect image files
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("No image files found in the provided folder.")
        return

    # Select attack type
    print("\nSelect Attack:")
    print("1 - FGSM")
    print("2 - PGD")
    attack_choice = input("Enter attack number: ")

    if attack_choice == "1":
        attack_name = "FGSM"
        attack_params = "Epsilon: 0.01"
    elif attack_choice == "2":
        attack_name = "PGD"
        attack_params = "Epsilon: 0.03, Alpha: 0.005, Iterations: 10"
    else:
        print("Invalid choice. Defaulting to FGSM.")
        attack_name = "FGSM"
        attack_params = "Epsilon: 0.01"

    os.makedirs("outputs", exist_ok=True)

    print("\n---")
    print("## ADVRET Adversarial Test")

    # Counters for attack success rates
    total_images = 0
    successful_attacks = 0

    for image_file in image_files:
        total_images += 1
        image_path = os.path.join(image_folder, image_file)

        # Load and preprocess image
        image = load_image(image_path, device)

        # Predict original image
        predicted_class, confidence_score = predict(model, image)

        print("\n----------------------------------")
        print(f"Image: {image_file}\n")
        print("Original Prediction:")
        print(f"{IMAGENET_CLASSES[predicted_class]} ({confidence_score:.2f})\n")

        # Create label tensor
        label = torch.tensor([predicted_class]).to(device)

        # Run selected attack
        if attack_name == "FGSM":
            adversarial_image = fgsm_attack(model, image, label, epsilon=0.01)
        else:
            adversarial_image = pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, iterations=10)

        # Compute the perturbation between the adversarial and original image
        perturbation = adversarial_image - image

        # Normalize perturbation for visualization
        perturbation_vis = perturbation.clone().detach()
        perturbation_vis = perturbation_vis - perturbation_vis.min()
        max_val = perturbation_vis.max()
        if max_val > 0:
            perturbation_vis = perturbation_vis / max_val

        # Save images with unique names per input file
        base_name = os.path.splitext(image_file)[0]
        orig_path = f"outputs/original_{base_name}.png"
        adv_path = f"outputs/adversarial_{base_name}.png"
        pert_path = f"outputs/perturbation_{base_name}.png"

        tv_utils.save_image(image, orig_path)
        tv_utils.save_image(adversarial_image, adv_path)
        tv_utils.save_image(perturbation_vis, pert_path)

        print("Saved images:")
        print(orig_path)
        print(adv_path)
        print(pert_path)

        print(f"\nAttack: {attack_name}")
        print(attack_params)

        # Predict adversarial image
        adv_predicted_class, adv_confidence_score = predict(model, adversarial_image)

        if adv_predicted_class != predicted_class:
            successful_attacks += 1

        print("\nAdversarial Prediction:")
        print(f"{IMAGENET_CLASSES[adv_predicted_class]} ({adv_confidence_score:.2f})")

    print("\n----------------------------------")

    # Attack success rate summary
    if total_images > 0:
        success_rate = (successful_attacks / total_images) * 100
    else:
        success_rate = 0.0

    print("\nAttack Summary")
    print(f"\nTotal Images Tested: {total_images}")
    print(f"Successful Attacks: {successful_attacks}")
    print(f"Attack Success Rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()