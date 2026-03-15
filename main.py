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
from attacks.deepfool_attack import deepfool_attack
from attacks.cw_attack import cw_attack
from utils.imagenet_labels import IMAGENET_CLASSES
from evaluation.metrics import update_attack_metrics, print_attack_benchmark


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

    # Select mode
    print("\nSelect Mode:")
    print("1 - Run Single Attack")
    print("2 - Run All Attacks (Benchmark Mode)")
    mode_choice = input("Enter mode number: ")

    if mode_choice not in ["1", "2"]:
        print("Invalid choice. Defaulting to Single Attack.")
        mode_choice = "1"

    # Select attack type if single mode
    if mode_choice == "1":
        print("\nSelect Attack:")
        print("1 - FGSM")
        print("2 - PGD")
        print("3 - DeepFool")
        print("4 - Carlini-Wagner")
        attack_choice = input("Enter attack number: ")

        if attack_choice == "1":
            attack_name = "FGSM"
            attack_params = "Epsilon: 0.01"
        elif attack_choice == "2":
            attack_name = "PGD"
            attack_params = "Epsilon: 0.03, Alpha: 0.005, Iterations: 10"
        elif attack_choice == "3":
            attack_name = "DeepFool"
            attack_params = "Max Iterations: 50\nOvershoot: 0.02"
        elif attack_choice == "4":
            attack_name = "Carlini-Wagner"
            attack_params = "Iterations: 100\nLearning Rate: 0.01"
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

    # Metrics for benchmark
    attack_metrics = {}

    # Define attacks
    attacks = {
        "FGSM": lambda model, image, label: fgsm_attack(model, image, label, epsilon=0.01),
        "PGD": lambda model, image, label: pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, iterations=10),
        "DeepFool": lambda model, image, label: deepfool_attack(model, image, max_iter=50, overshoot=0.02),
        "Carlini-Wagner": lambda model, image, label: cw_attack(model, image, label, c=1, kappa=0, iterations=100, learning_rate=0.01)
    }

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

        if mode_choice == "1":
            # Run selected attack
            adversarial_image = attacks[attack_name](model, image, label)

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

        else:  # Benchmark mode
            for attack_name in attacks:
                adversarial_image = attacks[attack_name](model, image, label)

                # Predict adversarial image
                adv_predicted_class, adv_confidence_score = predict(model, adversarial_image)

                is_success = adv_predicted_class != predicted_class
                conf_drop = confidence_score - adv_confidence_score

                update_attack_metrics(attack_metrics, attack_name, is_success, conf_drop)

                print(f"Attack: {attack_name} - Success: {is_success} - Conf Drop: {conf_drop:.2f}")

    print("\n----------------------------------")

    if mode_choice == "1":
        # Attack success rate summary
        if total_images > 0:
            success_rate = (successful_attacks / total_images) * 100
        else:
            success_rate = 0.0

        print("\nAttack Summary")
        print(f"\nTotal Images Tested: {total_images}")
        print(f"Successful Attacks: {successful_attacks}")
        print(f"Attack Success Rate: {success_rate:.2f}%")
    else:
        # Print benchmark
        print_attack_benchmark(attack_metrics, total_images)


if __name__ == "__main__":
    main()