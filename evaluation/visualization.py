# evaluation/visualization.py
# This module provides visualization and analysis for attack benchmarking results.

import matplotlib.pyplot as plt
import pandas as pd
import os

plt.style.use("seaborn-v0_8")

def plot_attack_success_rates(results):
    """
    Generate a bar chart of attack success rates.

    Parameters:
    - results: dict with attack results
    """
    attacks = list(results.keys())
    success_rates = [results[attack]['success_rate'] * 100 for attack in attacks]

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    bars = plt.bar(attacks, success_rates, color='skyblue', edgecolor='black', linewidth=1)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Adversarial Attack Success Rate Comparison', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    # Annotate values on the bars (including zeros)
    for i, v in enumerate(success_rates):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/attack_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_drop(results):
    """
    Generate a bar chart of average confidence drop.

    Parameters:
    - results: dict with attack results
    """
    attacks = list(results.keys())
    conf_drops = [results[attack]['confidence_drop'] for attack in attacks]

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    bars = plt.bar(attacks, conf_drops, color='salmon', edgecolor='black', linewidth=1)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Average Confidence Drop', fontsize=12)
    plt.title('Average Confidence Drop by Attack', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)

    # Determine y-axis limits to show negative values clearly
    min_drop = min(conf_drops + [0])
    max_drop = max(conf_drops + [0])
    padding = max(0.1, (max_drop - min_drop) * 0.1)
    plt.ylim(min_drop - padding, max_drop + padding)

    # Annotate values on the bars (handles negative values)
    for bar, value in zip(bars, conf_drops):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.01 if height >= 0 else -0.01
        plt.text(bar.get_x() + bar.get_width() / 2, height + offset, f"{value:.2f}", ha='center', va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/confidence_drop.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_attack_comparison(results):
    """
    Generate a combined bar chart comparing success rates and confidence drops.

    Parameters:
    - results: dict with attack results
    """
    attacks = list(results.keys())
    success_rates = [results[attack]['success_rate'] * 100 for attack in attacks]
    conf_drops = [results[attack]['confidence_drop'] for attack in attacks]

    x = range(len(attacks))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    bars1 = ax1.bar([i - width/2 for i in x], success_rates, width, label='Success Rate (%)', color='skyblue', edgecolor='black', linewidth=1)
    ax1.set_xlabel('Attack Type', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attacks, rotation=45)

    # Ensure y-axis covers 0 for clear reference
    ax1.set_ylim(0, max(100, max(success_rates) * 1.1))

    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], conf_drops, width, label='Confidence Drop', color='salmon', edgecolor='black', linewidth=1)
    ax2.set_ylabel('Average Confidence Drop', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')

    # Adjust y-axis range to show negative drops clearly
    min_drop = min(conf_drops + [0])
    max_drop = max(conf_drops + [0])
    padding = max(0.1, (max_drop - min_drop) * 0.1)
    ax2.set_ylim(min_drop - padding, max_drop + padding)

    # Annotate bars for clarity
    for bar, value in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{value:.0f}%", ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars2, conf_drops):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.01 if height >= 0 else -0.01
        ax2.text(bar.get_x() + bar.get_width() / 2, height + offset, f"{value:.2f}", ha='center', va=va, fontsize=9)

    plt.title('Attack Comparison: Success Rate vs Confidence Drop', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig('outputs/attack_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_robustness_curve(model, device, image_paths, predicted_classes):
    """
    Generate a robustness curve showing model accuracy vs attack strength (epsilon).

    Parameters:
    - model: the loaded model
    - device: torch device
    - image_paths: list of image paths
    - predicted_classes: list of predicted classes
    """
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
    accuracies = []

    from attacks.fgsm_attack import fgsm_attack
    from dataset.image_loader import load_image
    from utils.predict import predict
    import torch

    for eps in epsilons:
        correct, total = 0, 0
        for path, label in zip(image_paths, predicted_classes):
            img = load_image(path, device)
            if eps == 0.0:
                adv = img  # No attack
            else:
                adv = fgsm_attack(model, img, torch.tensor([label]).to(device), epsilon=eps)
            adv_pred, _ = predict(model, adv)
            total += 1
            if adv_pred == label:
                correct += 1
        accuracies.append((correct / total) * 100 if total > 0 else 0)

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.plot(epsilons, accuracies, marker="o", linewidth=2, color="#1F77B4")
    plt.title("Model Robustness Curve: Accuracy vs Attack Strength", fontsize=16, fontweight="bold")
    plt.xlabel("Epsilon (Attack Strength)", fontsize=12)
    plt.ylabel("Model Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)

    for x, y in zip(epsilons, accuracies):
        plt.text(x, y + 1, f"{y:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/robustness_curve.png', dpi=300, bbox_inches="tight")
    plt.close()

def print_detailed_summary(results):
    """
    Analyze attack results and print summary.

    Parameters:
    - results: dict with attack results
    """
    if not results:
        print("No results to analyze.")
        return

    # Find highest success rate
    max_success_attack = max(results, key=lambda x: results[x]['success_rate'])
    max_success_rate = results[max_success_attack]['success_rate'] * 100

    # Find largest confidence drop
    max_conf_drop_attack = max(results, key=lambda x: results[x]['confidence_drop'])
    max_conf_drop = results[max_conf_drop_attack]['confidence_drop']

    # Average success rate
    avg_success_rate = sum(results[attack]['success_rate'] for attack in results) / len(results) * 100

    print("\n" + "="*30)
    print("Analysis Summary")
    print("="*30)
    print(f"Most Powerful Attack: {max_success_attack} ({max_success_rate:.1f}%)")
    print(f"Highest Confidence Impact: {max_conf_drop_attack} ({max_conf_drop:.2f})")
    print(f"Average Attack Success Rate: {avg_success_rate:.1f}%")

def suggest_defenses(results):
    """
    Suggest defense strategies based on attack performance.

    Parameters:
    - results: dict with attack results
    """
    print("\n" + "="*30)
    print("Suggested Defense Strategies")
    print("="*30)

    suggestions = [
        "1. Adversarial Training\n   Train the model using adversarial examples to improve robustness.",
        "2. Input Preprocessing\n   Apply image denoising or feature squeezing before prediction.",
        "3. Gradient Masking Defense\n   Reduce gradient sensitivity to prevent gradient-based attacks.",
        "4. Defensive Distillation\n   Train a secondary model to reduce adversarial vulnerability.",
        "5. Model Ensemble\n   Combine predictions from multiple models."
    ]

    for suggestion in suggestions:
        print(suggestion)
    print()

def generate_visualizations(results, model=None, device=None, image_paths=None, predicted_classes=None):
    """
    Generate all visualizations and analysis.

    Parameters:
    - results: dict with attack results
    - model: the loaded model (for robustness curve)
    - device: torch device
    - image_paths: list of image paths
    - predicted_classes: list of predicted classes
    """
    print("\nGenerating benchmark graphs...")

    plot_attack_success_rates(results)
    plot_confidence_drop(results)
    plot_attack_comparison(results)

    if model and device and image_paths and predicted_classes:
        generate_robustness_curve(model, device, image_paths, predicted_classes)

    print("Saved:")
    print("outputs/attack_success_rates.png")
    print("outputs/confidence_drop.png")
    print("outputs/attack_comparison.png")
    if model:
        print("outputs/robustness_curve.png")

    print_detailed_summary(results)
    suggest_defenses(results)