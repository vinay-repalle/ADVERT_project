# metrics.py
# This file will compute evaluation metrics such as attack success rate, time cost, and perturbation distance.

def update_attack_metrics(metrics, attack_name, is_success, conf_drop):
    """
    Update metrics for a specific attack.

    Parameters:
    - metrics: dict of attack metrics
    - attack_name: name of the attack
    - is_success: bool, whether attack was successful
    - conf_drop: float, confidence drop
    """
    if attack_name not in metrics:
        metrics[attack_name] = {'successes': 0, 'total': 0, 'conf_drop_sum': 0.0}
    metrics[attack_name]['total'] += 1
    if is_success:
        metrics[attack_name]['successes'] += 1
    metrics[attack_name]['conf_drop_sum'] += conf_drop

def print_attack_benchmark(metrics, total_images):
    """
    Print the attack benchmark table.

    Parameters:
    - metrics: dict of attack metrics
    - total_images: total number of images tested
    """
    print("\n" + "="*30)
    print("ADVRET Attack Benchmark")
    print("="*30)
    print("\n## Attack            Success Rate    Avg Confidence Drop")
    print("-" * 55)

    for attack_name in ['FGSM', 'PGD', 'DeepFool', 'Carlini-Wagner']:
        if attack_name in metrics:
            data = metrics[attack_name]
            success_rate = (data['successes'] / data['total']) * 100 if data['total'] > 0 else 0
            avg_conf_drop = data['conf_drop_sum'] / data['total'] if data['total'] > 0 else 0
            print(f"{attack_name:<15} {success_rate:>12.2f}% {avg_conf_drop:>18.2f}")
        else:
            print(f"{attack_name:<15} {'N/A':>12} {'N/A':>18}")

    print(f"\nImages Tested: {total_images}")
