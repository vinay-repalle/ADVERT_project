import os
import zipfile
from datetime import datetime

import torch
from flask import Flask, flash, redirect, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from attacks.cw_attack import cw_attack
from attacks.deepfool_attack import deepfool_attack
from attacks.fgsm_attack import fgsm_attack
from attacks.pgd_attack import pgd_attack
from dataset.image_loader import load_image
from evaluation.metrics import get_attack_results, update_attack_metrics
from evaluation.visualization import generate_visualizations
from models.model_loader import load_model
from utils.predict import predict

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "advret-secret")

# Force template auto-reload (helps when running from synced folders like OneDrive)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
MODEL_UPLOAD_FOLDER = os.path.join(UPLOAD_ROOT, "models")
IMAGE_UPLOAD_FOLDER = os.path.join(UPLOAD_ROOT, "images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_MODEL_EXT = {".pt", ".pth"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}

ACTIVE_MODEL = {"path": None, "name": None, "model": None, "device": None}


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def clear_folder(folder_path: str) -> None:
    if not os.path.isdir(folder_path):
        return
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if os.path.isfile(path):
            os.remove(path)


def clear_active_model() -> None:
    """Delete the current uploaded model file and clear in-memory model."""
    model_path = ACTIVE_MODEL.get("path")
    if model_path and isinstance(model_path, str):
        try:
            abs_path = os.path.abspath(model_path)
            abs_models_dir = os.path.abspath(MODEL_UPLOAD_FOLDER)
            if abs_path.startswith(abs_models_dir) and os.path.isfile(abs_path):
                os.remove(abs_path)
        except Exception:
            pass
    ACTIVE_MODEL.update({"path": None, "name": None, "model": None, "device": None})


def clear_uploads() -> None:
    """Clear all uploaded files (models + images)."""
    clear_folder(IMAGE_UPLOAD_FOLDER)
    clear_folder(MODEL_UPLOAD_FOLDER)
    clear_active_model()


def save_uploaded_file(file_storage, dest_folder: str) -> str:
    filename = file_storage.filename
    if not filename:
        return ""

    filename = secure_filename(os.path.basename(filename))
    if not filename:
        return ""
    os.makedirs(dest_folder, exist_ok=True)
    target_path = os.path.join(dest_folder, filename)

    # Uploads are temporary: overwrite instead of keeping history.
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception:
            pass

    file_storage.save(target_path)
    return target_path


def delete_file_if_in_dir(file_path: str, allowed_dir: str) -> None:
    """Best-effort delete for a file within a specific directory."""
    try:
        abs_path = os.path.abspath(file_path)
        abs_allowed_dir = os.path.abspath(allowed_dir)
        if abs_path.startswith(abs_allowed_dir) and os.path.isfile(abs_path):
            os.remove(abs_path)
    except Exception:
        pass


def load_custom_model(model_path: str, device: torch.device):
    loaded = torch.load(model_path, map_location=device)
    if hasattr(loaded, "eval") and hasattr(loaded, "state_dict"):
        model = loaded
    else:
        model, _ = load_model("1", device)
        model.load_state_dict(loaded)
    model = model.to(device)
    model.eval()
    return model


def generate_reports_text(results: dict) -> tuple[str, str]:
    if not results:
        return "No results available.", "No defense suggestions available."

    max_success_attack = max(results, key=lambda x: results[x]["success_rate"])
    max_success_rate = results[max_success_attack]["success_rate"] * 100

    max_conf_drop_attack = max(results, key=lambda x: results[x]["confidence_drop"])
    max_conf_drop = results[max_conf_drop_attack]["confidence_drop"]

    avg_success_rate = sum(results[a]["success_rate"] for a in results) / len(results) * 100

    summary_text = (
        f"Most Powerful Attack: {max_success_attack} ({max_success_rate:.1f}%)\n"
        f"Highest Confidence Impact: {max_conf_drop_attack} ({max_conf_drop:.2f})\n"
        f"Average Attack Success Rate: {avg_success_rate:.1f}%\n"
    )

    suggestions_text = "\n".join(
        [
            "1. Adversarial Training: Train the model using adversarial examples to improve robustness.",
            "2. Input Preprocessing: Apply image denoising or feature squeezing before prediction.",
            "3. Gradient Masking Defense: Reduce gradient sensitivity to prevent gradient-based attacks.",
            "4. Defensive Distillation: Train a secondary model to reduce adversarial vulnerability.",
            "5. Model Ensemble: Combine predictions from multiple models.",
        ]
    )
    return summary_text, suggestions_text


def make_report_zip(output_folder: str) -> str:
    report_path = os.path.join(output_folder, "advret_report.zip")
    with zipfile.ZipFile(report_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for fname in [
            "attack_success_rates.png",
            "confidence_drop.png",
            "attack_comparison.png",
            "robustness_curve.png",
            "summary.txt",
            "defense_suggestions.txt",
        ]:
            fpath = os.path.join(output_folder, fname)
            if os.path.exists(fpath):
                archive.write(fpath, arcname=fname)
    return report_path


@app.route("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/download-report")
def download_report():
    report_path = os.path.join(OUTPUT_FOLDER, "advret_report.zip")
    if not os.path.exists(report_path):
        flash("No report available. Run an analysis first.", "warning")
        return redirect(url_for("index"))
    return send_file(report_path, as_attachment=True, download_name="advret_report.zip")


@app.route("/", methods=["GET"])
def index():
    # Treat uploads as temporary: a page refresh clears previous uploads.
    clear_uploads()
    return render_template(
        "index.html",
        active_model=ACTIVE_MODEL.get("name"),
        attack_mode="single",
        attack_name="FGSM",
    )


@app.route("/run", methods=["POST"])
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_choice = request.form.get("pretrained_model", "1")
    custom_model_file = request.files.get("custom_model")
    custom_model_uploaded = False

    if custom_model_file and custom_model_file.filename:
        # New model upload replaces any previous uploaded model.
        clear_folder(MODEL_UPLOAD_FOLDER)
        clear_active_model()
        if not allowed_file(custom_model_file.filename, ALLOWED_MODEL_EXT):
            flash("Invalid model file type. Use .pt or .pth.", "error")
            return redirect(url_for("index"))
        model_path = save_uploaded_file(custom_model_file, MODEL_UPLOAD_FOLDER)
        try:
            model = load_custom_model(model_path, device)
        except Exception as exc:
            flash(f"Failed to load custom model: {exc}", "error")
            return redirect(url_for("index"))
        custom_model_uploaded = True
        ACTIVE_MODEL.update(
            {"path": model_path, "name": os.path.basename(model_path), "model": model, "device": device}
        )
        active_model_name = ACTIVE_MODEL["name"]
    else:
        if ACTIVE_MODEL.get("model") is not None:
            model = ACTIVE_MODEL["model"]
            device = ACTIVE_MODEL.get("device", device)
            active_model_name = ACTIVE_MODEL.get("name")
        else:
            model, device = load_model(pretrained_choice, device)
            active_model_name = {
                "1": "ResNet18",
                "2": "MobileNetV2",
                "3": "VGG16",
            }.get(str(pretrained_choice), "ResNet18")

    uploaded_images = request.files.getlist("images")
    if not uploaded_images or all(img.filename == "" for img in uploaded_images):
        flash("Please upload at least one image file.", "warning")
        return redirect(url_for("index"))

    # New image upload replaces any previous uploaded images.
    clear_folder(IMAGE_UPLOAD_FOLDER)
    saved_image_paths: list[str] = []
    for img in uploaded_images:
        if img and img.filename and allowed_file(img.filename, ALLOWED_IMAGE_EXT):
            saved_path = save_uploaded_file(img, IMAGE_UPLOAD_FOLDER)
            if saved_path:
                saved_image_paths.append(saved_path)

    if not saved_image_paths:
        flash("No valid images were uploaded. Supported formats: .jpg, .jpeg, .png", "warning")
        return redirect(url_for("index"))

    attack_mode = request.form.get("attack_mode", "single")
    single_attack = request.form.get("single_attack", "FGSM")

    for fname in [
        "attack_success_rates.png",
        "confidence_drop.png",
        "attack_comparison.png",
        "robustness_curve.png",
        "summary.txt",
        "defense_suggestions.txt",
        "advret_report.zip",
    ]:
        fpath = os.path.join(OUTPUT_FOLDER, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    attacks = {
        "FGSM": lambda m, x, y: fgsm_attack(m, x, y, epsilon=0.01),
        "PGD": lambda m, x, y: pgd_attack(m, x, y, epsilon=0.03, alpha=0.005, iterations=10),
        "DeepFool": lambda m, x, y: deepfool_attack(m, x, max_iter=50, overshoot=0.02),
        "Carlini-Wagner": lambda m, x, y: cw_attack(m, x, y, c=1, kappa=0, iterations=100, learning_rate=0.01),
    }

    image_paths: list[str] = []
    predicted_classes: list[int] = []
    total_images = 0
    successful_attacks = 0
    attack_metrics: dict = {}

    for image_path in saved_image_paths:
        total_images += 1
        image_tensor = load_image(image_path, device)
        predicted_class, confidence_score = predict(model, image_tensor)

        image_paths.append(image_path)
        predicted_classes.append(predicted_class)

        label_tensor = torch.tensor([predicted_class]).to(device)

        if attack_mode == "single":
            attack_name = single_attack
            adversarial = attacks[attack_name](model, image_tensor, label_tensor)
            adv_pred, adv_conf = predict(model, adversarial)
            is_success = adv_pred != predicted_class
            if is_success:
                successful_attacks += 1
            update_attack_metrics(attack_metrics, attack_name, is_success, confidence_score - adv_conf)
        else:
            for attack_name, attack_fn in attacks.items():
                adversarial = attack_fn(model, image_tensor, label_tensor)
                adv_pred, adv_conf = predict(model, adversarial)
                is_success = adv_pred != predicted_class
                update_attack_metrics(attack_metrics, attack_name, is_success, confidence_score - adv_conf)

    results = get_attack_results(attack_metrics)
    generate_visualizations(results, model, device, image_paths, predicted_classes)

    summary_text, suggestions_text = generate_reports_text(results)
    with open(os.path.join(OUTPUT_FOLDER, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open(os.path.join(OUTPUT_FOLDER, "defense_suggestions.txt"), "w", encoding="utf-8") as f:
        f.write(suggestions_text)

    make_report_zip(OUTPUT_FOLDER)

    # Uploaded user files are temporary: delete them after processing.
    for p in saved_image_paths:
        delete_file_if_in_dir(p, IMAGE_UPLOAD_FOLDER)
    if custom_model_uploaded:
        delete_file_if_in_dir(ACTIVE_MODEL.get("path") or "", MODEL_UPLOAD_FOLDER)
        # Keep the in-memory model for this session, but don't keep the file path.
        ACTIVE_MODEL["path"] = None

    graphs = {
        "success": url_for("outputs", filename="attack_success_rates.png"),
        "confidence": url_for("outputs", filename="confidence_drop.png"),
        "comparison": url_for("outputs", filename="attack_comparison.png"),
        "robustness": url_for("outputs", filename="robustness_curve.png"),
    }

    return render_template(
        "results.html",
        active_model=active_model_name,
        attack_mode=attack_mode,
        attack_name=single_attack,
        total_images=total_images,
        successful_attacks=successful_attacks,
        results=results,
        graphs=graphs,
        summary_text=summary_text,
        suggestions_text=suggestions_text,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
