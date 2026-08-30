"""Gradio demo app for the frozen-v1 BreastUS-CAD pipeline (manual Step 11).

Research prototype — not for clinical use. Not a medical device.

Inference (CPU, loaded once at startup) is the frozen pipeline exactly:
5x cv_vit_fold{1-5} checkpoints x {original, hflip} sigmoid probs averaged,
temperature scaling and decision threshold read from models/calibration.json
and models/operating_point.json. The lesion contour comes from the
segmentation demo model (seg_unet_effb0) and the heatmap is the frozen
saliency method (fold-5 ckpt, Grad-CAM @ blocks[-2].norm1, predicted-class
target) — a single-model visualization, while the decision is the ensemble.

Usage: python app/app.py
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_repo_src = HERE.parent / "src"
if _repo_src.exists():  # repo layout: app/app.py beside src/ and models/
    sys.path.insert(0, str(_repo_src))
    os.chdir(HERE.parent)  # weights resolve via models/ relative to repo root
# Space layout: inference.py sits next to app.py; weights come from the HF Hub

import cv2
import gradio as gr
import numpy as np
import torch

from inference import (
    CAM_CKPT_FILE,
    CKPT_FILES,
    calibrate_probs,
    cam_overlay,
    ensemble_prob,
    load_calibration,
    load_classifier,
    load_operating_point,
    load_seg_model,
    make_cam,
    preprocess_gray,
    seg_prob_map,
)

EXAMPLES_DIR = HERE / "examples"
CONTOUR_BGR_RGB = (57, 255, 20)  # neon green, visible on grayscale US

BANNER_HTML = (
    "<div style='background:#7f1d1d;color:#fff;padding:10px 16px;border-radius:8px;"
    "font-weight:600;text-align:center;font-size:1.05em'>"
    "Research prototype — not for clinical use. Not a medical device.</div>"
)

ABOUT_MD = """\
**Data & scope.** The classifier was trained on the public **BUS-BRA** dataset
(1,875 images / 1,064 patients, official patient-level 5-fold splits) and
externally validated single-shot on four cohorts (BrEaST, BUSI, GDPH, SYSUCC;
AUC 0.84–0.93). The example images below are from the BUS-BRA validation fold
(fold 5), held out from the checkpoint used for the heatmap.
*Example image attribution: BUS-BRA dataset (Gómez-Flores et al., Medical
Physics 2024), used under its research license.*

**Known limitations.** Calibrated probabilities shift under domain shift: on
external cohorts, sensitivity held ≥ 0.92 at the frozen threshold but
specificity dropped from 0.77 (internal) to 0.41–0.63 — the operating point is
calibrated on internal validation only and does not transfer across scanners
and populations. The Grad-CAM heatmap is a single-model visualization, not the
ensemble decision path, and the lesion contour is a separate segmentation
model provided for orientation only.
"""


def load_pipeline():
    """Load every frozen artifact once, on CPU. Nothing hardcoded."""
    classifiers = [load_classifier(f) for f in CKPT_FILES]
    seg_model = load_seg_model()
    cam = make_cam(load_classifier(CAM_CKPT_FILE))
    return classifiers, seg_model, cam, load_calibration(), load_operating_point()


CLASSIFIERS, SEG_MODEL, CAM, TEMPERATURE, THRESHOLD = load_pipeline()


def preprocess(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Grayscale -> 3ch -> frozen val transforms. Returns (1,3,H,W) and raw gray."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise gr.Error("Could not read the image file.")
    return preprocess_gray(gray), gray


def lesion_contour_panel(x: torch.Tensor, gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Original image with the predicted lesion contour; returns (panel, n_contours)."""
    prob_map = seg_prob_map(SEG_MODEL, x)
    mask = cv2.resize(
        (prob_map >= 0.5).astype(np.uint8),
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel = np.repeat(gray[:, :, None], 3, axis=2)
    thickness = max(2, round(min(gray.shape[:2]) / 150))
    cv2.drawContours(panel, contours, -1, CONTOUR_BGR_RGB, thickness)
    return panel, len(contours)


def result_card(p_cal: float, suspicious: bool) -> str:
    pct = p_cal * 100
    thr_pct = THRESHOLD * 100
    color = "#dc2626" if suspicious else "#16a34a"
    call = "SUSPICIOUS" if suspicious else "LIKELY BENIGN"
    return f"""
<div style='border:1px solid #d1d5db;border-radius:10px;padding:16px;font-family:sans-serif'>
  <div style='font-size:0.9em;color:#6b7280'>Calibrated malignancy probability
  (5-model ensemble, hflip TTA, temperature-scaled)</div>
  <div style='font-size:2.2em;font-weight:700;color:{color};margin:4px 0'>{pct:.1f}%</div>
  <div style='position:relative;height:22px;background:#e5e7eb;border-radius:11px;overflow:hidden'>
    <div style='position:absolute;left:0;top:0;height:100%;width:{pct:.2f}%;
        background:{color};border-radius:11px'></div>
    <div style='position:absolute;left:{thr_pct:.2f}%;top:0;height:100%;width:2px;
        background:#111827' title='operating point'></div>
  </div>
  <div style='font-size:0.75em;color:#6b7280;margin-top:2px'>
    black tick = operating point ({THRESHOLD:.4f})</div>
  <div style='font-size:1.3em;font-weight:600;color:{color};margin-top:10px'>{call}</div>
  <div style='font-size:0.85em;color:#374151'>
    threshold set for sensitivity ≥ 0.90 on internal validation</div>
  <div style='font-size:0.75em;color:#6b7280;margin-top:10px'>
    Heatmap is a single-model (fold-5) Grad-CAM visualization; the decision above
    comes from the 5-model ensemble. Research prototype — not for clinical use.</div>
</div>"""


def predict(image_path: str | None):
    if image_path is None:
        raise gr.Error("Please upload a breast ultrasound image.")
    x, gray = preprocess(image_path)

    raw = ensemble_prob(x, CLASSIFIERS)
    p_cal = float(calibrate_probs(np.array([raw]), TEMPERATURE)[0])
    suspicious = p_cal >= THRESHOLD

    contour_panel, n_contours = lesion_contour_panel(x, gray)
    cam_panel = cam_overlay(CAM, image_path, int(suspicious))
    cam_panel = cv2.resize(
        cam_panel, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    card = result_card(p_cal, suspicious)
    if n_contours == 0:
        card += ("<div style='font-size:0.8em;color:#6b7280;margin-top:4px'>"
                 "No lesion contour detected by the segmentation model.</div>")
    # full-precision values for the deployment verification script (invisible)
    card += f"<!-- raw={raw!r} cal={p_cal!r} -->"
    return contour_panel, cam_panel, card


def startup_latency(example: Path) -> None:
    """One warm-up + one timed full inference on CPU (target < 3 s)."""
    predict(str(example))  # warm-up (first call pays lazy-init costs)
    t0 = time.perf_counter()
    predict(str(example))
    dt = time.perf_counter() - t0
    print(f"end-to-end CPU latency (1 image, ensemble+TTA+seg+CAM): {dt:.2f} s "
          f"(target < 3 s) — measured on {example.name}")


def build_ui() -> gr.Blocks:
    examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.png"))
    with gr.Blocks(title="BreastUS-CAD — research prototype") as demo:
        gr.HTML(BANNER_HTML)
        gr.Markdown("## BreastUS-CAD demo — benign/malignant classifier (frozen-v1)")
        with gr.Row():
            inp = gr.Image(type="filepath", label="Breast ultrasound image (grayscale)")
            out_contour = gr.Image(label="Predicted lesion contour (U-Net)")
            out_cam = gr.Image(label="Grad-CAM (fold-5 model, predicted class)")
        out_card = gr.HTML(label="Result")
        btn = gr.Button("Analyze", variant="primary")
        btn.click(predict, inputs=inp, outputs=[out_contour, out_cam, out_card],
                  api_name="predict")
        gr.Examples(
            examples=examples,
            inputs=inp,
            label="BUS-BRA fold-5 examples (2 benign, 2 malignant — see About for attribution)",
        )
        with gr.Accordion("About this demo", open=False):
            gr.Markdown(ABOUT_MD)
        gr.HTML(BANNER_HTML)
    return demo


if __name__ == "__main__":
    if not os.environ.get("SPACE_ID"):  # skip the timing run on Spaces (cold start)
        startup_latency(EXAMPLES_DIR / "benign_bus_0186-r.png")
    build_ui().launch()
