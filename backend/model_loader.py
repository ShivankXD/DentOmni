"""
model_loader.py
Handles loading and inference for both DentOmni models:
  1. Faster R-CNN + ResNet-50  (Detectron2)   → bounding boxes + confidence
  2. ResNet-50 Classifier      (torchvision)   → caries/periapical classification
"""

from __future__ import annotations

import io
import os
import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths (relative to project root — adjust if needed)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FRCNN_WEIGHTS  = ROOT / "fasterrcnn_resnet50_detectron2_model.pth"
RESNET_WEIGHTS = ROOT / "resnet50_weights.pth"

# Class names used during training (order must match training label map)
CLASS_NAMES = ["caries", "periapical_lesion"]   # index 0, 1

# Detection confidence threshold
DETECTION_THRESHOLD = 0.40

logger = logging.getLogger("dentomni")


# ===========================================================================
# Utility helpers
# ===========================================================================

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image → OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_b64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image → base64 PNG string (for JSON transport)."""
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def draw_boxes_pil(img: Image.Image, boxes: list[dict]) -> Image.Image:
    """Draw bounding boxes + labels on a PIL image."""
    draw   = ImageDraw.Draw(img)
    colors = {"caries": "#ffb347", "periapical_lesion": "#ff3c6e"}
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        cls   = box["class"]
        conf  = box["confidence"]
        x1,y1,x2,y2 = box["bbox"]
        color = colors.get(cls, "#00c8f8")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls.replace('_',' ').title()} {conf:.0%}"
        # small filled background for label
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 2), label, fill="#000000", font=font)

    return img


# ===========================================================================
# 1. Faster R-CNN (Detectron2)
# ===========================================================================

class FasterRCNNModel:
    """Wraps Detectron2 DefaultPredictor for DentOmni inference."""

    def __init__(self):
        self._predictor = None

    def load(self):
        if not FRCNN_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Faster R-CNN weights not found: {FRCNN_WEIGHTS}\n"
                "Place 'fasterrcnn_resnet50_detectron2_model.pth' in the project root."
            )
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.model_zoo import model_zoo
        except ImportError:
            raise ImportError(
                "Detectron2 is not installed.\n"
                "Install via: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS          = str(FRCNN_WEIGHTS)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THRESHOLD
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self._predictor = DefaultPredictor(cfg)
        logger.info("Faster R-CNN model loaded (device=%s)", cfg.MODEL.DEVICE)

    def predict(self, img: Image.Image) -> dict:
        if self._predictor is None:
            self.load()

        img_rgb  = np.array(img.convert("RGB"))
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        outputs  = self._predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        boxes_raw  = instances.pred_boxes.tensor.numpy()   # (N,4) x1y1x2y2
        scores_raw = instances.scores.numpy()               # (N,)
        classes_raw = instances.pred_classes.numpy()        # (N,)

        # ── per-class aggregation ──────────────────────────────────────
        class_best: dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
        bboxes: list[dict] = []

        for i, (score, cls_idx) in enumerate(zip(scores_raw, classes_raw)):
            cls_name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else "unknown"
            class_best[cls_name] = max(class_best.get(cls_name, 0.0), float(score))
            x1, y1, x2, y2 = boxes_raw[i].tolist()
            bboxes.append({
                "class":      cls_name,
                "confidence": round(float(score), 4),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            })

        # ── annotated image ───────────────────────────────────────────
        annotated_pil = draw_boxes_pil(img.copy(), bboxes)
        annotated_b64 = cv2_to_b64(pil_to_cv2(annotated_pil))

        return {
            "caries": {
                "detected":    class_best["caries"] >= DETECTION_THRESHOLD,
                "confidence":  round(class_best["caries"], 4),
            },
            "periapical_lesion": {
                "detected":    class_best["periapical_lesion"] >= DETECTION_THRESHOLD,
                "confidence":  round(class_best["periapical_lesion"], 4),
            },
            "bounding_boxes":   bboxes,
            "annotated_image":  annotated_b64,
        }


# ===========================================================================
# 2. ResNet-50 Classifier
# ===========================================================================

class ResNet50Classifier:
    """
    Multi-label ResNet-50 classifier.
    Expected output layer: Linear(in_features, 2)
    Classes: [caries, periapical_lesion]
    Activation: sigmoid  (independent probabilities per class)
    """

    def __init__(self):
        self._model  = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if not RESNET_WEIGHTS.exists():
            raise FileNotFoundError(
                f"ResNet-50 weights not found: {RESNET_WEIGHTS}\n"
                "Place 'resnet50_weights.pth' in the project root."
            )

        model = tv_models.resnet50(weights=None)
        # Replace final FC to match training (2 classes)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.to(self._device)

        # Load weights — handle both raw state-dict and checkpoint dicts
        ckpt = torch.load(str(RESNET_WEIGHTS), map_location=self._device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model.load_state_dict(ckpt, strict=False)
        model.eval()
        self._model = model
        logger.info("ResNet-50 classifier loaded (device=%s)", self._device)

    def predict(self, img: Image.Image) -> dict:
        if self._model is None:
            self.load()

        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        tensor = transform(img.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)          # (1, 2)
            probs  = torch.sigmoid(logits)[0]     # (2,) — independent per class

        p_caries  = float(probs[0].item())
        p_lesion  = float(probs[1].item())

        return {
            "caries": {
                "detected":   p_caries  >= 0.5,
                "confidence": round(p_caries,  4),
            },
            "periapical_lesion": {
                "detected":   p_lesion  >= 0.5,
                "confidence": round(p_lesion,  4),
            },
            "bounding_boxes":  [],          # classifier doesn't produce boxes
            "annotated_image": None,
        }


# ===========================================================================
# Singleton registry — load each model once on first use
# ===========================================================================

_frcnn    = FasterRCNNModel()
_resnet50 = ResNet50Classifier()


def get_model(name: str) -> FasterRCNNModel | ResNet50Classifier:
    if name == "faster_rcnn":
        return _frcnn
    elif name == "resnet50":
        return _resnet50
    raise ValueError(f"Unknown model '{name}'. Choose 'faster_rcnn' or 'resnet50'.")
