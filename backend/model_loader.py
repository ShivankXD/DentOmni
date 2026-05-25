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
FRCNN_WEIGHTS  = ROOT / "FINAL_dental_model.pth"
RESNET_WEIGHTS = ROOT / "FINAL_resnet50.pth"

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
# 1. Faster R-CNN  (torchvision — no Detectron2 dependency)
# ===========================================================================

def _remap_detectron2_to_torchvision(d2_state: dict) -> dict:
    """
    Convert a Detectron2-format state dict to torchvision FasterRCNN keys.

    Detectron2 naming                      → torchvision naming
    ─────────────────────────────────────────────────────────────────
    backbone.bottom_up.stem.conv1.*        → backbone.body.conv1.*
    backbone.bottom_up.stem.conv1.norm.*   → backbone.body.bn1.*
    backbone.bottom_up.res{N}.{i}.*        → backbone.body.layer{N-1}.{i}.*
      *.shortcut.*                         → *.downsample.*
      *.conv{k}.*                          → *.conv{k}.*
      *.conv{k}.norm.*                     → *.bn{k}.*
    backbone.fpn_lateral{N}.*             → backbone.fpn.inner_blocks.{N-2}.*
    backbone.fpn_output{N}.*              → backbone.fpn.layer_blocks.{N-2}.*
    proposal_generator.rpn_head.conv.*    → rpn.head.conv.*
    proposal_generator.rpn_head.anchor_deltas.* → rpn.head.bbox_pred.*
    proposal_generator.rpn_head.objectness_logits.* → rpn.head.cls_logits.*
    proposal_generator.anchor_generator.* → rpn.anchor_generator.*
    roi_heads.box_head.fc1.*              → roi_heads.box_head.fc6.*
    roi_heads.box_head.fc2.*              → roi_heads.box_head.fc7.*
    roi_heads.box_predictor.cls_score.*   → roi_heads.box_predictor.cls_score.*
    roi_heads.box_predictor.bbox_pred.*   → roi_heads.box_predictor.bbox_pred.*
    """
    import re

    def _convert(key: str) -> str | None:
        # ── backbone stem ─────────────────────────────────────────────
        # backbone.bottom_up.stem.conv1.norm.* → backbone.body.bn1.*
        m = re.match(r"backbone\.bottom_up\.stem\.conv1\.norm\.(.*)", key)
        if m:
            return f"backbone.body.bn1.{m.group(1)}"

        # backbone.bottom_up.stem.conv1.* → backbone.body.conv1.*
        m = re.match(r"backbone\.bottom_up\.stem\.conv1\.(.*)", key)
        if m:
            return f"backbone.body.conv1.{m.group(1)}"

        # ── ResNet stages ─────────────────────────────────────────────
        # backbone.bottom_up.res{N}.{i}.shortcut.norm.* →
        #   backbone.body.layer{N-1}.{i}.downsample.1.*
        m = re.match(
            r"backbone\.bottom_up\.res(\d+)\.(\d+)\.shortcut\.norm\.(.*)", key)
        if m:
            layer = int(m.group(1)) - 1
            return (f"backbone.body.layer{layer}.{m.group(2)}"
                    f".downsample.1.{m.group(3)}")

        # backbone.bottom_up.res{N}.{i}.shortcut.* →
        #   backbone.body.layer{N-1}.{i}.downsample.0.*
        m = re.match(
            r"backbone\.bottom_up\.res(\d+)\.(\d+)\.shortcut\.(.*)", key)
        if m:
            layer = int(m.group(1)) - 1
            return (f"backbone.body.layer{layer}.{m.group(2)}"
                    f".downsample.0.{m.group(3)}")

        # backbone.bottom_up.res{N}.{i}.conv{k}.norm.* →
        #   backbone.body.layer{N-1}.{i}.bn{k}.*
        m = re.match(
            r"backbone\.bottom_up\.res(\d+)\.(\d+)\.conv(\d+)\.norm\.(.*)", key)
        if m:
            layer = int(m.group(1)) - 1
            return (f"backbone.body.layer{layer}.{m.group(2)}"
                    f".bn{m.group(3)}.{m.group(4)}")

        # backbone.bottom_up.res{N}.{i}.conv{k}.* →
        #   backbone.body.layer{N-1}.{i}.conv{k}.*
        m = re.match(
            r"backbone\.bottom_up\.res(\d+)\.(\d+)\.conv(\d+)\.(.*)", key)
        if m:
            layer = int(m.group(1)) - 1
            return (f"backbone.body.layer{layer}.{m.group(2)}"
                    f".conv{m.group(3)}.{m.group(4)}")

        # ── FPN ───────────────────────────────────────────────────────
        # backbone.fpn_lateral{N}.* → backbone.fpn.inner_blocks.{N-2}.*
        m = re.match(r"backbone\.fpn_lateral(\d+)\.(.*)", key)
        if m:
            idx = int(m.group(1)) - 2
            return f"backbone.fpn.inner_blocks.{idx}.0.{m.group(2)}"

        # backbone.fpn_output{N}.* → backbone.fpn.layer_blocks.{N-2}.*
        m = re.match(r"backbone\.fpn_output(\d+)\.(.*)", key)
        if m:
            idx = int(m.group(1)) - 2
            return f"backbone.fpn.layer_blocks.{idx}.0.{m.group(2)}"

        # ── RPN ───────────────────────────────────────────────────────
        m = re.match(r"proposal_generator\.rpn_head\.conv\.(.*)", key)
        if m:
            return f"rpn.head.conv.0.0.{m.group(1)}"

        m = re.match(
            r"proposal_generator\.rpn_head\.anchor_deltas\.(.*)", key)
        if m:
            return f"rpn.head.bbox_pred.{m.group(1)}"

        m = re.match(
            r"proposal_generator\.rpn_head\.objectness_logits\.(.*)", key)
        if m:
            return f"rpn.head.cls_logits.{m.group(1)}"

        m = re.match(r"proposal_generator\.anchor_generator\.(.*)", key)
        if m:
            return f"rpn.anchor_generator.{m.group(1)}"

        # ── ROI Heads ─────────────────────────────────────────────────
        m = re.match(r"roi_heads\.box_head\.fc1\.(.*)", key)
        if m:
            return f"roi_heads.box_head.fc6.{m.group(1)}"

        m = re.match(r"roi_heads\.box_head\.fc2\.(.*)", key)
        if m:
            return f"roi_heads.box_head.fc7.{m.group(1)}"

        m = re.match(r"roi_heads\.box_predictor\.(.*)", key)
        if m:
            return f"roi_heads.box_predictor.{m.group(1)}"

        return None   # unknown key — skip

    new_sd: dict = {}
    skipped: list[str] = []
    for k, v in d2_state.items():
        new_key = _convert(k)
        if new_key is not None:
            new_sd[new_key] = v
        else:
            skipped.append(k)

    if skipped:
        logger.debug("D2→TV: skipped %d keys: %s…", len(skipped), skipped[:5])
    logger.info("D2→TV: mapped %d / %d keys", len(new_sd), len(d2_state))
    return new_sd


class FasterRCNNModel:
    """
    Faster R-CNN + ResNet50-FPN detector.

    Loads a Detectron2-format checkpoint and remaps keys to torchvision's
    FasterRCNN — no Detectron2 installation required.

    Checkpoint: FINAL_dental_model.pth
      { 'model': OrderedDict(<295 Detectron2 keys>),
        'trainer': ..., 'iteration': ... }

    Training num_classes: 1 (Detectron2) → 2 (torchvision, including background)
    cls_score output: [2, 1024]  → background + 1 dental class
    """

    # Map Detectron2 1-class index → CLASS_NAMES index
    # Detectron2 class 0 → could be caries or periapical_lesion
    # We cannot know which without training metadata, so we map to index 0 (caries)
    # as a conservative default. Adjust _D2_CLASS_MAP if needed.
    _D2_CLASS_MAP: dict[int, int] = {0: 0}   # d2_cls → CLASS_NAMES index

    def __init__(self):
        self._model  = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self._model is not None:
            return   # already loaded
        if not FRCNN_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Faster R-CNN weights not found: {FRCNN_WEIGHTS}\n"
                "Place 'FINAL_dental_model.pth' in the project root."
            )

        import torchvision.models.detection as tvd
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # ── load raw checkpoint ────────────────────────────────────────
        raw = torch.load(str(FRCNN_WEIGHTS), map_location=self._device,
                         weights_only=False)
        d2_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

        # Determine number of classes from cls_score shape
        cls_w = d2_sd.get("roi_heads.box_predictor.cls_score.weight")
        num_classes_tv = int(cls_w.shape[0]) if cls_w is not None else 2
        logger.info("Faster R-CNN: detected num_classes=%d (torchvision)", num_classes_tv)

        # ── build torchvision model ────────────────────────────────────
        model = tvd.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_tv)
        model.to(self._device)

        # ── remap & load state dict ────────────────────────────────────
        tv_sd = _remap_detectron2_to_torchvision(d2_sd)

        # bbox_pred: Detectron2 uses class-agnostic regression [4, in_features]
        # but torchvision expects per-class [num_classes*4, in_features].
        # Fix: tile the class-agnostic weights to match the torchvision shape.
        for key_suffix in ("weight", "bias"):
            roi_key = f"roi_heads.box_predictor.bbox_pred.{key_suffix}"
            rpn_key = f"rpn.head.bbox_pred.{key_suffix}"
            for bbox_key in (roi_key, rpn_key):
                if bbox_key not in tv_sd:
                    continue
                src = tv_sd[bbox_key]
                try:
                    tgt_shape = model.state_dict()[bbox_key].shape
                    if src.shape == tgt_shape:
                        pass  # already matches
                    elif tgt_shape[0] % src.shape[0] == 0:
                        # tile to match: e.g. [4] -> [8] for 2 classes
                        repeats = tgt_shape[0] // src.shape[0]
                        tv_sd[bbox_key] = src.repeat(repeats, *([1] * (src.dim() - 1)))
                        logger.info("bbox_pred tiled: %s -> %s",
                                    tuple(src.shape), tuple(tv_sd[bbox_key].shape))
                    else:
                        # shapes incompatible - remove to avoid crash
                        del tv_sd[bbox_key]
                except Exception:
                    del tv_sd[bbox_key]

        missing, unexpected = model.load_state_dict(tv_sd, strict=False)
        if missing:
            logger.warning("Missing keys after remap (%d): %s…",
                           len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys after remap (%d): %s…",
                           len(unexpected), unexpected[:5])

        model.eval()
        self._model = model
        logger.info("Faster R-CNN loaded via torchvision (device=%s)", self._device)

    def predict(self, img: Image.Image) -> dict:
        if self._model is None:
            self.load()

        import torchvision.transforms.functional as TF

        # torchvision FasterRCNN expects a list of float tensors [C,H,W] in [0,1]
        tensor = TF.to_tensor(img.convert("RGB")).to(self._device)

        with torch.no_grad():
            outputs = self._model([tensor])   # list of dict

        out = outputs[0]
        boxes_raw   = out["boxes"].cpu().numpy()    # (N,4) x1y1x2y2
        scores_raw  = out["scores"].cpu().numpy()   # (N,)
        classes_raw = out["labels"].cpu().numpy()   # (N,)  1-indexed in torchvision

        # ── filter by threshold ────────────────────────────────────────
        mask = scores_raw >= DETECTION_THRESHOLD
        boxes_raw   = boxes_raw[mask]
        scores_raw  = scores_raw[mask]
        classes_raw = classes_raw[mask]

        # ── per-class aggregation ──────────────────────────────────────
        class_best: dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
        bboxes: list[dict] = []

        for i, (score, cls_idx) in enumerate(zip(scores_raw, classes_raw)):
            # torchvision: class 0 = background, 1..N = actual classes
            d2_cls = int(cls_idx) - 1          # convert to 0-based
            mapped  = self._D2_CLASS_MAP.get(d2_cls, d2_cls % len(CLASS_NAMES))
            cls_name = CLASS_NAMES[mapped] if 0 <= mapped < len(CLASS_NAMES) else "unknown"
            class_best[cls_name] = max(class_best.get(cls_name, 0.0), float(score))
            x1, y1, x2, y2 = boxes_raw[i].tolist()
            bboxes.append({
                "class":      cls_name,
                "confidence": round(float(score), 4),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            })

        # ── annotated image ───────────────────────────────────────────
        # Cap bboxes to top-20 by confidence to keep response size manageable
        bboxes_top = sorted(bboxes, key=lambda x: x["confidence"], reverse=True)[:20]
        annotated_pil = draw_boxes_pil(img.copy(), bboxes_top)
        annotated_b64 = cv2_to_b64(pil_to_cv2(annotated_pil)) if bboxes_top else None

        logger.info(
            "Faster R-CNN: %d detections above %.0f%% threshold (showing top %d)",
            len(bboxes), DETECTION_THRESHOLD * 100, len(bboxes_top),
        )

        return {
            "caries": {
                "detected":   class_best["caries"] >= DETECTION_THRESHOLD,
                "confidence": round(class_best["caries"], 4),
            },
            "periapical_lesion": {
                "detected":   class_best["periapical_lesion"] >= DETECTION_THRESHOLD,
                "confidence": round(class_best["periapical_lesion"], 4),
            },
            "bounding_boxes":  bboxes_top,
            "annotated_image": annotated_b64,
        }


# ===========================================================================
# 2. ResNet-50 Classifier
# ===========================================================================

class _DentOmniResNet(nn.Module):
    """
    Custom model matching the training architecture:
      ResNet50 backbone (avgpool output 2048)
      → Linear(2048 → 128) + ReLU
      → Linear(128  → 2)
      → Sigmoid (applied at inference, not here)

    Weights are stored in a Keras-style checkpoint with keys:
      'resnet50' : list of 318 tensors (Keras HWIO ordering for convs)
      'dense'    : [weight(2048,128), bias(128)]
      'dense_1'  : [weight(128,2),   bias(2)]
    """

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        # Remove the original ImageNet FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.flatten  = nn.Flatten()
        self.dense    = nn.Linear(2048, 128)
        self.relu     = nn.ReLU(inplace=True)
        self.dense_1  = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 2048, 1, 1)
        x = self.flatten(x)    # (B, 2048)
        x = self.relu(self.dense(x))   # (B, 128)
        x = self.dense_1(x)            # (B, 2)
        return x


def _load_keras_weights_into_resnet(model: _DentOmniResNet, ckpt: dict,
                                     device: str) -> None:
    """
    Map Keras-ordered backbone tensors → PyTorch ResNet50 parameters.

    Keras ResNet50 weight ordering (from tf.keras.applications.ResNet50 source):

    Within each bottleneck block, Keras stores:
      [conv1_kernel, bn1_γ, bn1_β, bn1_mean, bn1_var, bn1_n,
       conv2_kernel, bn2_γ, bn2_β, bn2_mean, bn2_var, bn2_n,
       conv3_kernel,
       bn3_γ,
       [shortcut_conv_kernel, shortcut_bn_γ, shortcut_bn_β,
        shortcut_bn_mean, shortcut_bn_var, shortcut_bn_n]   ← ONLY if shortcut exists
       bn3_β, bn3_mean, bn3_var, bn3_n]

    Stem: [conv1_kernel, bn1_γ, bn1_β, bn1_mean, bn1_var, bn1_n]

    Keras conv kernels are HWIO; PyTorch OIHW — need permute(3,2,0,1).
    """
    keras_tensors: list = ckpt["resnet50"]       # 318 tensors total
    keras_dense:   list = ckpt["dense"]           # [W(2048,128), b(128)]
    keras_dense_1: list = ckpt["dense_1"]         # [W(128,2),   b(2)]

    cursor = [0]   # mutable cursor through keras_tensors

    def _next() -> torch.Tensor:
        """Consume and return the next Keras tensor."""
        t = keras_tensors[cursor[0]].to(device)
        cursor[0] += 1
        return t

    def _load_conv(conv: nn.Conv2d) -> None:
        """Load next conv kernel (transpose HWIO→OIHW)."""
        kt = _next()
        with torch.no_grad():
            conv.weight.copy_(kt.permute(3, 2, 0, 1).contiguous())

    def _load_bn(bn: nn.BatchNorm2d) -> None:
        """
        Load 5 Keras BN tensors in their actual order:
          [0] beta  (bias, ~0)        → bn.bias
          [1] gamma (weight, ~1.0)    → bn.weight
          [2] running_mean            → bn.running_mean
          [3] unknown/skip (can be negative)
          [4] running_var (large +ve) → bn.running_var
        """
        with torch.no_grad():
            bn.bias.copy_(_next())         # β
            bn.weight.copy_(_next())       # γ
            bn.running_mean.copy_(_next()) # mean
        _next()                            # [3] unknown — skip
        with torch.no_grad():
            bn.running_var.copy_(_next())  # var (large positive values)

    def _load_bn_shortcut_before(bn: nn.BatchNorm2d) -> None:
        """Load ONLY beta (1st tensor) of shortcut BN before conv3 is inserted."""
        with torch.no_grad():
            bn.bias.copy_(_next())    # β (only 1 tensor, conv3 interrupts here)

    def _load_bn_shortcut_after(bn: nn.BatchNorm2d) -> None:
        """Load remaining 4 shortcut BN tensors after conv3: γ, mean, skip, var."""
        with torch.no_grad():
            bn.weight.copy_(_next())       # γ
            bn.running_mean.copy_(_next()) # mean
        _next()                            # unknown — skip
        with torch.no_grad():
            bn.running_var.copy_(_next())  # var

    def _load_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        """Load a complete conv+BN pair (6 tensors)."""
        _load_conv(conv)
        _load_bn(bn)

    def _load_bn_shortcut(bn: nn.BatchNorm2d) -> None:
        """Load ALL 5 shortcut BN tensors: beta, gamma, mean, skip, var."""
        with torch.no_grad():
            bn.bias.copy_(_next())         # beta
            bn.weight.copy_(_next())       # gamma
            bn.running_mean.copy_(_next()) # mean
        _next()                            # unknown - skip
        with torch.no_grad():
            bn.running_var.copy_(_next())  # var

    def _load_bottleneck(block, has_shortcut: bool) -> None:
        """
        Load one ResNet bottleneck block following the VERIFIED Keras ordering.

        Confirmed by inspecting tensor positional values in checkpoint.

        Shortcut block (24 tensors):
          conv1 + bn1(beta,gamma,mean,unk,var)     [6]
          conv2 + bn2(beta,gamma,mean,unk,var)     [6]
          shortcut_conv_kernel                      [1]
          bn3_beta    <- bn3.bias loaded early      [1]
          conv3_kernel                              [1]
          shortcut_bn(beta,gamma,mean,unk,var)      [5]
          bn3_rest(gamma,mean,unk,var)              [4]

        Non-shortcut block (18 tensors):
          conv1+bn1, conv2+bn2, conv3+bn3
        """
        _load_conv_bn_pair(block.conv1, block.bn1)
        _load_conv_bn_pair(block.conv2, block.bn2)

        if has_shortcut:
            shortcut_conv = block.downsample[0]
            shortcut_bn   = block.downsample[1]

            # Shortcut conv kernel
            _load_conv(shortcut_conv)
            # bn3.bias (beta) comes BEFORE conv3 in Keras ordering
            with torch.no_grad():
                block.bn3.bias.copy_(_next())
            # Conv3 kernel
            _load_conv(block.conv3)
            # Full shortcut BN: beta, gamma, mean, unk, var (5 tensors)
            _load_bn_shortcut(shortcut_bn)
            # bn3 remaining: gamma, mean, unk, var (4 tensors)
            with torch.no_grad():
                block.bn3.weight.copy_(_next())       # bn3 gamma
                block.bn3.running_mean.copy_(_next()) # bn3 mean
            _next()                                    # unknown - skip
            with torch.no_grad():
                block.bn3.running_var.copy_(_next())  # bn3 var
        else:
            # Non-shortcut: conv3 then full bn3 (5 tensors)
            _load_conv(block.conv3)
            _load_bn(block.bn3)




    # ── Stem (conv1 + bn1) ────────────────────────────────────────────
    # backbone[0] = conv1, backbone[1] = bn1 in PyTorch Sequential
    stem_conv = model.backbone[0]
    stem_bn   = model.backbone[1]
    _load_conv_bn_pair(stem_conv, stem_bn)
    logger.debug("Stem loaded, cursor=%d", cursor[0])

    # ── ResNet layers (backbone[4]→layer1, [5]→layer2, [6]→layer3, [7]→layer4) ──
    # PyTorch backbone Sequential: [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool,
    #   [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool
    layer_configs = [
        (model.backbone[4], [True, False, False]),         # layer1: 3 blocks
        (model.backbone[5], [True, False, False, False]),  # layer2: 4 blocks
        (model.backbone[6], [True, False, False, False,
                             False, False]),                # layer3: 6 blocks
        (model.backbone[7], [True, False, False]),         # layer4: 3 blocks
    ]

    for layer, shortcut_flags in layer_configs:
        for block, has_sc in zip(layer, shortcut_flags):
            _load_bottleneck(block, has_sc)
        logger.debug("Layer loaded, cursor=%d", cursor[0])

    if cursor[0] != len(keras_tensors):
        logger.warning(
            "Keras backbone cursor=%d but total tensors=%d — "
            "%d tensors were not consumed",
            cursor[0], len(keras_tensors), len(keras_tensors) - cursor[0],
        )
    else:
        logger.info("All %d Keras backbone tensors consumed correctly", cursor[0])

    # -----------------------------------------------------------------
    # Dense layers (Keras weight is (in, out) → transpose to (out, in))
    # -----------------------------------------------------------------
    with torch.no_grad():
        # dense: weight(2048,128) → Linear.weight(128,2048)
        w_dense = keras_dense[0].to(device).t().contiguous()
        b_dense = keras_dense[1].to(device)
        model.dense.weight.copy_(w_dense)
        model.dense.bias.copy_(b_dense)

        # dense_1: weight(128,2) → Linear.weight(2,128)
        w_d1 = keras_dense_1[0].to(device).t().contiguous()
        b_d1 = keras_dense_1[1].to(device)
        model.dense_1.weight.copy_(w_d1)
        model.dense_1.bias.copy_(b_d1)

    logger.info("Keras weights mapped → PyTorch model successfully")




def _shape_based_load(backbone: nn.Sequential,
                       keras_tensors: list,
                       device: str) -> None:
    """
    Fallback: match by shape when count doesn't align.
    Groups Keras tensors by shape and assigns them greedily.
    """
    from collections import defaultdict
    keras_by_shape: dict[tuple, list] = defaultdict(list)
    for kt in keras_tensors:
        keras_by_shape[tuple(kt.shape)].append(kt)

    usage: dict[tuple, int] = defaultdict(int)

    with torch.no_grad():
        for mod_name, mod in backbone.named_modules():
            if isinstance(mod, nn.Conv2d):
                k_shape_hwio = (mod.weight.shape[2], mod.weight.shape[3],
                                mod.weight.shape[1], mod.weight.shape[0])
                idx = usage[k_shape_hwio]
                if idx < len(keras_by_shape[k_shape_hwio]):
                    kt = keras_by_shape[k_shape_hwio][idx].to(device)
                    mod.weight.copy_(kt.permute(3, 2, 0, 1).contiguous())
                    usage[k_shape_hwio] += 1
            elif isinstance(mod, nn.BatchNorm2d):
                ch = (mod.weight.shape[0],)
                for attr in ("weight", "bias", "running_mean", "running_var"):
                    idx = usage[ch]
                    if idx < len(keras_by_shape[ch]):
                        kt = keras_by_shape[ch][idx].to(device)
                        getattr(mod, attr).copy_(kt)
                        usage[ch] += 1


class ResNet50Classifier:
    """
    Multi-label ResNet-50 classifier.

    The weights file 'FINAL_resnet50.pth' uses a Keras-style format:
      { 'resnet50': [318 tensors], 'dense': [W,b], 'dense_1': [W,b] }

    Architecture:
      ResNet50 backbone (no top) → GlobalAvgPool(2048)
      → Linear(2048→128) + ReLU
      → Linear(128→2)
      → Sigmoid (applied during inference only)

    Classes: [caries(0), periapical_lesion(1)]
    """

    def __init__(self):
        self._model  = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self._model is not None:
            return   # already loaded
        if not RESNET_WEIGHTS.exists():
            raise FileNotFoundError(
                f"ResNet-50 weights not found: {RESNET_WEIGHTS}\n"
                "Place 'FINAL_resnet50.pth' in the project root."
            )

        ckpt = torch.load(str(RESNET_WEIGHTS), map_location=self._device,
                          weights_only=False)

        # ── detect checkpoint format ──────────────────────────────────
        is_keras_style = (
            isinstance(ckpt, dict)
            and "resnet50" in ckpt
            and isinstance(ckpt["resnet50"], list)
        )

        if is_keras_style:
            logger.info("Detected Keras-style checkpoint — using custom loader")
            model = _DentOmniResNet()
            model.to(self._device)
            _load_keras_weights_into_resnet(model, ckpt, self._device)
        else:
            # Fallback: standard PyTorch state-dict format
            logger.info("Detected PyTorch state-dict checkpoint — using standard loader")
            model = tv_models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            model.to(self._device)

            sd = ckpt
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            elif isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)

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

        p_caries = float(probs[0].item())
        p_lesion = float(probs[1].item())

        logger.info("ResNet50 raw probs → caries=%.4f  lesion=%.4f",
                    p_caries, p_lesion)

        return {
            "caries": {
                "detected":   p_caries >= 0.5,
                "confidence": round(p_caries, 4),
            },
            "periapical_lesion": {
                "detected":   p_lesion >= 0.5,
                "confidence": round(p_lesion, 4),
            },
            "bounding_boxes":  [],
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
