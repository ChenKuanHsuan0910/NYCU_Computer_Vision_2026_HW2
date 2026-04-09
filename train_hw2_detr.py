"""
NYCU Visual Recognition 2026 Spring — HW2
Digit Detection with DETR + ResNet-50

Usage:
  # Training
  python train_hw2_detr.py --data_root /path/to/nycu-hw2-data --output_dir ./runs/exp1 --do_train --amp

  # Inference (generates pred.json)
  python train_hw2_detr.py --data_root /path/to/nycu-hw2-data --output_dir ./runs/exp1 \
      --checkpoint ./runs/exp1/checkpoints/best.pt --do_infer

  # Validate pred.json format only
  python train_hw2_detr.py --validate_pred /path/to/pred.json

Smoke-test parameters (quick sanity check, ~5-10 min on GPU):
  --epochs 2 --batch_size 4 --grad_accum_steps 1 --eval_every 1 --amp

Full training parameters (leaderboard target):
  --epochs 50 --batch_size 4 --grad_accum_steps 2 --eval_every 3 --amp
"""

import os
import json
import math
import time
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    get_cosine_schedule_with_warmup,
)
from torchvision.ops import nms as torchvision_nms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DigitCocoDataset(Dataset):
    """
    COCO-format digit detection dataset.
    Bboxes in JSON are [x_min, y_min, w, h].
    category_id: 1-10 (mapping digit '0'-'9').
    """

    def __init__(self, images_dir, annotation_data, processor, transforms=None):
        self.images_dir = images_dir
        self.processor = processor
        self.transforms = transforms

        self.images = annotation_data["images"]
        self.annotations = annotation_data["annotations"]

        self.image_id_to_info = {img["id"]: img for img in self.images}
        self.image_id_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.image_id_to_anns[ann["image_id"]].append(ann)

        self.image_ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.image_ids)

    def _sanitize_bboxes(self, anns, img_w, img_h):
        """Clip bboxes to image bounds and remove degenerate boxes."""
        cleaned_boxes = []
        cleaned_labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]

            x, y, w, h = float(x), float(y), float(w), float(h)
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(img_w), x + w)
            y2 = min(float(img_h), y + h)
            new_w = x2 - x1
            new_h = y2 - y1

            if new_w <= 1e-6 or new_h <= 1e-6:
                continue

            cleaned_boxes.append([x1, y1, new_w, new_h])   # keep [x,y,w,h]
            # CRITICAL: Remap category_id 1-10 → 0-9 for DETR.
            # DETR no-object class = index num_labels = 10.
            # Using category_id=10 directly causes digit "9" to collide with
            # the no-object class → digit "9" is NEVER detected at inference!
            cleaned_labels.append(int(cat_id) - 1)

        return cleaned_boxes, cleaned_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        anns = self.image_id_to_anns[image_id]
        boxes, category_ids = self._sanitize_bboxes(anns, img_w, img_h)

        # Albumentations augmentation (train only)
        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                category_ids=category_ids,
            )
            image = transformed["image"]
            boxes = list(transformed["bboxes"])
            category_ids = list(transformed["category_ids"])

        # Final bbox sanity pass after augmentation
        final_annotations = []
        for box, cat_id in zip(boxes, category_ids):
            x, y, w, h = box
            x = max(0.0, float(x))
            y = max(0.0, float(y))
            w = float(w)
            h = float(h)

            if x + w > img_w:
                w = float(img_w) - x
            if y + h > img_h:
                h = float(img_h) - y
            if w <= 1e-6 or h <= 1e-6:
                continue

            final_annotations.append({
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })

        target = {"image_id": image_id, "annotations": final_annotations}

        # DetrImageProcessor expects PIL image or numpy HWC RGB
        encoded = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"][0]
        return pixel_values, labels


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_transforms():
    """
    Train augmentation for digit detection.
    - NO horizontal flip (flipping digits changes their identity, e.g. 6 <-> 9)
    - Scale jitter (0.75-1.25x) is the single biggest gain for detection
    - Color augmentation for lighting/camera robustness
    - HueSaturationValue + RandomGamma for diverse appearance
    """
    train_transform = A.Compose(
        [
            # Noise / blur  (simulate different camera / compression quality)
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.35),

            # Brightness / contrast / gamma
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=0.5),

            # Color / saturation  (digits appear in many contexts: signs, plates, screens)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.45,
            ),

            # Geometric: scale jitter (0.75-1.25x) + rotation + translation
            # Larger scale range than before (was 0.90-1.10) → better scale robustness
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.75, 1.25),
                rotate=(-8, 8),
                fill=0,
                p=0.6,
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.3,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )
    return train_transform, None   # val transform = None (processor handles resize)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch, processor):
    pixel_values = [item[0] for item in batch]   # list of [C, H, W] tensors
    labels = [item[1] for item in batch]

    # Manual padding: find max spatial dims in this batch
    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)

    padded_images = []
    pixel_masks = []
    for img in pixel_values:
        c, h, w = img.shape
        padded = torch.zeros(c, max_h, max_w, dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_images.append(padded)

        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def build_optimizer(model, lr, lr_backbone, weight_decay):
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler,
    device, grad_accum_steps, grad_clip, amp, epoch_idx
):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch_idx + 1}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps

        scaler.scale(loss).backward()

        do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(dataloader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        avg_loss = total_loss / (step + 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            lr=f"{optimizer.param_groups[1]['lr']:.2e}",
        )

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------
def validate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid Loss"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# COCO mAP evaluation on validation set
# ---------------------------------------------------------------------------
def evaluate_coco_map(model, dataloader, processor, coco_gt, output_dir, device, threshold):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="COCO Eval"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = batch["labels"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # orig_size from DetrImageProcessor labels: (height, width)
            target_sizes = torch.stack([t["orig_size"] for t in labels]).to(device)
            results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )

            for result, label in zip(results, labels):
                image_id = int(label["image_id"].item())
                boxes = result["boxes"].cpu().numpy()    # xyxy
                scores = result["scores"].cpu().numpy()
                pred_labels = result["labels"].cpu().numpy()

                for box, score, pred_label in zip(boxes, scores, pred_labels):
                    x1, y1, x2, y2 = box.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(pred_label) + 1,  # model 0-9 → COCO GT 1-10
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    })

    if len(predictions) == 0:
        print("[WARN] No predictions produced during eval — returning mAP=0.0")
        empty = {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
                 "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}
        return 0.0, empty

    pred_json_tmp = os.path.join(output_dir, "valid_preds_tmp.json")
    with open(pred_json_tmp, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_json_tmp)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "mAP": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
    }
    return stats["mAP"], stats


# ---------------------------------------------------------------------------
# Per-class NMS helper
# ---------------------------------------------------------------------------
def per_class_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float
):
    """Apply NMS independently per class. Returns (boxes, scores, labels) after NMS."""
    if len(scores) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    keep_b, keep_s, keep_l = [], [], []
    for cls in np.unique(labels):
        m = labels == cls
        b, s = boxes[m], scores[m]
        if len(b) == 1:
            keep_b.append(b)
            keep_s.append(s)
            keep_l.append(np.array([cls], dtype=np.int64))
            continue
        idx = torchvision_nms(
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
            iou_threshold=iou_threshold,
        ).numpy()
        keep_b.append(b[idx])
        keep_s.append(s[idx])
        keep_l.append(np.full(len(idx), cls, dtype=np.int64))
    return (
        np.concatenate(keep_b, axis=0),
        np.concatenate(keep_s, axis=0),
        np.concatenate(keep_l, axis=0),
    )


# ---------------------------------------------------------------------------
# Test inference  →  pred.json
# ---------------------------------------------------------------------------
def inference_test(model, processor, test_dir, device, threshold, max_dets_per_image,
                   pred_json_path, nms_iou=0.5, tta_processors=None):
    """
    Run inference on test images and save pred.json.
    Each prediction: {"image_id": int, "bbox": [x,y,w,h], "score": float, "category_id": int}
    image_id    = integer parsed from filename (e.g. '1.png' -> 1)
    bbox        = [x_min, y_min, w, h]   (COCO format)
    score       = float confidence in [0,1]
    category_id = 1-10  (model outputs 0-indexed 0-9; +1 applied here)
    tta_processors: optional list of extra DetrImageProcessor for multi-scale TTA
    """
    model.eval()
    all_procs = [processor]
    if tta_processors:
        all_procs.extend(tta_processors)

    # Sort by numeric value of stem
    test_files = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    all_predictions = []

    with torch.no_grad():
        for file_name in tqdm(test_files, desc="Test Inference"):
            image_id = int(os.path.splitext(file_name)[0])
            img_path = os.path.join(test_dir, file_name)

            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            orig_h, orig_w = image_np.shape[:2]

            tta_b, tta_s, tta_l = [], [], []
            for proc in all_procs:
                inputs = proc(images=image_np, return_tensors="pt").to(device)
                outputs = model(**inputs)
                target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
                results = proc.post_process_object_detection(
                    outputs=outputs,
                    threshold=threshold,
                    target_sizes=target_sizes,
                )[0]
                b = results["boxes"].cpu().numpy()    # xyxy
                s = results["scores"].cpu().numpy()
                l = results["labels"].cpu().numpy()
                if len(s) > 0:
                    tta_b.append(b)
                    tta_s.append(s)
                    tta_l.append(l)

            if not tta_b:
                continue  # no predictions for this image at any scale

            boxes  = np.concatenate(tta_b, axis=0)
            scores = np.concatenate(tta_s, axis=0)
            labels = np.concatenate(tta_l, axis=0)

            # Per-class NMS: remove overlapping predictions within each digit class
            if nms_iou > 0 and len(scores) > 1:
                boxes, scores, labels = per_class_nms(boxes, scores, labels, nms_iou)

            # Keep top-k by confidence
            if len(scores) > max_dets_per_image:
                keep = np.argsort(scores)[::-1][:max_dets_per_image]
                boxes  = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                all_predictions.append({
                    "image_id": int(image_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "category_id": int(label) + 1,  # model 0-indexed → COCO 1-10
                })

    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f)

    print(f"[INFO] Saved test predictions -> {pred_json_path}")
    print(f"[INFO] Total predictions: {len(all_predictions)}")
    return all_predictions


# ---------------------------------------------------------------------------
# pred.json format validator
# ---------------------------------------------------------------------------
def validate_pred_json(pred_json_path):
    """
    Validate pred.json format against competition requirements.
    Prints a summary and returns True if all checks pass.
    """
    print(f"\n=== Validating {pred_json_path} ===")

    if not os.path.exists(pred_json_path):
        print("[ERROR] File not found!")
        return False

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if not isinstance(preds, list):
        print("[ERROR] pred.json must be a JSON list (array of dicts)")
        return False

    print(f"  Total predictions: {len(preds)}")
    errors = []
    required_keys = {"image_id", "bbox", "score", "category_id"}

    for i, p in enumerate(preds):
        if not isinstance(p, dict):
            errors.append(f"  [{i}] Not a dict")
            continue
        missing = required_keys - set(p.keys())
        if missing:
            errors.append(f"  [{i}] Missing keys: {missing}")
            continue

        # image_id: must be int
        if not isinstance(p["image_id"], int):
            errors.append(f"  [{i}] image_id must be int, got {type(p['image_id'])}")

        # bbox: must be list of 4 floats/ints, [x,y,w,h] with w,h > 0
        bbox = p["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            errors.append(f"  [{i}] bbox must be list of 4 elements, got: {bbox}")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                errors.append(f"  [{i}] bbox w/h must be > 0, got w={w} h={h}")
            for v in bbox:
                if not isinstance(v, (int, float)):
                    errors.append(f"  [{i}] bbox elements must be numeric, got {type(v)}")
                    break

        # score: must be float in [0, 1]
        if not isinstance(p["score"], (int, float)):
            errors.append(f"  [{i}] score must be numeric, got {type(p['score'])}")
        elif not (0.0 <= float(p["score"]) <= 1.0):
            errors.append(f"  [{i}] score out of [0,1]: {p['score']}")

        # category_id: must be int in 1-10
        cat = p["category_id"]
        if not isinstance(cat, int):
            errors.append(f"  [{i}] category_id must be int, got {type(cat)}")
        elif not (1 <= cat <= 10):
            errors.append(f"  [{i}] category_id must be 1-10, got {cat}")

        if len(errors) > 20:
            errors.append("  ... (too many errors, stopping early)")
            break

    if errors:
        print("[FAIL] Format errors found:")
        for e in errors:
            print(e)
        return False
    else:
        # Summary stats
        image_ids = set(p["image_id"] for p in preds)
        cats = set(p["category_id"] for p in preds)
        scores = [p["score"] for p in preds]
        print(f"  Unique image_ids: {len(image_ids)}")
        print(f"  Category_ids present: {sorted(cats)}")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print("[PASS] pred.json format is valid!")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DETR ResNet-50 Digit Detection HW2")
    # ---- paths ----
    parser.add_argument("--data_root", type=str, default="",
                        help="Path to nycu-hw2-data/ (contains train/, valid/, test/, train.json, valid.json)")
    parser.add_argument("--output_dir", type=str, default="./runs/exp1",
                        help="Directory to save checkpoints, logs, pred.json")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint .pt file to load (for inference or resume)")
    parser.add_argument("--validate_pred", type=str, default="",
                        help="If set, only validate this pred.json file and exit (no --data_root needed)")

    # ---- actions ----
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_infer", action="store_true", help="Run test inference -> pred.json")

    # ---- training hyper-parameters ----
    # Smoke test:  --epochs 2 --batch_size 4 --grad_accum_steps 1 --eval_every 1
    # Full train:  --epochs 50 --batch_size 4 --grad_accum_steps 2 --eval_every 3 --amp
    parser.add_argument("--epochs", type=int, default=50,
                        help="[Full:50] [Smoke:2]")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-GPU batch size (4 works on H200 with shortest_edge=800)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Transformer head LR (DETR paper: 1e-4)")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="ResNet-50 backbone LR (DETR paper: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.1,
                        help="Gradient clipping max-norm (DETR paper: 0.1)")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation; effective_batch = batch_size * accum")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--amp", action="store_true",
                        help="Automatic mixed precision (recommended on H200)")
    parser.add_argument("--seed", type=int, default=42)

    # ---- image size ----
    parser.add_argument("--shortest_edge", type=int, default=800)
    parser.add_argument("--longest_edge", type=int, default=1333)

    # ---- inference / eval ----
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="[Full:0.3] [Smoke:0.05]  Lower = more recall during eval/infer")
    parser.add_argument("--max_dets_per_image", type=int, default=20,
                        help="Max detections per test image (digit images rarely have >10 digits)")
    parser.add_argument("--nms_iou", type=float, default=0.5,
                        help="NMS IoU threshold applied at inference to remove duplicates (0=disabled)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable multi-scale test-time augmentation (TTA) at inference")
    parser.add_argument("--tta_scales", type=str, default="0.8,1.0,1.2",
                        help="Comma-separated TTA scale factors applied to shortest/longest_edge")
    parser.add_argument("--combine_trainval", action="store_true",
                        help="Add validation images to training set (11%% more data)")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Run COCO mAP eval every N epochs [Full:3] [Smoke:1]")

    args = parser.parse_args()

    # ---- validate-only mode -----------------------------------------------
    if args.validate_pred:
        ok = validate_pred_json(args.validate_pred)
        exit(0 if ok else 1)

    if not args.data_root:
        parser.error("--data_root is required unless using --validate_pred")

    # ---- Setup ---------------------------------------------------------------
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dir  = os.path.join(args.data_root, "train")
    valid_dir  = os.path.join(args.data_root, "valid")
    test_dir   = os.path.join(args.data_root, "test")
    train_json = os.path.join(args.data_root, "train.json")
    valid_json = os.path.join(args.data_root, "valid.json")

    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(valid_json, "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    # CRITICAL FIX: id2label keys must be 0-indexed model output indices ("0"-"9").
    # DETR no-object class = index num_labels = 10.
    # Old code used category_ids 1-10 as keys → category_id=10 (digit '9') collided
    # with no-object class index 10 → digit '9' was NEVER detected!
    # Fix: sort categories by id, then enumerate 0-9 as the model output indices.
    sorted_cats = sorted(train_data["categories"], key=lambda x: x["id"])
    id2label = {str(i): cat["name"] for i, cat in enumerate(sorted_cats)}
    label2id = {cat["name"]: str(i) for i, cat in enumerate(sorted_cats)}
    num_labels = len(id2label)   # 10
    print(f"[INFO] num_labels={num_labels}  id2label={id2label}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # FIX: Pass size at construction so resize is applied consistently for both
    # training and inference. Assigning processor.size after the fact is
    # unreliable in newer versions of transformers.
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
    )

    # ---- Datasets & Loaders --------------------------------------------------
    train_transform, _ = build_transforms()

    train_dataset = DigitCocoDataset(train_dir, train_data, processor, transforms=train_transform)
    valid_dataset = DigitCocoDataset(valid_dir, valid_data, processor, transforms=None)

    # Optionally add valid images to training (11% more data; val still used for monitoring)
    if args.combine_trainval:
        from torch.utils.data import ConcatDataset
        valid_as_train = DigitCocoDataset(valid_dir, valid_data, processor,
                                          transforms=train_transform)
        train_dataset = ConcatDataset([train_dataset, valid_as_train])
        print(f"[INFO] combine_trainval: {len(train_dataset)} training images total")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, processor),
        drop_last=True,   # avoids broken final batch with grad accumulation
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, processor),
    )

    # ---- Model ---------------------------------------------------------------
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # replace detection head for 10 classes
    )
    model.to(device)

    # ---- Load checkpoint (if provided) ---------------------------------------
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # ==========================================================================
    # TRAINING
    # ==========================================================================
    if args.do_train:
        optimizer = build_optimizer(model, args.lr, args.lr_backbone, args.weight_decay)

        # Total optimiser steps (after accumulation)
        steps_per_epoch    = math.ceil(len(train_loader) / args.grad_accum_steps)
        num_training_steps = steps_per_epoch * args.epochs
        num_warmup_steps   = max(1, int(num_training_steps * args.warmup_ratio))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # FIX: GradScaler created ONCE outside epoch loop (was re-created every epoch)
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

        coco_gt  = COCO(valid_json)
        best_map = -1.0
        history  = []

        print(f"\n[INFO] Training: {args.epochs} epochs, "
              f"batch={args.batch_size}x accum={args.grad_accum_steps}, "
              f"effective_batch={args.batch_size * args.grad_accum_steps}, "
              f"lr={args.lr}, lr_backbone={args.lr_backbone}")

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                device, args.grad_accum_steps, args.grad_clip, args.amp, epoch,
            )
            valid_loss = validate_loss(model, valid_loader, device)
            elapsed = time.time() - t0

            # COCO mAP every eval_every epochs; always on first and last epoch
            do_eval = (
                ((epoch + 1) % args.eval_every == 0)
                or (epoch == 0)
                or (epoch == args.epochs - 1)
            )
            if do_eval:
                valid_map, valid_stats = evaluate_coco_map(
                    model, valid_loader, processor, coco_gt,
                    args.output_dir, device, args.conf_threshold,
                )
            else:
                valid_map = -1.0
                valid_stats = {k: -1.0 for k in
                               ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"]}

            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "valid_loss": round(valid_loss, 6),
                "elapsed_s": round(elapsed, 1),
                **valid_stats,
            }
            history.append(record)
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(json.dumps(record, indent=2))

            # Always save latest checkpoint
            latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_map": best_map,
                "history": history,
            }, latest_ckpt)

            # Save best if mAP improved
            if do_eval and valid_map > best_map:
                best_map = valid_map
                best_ckpt = os.path.join(ckpt_dir, "best.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_map": best_map,
                    "history": history,
                }, best_ckpt)
                print(f"[INFO] New best model  mAP={best_map:.4f}  -> {best_ckpt}")

            with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        print(f"\n[INFO] Training finished. Best mAP = {best_map:.4f}")

    # ==========================================================================
    # INFERENCE  ->  pred.json
    # ==========================================================================
    if args.do_infer:
        pred_json_path = os.path.join(args.output_dir, "pred.json")
        # Build TTA processors for multi-scale inference
        tta_procs = None
        if args.tta:
            tta_procs = []
            for scale_str in args.tta_scales.split(","):
                scale = float(scale_str.strip())
                if abs(scale - 1.0) < 0.01:
                    continue  # skip 1.0 — that's the main processor
                s_short = int(args.shortest_edge * scale)
                s_long  = int(args.longest_edge  * scale)
                tta_proc = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50",
                    size={"shortest_edge": s_short, "longest_edge": s_long},
                )
                tta_procs.append(tta_proc)
                print(f"[INFO] TTA scale={scale:.1f}: shortest_edge={s_short} longest_edge={s_long}")
        inference_test(
            model=model,
            processor=processor,
            test_dir=test_dir,
            device=device,
            threshold=args.conf_threshold,
            max_dets_per_image=args.max_dets_per_image,
            pred_json_path=pred_json_path,
            nms_iou=args.nms_iou,
            tta_processors=tta_procs,
        )
        # Automatically validate the output format
        validate_pred_json(pred_json_path)


if __name__ == "__main__":
    main()
