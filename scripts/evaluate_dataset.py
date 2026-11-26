import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.inference import SmartQueueDetector


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalized YOLO (class cx cy w h) labels to pixel [x1, y1, x2, y2]."""
    if not os.path.exists(label_path):
        return np.empty((0, 4), dtype=np.float32)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two (N, 4) and (M, 4) xyxy arrays."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


def evaluate(images_dir: str, labels_dir: str, iou_thresh: float = 0.50, max_images: int = 0) -> None:
    detector = SmartQueueDetector(model_path="models/yolo11n.onnx", conf_threshold=0.30)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    tp_total, fp_total, fn_total = 0, 0, 0
    latencies = []
    confidences = []

    print(f"Evaluating {len(image_paths)} images from {images_dir}...")
    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        gt_boxes = load_yolo_labels(label_path, w, h)

        # Run raw ONNX detection + postprocessing
        t0 = time.perf_counter()
        blob, scale, pad = detector._preprocess(frame)
        raw_out = detector.session.run(None, {detector.input_name: blob})[0]
        preds = detector._postprocess(raw_out, scale, pad, (h, w))
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_boxes = preds.xyxy
        if preds.confidence is not None and len(preds.confidence) > 0:
            confidences.extend(preds.confidence.tolist())

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            fn_total += len(gt_boxes)
            continue
        elif len(gt_boxes) == 0:
            fp_total += len(pred_boxes)
            continue

        # Greedy IoU matching
        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()
        tp = 0
        for p_idx in range(len(pred_boxes)):
            best_gt = int(np.argmax(iou_mat[p_idx]))
            if iou_mat[p_idx, best_gt] >= iou_thresh and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                tp += 1

        tp_total += tp
        fp_total += len(pred_boxes) - tp
        fn_total += len(gt_boxes) - len(matched_gt)

        if idx % 200 == 0 or idx == len(image_paths):
            print(f"  Processed {idx}/{len(image_paths)} images...")

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n" + "=" * 55)
    print("OFFLINE DATASET EVALUATION REPORT (YOLO11n ONNX)")
    print("=" * 55)
    print(f"Images Evaluated   : {len(image_paths)}")
    print(f"Ground Truth Boxes : {tp_total + fn_total}")
    print(f"True Positives     : {tp_total} | False Positives: {fp_total} | False Negatives: {fn_total}")
    print(f"Precision @0.50    : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall @0.50       : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score @0.50     : {f1:.4f}")
    print(f"Avg Confidence     : {np.mean(confidences):.3f}" if confidences else "Avg Confidence     : N/A")
    print(f"Mean Latency       : {np.mean(latencies):.2f} ms")
    print(f"p95 Latency        : {np.percentile(latencies, 95):.2f} ms")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ONNX detector against YOLO ground-truth labels.")
    parser.add_argument("--images", required=True, help="Path to images directory (e.g., dataset/valid/images)")
    parser.add_argument("--labels", required=True, help="Path to labels directory (e.g., dataset/valid/labels)")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU matching threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images (0 = all)")
    args = parser.parse_args()

    evaluate(args.images, args.labels, iou_thresh=args.iou, max_images=args.max_images)