import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

try:
    from trackers import ByteTrackTracker as ByteTrack
except ImportError:
    ByteTrack = sv.ByteTrack


@dataclass
class QueueMetrics:
    frame_id: int
    inference_ms: float
    queue_count: int
    max_dwell_sec: float
    avg_dwell_sec: float
    avg_confidence: float
    alert_triggered: bool
    alert_reasons: List[str] = field(default_factory=list)
    active_tracks: Dict[int, float] = field(default_factory=dict)  # {tracker_id: dwell_seconds}


def create_ort_session(model_path: str) -> ort.InferenceSession:
    """Configure ONNX Runtime for deterministic physical-core execution."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True

    # Avoid hyperthreading oversubscription by targeting physical cores
    physical_cores = max(1, (os.cpu_count() or 2) // 2)
    sess_options.intra_op_num_threads = physical_cores
    sess_options.inter_op_num_threads = 1

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)


class SmartQueueDetector:
    """
    Production ONNX Runtime detector + ByteTrack tracker + Polygon Dwell-Time analyzer.
    Filters strictly for COCO class 0 ('person') and logs low-confidence edge cases.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "models/yolo11n.onnx",
        zone_polygon: Optional[np.ndarray] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        max_queue_threshold: int = 4,
        max_dwell_threshold_sec: float = 60.0,
        hard_example_dir: str = "hard_examples",
        fps: float = 30.0,
    ):
        self.input_size = 640
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_queue_threshold = max_queue_threshold
        self.max_dwell_threshold_sec = max_dwell_threshold_sec
        self.hard_example_dir = hard_example_dir
        self.last_hard_example_ts = 0.0

        os.makedirs(self.hard_example_dir, exist_ok=True)

        self.session = create_ort_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Normalized checkout queue polygon [x_ratio, y_ratio] covering the kiosk/queue lane
        if zone_polygon is None:
            self.norm_polygon = np.array(
                [[0.05, 0.45], [0.72, 0.45], [0.72, 0.98], [0.05, 0.98]],
                dtype=np.float32,
            )
            self._is_normalized = True
        else:
            self.norm_polygon = zone_polygon
            self._is_normalized = float(np.max(zone_polygon)) <= 1.0

        self.zone: Optional[sv.PolygonZone] = None
        self.zone_annotator: Optional[sv.PolygonZoneAnnotator] = None
        self._current_frame_shape: Optional[Tuple[int, int]] = None

        self.tracker = ByteTrack()

        # Track entry timestamps: {tracker_id: first_seen_in_zone_timestamp}
        self.zone_entry_times: Dict[int, float] = {}
        self.frame_counter = 0

        # Annotators for visual debugging stream
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize to 640x640, normalize to [0, 1], and convert HWC -> NCHW."""
        orig_h, orig_w = frame.shape[:2]
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 640, 640)
        return blob, scale, (pad_x, pad_y)

    def _postprocess(
        self, raw_output: np.ndarray, scale: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]
    ) -> sv.Detections:
        """Decode YOLO11 (1, 84, 8400) output, filter 'person' class, apply NMS, and rescale boxes."""
        preds = np.squeeze(raw_output).T  # (8400, 84)
        person_scores = preds[:, 4 + self.PERSON_CLASS_ID]

        mask = person_scores >= self.conf_threshold
        preds = preds[mask]
        scores = person_scores[mask]

        if len(preds) == 0:
            return sv.Detections.empty()

        # Convert cx, cy, w, h -> x1, y1, x2, y2 in original image coordinates
        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

        x1 = np.clip(((cx - w / 2) - pad_x) / scale, 0, orig_w)
        y1 = np.clip(((cy - h / 2) - pad_y) / scale, 0, orig_h)
        x2 = np.clip(((cx + w / 2) - pad_x) / scale, 0, orig_w)
        y2 = np.clip(((cy + h / 2) - pad_y) / scale, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # OpenCV NMSBoxes expects [x, y, width, height]
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.conf_threshold, self.iou_threshold)

        if len(indices) == 0:
            return sv.Detections.empty()

        indices = np.array(indices).flatten()
        return sv.Detections(
            xyxy=boxes[indices],
            confidence=scores[indices],
            class_id=np.zeros(len(indices), dtype=int),
        )

    def _active_learning_hook(self, raw_frame: np.ndarray, detections: sv.Detections) -> None:
        """Save frames with uncertain detections (0.30 <= conf <= 0.50), rate-limited to 1 frame / 5s."""
        if len(detections) == 0 or detections.confidence is None:
            return

        uncertain_mask = (detections.confidence >= 0.30) & (detections.confidence <= 0.50)
        now = time.time()
        if np.any(uncertain_mask) and (now - self.last_hard_example_ts) > 5.0:
            self.last_hard_example_ts = now
            filename = os.path.join(self.hard_example_dir, f"uncertain_{int(now)}_{self.frame_counter}.jpg")
            cv2.imwrite(filename, raw_frame)

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> Tuple[QueueMetrics, Optional[np.ndarray]]:
        """Run end-to-end detection, tracking, dwell-time computation, and optional frame annotation."""
        self.frame_counter += 1
        now = time.time()

        # 1. Inference
        t0 = time.perf_counter()
        blob, scale, pad = self._preprocess(frame)
        raw_out = self.session.run(None, {self.input_name: blob})[0]
        detections = self._postprocess(raw_out, scale, pad, frame.shape[:2])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Active learning capture
        self._active_learning_hook(frame, detections)

        # 3. ByteTrack multi-object tracking (compatible with both `trackers` and `supervision`)
        if hasattr(self.tracker, "update_with_detections"):
            tracked_detections = self.tracker.update_with_detections(detections)
        else:
            tracked_detections = self.tracker.update(detections)

        # 4. Spatial zone check & dwell-time state update (auto-scaled to frame resolution)
        h, w = frame.shape[:2]
        if self.zone is None or self._current_frame_shape != (h, w):
            if self._is_normalized:
                scaled_poly = (self.norm_polygon * np.array([w, h])).astype(np.int32)
            else:
                scaled_poly = self.norm_polygon.astype(np.int32)
            self.zone = sv.PolygonZone(polygon=scaled_poly)
            self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=sv.Color.YELLOW, thickness=2)
            self._current_frame_shape = (h, w)

        in_zone_mask = self.zone.trigger(detections=tracked_detections)
        active_zone_ids = set()
        active_dwells: Dict[int, float] = {}

        if tracked_detections.tracker_id is not None:
            for tracker_id, is_inside in zip(tracked_detections.tracker_id, in_zone_mask):
                tid = int(tracker_id)
                if is_inside:
                    active_zone_ids.add(tid)
                    if tid not in self.zone_entry_times:
                        self.zone_entry_times[tid] = now
                    active_dwells[tid] = round(now - self.zone_entry_times[tid], 1)

        # Clean up IDs that have exited the checkout zone
        for tid in list(self.zone_entry_times.keys()):
            if tid not in active_zone_ids:
                del self.zone_entry_times[tid]

        # 5. Evaluate business alert rules
        queue_count = len(active_dwells)
        max_dwell = max(active_dwells.values(), default=0.0)
        avg_dwell = round(float(np.mean(list(active_dwells.values()))), 1) if active_dwells else 0.0
        avg_conf = (
            round(float(np.mean(tracked_detections.confidence)), 3)
            if tracked_detections.confidence is not None and len(tracked_detections) > 0
            else 0.0
        )

        alert_reasons = []
        if queue_count > self.max_queue_threshold:
            alert_reasons.append(f"Queue length {queue_count} > {self.max_queue_threshold}")
        if max_dwell > self.max_dwell_threshold_sec:
            alert_reasons.append(f"Max wait {max_dwell}s > {self.max_dwell_threshold_sec}s")

        metrics = QueueMetrics(
            frame_id=self.frame_counter,
            inference_ms=round(inference_ms, 2),
            queue_count=queue_count,
            max_dwell_sec=max_dwell,
            avg_dwell_sec=avg_dwell,
            avg_confidence=avg_conf,
            alert_triggered=len(alert_reasons) > 0,
            alert_reasons=alert_reasons,
            active_tracks=active_dwells,
        )

        # 6. Optional visual annotation for UI / demo stream
        annotated_frame = None
        if annotate and self.zone_annotator is not None:
            annotated_frame = self.zone_annotator.annotate(scene=frame.copy())
            if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
                labels = [
                    f"#{int(tid)} | {active_dwells[int(tid)]}s" if int(tid) in active_dwells else f"#{int(tid)} (Passerby)"
                    for tid in tracked_detections.tracker_id
                ]
                annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, detections=tracked_detections, labels=labels
                )

        return metrics, annotated_frame