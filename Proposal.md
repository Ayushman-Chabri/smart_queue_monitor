# Technical Proposal: Smart Queue & Retail Dwell-Time Monitor

**Prepared by:** Ayushman Chabri, Priya Agarwala, and Nikita Kumari  
**Target Environment:** Edge / On-Premise CPU Deployment (Docker Compose)  
**Model Artifact:** Static `YOLO11n` ONNX (`opset=17`, `10.2 MB`)

---

## 1. Executive Summary & Problem Statement

Retail checkout bottlenecks directly impact customer satisfaction and store conversion rates. However, deploying heavy GPU-accelerated video analytics across multiple store branches is cost-prohibitive, while naive frame-by-frame Python scripts fail in production due to unbounded memory growth, flickering detections across occlusions, and lack of operational visibility.

The **Smart Queue & Retail Dwell-Time Monitor** solves this by delivering an end-to-end, containerized computer vision microservice capable of real-time queue counting and per-customer wait-time measurement on standard CPU hardware.

---

## 2. System Architecture & Engineering Trade-Offs

```text
[Camera / Stream Simulator] ──(XADD MAXLEN ~5)──▶ [Redis Stream]
                                                        │
                                                        ▼
[Grafana Dashboard] ◀──(Scrape 5s)── [Prometheus] ◀── [FastAPI + ONNX Runtime + ByteTrack]
                                                        │
                                                        └──(0.30 <= conf <= 0.50)──▶ [hard_examples/]
```

### Key Architectural Decisions
1. **Static ONNX Runtime over PyTorch (`models/yolo11n.onnx`):**
   * **Trade-off:** Eliminates the multi-gigabyte PyTorch/CUDA runtime footprint in favor of a `10.2 MB` static graph (`opset=17`).
   * **Optimization:** Configured with `ORT_ENABLE_ALL` graph optimizations and `ORT_SEQUENTIAL` execution pinned to physical CPU cores (`os.cpu_count() // 2`), avoiding thread-contention latency spikes.
2. **Bounded Redis Stream Ingestion (`MAXLEN ~ 5`):**
   * **Trade-off:** Prioritizes real-time temporal freshness over processing every historical frame during CPU spikes.
   * **Reliability:** Decouples camera frame ingestion from inference compute; if stream ingestion exceeds CPU throughput, Redis sheds stale frames automatically so congestion alerts reflect current store conditions.
3. **ByteTrack + Resolution-Normalized Spatial Zones:**
   * **Persistence:** Associates detections across frames so brief occlusions (e.g., stanchion barriers or passing shoppers) do not reset a customer's dwell-time timer.
   * **Portability:** Defines checkout regions as normalized `[0.0, 1.0]` polygons that dynamically scale to any camera resolution (`640×480` up to `1080p+`) without manual coordinate recalibration.
4. **Automated Active-Learning Data Flywheel:**
   * Borderline detections (`0.30 <= conf <= 0.50`) are rate-limited (1 frame / 5 seconds) and written to a persistent `hard_examples/` volume, continuously curating edge cases for `v2.0` domain fine-tuning.

---

## 3. Empirical Benchmarks & Calibrated SLAs

Evaluated end-to-end on CPU across a **1,260-image labeled retail checkout dataset** (`12,855` annotated ground-truth boxes) using `scripts/evaluate_dataset.py`:

| Metric | Target SLA | Measured Result (1,260-Image Benchmark) |
| :--- | :--- | :--- |
| **Model Artifact Size** | `< 25.0 MB` | **10.2 MB** (`yolo11n.onnx`) |
| **Mean Inference Latency** | `< 65.0 ms` | **40.45 ms** (~24.7 FPS on CPU) |
| **p95 Inference Latency** | `< 85.0 ms` (CI Gate) | **44.96 ms** |
| **Zero-Shot Precision (@IoU 0.50)** | `> 80.0%` | **87.98%** (`5,531 TP / 756 FP`) |
| **Zero-Shot Recall (@IoU 0.50)** | Baseline Diagnostic | **43.03%** (`F1: 0.5779`, `Avg Conf: 0.743`) |

### Engineering Error Analysis & Active Learning Justification
* **Latency Headroom:** At **44.96 ms p95 latency**, the system maintains nearly 2x headroom under its 85 ms CI/CD SLA gate on CPU.
* **High Precision (`87.98%`) vs. Zero-Shot Recall (`43.03%`):** High precision ensures minimal false congestion alerts. Recall diagnostics across the 12,855 ground-truth labels highlight two clear factors: (1) non-person class labels in the multi-class dataset (`kiosks`/`barriers` intentionally excluded by our COCO class `0` `person`-only filter) and (2) severe camera-tilt augmentations and lower-body stanchion occlusions—empirically justifying our automated `hard_examples/` capture hook for `v2.0` fine-tuning.

---

## 4. Production Observability & CI/CD Governance

* **Prometheus & Grafana Telemetry:** Real-time metrics (`queue_active_customers`, `queue_max_dwell_seconds`, `queue_inference_latency_ms`, and `queue_avg_detection_confidence`) are scraped every 5 seconds and visualized in Grafana with threshold alerts for queue length (`> 4` customers) and dwell duration (`> 60.0s`).
* **Automated CI/CD Quality Gates (`.github/workflows/ml-ci-cd.yml`):** Every commit triggers automated `ruff` static linting, `pytest` FastAPI endpoint verification, and an automated latency SLA gate that blocks merges if p95 inference latency exceeds `85.0 ms`.

---

## 5. Roadmap (`v2.0`)

1. **Domain Fine-Tuning:** Retrain `YOLO11n` on the 1,260-image checkout dataset plus frames harvested from `hard_examples/` to raise recall under heavy stanchion occlusion and oblique camera angles.
2. **INT8 Post-Training Quantization:** Benchmark static INT8 ONNX quantization to target sub-25 ms CPU latency for multi-camera edge deployments.
3. **Webhook Alerting:** Integrate Grafana Alertmanager webhooks (Slack/Teams/POS notifications) when sustained queue congestion exceeds 60 seconds.