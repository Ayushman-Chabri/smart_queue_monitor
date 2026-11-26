# Smart Queue & Retail Dwell-Time Monitor

**An AI-Powered Retail Analytics Pipeline for CPU-Optimized Checkout Queue Monitoring, Multi-Object Dwell-Time Tracking, and Real-Time Observability.**

Developed by **Ayushman Chabri**, **Priya Agarwala**, and **Nikita Kumari**.

---

## Overview

The **Smart Queue & Retail Dwell-Time Monitor** is an end-to-end computer vision microservice engineered to detect retail checkout queue congestion and calculate per-customer dwell times on standard CPU hardware. Rather than running raw inference in a standalone loop, the system implements bounded stream backpressure, persistent multi-object tracking, resolution-independent spatial zoning, live Prometheus/Grafana telemetry, and an automated active-learning data flywheel.

---

## System Architecture

```text
[Camera / Stream Simulator] ──(XADD MAXLEN ~5)──▶ [Redis Stream]
                                                        │
                                                        ▼
[Grafana Dashboard] ◀──(Scrape 5s)── [Prometheus] ◀── [FastAPI + ONNX Runtime + ByteTrack]
                                                        │
                                                        └──(0.30 <= conf <= 0.50)──▶ [hard_examples/]
```

### Core Engineering Components
* **Deterministic CPU Inference (`app/inference.py`):** Static `YOLO11n` exported to ONNX (`opset=17`, `10.2 MB`) running via ONNX Runtime (`ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`) pinned to physical CPU cores to prevent hyperthreading oversubscription.
* **Multi-Object Tracking & Dwell-Time Engine (`app/inference.py`):** Filters strictly for COCO class `0` (`person`), maintains persistent customer IDs across occlusions using **ByteTrack**, and maps foot-anchor coordinates into a resolution-independent normalized polygon (`Supervision PolygonZone`) to track real-time wait durations and trigger congestion alerts.
* **Bounded Stream Backpressure (`app/main.py`):** Asynchronous FastAPI background worker consumes frames from a Redis Stream capped at `MAXLEN ~ 5`, automatically shedding stale frames during CPU bursts to prevent memory bloat and delayed alerts.
* **Active Learning Capture Loop (`app/inference.py`):** Rate-limits (1 frame / 5s) and saves borderline detections (`0.30 <= conf <= 0.50`) directly to `hard_examples/` for domain-specific `v2.0` fine-tuning.
* **Observability & Automated CI/CD Gates (`monitoring/`, `.github/workflows/`):** Exposes real-time metrics at `/metrics` for Prometheus and Grafana, paired with GitHub Actions pipelines enforcing `ruff` linting, `pytest` API verification, and a strict `p95 < 85 ms` latency SLA gate.

---

## Empirical Benchmarks (1,260-Image Evaluation)

Evaluated end-to-end on CPU across **1,260 labeled retail checkout images** (`12,855` annotated ground-truth boxes) using `scripts/evaluate_dataset.py`:

| Metric | Target SLA | Measured Result |
| :--- | :--- | :--- |
| **Model Artifact Size** | `< 25.0 MB` | **10.2 MB** (`models/yolo11n.onnx`) |
| **Mean Inference Latency** | `< 65.0 ms` | **40.45 ms** (~24.7 FPS on CPU) |
| **p95 Inference Latency** | `< 85.0 ms` (CI Gate) | **44.96 ms** |
| **Zero-Shot Precision (@IoU 0.50)** | `> 80.0%` | **87.98%** (`5,531 TP / 756 FP`) |
| **Zero-Shot Recall (@IoU 0.50)** | Baseline Diagnostic | **43.03%** (`F1: 0.5779`, `Avg Conf: 0.743`) |

### Diagnostic Takeaways
* **Sub-45 ms Tail Latency:** Static ONNX Runtime execution achieves a **44.96 ms p95 latency**, beating the 85 ms CI/CD SLA gate by nearly 2x without GPU acceleration.
* **High Precision (`87.98%`) vs. Zero-Shot Recall (`43.03%`):** Low false-positive rates prevent spurious congestion alerts, while lower recall on heavily tilted augmentations, lower-body stanchion occlusions, and multi-class dataset annotations (`kiosks`/`barriers` excluded by the `person`-only filter) empirically validates the active-learning capture pipeline for `v2.0` fine-tuning.

---

## Quickstart (Docker Compose)

### 1. Launch the 4-Container Stack
```bash
docker compose up --build -d
```
Verify that `queue-redis`, `queue-detector-api`, `queue-prometheus`, and `queue-grafana` are healthy:
```bash
curl http://127.0.0.1:8000/health
```
Expected response:
```json
{"status":"healthy","model_loaded":true,"redis_connected":true}
```

### 2. Ingest Frames into the Pipeline
* **Single Image Diagnostics (REST API):**
  ```bash
  curl -X POST "http://127.0.0.1:8000/analyze/frame" -F "file=@sample_frame.jpg"
  ```
* **Continuous Live Stream Simulation (Redis):**
  ```bash
  # Loop a single image to simulate dwell-time accumulation
  python scripts/simulate_stream.py --image sample_frame.jpg --fps 5

  # Or stream an entire directory of sequential frames
  python scripts/simulate_stream.py --image-dir dataset/images --fps 10
  ```

### 3. Monitor Live Telemetry
* **FastAPI Swagger UI:** `http://127.0.0.1:8000/docs`
* **Live Stream JSON State:** `http://127.0.0.1:8000/stream/status`
* **Prometheus Metrics Endpoint:** `http://127.0.0.1:8000/metrics`
* **Grafana Dashboard:** `http://127.0.0.1:3000` (Default credentials: `admin` / `admin`)

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Verifies ONNX model readiness and Redis stream connectivity |
| `POST` | `/analyze/frame` | Synchronous single-image detection, zone count, and confidence analysis |
| `POST` | `/analyze/frame/annotated` | Returns JPEG image overlaid with bounding boxes, tracker IDs, and zone polygon |
| `GET` | `/stream/status` | Returns real-time queue count, max/avg dwell time, and active alert status from Redis worker |
| `GET` | `/metrics` | Exposes Prometheus counters and latency histograms |

---

## Offline Dataset Evaluation & Testing

### Run Offline Accuracy & Latency Benchmark
```bash
python scripts/evaluate_dataset.py --images dataset/images --labels dataset/labels
```

### Run Automated CI/CD Test Suite Locally
```bash
pytest -v -s tests/
```

---

## Repository Structure

```text
smart_queue_monitor/
├── .github/workflows/ml-ci-cd.yml   # Automated Ruff lint, API tests & p95 < 85ms SLA gate
├── app/
│   ├── inference.py                 # ONNX Runtime + ByteTrack + PolygonZone dwell engine
│   └── main.py                      # FastAPI endpoints, Redis stream consumer & Prometheus metrics
├── hard_examples/                   # Auto-captured uncertain frames (0.30 <= conf <= 0.50)
├── models/
│   └── yolo11n.onnx                 # Static ONNX detection model (10.2 MB)
├── monitoring/
│   ├── prometheus.yml               # Prometheus 5s scrape configuration
│   └── grafana/provisioning/        # Provisioned Prometheus datasource & Smart Queue dashboard
├── scripts/
│   ├── evaluate_dataset.py          # Offline Precision/Recall/F1 & latency benchmark script
│   └── simulate_stream.py           # Image/directory/video Redis stream producer
├── tests/
│   ├── test_api.py                  # FastAPI endpoint functional tests
│   └── test_model_sla.py            # Automated p95 latency & schema validation gate
├── docker-compose.yml               # 4-container orchestration (Redis, API, Prometheus, Grafana)
├── Dockerfile                       # Slim Python 3.11 production container
├── PROPOSAL.md                      # Technical architecture & engineering trade-off proposal
└── requirements.txt                 # Pinned Python dependencies
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.