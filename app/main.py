import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response

from app.inference import SmartQueueDetector


# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.onnx")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_KEY = "video:frames"
MAX_STREAM_LEN = 5  # Bounded ring buffer: drops stale frames to prevent lag buildup

# --- Prometheus Telemetry Metrics ---
INFERENCE_LATENCY = Histogram(
    "queue_inference_latency_ms",
    "ONNX model inference latency in milliseconds",
    buckets=(10, 20, 30, 40, 50, 60, 75, 100, 150, 250),
)
QUEUE_COUNT_GAUGE = Gauge("queue_active_customers", "Current number of customers in checkout zone")
MAX_DWELL_GAUGE = Gauge("queue_max_dwell_seconds", "Maximum customer wait time in seconds")
AVG_CONF_GAUGE = Gauge("queue_avg_detection_confidence", "Rolling average person detection confidence")
ALERTS_COUNTER = Counter("queue_alerts_total", "Total queue congestion or dwell-time alerts triggered")
FRAMES_DROPPED_COUNTER = Counter("queue_frames_dropped_total", "Stale frames dropped due to backpressure")

detector: Optional[SmartQueueDetector] = None
redis_client: Optional[redis.Redis] = None
worker_task: Optional[asyncio.Task] = None
latest_state: dict = {"status": "idle"}


async def inference_consumer_loop():
    """Continuously consumes the newest frame from Redis, skipping stale backlog frames."""
    global latest_state
    last_id = "$"

    while True:
        try:
            if redis_client is None or detector is None:
                await asyncio.sleep(0.2)
                continue

            # Block briefly for new frames
            entries = await redis_client.xread({STREAM_KEY: last_id}, count=10, block=500)
            if not entries:
                continue

            _, messages = entries[0]
            # If multiple frames queued up while we were busy, skip to the newest one
            if len(messages) > 1:
                FRAMES_DROPPED_COUNTER.inc(len(messages) - 1)

            newest_id, payload = messages[-1]
            last_id = newest_id

            img_bytes = payload[b"frame"]
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            metrics, _ = await asyncio.to_thread(detector.process_frame, frame, False)

            # Update Prometheus gauges & histograms
            INFERENCE_LATENCY.observe(metrics.inference_ms)
            QUEUE_COUNT_GAUGE.set(metrics.queue_count)
            MAX_DWELL_GAUGE.set(metrics.max_dwell_sec)
            if metrics.avg_confidence > 0:
                AVG_CONF_GAUGE.set(metrics.avg_confidence)
            if metrics.alert_triggered:
                ALERTS_COUNTER.inc()

            latest_state = {
                "frame_id": metrics.frame_id,
                "inference_ms": metrics.inference_ms,
                "queue_count": metrics.queue_count,
                "max_dwell_sec": metrics.max_dwell_sec,
                "avg_dwell_sec": metrics.avg_dwell_sec,
                "avg_confidence": metrics.avg_confidence,
                "alert_triggered": metrics.alert_triggered,
                "alert_reasons": metrics.alert_reasons,
                "active_tracks": metrics.active_tracks,
            }
        except asyncio.CancelledError:
            break
        except Exception as exc:
            latest_state = {"status": "error", "detail": str(exc)}
            await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, redis_client, worker_task
    detector = SmartQueueDetector(model_path=MODEL_PATH)
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        worker_task = asyncio.create_task(inference_consumer_loop())
    except Exception:
        # Allows stateless REST endpoints (/health, /analyze/frame) to run even without Redis in CI
        redis_client = None
    yield
    if worker_task:
        worker_task.cancel()
    if redis_client:
        await redis_client.close()


app = FastAPI(title="Smart Queue & Dwell-Time Monitor", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "redis_connected": redis_client is not None,
    }


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stream/status")
async def get_stream_status():
    return latest_state


@app.post("/analyze/frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """Synchronous endpoint for single-image inspection and CI verification."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    metrics, _ = detector.process_frame(frame, annotate=False)
    INFERENCE_LATENCY.observe(metrics.inference_ms)
    QUEUE_COUNT_GAUGE.set(metrics.queue_count)
    MAX_DWELL_GAUGE.set(metrics.max_dwell_sec)

    return {
        "inference_ms": metrics.inference_ms,
        "queue_count": metrics.queue_count,
        "max_dwell_sec": metrics.max_dwell_sec,
        "avg_confidence": metrics.avg_confidence,
        "alert_triggered": metrics.alert_triggered,
        "alert_reasons": metrics.alert_reasons,
    }
    
@app.post("/analyze/frame/annotated")
async def analyze_frame_annotated(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    _, annotated = detector.process_frame(frame, annotate=True)
    _, jpeg_bytes = cv2.imencode(".jpg", annotated)
    return Response(content=jpeg_bytes.tobytes(), media_type="image/jpeg")
    