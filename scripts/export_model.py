import os
import shutil
import time
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def export_and_benchmark(output_dir: str = "models", imgsz: int = 640):
    os.makedirs(output_dir, exist_ok=True)
    target_onnx_path = os.path.join(output_dir, "yolo11n.onnx")

    print("1. Loading pretrained YOLO11n weights...")
    model = YOLO("yolo11n.pt")

    print("2. Exporting to ONNX (opset=17, simplified static graph)...")
    exported_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
    shutil.move(exported_path, target_onnx_path)
    print(f"Saved ONNX model to: {target_onnx_path}")

    # Quick CPU latency check using ONNX Runtime
    session = ort.InferenceSession(target_onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times_ms.append((time.perf_counter() - t0) * 1000)

    print(f"ONNX CPU Benchmark -> Mean: {np.mean(times_ms):.2f} ms | p95: {np.percentile(times_ms, 95):.2f} ms")
    
import os
import onnxruntime as ort

def create_ort_session(model_path: str) -> ort.InferenceSession:
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


if __name__ == "__main__":
    export_and_benchmark()