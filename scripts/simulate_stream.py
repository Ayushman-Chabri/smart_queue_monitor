import argparse
import glob
import os
import time
import cv2
import numpy as np
import redis


def connect_redis() -> redis.Redis:
    return redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        protocol=2,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def push_frame(r: redis.Redis, frame: np.ndarray, fps: float) -> None:
    ok, jpeg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        r.xadd("video:frames", {"frame": jpeg_buf.tobytes()}, maxlen=5, approximate=True)
    time.sleep(1.0 / fps)


def stream_images(image_paths: list[str], fps: float = 5.0, loop: bool = True) -> None:
    r = connect_redis()
    print(f"Streaming {len(image_paths)} image(s) at {fps} FPS to Redis...")
    while True:
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            push_frame(r, frame, fps)
        if not loop:
            break


def stream_video(video_path: str, fps: float = 10.0) -> None:
    r = connect_redis()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    print(f"Streaming video {video_path} at {fps} FPS to Redis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        push_frame(r, frame, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream images or video frames into Redis.")
    parser.add_argument("--image", type=str, help="Path to a single image to loop continuously")
    parser.add_argument("--image-dir", type=str, help="Directory of sequential .jpg/.png frames")
    parser.add_argument("--video", type=str, help="Path to an MP4 video file")
    parser.add_argument("--fps", type=float, default=5.0, help="Simulated camera frame rate")
    args = parser.parse_args()

    if args.image:
        stream_images([args.image], fps=args.fps, loop=True)
    elif args.image_dir:
        files = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg"))
            + glob.glob(os.path.join(args.image_dir, "*.png"))
        )
        if not files:
            raise FileNotFoundError(f"No .jpg or .png images found in {args.image_dir}")
        stream_images(files, fps=args.fps, loop=True)
    elif args.video:
        stream_video(args.video, fps=args.fps)
    else:
        r = connect_redis()
        print("Streaming synthetic test frames...")
        while True:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            push_frame(r, dummy, args.fps)