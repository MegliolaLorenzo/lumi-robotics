"""
Fight Detection — Main pipeline

Usage:
    python main.py                        # webcam (default)
    python main.py --source robot         # Unitree Go2 via WiFi
    python main.py --source video --video path.mp4
"""

import cv2
import time
import argparse
import sys

from detector import ViolenceDetector
from alert import send_alert, notify_clear
from video_source import create_source


def run_pipeline(source_type: str = "webcam", video_path: str = None):
    detector = ViolenceDetector()

    try:
        kwargs = {}
        if video_path:
            kwargs["path"] = video_path
        source = create_source(source_type, **kwargs)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"Video source error: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Fight Detection Pipeline")
    print(f"{'='*50}")
    print(f"  Source : {source_type.upper()}")
    print(f"  Model  : {detector.mode}")
    print(f"  Press Q to quit")
    print(f"{'='*50}\n")

    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = source.read()

        if not ret or frame is None:
            if source_type == "robot":
                print("  No frame received from robot — waiting...")
                time.sleep(0.5)
                continue
            else:
                print("  Source ended.")
                break

        frame_count += 1

        detected, confidence, annotated_frame = detector.analyze_frame(frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - fps_time
            fps = 30 / elapsed if elapsed > 0 else 0
            fps_time = time.time()
            print(f"  FPS: {fps:.1f} | Frames: {frame_count} | Source: {source_type}")

        if detected:
            send_alert(event_type="fight", confidence=confidence)
        else:
            notify_clear("fight")

        cv2.imshow("Fight Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Quit.")
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Detection Pipeline")
    parser.add_argument("--source", type=str, default="webcam",
                        choices=["webcam", "robot", "video"],
                        help="Video source (default: webcam)")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file path (only with --source video)")
    args = parser.parse_args()

    run_pipeline(source_type=args.source, video_path=args.video)
