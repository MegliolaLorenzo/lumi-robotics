"""
Video source abstraction — webcam / robot (MJPEG HTTP) / video file.
Set VIDEO_SOURCE in .env to switch source without touching main.py.
"""

import cv2
import numpy as np
import requests
import time
from abc import ABC, abstractmethod
from config import ROBOT_STREAM_URL, VIDEO_SOURCE


class VideoSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame. Returns (ok, frame)."""
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass


class WebcamSource(VideoSource):
    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()


class RobotMJPEGSource(VideoSource):
    """
    Reads frames from the hfarmai MJPEG HTTP stream.
    Compatible with: http://<robot_ip>:8080/api/video/stream
    """

    def __init__(self, stream_url: str, timeout: float = 10.0):
        self._url = stream_url
        self._timeout = timeout
        self._stream = None
        self._iter = None   # persistent iterator — do NOT recreate on every read()
        self._buffer = b""
        self._opened = False
        self._connect()

    def _connect(self):
        print(f"  Connecting to robot stream: {self._url}")
        try:
            self._stream = requests.get(
                self._url,
                stream=True,
                timeout=self._timeout
            )
            if self._stream.status_code == 200:
                self._iter = self._stream.iter_content(chunk_size=4096)
                self._buffer = b""
                self._opened = True
                print("  Robot stream connected")
            else:
                print(f"  Robot stream HTTP error {self._stream.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  Cannot connect to {self._url}")
            print("  Make sure the hfarmai server is running (start.py handles this automatically)")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._iter is None:
            return False, None

        try:
            while True:
                chunk = next(self._iter)
                if not chunk:
                    continue
                self._buffer += chunk

                start = self._buffer.find(b'\xff\xd8')
                end = self._buffer.find(b'\xff\xd9')

                if start != -1 and end != -1 and end > start:
                    jpg_data = self._buffer[start:end + 2]
                    self._buffer = self._buffer[end + 2:]

                    frame = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        return True, frame

        except StopIteration:
            print("  Stream ended — reconnecting...")
            time.sleep(1.0)
            self._connect()
        except requests.exceptions.ChunkedEncodingError:
            print("  Stream interrupted — reconnecting...")
            time.sleep(1.0)
            self._connect()
        except Exception as e:
            print(f"  Stream read error: {e}")

        return False, None

    def release(self):
        self._opened = False
        if self._stream:
            self._stream.close()

    def is_opened(self) -> bool:
        return self._opened


class VideoFileSource(VideoSource):
    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Video not found: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._frame_delay = 1.0 / fps

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            # Loop video
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        time.sleep(self._frame_delay)
        return ret, frame

    def release(self):
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()


def create_source(source_type: str = None, **kwargs) -> VideoSource:
    """
    Factory — creates the right video source based on config.

    source_type: "webcam" | "robot" | "video"
    """
    t = (source_type or VIDEO_SOURCE).lower()

    if t == "webcam":
        index = kwargs.get("index", 0)
        print(f"  Source: Webcam (index {index})")
        src = WebcamSource(index)
        if not src.is_opened():
            raise RuntimeError("Webcam not available")
        return src

    elif t == "robot":
        url = kwargs.get("stream_url", ROBOT_STREAM_URL)
        if not url:
            raise ValueError(
                "ROBOT_STREAM_URL not set in .env\n"
                "Run: python discovery.py to find the robot IP"
            )
        print(f"  Source: Robot MJPEG ({url})")
        return RobotMJPEGSource(url)

    elif t == "video":
        path = kwargs.get("path")
        if not path:
            raise ValueError("Specify a video path with --video <path>")
        print(f"  Source: Video file ({path})")
        return VideoFileSource(path)

    else:
        raise ValueError(f"Invalid source: '{t}'. Use: webcam | robot | video")
