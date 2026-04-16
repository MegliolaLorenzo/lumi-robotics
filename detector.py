"""
Violence detector — GPT-4o Vision (live stream).

States: safe / suspect / aggression
- Accumulates frames from the live stream into a buffer of CHUNK_SECONDS seconds
- Every CHUNK_SECONDS sends the frames to GPT-4o in a background thread
- Display is always live; overlay updates as soon as the response arrives
"""

import cv2
import json
import numpy as np
import base64
import threading
import time
from openai import OpenAI
from ultralytics import YOLO
from config import OPENAI_API_KEY

CHUNK_SECONDS  = 10
CAPTURE_FPS    = 2
FRAME_INTERVAL = 1.0 / CAPTURE_FPS

STATE_CONF = {
    "safe":       0.0,
    "suspect":    0.5,
    "aggression": 0.9,
}

STATE_COLOR = {
    "safe":       (0, 200, 0),
    "suspect":    (0, 165, 255),
    "aggression": (0, 0, 255),
}

PROMPT = """You are a security camera AI. Analyze these frames from a live security feed.

Classify the scene into exactly one of three states:

"safe": people talking, standing, sitting, walking — even if close together. Looking at camera is safe. Normal social interaction is safe.

"suspect": one person moves hands/arms TOWARD another person's body in a non-casual way — e.g. reaching to grab, push, or strike. Aggressive posture with intent. NOT just standing close.

"aggression": actual physical contact with aggressive intent — grabbing, pushing, wrestling, punching, slapping, headlock, arms wrapped around someone to restrain/fight. Real OR simulated. If hands are raised and touching the other person → aggression.

RULES:
1. Standing close or looking at each other → "safe"
2. Hands raised toward the other person but not touching → "suspect"
3. Any physical contact between people (grab, push, wrap arms, shove) → "aggression"
4. Smiling/laughing does NOT matter — classify on body contact only
5. Simulated or staged fights → still "aggression"

Respond ONLY with valid JSON:
{
  "state": "safe" or "suspect" or "aggression",
  "confidence": "high" or "medium" or "low",
  "description": "one sentence describing what you see",
  "people_involved": number or null
}"""


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class ViolenceDetector:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY not found in .env\n"
                "Add it: OPENAI_API_KEY=sk-..."
            )

        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self.mode = "gpt-4o-vision"

        print("  Loading YOLOv8n-pose (visualization)...")
        self._pose = YOLO("yolov8n-pose.pt")

        self._frame_buffer: list[np.ndarray] = []
        self._last_capture_time = 0.0
        self._chunk_start_time = time.time()

        self._lock = threading.Lock()
        self._state = "safe"
        self._confidence = 0.0
        self._description = "Starting analysis..."
        self._is_analyzing = False

        print("  GPT-4o Vision detector ready")
        print(f"  Analysis every {CHUNK_SECONDS}s at {CAPTURE_FPS} FPS → {CHUNK_SECONDS * CAPTURE_FPS} frames/chunk")

    def analyze_frame(self, frame: np.ndarray) -> tuple[bool, float, np.ndarray]:
        now = time.time()

        if now - self._last_capture_time >= FRAME_INTERVAL:
            self._frame_buffer.append(frame.copy())
            self._last_capture_time = now

        chunk_elapsed = now - self._chunk_start_time
        with self._lock:
            already_analyzing = self._is_analyzing

        if chunk_elapsed >= CHUNK_SECONDS and not already_analyzing and len(self._frame_buffer) > 0:
            frames_to_send = self._frame_buffer.copy()
            self._frame_buffer.clear()
            self._chunk_start_time = now
            with self._lock:
                self._is_analyzing = True
            threading.Thread(target=self._run_gpt, args=(frames_to_send,), daemon=True).start()

        with self._lock:
            state     = self._state
            conf      = self._confidence
            desc      = self._description
            analyzing = self._is_analyzing

        fight_detected = state == "aggression"
        pose_result = self._pose(frame, verbose=False)[0]
        annotated = pose_result.plot()
        self._draw_overlay(annotated, state, conf, desc, analyzing, chunk_elapsed)
        return fight_detected, conf, annotated

    def _run_gpt(self, frames: list[np.ndarray]):
        try:
            content = [{"type": "text", "text": PROMPT}]
            for f in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_frame(f)}",
                        "detail": "low"
                    }
                })

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=200
            )
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            state = result.get("state", "safe")
            if state not in STATE_CONF:
                state = "safe"
            conf     = STATE_CONF[state]
            desc     = result.get("description", "")
            conf_str = result.get("confidence", "?")
            people   = result.get("people_involved")

            print(f"  [GPT-4o] {state.upper()} ({conf_str}) people={people}")
            print(f"  [GPT-4o] {desc}")

        except Exception as e:
            print(f"  [GPT-4o] Error: {e}")
            state = "safe"
            conf  = 0.0
            desc  = f"Error: {e}"

        with self._lock:
            self._state       = state
            self._confidence  = conf
            self._description = desc
            self._is_analyzing = False

    def _draw_overlay(self, frame: np.ndarray, state: str, conf: float,
                      desc: str, analyzing: bool, chunk_elapsed: float):
        h, w = frame.shape[:2]
        color = STATE_COLOR.get(state, (0, 200, 0))

        # Confidence bar at top
        bar_w = int(w * conf)
        cv2.rectangle(frame, (0, 0), (bar_w, 10), color, -1)
        cv2.rectangle(frame, (0, 0), (w, 10), (80, 80, 80), 1)

        # Red border on aggression
        if state == "aggression":
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        if analyzing:
            label = "ANALYZING..."
            label_color = (0, 165, 255)
        elif state == "aggression":
            label = "AGGRESSION DETECTED"
            label_color = (0, 0, 255)
        elif state == "suspect":
            label = "SUSPECT BEHAVIOUR"
            label_color = (0, 165, 255)
        else:
            label = "SAFE"
            label_color = (0, 200, 0)

        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

        if desc and not analyzing:
            short = desc[:75] + "..." if len(desc) > 75 else desc
            cv2.putText(frame, short, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Chunk progress bar at bottom
        progress = min(1.0, chunk_elapsed / CHUNK_SECONDS)
        cv2.rectangle(frame, (0, h - 14), (w, h), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, h - 14), (int(w * progress), h), (100, 100, 100), -1)
        next_in = max(0, CHUNK_SECONDS - chunk_elapsed)
        cv2.putText(frame, f"Next analysis: {next_in:.0f}s", (10, h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
