"""
Alert — local notifications when a fight is detected.
No external backend required.

Modes (set ALERT_MODE in .env):
  print   → print to terminal (default)
  webhook → send POST JSON to ALERT_WEBHOOK_URL
  file    → append a line to alerts.log
"""

import time
from datetime import datetime
from config import ALERT_MODE, ALERT_WEBHOOK_URL

SESSION_CLEAR_TIMEOUT = 8  # seconds without signal before incident is closed

_sessions: dict[str, dict] = {}


def send_alert(event_type: str, confidence: float):
    """Call when an event is detected (fight, etc.)."""
    session = _sessions.get(event_type, {})
    if session.get("active"):
        session["clear_since"] = None  # signal still present
        now = time.time()
        if now - session.get("last_log", 0) >= 5:
            duration = now - session["started_at"]
            print(f"  Incident ongoing for {duration:.0f}s...")
            session["last_log"] = now
        return

    # New incident
    _sessions[event_type] = {
        "active": True,
        "started_at": time.time(),
        "clear_since": None,
        "last_log": time.time()
    }

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{now_str}] FIGHT DETECTED — confidence {confidence:.0%}"

    print(f"\n{'='*50}")
    print(f"  ALERT: {message}")
    print(f"{'='*50}\n")

    if ALERT_MODE == "file":
        try:
            with open("alerts.log", "a") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"  Log write error: {e}")

    elif ALERT_MODE == "webhook":
        if not ALERT_WEBHOOK_URL:
            print("  ALERT_WEBHOOK_URL not set in .env")
            return
        try:
            import requests
            requests.post(
                ALERT_WEBHOOK_URL,
                json={
                    "event": event_type,
                    "confidence": confidence,
                    "timestamp": now_str
                },
                timeout=5
            )
            print(f"  Webhook sent to {ALERT_WEBHOOK_URL}")
        except Exception as e:
            print(f"  Webhook error: {e}")


def notify_clear(event_type: str):
    """Call when the signal disappears — closes the session after timeout."""
    session = _sessions.get(event_type)
    if session and session["active"]:
        if session.get("clear_since") is None:
            session["clear_since"] = time.time()
        elif time.time() - session["clear_since"] > SESSION_CLEAR_TIMEOUT:
            duration = time.time() - session["started_at"]
            print(f"  Incident '{event_type}' ended — duration: {duration:.0f}s")
            _sessions[event_type] = {"active": False}
