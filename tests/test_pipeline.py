"""
Unit tests — Fight Detection Pipeline
Run: python -m pytest tests/
No robot, no webcam, no OpenAI key required.
"""

import sys
import os

# Ensure the project root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import time
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# ─── helpers ────────────────────────────────────────────────────────────────

def make_frame(h=480, w=640):
    """Returns a dummy BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_jpeg_bytes():
    """Returns a minimal valid JPEG byte sequence (SOI + EOI markers)."""
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00' + b'\x00' * 100 + b'\xff\xd9'


# ════════════════════════════════════════════════════════════════════════════
# config.py
# ════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_defaults_without_env(self):
        """Config provides sensible defaults when no .env is present."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib, config
            importlib.reload(config)
            self.assertEqual(config.ALERT_MODE, "print")
            self.assertEqual(config.VIDEO_SOURCE, "webcam")
            self.assertEqual(config.ROBOT_IP, "")
            self.assertGreater(config.CONFIDENCE_THRESHOLD, 0)

    def test_env_override(self):
        """Environment variables override defaults."""
        with patch.dict(os.environ, {"ALERT_MODE": "file", "ROBOT_IP": "192.168.1.42"}):
            import importlib, config
            importlib.reload(config)
            self.assertEqual(config.ALERT_MODE, "file")
            self.assertEqual(config.ROBOT_IP, "192.168.1.42")


# ════════════════════════════════════════════════════════════════════════════
# alert.py
# ════════════════════════════════════════════════════════════════════════════

class TestAlert(unittest.TestCase):

    def setUp(self):
        # Reset module state before every test
        import alert
        alert._sessions.clear()

    def test_new_incident_prints(self):
        with patch.dict(os.environ, {"ALERT_MODE": "print"}):
            import importlib, config, alert
            importlib.reload(config)
            alert.ALERT_MODE = "print"
            alert._sessions.clear()
            with patch("builtins.print") as mock_print:
                alert.send_alert("fight", confidence=0.9)
                printed = " ".join(str(c) for c in mock_print.call_args_list)
                self.assertIn("ALERT", printed)

    def test_ongoing_incident_does_not_reprint(self):
        import alert
        alert._sessions.clear()
        alert.send_alert("fight", confidence=0.9)
        with patch("builtins.print") as mock_print:
            alert.send_alert("fight", confidence=0.9)
            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertNotIn("ALERT", printed)

    def test_incident_written_to_file(self):
        import alert
        alert._sessions.clear()
        alert.ALERT_MODE = "file"
        m = mock_open()
        with patch("builtins.open", m):
            alert.send_alert("fight", confidence=0.75)
        m.assert_called_once_with("alerts.log", "a")
        written = m().write.call_args[0][0]
        self.assertIn("FIGHT DETECTED", written)
        self.assertIn("75%", written)

    def test_webhook_posts_json(self):
        import alert
        alert._sessions.clear()
        alert.ALERT_MODE = "webhook"
        alert.ALERT_WEBHOOK_URL = "http://example.com/hook"
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            alert.send_alert("fight", confidence=0.8)
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(payload["event"], "fight")
            self.assertAlmostEqual(payload["confidence"], 0.8)

    def test_webhook_skips_if_no_url(self):
        import alert
        alert._sessions.clear()
        alert.ALERT_MODE = "webhook"
        alert.ALERT_WEBHOOK_URL = ""
        with patch("requests.post") as mock_post:
            alert.send_alert("fight", confidence=0.9)
            mock_post.assert_not_called()

    def test_notify_clear_closes_session_after_timeout(self):
        import alert
        alert._sessions.clear()
        alert.SESSION_CLEAR_TIMEOUT = 0  # instant
        alert.send_alert("fight", confidence=0.9)
        alert._sessions["fight"]["clear_since"] = time.time() - 1
        alert.notify_clear("fight")
        self.assertFalse(alert._sessions["fight"]["active"])

    def test_notify_clear_noop_when_no_session(self):
        import alert
        alert._sessions.clear()
        # Should not raise
        alert.notify_clear("fight")

    def test_notify_clear_resets_timer_when_signal_returns(self):
        """If send_alert is called again while waiting to clear, timer resets."""
        import alert
        alert._sessions.clear()
        alert.send_alert("fight", confidence=0.9)
        alert.notify_clear("fight")                      # starts clear_since
        alert.send_alert("fight", confidence=0.9)        # signal back → clear_since = None
        self.assertIsNone(alert._sessions["fight"]["clear_since"])


# ════════════════════════════════════════════════════════════════════════════
# video_source.py
# ════════════════════════════════════════════════════════════════════════════

class TestCreateSource(unittest.TestCase):

    def test_invalid_source_raises(self):
        from video_source import create_source
        with self.assertRaises(ValueError):
            create_source("banana")

    def test_video_source_no_path_raises(self):
        from video_source import create_source
        with self.assertRaises(ValueError):
            create_source("video")

    def test_video_source_missing_file_raises(self):
        from video_source import create_source
        with self.assertRaises(FileNotFoundError):
            create_source("video", path="/nonexistent/file.mp4")

    def test_robot_source_no_url_raises(self):
        from video_source import create_source
        with patch("video_source.ROBOT_STREAM_URL", ""):
            with self.assertRaises(ValueError):
                create_source("robot")

    def test_webcam_not_available_raises(self):
        from video_source import create_source
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(RuntimeError):
                create_source("webcam")

    def test_webcam_returns_source(self):
        from video_source import create_source, WebcamSource
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("cv2.VideoCapture", return_value=mock_cap):
            src = create_source("webcam")
        self.assertIsInstance(src, WebcamSource)


class TestWebcamSource(unittest.TestCase):

    def _make_source(self):
        from video_source import WebcamSource
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, make_frame())
        with patch("cv2.VideoCapture", return_value=mock_cap):
            src = WebcamSource()
        src._cap = mock_cap
        return src, mock_cap

    def test_read_returns_frame(self):
        src, mock_cap = self._make_source()
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)

    def test_release_called(self):
        src, mock_cap = self._make_source()
        src.release()
        mock_cap.release.assert_called_once()

    def test_is_opened(self):
        src, mock_cap = self._make_source()
        self.assertTrue(src.is_opened())


class TestRobotMJPEGSource(unittest.TestCase):

    def _make_connected_source(self, chunks):
        from video_source import RobotMJPEGSource
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = iter(chunks)
        with patch("requests.get", return_value=mock_resp):
            src = RobotMJPEGSource("http://localhost:8080/stream")
        return src

    def test_connects_and_opens(self):
        src = self._make_connected_source([make_jpeg_bytes()])
        self.assertTrue(src.is_opened())

    def test_connection_error_leaves_closed(self):
        from video_source import RobotMJPEGSource
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            src = RobotMJPEGSource("http://localhost:8080/stream")
        self.assertFalse(src.is_opened())

    def test_read_returns_false_when_not_opened(self):
        from video_source import RobotMJPEGSource
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            src = RobotMJPEGSource("http://localhost:8080/stream")
        ok, frame = src.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_read_decodes_jpeg_frame(self):
        from video_source import RobotMJPEGSource
        # Build a real JPEG in memory using cv2
        import cv2
        dummy = make_frame()
        _, buf = cv2.imencode(".jpg", dummy)
        jpeg = buf.tobytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Wrap JPEG in MJPEG framing (SOI … EOI)
        mock_resp.iter_content.return_value = iter([jpeg])
        with patch("requests.get", return_value=mock_resp):
            src = RobotMJPEGSource("http://localhost:8080/stream")

        with patch("cv2.imdecode", return_value=make_frame()):
            ok, frame = src.read()
        # imdecode mock returns a frame → should succeed
        self.assertTrue(ok)

    def test_http_error_leaves_closed(self):
        from video_source import RobotMJPEGSource
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("requests.get", return_value=mock_resp):
            src = RobotMJPEGSource("http://localhost:8080/stream")
        self.assertFalse(src.is_opened())

    def test_release_closes_stream(self):
        src = self._make_connected_source([make_jpeg_bytes()])
        src.release()
        self.assertFalse(src.is_opened())


class TestVideoFileSource(unittest.TestCase):

    def _make_source(self, path="/fake/video.mp4"):
        from video_source import VideoFileSource
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (True, make_frame())
        with patch("cv2.VideoCapture", return_value=mock_cap):
            src = VideoFileSource(path)
        src._cap = mock_cap
        return src, mock_cap

    def test_reads_frame(self):
        src, _ = self._make_source()
        with patch("time.sleep"):
            ok, frame = src.read()
        self.assertTrue(ok)

    def test_loops_on_end(self):
        from video_source import VideoFileSource
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        # First read() returns (False, None) → triggers loop reset → second returns frame
        mock_cap.read.side_effect = [(False, None), (True, make_frame())]
        with patch("cv2.VideoCapture", return_value=mock_cap):
            src = VideoFileSource("/fake/video.mp4")
        src._cap = mock_cap
        with patch("time.sleep"):
            ok, frame = src.read()
        self.assertTrue(ok)
        # set() should have been called to reset position
        mock_cap.set.assert_called()


# ════════════════════════════════════════════════════════════════════════════
# discovery.py
# ════════════════════════════════════════════════════════════════════════════

class TestDiscovery(unittest.TestCase):

    def test_get_local_network_returns_subnet(self):
        from discovery import get_local_network
        subnet = get_local_network()
        self.assertRegex(subnet, r"\d+\.\d+\.\d+\.0/24")

    def test_get_local_ip_returns_ip(self):
        from discovery import get_local_ip
        ip = get_local_ip()
        self.assertRegex(ip, r"\d+\.\d+\.\d+\.\d+")

    def test_is_unitree_mac_known_prefix(self):
        from discovery import is_unitree_mac
        self.assertTrue(is_unitree_mac("b0:a7:b9:xx:xx:xx"))
        self.assertTrue(is_unitree_mac("7c:df:a1:xx:xx:xx"))
        self.assertFalse(is_unitree_mac("aa:bb:cc:dd:ee:ff"))

    def test_is_unitree_hostname(self):
        from discovery import is_unitree_hostname
        self.assertTrue(is_unitree_hostname("unitree-go2.local"))
        self.assertTrue(is_unitree_hostname("go2.home"))
        self.assertTrue(is_unitree_hostname("my-robot"))
        self.assertFalse(is_unitree_hostname("my-laptop"))

    def test_check_port_closed(self):
        from discovery import check_port
        # Port 1 is almost always closed
        result = check_port("127.0.0.1", 1, timeout=0.3)
        self.assertFalse(result)

    def test_check_port_open(self):
        """Opens a real local socket and checks that check_port detects it."""
        import socket, threading
        from discovery import check_port

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        def _serve():
            try:
                conn, _ = server.accept()
                conn.close()
            except Exception:
                pass
            finally:
                server.close()

        threading.Thread(target=_serve, daemon=True).start()
        result = check_port("127.0.0.1", port, timeout=2.0)
        self.assertTrue(result)

    def test_scan_host_returns_none_on_dead_ip(self):
        from discovery import scan_host
        # 192.0.2.x is reserved/unroutable — should return None quickly
        with patch("discovery.check_unitree_ports", return_value=[]):
            with patch("discovery.ping_host", return_value=False):
                result = scan_host("192.0.2.1")
        self.assertIsNone(result)

    def test_scan_host_returns_candidate_on_open_ports(self):
        from discovery import scan_host
        with patch("discovery.check_unitree_ports", return_value=[8080]):
            with patch("discovery.get_arp_table", return_value={}):
                with patch("discovery.try_resolve_hostname", return_value=""):
                    result = scan_host("192.168.1.42")
        self.assertIsNotNone(result)
        self.assertEqual(result["ip"], "192.168.1.42")
        self.assertIn(8080, result["open_ports"])
        self.assertGreater(result["confidence"], 0)

    def test_discover_robot_returns_sorted_candidates(self):
        from discovery import discover_robot

        fake_candidates = [
            {"ip": "192.168.1.10", "confidence": 40, "evidence": [], "open_ports": [8080], "mac": "", "hostname": ""},
            {"ip": "192.168.1.20", "confidence": 90, "evidence": [], "open_ports": [8080, 8081], "mac": "b0:a7:b9:00:00:00", "hostname": ""},
        ]

        def fake_scan(ip):
            for c in fake_candidates:
                if c["ip"] == str(ip):
                    return c
            return None

        # /28 gives 14 hosts including .10 and .20
        with patch("discovery.scan_host", side_effect=fake_scan):
            results = discover_robot("192.168.1.0/28")

        self.assertGreaterEqual(len(results), 1)
        confidences = [r["confidence"] for r in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_find_robot_stream_url_returns_none_when_unreachable(self):
        from discovery import find_robot_stream_url
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            url = find_robot_stream_url("192.0.2.1")
        self.assertIsNone(url)

    def test_find_robot_stream_url_returns_url_on_200(self):
        from discovery import find_robot_stream_url
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            url = find_robot_stream_url("192.168.1.42")
        self.assertIsNotNone(url)
        self.assertIn("192.168.1.42", url)


# ════════════════════════════════════════════════════════════════════════════
# detector.py  (mocked — no OpenAI / YOLO download required)
# ════════════════════════════════════════════════════════════════════════════

class TestViolenceDetectorLogic(unittest.TestCase):
    """
    Tests ViolenceDetector internal logic with fully mocked YOLO and OpenAI.
    """

    def _make_detector(self):
        """Returns a ViolenceDetector with all external I/O mocked."""
        mock_pose_result = MagicMock()
        mock_pose_result.plot.return_value = make_frame()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_pose_result]
        mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

        mock_openai_client = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_openai_client

        # Patch ultralytics at the sys.modules level so the import inside
        # detector.py succeeds even without the package installed.
        import sys
        fake_ultralytics = MagicMock()
        fake_ultralytics.YOLO = mock_yolo_cls

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics,
                                       "openai": mock_openai_module}):
            # Force re-import so the mocked modules are used
            if "detector" in sys.modules:
                del sys.modules["detector"]
            import detector as det_module
            with patch.object(det_module, "OPENAI_API_KEY", "sk-fake"):
                det = det_module.ViolenceDetector()

        det._pose = mock_yolo_instance
        det._client = mock_openai_client
        return det, mock_openai_client

    def test_initial_state_is_safe(self):
        det, _ = self._make_detector()
        self.assertEqual(det._state, "safe")
        self.assertEqual(det._confidence, 0.0)

    def test_analyze_frame_returns_tuple(self):
        det, _ = self._make_detector()
        detected, conf, annotated = det.analyze_frame(make_frame())
        self.assertIsInstance(detected, bool)
        self.assertIsInstance(conf, float)
        self.assertIsNotNone(annotated)

    def test_aggression_state_triggers_detected(self):
        det, _ = self._make_detector()
        det._state = "aggression"
        det._confidence = 0.9
        detected, conf, _ = det.analyze_frame(make_frame())
        self.assertTrue(detected)
        self.assertAlmostEqual(conf, 0.9)

    def test_safe_state_does_not_trigger(self):
        det, _ = self._make_detector()
        det._state = "safe"
        detected, _, _ = det.analyze_frame(make_frame())
        self.assertFalse(detected)

    def test_gpt_response_sets_state(self):
        det, mock_client = self._make_detector()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "state": "aggression",
            "confidence": "high",
            "description": "Two people fighting",
            "people_involved": 2
        })
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        frames = [make_frame(), make_frame()]
        det._run_gpt(frames)

        self.assertEqual(det._state, "aggression")
        self.assertEqual(det._confidence, 0.9)
        self.assertFalse(det._is_analyzing)

    def test_gpt_markdown_json_stripped(self):
        det, mock_client = self._make_detector()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "```json\n"
            '{"state": "suspect", "confidence": "medium", "description": "Hands raised", "people_involved": 2}'
            "\n```"
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        det._run_gpt([make_frame()])
        self.assertEqual(det._state, "suspect")

    def test_gpt_invalid_state_falls_back_to_safe(self):
        det, mock_client = self._make_detector()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "state": "unknown_state",
            "confidence": "low",
            "description": "N/A",
            "people_involved": None
        })
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        det._run_gpt([make_frame()])
        self.assertEqual(det._state, "safe")

    def test_gpt_exception_falls_back_to_safe(self):
        det, mock_client = self._make_detector()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        det._run_gpt([make_frame()])
        self.assertEqual(det._state, "safe")
        self.assertFalse(det._is_analyzing)

    def test_frame_buffer_accumulates(self):
        det, _ = self._make_detector()
        det._last_capture_time = 0.0  # force capture
        det.analyze_frame(make_frame())
        self.assertEqual(len(det._frame_buffer), 1)

    def test_chunk_triggers_gpt_thread(self):
        det, _ = self._make_detector()
        det._chunk_start_time = time.time() - 15  # past chunk window
        det._frame_buffer = [make_frame(), make_frame()]

        with patch.object(det, "_run_gpt") as mock_run, \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            det.analyze_frame(make_frame())
            mock_thread.assert_called_once()

    def test_no_double_analysis(self):
        """If already analyzing, no new thread should start."""
        det, _ = self._make_detector()
        det._chunk_start_time = time.time() - 15
        det._frame_buffer = [make_frame()]
        det._is_analyzing = True

        with patch("threading.Thread") as mock_thread:
            det.analyze_frame(make_frame())
            mock_thread.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# start.py  (launcher logic)
# ════════════════════════════════════════════════════════════════════════════

class TestStartLauncher(unittest.TestCase):

    def test_check_hfarmai_installed_false_if_missing(self):
        import start
        with patch("os.path.exists", return_value=False):
            self.assertFalse(start.check_hfarmai_installed())

    def test_check_hfarmai_installed_true_if_present(self):
        import start
        with patch("os.path.exists", return_value=True):
            self.assertTrue(start.check_hfarmai_installed())

    def test_ensure_hfarmai_installed_skips_if_present(self):
        import start
        with patch.object(start, "check_hfarmai_installed", return_value=True):
            with patch("subprocess.run") as mock_run:
                start.ensure_hfarmai_installed()
                mock_run.assert_not_called()

    def test_ensure_hfarmai_exits_if_git_missing(self):
        import start
        with patch.object(start, "check_hfarmai_installed", return_value=False):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaises(SystemExit):
                    start.ensure_hfarmai_installed()

    def test_wait_for_hfarmai_boot_returns_false_on_timeout(self):
        import start
        with patch("requests.get", side_effect=Exception("no server")):
            result = start.wait_for_hfarmai_boot(timeout=0.1)
        self.assertFalse(result)

    def test_wait_for_hfarmai_boot_returns_true_on_200(self):
        import start
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            result = start.wait_for_hfarmai_boot(timeout=2.0)
        self.assertTrue(result)

    def test_update_robot_ip_exits_if_no_robot(self):
        import start
        # discover_robot is imported inside the function via `from discovery import`
        with patch("discovery.discover_robot", return_value=[]):
            with self.assertRaises(SystemExit):
                start.update_robot_ip_from_discovery()

    def test_update_robot_ip_saves_to_env(self):
        import start
        candidates = [{"ip": "192.168.1.99", "confidence": 80}]
        env_content = "OPENAI_API_KEY=sk-test\nROBOT_IP=\n"
        m = mock_open(read_data=env_content)
        with patch("discovery.discover_robot", return_value=candidates), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", m):
            ip = start.update_robot_ip_from_discovery()
        self.assertEqual(ip, "192.168.1.99")
        written = "".join(call.args[0] for call in m().write.call_args_list)
        self.assertIn("192.168.1.99", written)


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
