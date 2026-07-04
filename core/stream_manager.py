"""
Multi-stream orchestration for the web dashboard.
===================================================
Runs one ThreadSafeDetector + capture loop per registered camera so several
video sources (files, webcams, RTSP) can be monitored concurrently. This
module only orchestrates existing detection code — it does not change
preprocessing, sequence length, or thresholding in core/detection_engine.py.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from config.settings import DetectionConfig, WebConfig
from .detection_engine import ThreadSafeDetector

logger = logging.getLogger(__name__)

SCREENSHOT_COOLDOWN = 10   # seconds between saved screenshots per stream
LOG_INTERVAL = 30          # write a DetectionLog row every N frames


def coerce_source(source_url):
    """'0' -> webcam index 0; anything else (file path / rtsp/http URL) stays a string."""
    source_url = str(source_url).strip()
    return int(source_url) if source_url.isdigit() else source_url


def _save_screenshot(annotated_frame) -> Optional[str]:
    try:
        screenshot_dir = Path(__file__).parent.parent / 'alerts' / 'screenshots'
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        fname = screenshot_dir / f"incident_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
        cv2.imwrite(str(fname), annotated_frame)
        return str(fname)
    except Exception as exc:
        logger.warning("Screenshot save failed: %s", exc)
        return None


class StreamWorker:
    """Owns one camera's capture + detector + frame buffers, running in its own thread."""

    def __init__(
        self,
        stream_id: str,
        source_url,
        model_path: Optional[str],
        use_yolo: bool,
        on_violence: Optional[Callable] = None,
        on_alert: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ):
        self.stream_id = stream_id
        self.source_url = source_url
        self.source = coerce_source(source_url)
        self.model_path = model_path
        self.use_yolo = use_yolo
        self.on_violence = on_violence
        self.on_alert = on_alert
        self.on_log = on_log

        self.detector: Optional[ThreadSafeDetector] = None
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self.current_frame = None
        self.current_raw_frame = None
        self.stats = {
            'total_frames': 0,
            'violence_detections': 0,
            'alerts_triggered': 0,
            'current_fps': 0,
            'start_time': None,
        }

    def start(self):
        self.detector = ThreadSafeDetector(
            lstm_model_path=self.model_path,
            use_yolo=self.use_yolo,
            use_scene_classifier=True,
            use_person_classifier=False,
        )
        self.detector.start()
        self._running = True
        self.stats['start_time'] = time.time()
        self._thread = threading.Thread(
            target=self._run,
            name=f"StreamWorker-{self.stream_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Stream '%s' started (source=%s)", self.stream_id, self.source)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
        if self.detector:
            self.detector.stop()
        logger.info("Stream '%s' stopped", self.stream_id)

    def get_frame(self, raw: bool = False):
        with self._frame_lock:
            frame = self.current_raw_frame if raw else self.current_frame
            return frame.copy() if frame is not None else None

    def _run(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error("Stream '%s': failed to open source %s", self.stream_id, self.source)
            self._running = False
            return

        last_screenshot_time = 0.0
        last_result = None
        frame_idx = 0
        frame_skip = max(1, DetectionConfig.DETECT_FRAME_SKIP)
        loop_fps_ema = None
        last_loop_t = time.time()

        try:
            while self._running:
                t0 = time.time()
                ret, frame = self._cap.read()
                if not ret:
                    # Loop file sources back to the start instead of dying at EOF.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Under multi-stream load, only run the full detector every
                # Nth frame (DetectionConfig.DETECT_FRAME_SKIP) and reuse the
                # last result in between — cuts CPU load roughly
                # proportionally while every frame still gets drawn+streamed
                # for smooth video. Incident/alert/log side effects only run
                # on frames that actually went through detection, so this
                # never multiplies incident writes for one detection event.
                frame_idx += 1
                did_detect = last_result is None or frame_idx % frame_skip == 0
                if did_detect:
                    result = self.detector.process_frame(frame)
                    last_result = result
                    processing_ms = (time.time() - t0) * 1000
                else:
                    result = last_result

                self.stats['total_frames'] += 1

                annotated = self.detector.draw_results(frame, result)

                with self._frame_lock:
                    self.current_raw_frame = frame.copy()
                    self.current_frame = annotated.copy()

                # FPS measured on the actual capture/draw cadence (what the
                # browser sees), not just the — possibly skipped —
                # inference rate, so the stats panel reflects real smoothness.
                now = time.time()
                inst_fps = 1.0 / max(now - last_loop_t, 1e-6)
                loop_fps_ema = inst_fps if loop_fps_ema is None else (0.9 * loop_fps_ema + 0.1 * inst_fps)
                last_loop_t = now
                self.stats['current_fps'] = round(loop_fps_ema, 1)

                if did_detect and result.has_violence:
                    self.stats['violence_detections'] += 1

                    screenshot_path = None
                    if t0 - last_screenshot_time >= SCREENSHOT_COOLDOWN:
                        screenshot_path = _save_screenshot(annotated)
                        if screenshot_path:
                            last_screenshot_time = t0

                    for det in result.detections:
                        if det.is_violent:
                            self.stats['alerts_triggered'] += 1
                            if self.on_violence:
                                incident_data = self.on_violence(
                                    self.stream_id,
                                    det,
                                    screenshot_path,
                                    round(result.scene_violence_prob, 4) if result.scene_violence_prob else None,
                                    len(result.detections),
                                )
                                if incident_data and self.on_alert:
                                    self.on_alert(incident_data)

                if did_detect and self.on_log and self.stats['total_frames'] % LOG_INTERVAL == 0:
                    self.on_log(self.stream_id, result, processing_ms)

                time.sleep(1.0 / WebConfig.STREAM_FPS)
        finally:
            if self._cap is not None:
                self._cap.release()
            self._running = False


class StreamManager:
    """Registry of running StreamWorkers, keyed by stream_id."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_yolo: bool = True,
        on_violence: Optional[Callable] = None,
        on_alert: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ):
        self.model_path = model_path
        self.use_yolo = use_yolo
        self.on_violence = on_violence
        self.on_alert = on_alert
        self.on_log = on_log
        self._workers = {}
        self._registry_lock = threading.Lock()

    def start_stream(self, stream_id: str, source_url) -> StreamWorker:
        with self._registry_lock:
            existing = self._workers.get(stream_id)
            if existing:
                return existing
            worker = StreamWorker(
                stream_id, source_url, self.model_path, self.use_yolo,
                on_violence=self.on_violence,
                on_alert=self.on_alert,
                on_log=self.on_log,
            )
            self._workers[stream_id] = worker
        worker.start()
        return worker

    def stop_stream(self, stream_id: str):
        with self._registry_lock:
            worker = self._workers.pop(stream_id, None)
        if worker:
            worker.stop()

    def restart_stream(self, stream_id: str, source_url) -> StreamWorker:
        self.stop_stream(stream_id)
        return self.start_stream(stream_id, source_url)

    def get_worker(self, stream_id: str) -> Optional[StreamWorker]:
        return self._workers.get(stream_id)

    def is_live(self, stream_id: str) -> bool:
        worker = self._workers.get(stream_id)
        return bool(worker and worker._running)

    def primary_worker(self) -> Optional[StreamWorker]:
        """The first stream started this session — used by the legacy single-stream routes."""
        with self._registry_lock:
            return next(iter(self._workers.values()), None)

    def sync_from_db(self):
        """Start a worker for every Stream row already marked active in the DB."""
        from database.db import get_session
        from database.models import Stream

        session = get_session()
        try:
            rows = session.query(Stream).filter_by(is_active=True).all()
            for row in rows:
                self.start_stream(row.stream_id, row.source_url)
        finally:
            session.close()

    def aggregate_stats(self) -> dict:
        with self._registry_lock:
            workers = list(self._workers.values())
        return {
            'total_frames': sum(w.stats['total_frames'] for w in workers),
            'violence_detections': sum(w.stats['violence_detections'] for w in workers),
            'alerts_triggered': sum(w.stats['alerts_triggered'] for w in workers),
            'current_fps': round(sum(w.stats['current_fps'] for w in workers) / len(workers), 1) if workers else 0,
            'is_running': any(w._running for w in workers),
        }
