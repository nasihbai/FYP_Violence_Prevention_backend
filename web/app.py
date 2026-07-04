"""
Flask Web Dashboard for Violence Detection System
=================================================
Real-time web interface powered by Flask-SocketIO and SQLAlchemy.

Changes from original:
- Flask-SocketIO replaces REST polling for stats and alerts
- Every violence detection is persisted to the database
- Server pushes 'violence_alert' and 'stats_update' events to all clients
"""

import os
import sys
import cv2
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

# Load .env before any config import reads os.environ
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_jwt_extended import JWTManager
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WebConfig, VideoConfig, AlertConfig
from core.stream_manager import StreamManager
from database import init_db, User, Stream, Incident, Alert, DetectionLog
from database.db import get_session
from .auth import auth_bp, require_manage_role, seed_demo_users
from .api import api_bp

logger = logging.getLogger(__name__)

# ==================== APP SETUP ====================

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = WebConfig.SECRET_KEY
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', WebConfig.SECRET_KEY)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # No expiry for dev; set timedelta in prod

_cors_origins = WebConfig.CORS_ORIGINS
_cors_value = "*" if _cors_origins.strip() == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": _cors_value}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=_cors_value, async_mode='threading')
jwt = JWTManager(app)

app.register_blueprint(auth_bp)
app.register_blueprint(api_bp)


# ==================== ERROR HANDLERS ====================
# Normalize all error responses to the shape the FE expects:
#   { "errors": { "<field>": ["message", ...] } }
# Use "_" as the field for non-validation / generic errors.

@app.errorhandler(Exception)
def _handle_uncaught(e):
    code = getattr(e, "code", 500)
    msg = getattr(e, "description", None) or str(e) or "Internal Server Error"
    logger.exception("Unhandled error on %s %s", request.method, request.path)
    return jsonify({"errors": {"_": [msg]}}), code


@app.errorhandler(404)
def _handle_404(e):
    return jsonify({"errors": {"_": [getattr(e, "description", None) or "Not Found"]}}), 404


@app.errorhandler(405)
def _handle_405(e):
    return jsonify({"errors": {"_": [getattr(e, "description", None) or "Method Not Allowed"]}}), 405

# Initialise database (SQLite by default; set DATABASE_URL env var for PostgreSQL)
init_db()
seed_demo_users()

# ==================== GLOBAL STATE ====================

# One StreamWorker per registered camera, orchestrated by this manager.
# `primary_stream_id` is the source the process was launched with (CLI arg /
# settings) — the legacy single-stream routes (/api/start, /api/stats, ...)
# stay scoped to it rather than being made per-stream.
stream_manager: StreamManager = None
primary_stream_id: str = None


# ==================== HELPERS ====================

def _resolve_model_path(model_path: str = None) -> str | None:
    if model_path:
        return model_path
    root = Path(__file__).parent.parent
    # Prefer the 132-feature model that matches the current pipeline
    candidates = [
        root / 'models' / 'violence_lstm_dataset.h5',   # 309-feature, proven pipeline
        root / 'lstm-model.h5',
        root / 'models' / 'violence_lstm_rwf2000.h5',
        root / 'models' / 'violence_lstm_enhanced.h5',
    ]
    return next((str(p) for p in candidates if p.exists()), None)


def initialize_detector(model_path: str = None, source=0, use_yolo: bool = True):
    """Initialise multi-stream orchestration and start the primary (CLI) source."""
    global stream_manager, primary_stream_id

    resolved_model_path = _resolve_model_path(model_path)
    model_path = resolved_model_path if resolved_model_path and Path(resolved_model_path).exists() else None

    stream_manager = StreamManager(
        model_path=model_path,
        use_yolo=use_yolo,
        on_violence=_save_incident,
        on_alert=lambda incident_data: socketio.emit('violence_alert', incident_data),
        on_log=_write_detection_log,
    )

    # Ensure the primary Stream row exists, then start it explicitly so the
    # CLI-supplied source always wins even if an older row with the same
    # stream_id has a stale source_url.
    session = get_session()
    try:
        stream = _get_or_create_stream(session, source)
        session.commit()
        primary_stream_id = stream.stream_id
    finally:
        session.close()

    stream_manager.start_stream(primary_stream_id, source)
    logger.info(f"Primary stream '{primary_stream_id}' started — source: {source}")

    # Resume any other cameras that were left active from a previous run.
    stream_manager.sync_from_db()


def _get_or_create_stream(session, source) -> Stream:
    """Return the Stream record for the current video source, creating it if absent."""
    stream_id = f"CAM_{source}" if isinstance(source, int) else Path(str(source)).stem.upper()
    stream = session.query(Stream).filter_by(stream_id=stream_id).first()
    if not stream:
        stream = Stream(
            stream_id=stream_id,
            name=f"Camera {source}",
            source_url=str(source),
            location=None,
            is_active=True,
        )
        session.add(stream)
        session.flush()  # get id without committing
    return stream


def _save_incident(
    stream_id: str,
    det,
    screenshot_path: str = None,
    scene_violence_score: float = None,
    person_count: int = None,
) -> dict | None:
    """
    Persist a violence detection event for the given stream.
    Creates: Incident → Alert. The Stream row is expected to already exist
    (every running StreamWorker's stream_id is backed by a Stream row).
    Returns the Alert dict (matches the shape the Vue store expects).
    """
    session = get_session()
    try:
        confidence = float(det.confidence)
        if confidence > 0.85:
            severity = 'high'
        elif confidence > 0.70:
            severity = 'medium'
        elif confidence > 0.55:
            severity = 'low'
        else:
            severity = 'low'

        alert_type = 'violent' if severity in ('high', 'medium') else 'threatening'

        stream = session.query(Stream).filter_by(stream_id=stream_id).first()
        if not stream:
            logger.warning(f"No Stream row for '{stream_id}'; incident logged without location")

        # Generate a human-readable incident code. With multiple StreamWorkers
        # detecting violence concurrently, two threads can read the same
        # count() before either commits — retry on the resulting unique
        # constraint violation instead of losing the incident.
        year = datetime.utcnow().year
        incident = None
        for attempt in range(5):
            count = session.query(Incident).count() + 1
            incident_code = f"INC-{year}-{count:04d}"
            incident = Incident(
                incident_code=incident_code,
                stream_id=stream_id,
                type=alert_type,
                confidence=confidence,
                scene_violence_score=scene_violence_score,
                person_count=person_count,
                timestamp=datetime.utcnow(),
                location=stream.location if stream else None,
                screenshot_path=screenshot_path,
                severity=severity,
                status='open',
            )
            session.add(incident)
            try:
                session.flush()
                break
            except IntegrityError:
                session.rollback()
                incident = None
        if incident is None:
            logger.error(f"Failed to generate a unique incident_code after retries (stream={stream_id})")
            return None

        alert = Alert(
            incident_id=incident.id,
            type=alert_type,
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )
        session.add(alert)
        session.commit()

        # Return the full Alert shape — identical to POST /api/test/fire-alert
        # and to what the FE's Alert type + alerts store expect (incident_id,
        # type, acknowledged, dismissed, ...). Built before the session closes
        # so the alert.incident relationship can still lazy-load.
        return alert.to_dict()
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save incident: {exc}")
        return None
    finally:
        session.close()


def _write_detection_log(stream_id: str, result, processing_ms: float):
    """Write one DetectionLog row. Called every LOG_INTERVAL frames."""
    session = get_session()
    try:
        detections_data = [
            {
                'person_id':  d.person_id,
                'confidence': round(float(d.confidence), 4),
                'is_violent': d.is_violent,
                'bbox':       d.bbox,
            }
            for d in result.detections
        ] if result.detections else []

        session.add(DetectionLog(
            stream_id=stream_id,
            timestamp=datetime.utcnow(),
            person_count=len(detections_data),
            has_violence=result.has_violence,
            scene_violence_score=round(result.scene_violence_prob, 4) if result.scene_violence_prob else None,
            detections=detections_data,
            processing_time_ms=round(processing_ms, 2),
        ))
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.debug(f"Detection log write failed: {exc}")
    finally:
        session.close()


# ==================== VIDEO / DETECTION ====================
# Per-stream capture, detection, incident/log persistence, and screenshot
# saving now live in core/stream_manager.py (StreamWorker). This module just
# wires StreamManager's callbacks to the DB/SocketIO and exposes per-stream
# MJPEG routes.


def _stats_broadcaster():
    """Background thread: push aggregate live stats to all connected clients every second."""
    while True:
        agg = stream_manager.aggregate_stats() if stream_manager else {
            'total_frames': 0, 'violence_detections': 0, 'alerts_triggered': 0,
            'current_fps': 0, 'is_running': False,
        }
        primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
        uptime = None
        if primary and primary.stats['start_time']:
            uptime = str(datetime.now() - datetime.fromtimestamp(primary.stats['start_time'])).split('.')[0]

        socketio.emit('stats_update', {
            'total_frames': agg['total_frames'],
            'violence_detections': agg['violence_detections'],
            'alerts_triggered': agg['alerts_triggered'],
            'current_fps': agg['current_fps'],
            'uptime': uptime or '00:00:00',
            'is_running': agg['is_running'],
        })
        time.sleep(1)


# Start stats broadcaster as a daemon thread
threading.Thread(target=_stats_broadcaster, daemon=True).start()


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


def _stream_worker_frames(stream_id: str, raw: bool = False):
    """
    MJPEG generator that reads from one StreamWorker's frame buffer.
    Multiple simultaneous clients for the same stream_id share the same
    worker — the worker writes frames once; all clients read that copy.
    """
    while True:
        worker = stream_manager.get_worker(stream_id) if stream_manager else None
        if worker is None:
            # Unknown or not-yet-started stream — wait and retry rather than
            # erroring, in case the worker is still starting up.
            time.sleep(0.5)
            continue

        frame = worker.get_frame(raw=raw)
        if frame is None:
            time.sleep(0.1)
            continue

        _, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, WebConfig.STREAM_QUALITY]
        )
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1.0 / WebConfig.STREAM_FPS)


@app.route('/video_feed')
@app.route('/video_feed/<stream_id>')
def video_feed(stream_id=None):
    return Response(
        _stream_worker_frames(stream_id or primary_stream_id, raw=False),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/video_feed/raw')
@app.route('/video_feed/<stream_id>/raw')
def video_feed_raw(stream_id=None):
    """Clean camera feed — no skeleton or bounding-box overlay."""
    return Response(
        _stream_worker_frames(stream_id or primary_stream_id, raw=True),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def get_stats():
    agg = stream_manager.aggregate_stats() if stream_manager else {
        'total_frames': 0, 'violence_detections': 0, 'alerts_triggered': 0,
        'current_fps': 0, 'is_running': False,
    }
    primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
    uptime = None
    if primary and primary.stats['start_time']:
        uptime = str(datetime.now() - datetime.fromtimestamp(primary.stats['start_time'])).split('.')[0]
    return jsonify({
        'total_frames': agg['total_frames'],
        'violence_detections': agg['violence_detections'],
        'alerts_triggered': agg['alerts_triggered'],
        'current_fps': agg['current_fps'],
        'uptime': uptime,
        'is_running': agg['is_running'],
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
    detector = primary.detector if primary else None
    if request.method == 'GET':
        return jsonify({
            'video_source': primary.source_url if primary else None,
            'violence_threshold': detector.violence_threshold if detector else 0.6,
            'use_yolo': detector.use_yolo if detector else True,
            'warmup_frames': detector.warmup_frames if detector else 30,
        })
    data = request.json or {}
    if detector and 'violence_threshold' in data:
        detector.violence_threshold = data['violence_threshold']
    return jsonify({'status': 'updated'})


@app.route('/api/start', methods=['POST'])
def start_detection():
    _, err = require_manage_role()
    if err:
        return err
    if not primary_stream_id:
        return jsonify({'error': 'No primary stream configured'}), 400
    worker = stream_manager.get_worker(primary_stream_id)
    if worker is None or not worker._running:
        session = get_session()
        try:
            stream = session.query(Stream).filter_by(stream_id=primary_stream_id).first()
            source_url = stream.source_url if stream else primary_stream_id
        finally:
            session.close()
        stream_manager.start_stream(primary_stream_id, source_url)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    _, err = require_manage_role()
    if err:
        return err
    if primary_stream_id:
        stream_manager.stop_stream(primary_stream_id)
    return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    _, err = require_manage_role()
    if err:
        return err
    primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
    if primary:
        primary.stats = {
            'total_frames': 0,
            'violence_detections': 0,
            'alerts_triggered': 0,
            'start_time': time.time() if primary._running else None,
            'current_fps': 0,
        }
        if primary.detector:
            primary.detector.reset()
    return jsonify({'status': 'reset'})


@app.route('/api/snapshot')
def snapshot():
    primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
    frame = primary.get_frame() if primary else None
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return jsonify({'error': 'No frame available'}), 404


@app.route('/health')
def health():
    primary = stream_manager.get_worker(primary_stream_id) if stream_manager and primary_stream_id else None
    return jsonify({
        'status': 'healthy',
        'detector_loaded': primary is not None and primary.detector is not None,
        'is_running': primary is not None and primary._running,
    })


# ==================== MAIN ====================

def create_app(model_path: str = None, source=0, use_yolo: bool = True):
    initialize_detector(model_path, source, use_yolo)
    return app


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False,
               use_reloader: bool = False):
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader,
        allow_unsafe_werkzeug=True,  # dev server; fine for an FYP demo
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Violence Detection Web Dashboard')
    parser.add_argument('--model', type=str)
    parser.add_argument('--source', default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--no-yolo', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    initialize_detector(args.model, source, not args.no_yolo)
    run_server(args.host, args.port, args.debug)
