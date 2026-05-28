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

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_jwt_extended import JWTManager

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WebConfig, VideoConfig, AlertConfig
from core.detection_engine import ThreadSafeDetector, FrameResult
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

detector: ThreadSafeDetector = None
video_source = None
is_running = False
current_frame = None
frame_lock = threading.Lock()

stats = {
    'total_frames': 0,
    'violence_detections': 0,
    'alerts_triggered': 0,
    'start_time': None,
    'current_fps': 0,
}


# ==================== HELPERS ====================

def initialize_detector(model_path: str = None, source=0, use_yolo: bool = True):
    """Initialise the detection system."""
    global detector, video_source

    video_source = source

    if model_path is None:
        root = Path(__file__).parent.parent
        # Prefer the 132-feature model that matches the current pipeline
        candidates = [
            root / 'models' / 'violence_lstm_dataset.h5',   # 309-feature, proven pipeline
            root / 'lstm-model.h5',
            root / 'models' / 'violence_lstm_rwf2000.h5',
            root / 'models' / 'violence_lstm_enhanced.h5',
        ]
        model_path = next((str(p) for p in candidates if p.exists()), None)

    detector = ThreadSafeDetector(
        lstm_model_path=model_path if model_path and Path(model_path).exists() else None,
        use_yolo=use_yolo,
        use_scene_classifier=True,
        use_person_classifier=False,
    )
    detector.start()
    logger.info(f"Detector initialised — source: {source}")


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


def _save_incident(det, screenshot_path: str = None) -> dict | None:
    """
    Persist a violence detection event.
    Creates: Stream (if needed) → Incident → Alert.
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

        stream = _get_or_create_stream(session, video_source)

        # Generate human-readable incident code
        year = datetime.utcnow().year
        count = session.query(Incident).count() + 1
        incident_code = f"INC-{year}-{count:04d}"

        incident = Incident(
            incident_code=incident_code,
            stream_id=stream.stream_id,
            type=alert_type,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            location=stream.location,
            screenshot_path=screenshot_path,
            severity=severity,
            status='open',
        )
        session.add(incident)
        session.flush()

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

LOG_INTERVAL = 30   # write a DetectionLog row every N frames


def generate_frames():
    """Generator for MJPEG video streaming. Saves incidents and emits SocketIO events."""
    global current_frame, is_running, stats

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return

    is_running = True
    stats['start_time'] = datetime.now()

    # Resolve stream_id once so we don't recalculate every frame
    stream_id = f"CAM_{video_source}" if isinstance(video_source, int) \
        else Path(str(video_source)).stem.upper()

    try:
        while is_running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = detector.process_frame(frame)
            stats['total_frames'] += 1
            stats['current_fps'] = result.fps
            processing_ms = (time.time() - t0) * 1000

            if result.has_violence:
                stats['violence_detections'] += 1
                for det in result.detections:
                    if det.is_violent:
                        stats['alerts_triggered'] += 1
                        incident_data = _save_incident(det)
                        if incident_data:
                            socketio.emit('violence_alert', incident_data)

            # Write detection log every LOG_INTERVAL frames
            if stats['total_frames'] % LOG_INTERVAL == 0:
                _write_detection_log(stream_id, result, processing_ms)

            annotated = detector.draw_results(frame, result)

            with frame_lock:
                current_frame = annotated.copy()

            _, buffer = cv2.imencode(
                '.jpg', annotated,
                [cv2.IMWRITE_JPEG_QUALITY, WebConfig.STREAM_QUALITY]
            )
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(1.0 / WebConfig.STREAM_FPS)
    finally:
        cap.release()
        is_running = False


def _stats_broadcaster():
    """Background thread: push live stats to all connected clients every second."""
    while True:
        uptime = None
        if stats['start_time']:
            uptime = str(datetime.now() - stats['start_time']).split('.')[0]

        socketio.emit('stats_update', {
            'total_frames': stats['total_frames'],
            'violence_detections': stats['violence_detections'],
            'alerts_triggered': stats['alerts_triggered'],
            'current_fps': round(stats['current_fps'], 1),
            'uptime': uptime or '00:00:00',
            'is_running': is_running,
        })
        time.sleep(1)


# Start stats broadcaster as a daemon thread
threading.Thread(target=_stats_broadcaster, daemon=True).start()


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def get_stats():
    uptime = None
    if stats['start_time']:
        uptime = str(datetime.now() - stats['start_time']).split('.')[0]
    return jsonify({
        'total_frames': stats['total_frames'],
        'violence_detections': stats['violence_detections'],
        'alerts_triggered': stats['alerts_triggered'],
        'current_fps': round(stats['current_fps'], 1),
        'uptime': uptime,
        'is_running': is_running,
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify({
            'video_source': video_source,
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
    global is_running
    if not is_running:
        threading.Thread(
            target=lambda: list(generate_frames()),
            daemon=True
        ).start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    _, err = require_manage_role()
    if err:
        return err
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    _, err = require_manage_role()
    if err:
        return err
    global stats
    stats = {
        'total_frames': 0,
        'violence_detections': 0,
        'alerts_triggered': 0,
        'start_time': datetime.now() if is_running else None,
        'current_fps': 0,
    }
    if detector:
        detector.reset()
    return jsonify({'status': 'reset'})


@app.route('/api/snapshot')
def snapshot():
    with frame_lock:
        if current_frame is not None:
            _, buffer = cv2.imencode('.jpg', current_frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return jsonify({'error': 'No frame available'}), 404


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'is_running': is_running,
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
