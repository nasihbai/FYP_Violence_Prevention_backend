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
from database import init_db, Incident
from database.db import get_session
from auth import auth_bp, require_manage_role

logger = logging.getLogger(__name__)

# ==================== APP SETUP ====================

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = WebConfig.SECRET_KEY
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', WebConfig.SECRET_KEY)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # No expiry for dev; set timedelta in prod

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
jwt = JWTManager(app)

app.register_blueprint(auth_bp)

# Initialise database (SQLite by default; set DATABASE_URL env var for PostgreSQL)
init_db()

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
        lstm_model_path=model_path if Path(model_path).exists() else None,
        use_yolo=use_yolo,
    )
    detector.start()
    logger.info(f"Detector initialised — source: {source}")


def _save_incident(det, screenshot_path: str = None) -> dict | None:
    """Persist a violence detection event and return its dict representation."""
    session = get_session()
    try:
        confidence = float(det.confidence)
        if confidence > 0.85:
            severity = 'high'
        elif confidence > 0.70:
            severity = 'medium'
        else:
            severity = 'low'

        incident = Incident(
            timestamp=datetime.utcnow(),
            person_id=int(det.person_id) if det.person_id is not None else None,
            confidence=confidence,
            bbox=json.dumps(det.bbox) if det.bbox else None,
            screenshot_path=screenshot_path,
            severity=severity,
            camera_id=str(video_source),
        )
        session.add(incident)
        session.commit()
        return incident.to_dict()
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save incident: {exc}")
        return None
    finally:
        session.close()


# ==================== VIDEO / DETECTION ====================

def generate_frames():
    """Generator for MJPEG video streaming. Saves incidents and emits SocketIO events."""
    global current_frame, is_running, stats

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return

    is_running = True
    stats['start_time'] = datetime.now()

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = detector.process_frame(frame)
            stats['total_frames'] += 1
            stats['current_fps'] = result.fps

            if result.has_violence:
                stats['violence_detections'] += 1
                for det in result.detections:
                    if det.is_violent:
                        stats['alerts_triggered'] += 1
                        incident_data = _save_incident(det)
                        if incident_data:
                            socketio.emit('violence_alert', incident_data)

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


@app.route('/api/alerts')
def get_alerts():
    """Return incident history — requires valid JWT."""
    from flask_jwt_extended import verify_jwt_in_request
    try:
        verify_jwt_in_request()
    except Exception:
        return jsonify({'message': 'Unauthorized'}), 401

    limit = request.args.get('limit', 50, type=int)
    session = get_session()
    try:
        incidents = (
            session.query(Incident)
            .order_by(Incident.timestamp.desc())
            .limit(limit)
            .all()
        )
        return jsonify([i.to_dict() for i in incidents])
    finally:
        session.close()


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


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    socketio.run(app, host=host, port=port, debug=debug)


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
