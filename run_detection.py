"""
Main Violence Detection Runner
==============================
Run real-time violence detection with YOLO + LSTM.

Usage:
    # Webcam (default)
    python run_detection.py

    # Specific camera
    python run_detection.py --source 1

    # Video file
    python run_detection.py --source path/to/video.mp4

    # RTSP stream
    python run_detection.py --source rtsp://192.168.1.100:554/stream

    # Without YOLO (single person)
    python run_detection.py --no-yolo

    # With web dashboard
    python run_detection.py --web
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import VideoConfig, ModelConfig, AlertConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_detection(args):
    """Run detection in OpenCV window mode."""
    from core.detection_engine import ThreadSafeDetector, VideoProcessor
    from alerts.alert_system import AlertManager

    # Determine model path
    model_path = args.model
    if model_path is None:
        root = Path(__file__).parent
        candidates = [
            root / 'models' / 'violence_lstm_dataset.h5',   # 309-feature, proven pipeline
            root / 'lstm-model.h5',
            root / 'models' / 'violence_lstm_rwf2000.h5',
            root / 'models' / 'violence_lstm_enhanced.h5',
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            model_path = str(found)
        else:
            logger.warning("No model found. Running without LSTM classification.")
            model_path = None

    logger.info(f"Model path: {model_path}")
    logger.info(f"Video source: {args.source}")
    logger.info(f"YOLO enabled: {not args.no_yolo}")

    # Initialize detector
    detector = ThreadSafeDetector(
        lstm_model_path=model_path,
        use_yolo=not args.no_yolo,
        violence_threshold=args.threshold,
        warmup_frames=args.warmup
    )

    # Initialize alert manager
    alert_manager = AlertManager(
        cooldown_seconds=args.alert_cooldown,
        sound_file=str(Path(__file__).parent / 'alert.wav') if not args.no_sound else None,
        screenshot_dir=str(Path(__file__).parent / 'alerts' / 'screenshots')
    )
    alert_manager.start()

    # Define violence callback
    def on_violence(result):
        detections = [
            {
                'person_id': d.person_id,
                'confidence': d.confidence,
                'class_name': d.class_name,
                'bbox': d.bbox
            }
            for d in result.detections if d.is_violent
        ]
        max_conf = max(d.confidence for d in result.detections if d.is_violent) if detections else 0
        alert_manager.trigger_alert(
            frame=result.frame,
            detections=detections,
            confidence=max_conf,
            source=str(args.source)
        )

    # Create video processor
    processor = VideoProcessor(
        source=args.source,
        detector=detector,
        on_violence_detected=on_violence
    )

    logger.info("Starting detection... Press 'q' to quit.")

    try:
        processor.run(display=True, window_name='Violence Detection System')
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()
        alert_manager.stop()


def run_web_dashboard(args):
    """Run detection with web dashboard."""
    from web.app import initialize_detector, run_server

    # Determine model path
    model_path = args.model
    if model_path is None:
        root = Path(__file__).parent
        candidates = [
            root / 'models' / 'violence_lstm_dataset.h5',   # 309-feature, proven pipeline
            root / 'lstm-model.h5',
            root / 'models' / 'violence_lstm_rwf2000.h5',
            root / 'models' / 'violence_lstm_enhanced.h5',
        ]
        model_path = next((str(p) for p in candidates if p.exists()), None)

    logger.info(f"Starting web dashboard on http://{args.host}:{args.port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Source: {args.source}")

    initialize_detector(model_path, args.source, not args.no_yolo)
    run_server(host=args.host, port=args.port, debug=args.debug)


def main():
    parser = argparse.ArgumentParser(
        description='Violence Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py                          # Webcam
  python run_detection.py --source 1               # Camera 1
  python run_detection.py --source video.mp4       # Video file
  python run_detection.py --source rtsp://ip/stream  # RTSP stream
  python run_detection.py --web                    # Web dashboard
  python run_detection.py --no-yolo                # Single person mode
        """
    )

    # Video source
    parser.add_argument(
        '--source', '-s',
        default=0,
        help='Video source: camera index (0,1,...), file path, or URL'
    )

    # Model
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to LSTM model file'
    )

    # Detection options
    parser.add_argument(
        '--no-yolo',
        action='store_true',
        help='Disable YOLO multi-person detection'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.6,
        help='Violence detection threshold (0-1)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=30,
        help='Warmup frames to skip'
    )

    # Alert options
    parser.add_argument(
        '--no-sound',
        action='store_true',
        help='Disable sound alerts'
    )
    parser.add_argument(
        '--alert-cooldown',
        type=int,
        default=10,
        help='Seconds between alerts'
    )

    # Web dashboard
    parser.add_argument(
        '--web',
        action='store_true',
        help='Run with web dashboard'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Web server host'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web server port'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # Convert source to int if it's a camera index
    try:
        args.source = int(args.source)
    except ValueError:
        pass

    # Run appropriate mode
    if args.web:
        run_web_dashboard(args)
    else:
        run_detection(args)


if __name__ == '__main__':
    main()
