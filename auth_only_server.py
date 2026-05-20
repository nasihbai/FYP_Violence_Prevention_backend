"""
Auth-only Flask server — Phase 1 verification helper.

Boots only what's needed to test the FE auth flow:
  - Flask + Flask-CORS + Flask-JWT-Extended
  - SQLAlchemy DB + demo user seeder
  - The /auth/* blueprint

Deliberately skips the detector pipeline so you don't have to fight
TensorFlow/MediaPipe/Protobuf dependency conflicts to verify login,
/auth/me, and /auth/logout work end-to-end with the Vue frontend.

Run:
    python auth_only_server.py

Endpoints:
    POST /auth/login           — body: { email, password }
    GET|POST /auth/me          — JWT required
    POST /auth/logout
    GET  /health               — sanity check

When you eventually fix the TF/MP install, switch back to:
    python run_detection.py --web

That uses the real entry point with the detector enabled.
"""

import os
import sys
import types
import logging
import importlib
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

# Pre-stub sys.modules['web'] BEFORE any `from web.auth import ...` happens.
# The real web/__init__.py eagerly imports web/app.py, which drags in
# TensorFlow/MediaPipe/OpenCV via core/detection_engine — exactly what
# we're trying to skip. By inserting a bare module object with __path__
# pointing at the web/ directory, `from web.auth import X` still resolves
# but the real __init__.py never runs.
_web_stub = types.ModuleType("web")
_web_stub.__path__ = [str(_REPO_ROOT / "web")]
sys.modules["web"] = _web_stub

from config import WebConfig
from database import init_db

_auth_module = importlib.import_module("web.auth")
auth_bp = _auth_module.auth_bp
seed_demo_users = _auth_module.seed_demo_users

_api_module = importlib.import_module("web.api")
api_bp = _api_module.api_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== APP SETUP ====================

app = Flask(__name__)
app.config["SECRET_KEY"] = WebConfig.SECRET_KEY
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", WebConfig.SECRET_KEY)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False  # No expiry for dev

# CORS — mirrors the production app's behavior
_cors_origins = WebConfig.CORS_ORIGINS
_cors_value = (
    "*"
    if _cors_origins.strip() == "*"
    else [o.strip() for o in _cors_origins.split(",") if o.strip()]
)
CORS(app, resources={r"/*": {"origins": _cors_value}}, supports_credentials=True)

jwt = JWTManager(app)
app.register_blueprint(auth_bp)
app.register_blueprint(api_bp)


# ==================== DB BOOT ====================

init_db()
seed_demo_users()
logger.info("DB initialized; demo users seeded")


# ==================== ERROR HANDLERS ====================
# Mirror the FE-friendly shape: { errors: { _: ["msg"] } }

@app.errorhandler(Exception)
def _handle_uncaught(e):
    code = getattr(e, "code", 500)
    msg = getattr(e, "description", None) or str(e) or "Internal Server Error"
    logger.exception("Unhandled error on %s %s", request.method, request.path)
    return jsonify({"errors": {"_": [msg]}}), code


@app.errorhandler(404)
def _handle_404(e):
    return (
        jsonify({"errors": {"_": [getattr(e, "description", None) or "Not Found"]}}),
        404,
    )


@app.errorhandler(405)
def _handle_405(e):
    return (
        jsonify({"errors": {"_": [getattr(e, "description", None) or "Method Not Allowed"]}}),
        405,
    )


# ==================== SMOKE ROUTES ====================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "auth-only"})


# ==================== MAIN ====================

if __name__ == "__main__":
    host = WebConfig.HOST
    port = WebConfig.PORT
    print()
    print("=" * 60)
    print(f"  Auth-only server  →  http://localhost:{port}")
    print("=" * 60)
    print("  Mode:     auth-only (no detector / TF / MediaPipe)")
    print("  Database: sqlite:///violence_detection.db")
    print("  CORS:     ", _cors_value)
    print()
    print("  Endpoints:")
    print("    POST  /auth/login                          body: { email, password }")
    print("    GET   /auth/me                             JWT required")
    print("    POST  /auth/logout")
    print("    GET   /api/alerts                          ?status&severity&acknowledged&limit&offset")
    print("    POST  /api/alerts/<id>/acknowledge")
    print("    POST  /api/alerts/<id>/dismiss")
    print("    GET   /api/incidents                       ?status&severity&limit&offset")
    print("    GET   /api/incidents/<id>")
    print("    PATCH /api/incidents/<id>                  manage role; body: { status?, severity?, notes? }")
    print("    GET   /health")
    print()
    print("  Demo users (seeded automatically):")
    print("    superadmin@example.com / superadmin123")
    print("    admin@example.com      / admin123")
    print("    user@example.com       / user123")
    print()
    print("  Ctrl+C to stop.")
    print("=" * 60)
    print()
    app.run(host=host, port=port, debug=False)
