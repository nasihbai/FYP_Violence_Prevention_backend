"""
CRUD API blueprint — detector-independent routes.

Lives separate from web/app.py so it can be registered in both:
  - The full detector server (web/app.py)
  - The slim auth-only server (auth_only_server.py)
without dragging in TensorFlow/MediaPipe.

All routes require a valid JWT. Routes that modify state under "manage"
control (currently: PATCH /api/incidents/<id>) require superadmin or admin role.

Response shapes:
  - List endpoints:   { items: [...], total: N, limit, offset }
  - Detail endpoints: the entity itself (sometimes with embedded sub-objects)
  - Errors:           { errors: { <field>: ["msg"] } } — matches the FE shape
"""

import json
from datetime import datetime, timedelta
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity, verify_jwt_in_request
from werkzeug.security import generate_password_hash
from sqlalchemy import func

from database.db import get_session
from database.models import Alert, Incident, Stream, User, Setting

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Mirrors MANAGE_ROLES in web/auth.py — kept duplicated to avoid an import
# cycle. If you change one, change the other.
_MANAGE_ROLES = {"superadmin", "admin"}

_ALLOWED_INCIDENT_STATUSES = {"open", "investigating", "resolved", "false_positive"}
_ALLOWED_INCIDENT_SEVERITIES = {"low", "medium", "high", "critical"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_manage_role():
    """
    Verify JWT and check the caller's role is admin/superadmin.
    Returns (claims, None) on success, (None, error_response) on failure.

    The JWT identity (sub) is a string user id; role / email / fullname
    ride along as additional claims, read via get_jwt().
    """
    return _require_role(_MANAGE_ROLES)


def _require_superadmin():
    """Verify JWT and require the superadmin role specifically."""
    return _require_role({"superadmin"})


def _require_role(allowed):
    """
    Shared role gate. `allowed` is a set of acceptable user_type values.
    Returns (claims, None) on success, (None, error_response) on failure.
    """
    try:
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("user_type") not in allowed:
            return None, (jsonify({"errors": {"_": ["Insufficient role"]}}), 403)
        return claims, None
    except Exception as exc:
        return None, (jsonify({"errors": {"_": [f"Unauthorized: {exc}"]}}), 401)


def _parse_int(name, default, mn=None, mx=None):
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    if mn is not None:
        v = max(mn, v)
    if mx is not None:
        v = min(mx, v)
    return v


def _parse_bool(name):
    raw = request.args.get(name)
    if raw is None:
        return None
    return raw.lower() in ("true", "1", "yes", "on")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@api_bp.route("/alerts", methods=["GET"])
@jwt_required()
def list_alerts():
    """
    List alerts. Query params:
      ?status=open|investigating|resolved|false_positive  (filters by parent incident)
      ?severity=low|medium|high|critical                  (filters by parent incident)
      ?acknowledged=true|false
      ?limit=N  (default 50, max 200)
      ?offset=M (default 0)
    """
    limit = _parse_int("limit", 50, 1, 200)
    offset = _parse_int("offset", 0, 0)
    status = request.args.get("status")
    severity = request.args.get("severity")
    acknowledged = _parse_bool("acknowledged")

    session = get_session()
    try:
        q = session.query(Alert).join(Incident, Alert.incident_id == Incident.id)

        if status:
            q = q.filter(Incident.status == status)
        if severity:
            q = q.filter(Incident.severity == severity)
        if acknowledged is not None:
            q = q.filter(Alert.acknowledged == acknowledged)

        total = q.count()
        items = (
            q.order_by(Alert.timestamp.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return jsonify({
            "items": [a.to_dict() for a in items],
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    finally:
        session.close()


@api_bp.route("/alerts/<int:alert_id>/acknowledge", methods=["POST"])
@jwt_required()
def acknowledge_alert(alert_id):
    # JWT identity (sub) is the string user id
    user_id = get_jwt_identity()
    session = get_session()
    try:
        alert = session.query(Alert).get(alert_id)
        if not alert:
            return jsonify({"errors": {"_": ["Alert not found"]}}), 404
        alert.acknowledged = True
        try:
            alert.acknowledged_by = int(user_id) if user_id else None
        except (TypeError, ValueError):
            alert.acknowledged_by = None
        alert.acknowledged_at = datetime.utcnow()
        session.commit()
        return jsonify(alert.to_dict())
    finally:
        session.close()


@api_bp.route("/alerts/<int:alert_id>/dismiss", methods=["POST"])
@jwt_required()
def dismiss_alert(alert_id):
    session = get_session()
    try:
        alert = session.query(Alert).get(alert_id)
        if not alert:
            return jsonify({"errors": {"_": ["Alert not found"]}}), 404
        alert.dismissed = True
        session.commit()
        return jsonify(alert.to_dict())
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@api_bp.route("/incidents", methods=["GET"])
@jwt_required()
def list_incidents():
    """
    List incidents. Query params:
      ?status=open|investigating|resolved|false_positive
      ?severity=low|medium|high|critical
      ?limit=N (default 50, max 200)
      ?offset=M (default 0)
    """
    limit = _parse_int("limit", 50, 1, 200)
    offset = _parse_int("offset", 0, 0)
    status = request.args.get("status")
    severity = request.args.get("severity")

    session = get_session()
    try:
        q = session.query(Incident)
        if status:
            q = q.filter(Incident.status == status)
        if severity:
            q = q.filter(Incident.severity == severity)

        total = q.count()
        items = (
            q.order_by(Incident.timestamp.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return jsonify({
            "items": [i.to_dict() for i in items],
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    finally:
        session.close()


@api_bp.route("/incidents/<int:incident_id>", methods=["GET"])
@jwt_required()
def get_incident(incident_id):
    """Fetch one incident with its embedded alerts."""
    session = get_session()
    try:
        incident = session.query(Incident).get(incident_id)
        if not incident:
            return jsonify({"errors": {"_": ["Incident not found"]}}), 404
        payload = incident.to_dict()
        payload["alerts"] = [a.to_dict() for a in incident.alerts]
        return jsonify(payload)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Streams (cameras)
# ---------------------------------------------------------------------------

@api_bp.route("/streams", methods=["GET"])
@jwt_required()
def list_streams():
    """
    List camera streams. Query params:
      ?is_active=true|false   (omit for all)
    Returns the list envelope: { items, total, limit, offset }.
    """
    is_active = _parse_bool("is_active")
    session = get_session()
    try:
        q = session.query(Stream)
        if is_active is not None:
            q = q.filter(Stream.is_active == is_active)
        items = q.order_by(Stream.stream_id.asc()).all()
        return jsonify({
            "items": [s.to_dict() for s in items],
            "total": len(items),
            "limit": len(items),
            "offset": 0,
        })
    finally:
        session.close()


@api_bp.route("/streams/<int:stream_pk>", methods=["GET"])
@jwt_required()
def get_stream(stream_pk):
    session = get_session()
    try:
        stream = session.query(Stream).get(stream_pk)
        if not stream:
            return jsonify({"errors": {"_": ["Stream not found"]}}), 404
        return jsonify(stream.to_dict())
    finally:
        session.close()


@api_bp.route("/streams", methods=["POST"])
def create_stream():
    """
    Register a new camera. Manage role required.
    Body: { stream_id, name, source_url, location?, is_active? }
    """
    _, err = _require_manage_role()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    errors = {}
    for field in ("stream_id", "name", "source_url"):
        if not str(data.get(field, "")).strip():
            errors[field] = [f"{field} is required"]
    if errors:
        return jsonify({"errors": errors}), 422

    session = get_session()
    try:
        existing = session.query(Stream).filter_by(stream_id=data["stream_id"].strip()).first()
        if existing:
            return jsonify({"errors": {"stream_id": ["A stream with this ID already exists"]}}), 422

        stream = Stream(
            stream_id=data["stream_id"].strip(),
            name=data["name"].strip(),
            source_url=str(data["source_url"]).strip(),
            location=(data.get("location") or "").strip() or None,
            is_active=bool(data.get("is_active", True)),
        )
        session.add(stream)
        session.commit()
        return jsonify(stream.to_dict()), 201
    finally:
        session.close()


@api_bp.route("/streams/<int:stream_pk>", methods=["PATCH"])
def update_stream(stream_pk):
    """
    Update a camera. Manage role required.
    Body may include: name, source_url, location, is_active.
    stream_id is immutable (it's the FK target for incidents/logs).
    """
    _, err = _require_manage_role()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    session = get_session()
    try:
        stream = session.query(Stream).get(stream_pk)
        if not stream:
            return jsonify({"errors": {"_": ["Stream not found"]}}), 404

        if "name" in data:
            if not str(data["name"]).strip():
                return jsonify({"errors": {"name": ["name cannot be empty"]}}), 422
            stream.name = data["name"].strip()
        if "source_url" in data:
            if not str(data["source_url"]).strip():
                return jsonify({"errors": {"source_url": ["source_url cannot be empty"]}}), 422
            stream.source_url = str(data["source_url"]).strip()
        if "location" in data:
            stream.location = (data.get("location") or "").strip() or None
        if "is_active" in data:
            stream.is_active = bool(data["is_active"])

        session.commit()
        return jsonify(stream.to_dict())
    finally:
        session.close()


@api_bp.route("/streams/<int:stream_pk>", methods=["DELETE"])
def delete_stream(stream_pk):
    """
    Soft-delete a camera (set is_active=false). Manage role required.
    Hard delete is avoided — Stream.stream_id is the FK target for
    incidents and detection logs; removing the row would orphan history.
    """
    _, err = _require_manage_role()
    if err:
        return err

    session = get_session()
    try:
        stream = session.query(Stream).get(stream_pk)
        if not stream:
            return jsonify({"errors": {"_": ["Stream not found"]}}), 404
        stream.is_active = False
        session.commit()
        return jsonify(stream.to_dict())
    finally:
        session.close()


@api_bp.route("/incidents/<int:incident_id>", methods=["PATCH"])
def update_incident(incident_id):
    """
    Update an incident's status / severity / notes. Manage role required.
    Body: { status?, severity?, notes? }
    """
    _, err = _require_manage_role()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    session = get_session()
    try:
        incident = session.query(Incident).get(incident_id)
        if not incident:
            return jsonify({"errors": {"_": ["Incident not found"]}}), 404

        if "status" in data:
            if data["status"] not in _ALLOWED_INCIDENT_STATUSES:
                return jsonify({"errors": {"status": [f"Invalid status: {data['status']}"]}}), 422
            incident.status = data["status"]

        if "severity" in data:
            if data["severity"] not in _ALLOWED_INCIDENT_SEVERITIES:
                return jsonify({"errors": {"severity": [f"Invalid severity: {data['severity']}"]}}), 422
            incident.severity = data["severity"]

        if "notes" in data:
            incident.notes = data["notes"]

        incident.updated_at = datetime.utcnow()
        session.commit()
        return jsonify(incident.to_dict())
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
# All user routes are superadmin-only. DELETE is a soft delete — User.id is
# the FK target for Incident.created_by and Alert.acknowledged_by, so a hard
# delete would orphan history.

_ALLOWED_USER_ROLES = {"superadmin", "admin", "user"}


@api_bp.route("/users", methods=["GET"])
def list_users():
    """List users. Superadmin only. Query params: ?role= ?is_active=."""
    _, err = _require_superadmin()
    if err:
        return err

    role = request.args.get("role")
    is_active = _parse_bool("is_active")

    session = get_session()
    try:
        q = session.query(User)
        if role:
            q = q.filter(User.role == role)
        if is_active is not None:
            q = q.filter(User.is_active == is_active)
        items = q.order_by(User.id.asc()).all()
        return jsonify({
            "items": [u.to_dict() for u in items],
            "total": len(items),
            "limit": len(items),
            "offset": 0,
        })
    finally:
        session.close()


@api_bp.route("/users/<int:user_pk>", methods=["GET"])
def get_user(user_pk):
    _, err = _require_superadmin()
    if err:
        return err
    session = get_session()
    try:
        user = session.query(User).get(user_pk)
        if not user:
            return jsonify({"errors": {"_": ["User not found"]}}), 404
        return jsonify(user.to_dict())
    finally:
        session.close()


@api_bp.route("/users", methods=["POST"])
def create_user():
    """
    Create a user. Superadmin only.
    Body: { username, email, password, role?, is_active? }
    """
    _, err = _require_superadmin()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    errors = {}
    for field in ("username", "email", "password"):
        if not str(data.get(field, "")).strip():
            errors[field] = [f"{field} is required"]
    role = (data.get("role") or "user").strip()
    if role not in _ALLOWED_USER_ROLES:
        errors["role"] = [f"Invalid role: {role}"]
    if errors:
        return jsonify({"errors": errors}), 422

    username = data["username"].strip()
    email = data["email"].strip().lower()

    session = get_session()
    try:
        if session.query(User).filter_by(username=username).first():
            return jsonify({"errors": {"username": ["Username already taken"]}}), 422
        if session.query(User).filter_by(email=email).first():
            return jsonify({"errors": {"email": ["Email already registered"]}}), 422

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(data["password"]),
            role=role,
            is_active=bool(data.get("is_active", True)),
        )
        session.add(user)
        session.commit()
        return jsonify(user.to_dict()), 201
    finally:
        session.close()


@api_bp.route("/users/<int:user_pk>", methods=["PATCH"])
def update_user(user_pk):
    """
    Update a user. Superadmin only.
    Body may include: username, email, password, role, is_active.
    A password, if present, is re-hashed.
    """
    _, err = _require_superadmin()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    session = get_session()
    try:
        user = session.query(User).get(user_pk)
        if not user:
            return jsonify({"errors": {"_": ["User not found"]}}), 404

        if "username" in data:
            new_username = str(data["username"]).strip()
            if not new_username:
                return jsonify({"errors": {"username": ["username cannot be empty"]}}), 422
            clash = session.query(User).filter(
                User.username == new_username, User.id != user_pk
            ).first()
            if clash:
                return jsonify({"errors": {"username": ["Username already taken"]}}), 422
            user.username = new_username

        if "email" in data:
            new_email = str(data["email"]).strip().lower()
            if not new_email:
                return jsonify({"errors": {"email": ["email cannot be empty"]}}), 422
            clash = session.query(User).filter(
                User.email == new_email, User.id != user_pk
            ).first()
            if clash:
                return jsonify({"errors": {"email": ["Email already registered"]}}), 422
            user.email = new_email

        if "role" in data:
            if data["role"] not in _ALLOWED_USER_ROLES:
                return jsonify({"errors": {"role": [f"Invalid role: {data['role']}"]}}), 422
            user.role = data["role"]

        if "is_active" in data:
            user.is_active = bool(data["is_active"])

        if data.get("password"):
            user.password = generate_password_hash(data["password"])

        session.commit()
        return jsonify(user.to_dict())
    finally:
        session.close()


@api_bp.route("/users/<int:user_pk>", methods=["DELETE"])
def delete_user(user_pk):
    """Soft-delete a user (set is_active=false). Superadmin only."""
    _, err = _require_superadmin()
    if err:
        return err
    session = get_session()
    try:
        user = session.query(User).get(user_pk)
        if not user:
            return jsonify({"errors": {"_": ["User not found"]}}), 404
        user.is_active = False
        session.commit()
        return jsonify(user.to_dict())
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@api_bp.route("/analytics/incidents", methods=["GET"])
def incident_analytics():
    """
    Aggregated incident stats for the analytics page. Manage role.
    Returns counts grouped by day (last 30d), severity, status, type, camera.
    """
    _, err = _require_manage_role()
    if err:
        return err

    session = get_session()
    try:
        total = session.query(Incident).count()

        def _grouped(column, allowed):
            """Count incidents grouped by `column`, seeded with allowed keys at 0."""
            out = {k: 0 for k in allowed}
            for value, count in (
                session.query(column, func.count(Incident.id)).group_by(column).all()
            ):
                if value in out:
                    out[value] = count
            return out

        by_severity = _grouped(Incident.severity, _ALLOWED_INCIDENT_SEVERITIES)
        by_status = _grouped(Incident.status, _ALLOWED_INCIDENT_STATUSES)
        by_type = _grouped(Incident.type, ("violent", "threatening"))

        # by camera — every camera that has incidents, busiest first
        by_camera = [
            {"stream_id": sid, "count": count}
            for sid, count in (
                session.query(Incident.stream_id, func.count(Incident.id))
                .group_by(Incident.stream_id)
                .order_by(func.count(Incident.id).desc())
                .all()
            )
        ]

        # by day — last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        by_day = [
            {"date": str(day), "count": count}
            for day, count in (
                session.query(func.date(Incident.timestamp), func.count(Incident.id))
                .filter(Incident.timestamp >= cutoff)
                .group_by(func.date(Incident.timestamp))
                .order_by(func.date(Incident.timestamp))
                .all()
            )
        ]

        return jsonify({
            "total": total,
            "by_day": by_day,
            "by_severity": by_severity,
            "by_status": by_status,
            "by_type": by_type,
            "by_camera": by_camera,
        })
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Settings (key-value config, grouped by namespace)
# ---------------------------------------------------------------------------
# Backs the superadmin settings pages. Manage role required.
# Values are JSON-encoded in the DB so any JSON type round-trips.

_ALLOWED_SETTING_NAMESPACES = {"app", "email", "seo"}


def _decode_setting_value(raw):
    """DB stores JSON-encoded strings; decode, tolerating legacy plain text."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw  # legacy / non-JSON value — return as-is


@api_bp.route("/settings/<namespace>", methods=["GET"])
def get_settings(namespace):
    """Return all settings in a namespace as a flat { key: value } map."""
    _, err = _require_manage_role()
    if err:
        return err
    if namespace not in _ALLOWED_SETTING_NAMESPACES:
        return jsonify({"errors": {"_": [f"Unknown namespace: {namespace}"]}}), 404

    session = get_session()
    try:
        rows = session.query(Setting).filter_by(namespace=namespace).all()
        return jsonify({r.key: _decode_setting_value(r.value) for r in rows})
    finally:
        session.close()


@api_bp.route("/settings/<namespace>", methods=["PUT"])
def put_settings(namespace):
    """
    Upsert settings in a namespace. Body is a flat { key: value } map;
    each entry is created or updated. Keys not in the body are left alone.
    """
    _, err = _require_manage_role()
    if err:
        return err
    if namespace not in _ALLOWED_SETTING_NAMESPACES:
        return jsonify({"errors": {"_": [f"Unknown namespace: {namespace}"]}}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"errors": {"_": ["Body must be a JSON object of key/value pairs"]}}), 422

    session = get_session()
    try:
        for key, value in data.items():
            encoded = json.dumps(value)
            row = session.query(Setting).filter_by(namespace=namespace, key=str(key)).first()
            if row:
                row.value = encoded
                row.updated_at = datetime.utcnow()
            else:
                session.add(Setting(namespace=namespace, key=str(key), value=encoded))
        session.commit()

        rows = session.query(Setting).filter_by(namespace=namespace).all()
        return jsonify({r.key: _decode_setting_value(r.value) for r in rows})
    finally:
        session.close()


@api_bp.route("/settings/<namespace>/<key>", methods=["DELETE"])
def delete_setting(namespace, key):
    """Remove a single setting key from a namespace."""
    _, err = _require_manage_role()
    if err:
        return err
    if namespace not in _ALLOWED_SETTING_NAMESPACES:
        return jsonify({"errors": {"_": [f"Unknown namespace: {namespace}"]}}), 404

    session = get_session()
    try:
        row = session.query(Setting).filter_by(namespace=namespace, key=key).first()
        if not row:
            return jsonify({"errors": {"_": ["Setting not found"]}}), 404
        session.delete(row)
        session.commit()
        return jsonify({"ok": True})
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test / demo helper
# ---------------------------------------------------------------------------

@api_bp.route("/test/fire-alert", methods=["POST"])
@jwt_required()
def fire_test_alert():
    """
    Insert a synthetic incident + alert and emit a 'violence_alert' SocketIO
    event. Lets the FE / demo trigger an alert on cue without real violence
    to detect. Body (all optional):
      { type?: "violent"|"threatening", severity?: low|medium|high|critical,
        confidence?: 0.0-1.0 }
    """
    data = request.get_json(silent=True) or {}
    kind = data.get("type", "violent")
    severity = data.get("severity", "high")
    try:
        confidence = float(data.get("confidence", 0.85))
    except (TypeError, ValueError):
        confidence = 0.85

    session = get_session()
    try:
        stream = session.query(Stream).filter_by(stream_id="CAM_DEMO").first()
        if not stream:
            stream = Stream(
                stream_id="CAM_DEMO",
                name="Demo Camera",
                source_url="0",
                location="Lab / FYP demo",
                is_active=True,
            )
            session.add(stream)
            session.commit()

        now = datetime.utcnow()
        incident = Incident(
            incident_code=f"INC-LIVE-{int(now.timestamp())}",
            stream_id=stream.stream_id,
            type=kind,
            confidence=confidence,
            timestamp=now,
            location=stream.location,
            severity=severity,
            status="open",
            notes="Manually triggered via /api/test/fire-alert",
        )
        session.add(incident)
        session.flush()

        alert = Alert(
            incident_id=incident.id,
            type=kind,
            confidence=confidence,
            timestamp=now,
            acknowledged=False,
            dismissed=False,
        )
        session.add(alert)
        session.commit()

        payload = alert.to_dict()
        # Emit over SocketIO. The instance is registered on the Flask app
        # by flask-socketio's init_app — grab it via current_app so this
        # blueprint route doesn't need a direct import of web/app.py.
        socketio = current_app.extensions.get("socketio")
        if socketio is not None:
            socketio.emit("violence_alert", payload)
        return jsonify(payload), 201
    finally:
        session.close()
