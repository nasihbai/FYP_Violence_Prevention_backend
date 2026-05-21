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

from datetime import datetime
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request

from database.db import get_session
from database.models import Alert, Incident, Stream

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
    Returns (identity, None) on success, (None, error_response) on failure.
    """
    try:
        verify_jwt_in_request()
        identity = get_jwt_identity()
        if identity.get("user_type") not in _MANAGE_ROLES:
            return None, (jsonify({"errors": {"_": ["Insufficient role"]}}), 403)
        return identity, None
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
    identity = get_jwt_identity() or {}
    session = get_session()
    try:
        alert = session.query(Alert).get(alert_id)
        if not alert:
            return jsonify({"errors": {"_": ["Alert not found"]}}), 404
        alert.acknowledged = True
        try:
            alert.acknowledged_by = int(identity.get("id")) if identity.get("id") else None
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
