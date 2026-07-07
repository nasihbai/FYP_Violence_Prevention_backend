"""
Authentication Blueprint
========================
JWT-based authentication backed by the Users table.

seed_demo_users() is called at app startup to ensure the default
accounts (matching jBoilerplate example-users.ts) always exist.
"""

from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt,
    get_jwt_identity,
    verify_jwt_in_request,
)
from werkzeug.security import generate_password_hash, check_password_hash

from config import EmailConfig
from .email_utils import generate_verification_token, send_verification_email

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Roles allowed to start/stop/reset detection
MANAGE_ROLES = {'superadmin', 'admin'}


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def seed_demo_users():
    """
    Insert default demo accounts if they don't already exist.
    Mirrors jBoilerplate src/constants/example-users.ts so the
    frontend can log in without a separate registration step.
    """
    from database.db import get_session
    from database.models import User

    demo = [
        dict(username='superadmin', email='superadmin@example.com',
             password='superadmin123', role='superadmin'),
        dict(username='admin',      email='admin@example.com',
             password='admin123',      role='admin'),
        dict(username='user',       email='user@example.com',
             password='user123',       role='user'),
    ]

    session = get_session()
    try:
        for u in demo:
            exists = session.query(User).filter_by(email=u['email']).first()
            if not exists:
                session.add(User(
                    username=u['username'],
                    email=u['email'],
                    password=generate_password_hash(u['password']),
                    role=u['role'],
                    is_active=True,
                    is_verified=True,  # demo accounts must not be blocked by the verification gate
                ))
        session.commit()
    except Exception as exc:
        session.rollback()
        import logging
        logging.getLogger(__name__).warning(f'seed_demo_users failed: {exc}')
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate and return a JWT access token."""
    from database.db import get_session
    from database.models import User

    data = request.get_json(silent=True) or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    session = get_session()
    try:
        user = session.query(User).filter_by(email=email, is_active=True).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'message': 'Invalid email or password'}), 401

        if not user.is_verified:
            return jsonify({'message': 'Please verify your email before logging in.'}), 403

        # Stamp last_login
        user.last_login = datetime.utcnow()
        session.commit()

        # flask-jwt-extended 4.6+ requires the identity (the JWT "sub"
        # claim) to be a string. The user id is the identity; everything
        # else rides along as additional claims, read back via get_jwt().
        access_token = create_access_token(
            identity=str(user.id),
            additional_claims={
                'email':     user.email,
                'user_type': user.role,   # matches jBoilerplate User.user_type
                'fullname':  user.username,
            },
        )

        return jsonify({
            'access_token': access_token,
            'user': {
                'id':        str(user.id),
                'fullname':  user.username,
                'email':     user.email,
                'user_type': user.role,
                'avatar':    '',
            },
        })
    finally:
        session.close()


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Create an unverified account and email a verification link.
    Body:     { username, email, password }
    Response: 201 { id, username, email, message }
              422 { errors: { <field>: [msg] } }  (missing fields / duplicate username or email)
    Does NOT log the user in — no access_token is returned (login is gated on is_verified).
    """
    from database.db import get_session
    from database.models import User

    data = request.get_json(silent=True) or {}
    errors = {}
    for field in ("username", "email", "password"):
        if not str(data.get(field, "")).strip():
            errors[field] = [f"{field} is required"]
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

        token = generate_verification_token()
        user = User(
            username=username,
            email=email,
            password=generate_password_hash(data["password"]),
            role="user",
            is_active=True,
            is_verified=False,
            verification_token=token,
            verification_token_expires_at=(
                datetime.utcnow() + timedelta(hours=EmailConfig.VERIFICATION_TOKEN_TTL_HOURS)
            ),
        )
        session.add(user)
        session.commit()

        send_verification_email(user.email, user.username, token)

        return jsonify({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "message": "Registration successful. Please check your email to verify your account.",
        }), 201
    finally:
        session.close()


@auth_bp.route('/verify-email', methods=['POST'])
def verify_email():
    """
    Body:     { token }
    Response: 200 { message, email }
              400 { errors: { token: ["Invalid verification token"] } }
              400 { errors: { token: ["Verification link has expired"] } }
    """
    from database.db import get_session
    from database.models import User

    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    if not token:
        return jsonify({"errors": {"token": ["Token is required"]}}), 400

    session = get_session()
    try:
        user = session.query(User).filter_by(verification_token=token).first()
        if not user:
            return jsonify({"errors": {"token": ["Invalid verification token"]}}), 400
        if not user.verification_token_expires_at or user.verification_token_expires_at < datetime.utcnow():
            return jsonify({"errors": {"token": ["Verification link has expired"]}}), 400

        user.is_verified = True
        user.verification_token = None
        user.verification_token_expires_at = None
        session.commit()

        return jsonify({"message": "Email verified successfully. You can now log in.", "email": user.email})
    finally:
        session.close()


@auth_bp.route('/resend-verification', methods=['POST'])
def resend_verification():
    """
    Body:     { email }
    Response: 200 { message }  (also 200 if already verified — informational, not an error)
              422 { errors: { email: ["No account found with that email"] } }
    """
    from database.db import get_session
    from database.models import User

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"errors": {"email": ["Email is required"]}}), 422

    session = get_session()
    try:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return jsonify({"errors": {"email": ["No account found with that email"]}}), 422
        if user.is_verified:
            return jsonify({"message": "Your email is already verified — you can log in."})

        token = generate_verification_token()
        user.verification_token = token
        user.verification_token_expires_at = (
            datetime.utcnow() + timedelta(hours=EmailConfig.VERIFICATION_TOKEN_TTL_HOURS)
        )
        session.commit()
        send_verification_email(user.email, user.username, token)
        return jsonify({"message": "Verification email resent."})
    finally:
        session.close()


@auth_bp.route('/me', methods=['GET', 'POST'])
@jwt_required()
def me():
    """Return current user info from the JWT."""
    user_id = get_jwt_identity()      # string user id (the "sub" claim)
    claims = get_jwt()                # additional claims set at login
    return jsonify({
        'id':         str(user_id),
        'fullname':   claims.get('fullname', ''),
        'email':      claims.get('email', ''),
        'user_type':  claims.get('user_type', ''),
        'avatar':     '',
        'created_at': '',
        'updated_at': '',
    })


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Stateless logout. JWT is held client-side and dropped by the FE;
    nothing to invalidate server-side until refresh-token rotation lands
    (Phase 6 / hardening). Returning a body so the FE can confirm the
    round-trip succeeded.
    """
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Helper used by protected app.py routes
# ---------------------------------------------------------------------------

def require_manage_role():
    """
    Verify JWT and check for admin/superadmin role.
    Returns (claims, None) on success, (None, error_response) on failure.
    `claims` carries user_type / email / fullname; the user id is in
    get_jwt_identity().
    """
    try:
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get('user_type') not in MANAGE_ROLES:
            return None, (jsonify({'message': 'Insufficient role'}), 403)
        return claims, None
    except Exception as exc:
        return None, (jsonify({'message': f'Unauthorized: {exc}'}), 401)
