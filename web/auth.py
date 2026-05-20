"""
Authentication Blueprint
========================
JWT-based authentication backed by the Users table.

seed_demo_users() is called at app startup to ensure the default
accounts (matching jBoilerplate example-users.ts) always exist.
"""

from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    verify_jwt_in_request,
)
from werkzeug.security import generate_password_hash, check_password_hash

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

        # Stamp last_login
        user.last_login = datetime.utcnow()
        session.commit()

        identity = {
            'id':        user.id,
            'email':     user.email,
            'user_type': user.role,      # matches jBoilerplate User.user_type
            'fullname':  user.username,
        }
        access_token = create_access_token(identity=identity)

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


@auth_bp.route('/me', methods=['GET', 'POST'])
@jwt_required()
def me():
    """Return current user info from the JWT."""
    identity = get_jwt_identity()
    return jsonify({
        'id':         str(identity['id']),
        'fullname':   identity['fullname'],
        'email':      identity['email'],
        'user_type':  identity['user_type'],
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
    Returns (identity, None) on success, (None, error_response) on failure.
    """
    try:
        verify_jwt_in_request()
        identity = get_jwt_identity()
        if identity.get('user_type') not in MANAGE_ROLES:
            return None, (jsonify({'message': 'Insufficient role'}), 403)
        return identity, None
    except Exception as exc:
        return None, (jsonify({'message': f'Unauthorized: {exc}'}), 401)
