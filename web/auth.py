"""
Authentication Blueprint
========================
JWT-based authentication for the Violence Detection API.

Users are seeded to match the jBoilerplate example-users.ts so the
frontend can log in without a separate registration flow.
"""

import hashlib
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# ---------------------------------------------------------------------------
# Hardcoded users — mirror of jBoilerplate src/constants/example-users.ts
# In production, replace with a proper users table.
# ---------------------------------------------------------------------------

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


_USERS = {
    'superadmin@example.com': {
        'id': '1',
        'fullname': 'Super Admin',
        'email': 'superadmin@example.com',
        'password_hash': _hash('superadmin123'),
        'user_type': 'superadmin',
    },
    'admin@example.com': {
        'id': '2',
        'fullname': 'Admin User',
        'email': 'admin@example.com',
        'password_hash': _hash('admin123'),
        'user_type': 'admin',
    },
    'user@example.com': {
        'id': '3',
        'fullname': 'Regular User',
        'email': 'user@example.com',
        'password_hash': _hash('user123'),
        'user_type': 'user',
    },
}

# Roles that can manage the detection system
_MANAGE_ROLES = {'superadmin', 'admin'}


def _user_to_public(user: dict) -> dict:
    """Return user dict safe to send to the client."""
    return {
        'id': user['id'],
        'fullname': user['fullname'],
        'email': user['email'],
        'user_type': user['user_type'],
        'avatar': '',
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate and return a JWT access token."""
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    user = _USERS.get(email)
    if not user or user['password_hash'] != _hash(password):
        return jsonify({'message': 'Invalid email or password'}), 401

    identity = {
        'id': user['id'],
        'email': user['email'],
        'user_type': user['user_type'],
        'fullname': user['fullname'],
    }
    access_token = create_access_token(identity=identity)

    return jsonify({
        'access_token': access_token,
        'user': _user_to_public(user),
    })


@auth_bp.route('/me', methods=['GET', 'POST'])
@jwt_required()
def me():
    """Return current user info from the JWT."""
    identity = get_jwt_identity()
    return jsonify({
        'id': identity['id'],
        'fullname': identity['fullname'],
        'email': identity['email'],
        'user_type': identity['user_type'],
        'avatar': '',
        'created_at': '',
        'updated_at': '',
    })


# ---------------------------------------------------------------------------
# Helper: check manage permission (used by app.py routes)
# ---------------------------------------------------------------------------

def require_manage_role():
    """Return (identity, None) or (None, error_response) for management routes."""
    from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
    try:
        verify_jwt_in_request()
        identity = get_jwt_identity()
        if identity.get('user_type') not in _MANAGE_ROLES:
            return None, (jsonify({'message': 'Insufficient role'}), 403)
        return identity, None
    except Exception as exc:
        return None, (jsonify({'message': f'Unauthorized: {exc}'}), 401)
