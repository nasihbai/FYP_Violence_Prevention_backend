"""
SQLAlchemy ORM models — School Violence Prevention System
=========================================================
5 tables matching the project data dictionary:
  Users → Streams → Incidents → Alerts
                 → Detection_Logs
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean,
    ForeignKey, JSON, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from .db import Base


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class User(Base):
    """System user accounts with role-based access control."""
    __tablename__ = 'users'

    id          = Column(Integer, primary_key=True, autoincrement=True)
    username    = Column(String(50),  unique=True, nullable=False)
    email       = Column(String(100), unique=True, nullable=False)
    password    = Column(String(255), nullable=False)   # werkzeug pbkdf2 hash
    role        = Column(String(20),  nullable=False, default='user')
    is_active   = Column(Boolean,     nullable=False, default=True)
    created_at  = Column(DateTime,    nullable=False, default=datetime.utcnow)
    last_login  = Column(DateTime,    nullable=True)

    # Relationships
    incidents_created    = relationship('Incident', back_populates='creator',
                                        foreign_keys='Incident.created_by')
    alerts_acknowledged  = relationship('Alert',    back_populates='acknowledger',
                                        foreign_keys='Alert.acknowledged_by')

    def to_dict(self):
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'role':       self.role,
            'is_active':  self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat()  if self.last_login  else None,
        }


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class Stream(Base):
    """Camera / video stream configuration."""
    __tablename__ = 'streams'

    id         = Column(Integer,      primary_key=True, autoincrement=True)
    stream_id  = Column(String(50),   unique=True, nullable=False)   # e.g. "CAM_01"
    name       = Column(String(100),  nullable=False)
    source_url = Column(String(255),  nullable=False)                # RTSP / webcam index
    location   = Column(String(100),  nullable=True)
    is_active  = Column(Boolean,      nullable=False, default=True)

    # Relationships
    incidents      = relationship('Incident',     back_populates='stream',
                                   foreign_keys='Incident.stream_id')
    detection_logs = relationship('DetectionLog', back_populates='stream',
                                   foreign_keys='DetectionLog.stream_id',
                                   cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id':         self.id,
            'stream_id':  self.stream_id,
            'name':       self.name,
            'source_url': self.source_url,
            'location':   self.location,
            'is_active':  self.is_active,
        }


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

class Incident(Base):
    """
    Core table. One row per violence/threat event that crossed the
    alert confidence threshold.
    """
    __tablename__ = 'incidents'

    id              = Column(Integer,      primary_key=True, autoincrement=True)
    incident_code   = Column(String(50),   unique=True, nullable=False)  # INC-2026-0042
    stream_id       = Column(String(50),   ForeignKey('streams.stream_id'), nullable=False)
    type            = Column(String(20),   nullable=False)               # threatening | violent
    confidence      = Column(Float,        nullable=False)
    timestamp       = Column(DateTime,     nullable=False, default=datetime.utcnow)
    location        = Column(String(100),  nullable=True)
    screenshot_path = Column(String(255),  nullable=True)
    video_path      = Column(String(255),  nullable=True)
    duration_seconds = Column(Integer,     nullable=True)
    notes           = Column(Text,         nullable=True)
    severity        = Column(String(20),   nullable=False, default='medium')  # low/medium/high/critical
    status          = Column(String(20),   nullable=False, default='open')    # open/investigating/resolved/false_positive
    created_by      = Column(Integer,      ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    created_at      = Column(DateTime,     nullable=False, default=datetime.utcnow)
    updated_at      = Column(DateTime,     nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stream   = relationship('Stream', back_populates='incidents', foreign_keys=[stream_id])
    creator  = relationship('User',   back_populates='incidents_created', foreign_keys=[created_by])
    alerts   = relationship('Alert',  back_populates='incident', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id':             self.id,
            'incident_code':  self.incident_code,
            'stream_id':      self.stream_id,
            'type':           self.type,
            'confidence':     self.confidence,
            'timestamp':      self.timestamp.isoformat(),
            'location':       self.location,
            'screenshot_path': self.screenshot_path,
            'video_path':     self.video_path,
            'severity':       self.severity,
            'status':         self.status,
            'notes':          self.notes,
            'created_by':     self.created_by,
            'created_at':     self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alert(Base):
    """
    Notification record per incident.
    One incident can fire multiple alert channels (sound, email, webhook).
    Cascade-deletes when the parent incident is removed.
    """
    __tablename__ = 'alerts'

    id              = Column(Integer,  primary_key=True, autoincrement=True)
    incident_id     = Column(Integer,  ForeignKey('incidents.id', ondelete='CASCADE'), nullable=False)
    type            = Column(String(20), nullable=False)   # threatening | violent
    confidence      = Column(Float,    nullable=False)
    timestamp       = Column(DateTime, nullable=False, default=datetime.utcnow)
    acknowledged    = Column(Boolean,  nullable=False, default=False)
    acknowledged_by = Column(Integer,  ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    dismissed       = Column(Boolean,  nullable=False, default=False)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    incident     = relationship('Incident', back_populates='alerts')
    acknowledger = relationship('User',     back_populates='alerts_acknowledged',
                                foreign_keys=[acknowledged_by])

    def to_dict(self):
        return {
            'id':              self.id,
            'incident_id':     self.incident_id,
            'type':            self.type,
            'confidence':      self.confidence,
            'timestamp':       self.timestamp.isoformat(),
            'acknowledged':    self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'dismissed':       self.dismissed,
            'severity':        self.incident.severity if self.incident else None,
            'camera_id':       self.incident.stream_id if self.incident else None,
        }


# ---------------------------------------------------------------------------
# Detection Logs
# ---------------------------------------------------------------------------

class DetectionLog(Base):
    """
    High-volume append-only log. One row per processed frame batch per camera.
    Used for analytics and performance monitoring — not for incident reporting.
    """
    __tablename__ = 'detection_logs'

    id                 = Column(Integer,  primary_key=True, autoincrement=True)
    stream_id          = Column(String(50), ForeignKey('streams.stream_id', ondelete='CASCADE'), nullable=False)
    timestamp          = Column(DateTime, nullable=False, default=datetime.utcnow)
    person_count       = Column(Integer,  nullable=True, default=0)
    detections         = Column(JSON,     nullable=True)  # JSONB in PostgreSQL, TEXT in SQLite
    processing_time_ms = Column(Float,    nullable=True)
    created_at         = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    stream = relationship('Stream', back_populates='detection_logs', foreign_keys=[stream_id])

    def to_dict(self):
        return {
            'id':                 self.id,
            'stream_id':          self.stream_id,
            'timestamp':          self.timestamp.isoformat() if self.timestamp else None,
            'person_count':       self.person_count,
            'processing_time_ms': self.processing_time_ms,
            'detections':         self.detections,
        }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Setting(Base):
    """
    Key-value configuration store, grouped by namespace.

    Backs the superadmin settings pages (namespace 'app' / 'email' / 'seo')
    so their values persist server-side instead of in browser localStorage.
    `value` holds a JSON-encoded string so any JSON type (string, bool,
    number, object) round-trips.
    """
    __tablename__ = 'settings'

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    namespace  = Column(String(50),  nullable=False)
    key        = Column(String(100), nullable=False)
    value      = Column(Text,        nullable=True)   # JSON-encoded
    updated_at = Column(DateTime,    nullable=False,
                        default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('namespace', 'key', name='uq_setting_namespace_key'),
    )

    def to_dict(self):
        return {
            'namespace':  self.namespace,
            'key':        self.key,
            'value':      self.value,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
