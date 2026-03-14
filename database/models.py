"""
SQLAlchemy ORM models for the Violence Detection System.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from .db import Base


class Incident(Base):
    """
    Represents a single violence detection event.
    Written whenever the ML pipeline fires an alert.
    """
    __tablename__ = 'incidents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    person_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)
    bbox = Column(Text, nullable=True)           # JSON string: [x1, y1, x2, y2]
    screenshot_path = Column(String(512), nullable=True)
    severity = Column(String(20), nullable=False, default='medium')  # low / medium / high
    camera_id = Column(String(100), nullable=False, default='cam_0')

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'person_id': self.person_id,
            'confidence': self.confidence,
            'severity': self.severity,
            'camera_id': self.camera_id,
            'screenshot_path': self.screenshot_path,
        }
