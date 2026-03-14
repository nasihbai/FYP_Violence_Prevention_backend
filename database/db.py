"""
Database connection management.
Defaults to SQLite for development.
Set DATABASE_URL env var for PostgreSQL:
  DATABASE_URL=postgresql://user:password@localhost/violence_db
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

_SessionFactory = None


def init_db(database_url: str = None):
    """
    Initialise the database engine and create all tables.
    Returns a session factory callable.
    """
    global _SessionFactory

    if database_url is None:
        database_url = os.environ.get(
            'DATABASE_URL',
            'sqlite:///violence_detection.db'
        )

    connect_args = {}
    if database_url.startswith('sqlite'):
        # SQLite needs check_same_thread=False when used from multiple threads
        connect_args['check_same_thread'] = False

    engine = create_engine(database_url, connect_args=connect_args)

    # Import models so their tables are registered on Base.metadata
    from . import models  # noqa: F401
    Base.metadata.create_all(engine)

    _SessionFactory = sessionmaker(bind=engine)
    logger.info(f"Database initialised: {database_url}")
    return _SessionFactory


def get_session():
    """Return a new database session. Caller is responsible for closing it."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _SessionFactory()
