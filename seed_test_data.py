"""
Synthetic incident / alert seeder — for Phase 2 FE testing.

Run once to populate the DB with a handful of demo incidents and alerts
so the alerts dashboard has something to display while the real detector
is unavailable (TF/MediaPipe dep wall).

Usage:
    python seed_test_data.py             # adds 6 demo incidents
    python seed_test_data.py --clear     # wipe demo data first
    python seed_test_data.py --count 20  # change the count

Idempotent on incident_code — re-running won't duplicate entries.
"""

import argparse
import random
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database.db import init_db, get_session
from database.models import Stream, Incident, Alert


_DEMO_PREFIX = "INC-DEMO-"
_TYPES = ["violent", "threatening"]
_SEVERITIES = ["low", "medium", "high", "critical"]
_STATUSES = ["open", "investigating", "resolved", "false_positive"]


def _ensure_demo_stream(session) -> str:
    """Ensure a CAM_DEMO stream exists so foreign keys are satisfied."""
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
        print(f"  + created stream CAM_DEMO")
    return stream.stream_id


def seed(count: int, clear: bool):
    init_db()
    session = get_session()
    try:
        if clear:
            # Delete demo alerts first (FK cascade), then incidents
            demo_incidents = (
                session.query(Incident)
                .filter(Incident.incident_code.like(f"{_DEMO_PREFIX}%"))
                .all()
            )
            for inc in demo_incidents:
                session.delete(inc)  # cascade removes child alerts
            session.commit()
            print(f"  - cleared {len(demo_incidents)} demo incident(s)")

        stream_id = _ensure_demo_stream(session)

        existing = {
            i.incident_code
            for i in session.query(Incident)
            .filter(Incident.incident_code.like(f"{_DEMO_PREFIX}%"))
            .all()
        }

        added = 0
        now = datetime.utcnow()
        for n in range(count):
            code = f"{_DEMO_PREFIX}{n + 1:04d}"
            if code in existing:
                continue

            severity = random.choice(_SEVERITIES)
            status = random.choice(_STATUSES)
            kind = random.choice(_TYPES)
            ts = now - timedelta(minutes=random.randint(0, 60 * 24 * 7))

            incident = Incident(
                incident_code=code,
                stream_id=stream_id,
                type=kind,
                confidence=round(random.uniform(0.60, 0.99), 3),
                timestamp=ts,
                location="Lab / FYP demo",
                severity=severity,
                status=status,
                notes=f"Synthetic test data for Phase 2 FE verification (run #{n + 1}).",
            )
            session.add(incident)
            session.flush()  # get incident.id

            alert = Alert(
                incident_id=incident.id,
                type=kind,
                confidence=incident.confidence,
                timestamp=ts,
                acknowledged=(status in ("resolved", "false_positive")),
                acknowledged_at=(ts + timedelta(minutes=2)) if status in ("resolved", "false_positive") else None,
                dismissed=(status == "false_positive"),
            )
            session.add(alert)
            added += 1

        session.commit()
        total = session.query(Incident).filter(Incident.incident_code.like(f"{_DEMO_PREFIX}%")).count()
        print(f"  + added {added} demo incident(s) (total demo rows: {total})")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6, help="how many demo incidents to add")
    parser.add_argument("--clear", action="store_true", help="wipe existing demo rows first")
    args = parser.parse_args()

    print(f"Seeding test data (count={args.count}, clear={args.clear}) ...")
    seed(args.count, args.clear)
    print("Done.")
