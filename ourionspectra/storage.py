"""
Lightweight local persistence for past recovery runs.

Sessions are stored as one JSON file per run under <project root>/data/sessions/.
No database, no external dependencies — just enough to let the user reopen
the app and see/reload their history.
"""

import json
import os
import time
import uuid

from .config import APP_DIR

SESSIONS_DIR = os.path.join(APP_DIR, "data", "sessions")


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(session: dict) -> str:
    """Save a session dict to disk. Returns the session id, or None on failure."""
    _ensure_dir()
    session_id = session.get("id") or str(uuid.uuid4())[:8]
    session["id"] = session_id
    session.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(path, "w") as f:
            json.dump(session, f)
    except OSError:
        return None
    return session_id


def list_sessions() -> list:
    """Return metadata for all saved sessions, newest first."""
    _ensure_dir()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id"),
                "label": data.get("dataset", data.get("source_label", "Untitled")),
                "saved_at": data.get("saved_at", ""),
                "recovery_improvement": (
                    data.get("metrics", {}).get("recovery_improvement")
                    if data.get("metrics_reference_available")
                    else None
                ),
            })
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s["saved_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> dict:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path) as f:
        return json.load(f)


def delete_session(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
