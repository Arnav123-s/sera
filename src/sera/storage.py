"""Crash-resistant JSON artifacts and a local append-only experiment journal."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class Journal:
    """Single-machine evidence ledger; this is not a hostile-code security boundary."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL,
                    kind TEXT NOT NULL, body TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, hash TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    dataset_id TEXT PRIMARY KEY, round_index INTEGER UNIQUE NOT NULL
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def append(self, kind, payload):
        body = canonical(payload)
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
            previous = last[0] if last else "0" * 64
            event_hash = digest([previous, now, kind, body])
            db.execute(
                "INSERT INTO events(timestamp,kind,body,previous_hash,hash) VALUES(?,?,?,?,?)",
                (now, kind, body, previous, event_hash),
            )
        return event_hash

    def reserve_evaluation(self, dataset_id):
        """Consume before evaluating: failed attempts cannot silently reuse an exposed suite."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT MAX(round_index) FROM evaluations").fetchone()[0]
            round_index = 0 if row is None else row + 1
            try:
                db.execute("INSERT INTO evaluations VALUES (?,?)", (dataset_id, round_index))
            except sqlite3.IntegrityError as error:
                raise ValueError("Evaluation suite already consumed") from error
        return round_index

    def events(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,timestamp,kind,body,previous_hash,hash FROM events ORDER BY id"
            ).fetchall()
        return [
            dict(zip(("id", "timestamp", "kind", "body", "previous_hash", "hash"), row))
            for row in rows
        ]

    def verify(self):
        previous = "0" * 64
        for event in self.events():
            expected = digest([previous, event["timestamp"], event["kind"], event["body"]])
            if event["previous_hash"] != previous or event["hash"] != expected:
                raise ValueError("Experiment journal integrity failure")
            previous = event["hash"]
        return True
