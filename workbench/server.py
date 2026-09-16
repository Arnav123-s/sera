"""Loopback application host; numerical requests use the existing bounded worker."""

import argparse
import csv
import hashlib
import io
import json
import secrets
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .storage import Store, encoded

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent/"static"


class Application:
    def __init__(self, workspace, *, watch=True):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_relative_to(ROOT/"runs"):
            raise ValueError("Use an isolated workspace under runs/")
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace/"inbox").mkdir(exist_ok=True)
        (self.workspace/"outputs").mkdir(exist_ok=True)
        (self.workspace/"requests").mkdir(exist_ok=True)
        self.lock, self.token = threading.Lock(), secrets.token_urlsafe(32)
        self.watch, self.stop = watch, threading.Event()
        self.seen, self.task_errors = {}, {}
        self.last_activity, self.busy = "Ready", False

    def state(self):
        record = Store(self.workspace).read()
        ledger = json.loads((ROOT/"runs/v3-batch-001/budget.json").read_text())
        return {"state": None if record is None else record["view"], "watch": self.watch,
                "language_ready": (ROOT/"research-continuation/21_grounded_language/qualification.json").exists(),
                "inbox": str(self.workspace/"inbox"), "outputs": str(self.workspace/"outputs"),
                "remaining_seconds": ledger["remaining_seconds"], "busy": self.busy,
                "last_activity": self.last_activity, "errors": dict(self.task_errors),
                "session_token": self.token, "server_time": datetime.now(timezone.utc).isoformat()}

    def execute(self, payload):
        if (ROOT/"runs/v3-batch-001/active.lock").exists():
            raise ValueError("Another bounded local job is using the worker. This task can run when it finishes.")
        if not self.lock.acquire(blocking=False):
            raise ValueError("A task is running. Please wait for it to finish.")
        self.busy = True
        try:
            identity = str(uuid.uuid4())
            directory = self.workspace/"requests"/identity
            directory.mkdir()
            payload = {**payload, "request_id": payload.get("request_id", identity)}
            request, response = directory/"request.json", directory/"response.json"
            request.write_bytes(encoded(payload))
            seconds = 90 if payload.get("operation") == "learn_language" else 20
            command = [sys.executable, "-X", "utf8", str(ROOT/"scripts/run_workbench_bounded.py"),
                       "--seconds", str(seconds), "--output", str(directory/"worker"), "--module", "workbench.worker", "--",
                       str(request), str(response), "--workspace", str(self.workspace)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=seconds+20)
            (directory/"supervisor.log").write_text(result.stdout+result.stderr, encoding="utf-8")
            if result.returncode or not response.exists():
                log = directory/"worker/process.log"
                detail = log.read_text(encoding="utf-8").splitlines()[-1] if log.exists() and log.stat().st_size else "Resource allowance or an owned job prevented this operation"
                raise ValueError(detail[:400])
            answer = json.loads(response.read_text())
            self.last_activity = payload["operation"]+" completed at "+datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            if payload["operation"] == "learn_csv":
                self.export_report(payload["name"], answer["result"]["report"])
            return answer
        finally:
            self.busy = False
            self.lock.release()

    def export_report(self, name, report):
        # Name has already been validated by the numerical worker.
        folder = self.workspace/"outputs"/name
        folder.mkdir(exist_ok=True)
        identity = uuid.uuid4().hex[:12]
        (folder/(identity+".json")).write_bytes(encoded(report))
        text = (f"# {name}\n\nUpdated {datetime.now(timezone.utc).isoformat()}\n\n"
                f"Read {report['rows']} observations through {report['last_timestamp']}. "
                f"Selected {report['method']} using observed prior prediction errors.\n\n"
                f"Causal historical MAE: {report['mae']:.6g}; persistence MAE: {report['persistence_mae']:.6g}. "
                f"Flagged {report['flags']} surprising observations.\n\n"
                "| Future observation | Forecast |\n|---|---:|\n"+
                "".join(f"| +{r['step']} | {r['value']:.8g} |\n" for r in report["forecast"])+
                "\n"+report["assumptions"]+"\n")
        (folder/(identity+".md")).write_text(text, encoding="utf-8")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["future_observation", "forecast"])
        writer.writerows((r["step"], r["value"]) for r in report["forecast"])
        (folder/(identity+".csv")).write_text(buffer.getvalue(), encoding="utf-8")

    def watch_once(self):
        if not self.watch or self.busy or (ROOT/"runs/v3-batch-001/active.lock").exists():
            return
        for path in sorted((self.workspace/"inbox").glob("*.csv"))[:8]:
            if not path.resolve().is_relative_to(self.workspace/"inbox") or path.stat().st_size > 250_000:
                continue
            raw = path.read_bytes()
            fingerprint = hashlib.sha256(raw).hexdigest()
            if self.seen.get(path.name) == fingerprint:
                continue
            try:
                current = Store(self.workspace).read()
                if current is not None:
                    # A restart may see the same file; the worker's unchanged check is a final guard.
                    previous = current.get("streams", {}).get(path.stem)
                    saved_digest = self.workspace/"inbox"/(path.name+".accepted.sha256")
                    if previous is not None and saved_digest.exists() and saved_digest.read_text() == fingerprint:
                        self.seen[path.name] = fingerprint
                        continue
                self.execute({"operation": "learn_csv", "name": path.stem, "csv": raw.decode("utf-8-sig")})
                (self.workspace/"inbox"/(path.name+".accepted.sha256")).write_text(fingerprint)
                self.task_errors.pop(path.name, None)
            except (ValueError, UnicodeError, OSError, subprocess.TimeoutExpired) as error:
                self.task_errors[path.name] = str(error)
            self.seen[path.name] = fingerprint
            break

    def watch_loop(self):
        while not self.stop.wait(3):
            self.watch_once()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin", f"http://127.0.0.1:{port}") in {"http://"+host for host in hosts}

        def do_GET(self):
            if not self.allowed():
                self.reply(403, encoded({"error": "Local requests only"}))
                return
            path = urlparse(self.path).path
            if path == "/api/state":
                self.reply(200, encoded(app.state()))
            elif path == "/api/export":
                value = Store(app.workspace).read()
                self.reply(200, encoded(value))
            elif path in ("/", "/app.js", "/style.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                kind = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[path]
                self.reply(200, (STATIC/name).read_bytes(), kind+"; charset=utf-8")
            else:
                self.reply(404, encoded({"error": "Not found"}))

        def do_POST(self):
            if not self.allowed() or self.headers.get("X-Sera-Session") != app.token:
                self.reply(403, encoded({"error": "Local session token required"}))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 300_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/api/watch":
                    if type(payload.get("enabled")) is not bool:
                        raise ValueError("Watch state must be true or false")
                    app.watch = payload["enabled"]
                    answer = app.state()
                elif self.path == "/api/execute":
                    answer = app.execute(payload)
                else:
                    raise ValueError("Unknown operation")
                self.reply(200, encoded(answer))
            except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                self.reply(400, encoded({"error": str(error)}))

        def log_message(self, message, *args):
            print(message % args, flush=True)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT/"runs/sera-workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-watch", action="store_true")
    args = parser.parse_args()
    app = Application(args.workspace, watch=not args.no_watch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    worker = threading.Thread(target=app.watch_loop, daemon=True)
    worker.start()
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "workspace": str(app.workspace)}), flush=True)
    try:
        server.serve_forever()
    finally:
        app.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
