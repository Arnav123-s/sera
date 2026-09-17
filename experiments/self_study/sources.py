"""Bounded read-only arXiv access; retrieved material never becomes a command."""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
ID = re.compile(r"(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v[1-9]\d*)?\Z")


def valid_url(url):
    p = urlparse(url)
    if (p.scheme != "https" or p.hostname not in {"arxiv.org", "export.arxiv.org"}
            or p.username or p.password or p.port not in (None, 443)
            or not p.path.startswith(("/abs/", "/html/", "/pdf/", "/api/query"))):
        raise ValueError("Only public HTTPS arXiv research endpoints are admitted")
    return url


class Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        valid_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Maths(HTMLParser):
    def __init__(self):
        super().__init__()
        self.formulas = []

    def handle_starttag(self, tag, attrs):
        if tag == "math":
            data = dict(attrs)
            if data.get("alttext"):
                self.formulas.append(data["alttext"])


class Arxiv:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(ROOT / "runs"):
            raise ValueError("Source cache must remain under the local runs directory")
        self.directory.mkdir(parents=True, exist_ok=True)

    def fetch(self, url, *, refresh=False):
        valid_url(url)
        key = hashlib.sha256(url.encode()).hexdigest()
        record = self.directory / f"{key}.json"
        if record.exists() and not refresh:
            saved = json.loads(record.read_text(encoding="utf-8"))
            sha = saved.get("sha256", "")
            if (not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                    or saved.get("url") != url or saved.get("body") != sha + ".body"):
                raise ValueError("Cached source identity or body path changed")
            body = (self.directory / saved["body"]).resolve()
            if body.parent != self.directory:
                raise ValueError("Cached source body escaped the cache")
            raw = body.read_bytes()
            if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
                raise ValueError("Cached source bytes changed")
            return saved, raw
        # One shared cache lock prevents concurrent requests from this adapter.
        with (self.directory / "request.lock").open("x") as owned_lock:
            try:
                rate = self.directory / "last-request.json"
                last = json.loads(rate.read_text())["time"] if rate.exists() else 0
                time.sleep(min(3.1, max(0., 3.1 - (time.time() - last))))
                start = time.perf_counter()
                rate.write_text(json.dumps({"time": time.time()}))
                event = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                         "read_only": True, "commands_executed": 0}
                try:
                    request = Request(url, headers={"User-Agent": "SERA-Research/0.1 (+https://github.com/Arnav123-s/sera)"})
                    with build_opener(Redirect()).open(request, timeout=25) as response:
                        valid_url(response.url)
                        raw = response.read(8 * 1024 * 1024 + 1)
                        if len(raw) > 8 * 1024 * 1024:
                            raise ValueError("Source exceeds the 8 MiB acquisition cap")
                        event.update(final_url=response.url, content_type=response.headers.get("Content-Type"),
                                     etag=response.headers.get("ETag"), status="received", bytes=len(raw))
                    sha = hashlib.sha256(raw).hexdigest()
                    body = self.directory / f"{sha}.body"
                    if not body.exists():
                        body.write_bytes(raw)
                    event.update(sha256=sha, body=body.name)
                except Exception as exc:
                    event.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    event["wall_seconds"] = time.perf_counter() - start
                    with (self.directory / "acquisitions.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event) + "\n")
                record.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
                return event, raw
            finally:
                owned_lock.close()
                (self.directory / "request.lock").unlink()

    def paper(self, identifier, kind="html", *, refresh=False):
        if not ID.fullmatch(identifier) or kind not in {"abs", "html", "pdf"}:
            raise ValueError("Use an arXiv paper identifier and abs/html/pdf")
        return self.fetch(f"https://arxiv.org/{kind}/{identifier}", refresh=refresh)

    def search(self, query, limit=3, *, refresh=False):
        if not isinstance(query, str) or not 1 <= len(query) <= 256 or not 1 <= limit <= 5:
            raise ValueError("A bounded public research query and 1..5 results are required")
        return self.fetch("https://export.arxiv.org/api/query?" + urlencode(
            {"search_query": query, "start": 0, "max_results": limit, "sortBy": "relevance"}), refresh=refresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="math/9207222v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/SS-sources")
    args = parser.parse_args()
    client = Arxiv(args.cache)
    meta, raw = client.paper(args.id)
    parser = Maths()
    parser.feed(raw.decode("utf-8"))
    print(json.dumps({"source": meta, "math_count": len(parser.formulas),
                      "first_formulas": parser.formulas[:22]}, indent=2))


if __name__ == "__main__":
    main()
