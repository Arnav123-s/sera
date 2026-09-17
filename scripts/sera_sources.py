"""Read public arXiv sources with bounded requests, caching and provenance."""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.self_study.sources import Arxiv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("search", "paper"))
    p.add_argument("value", help="A public research search query or arXiv paper ID")
    p.add_argument("--kind", choices=("abs", "html", "pdf"), default="html")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--refresh", action="store_true", help="Fetch a current response; preserve content-addressed older bodies")
    args = p.parse_args()
    client = Arxiv(ROOT / "runs/SS-sources")
    if args.action == "search":
        receipt, raw = client.search(args.value, args.limit, refresh=args.refresh)
        ns = "{http://www.w3.org/2005/Atom}"
        entries = [{"id": e.findtext(ns + "id"), "title": e.findtext(ns + "title"),
                    "summary": e.findtext(ns + "summary")} for e in ET.fromstring(raw).findall(ns + "entry")]
        result = {"sources": entries, "receipt": receipt}
    else:
        receipt, _ = client.paper(args.value, args.kind, refresh=args.refresh)
        result = {"receipt": receipt, "local_file": str(client.directory / receipt["body"])}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
