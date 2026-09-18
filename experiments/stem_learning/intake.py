"""Acquire pinned human-authored textbook chapters; never execute source content."""

import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research-continuation/31_stem_acquisition"
RAW = ROOT / "runs/ST-sources-001"
PIN = "2300d5b8ca2571dacc6ec25ebef56ba8d49000a2"
BASE = f"https://raw.githubusercontent.com/openstax/osbooks-university-physics-bundle/{PIN}/"
CHAPTERS = {1: {"Newton's Laws of Motion", "Work and Kinetic Energy",
                "Potential Energy and Conservation of Energy", "Linear Momentum and Collisions"},
            2: {"Temperature and Heat", "Current and Resistance"}}


def main():
    RAW.mkdir(parents=True, exist_ok=False)
    records = []

    def get(path):
        started = time.perf_counter()
        with urllib.request.urlopen(BASE + path, timeout=25) as response:
            raw = response.read(8*1024*1024+1)
        if len(raw) > 8*1024*1024:
            raise ValueError("Textbook resource exceeds intake bound")
        target = RAW / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        records.append({"path": target.relative_to(ROOT).as_posix(), "url": BASE+path,
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                        "seconds": time.perf_counter()-started})
        (RAW / "receipt.json").write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")
        return raw

    get("LICENSE")
    get("README.md")
    ns = {"c": "http://cnx.rice.edu/collxml", "m": "http://cnx.rice.edu/mdml"}
    for volume, chapters in CHAPTERS.items():
        root = ET.fromstring(get(f"collections/university-physics-volume-{volume}.collection.xml"))
        for chapter in root.findall(".//c:subcollection", ns):
            title = chapter.findtext("m:title", "", ns)
            if title not in chapters:
                continue
            for module in chapter.findall(".//c:module", ns):
                name = module.attrib["document"]
                get(f"modules/{name}/index.cnxml")
    print(json.dumps({"resources": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
