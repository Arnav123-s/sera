"""Use the retained continuing owner for attributed reading, requests and exact work."""

import argparse
import json

import torch

from experiments.stream_curriculum.model import encode
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.credit import decode
from sera.session_state import model_identity
from workbench.storage import Store

from .common import DECISIONS, OUT, ROOT, RUN, read
from .runtime import GrowthSession
from .study import load_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("integrate", "status", "read", "request", "sum", "integral"))
    parser.add_argument("--text")
    parser.add_argument("--input")
    parser.add_argument("--coefficients", default="2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = ROOT / "runs/sera-growth-live"
    path.mkdir(exist_ok=True)
    with lock(path):
        store = Store(path)
        if args.action == "integrate":
            if store.read() is not None:
                raise FileExistsError("Preserve an existing integrated owner")
            final, audit = read(OUT / "final.json"), read(OUT / "audit.json")
            if not final["knowledge_admitted"] or not audit["protected_equal"]:
                raise ValueError("Independent knowledge and retention gates required")
            session = GrowthSession(read(RUN / "parent.json"), saved=load_state(
                RUN / f"{final['selected']}-{DECISIONS:04d}.pt"))
            if model_identity(session.owner) != audit["owner"]:
                raise ValueError("Audited owner differs from integrated owner")
            store.commit(session.snapshot(), None)
            result = {"status": "INTEGRATED", "owner": audit["owner"], "store": str(path)}
        else:
            saved = store.read()
            if saved is None:
                raise ValueError("Integrate the independently admitted continuation first")
            session = GrowthSession(saved["parent"], saved=decode(saved["state"]))
            if model_identity(session.owner) != saved["owner"]:
                raise ValueError("Changed saved owner")
            if args.action == "request":
                if not args.text:
                    raise ValueError("A human request is required")
                with torch.no_grad():
                    intent, slots = session.owner.request_logits(encode([args.text]))
                vocabulary = session.owner.stream_config["vocabulary"]
                result = {"text": args.text, "intent": vocabulary["intents"][int(intent.argmax(-1)[0])],
                          "slots": [vocabulary["tags"][i] for i in slots.argmax(-1)[0].tolist()],
                          "status": "INTERPRETED_REQUEST"}
            elif args.action == "read":
                request = read(args.input)
                if not request.get("source") or not request.get("question"):
                    raise ValueError("An attributed source question is required")
                with torch.no_grad():
                    x, mask = session.owner.reading_features([request])
                    selected = int(session.owner.reading_logits(x, mask).argmax(-1)[0])
                result = {"status": "ATTRIBUTED_SENTENCE_SELECTION", "question": request["question"],
                          "answer": request["sentences"][selected]["text"], "source": request["source"]}
            elif args.action in ("sum", "integral"):
                from experiments.self_study.algebra import check, independent
                p = args.coefficients.split(",")
                proposed = session.owner.study_maps[args.action].propose(p)
                verified = check(args.action, p, proposed)["accepted"] and independent(args.action, p, proposed)
                result = {"status": "CERTIFIED_ALGEBRA" if verified else "NEEDS_ACQUISITION",
                          "operator": args.action, "input": p, "coefficients": proposed, "independent_check": verified}
            else:
                result = {"status": "READY", "goal": session.goal, "decisions": len(session.events),
                          "default_controller": read(OUT / "procedure.json")["default"],
                          "strands": sorted({e["skill"] for e in session.events}),
                          "open_goals": ["Continue unresolved attributed-reading corrections",
                                         "Ground missing concepts under the preserved source gate",
                                         "Extend independently verified composition frontiers"]}
            result["owner"] = model_identity(session.owner)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
