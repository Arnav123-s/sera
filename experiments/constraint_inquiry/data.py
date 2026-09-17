"""Supplied finite teaching grammar; roles are targets, never runtime parsing rules."""

import itertools

ENTITIES = ("orbit", "beacon", "marker", "anchor")
MODES = ("can", "imagine", "did", "avoid")
TEMPLATES = (
    ("can {a} reach {b}", "please {b} can be reached by {a}"),
    ("imagine {a} reaching {b}", "please imagine {b} being reached by {a}"),
    ("did {a} reach {b}", "please was {b} reached by {a}"),
    ("{a} must not reach {b}", "please {b} must not be reached by {a}"),
)
WORDS = ("<pad>",) + tuple(sorted(set(" ".join(
    t.format(a="orbit beacon", b="marker anchor") for pair in TEMPLATES for t in pair
).split())))
VOCAB = {w: i for i, w in enumerate(WORDS)}
LENGTH = 10
HELD = {(0, 1), (1, 0)}


def corpus(partition):
    result = []
    for a, b in itertools.permutations(range(4), 2):
        for mode, pair in enumerate(TEMPLATES):
            for template, text in enumerate(pair):
                if partition == "surface":
                    if template:
                        continue
                    text = "please " + text
                elif ((a, b) in HELD) != (partition == "pairs"):
                    continue
                result.append({"text": text.format(a=ENTITIES[a], b=ENTITIES[b]),
                               "target": [a, b, mode], "template": template})
    return result


def encode(texts):
    rows = []
    for text in texts:
        words = text.lower().strip().rstrip(".?").split()
        if not words or len(words) > LENGTH or any(w not in VOCAB for w in words):
            raise ValueError("Missing language meaning or unsupported length")
        if sum(w in ENTITIES for w in words) != 2 or len({w for w in words if w in ENTITIES}) != 2:
            raise ValueError("Two distinct named entities are required; ambiguous reference stays unresolved")
        rows.append([VOCAB[w] for w in words] + [0] * (LENGTH - len(words)))
    return rows
