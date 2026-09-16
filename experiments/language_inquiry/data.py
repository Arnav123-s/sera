"""Supplied bounded syntax; learned span semantics; no open-English claim."""
import itertools
import re

PHRASES = {
    -1: ("push left", "apply negative control", "not right but left"),
    0: ("wait", "apply zero control", "remain still"),
    1: ("push right", "apply positive control", "not left but right"),
}
FIELD_WORDS = {"position": ("position", "location"), "velocity": ("velocity", "angular speed")}
LESSON = "coast"
TEACHING = [(text, action+1) for action, texts in PHRASES.items() for text in texts]
TEACHING += [(text, 3+i) for i, texts in enumerate(FIELD_WORDS.values()) for text in texts]
WORDS = ["<pad>", "<unknown>"] + sorted({w for text, _ in TEACHING for w in text.split()} | {LESSON})
VOCAB = {w: i for i, w in enumerate(WORDS)}
TEMPLATES = (
    "predict the {field} of {entity} after {body}",
    "if {entity} does {body} then report its {field}",
    "after {body} report the {field} of {entity}",
)
WITHHELD = ((1, 0, -1), (-1, 0, 1), (1, -1, 1), (-1, 1, -1))
PATTERNS = (
    r"predict the (?P<field>.+?) of (?P<entity>[a-z]+) after (?P<body>.+)",
    r"if (?P<entity>[a-z]+) does (?P<body>.+) then report its (?P<field>.+)",
    r"after (?P<body>.+) report the (?P<field>.+?) of (?P<entity>[a-z]+)",
)


def encode(texts):
    result = []
    for text in texts:
        words = text.split()
        if not 1 <= len(words) <= 6:
            raise ValueError("Span length outside the six-token codec")
        result.append([VOCAB.get(w, 1) for w in words]+[0]*(6-len(words)))
    return result


def surface(text, entity="orbit", *, admitted=()):
    if not isinstance(text, str) or len(text) > 400 or not re.fullmatch(r"[a-zA-Z ]+[.?]?", text):
        raise ValueError("Unsupported characters, units or sentence length")
    normalized = text.lower().rstrip(".?")
    match = next((m for p in PATTERNS if (m := re.fullmatch(p, normalized))), None)
    if match is None or match["entity"] != entity:
        raise ValueError("Unsupported syntax or entity; name the active entity explicitly")
    clauses = match["body"].split(" then ")
    if not 1 <= len(clauses) <= 3:
        raise ValueError("Use one to three ordered controls")
    fields = {word for words in FIELD_WORDS.values() for word in words}
    allowed = {phrase for phrases in PHRASES.values() for phrase in phrases} | set(admitted)
    ambiguous = {"do not wait": [-1, 1], "do not push right": [-1, 0], "do not push left": [0, 1]}
    if match["field"] not in fields or any(c not in allowed and c not in ambiguous for c in clauses):
        raise ValueError("Unsupported span meaning; a labeled lesson or clarification is required")
    offset = match.start("body")
    spans = []
    for c in clauses:
        spans.append([offset, offset+len(c)])
        offset += len(c)+6
    return {"entity": entity, "field_span": match["field"], "clauses": clauses,
            "spans": spans, "field_support": list(match.span("field")),
            "ambiguities": {str(i): ambiguous[c] for i, c in enumerate(clauses) if c in ambiguous}}


def queries(sequences, templates=(0, 1), entities=("orbit",)):
    """Complete finite compositions; no synthetic sentence is an independent world."""
    result = []
    for acts, field, template, entity in itertools.product(sequences, FIELD_WORDS, templates, entities):
        for wording in range(3):
            text = TEMPLATES[template].format(field=field, entity=entity,
                body=" then ".join(PHRASES[a][wording] for a in acts))
            result.append({"text": text, "entity": entity, "actions": list(acts), "field": field})
    return result


def sequences():
    return [s for n in (1, 2, 3) for s in itertools.product((-1, 0, 1), repeat=n)]
