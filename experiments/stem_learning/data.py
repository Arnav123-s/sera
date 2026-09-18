"""Auditable transcriptions of human textbook calculations; no generated teaching."""

import hashlib
import json
import xml.etree.ElementTree as ET
from fractions import Fraction as Q

from .intake import BASE, OUT, RAW


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build():
    # Dimensions: mass, length, time, current, temperature.
    units = {
        "kg": [1, 0, 0, 0, 0], "m": [0, 1, 0, 0, 0], "s": [0, 0, 1, 0, 0],
        "m/s": [0, 1, -1, 0, 0], "m/s^2": [0, 1, -2, 0, 0],
        "N": [1, 1, -2, 0, 0], "kg*m/s": [1, 1, -1, 0, 0],
        "N*s": [1, 1, -1, 0, 0], "J": [1, 2, -2, 0, 0], "W": [1, 2, -3, 0, 0],
        "N/m": [1, 0, -2, 0, 0], "A": [0, 0, 0, 1, 0],
        "V": [1, 2, -3, -1, 0], "ohm": [1, 2, -3, -2, 0],
        "C": [0, 0, 1, 1, 0], "K": [0, 0, 0, 0, 1], "J/(kg*K)": [0, 2, -2, 0, -1],
    }
    relations = {}

    def relation(name, inputs, output, coefficient, powers, assumptions, positive=()):
        relations[name] = {"inputs": inputs, "output": output,
                           "reference": {"coefficient": coefficient, "powers": powers},
                           "assumptions": assumptions, "positive": list(positive), "lessons": []}

    mechanics = ["one_dimension", "nonrelativistic", "inertial_frame"]
    relation("momentum", {"mass": "kg", "velocity": "m/s"}, ["momentum", "kg*m/s"],
             "1", [1, 1], mechanics, ["mass"])
    relation("impulse", {"force": "N", "time": "s"}, ["impulse", "N*s"],
             "1", [1, 1], mechanics+["net_force", "constant_or_average_force"], ["time"])
    relation("newton", {"mass": "kg", "acceleration": "m/s^2"}, ["force", "N"],
             "1", [1, 1], mechanics+["constant_mass", "net_force"], ["mass"])
    relation("kinetic_energy", {"mass": "kg", "speed": "m/s"}, ["kinetic_energy", "J"],
             "1/2", [1, 2], mechanics+["translational_motion"], ["mass"])
    relation("work", {"parallel_force": "N", "displacement": "m"}, ["work", "J"],
             "1", [1, 1], ["constant_force", "signed_parallel_component"])
    relation("power", {"work": "J", "time": "s"}, ["power", "W"],
             "1", [1, -1], ["average_over_interval"], ["time"])
    relation("spring_energy", {"spring_constant": "N/m", "extension": "m"}, ["spring_energy", "J"],
             "1/2", [1, 2], ["hooke_regime", "zero_at_equilibrium"], ["spring_constant"])
    relation("gravitational_energy", {"mass": "kg", "gravity": "m/s^2", "height": "m"}, ["potential_energy", "J"],
             "1", [1, 1, 1], ["uniform_gravity", "specified_height_zero"], ["mass", "gravity"])
    relation("ohm", {"current": "A", "resistance": "ohm"}, ["voltage", "V"],
             "1", [1, 1], ["ohmic_element", "fixed_temperature"], ["resistance"])
    relation("electric_power", {"voltage": "V", "current": "A"}, ["power", "W"],
             "1", [1, 1], ["dc_or_instantaneous", "passive_sign_convention"])
    relation("charge", {"current": "A", "time": "s"}, ["charge", "C"],
             "1", [1, 1], ["constant_or_average_current"], ["time"])
    relation("heat", {"mass": "kg", "specific_heat": "J/(kg*K)", "temperature_change": "K"}, ["heat", "J"],
             "1", [1, 1, 1], ["constant_specific_heat", "single_phase"], ["mass", "specific_heat"])

    sources = {}

    def lesson(law, identifier, module, anchor, values, target, note, split="train"):
        path = RAW / "modules" / module / "index.cnxml"
        raw = path.read_bytes()
        tree = ET.fromstring(raw)
        node = next(n for n in tree.iter() if n.get("id") == anchor)
        excerpt = " ".join("".join(node.itertext()).split())
        source = {"url": BASE+f"modules/{module}/index.cnxml", "anchor": anchor,
                  "sha256": hashlib.sha256(raw).hexdigest(), "excerpt": excerpt,
                  "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest()}
        source_id = module+"#"+anchor
        sources[source_id] = source
        relations[law]["lessons"].append({"id": identifier, "source_id": source_id, "split": split,
                                          "inputs": list(map(str, values)), "output": str(target),
                                          "transcription": note})

    lesson("momentum", "car-momentum", "m58318", "fs-id1167133647686", [1400, 15], 21000,
           "Original car mass, velocity and momentum.")
    lesson("momentum", "box-momentum", "m58318", "fs-id1167132413861", [5, 5], 25,
           "Box mass and speed; published answer is 25 kg m/s.", "human_check")
    lesson("newton", "soccer-x", "m58297", "fs-id1165039045618", ["0.400", 3], "1.2", "Original x component.")
    lesson("newton", "car-force", "m58297", "fs-id1165039443202", [3000, "-0.2"], -600,
           "Original signed component with published solved mass.")
    lesson("newton", "soccer-y", "m58297", "fs-id1165039045618", ["0.400", 7], "2.8",
           "Untaught y component of same example; part overlap is explicit.", "human_check")
    lesson("kinetic_energy", "athlete-energy", "m58308", "fs-id1165038048633", [80, 10], 4000,
           "Original athlete example, kJ converted to J.")
    lesson("kinetic_energy", "subway-person", "m58308", "fs-id1165037049856", [75, "1.50"], "84.375",
           "Original 1/2*75*1.50^2 before published rounding to 84.4 J.")
    lesson("kinetic_energy", "basketball", "m58308", "fs-id1165038198441", ["0.624", "7.5"], "17.55",
           "Original 1/2*0.624*7.5^2 before published rounding to 17.6 J.", "human_check")
    lesson("work", "lower-book", "m58719", "fs-id1165039051160", [20, 1], 20,
           "Choose downward positive for the source's gravitational work.")
    lesson("work", "couch-friction", "m58719", "fs-id1165039346377", [-600, 4], -2400,
           "Source friction 0.6*1000 N opposite its 3+1 m path.")
    lesson("power", "pull-up", "m58310", "fs-id1165037910246", ["423.36", "0.8"], "529.2",
           "Source work 0.9*80*9.8*0.60 J, before rounding power to 529 W.")
    lesson("power", "winch-mechanical", "m58734", "fs-id1170902335576", [49000, 30], Q(4900, 3),
           "Source 4900*10 J over 30 s, before rounding.")
    lesson("spring_energy", "spring-6cm", "m58719", "fs-id1165039118668", [300, "0.06"], "0.54",
           "Original spring constant 3 N/cm converted to N/m.")
    lesson("spring_energy", "spring-3cm", "m58312", "fs-id1165035980783", [400, "0.03"], "0.18",
           "Original 4 N/cm, displacement 23-20 cm.", "human_check")
    lesson("gravitational_energy", "hiker-summit", "m58312", "fs-id1165038964832", [75, "9.8", 147], 108045,
           "Original 75*9.8*147 before published 108 kJ rounding.")
    lesson("gravitational_energy", "hiker-sea", "m58312", "fs-id1165038964832", [75, "9.8", -48], -35280,
           "Original 75*9.8*(-48) before published -35.3 kJ rounding.")
    lesson("ohm", "carbon-resistor", "m58732", "fs-id1170902116601", ["0.003", 3000], 9,
           "Original voltage, measured current and solved room-temperature resistance.")
    lesson("electric_power", "winch-electric", "m58734", "fs-id1170902335576", [115, 20], 2300,
           "Original motor electrical power calculation.")
    lesson("charge", "starter-charge", "m58729", "fs-id1170902884663", [180, 4], 720,
           "Original truck starter charge, time and solved current.")
    lesson("heat", "water-heating", "m58387", "fs-id1170903904170", ["0.250", 4186, 60], 62790,
           "Original water heat expression before rounding to 62.8 kJ.")
    lesson("heat", "pan-heating", "m58387", "fs-id1170903904170", ["0.500", 900, 60], 27000,
           "Original aluminum pan heat expression.")
    lesson("heat", "brake-heating", "m58387", "fs-id1170903814928", [10, 800, Q(735000, 8000)], 735000,
           "Original Q/(mc) temperature change before rounded 92 degrees.", "human_check")
    lesson("impulse", "restrained-driver", "m58319", "fs-id1167131159147", ["-948.24", "2.5"], "-2370.6",
           "Original 87.8*(-27)/2.5 force and 87.8*(-27) momentum change, before rounding.")
    lesson("impulse", "unrestrained-driver", "m58319", "fs-id1167131159147", [-11853, "0.20"], "-2370.6",
           "Second published collision interval; same original driver and velocity change.")
    return {"schema": "sera.stem-teaching.1", "units": units, "relations": relations,
            "sources": sources, "teaching_origin": "human textbook; supervised transcription of original calculations",
            "source_license": "CC BY-NC-SA 4.0", "generated_training_examples": 0}


def prepare():
    path = OUT / "teaching.json"
    if path.exists():
        raise FileExistsError("Preserve the frozen teaching catalog")
    value = build()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (OUT / "SOURCE-LICENSE.txt").write_bytes((RAW / "LICENSE").read_bytes())
    (OUT / "sources.json").write_bytes((RAW / "receipt.json").read_bytes())
    print(json.dumps({"relations": len(value["relations"]), "sources": len(value["sources"]),
                      "teaching_sha256": digest(value)}))


if __name__ == "__main__":
    prepare()
