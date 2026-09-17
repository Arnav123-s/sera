"""C01: conditional empirical parameters on the actual continuing StudyR1."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from experiments.self_study.model import StudyR1
from sera.shared import LowRankUpdate

KINDS = ("r1_frozen", "r1_adapted", "r1_reset", "gru", "window")


def source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class RefiningR1(StudyR1):
    @classmethod
    def attach(cls, owner, kind="r1_frozen", seed=2801):
        if type(owner) is not StudyR1 or kind not in KINDS:
            raise ValueError("Restore the actual StudyR1 and declare the empirical route")
        torch.manual_seed(seed)
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        owner.__class__ = cls
        width = owner.settings["width"]
        owner.concept_config = {"kind": kind, "seed": seed, "source": source(), "features": 6, "rank": 4}
        owner.concept_encoder = nn.Sequential(nn.Linear(6, width), nn.Tanh())
        owner.concept_head = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))
        owner.concept_parameter_names = []
        owner.concept_adapters = nn.ModuleList()
        if kind == "r1_adapted":
            for name, parameter in list(owner.named_parameters()):
                if (name.startswith("memory.cores.") or name.startswith("fusion.")) and parameter.ndim == 2:
                    owner.concept_parameter_names.append(name)
                    owner.concept_adapters.append(LowRankUpdate(*parameter.shape, 4))
        if kind == "gru":
            owner.concept_gru = nn.GRU(6, 32, batch_first=True)
            owner.concept_gru_head = nn.Linear(32, 1)
        if kind == "window":
            owner.concept_window = nn.Sequential(nn.Linear(48, 64), nn.Tanh(), nn.Linear(64, 1))
        # Online subject coefficients are registered computational weights, not a record-only substitute.
        owner.concept_subjects = nn.ParameterDict()
        return owner

    def export_config(self):
        return {**super().export_config(), "concept_refinement": self.concept_config}

    def forward(self, *args, route="world", **kwargs):
        if route == "empirical-core":
            return self.empirical_core(*args, **kwargs)
        return super().forward(*args, route=route, **kwargs)

    def empirical(self, features):
        if self.concept_config["kind"] == "r1_adapted":
            original = dict(self.named_parameters())
            replacements = {name: original[name] + adapter() for name, adapter in
                            zip(self.concept_parameter_names, self.concept_adapters)}
            return torch.func.functional_call(self, replacements, (features,), {"route": "empirical-core"})
        return self.empirical_core(features)

    def empirical_core(self, features):
        kind = self.concept_config["kind"]
        if kind == "gru":
            return self.concept_gru_head(self.concept_gru(features)[0]).squeeze(-1)
        if kind == "window":
            x = torch.nn.functional.pad(features, (0, 0, 7, 0))
            x = torch.stack([x[:, i:i+8].reshape(len(x), 48) for i in range(features.shape[1])], 1)
            return self.concept_window(x).squeeze(-1)
        encoded = self.concept_encoder(features)
        state, outputs = self.initial(len(features)), []
        for t in range(features.shape[1]):
            if kind == "r1_reset":
                state = self.initial(len(features))
            value, state = self.shared_step(state, encoded[:, t], encoded.new_ones((len(features), 1)))
            outputs.append(self.concept_head(value).squeeze(-1))
        return torch.stack(outputs, 1)


def delta(owner):
    return {name: value.detach().clone() for name, value in owner.state_dict().items() if name.startswith("concept_")}


def apply(owner, values):
    expected = delta(owner)
    if set(expected) != set(values) or any(expected[n].shape != v.shape or expected[n].dtype != v.dtype
            or not torch.isfinite(v).all() for n, v in values.items()):
        raise ValueError("Empirical checkpoint schema or finiteness changed")
    owner.load_state_dict({**owner.state_dict(), **values})
