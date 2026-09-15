"""Read-only post-run checks; never fit a model or write into a preserved run."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('D:/ai/projects/sera')
RUN = ROOT / 'runs/generative-memory-GG-P0-001'
OUTPUT = ROOT / 'research-continuation/01_audit/generative-memory-review-evidence.json'

state = json.loads((RUN / 'state.json').read_text())
protocol = json.loads((RUN / 'protocol.json').read_text())
summary = {'run': str(RUN), 'source_identity': state['contract'],
           'status': state['status'], 'cases': len(state['completed']),
           'charged_seconds': state['charged_seconds'], 'memory': state.get('memory'),
           'quantized_coordinate_collisions': [], 'nested_bank_failures': [],
           'artifact_float32_parameter_failures': [], 'mixed_precision_examples': [],
           'artifact_counts': collections.Counter(), 'numeric_payload': {},
           'reviewer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
inventory = collections.defaultdict(collections.Counter)
banks = {}
float32_fields = {'coordinates', 'outputs', 'residuals', 'weights', 'mean', 'covariance'}

def walk(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + (key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, path + (i,))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value

for completed in state['completed']:
    record_path = RUN / completed['path']
    record = json.loads(record_path.read_text())
    case = record['case']
    case_id = completed['case_id']
    identity = (case['seed'], case['family'], case['noise'])
    with np.load(record_path.parent / 'observations.npz', allow_pickle=False) as observations:
        saved = {key: observations[key].copy() for key in observations.files}
    banks.setdefault(identity, {})[case['support']] = saved
    role_coords = {role: saved[f'{role}_t'].astype(np.float32)
                   for role in ('support', 'selection', 'calibration')}
    for role, values in role_coords.items():
        if len(np.unique(values)) != len(values):
            summary['quantized_coordinate_collisions'].append({'case': case_id, 'roles': [role, role]})
    for i, (left, values) in enumerate(role_coords.items()):
        for right in list(role_coords)[i + 1:]:
            if np.intersect1d(values, role_coords[right]).size:
                summary['quantized_coordinate_collisions'].append({'case': case_id, 'roles': [left, right]})
    with np.load(record_path.parent / 'queries.npz', allow_pickle=False) as queries:
        for role in ('interpolation', 'withheld_arc', 'extrapolation'):
            for evidence_role, values in role_coords.items():
                if np.intersect1d(queries[f'{role}_t'].astype(np.float32), values).size:
                    summary['quantized_coordinate_collisions'].append(
                        {'case': case_id, 'roles': [role, evidence_role]})
    for model in record['records']:
        method = model['method']
        summary['artifact_counts'][method] += 1
        artifact = json.loads((record_path.parent / model['artifact']).read_text())
        for path, value in walk(artifact):
            explicit_f32 = any(field in path for field in float32_fields)
            kind = 'float32_parameter_scalars' if explicit_f32 else (
                'float64_scalars' if isinstance(value, float) else 'integer_scalars')
            inventory[method][kind] += 1
            inventory[method]['JSON_number_lexeme_bytes'] += len(json.dumps(value))
            if explicit_f32 and float(np.float32(value)) != value:
                summary['artifact_float32_parameter_failures'].append({'case': case_id, 'method': method,
                                                                       'path': list(path), 'value': value})
            if not explicit_f32 and isinstance(value, float) and float(np.float32(value)) != value:
                if len(summary['mixed_precision_examples']) < 12:
                    summary['mixed_precision_examples'].append({'case': case_id, 'method': method,
                        'path': list(path), 'stored': value, 'rounded_float32': float(np.float32(value))})
        inventory[method]['actual_JSON_artifact_bytes'] += model['artifact_bytes']

for identity, sizes in banks.items():
    full = sizes[max(protocol['support'])]
    for n, smaller in sizes.items():
        for key, values in smaller.items():
            expected = full[key][:n] if key.startswith('support_') else full[key]
            if not np.array_equal(values, expected):
                summary['nested_bank_failures'].append({'identity': identity, 'support': n, 'field': key})
for method, counts in inventory.items():
    counts['counterfactual_typed_floating_payload_bytes'] = (
        4 * counts['float32_parameter_scalars'] + 8 * counts['float64_scalars'])
summary['numeric_payload'] = dict(inventory)
summary['numeric_payload_boundary'] = (
    'The floating binary-equivalent count is counterfactual: no packed binary artifact was deployed. '
    'All listed float32 arrays contain exactly float32-representable values, but decoding and arithmetic use '
    'NumPy float64. Other continuous fields are stored from float64; integer AST/schema values are counted '
    'separately without inventing an integer codec. JSON number lexeme bytes exclude keys and container punctuation. '
    'Actual full JSON bytes remain the deployed cost.')
OUTPUT.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: summary[key] for key in ('status', 'cases', 'charged_seconds', 'memory',
    'quantized_coordinate_collisions', 'nested_bank_failures', 'artifact_float32_parameter_failures',
    'artifact_counts', 'numeric_payload')}, indent=2))
