"""Independent observation-space Gaussian and marginal mixture score checks.

Read-only analysis of an existing cohort. Does not optimize, select, or refit a model.
Design matrices come from the frozen decoder; linear algebra and NLL are independent.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('D:/ai/projects/sera')
RUN = ROOT / 'runs/generative-memory-GG-P0-001'
OUTPUT = ROOT / 'research-continuation/01_audit/generative-memory-math-evidence.json'
spec = importlib.util.spec_from_file_location('review_frozen_generator_core', RUN / 'source/core.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
state = json.loads((RUN / 'state.json').read_text())
protocol = json.loads((RUN / 'protocol.json').read_text())
report = {'run': str(RUN), 'source_identity': state['contract'], 'linear_models': 0,
          'score_partitions': 0, 'expected_score_partitions': 4 * len(protocol['methods']) * len(state['completed']),
          'tolerances': {'mean': {'rtol': 3e-6, 'atol': 2e-7},
                         'covariance': {'rtol': 3e-6, 'atol': 2e-7},
                         'log_evidence': {'rtol': 0, 'atol': 1e-7},
                         'marginal_nll': {'rtol': 1e-12, 'atol': 1e-12}},
          'max_absolute_error': {'mean': 0., 'covariance': 0., 'log_evidence': 0., 'marginal_nll': 0.},
          'failures': [], 'reviewer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'boundary': 'Observation-space Gaussian formula uses the frozen design matrix; not independent feature interpretation. '
                      'Covariance tolerances allow serialized float32. Evidence tolerance1e-7 is fixed from the existing reference test; '
                      'NLL uses original replay tolerance. No coefficients or artifacts are written or selected.'}
start, cpu_start = time.perf_counter(), time.process_time()
def check(field, actual, expected, identity):
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    report['max_absolute_error'][field] = max(report['max_absolute_error'][field], error)
    if not np.allclose(actual, expected, **report['tolerances'][field]):
        report['failures'].append({'field': field, 'identity': identity, 'max_absolute_error': error})

for case_number, completed in enumerate(state['completed'], 1):
    record_path = RUN / completed['path']
    record = json.loads(record_path.read_text())
    with np.load(record_path.parent / 'observations.npz', allow_pickle=False) as obs:
        t, y = obs['support_t'].copy(), obs['support_y'].ravel().copy()
    noise = record['case']['noise']
    with np.load(record_path.parent / 'queries.npz', allow_pickle=False) as raw_queries:
        queries = {role: raw_queries[f'{role}_y'].copy()
                   for role in ('recall', 'interpolation', 'withheld_arc', 'extrapolation')}
    for r in record['records']:
        artifact = json.loads((record_path.parent / r['artifact']).read_text())
        for i, model in enumerate(artifact.get('models', [])):
            a = core.design(model['spec'], t)
            marginal = noise ** 2 * np.eye(len(y)) + 4 * a @ a.T
            cholesky = np.linalg.cholesky(marginal)
            whitened = np.linalg.solve(cholesky, np.column_stack((y, a)))
            wy, wa = whitened[:, 0], whitened[:, 1:]
            posterior_mean = 4 * wa.T @ wy
            posterior_covariance = 4 * np.eye(a.shape[1]) - 16 * wa.T @ wa
            log_evidence = -.5 * (len(y) * np.log(2*np.pi)
                                  + 2*np.log(np.diag(cholesky)).sum() + wy @ wy)
            identity = [completed['case_id'], r['method'], i, model['spec']['name']]
            check('mean', model['mean'], posterior_mean, identity)
            check('covariance', model['covariance'], posterior_covariance, identity)
            check('log_evidence', model['log_evidence'], log_evidence, identity)
            report['linear_models'] += 1
        if set(r['evaluation']) != set(queries):
            report['failures'].append({'field': 'partition_set', 'identity': [completed['case_id'], r['method']]})
        with np.load(record_path.parent / f"{r['method']}-predictions.npz", allow_pickle=False) as raw:
            for role, labels in queries.items():
                means = raw[f'{role}_component_means']
                variances = raw[f'{role}_component_variances']
                weights = raw[f'{role}_class_weights']
                logweights = np.full_like(weights, -np.inf)
                np.log(weights, out=logweights, where=weights > 0)
                component_logp = (-.5*np.log(2*np.pi) - .5*np.log(variances)
                                  - .5*np.square((labels[None] - means) / np.sqrt(variances)))
                reference_nll = -np.mean(np.logaddexp.reduce(component_logp + logweights[:, None, None], axis=0))
                check('marginal_nll', r['evaluation'][role]['marginal_nll'], reference_nll,
                      [completed['case_id'], r['method'], role])
                report['score_partitions'] += 1
    if case_number % 160 == 0:
        print(json.dumps({'cases_checked': case_number, 'failures': len(report['failures'])}), flush=True)
report['cpu_seconds'] = time.process_time() - cpu_start
report['wall_seconds'] = time.perf_counter() - start
report['status'] = 'PASS' if not report['failures'] and report['score_partitions'] == report['expected_score_partitions'] else 'FAIL'
OUTPUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
