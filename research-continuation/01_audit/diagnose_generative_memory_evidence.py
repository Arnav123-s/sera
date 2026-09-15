"""Stable SVD evidence comparison following the preserved Cholesky-reference failure."""
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
AUDIT = ROOT / 'research-continuation/01_audit'
spec = importlib.util.spec_from_file_location('svd_review_frozen_core', RUN / 'source/core.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
state = json.loads((RUN / 'state.json').read_text())
previous = json.loads((AUDIT / 'generative-memory-math-evidence.json').read_text())
failed = {tuple(item['identity']) for item in previous['failures'] if item['field'] == 'log_evidence'}
result = {'reference': 'Thin SVD of A, spectrum of sigma^2 I + 4 A A^T; orthogonal residual quadratic',
          'source_identity': state['contract'], 'linear_models': 0, 'family_posteriors': 0,
          'absolute_log_evidence_tolerance': 1e-7,
          'failures': [], 'max_log_evidence_error': 0., 'max_family_weight_error': 0.,
          'prior_observation_cholesky_failures': [],
          'previous_diagnostic_status_preserved': previous['status'],
          'reviewer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
start, cpu = time.perf_counter(), time.process_time()
for completed in state['completed']:
    record_path = RUN / completed['path']
    record = json.loads(record_path.read_text())
    with np.load(record_path.parent / 'observations.npz', allow_pickle=False) as obs:
        t, y = obs['support_t'].copy(), obs['support_y'].ravel().copy()
    sigma2 = record['case']['noise'] ** 2
    for r in record['records']:
        artifact = json.loads((record_path.parent / r['artifact']).read_text())
        references = []
        for i, model in enumerate(artifact.get('models', [])):
            a = core.design(model['spec'], t)
            u, s, _ = np.linalg.svd(a, full_matrices=False)
            projected = u.T @ y
            residual = y - u @ projected
            eigenvalues = sigma2 + 4 * s**2
            logdet = len(y) * np.log(sigma2) + np.log1p(4 * s**2 / sigma2).sum()
            quadratic = residual @ residual / sigma2 + np.sum(projected**2 / eigenvalues)
            reference = -.5 * (len(y) * np.log(2*np.pi) + logdet + quadratic)
            references.append(reference)
            error = abs(reference - model['log_evidence'])
            identity = (completed['case_id'], r['method'], i, model['spec']['name'])
            if error > result['max_log_evidence_error']:
                result['max_log_evidence_error'] = float(error)
                result['max_log_evidence_identity'] = identity
            if error > result['absolute_log_evidence_tolerance']:
                result['failures'].append({'identity': identity, 'absolute_error': float(error)})
            if identity in failed:
                result['prior_observation_cholesky_failures'].append({
                    'identity': identity, 'svd_reference': float(reference),
                    'artifact_log_evidence': model['log_evidence'], 'artifact_vs_svd_error': float(error),
                    'observation_covariance_condition': float(eigenvalues.max()/sigma2)})
            result['linear_models'] += 1
        if r['method'] == 'D_family':
            references = np.asarray(references)
            weights = np.exp(references - references.max())
            weights /= weights.sum()
            difference = float(np.max(np.abs(weights - artifact['class_weights'])))
            result['max_family_weight_error'] = max(result['max_family_weight_error'], difference)
            result['family_posteriors'] += 1
result['cpu_seconds'] = time.process_time() - cpu
result['wall_seconds'] = time.perf_counter() - start
result['status'] = 'PASS' if not result['failures'] else 'FAIL'
result['interpretation'] = (
    'This follows up, rather than erases, the Cholesky diagnostic failure at the same1e-7 evidence threshold. '
    'The SVD formula avoids a dense ill-conditioned covariance factorization and cancellation in inverse-based quadratics. '
    'Shared frozen design matrices remain an explicit limitation of the independent reference.')
(AUDIT / 'generative-memory-svd-evidence.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'prior_observation_cholesky_failures'}, indent=2))
