"""Read-only conditioning diagnostics; does not rerun acquisition or fit models."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('D:/ai/projects/sera/research-continuation/12_reproductions/bundle2')
old=json.loads((root/'source/results/repair/219_sine_4.json').read_text())['result']
new=json.loads((root/'reproduction/results/repair/219_sine_4.json').read_text())['result']
x=np.asarray(old['evidence'][0]['X']);p,v,u=x.T
features=np.column_stack((np.ones(len(x)),p,v,u,p**3,v*np.abs(v),np.sin(old['selected_frequency']*p)))
normal=features.T@features+1e-5*np.eye(features.shape[1])
result={'record':'219_sine_4.json','true_frequency':4,'selected_frequency_archived':old['selected_frequency'],'selected_frequency_fresh':new['selected_frequency'],'admitted_archived':old['admitted'],'admitted_fresh':new['admitted'],
        'support_design_identical':old['evidence'][0]['X']==new['evidence'][0]['X'],
        'selected_design_condition_number':float(np.linalg.cond(features)),
        'regularized_normal_matrix_condition_number':float(np.linalg.cond(normal)),
        'largest_coefficient_absolute_difference':float(np.max(np.abs(np.asarray(old['artifact']['coefficients'])-np.asarray(new['artifact']['coefficients'])))),
        'summary_metrics':{split:{key:{'archived':old['evaluation'][split][key],'fresh':new['evaluation'][split][key],'absolute_difference':abs(old['evaluation'][split][key]-new['evaluation'][split][key])} for key in ('fixed_mse','repair_mse')} for split in ('interpolation','extrapolation')},
        'interpretation':'Conditioning is consistent with amplification of platform arithmetic. This read-only calculation does not isolate the causal library/hardware difference or repair the original strict replay failure.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'independent-audit/conditioning.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
