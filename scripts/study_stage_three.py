"""Run the declared architecture-completion study from a clean output directory."""

import argparse
from pathlib import Path

from sera.stage_three import StageConfig, run_stage_three

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
config = StageConfig(predictor_seconds=1, symbolic_steps=2, typed_steps=2, world_steps=2, reference_steps=2,
                     adaptation_steps=2, outer_inner_steps=1, outer_episodes=3,
                     evaluation_samples=8) if args.smoke else StageConfig()
run_stage_three(args.output, seeds=args.seeds, config=config)
