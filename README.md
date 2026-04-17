# MedFrameQA Order-Vote Evolve

Code and paper materials for:

**Inference-Time Agentic Decision Rules Beat Longer Evolving Search for Multi-Image Medical Reasoning**

## Repository contents

- `advanced_vqa_task_fixed/`, `advanced_vqa_task_reasoning/`, `advanced_vqa_task_order_vote/`, `advanced_vqa_task_order_rerank/`, `advanced_vqa_task_order_vote_plus/`:
  task-specific method definitions and evaluation entrypoints.
- `medframeqa_runtime.py`:
  shared runtime and evaluation utilities.
- `create_medframeqa_split_manifest.py`:
  reproducible internal frozen split generation.
- `run_medframeqa_paper_pipeline.py`, `run_medframeqa_repeats.py`, `medframeqa_run_paper_eval.py`:
  experiment, repeat-run, and post-hoc evaluation pipeline.
- `paper_analysis_output/`:
  publication-clean summary JSON files used by the paper.
- `main.tex`, `references.bib`, `figures/`:
  current paper draft and figure assets.
- `sakana_medframeqa_*.ipynb`:
  notebooks for running and analyzing the methods.

## Notes

- This repository excludes large raw run directories and local packaging artifacts.
- The paper uses an **internal frozen split** for MedFrameQA rather than an official hidden test set.
