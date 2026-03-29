# Tests

## Running tests locally (CPU-only, no GPU)

```bash
# From the project root
python -m pytest tests/ -v
```

Tests that require torch will be **automatically skipped** on machines where PyTorch cannot load (e.g. missing CUDA DLLs).

## Running full test suite on Kaggle / GPU machine

```bash
python -m pytest tests/ -v
```

All 5 torch-dependent test modules will run on a machine with a working PyTorch install:

| Module | Requires torch | Tests |
|---|---|---|
| `test_constants.py` | No | Joint names, edge indices, set partitioning |
| `test_config.py` | No | YAML loading, runtime profiles, deep update |
| `test_data.py` | No | Resample, extract joints, normalize, augment, splitter |
| `test_ddp.py` | Yes | Non-distributed helpers, NCCL env var fix |
| `test_model.py` | Yes | Adjacency matrix, SSRGCN forward shapes, NaN/Inf |
| `test_metrics.py` | Yes | MPJPE, bone loss, MetricTracker |
| `test_engine_helpers.py` | Yes | `_maybe_subset_metadata`, `_resolve_training_cli`, `_unwrap_model`, DDP `broadcast_buffers=False` fix |
| `test_pipeline_integration.py` | Yes | End-to-end: prepare → model → restore; train step forward+backward |

## Key regression tests

- **`test_jitter_only_modifies_input_not_target`** — Catches Bug #1: jitter was previously added independently to `target_25`, corrupting the ground truth. Fixed in `augment_pair`.
- **`test_broadcast_buffers_false_in_engine_ddp_call`** — Catches the NCCL 600s deadlock on Kaggle 2×T4: verifies `DDP(..., broadcast_buffers=False)`.
- **`test_nccl_env_vars_set_when_rank_present`** — Verifies `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` are set via `setup_distributed`.
- **`test_input13_target25_visible_joint_consistency`** — Data integrity: normalized input and target must agree at visible (shared) joints.
