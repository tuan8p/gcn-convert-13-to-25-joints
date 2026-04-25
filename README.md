# SSR GCN 13 to 25 Joints

Pipeline Skeleton Super-Resolution (SSR) de hoc hoi quy tu skeleton `13 joints`
(Toyota-compatible) sang `25 joints` theo layout NTU/ETRI.

## Bai toan

- Train tren `ETRI elderly` bang cach lay skeleton goc `(T, 25, 3)`.
- Tao input `13 joints` theo mapping Toyota-compatible.
- Output la skeleton day du `(T, 25, 3)`.
- Loss chinh: `joint MSE` (co the **trong so theo joint**: visible Toyota-13 vs thieu; thieu **torso** 0,1,2 vs **extremity** tay/chan/ngon) + `bone length loss` (trong so canh theo joint neu bat `use_weighted_bone`).
- Metric chinh: `MPJPE`; metric phu: `missing MPJPE`, `extremity_missing_mpjpe`, `torso_missing_mpjpe`, `visible MPJPE`, `bone error`.

## Checkpoint / early stopping (val)

- `experiment.best_val_metric`: `mpjpe` | `extremity_missing_mpjpe` | `torso_missing_mpjpe` | `missing_mpjpe` | `combined`
- Voi `combined`: `combined_val_w_mpjpe` + `combined_val_w_extremity` (tong quy chuan hoa 1) — uu tien can bang tay/chan voi MPJPE tong.
- `metrics.json` luu `best_val_score` (giong tieu chuan da chon), kem `best_val_mpjpe` / `best_val_extremity_missing_mpjpe` tai epoch do.

**Preset mau:** `configs/ssr_gcn_strong_extremity.yaml` — `missing_extremity_mult: 2.25` va `best_val_metric: extremity_missing_mpjpe` (patience 20).

## Cau truc

- `configs/ssr_gcn_kaggle.yaml`: config mac dinh cho Kaggle `2xT4` ( `best_val_metric: combined` + loss trong so joint).
- `configs/ssr_gcn_strong_extremity.yaml`: extremity nang + chon `best` theo `extremity_missing_mpjpe`.
- `ssr_gcn/data.py`: split theo subject, resample `T=150`, normalize, augmentation.
- `ssr_gcn/model.py`: baseline ST-GCN encoder-decoder nhe cho `13 -> 25`.
- `ssr_gcn/engine.py`: train/val/test, DDP, checkpoint, metrics, figures, wandb.
- `tools/train.py`: train SSR.
- `tools/evaluate.py`: evaluate checkpoint.
- `tools/smoke_test_preprocessing.py`: smoke test preprocessing.
- `tools/infer_toyota.py`: infer `25 joints` tu input Toyota `13 joints`.

## Du lieu

Mac dinh config dang tro den:

```text
../data/processed/npy_merged
```

Pipeline hien tai train tren cac record co prefix:

```text
etri_activity3d_elderly_
```

Input Toyota-compatible `13 joints` duoc rut ra tu `25 joints` theo mapping dang dung
trong `EAR-Anticipation-Model/ear/transform/toyota_13_to_25.py`.

## Cai dat

```bash
pip install -r requirements.txt
```

Neu chay tren Kaggle, nen dung image co PyTorch san va set them `WANDB_API_KEY`
neu muon log W&B.

## Chay smoke test preprocessing

```bash
python tools/smoke_test_preprocessing.py --runtime-profile local_debug --samples 3 --batch-size 2
```

Smoke test se:

- Doc mot vai sequence ETRI elderly that.
- Resample ve `T=150`.
- Tao cap `(input_13, target_25)`.
- In shape, scale, split info.

## Train

Chay tu thu muc goc cua repo (`gcn-convert-13-to-25-joints`).

**Quy tac tham so:** moi arg tren dong lenh la **tuy chon**. Khong truyen arg nao thi toan bo gia tri
lay tu file YAML (mac dinh `configs/ssr_gcn_kaggle.yaml`). **Truyen arg thi chi ghi de dung muc do**
trong config, cac muc khac van theo YAML.

| Arg CLI | Khi co truyen thi ghi de |
|---------|---------------------------|
| `--config` | Duong file YAML (mac dinh: `configs/ssr_gcn_kaggle.yaml`) |
| `--runtime-profile` | Profile trong `runtime_profile:` (neu bo trong, dung `runtime.profile` trong YAML) |
| `--max-epochs` | `training.epochs` |
| `--subset-ratio` | `experiment.subset_ratio` |
| `--lr` | `training.lr` |
| `--per-gpu-batch-size` | `training.per_gpu_batch_size` |

Single GPU — chi can config:

```bash
python tools/train.py
```

Multi-GPU (`torchrun`):

```bash
torchrun --standalone --nproc_per_node=2 tools/train.py
```

Vi du ghi de mot vai tham so, con lai van theo YAML:

```bash
python tools/train.py --runtime-profile local_debug --subset-ratio 0.02 --max-epochs 2 --lr 0.0005
```

Xem them: `python tools/train.py --help` (cuoi help co tom tat map arg → YAML).

## Output train

Moi run se tao thu muc:

```text
outputs/ssr_gcn/run_<timestamp>_<profile>/
```

Ben trong co:

- `best_model.pt`
- `run_config.yaml`
- `split_info.json`
- `split_manifest.json`
- `training_log.json`
- `metrics.json`
- `figures/`

## Evaluate checkpoint

```bash
python tools/evaluate.py --checkpoint outputs/ssr_gcn/<run>/best_model.pt --split test
```

## Infer Toyota

Script infer chap nhan sequence `13 joints` hoac `25 joints`.
Neu input la `25 joints`, script se tu rut ve `13 joints` Toyota-compatible truoc khi infer.

```bash
python tools/infer_toyota.py ^
  --checkpoint outputs/ssr_gcn/<run>/best_model.pt ^
  --input-dir ../data/processed/npy_merged/toyota_smarthome ^
  --output-dir outputs/toyota_infer
```

Output se gom:

- file `.npy` du doan `25 joints`
- `inference_manifest.json`

## Ghi chu Kaggle

- Config `t4` dang de `per_gpu_batch_size=8`, `val_per_gpu_batch_size=16`, `amp=float16`.
- Dung `torchrun --nproc_per_node=2` de khai thac 2 GPU T4.
- W&B la no-op neu khong cau hinh token/project.
