"""Training and evaluation orchestration for SSR GCN."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ssr_gcn.data import SSRDataset, SubjectSplit, build_etri_elderly_split, save_json
from ssr_gcn.ddp import (
    all_reduce_mean,
    barrier,
    broadcast_object,
    cleanup_distributed,
    is_distributed,
    is_rank0,
    rank,
    seed_everything,
    setup_distributed,
    world_size,
)
from ssr_gcn.figures import save_all_figures
from ssr_gcn.logging_wandb import WandbLogger
from ssr_gcn.metrics import (
    MetricTracker,
    bone_length_error,
    missing_joint_mpjpe,
    mpjpe,
    total_loss,
    visible_joint_mpjpe,
)
from ssr_gcn.model import create_model


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    experiment_cfg = cfg.get("experiment") or {}
    output_root = Path(experiment_cfg.get("output_dir", "outputs/ssr_gcn")).resolve()
    profile = (cfg.get("runtime") or {}).get("profile", "default")
    return output_root / f"run_{_timestamp()}_{profile}"


def _maybe_subset_metadata(
    split: SubjectSplit,
    subset_ratio: float,
    seed: int,
) -> SubjectSplit:
    if subset_ratio >= 1.0:
        return split

    generator = torch.Generator().manual_seed(seed)

    def take_subset(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return items
        n_keep = max(1, int(len(items) * subset_ratio))
        indices = torch.randperm(len(items), generator=generator)[:n_keep].tolist()
        keep = set(indices)
        return [item for idx, item in enumerate(items) if idx in keep]

    new_train = take_subset(split.train)
    new_val = take_subset(split.val)
    new_test = take_subset(split.test)
    info = dict(split.info)
    info["subset_ratio"] = subset_ratio
    info["subset_sizes"] = {
        "train": len(new_train),
        "val": len(new_val),
        "test": len(new_test),
    }
    return SubjectSplit(new_train, new_val, new_test, info)


def _build_datasets(
    cfg: dict[str, Any],
    subset_ratio: float = 1.0,
) -> tuple[Path, SubjectSplit, dict[str, SSRDataset]]:
    root, split = build_etri_elderly_split(cfg)
    seed = int((cfg.get("experiment") or {}).get("seed", 42))
    split = _maybe_subset_metadata(split, subset_ratio=subset_ratio, seed=seed)
    prep_cfg = cfg.get("preprocessing") or {}
    aug_cfg = cfg.get("augmentation") or {}
    fixed_len = int(prep_cfg.get("fixed_len", 150))

    datasets = {
        "train": SSRDataset(
            root=root,
            metadata=split.train,
            fixed_len=fixed_len,
            augment=bool(aug_cfg.get("enabled", True)),
            rotation_deg=float(aug_cfg.get("rotation_deg", 10.0)),
            jitter_std=float(aug_cfg.get("jitter_std", 0.005)),
            base_seed=seed,
        ),
        "val": SSRDataset(
            root=root,
            metadata=split.val,
            fixed_len=fixed_len,
            augment=False,
            base_seed=seed + 10_000,
        ),
        "test": SSRDataset(
            root=root,
            metadata=split.test,
            fixed_len=fixed_len,
            augment=False,
            base_seed=seed + 20_000,
        ),
    }
    return root, split, datasets


def _build_loaders(
    cfg: dict[str, Any],
    datasets: dict[str, SSRDataset],
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DistributedSampler | None]:
    training_cfg = cfg.get("training") or {}
    train_batch_size = int(training_cfg.get("per_gpu_batch_size", 8))
    eval_batch_size = int(training_cfg.get("val_per_gpu_batch_size", train_batch_size))
    num_workers = int(training_cfg.get("num_workers", 2))
    pin_memory = bool(training_cfg.get("pin_memory", True))
    persistent_workers = bool(training_cfg.get("persistent_workers", False))

    train_sampler = DistributedSampler(datasets["train"], shuffle=True) if is_distributed() else None
    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )

    if is_rank0():
        val_loader = DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
        )
        test_loader = DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
        )
    else:
        val_loader = None
        test_loader = None
    return train_loader, val_loader, test_loader, train_sampler


def _build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    training_cfg = cfg.get("training") or {}
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    training_cfg = cfg.get("training") or {}
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()
    epochs = int(training_cfg.get("epochs", 50))
    if scheduler_name == "exponential":
        gamma = float(training_cfg.get("lr_decay", 0.98))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if scheduler_name == "cosine":
        eta_min = float(training_cfg.get("min_lr", 1e-5))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)
    return None


def _maybe_compile(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.nn.Module:
    training_cfg = cfg.get("training") or {}
    if not bool(training_cfg.get("compile", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode=str(training_cfg.get("compile_mode", "default")))
    except Exception as exc:
        if is_rank0():
            print(f"[compile] WARNING: torch.compile disabled ({exc})")
        return model


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, float],
) -> Path:
    checkpoint_path = output_dir / "best_model.pt"
    payload = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    _unwrap_model(model).load_state_dict(payload["model_state_dict"])
    return payload


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["input_13"].to(device, non_blocking=True)
    targets = batch["target_25"].to(device, non_blocking=True)
    return inputs, targets


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    tracker = MetricTracker()
    use_amp = bool((cfg.get("training") or {}).get("use_amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16 if str((cfg.get("training") or {}).get("amp_dtype", "float16")) == "float16" else torch.bfloat16
    grad_clip = float((cfg.get("training") or {}).get("grad_clip", 0.0))
    joint_weight = float((cfg.get("loss") or {}).get("joint_weight", 1.0))
    bone_weight = float((cfg.get("loss") or {}).get("bone_weight", 0.1))
    start = time.perf_counter()

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}",
        leave=False,
        disable=not is_rank0(),
        unit="batch",
    )
    for batch in pbar:
        inputs, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            predictions = model(inputs)
            loss, loss_parts = total_loss(
                predictions,
                targets,
                joint_weight=joint_weight,
                bone_weight=bone_weight,
            )
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_metrics = {
                "loss": float(loss.detach().item()),
                "joint_loss": loss_parts["joint_loss"],
                "bone_loss": loss_parts["bone_loss"],
                "mpjpe": float(mpjpe(predictions, targets).detach().item()),
                "missing_mpjpe": float(missing_joint_mpjpe(predictions, targets).detach().item()),
                "visible_mpjpe": float(visible_joint_mpjpe(predictions, targets).detach().item()),
                "bone_error": float(bone_length_error(predictions, targets).detach().item()),
            }
            tracker.update(count=inputs.size(0), **batch_metrics)
            if is_rank0():
                elapsed = time.perf_counter() - start
                samples_per_sec = tracker.total / max(elapsed, 1e-9)
                pbar.set_postfix(
                    loss=f"{batch_metrics['loss']:.4f}",
                    mpjpe=f"{batch_metrics['mpjpe']:.4f}",
                    bone=f"{batch_metrics['bone_error']:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    sps=f"{samples_per_sec:.1f}",
                )

    epoch_sec = time.perf_counter() - start
    metrics = tracker.compute()
    samples_per_sec = tracker.total / max(epoch_sec, 1e-9)
    reduced_metrics = {key: all_reduce_mean(value, device) for key, value in metrics.items()}
    reduced_metrics["samples_per_sec"] = all_reduce_mean(samples_per_sec, device)
    reduced_metrics["epoch_time_sec"] = all_reduce_mean(epoch_sec, device)
    return reduced_metrics


@torch.no_grad()
def _run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    cfg: dict[str, Any],
    desc: str,
) -> dict[str, Any]:
    if loader is None:
        return {}

    model.eval()
    tracker = MetricTracker()
    per_joint_sum = torch.zeros(25, dtype=torch.float64, device=device)
    frame_count = 0
    use_amp = bool((cfg.get("training") or {}).get("use_amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16 if str((cfg.get("training") or {}).get("amp_dtype", "float16")) == "float16" else torch.bfloat16
    joint_weight = float((cfg.get("loss") or {}).get("joint_weight", 1.0))
    bone_weight = float((cfg.get("loss") or {}).get("bone_weight", 0.1))

    start = time.perf_counter()
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_rank0(), unit="batch")
    for batch in pbar:
        inputs, targets = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            predictions = model(inputs)
            loss, loss_parts = total_loss(
                predictions,
                targets,
                joint_weight=joint_weight,
                bone_weight=bone_weight,
            )

        joint_errors = torch.linalg.norm(predictions - targets, dim=-1)
        per_joint_sum += joint_errors.sum(dim=(0, 1)).to(torch.float64)
        frame_count += joint_errors.shape[0] * joint_errors.shape[1]

        batch_metrics = {
            "loss": float(loss.detach().item()),
            "joint_loss": loss_parts["joint_loss"],
            "bone_loss": loss_parts["bone_loss"],
            "mpjpe": float(mpjpe(predictions, targets).detach().item()),
            "missing_mpjpe": float(missing_joint_mpjpe(predictions, targets).detach().item()),
            "visible_mpjpe": float(visible_joint_mpjpe(predictions, targets).detach().item()),
            "bone_error": float(bone_length_error(predictions, targets).detach().item()),
        }
        tracker.update(count=inputs.size(0), **batch_metrics)
        pbar.set_postfix(
            loss=f"{batch_metrics['loss']:.4f}",
            mpjpe=f"{batch_metrics['mpjpe']:.4f}",
            bone=f"{batch_metrics['bone_error']:.4f}",
        )

    metrics = tracker.compute()
    elapsed = time.perf_counter() - start
    metrics["samples_per_sec"] = tracker.total / max(elapsed, 1e-9)
    if frame_count > 0:
        metrics["per_joint_mpjpe"] = {
            f"joint_{idx:02d}": float(value / frame_count) for idx, value in enumerate(per_joint_sum.tolist())
        }
    return metrics


def run(cfg: dict[str, Any], args: Any) -> int:
    """Run training, validation, test, and artifact export."""
    device = setup_distributed(backend=str((cfg.get("ddp") or {}).get("backend", "nccl")))
    seed_everything(int((cfg.get("experiment") or {}).get("seed", 42)) + rank())

    output_dir_obj = _resolve_output_dir(cfg) if is_rank0() else None
    output_dir = Path(broadcast_object(str(output_dir_obj) if output_dir_obj is not None else None))
    run_name = output_dir.name
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file, allow_unicode=True, sort_keys=False)

    root, split, datasets = _build_datasets(cfg, subset_ratio=float(getattr(args, "subset_ratio", 1.0)))
    train_loader, val_loader, test_loader, train_sampler = _build_loaders(cfg, datasets)

    if is_rank0():
        save_json(output_dir / "split_info.json", split.info)
        save_json(
            output_dir / "split_manifest.json",
            {
                "train": [item["npy_file"] for item in split.train],
                "val": [item["npy_file"] for item in split.val],
                "test": [item["npy_file"] for item in split.test],
                "npy_merged_root": str(root),
            },
        )

    model = create_model(cfg).to(device)
    model = _maybe_compile(model, cfg)
    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and bool((cfg.get("training") or {}).get("use_amp", True)))

    wandb_logger = WandbLogger.from_cfg(cfg, run_name=run_name) if is_rank0() else WandbLogger(False, None, None, None, [])
    if is_rank0():
        wandb_logger.log_config(cfg)

    training_log: list[dict[str, Any]] = []
    epochs = int(getattr(args, "max_epochs", None) or (cfg.get("training") or {}).get("epochs", 50))
    patience = int((cfg.get("experiment") or {}).get("early_stopping_patience", 10))
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    try:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_train_epoch(
                model=model,
                loader=train_loader,
                sampler=train_sampler,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                cfg=cfg,
                epoch=epoch,
            )
            if scheduler is not None:
                scheduler.step()

            barrier()

            if is_rank0():
                val_metrics = _run_eval_epoch(model, val_loader, device, cfg, desc=f"Val {epoch}")
                epoch_log = {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": train_metrics.get("loss", 0.0),
                    "train_mpjpe": train_metrics.get("mpjpe", 0.0),
                    "train_missing_mpjpe": train_metrics.get("missing_mpjpe", 0.0),
                    "train_visible_mpjpe": train_metrics.get("visible_mpjpe", 0.0),
                    "train_bone_error": train_metrics.get("bone_error", 0.0),
                    "train_samples_per_sec": train_metrics.get("samples_per_sec", 0.0),
                    "val_loss": val_metrics.get("loss", 0.0),
                    "val_mpjpe": val_metrics.get("mpjpe", 0.0),
                    "val_missing_mpjpe": val_metrics.get("missing_mpjpe", 0.0),
                    "val_visible_mpjpe": val_metrics.get("visible_mpjpe", 0.0),
                    "val_bone_error": val_metrics.get("bone_error", 0.0),
                    "val_samples_per_sec": val_metrics.get("samples_per_sec", 0.0),
                    "epoch_time_sec": train_metrics.get("epoch_time_sec", 0.0),
                }
                training_log.append(epoch_log)
                wandb_logger.log_epoch({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
                wandb_logger.log_epoch({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
                wandb_logger.log_epoch({"train/lr": epoch_log["lr"]}, step=epoch)

                if epoch_log["val_mpjpe"] < best_metric:
                    best_metric = epoch_log["val_mpjpe"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    _save_checkpoint(output_dir, model, optimizer, epoch, val_metrics)
                else:
                    epochs_without_improvement += 1

                print(
                    f"[epoch {epoch:03d}] "
                    f"train_loss={epoch_log['train_loss']:.4f} "
                    f"train_mpjpe={epoch_log['train_mpjpe']:.4f} "
                    f"val_loss={epoch_log['val_loss']:.4f} "
                    f"val_mpjpe={epoch_log['val_mpjpe']:.4f} "
                    f"val_bone={epoch_log['val_bone_error']:.4f} "
                    f"lr={epoch_log['lr']:.2e} "
                    f"train_sps={epoch_log['train_samples_per_sec']:.1f} "
                    f"val_sps={epoch_log['val_samples_per_sec']:.1f}"
                )

                if bool((cfg.get("experiment") or {}).get("early_stopping", True)) and epochs_without_improvement >= patience:
                    print(f"[early-stopping] Triggered at epoch {epoch}.")
                    stop_tensor = torch.tensor([1], dtype=torch.int64, device=device)
                else:
                    stop_tensor = torch.tensor([0], dtype=torch.int64, device=device)
            else:
                stop_tensor = torch.tensor([0], dtype=torch.int64, device=device)

            if is_distributed():
                dist.broadcast(stop_tensor, src=0)
            barrier()
            if int(stop_tensor.item()) == 1:
                break

        barrier()

        if is_rank0():
            if best_checkpoint_path.exists():
                _load_checkpoint(model, best_checkpoint_path, device=device)
            test_metrics = _run_eval_epoch(model, test_loader, device, cfg, desc="Test")
            test_metrics["best_epoch"] = best_epoch
            test_metrics["best_val_mpjpe"] = best_metric

            with (output_dir / "training_log.json").open("w", encoding="utf-8") as file:
                json.dump({"log": training_log}, file, indent=2, ensure_ascii=False)
            with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_val_mpjpe": best_metric,
                        "test": test_metrics,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            save_all_figures(training_log=training_log, test_metrics=test_metrics, output_dir=output_dir)
            wandb_logger.log_test_metrics(test_metrics)
            wandb_logger.log_figures_dir(output_dir / "figures")
            print(f"[done] Output saved to {output_dir}")
    finally:
        if is_rank0():
            wandb_logger.finish()
        cleanup_distributed()

    return 0


def evaluate_checkpoint(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    split_name: str = "test",
    subset_ratio: float = 1.0,
) -> dict[str, Any]:
    """Evaluate an existing checkpoint on val/test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int((cfg.get("experiment") or {}).get("seed", 42)))
    _, _, datasets = _build_datasets(cfg, subset_ratio=subset_ratio)
    loader = DataLoader(
        datasets[split_name],
        batch_size=int((cfg.get("training") or {}).get("val_per_gpu_batch_size", 16)),
        shuffle=False,
        num_workers=int((cfg.get("training") or {}).get("num_workers", 2)),
        pin_memory=bool((cfg.get("training") or {}).get("pin_memory", True)),
    )
    model = create_model(cfg).to(device)
    _load_checkpoint(model, Path(checkpoint_path), device=device)
    return _run_eval_epoch(model, loader, device, cfg, desc=f"Eval-{split_name}")
