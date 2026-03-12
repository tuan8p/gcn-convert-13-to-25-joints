"""Optional W&B logging with graceful fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WandbLogger:
    """No-op friendly wrapper around `wandb`."""

    def __init__(
        self,
        enabled: bool,
        project: str | None,
        entity: str | None,
        run_name: str | None,
        tags: list[str],
    ) -> None:
        self._enabled = enabled and bool(project or os.environ.get("WANDB_PROJECT"))
        self._run: Any = None

        if not self._enabled:
            return

        try:
            import wandb

            self._run = wandb.init(
                project=project or os.environ.get("WANDB_PROJECT", "ssr-gcn"),
                entity=entity or os.environ.get("WANDB_ENTITY") or None,
                name=run_name,
                tags=tags or [],
                reinit=True,
            )
        except Exception as exc:
            print(f"[wandb] WARNING: init failed ({exc}). Proceeding without wandb.")
            self._enabled = False
            self._run = None

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], run_name: str | None = None) -> "WandbLogger":
        wandb_cfg = cfg.get("wandb") or {}
        return cls(
            enabled=bool(wandb_cfg.get("enabled", False)),
            project=wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT"),
            entity=wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY"),
            run_name=run_name or wandb_cfg.get("run_name"),
            tags=list(wandb_cfg.get("tags") or []),
        )

    @property
    def active(self) -> bool:
        return self._enabled and self._run is not None

    def log_config(self, cfg: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            import wandb

            wandb.config.update(cfg, allow_val_change=True)
        except Exception as exc:
            print(f"[wandb] log_config failed: {exc}")

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        if not self.active:
            return
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as exc:
            print(f"[wandb] log_epoch failed: {exc}")

    def log_test_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            import wandb

            flat: dict[str, Any] = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat[f"test/{key}_{sub_key}"] = sub_value
                else:
                    flat[f"test/{key}"] = value
            wandb.log(flat)
        except Exception as exc:
            print(f"[wandb] log_test_metrics failed: {exc}")

    def log_figures_dir(self, figures_dir: Path) -> None:
        if not self.active:
            return
        try:
            import wandb

            for png in figures_dir.glob("*.png"):
                wandb.log({f"figures/{png.stem}": wandb.Image(str(png))})
        except Exception as exc:
            print(f"[wandb] log_figures_dir failed: {exc}")

    def finish(self) -> None:
        if not self.active:
            return
        try:
            import wandb

            wandb.finish()
        except Exception as exc:
            print(f"[wandb] finish failed: {exc}")
