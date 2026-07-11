"""Early stopping and checkpointing.

Both watch the same validation metric (default: macro-F1, not accuracy --
with 8 roughly-imbalanced classes, accuracy can look good while the model
ignores rare classes entirely; macro-F1 penalizes that).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" else (
            score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str, filename_prefix: str, mode: str = "max"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.best_score: float | None = None
        self.best_path: Path | None = None

    def step(self, score: float, model: torch.nn.Module, epoch: int, extra: dict | None = None) -> bool:
        is_best = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if is_best:
            self.best_score = score
            self.best_path = self.checkpoint_dir / f"{self.filename_prefix}_best.pt"
            payload = {"model_state": model.state_dict(), "epoch": epoch, "score": score}
            if extra:
                payload.update(extra)
            torch.save(payload, self.best_path)
            logger.info("New best checkpoint (score=%.4f) saved to %s", score, self.best_path)

        last_path = self.checkpoint_dir / f"{self.filename_prefix}_last.pt"
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "score": score}, last_path)
        return is_best
