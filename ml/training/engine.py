"""Single-backbone training engine.

Each of the three backbones is fine-tuned independently with this engine
(transfer learning + mixed precision + focal/label-smoothed loss + cosine
warmup schedule + gradient clipping + early stopping + checkpointing).
The ensemble combiner is fit afterward on out-of-fold predictions -- see
ml/training/ensemble_fit.py -- which is the leakage-safe way to train a
stacking/weighting layer on top of base learners.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.training.callbacks import EarlyStopping, ModelCheckpoint
from ml.training.schedulers import build_warmup_cosine_scheduler

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    logits: torch.Tensor = field(repr=False)
    labels: torch.Tensor = field(repr=False)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float,
    train: bool,
) -> EpochMetrics:
    model.train(mode=train)
    total_loss, total_correct, total_seen = 0.0, 0, 0
    all_logits, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_seen += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    return EpochMetrics(
        loss=total_loss / max(1, total_seen),
        accuracy=total_correct / max(1, total_seen),
        logits=torch.cat(all_logits, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )


def run_backbone_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    base_lr: float,
    weight_decay: float,
    warmup_epochs: int,
    grad_clip_norm: float,
    mixed_precision: bool,
    early_stopping_patience: int,
    checkpoint_dir: str,
    checkpoint_prefix: str,
    metric_fn,
    metric_name: str = "val_macro_f1",
    mlflow_run=None,
    tb_writer=None,
) -> dict:
    """Trains one backbone to convergence (or early stop). `metric_fn(logits,
    labels) -> float` computes the model-selection metric (default:
    macro-F1) each epoch on the val set.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    early_stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    checkpointer = ModelCheckpoint(checkpoint_dir, checkpoint_prefix, mode="max")

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer, scheduler, scaler, grad_clip_norm, train=True)
        val_metrics = _run_epoch(model, val_loader, loss_fn, device, None, None, None, grad_clip_norm, train=False)
        val_score = metric_fn(val_metrics.logits, val_metrics.labels)

        elapsed = time.time() - t0
        logger.info(
            "[%s] epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  %s=%.4f  (%.1fs)",
            checkpoint_prefix, epoch, epochs, train_metrics.loss, val_metrics.loss,
            val_metrics.accuracy, metric_name, val_score, elapsed,
        )

        if tb_writer is not None:
            tb_writer.add_scalar(f"{checkpoint_prefix}/train_loss", train_metrics.loss, epoch)
            tb_writer.add_scalar(f"{checkpoint_prefix}/val_loss", val_metrics.loss, epoch)
            tb_writer.add_scalar(f"{checkpoint_prefix}/{metric_name}", val_score, epoch)

        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics({
                f"{checkpoint_prefix}_train_loss": train_metrics.loss,
                f"{checkpoint_prefix}_val_loss": val_metrics.loss,
                f"{checkpoint_prefix}_{metric_name}": val_score,
            }, step=epoch)

        checkpointer.step(val_score, model, epoch)
        history.append({"epoch": epoch, "train_loss": train_metrics.loss, "val_loss": val_metrics.loss, metric_name: val_score})

        if early_stopper.step(val_score):
            logger.info("Early stopping triggered for %s at epoch %d (best %s=%.4f)", checkpoint_prefix, epoch, metric_name, early_stopper.best_score)
            break

    # Restore best checkpoint before returning, so the caller always gets the
    # best-validation-score weights, not whatever the last epoch happened to be.
    best_state = torch.load(checkpointer.best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])

    return {"history": history, "best_score": checkpointer.best_score, "best_checkpoint": str(checkpointer.best_path)}
