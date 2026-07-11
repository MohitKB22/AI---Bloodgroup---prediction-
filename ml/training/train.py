"""End-to-end training entrypoint.

    python -m ml.training.train --dataset-root dataset_blood_group

Pipeline:
  1. Build a manifest from the class-subfolder dataset layout, carve out a
     held-out test set, and build StratifiedGroupKFold splits over the rest
     -- all grouped by subject id so no fold or the test set ever shares a
     subject with another (see ml/data/splits.py).
  2. For each backbone (EfficientNetV2, ViT, ConvNeXt) independently: run
     k-fold cross-validation with transfer learning, mixed precision, focal
     loss + label smoothing, cosine warmup schedule, gradient clipping,
     early stopping, and checkpointing. Collect out-of-fold (OOF)
     predictions as we go.
  3. Fit the ensemble combiner (weighted-softmax or stacking) on the
     concatenated OOF predictions -- never on in-sample predictions.
  4. Fit temperature-scaling calibration, also on OOF predictions.
  5. Evaluate the fully-assembled ensemble exactly once on the held-out test
     set (optionally with TTA) and write the final metrics/plots.
  6. Save a model bundle manifest that ml/inference/predictor.py can load.

This script is compute-orchestration only; every actual "how" (loss
function, scheduler, calibration math, leakage-safe splitting) lives in the
modules it imports, so the pieces stay independently testable.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.config.config import CLASS_NAMES, Config
from ml.data.dataset import FingerprintDataset
from ml.data.splits import build_manifest, compute_class_alpha, make_kfold_splits
from ml.evaluation.calibration import fit_temperature
from ml.evaluation.evaluate import full_evaluation_report
from ml.evaluation.metrics import macro_f1_from_logits
from ml.models.ensemble import EnsembleModel
from ml.models.losses import FocalLossWithLabelSmoothing
from ml.training.engine import run_backbone_training
from ml.training.ensemble_fit import fit_stacking_meta_learner, fit_weighted_softmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BloodPrint ensemble.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config overriding defaults.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbones", type=str, default=None, help="Comma-separated subset, e.g. 'efficientnetv2_rw_s,convnext_tiny'")
    parser.add_argument("--quick-test", action="store_true",
                         help="Tiny epoch/fold counts, for pipeline-wiring verification only -- "
                              "never use this flag's output as a real model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _build_loader(manifest_subset, image_size, train, batch_size, num_workers):
    ds = FingerprintDataset(manifest_subset, image_size=image_size, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=False)


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.backbones:
        cfg.model.backbones = tuple(b.strip() for b in args.backbones.split(","))
    if args.quick_test:
        cfg.train.epochs = min(cfg.train.epochs, 2)
        cfg.data.k_folds = min(cfg.data.k_folds, 2)
        cfg.train.early_stopping_patience = 1
        logger.warning("--quick-test set: this run verifies wiring only, NOT a usable model.")

    cfg.ensure_dirs()
    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
        mlflow.set_experiment("bloodprint-ensemble")
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "epochs": cfg.train.epochs, "batch_size": cfg.train.batch_size,
            "base_lr": cfg.train.base_lr, "backbones": cfg.model.backbones,
            "ensemble_strategy": cfg.model.ensemble_strategy,
        })
    except ImportError:
        logger.warning("mlflow not installed; skipping experiment tracking.")
    except Exception as exc:  # tracking-backend issues should never abort a training run
        logger.warning("mlflow experiment tracking unavailable (%s); continuing without it.", exc)
        mlflow_run = None

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(cfg.paths.tensorboard_dir)
    except ImportError:
        logger.warning("tensorboard not installed; skipping TB logging.")

    # ---- 1. Manifest + leakage-safe splits ----
    manifest = build_manifest(cfg.data.dataset_root, cfg.data.subject_id_regex)
    test_manifest, folds = make_kfold_splits(manifest, cfg.data.k_folds, cfg.data.test_fraction, cfg.train.seed)
    test_subjects = {e.subject_id for e in test_manifest}
    trainval_manifest = [e for e in manifest if e.subject_id not in test_subjects]
    logger.info("Dataset: %d total | %d trainval | %d held-out test | %d folds",
                len(manifest), len(trainval_manifest), len(test_manifest), len(folds))

    labels_trainval = np.array([e.label for e in trainval_manifest])

    # ---- 2. Per-backbone k-fold training + OOF collection ----
    oof_probs: dict[str, np.ndarray] = {
        name: np.full((len(trainval_manifest), len(CLASS_NAMES)), np.nan) for name in cfg.model.backbones
    }
    backbone_best_checkpoints: dict[str, str] = {}
    fold_metric_summary: dict[str, list[float]] = {name: [] for name in cfg.model.backbones}

    for backbone_name in cfg.model.backbones:
        logger.info("=== Training backbone: %s ===", backbone_name)
        best_overall_score = -np.inf

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold_prefix = f"{backbone_name}_fold{fold_idx}"
            train_subset = [trainval_manifest[i] for i in train_idx]
            val_subset = [trainval_manifest[i] for i in val_idx]

            train_loader = _build_loader(train_subset, cfg.data.image_size, True, cfg.train.batch_size, cfg.data.num_workers)
            val_loader = _build_loader(val_subset, cfg.data.image_size, False, cfg.train.batch_size, cfg.data.num_workers)

            alpha = torch.tensor(compute_class_alpha(train_subset), dtype=torch.float32) if cfg.train.use_focal_loss else None
            loss_fn = FocalLossWithLabelSmoothing(
                num_classes=len(CLASS_NAMES), gamma=cfg.train.focal_gamma,
                label_smoothing=cfg.train.label_smoothing, alpha=alpha,
            )

            from ml.models.backbones import build_backbone
            model = build_backbone(backbone_name, len(CLASS_NAMES), pretrained=cfg.model.pretrained, drop_rate=cfg.model.drop_rate)

            result = run_backbone_training(
                model=model, train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn, device=device,
                epochs=cfg.train.epochs, base_lr=cfg.train.base_lr, weight_decay=cfg.train.weight_decay,
                warmup_epochs=cfg.train.warmup_epochs, grad_clip_norm=cfg.train.grad_clip_norm,
                mixed_precision=cfg.train.mixed_precision, early_stopping_patience=cfg.train.early_stopping_patience,
                checkpoint_dir=cfg.paths.checkpoint_dir, checkpoint_prefix=fold_prefix,
                metric_fn=macro_f1_from_logits, mlflow_run=mlflow_run, tb_writer=tb_writer,
            )
            fold_metric_summary[backbone_name].append(result["best_score"])

            # Collect this fold's OOF predictions (val_idx were never trained on).
            model.eval()
            with torch.no_grad():
                for batch_i, (x, _y) in enumerate(val_loader):
                    probs = F.softmax(model(x.to(device)), dim=-1).cpu().numpy()
                    start = batch_i * cfg.train.batch_size
                    oof_probs[backbone_name][val_idx[start:start + len(probs)]] = probs

            if result["best_score"] > best_overall_score:
                best_overall_score = result["best_score"]
                backbone_best_checkpoints[backbone_name] = result["best_checkpoint"]

        scores = fold_metric_summary[backbone_name]
        logger.info("%s: fold macro-F1 = %.4f +/- %.4f across %d folds", backbone_name, np.mean(scores), np.std(scores), len(scores))

    missing = np.isnan(oof_probs[cfg.model.backbones[0]]).any(axis=1)
    if missing.any():
        logger.warning("%d/%d trainval samples never landed in a validation fold; excluding from OOF ensemble fit.",
                        missing.sum(), len(missing))
    keep_mask = ~missing
    oof_probs_clean = {k: v[keep_mask] for k, v in oof_probs.items()}
    labels_clean = labels_trainval[keep_mask]

    # ---- 3. Fit ensemble combiner on OOF predictions ----
    ensemble_weights, meta_learner = None, None
    if cfg.model.ensemble_strategy == "weighted_softmax":
        ensemble_weights = fit_weighted_softmax(oof_probs_clean, labels_clean)
        logger.info("Fitted ensemble weights: %s", ensemble_weights)
    else:
        meta_learner = fit_stacking_meta_learner(oof_probs_clean, labels_clean, num_classes=len(CLASS_NAMES))
        logger.info("Fitted stacking meta-learner on %d OOF samples.", len(labels_clean))

    # ---- 4. Combine OOF probs the same way the ensemble will at serving time, then calibrate ----
    names = list(oof_probs_clean.keys())
    stacked_oof = np.stack([oof_probs_clean[n] for n in names], axis=0)  # (M, N, C)
    if ensemble_weights is not None:
        w = np.array([ensemble_weights[n] for n in names]).reshape(-1, 1, 1)
        combined_oof = (w * stacked_oof).sum(axis=0)
    else:
        concat = np.concatenate([oof_probs_clean[n] for n in names], axis=-1)
        with torch.no_grad():
            combined_oof = F.softmax(meta_learner(torch.tensor(concat, dtype=torch.float32)), dim=-1).numpy()

    pseudo_logits = torch.log(torch.tensor(combined_oof, dtype=torch.float32).clamp_min(1e-12))
    temperature = fit_temperature(pseudo_logits, torch.tensor(labels_clean, dtype=torch.long))
    logger.info("Fitted ensemble temperature: %.4f", temperature)

    # ---- 5. Assemble final EnsembleModel and evaluate ONCE on the held-out test set ----
    final_model = EnsembleModel(
        backbone_names=tuple(cfg.model.backbones), num_classes=len(CLASS_NAMES),
        pretrained=False, strategy=cfg.model.ensemble_strategy,
    )
    for name, ckpt_path in backbone_best_checkpoints.items():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        final_model.backbones[name].load_state_dict(state["model_state"])
    if ensemble_weights is not None:
        with torch.no_grad():
            w = torch.tensor([ensemble_weights[n] for n in final_model.backbone_names])
            final_model.raw_weights.copy_(torch.log(w.clamp_min(1e-8)))
    else:
        final_model.meta_learner.load_state_dict(meta_learner.state_dict())
    final_model.to(device).eval()

    test_loader = _build_loader(test_manifest, cfg.data.image_size, False, cfg.train.batch_size, cfg.data.num_workers)
    test_probs, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = final_model(x.to(device))
            test_probs.append(out["ensemble_probs"].cpu().numpy())
            test_labels.append(y.numpy())
    test_probs = np.concatenate(test_probs, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    calibrated_test_probs = F.softmax(
        torch.log(torch.tensor(test_probs).clamp_min(1e-12)) / temperature, dim=-1
    ).numpy()

    report = full_evaluation_report(
        probs=calibrated_test_probs, labels=test_labels, output_dir=cfg.paths.output_dir, split_name="test",
        temperature=temperature,
        extra_metadata={"n_test": len(test_manifest), "fold_macro_f1_by_backbone":
                         {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in fold_metric_summary.items()}},
    )
    logger.info("HELD-OUT TEST RESULTS: accuracy=%.4f macro_f1=%.4f roc_auc_macro=%s ece=%.4f",
                report["accuracy"], report["macro_f1"], report["roc_auc_macro_ovr"], report["ece"])

    # ---- 6. Save model bundle for serving ----
    bundle_dir = Path(cfg.paths.output_dir) / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_backbone_ckpts = {}
    for name, ckpt_path in backbone_best_checkpoints.items():
        dest = bundle_dir / f"{name}.pt"
        dest.write_bytes(Path(ckpt_path).read_bytes())
        bundle_backbone_ckpts[name] = dest.name

    bundle = {
        "version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "class_names": CLASS_NAMES,
        "image_size": cfg.data.image_size,
        "ensemble_strategy": cfg.model.ensemble_strategy,
        "backbone_checkpoints": bundle_backbone_ckpts,
        "ensemble_weights": ensemble_weights,
        "temperature": temperature,
        "test_metrics_summary": {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"], "ece": report["ece"]},
    }
    if meta_learner is not None:
        meta_path = bundle_dir / "meta_learner.pt"
        torch.save(meta_learner.state_dict(), meta_path)
        bundle["meta_learner_checkpoint"] = meta_path.name

    with open(bundle_dir / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info("Model bundle saved to %s", bundle_dir / "bundle.json")

    if mlflow_run is not None:
        import mlflow
        mlflow.log_metrics({"test_accuracy": report["accuracy"], "test_macro_f1": report["macro_f1"], "test_ece": report["ece"]})
        mlflow.end_run()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
