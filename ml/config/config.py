"""
Central configuration for the ML pipeline.

Single source of truth for class labels, image geometry, and training
hyperparameters. Everything here is overridable via environment variables
or a YAML file so experiments don't require code edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CLASS_NAMES: list[str] = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
NUM_CLASSES: int = len(CLASS_NAMES)


@dataclass
class DataConfig:
    dataset_root: str = "dataset_blood_group"
    image_size: int = 224
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    k_folds: int = 5
    # If filenames encode a subject id (e.g. "subj014_finger02_03.png"), set the
    # regex group used to extract it so k-fold splitting keeps a subject's
    # images together and never leaks the same subject across train/val/test.
    subject_id_regex: str = r"^(subj\d+|[A-Za-z0-9]+?)_"
    num_workers: int = 0


@dataclass
class ModelConfig:
    backbones: tuple[str, ...] = ("efficientnetv2_rw_s", "vit_base_patch16_224", "convnext_tiny")
    pretrained: bool = True
    drop_rate: float = 0.2
    ensemble_strategy: str = "weighted_softmax"  # "weighted_softmax" | "stacking"


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    base_lr: float = 3e-4
    weight_decay: float = 5e-5
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    use_focal_loss: bool = True
    mixed_precision: bool = True
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_macro_f1"
    grad_clip_norm: float = 1.0
    tta_num_augments: int = 6
    seed: int = 42


@dataclass
class PathsConfig:
    output_dir: str = "artifacts"
    checkpoint_dir: str = "artifacts/checkpoints"
    log_dir: str = "artifacts/logs"
    # mlflow's legacy filesystem store is in maintenance mode as of mlflow 3.x;
    # sqlite is the currently-recommended lightweight backend for single-node use.
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    tensorboard_dir: str = "artifacts/tensorboard"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, yaml_path: str | None = None) -> Config:
        cfg = cls()
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
            for section in ("data", "model", "train", "paths"):
                if section in raw:
                    section_obj = getattr(cfg, section)
                    for k, v in raw[section].items():
                        if hasattr(section_obj, k):
                            setattr(section_obj, k, v)
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        env_map = {
            "BP_DATASET_ROOT": ("data", "dataset_root"),
            "BP_IMAGE_SIZE": ("data", "image_size"),
            "BP_EPOCHS": ("train", "epochs"),
            "BP_BATCH_SIZE": ("train", "batch_size"),
            "BP_BASE_LR": ("train", "base_lr"),
        }
        for env_key, (section, attr) in env_map.items():
            if env_key in os.environ:
                section_obj = getattr(self, section)
                current = getattr(section_obj, attr)
                cast = type(current)
                setattr(section_obj, attr, cast(os.environ[env_key]))

    def ensure_dirs(self) -> None:
        for p in (self.paths.output_dir, self.paths.checkpoint_dir,
                  self.paths.log_dir, self.paths.tensorboard_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
        if self.paths.mlflow_tracking_uri.startswith("sqlite:///"):
            Path(self.paths.mlflow_tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
