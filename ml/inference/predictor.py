"""Serving-time inference.

This is the single entry point the FastAPI backend calls. It composes, in
order: preprocessing + quality assessment -> ensemble forward pass
(optionally with TTA) -> temperature-scaled calibration -> per-backbone
explainability (Grad-CAM for the two CNNs, attention rollout for ViT).

Calibration note: the ensemble's combined output is already a probability
distribution (a weighted average of per-backbone softmaxes, or a
meta-learner's softmax), not a single model's pre-softmax logits. We treat
log(combined_probs) as the "logit" input to temperature scaling -- this is
a well-defined rescaling of any probabilistic classifier's output
(softmax(log(p)) recovers p exactly at T=1) and lets us reuse the same
temperature-scaling machinery without pretending the ensemble has logits
it doesn't.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.config.config import CLASS_NAMES
from ml.evaluation.attention_rollout import rollout_heatmap
from ml.evaluation.gradcam import GradCAM
from ml.evaluation.viz_utils import overlay_heatmap
from ml.inference.tta import aggregate_tta_probs, build_tta_batch
from ml.models.backbones import CNNBackbone, ViTBackbone
from ml.models.ensemble import EnsembleModel
from ml.preprocessing.pipeline import FingerprintPreprocessor

RESEARCH_DISCLAIMER = (
    "This prediction comes from a research prototype exploring a statistically "
    "contested hypothesis (that fingerprint ridge patterns correlate with ABO "
    "blood group). It is NOT a validated diagnostic method. Do not use this "
    "output for any medical, clinical, or blood-donation/transfusion decision. "
    "See MODEL_CARD.md for known limitations."
)


@dataclass
class ExplainabilityAsset:
    backbone: str
    method: str  # "grad_cam" | "attention_rollout"
    overlay_png_path: str


@dataclass
class PredictionResult:
    predicted_label: str
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    confidence_calibrated: float
    confidence_raw: float
    ensemble_weights: dict[str, float]
    quality: dict
    explainability: list[ExplainabilityAsset]
    tta_used: bool
    tta_num_augments: int
    temperature: float
    model_version: str
    disclaimer: str = RESEARCH_DISCLAIMER
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        d = {
            "predicted_label": self.predicted_label,
            "raw_probabilities": self.raw_probabilities,
            "calibrated_probabilities": self.calibrated_probabilities,
            "confidence_calibrated": self.confidence_calibrated,
            "confidence_raw": self.confidence_raw,
            "ensemble_weights": self.ensemble_weights,
            "quality": self.quality,
            "explainability": [e.__dict__ for e in self.explainability],
            "tta_used": self.tta_used,
            "tta_num_augments": self.tta_num_augments,
            "temperature": self.temperature,
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
            "generated_at": self.generated_at,
        }
        return d


class BloodGroupPredictor:
    def __init__(
        self,
        model: EnsembleModel,
        temperature: float,
        class_names: list[str] = CLASS_NAMES,
        image_size: int = 224,
        model_version: str = "dev",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.temperature = temperature
        self.class_names = class_names
        self.image_size = image_size
        self.model_version = model_version
        self.device = torch.device(device)
        self.preprocessor = FingerprintPreprocessor(target_size=image_size)

    @classmethod
    def from_bundle(cls, bundle_path: str, device: str = "cpu") -> BloodGroupPredictor:
        bundle_dir = Path(bundle_path).parent
        with open(bundle_path) as f:
            bundle = json.load(f)

        model = EnsembleModel(
            backbone_names=tuple(bundle["backbone_checkpoints"].keys()),
            num_classes=len(bundle["class_names"]),
            pretrained=False,
            strategy=bundle["ensemble_strategy"],
        )
        for name, ckpt_rel in bundle["backbone_checkpoints"].items():
            state = torch.load(bundle_dir / ckpt_rel, map_location=device, weights_only=False)
            model.backbones[name].load_state_dict(state["model_state"])

        if bundle["ensemble_strategy"] == "weighted_softmax":
            with torch.no_grad():
                weights = torch.tensor([bundle["ensemble_weights"][n] for n in model.backbone_names])
                model.raw_weights.copy_(torch.log(weights.clamp_min(1e-8)))
        elif bundle.get("meta_learner_checkpoint"):
            meta_state = torch.load(bundle_dir / bundle["meta_learner_checkpoint"], map_location=device, weights_only=False)
            model.meta_learner.load_state_dict(meta_state)

        return cls(
            model=model,
            temperature=bundle["temperature"],
            class_names=bundle["class_names"],
            image_size=bundle["image_size"],
            model_version=bundle.get("version", "unknown"),
            device=device,
        )

    @torch.no_grad()
    def _forward_probs(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch.to(self.device))
        return out["ensemble_probs"].cpu()

    def predict(
        self,
        raw_image: np.ndarray,
        use_tta: bool = True,
        tta_num_augments: int = 6,
        explain: bool = True,
    ) -> PredictionResult:
        pre = self.preprocessor(raw_image)

        if use_tta:
            batch = build_tta_batch(self.preprocessor, pre.enhanced_display, tta_num_augments)
            probs_per_view = self._forward_probs(batch)
            raw_probs = aggregate_tta_probs(probs_per_view)
        else:
            tensor = torch.from_numpy(pre.model_input.transpose(2, 0, 1)).float().unsqueeze(0)
            raw_probs = self._forward_probs(tensor)[0]

        calibrated_probs = F.softmax(torch.log(raw_probs.clamp_min(1e-12)) / self.temperature, dim=-1)

        predicted_idx = int(calibrated_probs.argmax())
        predicted_label = self.class_names[predicted_idx]

        explain_assets: list[ExplainabilityAsset] = []
        if explain:
            explain_assets = self._generate_explainability(pre, predicted_idx)

        return PredictionResult(
            predicted_label=predicted_label,
            raw_probabilities=dict(zip(self.class_names, (float(p) for p in raw_probs), strict=True)),
            calibrated_probabilities=dict(zip(self.class_names, (float(p) for p in calibrated_probs), strict=True)),
            confidence_calibrated=float(calibrated_probs[predicted_idx]),
            confidence_raw=float(raw_probs[predicted_idx]),
            ensemble_weights=self.model.current_weights(),
            quality=pre.quality.to_dict(),
            explainability=explain_assets,
            tta_used=use_tta,
            tta_num_augments=tta_num_augments if use_tta else 0,
            temperature=self.temperature,
            model_version=self.model_version,
        )

    def _generate_explainability(self, pre, predicted_idx: int) -> list[ExplainabilityAsset]:
        tensor = torch.from_numpy(pre.model_input.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        class_idx = torch.tensor([predicted_idx], device=self.device)
        assets: list[ExplainabilityAsset] = []
        out_dir = Path("artifacts/explanations")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")

        for name, backbone in self.model.backbones.items():
            if isinstance(backbone, CNNBackbone):
                cam = GradCAM(backbone)
                try:
                    heatmap, _ = cam.generate(tensor, class_idx=class_idx)
                    overlay_img = cam.overlay(heatmap[0], pre.enhanced_display)
                    method = "grad_cam"
                finally:
                    cam.close()
            elif isinstance(backbone, ViTBackbone):
                _ = backbone(tensor)  # populate attention maps for this input
                heatmap = rollout_heatmap(backbone, patch_grid_size=self.image_size // 16)
                overlay_img = overlay_heatmap(heatmap[0], pre.enhanced_display)
                method = "attention_rollout"
            else:
                continue

            import cv2
            out_path = out_dir / f"{timestamp}_{name}_{method}.png"
            cv2.imwrite(str(out_path), overlay_img)
            assets.append(ExplainabilityAsset(backbone=name, method=method, overlay_png_path=str(out_path)))

        return assets
