import cv2
import pytest

from ml.config.config import CLASS_NAMES
from ml.data.splits import build_manifest, compute_class_alpha, make_kfold_splits
from ml.tests.synthetic_data import synthetic_fingerprint


@pytest.fixture(scope="module")
def synthetic_dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset_blood_group")
    for label_idx, cname in enumerate(CLASS_NAMES):
        class_dir = root / cname
        class_dir.mkdir()
        for subj in range(4):
            for capture in range(2):
                seed = label_idx * 100 + subj * 10 + capture
                img = synthetic_fingerprint(200, seed=seed)
                cv2.imwrite(str(class_dir / f"subj{label_idx}{subj:02d}_finger01_{capture}.png"), img)
    return str(root)


def test_manifest_finds_all_images(synthetic_dataset_root):
    manifest = build_manifest(synthetic_dataset_root, r"^(subj\d+|[A-Za-z0-9]+?)_")
    assert len(manifest) == len(CLASS_NAMES) * 4 * 2


def test_manifest_raises_on_missing_dataset():
    with pytest.raises(FileNotFoundError):
        build_manifest("/nonexistent/path/xyz", r"^(subj\d+)_")


def test_kfold_splits_have_zero_subject_leakage(synthetic_dataset_root):
    manifest = build_manifest(synthetic_dataset_root, r"^(subj\d+|[A-Za-z0-9]+?)_")
    test_set, folds = make_kfold_splits(manifest, n_splits=3, test_fraction=0.2, seed=42)

    test_subjects = {e.subject_id for e in test_set}
    trainval = [e for e in manifest if e.subject_id not in test_subjects]
    trainval_subjects = {e.subject_id for e in trainval}

    assert trainval_subjects.isdisjoint(test_subjects), "a subject appears in both trainval and test"

    for train_idx, val_idx in folds:
        train_subjects = {trainval[i].subject_id for i in train_idx}
        val_subjects = {trainval[i].subject_id for i in val_idx}
        assert train_subjects.isdisjoint(val_subjects), "a subject appears in both a fold's train and val split"


def test_class_alpha_upweights_rare_classes(synthetic_dataset_root):
    manifest = build_manifest(synthetic_dataset_root, r"^(subj\d+|[A-Za-z0-9]+?)_")
    # Artificially drop most AB- samples to simulate class imbalance.
    ab_neg_idx = CLASS_NAMES.index("AB-")
    trimmed = [e for e in manifest if e.label != ab_neg_idx] + [e for e in manifest if e.label == ab_neg_idx][:1]

    alpha = compute_class_alpha(trimmed)
    assert alpha[ab_neg_idx] > alpha.mean(), "the rarer class should get a larger alpha weight"
