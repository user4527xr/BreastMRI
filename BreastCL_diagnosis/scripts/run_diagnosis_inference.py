from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_diagnosis_dataset import PublicDiagnosisDataset
from module.util import get_multimodal_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public diagnosis inference for BreastCL.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "diagnosis_inference.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model"]
    args = SimpleNamespace(**model_cfg)
    t2_size = (1, 384, 256, 48)
    dwi_size = (1, 256, 128, 32)
    dce_size = (6, 384, 256, 128)

    model = get_multimodal_model(
        args,
        model_cfg["fusion_tag"],
        (model_cfg["t2_model_tag"], model_cfg["dwi_model_tag"], model_cfg["dce_model_tag"]),
        (t2_size, dwi_size, dce_size),
        model_cfg.get("num_classes", 2),
    ).to(device)

    ckpt_path = Path(model_cfg["checkpoint_path"])
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def safe_divide(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    metrics = {
        "num_cases": int(len(labels)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_divide(tp + tn, len(labels)),
        "sensitivity": safe_divide(tp, np.sum(labels == 1)),
        "specificity": safe_divide(tn, np.sum(labels == 0)),
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }

    if len(np.unique(labels)) > 1:
        metrics["average_precision"] = float(average_precision_score(labels, probs))
        metrics["auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["average_precision"] = None
        metrics["auc"] = None

    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    runtime_cfg = cfg["runtime"]
    data_cfg = cfg["data"]
    output_dir = Path(runtime_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(runtime_cfg.get("device", "cpu"))
    print(f"Using device: {device}")

    csv_path = Path(data_cfg["csv_path"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    dataset = PublicDiagnosisDataset(
        csv_path=csv_path,
        data_root=data_cfg.get("data_root"),
        label_column="malignant",
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg, device)

    all_indices, all_subjects = [], []
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            t2 = batch["t2"].to(device)
            dwi = batch["dwi"].to(device)
            sub = batch["sub"].to(device)
            logits = model(sub, dwi, t2)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_indices.extend(batch["index"].cpu().tolist())
            all_subjects.extend(batch["subject"])
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    results_df = dataset.df.copy()
    results_df["index"] = all_indices
    results_df["Subject"] = all_subjects
    results_df["Probability"] = all_probs
    results_df["Prediction"] = all_preds
    results_df["GT"] = all_labels
    results_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    valid_label_mask = np.array(all_labels) >= 0
    if valid_label_mask.any():
        labels = np.array(all_labels)[valid_label_mask]
        probs = np.array(all_probs)[valid_label_mask]
        preds = np.array(all_preds)[valid_label_mask]
        metrics = compute_metrics(labels, probs, preds)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print("No labels found in CSV. Saved predictions only.")


if __name__ == "__main__":
    main()
