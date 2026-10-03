"""Train the DistilBERT length predictor + prompt-length baseline, evaluate, save artifacts.

    python -m ml.train [--data PATH] [--epochs N] [--loss-lambda 0.95] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ml.baseline import PromptLengthBaseline
from ml.config import PROJECT_ROOT, Config, resolve_device
from ml.data import (LengthDataset, assert_disjoint_splits, compute_quantile_bins, generate_synthetic_dataset,
                     load_jsonl, make_collate, save_jsonl, split_dataset)
from ml.evaluate import compute_metrics, evaluate_split
from ml.model import LengthPredictorModel, joint_loss
from ml.predictor import LengthPredictor


SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_prompts.jsonl"
SYNTHETIC_WARNING = "SYNTHETIC or MOCK labels: numbers validate the code path only."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, config, mse_scale, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "mse": 0.0}
    preds, actual = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            expected = model.expected_tokens(logits)
            loss, ce, mse = joint_loss(logits, batch["soft_targets"], expected, batch["target_tokens"],
                                       config.loss_lambda, mse_scale)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = len(batch["target_tokens"])
            for k, v in (("loss", loss), ("ce", ce), ("mse", mse)):
                totals[k] += v.item() * n
            preds += expected.detach().cpu().tolist()
            actual += batch["target_tokens"].cpu().tolist()
    out = {k: v / len(actual) for k, v in totals.items()}
    out["mae"] = compute_metrics(preds, actual)["mae"]
    return out


def train(config: Config) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.output_dir)

    # --- data ---------------------------------------------------------------
    data_path = Path(config.data_path)
    if not data_path.exists():
        if data_path.resolve() != SYNTHETIC_DATA_PATH.resolve():
            raise FileNotFoundError(f"label file {data_path} not found; run datagen.generate_labels first")
        # SYNTHETIC: only the placeholder dataset is ever auto-generated.
        save_jsonl(generate_synthetic_dataset(seed=config.seed), data_path)
        print(f"generated SYNTHETIC dataset at {data_path}")
    examples = load_jsonl(data_path, config.target_field, config.allow_mock_labels)
    synthetic = any(ex.get("synthetic") or ex.get("is_mock") for ex in examples)
    train_ex, val_ex, test_ex = split_dataset(examples, config.train_frac, config.val_frac, config.seed,
                                              config.split_strategy)
    assert_disjoint_splits(train_ex, val_ex, test_ex)
    print(f"split ({config.split_strategy}): train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(
        {name: [ex["id"] for ex in split] for name, split in
         (("train", train_ex), ("val", val_ex), ("test", test_ex))}, indent=2))

    # Bins and MSE scale use TRAINING targets only (no val/test leakage).
    train_targets = np.array([ex["target_output_tokens"] for ex in train_ex], dtype=np.float64)
    boundaries, centers = compute_quantile_bins(train_targets, config.n_bins)
    mse_scale = config.mse_scale or float(train_targets.std()) or 1.0

    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    collate = make_collate(tokenizer, config.max_length)
    train_loader = DataLoader(LengthDataset(train_ex, boundaries, config.soft_label_scale),
                              batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(LengthDataset(val_ex, boundaries, config.soft_label_scale),
                            batch_size=config.batch_size, collate_fn=collate)

    # --- model ----------------------------------------------------------------
    model = LengthPredictorModel(config.backbone, config.n_bins, centers, config.pooling,
                                 config.head_hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    history = []
    for epoch in range(1, config.epochs + 1):
        tr = run_epoch(model, train_loader, config, mse_scale, device, optimizer)
        va = run_epoch(model, val_loader, config, mse_scale, device)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch}: train loss={tr['loss']:.4f} (ce={tr['ce']:.3f} mse={tr['mse']:.3f}) "
              f"mae={tr['mae']:.1f} | val loss={va['loss']:.4f} mae={va['mae']:.1f}")

    # --- save + evaluate --------------------------------------------------------
    predictor = LengthPredictor(model, tokenizer, config, boundaries, centers, mse_scale, device)
    predictor.save(out_dir)
    baseline = PromptLengthBaseline.from_spec(config.prompt_token_counter, tokenizer)
    baseline.fit([ex["prompt"] for ex in train_ex], train_targets)
    baseline.save(out_dir / "baseline.json")

    results = {
        "data_path": str(data_path),
        "target_field": config.target_field,
        "history": history,
        "val": evaluate_split(predictor, baseline, val_ex, config),
        "test": evaluate_split(predictor, baseline, test_ex, config),
    }
    if synthetic:
        results["warning"] = SYNTHETIC_WARNING
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print("test:", json.dumps(results["test"], indent=2))
    print(f"artifacts saved to {out_dir}")
    return results


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--real", action="store_true",
                     help="train on real Llama labels (Config.for_real_data) instead of the synthetic file")
    defaults = Config.for_real_data() if pre.parse_known_args(argv)[0].real else Config()
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--data", default=defaults.data_path)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--prompt-token-counter", default=defaults.prompt_token_counter,
                        help="baseline tokenizer: distilbert | llamacpp:<url> | hf:<name>")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--loss-lambda", type=float, default=defaults.loss_lambda)
    parser.add_argument("--pooling", choices=["mean", "cls"], default=defaults.pooling)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    config = dataclasses.replace(
        defaults, data_path=args.data, output_dir=args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, loss_lambda=args.loss_lambda,
        pooling=args.pooling, device=args.device, seed=args.seed,
        prompt_token_counter=args.prompt_token_counter)
    return train(config)


if __name__ == "__main__":
    main()
