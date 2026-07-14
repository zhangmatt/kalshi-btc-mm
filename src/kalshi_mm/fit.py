"""Fit and evaluate Models V and T from the feature/label stores (STRATEGY.md §5–§6).

Protocol enforcement: windows split chronologically 60/20/20; the final 20%
holdout is excluded from every code path unless --unlock-holdout is passed,
which prints a loud warning. Run this at ≥300 windows for screening and ≥1,000
for promotion; below that, results are smoke tests.
"""
from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import (
    FEATURE_SETS,
    TIME_BUCKETS,
    LinearModel,
    block_bootstrap_ci,
    bucket_of,
    chronological_split,
    fit_logistic,
    fit_ridge,
    save_models,
)

V_TARGET = "fwd_mid_delta_5s"
T_TARGET = "toxic_5s"


def _read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(next(iter(columns.values()))) if columns else 0
    return [{name: values[i] for name, values in columns.items()} for i in range(n)]


def load_store(pattern: str) -> dict[str, list[dict[str, Any]]]:
    """window ticker -> rows, chronologically keyed by ticker sort order."""
    store: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(pattern)):
        rows = _read_parquet_rows(path)
        if rows:
            store[str(rows[0]["ticker"])] = rows
    return store


def _matrix(
    rows: list[dict[str, Any]], features: tuple[str, ...], target: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        values = [row.get(name) for name in features]
        y = row.get(target)
        if y is None or any(value is None for value in values):
            continue
        xs.append([float(v) for v in values])
        ys.append(float(y))
    if not xs:
        return np.zeros((0, len(features))), np.zeros(0)
    return np.array(xs), np.array(ys)


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = bucket_of(float(row["seconds_to_close"]))
        if bucket is not None:
            grouped[bucket].append(row)
    return grouped


def calibration_report(store: dict[str, list[dict[str, Any]]]) -> list[str]:
    """P1: Brier/log-loss of gbm_fair vs market mid, by time bucket (resolved windows)."""
    lines = ["", "== P1 calibration: gbm_fair vs market mid (resolved windows) =="]
    lines.append(f"{'bucket_s':>12s} {'n':>7s} {'brier_gbm':>10s} {'brier_mid':>10s} {'ll_gbm':>8s} {'ll_mid':>8s}")
    for bucket in TIME_BUCKETS:
        rows = [
            row
            for window_rows in store.values()
            for row in window_rows
            if row.get("outcome_yes") is not None and bucket_of(float(row["seconds_to_close"])) == bucket
        ]
        if len(rows) < 100:
            continue
        briers = {"gbm": [], "mid": []}
        lls = {"gbm": [], "mid": []}
        for row in rows:
            outcome = float(row["outcome_yes"])
            for key, value in (("gbm", row["gbm_fair"]), ("mid", row["mid"])):
                p = min(1 - 1e-9, max(1e-9, float(value)))
                briers[key].append((p - outcome) ** 2)
                lls[key].append(-(outcome * math.log(p) + (1 - outcome) * math.log(1 - p)))
        lines.append(
            f"{f'{int(bucket[0])}-{int(bucket[1])}':>12s} {len(rows):7d} "
            f"{np.mean(briers['gbm']):10.4f} {np.mean(briers['mid']):10.4f} "
            f"{np.mean(lls['gbm']):8.4f} {np.mean(lls['mid']):8.4f}"
        )
    return lines


def _eval_model_v(
    model: LinearModel,
    windows: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Per-window IC of model prediction vs realized target on unseen windows."""
    ics: dict[str, float] = {}
    for ticker, rows in windows.items():
        pairs = []
        for row in rows:
            if bucket_of(float(row["seconds_to_close"])) != model.bucket:
                continue
            prediction = model.predict([row.get(name) for name in model.features])
            target = row.get(V_TARGET)
            if prediction is not None and target is not None:
                pairs.append((prediction, float(target)))
        if len(pairs) < 30:
            continue
        xs = np.array([p for p, _ in pairs])
        ys = np.array([y for _, y in pairs])
        if xs.std() > 0 and ys.std() > 0:
            ics[ticker] = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (xs.std() * ys.std()))
    return ics


def run_v(
    store: dict[str, list[dict[str, Any]]],
    train_ids: list[str],
    val_ids: list[str],
    out_dir: Path,
) -> list[str]:
    lines = ["", f"== Model V (ridge -> {V_TARGET}) train={len(train_ids)}w val={len(val_ids)}w =="]
    lines.append(f"{'set':>9s} {'bucket_s':>10s} {'n_train':>8s} {'val_windows':>11s} {'val_IC':>7s} {'ci_low':>7s} {'ci_high':>7s}")
    promoted: list[LinearModel] = []
    train_rows = [row for ticker in train_ids for row in store[ticker]]
    val_windows = {ticker: store[ticker] for ticker in val_ids}
    for set_name, features in FEATURE_SETS.items():
        for bucket, rows in sorted(_bucket_rows(train_rows).items()):
            x, y = _matrix(rows, features, V_TARGET)
            if len(y) < 500:
                continue
            model = fit_ridge(x, y, features=features, bucket=bucket)
            model.meta["feature_set"] = set_name
            ics = _eval_model_v(model, val_windows)
            mean, low, high = block_bootstrap_ci(list(ics.values()))
            lines.append(
                f"{set_name:>9s} {f'{int(bucket[0])}-{int(bucket[1])}':>10s} {len(y):8d} "
                f"{len(ics):11d} {mean:+7.3f} {low:+7.3f} {high:+7.3f}"
            )
            promoted.append(model)
    if promoted:
        save_models(promoted, out_dir / "model_v.json")
        lines.append(f"saved {len(promoted)} bucket models -> {out_dir / 'model_v.json'}")
    return lines


def run_t(
    fills_store: dict[str, list[dict[str, Any]]],
    train_ids: list[str],
    val_ids: list[str],
    out_dir: Path,
) -> list[str]:
    lines = ["", f"== Model T (logistic -> {T_TARGET}) train={len(train_ids)}w val={len(val_ids)}w =="]
    train_rows = [row for ticker in train_ids if ticker in fills_store for row in fills_store[ticker]]
    val_rows = [row for ticker in val_ids if ticker in fills_store for row in fills_store[ticker]]
    lines.append(f"train_fills={len(train_rows)} val_fills={len(val_rows)}")
    if not train_rows:
        lines.append("no training fills yet; skipped")
        return lines
    promoted: list[LinearModel] = []
    lines.append(f"{'set':>9s} {'bucket_s':>10s} {'n_train':>8s} {'base_rate':>9s} {'val_n':>6s} {'val_auc':>8s} {'val_brier':>9s}")
    for set_name, features in FEATURE_SETS.items():
        for bucket, rows in sorted(_bucket_rows(train_rows).items()):
            x, y = _matrix(rows, features, T_TARGET)
            if len(y) < 100 or len(set(y.tolist())) < 2:
                continue
            model = fit_logistic(x, y, features=features, bucket=bucket)
            model.meta["feature_set"] = set_name
            # validation AUC + Brier
            pairs = []
            for row in val_rows:
                if bucket_of(float(row["seconds_to_close"])) != bucket:
                    continue
                p = model.predict([row.get(name) for name in features])
                target = row.get(T_TARGET)
                if p is not None and target is not None:
                    pairs.append((p, float(target)))
            auc = _auc(pairs)
            brier = float(np.mean([(p - y_) ** 2 for p, y_ in pairs])) if pairs else float("nan")
            lines.append(
                f"{set_name:>9s} {f'{int(bucket[0])}-{int(bucket[1])}':>10s} {len(y):8d} "
                f"{float(y.mean()):9.3f} {len(pairs):6d} "
                f"{auc if auc is not None else float('nan'):8.3f} {brier:9.4f}"
            )
            promoted.append(model)
    if promoted:
        save_models(promoted, out_dir / "model_t.json")
        lines.append(f"saved {len(promoted)} bucket models -> {out_dir / 'model_t.json'}")
    return lines


def _auc(pairs: list[tuple[float, float]]) -> Optional[float]:
    positives = sorted(p for p, y in pairs if y == 1.0)
    negatives = sorted(p for p, y in pairs if y == 0.0)
    if not positives or not negatives:
        return None
    # rank-sum AUC
    wins = 0.0
    for p in positives:
        import bisect as _bisect

        wins += _bisect.bisect_left(negatives, p) + 0.5 * (
            _bisect.bisect_right(negatives, p) - _bisect.bisect_left(negatives, p)
        )
    return wins / (len(positives) * len(negatives))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit/evaluate Models V and T with holdout protection.")
    parser.add_argument("--features", default="data/features/*.parquet")
    parser.add_argument("--labels", default="data/labels/*.fills.parquet")
    parser.add_argument("--out", default="data/models")
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="include the final 20% holdout windows (ONE final promotion read only)",
    )
    args = parser.parse_args(argv)
    store = load_store(args.features)
    fills_store = load_store(args.labels)
    if not store:
        raise SystemExit("no feature windows found")
    train_ids, val_ids, holdout_ids = chronological_split(list(store.keys()))
    print(f"windows: train={len(train_ids)} val={len(val_ids)} holdout={len(holdout_ids)} (locked={not args.unlock_holdout})")
    if len(store) < 300:
        print(f"WARNING: {len(store)} windows < 300 — screening power is insufficient; treat as smoke test")
    if args.unlock_holdout:
        print("*** HOLDOUT UNLOCKED — this counts as the one promotion read (STRATEGY.md §6) ***")
        val_ids = val_ids + holdout_ids
    visible = {ticker: rows for ticker, rows in store.items() if ticker in set(train_ids) | set(val_ids)}
    out_dir = Path(args.out)
    for line in calibration_report({t: r for t, r in visible.items()}):
        print(line)
    for line in run_v(store, train_ids, val_ids, out_dir):
        print(line)
    for line in run_t(fills_store, train_ids, val_ids, out_dir):
        print(line)


if __name__ == "__main__":
    main()
