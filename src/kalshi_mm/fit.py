"""Fit and evaluate Models V and T from the feature/label stores (STRATEGY.md §5–§6).

Protocol enforcement: windows split chronologically 60/20/20; the final 20%
holdout is excluded from every code path unless --unlock-holdout is passed,
which prints a loud warning. Run this at ≥300 windows for screening and ≥1,000
for promotion; below that, results are smoke tests.
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import (
    FEATURE_SETS,
    T_MIRRORED_FEATURES,
    TIME_BUCKETS,
    LinearModel,
    block_bootstrap_ci,
    bucket_of,
    chronological_split,
    fit_logistic,
    fit_ridge,
    mirror_features,
    save_models,
)

V_TARGET = "fwd_mid_delta_5s"
T_TARGET = "toxic_5s"
# A market state is adverse for a side when the mid moves >= 1c against it in 5s.
STATE_TAIL_DOLLARS = 0.01


def window_order(pattern: str) -> list[tuple[str, str]]:
    """(ticker, path) sorted by first data timestamp, reading only ts_ms.

    Cheap metadata pass so the chronological split can be computed before the
    (memory-heavy) full store is loaded — letting us skip the holdout entirely.
    """
    import pyarrow.parquet as pq

    ordered = []
    for path in glob.glob(pattern):
        table = pq.read_table(path, columns=["ticker", "ts_ms"])
        if table.num_rows:
            first_ts = min(table.column("ts_ms").to_pylist())
            ordered.append((first_ts, str(table.column("ticker")[0].as_py()), path))
    return [(ticker, path) for _, ticker, path in sorted(ordered, key=lambda item: item[0])]


# Every float column the fit consumes. Loaded unboxed (float64, nulls -> NaN),
# ~20 arrays * n_rows * 8 bytes instead of n_rows python dicts (which OOM the box).
FLOAT_COLUMNS: tuple[str, ...] = (
    "seconds_to_close", "fwd_mid_delta_5s", "outcome_yes", "gbm_fair", "mid",
    "imb_top1", "imb_top3", "microprice_minus_mid", "flow_1s", "flow_5s",
    "bid_size_delta", "ask_size_delta", "spread", "gbm_fair_minus_mid", "jump_ratio",
    "ext_basis_bps", "ext_micro_lead_bps", "ext_imbalance", "ext_dispersion_bps", "trades_5s",
)
Columns = dict[str, np.ndarray]


def load_feature_store(pattern: str, *, tickers: Optional[set[str]] = None) -> dict[str, Columns]:
    """ticker -> {column: float64 ndarray}, chronological, columnar (not row dicts).

    tickers filters to (train+val) so the holdout never sits in memory — both a
    memory saving and a leakage guard. Absent columns (older schema) become NaN.
    """
    import pyarrow.parquet as pq

    store: dict[str, Columns] = {}
    for ticker, path in window_order(pattern):
        if tickers is not None and ticker not in tickers:
            continue
        table = pq.read_table(path)
        present = set(table.column_names)
        n = table.num_rows
        cols: Columns = {}
        for name in FLOAT_COLUMNS:
            if name in present:
                cols[name] = table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
            else:
                cols[name] = np.full(n, np.nan)
        store[ticker] = cols
    return store


def load_fills_store(pattern: str, *, tickers: Optional[set[str]] = None) -> dict[str, list[dict[str, Any]]]:
    """Fills stay row-dicts (small: ~fills/window, not ticks/window)."""
    from .features import read_parquet_rows

    store: dict[str, list[dict[str, Any]]] = {}
    for path in glob.glob(pattern):
        rows = read_parquet_rows(path)
        if rows and (tickers is None or str(rows[0]["ticker"]) in tickers):
            store[str(rows[0]["ticker"])] = rows
    return store


def _cat(store: dict[str, Columns], tickers: list[str], name: str) -> np.ndarray:
    parts = [store[t][name] for t in tickers if t in store]
    return np.concatenate(parts) if parts else np.zeros(0)


def _bucket_mask(seconds_to_close: np.ndarray, bucket: tuple[float, float]) -> np.ndarray:
    return (seconds_to_close >= bucket[0]) & (seconds_to_close < bucket[1])


def _vector_predict(model: LinearModel, x: np.ndarray) -> np.ndarray:
    """Vectorized standardized linear/logistic prediction; NaN feature -> NaN out."""
    means = np.asarray(model.means)
    stds = np.where(np.asarray(model.stds) > 0, np.asarray(model.stds), 1.0)
    score = ((x - means) / stds) @ np.asarray(model.coefs) + model.intercept
    if model.kind == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
    return score


def calibration_report(store: dict[str, Columns], tickers: list[str]) -> list[str]:
    """P1: Brier/log-loss of gbm_fair vs market mid, by time bucket (resolved windows)."""
    lines = ["", "== P1 calibration: gbm_fair vs market mid (resolved windows) =="]
    lines.append(f"{'bucket_s':>12s} {'n':>7s} {'brier_gbm':>10s} {'brier_mid':>10s} {'ll_gbm':>8s} {'ll_mid':>8s}")
    stc = _cat(store, tickers, "seconds_to_close")
    outcome = _cat(store, tickers, "outcome_yes")
    gbm = _cat(store, tickers, "gbm_fair")
    mid = _cat(store, tickers, "mid")
    resolved = np.isfinite(outcome) & np.isfinite(gbm) & np.isfinite(mid)
    for bucket in TIME_BUCKETS:
        mask = resolved & _bucket_mask(stc, bucket)
        if mask.sum() < 100:
            continue
        o = outcome[mask]
        row = [f"{f'{int(bucket[0])}-{int(bucket[1])}':>12s}", f"{int(mask.sum()):7d}"]
        stats = []
        for value in (gbm[mask], mid[mask]):
            p = np.clip(value, 1e-9, 1 - 1e-9)
            brier = float(np.mean((p - o) ** 2))
            ll = float(np.mean(-(o * np.log(p) + (1 - o) * np.log(1 - p))))
            stats.append((brier, ll))
        lines.append(
            f"{row[0]} {row[1]} {stats[0][0]:10.4f} {stats[1][0]:10.4f} {stats[0][1]:8.4f} {stats[1][1]:8.4f}"
        )
    return lines


def _eval_model_v(model: LinearModel, store: dict[str, Columns], val_ids: list[str]) -> dict[str, float]:
    """Per-window IC of model prediction vs realized target on unseen windows."""
    ics: dict[str, float] = {}
    for ticker in val_ids:
        cols = store.get(ticker)
        if cols is None:
            continue
        x = np.column_stack([cols[name] for name in model.features])
        y = cols[V_TARGET]
        mask = _bucket_mask(cols["seconds_to_close"], model.bucket) & np.isfinite(x).all(axis=1) & np.isfinite(y)
        if mask.sum() < 30:
            continue
        pred = _vector_predict(model, x[mask])
        yy = y[mask]
        if pred.std() > 0 and yy.std() > 0:
            ics[ticker] = float(((pred - pred.mean()) * (yy - yy.mean())).mean() / (pred.std() * yy.std()))
    return ics


def run_v(store: dict[str, Columns], train_ids: list[str], val_ids: list[str], out_dir: Path) -> list[str]:
    lines = ["", f"== Model V (ridge -> {V_TARGET}) train={len(train_ids)}w val={len(val_ids)}w =="]
    lines.append(f"{'set':>9s} {'bucket_s':>10s} {'n_train':>8s} {'val_windows':>11s} {'val_IC':>7s} {'ci_low':>7s} {'ci_high':>7s}")
    promoted: list[LinearModel] = []
    y_all = _cat(store, train_ids, V_TARGET)
    stc_all = _cat(store, train_ids, "seconds_to_close")
    for set_name, features in FEATURE_SETS.items():
        x_all = np.column_stack([_cat(store, train_ids, name) for name in features])
        finite = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
        for bucket in TIME_BUCKETS:
            mask = finite & _bucket_mask(stc_all, bucket)
            if mask.sum() < 500:
                continue
            model = fit_ridge(x_all[mask], y_all[mask], features=features, bucket=bucket)
            model.meta["feature_set"] = set_name
            ics = _eval_model_v(model, store, val_ids)
            mean, low, high = block_bootstrap_ci(list(ics.values()))
            lines.append(
                f"{set_name:>9s} {f'{int(bucket[0])}-{int(bucket[1])}':>10s} {int(mask.sum()):8d} "
                f"{len(ics):11d} {mean:+7.3f} {low:+7.3f} {high:+7.3f}"
            )
            promoted.append(model)
    if promoted:
        save_models(promoted, out_dir / "model_v.json")
        lines.append(f"saved {len(promoted)} bucket models -> {out_dir / 'model_v.json'}")
    return lines


# Raw feature columns feeding the side-mirroring; order matches T_MIRRORED_FEATURES.
def _mirror_matrix(cols: dict[str, np.ndarray], side: str) -> np.ndarray:
    sign = -1.0 if side == "bid" else 1.0
    own = cols["bid_size_delta"] if side == "bid" else cols["ask_size_delta"]
    opp = cols["ask_size_delta"] if side == "bid" else cols["bid_size_delta"]
    return np.column_stack([
        sign * cols["imb_top1"], sign * cols["imb_top3"], sign * cols["microprice_minus_mid"],
        sign * cols["flow_1s"], sign * cols["flow_5s"], -own, opp,
        sign * cols["ext_micro_lead_bps"], sign * cols["ext_imbalance"],
        sign * cols["gbm_fair_minus_mid"], cols["spread"], cols["trades_5s"],
    ])


def _mirrored_fill_row(row: dict[str, Any]) -> Optional[dict[str, Optional[float]]]:
    side = row.get("fill_side")
    if side not in {"bid", "ask"}:
        return None
    return mirror_features(row, str(side))


def _fill_auc_brier(
    model: LinearModel, val_rows: list[dict[str, Any]], bucket: tuple[float, float]
) -> tuple[Optional[float], float, int]:
    pairs = []
    for row in val_rows:
        if bucket_of(float(row["seconds_to_close"])) != bucket:
            continue
        mirrored = _mirrored_fill_row(row)
        target = row.get(T_TARGET)
        if mirrored is None or target is None:
            continue
        # Evaluate exactly as deployed: the gate imputes missing features to the
        # training mean, so validation must too.
        p = model.predict([mirrored.get(name) for name in model.features], impute_missing=True)
        if p is not None:
            pairs.append((p, float(target)))
    auc = _auc(pairs)
    brier = float(np.mean([(p - y_) ** 2 for p, y_ in pairs])) if pairs else float("nan")
    return auc, brier, len(pairs)


def run_t(
    store: dict[str, Columns],
    fills_store: dict[str, list[dict[str, Any]]],
    train_ids: list[str],
    val_ids: list[str],
    out_dir: Path,
) -> list[str]:
    """Model T, two trainings, one evaluation (see 2026-07-14 redesign).

    fills: logistic on simulated-fill outcomes (matches deployment, few samples).
    state: logistic on counterfactual adverse-move labels from EVERY feature row,
    both hypothetical sides (~20x samples). Both use side-mirrored features and
    are evaluated identically on actual val-window fill toxicity.
    """
    lines = ["", f"== Model T (side-mirrored logistic) train={len(train_ids)}w val={len(val_ids)}w =="]
    train_fills = [row for ticker in train_ids if ticker in fills_store for row in fills_store[ticker]]
    val_fills = [row for ticker in val_ids if ticker in fills_store for row in fills_store[ticker]]
    lines.append(f"train_fills={len(train_fills)} val_fills={len(val_fills)}")
    features = T_MIRRORED_FEATURES
    candidates: dict[str, list[LinearModel]] = {"fills": [], "state": []}
    lines.append(f"{'training':>9s} {'bucket_s':>10s} {'n_train':>8s} {'base_rate':>9s} {'val_n':>6s} {'val_auc':>8s} {'val_brier':>9s}")

    # -- fills training (row-dicts) --
    fill_samples: dict[tuple[float, float], tuple[list, list]] = defaultdict(lambda: ([], []))
    for row in train_fills:
        mirrored = _mirrored_fill_row(row)
        bucket = bucket_of(float(row["seconds_to_close"]))
        target = row.get(T_TARGET)
        if mirrored is None or bucket is None or target is None:
            continue
        values = [mirrored.get(name) for name in features]
        if any(value is None for value in values):
            continue
        fill_samples[bucket][0].append([float(v) for v in values])
        fill_samples[bucket][1].append(float(target))

    # -- state training (columnar) --
    src = {name: _cat(store, train_ids, name) for name in (
        "imb_top1", "imb_top3", "microprice_minus_mid", "flow_1s", "flow_5s",
        "bid_size_delta", "ask_size_delta", "ext_micro_lead_bps", "ext_imbalance",
        "gbm_fair_minus_mid", "spread", "trades_5s",
    )}
    move = _cat(store, train_ids, V_TARGET)
    stc = _cat(store, train_ids, "seconds_to_close")
    x_state = np.vstack([_mirror_matrix(src, "bid"), _mirror_matrix(src, "ask")])
    y_state = np.concatenate([(move <= -STATE_TAIL_DOLLARS).astype(float), (move >= STATE_TAIL_DOLLARS).astype(float)])
    stc_state = np.concatenate([stc, stc])
    finite_state = np.isfinite(x_state).all(axis=1) & np.isfinite(stc_state)

    for training in ("fills", "state"):
        for bucket in TIME_BUCKETS:
            if training == "fills":
                xs, ys = fill_samples.get(bucket, ([], []))
                if len(ys) < 200 or len(set(ys)) < 2:
                    continue
                x, y = np.array(xs), np.array(ys)
            else:
                mask = finite_state & _bucket_mask(stc_state, bucket)
                if mask.sum() < 200 or len(np.unique(y_state[mask])) < 2:
                    continue
                x, y = x_state[mask], y_state[mask]
            model = fit_logistic(x, y, features=features, bucket=bucket)
            model.meta["training"] = training
            auc, brier, val_n = _fill_auc_brier(model, val_fills, bucket)
            lines.append(
                f"{training:>9s} {f'{int(bucket[0])}-{int(bucket[1])}':>10s} {len(y):8d} "
                f"{float(np.mean(y)):9.3f} {val_n:6d} "
                f"{auc if auc is not None else float('nan'):8.3f} {brier:9.4f}"
            )
            candidates[training].append(model)
    for training, models in candidates.items():
        if models:
            save_models(models, out_dir / f"model_t_{training}.json")
            lines.append(f"saved {len(models)} bucket models -> {out_dir / f'model_t_{training}.json'}")
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
        help="include the final 20%% holdout windows (ONE final promotion read only)",
    )
    args = parser.parse_args(argv)
    # Split from a cheap metadata pass so the holdout is never loaded into memory
    # unless explicitly unlocked (both a memory saving and a leakage guard).
    order = [ticker for ticker, _ in window_order(args.features)]
    if not order:
        raise SystemExit("no feature windows found")
    train_ids, val_ids, holdout_ids = chronological_split(order)
    print(
        f"windows: train={len(train_ids)} val={len(val_ids)} holdout={len(holdout_ids)} "
        f"(locked={not args.unlock_holdout})"
    )
    if len(order) < 300:
        print(f"WARNING: {len(order)} windows < 300 — screening power is insufficient; treat as smoke test")
    if args.unlock_holdout:
        print("*** HOLDOUT UNLOCKED — this counts as the one promotion read (STRATEGY.md §6) ***")
        val_ids = val_ids + holdout_ids
    load_tickers = set(train_ids) | set(val_ids)
    store = load_feature_store(args.features, tickers=load_tickers)
    fills_store = load_fills_store(args.labels, tickers=load_tickers)
    out_dir = Path(args.out)
    for line in calibration_report(store, train_ids + val_ids):
        print(line)
    for line in run_v(store, train_ids, val_ids, out_dir):
        print(line)
    for line in run_t(store, fills_store, train_ids, val_ids, out_dir):
        print(line)


if __name__ == "__main__":
    main()
