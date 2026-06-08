"""Verification suite for DSD using dense FP32 as ground truth.

These tests intentionally avoid importing runtime modules so they can audit the
DSD invariants without modifying or depending on runtime code paths.  The helper
functions below are a small, deterministic FP32 reference implementation built
only from the Python standard library.
"""

from __future__ import annotations

import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path


N_RANDOM_SEEDS = 1000
VOCAB_SIZE = 24
HIDDEN_DIM = 16
CHUNK_SIZE = 4
TOP5_K = 5
TOP10_K = 10


def f32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32 and return it as a float."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def fp32_mul(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def fp32_add(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def manual_dot_fp32(row: list[float], h: list[float]) -> float:
    """Manual FP32 dot product with FP32 multiply and FP32 accumulation."""
    acc = f32(0.0)
    for w_i, h_i in zip(row, h, strict=True):
        acc = fp32_add(acc, fp32_mul(w_i, h_i))
    return acc


def gemv_fp32(W: list[list[float]], h: list[float], bias: list[float] | None = None) -> list[float]:
    """Dense FP32 GEMV reference: one manual FP32 dot product per vocab row."""
    logits = []
    for row_idx, row in enumerate(W):
        acc = manual_dot_fp32(row, h)
        if bias is not None:
            acc = fp32_add(acc, bias[row_idx])
        logits.append(acc)
    return logits


def top_k_indices(values: list[float], k: int) -> list[int]:
    """Stable top-k: higher value first, lower index breaks ties."""
    return sorted(range(len(values)), key=lambda idx: (-values[idx], idx))[:k]


def weight_col_max(W: list[list[float]]) -> list[float]:
    return [max(abs(row[col]) for row in W) for col in range(len(W[0]))]


def bound_first_order(h: list[float], col_max: list[float]) -> list[int]:
    contrib = [col_max[i] * abs(h[i]) for i in range(len(h))]
    return sorted(range(len(h)), key=lambda idx: (-contrib[idx], idx))


def suffix_bounds(order: list[int], h_ord: list[float], col_max: list[float]) -> list[float]:
    """suffix_B[k] = sum_{j>=k} max_v|W[v, order[j]]| * |h_ord[j]|."""
    H = len(order)
    suffix_B = [0.0] * (H + 1)
    running = 0.0
    for pos in range(H - 1, -1, -1):
        running += col_max[order[pos]] * abs(h_ord[pos])
        suffix_B[pos] = running
    return suffix_B


@dataclass(frozen=True)
class DSDResult:
    token_id: int
    top5_ids: list[int]
    top10_ids: list[int]
    stop_tile: int
    n_tiles: int
    skip_rate: float
    partial_logits: list[float]
    history: list[dict[str, float | int | bool]]


def dsd_fp32(
    W: list[list[float]],
    h: list[float],
    bias: list[float] | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> DSDResult:
    """Tile-by-tile DSD FP32 verifier with the runtime stopping rule Δ > 2B."""
    H = len(h)
    n_tiles = math.ceil(H / chunk_size)
    col_max = weight_col_max(W)
    order = bound_first_order(h, col_max)
    h_ord = [h[i] for i in order]
    suffix_B = suffix_bounds(order, h_ord, col_max)
    partial = [f32(bias_i) for bias_i in bias] if bias is not None else [f32(0.0) for _ in W]
    history: list[dict[str, float | int | bool]] = []
    stop_tile = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end = min(tile * chunk_size, H)
        for vocab_idx, row in enumerate(W):
            acc = partial[vocab_idx]
            for ord_pos in range(dim_start, dim_end):
                original_dim = order[ord_pos]
                acc = fp32_add(acc, fp32_mul(row[original_dim], h_ord[ord_pos]))
            partial[vocab_idx] = acc

        top2 = top_k_indices(partial, 2)
        delta = partial[top2[0]] - partial[top2[1]]
        delta_bound = 2.0 * suffix_B[dim_end]
        stopped = delta > delta_bound
        history.append(
            {
                "tile": tile,
                "dim_end": dim_end,
                "delta": delta,
                "delta_bound": delta_bound,
                "stopped": stopped,
            }
        )
        if stopped:
            stop_tile = tile
            break

    return DSDResult(
        token_id=top_k_indices(partial, 1)[0],
        top5_ids=top_k_indices(partial, TOP5_K),
        top10_ids=top_k_indices(partial, TOP10_K),
        stop_tile=stop_tile,
        n_tiles=n_tiles,
        skip_rate=(n_tiles - stop_tile) / n_tiles,
        partial_logits=partial,
        history=history,
    )


def random_power_of_two_matrix(seed: int) -> tuple[list[list[float]], list[float], list[float]]:
    """Generate exactly representable FP32-ish toy weights for deterministic audits."""
    rng = random.Random(seed)
    W = [[f32(rng.randint(-8, 8) / 8.0) for _ in range(HIDDEN_DIM)] for _ in range(VOCAB_SIZE)]
    h = [f32(rng.randint(-8, 8) / 8.0) for _ in range(HIDDEN_DIM)]
    bias = [f32(rng.randint(-4, 4) / 8.0) for _ in range(VOCAB_SIZE)]
    return W, h, bias


def test_h_ord_equals_h_order_invariant() -> None:
    W, h, _ = random_power_of_two_matrix(seed=17)
    col_max = weight_col_max(W)
    order = bound_first_order(h, col_max)
    h_ord = [h[i] for i in order]

    assert h_ord == [h[i] for i in order]
    assert sorted(order) == list(range(HIDDEN_DIM))


def test_gemv_fp32_equals_manual_fp32_invariant() -> None:
    W, h, bias = random_power_of_two_matrix(seed=29)
    logits = gemv_fp32(W, h, bias)

    for vocab_idx, row in enumerate(W):
        expected = fp32_add(manual_dot_fp32(row, h), bias[vocab_idx])
        assert logits[vocab_idx] == expected


def test_suffix_B_monotonicity_invariant() -> None:
    W, h, _ = random_power_of_two_matrix(seed=41)
    col_max = weight_col_max(W)
    order = bound_first_order(h, col_max)
    h_ord = [h[i] for i in order]
    suffix_B = suffix_bounds(order, h_ord, col_max)

    assert suffix_B[-1] == 0.0
    assert all(suffix_B[i] >= suffix_B[i + 1] for i in range(len(suffix_B) - 1))


def test_delta_greater_than_bound_correctness_on_toy_matrices() -> None:
    W = [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    early_h = [1.0, 0.1, 0.1]
    early = dsd_fp32(W, early_h, chunk_size=1)
    assert early.stop_tile == 1
    assert early.history[0]["delta"] > early.history[0]["delta_bound"]
    assert early.token_id == top_k_indices(gemv_fp32(W, early_h), 1)[0]

    delayed_h = [1.0, 0.9, 0.0]
    delayed = dsd_fp32(W, delayed_h, chunk_size=1)
    assert delayed.stop_tile == 2
    assert delayed.history[0]["delta"] <= delayed.history[0]["delta_bound"]
    assert delayed.history[1]["delta"] > delayed.history[1]["delta_bound"]
    assert delayed.token_id == top_k_indices(gemv_fp32(W, delayed_h), 1)[0]


def margin(logits: list[float]) -> float:
    """Return top1 - top2 margin for a logit vector."""
    top2 = top_k_indices(logits, 2)
    return logits[top2[0]] - logits[top2[1]]


def margin_statistics(margins: list[float]) -> dict[str, float]:
    """Summarize dense FP32 top1/top2 margins over the randomized audit."""
    sorted_margins = sorted(margins)
    n = len(sorted_margins)
    return {
        "min": sorted_margins[0],
        "avg": sum(sorted_margins) / n,
        "p50": sorted_margins[n // 2],
        "max": sorted_margins[-1],
    }


def dump_failure_cases(failure_cases: list[dict[str, object]], dump_path: Path) -> None:
    """Persist top1 mismatches with enough context to reproduce the failure."""
    if failure_cases:
        dump_path.write_text(json.dumps(failure_cases, indent=2), encoding="utf-8")


def test_randomized_dense_fp32_vs_dsd_fp32_1000_seeds(tmp_path: Path) -> None:
    top1_agree = 0
    top5_agree = 0
    top5_set_agree = 0
    top10_agree = 0
    stop_tiles: list[int] = []
    skip_rates: list[float] = []
    dense_margins: list[float] = []
    failure_cases: list[dict[str, object]] = []

    for seed in range(N_RANDOM_SEEDS):
        W, h, bias = random_power_of_two_matrix(seed)
        dense_logits = gemv_fp32(W, h, bias)
        dense_top1 = top_k_indices(dense_logits, 1)[0]
        dense_top5 = top_k_indices(dense_logits, TOP5_K)
        dense_top10 = top_k_indices(dense_logits, TOP10_K)
        dsd = dsd_fp32(W, h, bias, chunk_size=CHUNK_SIZE)

        top1_match = dsd.token_id == dense_top1
        top1_agree += int(top1_match)
        top5_agree += int(dsd.top5_ids == dense_top5)
        top5_set_agree += int(set(dsd.top5_ids) == set(dense_top5))
        top10_agree += int(dsd.top10_ids == dense_top10)
        dense_margins.append(margin(dense_logits))
        if not top1_match:
            failure_cases.append(
                {
                    "seed": seed,
                    "dense_top10": dense_top10,
                    "dsd_top10": dsd.top10_ids,
                    "stop_tile": dsd.stop_tile,
                }
            )
        stop_tiles.append(dsd.stop_tile)
        skip_rates.append(dsd.skip_rate)

    failure_dump_path = tmp_path / "dsd_top1_failures.json"
    dump_failure_cases(failure_cases, failure_dump_path)

    report = {
        "seeds": N_RANDOM_SEEDS,
        "top1_agreement": top1_agree / N_RANDOM_SEEDS,
        "top5_agreement": top5_agree / N_RANDOM_SEEDS,
        "top5_set_agreement": top5_set_agree / N_RANDOM_SEEDS,
        "top10_agreement": top10_agree / N_RANDOM_SEEDS,
        "margin_stats": margin_statistics(dense_margins),
        "average_stop_tile": sum(stop_tiles) / N_RANDOM_SEEDS,
        "average_skip_rate": sum(skip_rates) / N_RANDOM_SEEDS,
        "failure_case_count": len(failure_cases),
        "failure_dump_path": str(failure_dump_path) if failure_cases else None,
    }
    print(f"DSD FP32 randomized verification report: {report}")

    assert not failure_cases, f"top1 mismatches dumped to {failure_dump_path}"
    assert top1_agree == N_RANDOM_SEEDS
    assert all(1 <= stop <= math.ceil(HIDDEN_DIM / CHUNK_SIZE) for stop in stop_tiles)
    assert all(0.0 <= skip <= 1.0 for skip in skip_rates)
