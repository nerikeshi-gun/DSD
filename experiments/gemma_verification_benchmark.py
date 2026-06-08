"""
experiments/gemma_verification_benchmark.py -- Gemma hidden-state DSD benchmark

Goal:
  Evaluate DSD on real Gemma hidden states using Dense FP32 GEMV as ground truth.

Compares:
  A. Dense FP32 GEMV (ground truth)
  B. DSD FP32 (tile-by-tile, Δ > 2B, FP32 accumulation)
  C. DSD BF16 (tile-by-tile, Δ > 2B, BF16 accumulation)

Metrics:
  - Top1 agreement vs Dense FP32
  - Top5 set agreement vs Dense FP32
  - Top10 set agreement vs Dense FP32
  - Average stop tile
  - Average skip rate

Output:
  JSON report with summary metrics and per-sample records.

Runtime code is intentionally not imported or modified; this is a standalone
benchmark script under experiments/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.prompts import PROMPTS

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"
PROMPT_COUNT = 1000
CHUNK_SIZE = 80
BATCH_SIZE = 16
VOCAB_CHUNK = 4096
TOPK = 10


def load_gemma(model_name: str):
    print(f"[load] {model_name}")
    print("  torch_dtype=bfloat16  device_map=auto  local_files_only=True")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_model_dims(model) -> tuple[int, int]:
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    return int(cfg.hidden_size), int(cfg.vocab_size)


def get_lm_head_bf16(model) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return lm_head weights in model dtype/device; avoid full FP32 copies."""
    lm_head = model.lm_head
    device = next(model.parameters()).device
    W = lm_head.weight.detach().to(device=device, dtype=torch.bfloat16)
    bias = (
        lm_head.bias.detach().to(device=device, dtype=torch.bfloat16)
        if lm_head.bias is not None
        else None
    )
    return W, bias


def build_prompt_dataset(prompt_count: int) -> list[str]:
    """Build exactly prompt_count prompts by cycling the repository prompt set."""
    return [PROMPTS[i % len(PROMPTS)] for i in range(prompt_count)]


def iter_batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def batch_hidden_states(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """Return final-token Gemma hidden states as BF16 tensors on the model device."""
    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    last_hidden = out.hidden_states[-1]
    last_token = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden.size(0), device=device)
    return last_hidden[rows, last_token, :].detach()


def compute_weight_col_max_fp32(W: torch.Tensor, col_chunk: int = 256) -> torch.Tensor:
    """Compute max_v |W[v, i]| in FP32 without materializing W.float()."""
    _, hidden_dim = W.shape
    col_max = torch.empty(hidden_dim, dtype=torch.float32, device=W.device)
    for start in range(0, hidden_dim, col_chunk):
        end = min(start + col_chunk, hidden_dim)
        col_max[start:end] = W[:, start:end].float().abs().max(dim=0).values
    return col_max


def compute_weight_col_max_bf16(W: torch.Tensor, col_chunk: int = 256) -> torch.Tensor:
    """Compute max_v |W[v, i]| in BF16 for BF16 DSD bounds."""
    _, hidden_dim = W.shape
    col_max = torch.empty(hidden_dim, dtype=torch.bfloat16, device=W.device)
    for start in range(0, hidden_dim, col_chunk):
        end = min(start + col_chunk, hidden_dim)
        col_max[start:end] = W[:, start:end].abs().max(dim=0).values
    return col_max


def topk_summary_from_logits(logits: torch.Tensor, k: int = TOPK) -> dict[str, Any]:
    topk = logits.topk(k)
    ids = [int(x) for x in topk.indices.detach().cpu().tolist()]
    values = [float(x) for x in topk.values.detach().float().cpu().tolist()]
    return {
        "top1_id": ids[0],
        "top5_ids": ids[:5],
        "top10_ids": ids[:10],
        "top10_logits": values[:10],
        "margin": values[0] - values[1] if len(values) > 1 else None,
    }


def dense_fp32_top10(
    W: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_chunk: int,
) -> dict[str, Any]:
    """Dense FP32 GEMV top10 ground truth using row chunks to avoid W.float()."""
    vocab_size = W.shape[0]
    h32 = h.float()
    bias32 = bias.float() if bias is not None else None
    top_vals = torch.full((TOPK,), float("-inf"), dtype=torch.float32, device=W.device)
    top_ids = torch.zeros(TOPK, dtype=torch.long, device=W.device)

    for start in range(0, vocab_size, vocab_chunk):
        end = min(start + vocab_chunk, vocab_size)
        logits = W[start:end].float() @ h32
        if bias32 is not None:
            logits = logits + bias32[start:end]

        k = min(TOPK, end - start)
        chunk_top = logits.topk(k)
        candidate_vals = torch.cat([top_vals, chunk_top.values])
        candidate_ids = torch.cat([top_ids, chunk_top.indices + start])
        merged = candidate_vals.topk(TOPK)
        top_vals = merged.values
        top_ids = candidate_ids[merged.indices]

    ids = [int(x) for x in top_ids.detach().cpu().tolist()]
    values = [float(x) for x in top_vals.detach().cpu().tolist()]
    return {
        "top1_id": ids[0],
        "top5_ids": ids[:5],
        "top10_ids": ids,
        "top10_logits": values,
        "margin": values[0] - values[1],
    }


def prepare_ordering(
    h: torch.Tensor,
    weight_col_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bound-first order, h[order], and suffix_B for one hidden state."""
    h_for_bound = h.to(dtype=weight_col_max.dtype)
    bound_contrib = weight_col_max * h_for_bound.abs()
    order = bound_contrib.argsort(descending=True)
    h_ord = h_for_bound[order]
    suffix_B = (weight_col_max[order] * h_ord.abs()).flip(0).cumsum(0).flip(0)
    return order, h_ord, suffix_B


def dsd_fp32_top10(
    W: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor | None,
    weight_col_max_fp32: torch.Tensor,
    chunk_size: int,
    n_tiles: int,
) -> tuple[dict[str, Any], int, float]:
    """DSD FP32 with FP32 GEMV tile accumulation and Δ > 2B stopping."""
    hidden_dim = h.shape[0]
    vocab_size = W.shape[0]
    order, h_ord, suffix_B = prepare_ordering(h.float(), weight_col_max_fp32)
    partial = (
        bias.float().clone()
        if bias is not None
        else torch.zeros(vocab_size, dtype=torch.float32, device=W.device)
    )
    stop_tile = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end = min(tile * chunk_size, hidden_dim)
        idx = order[dim_start:dim_end]
        partial.add_(W[:, idx].float() @ h_ord[dim_start:dim_end])

        top2 = partial.topk(2)
        delta = top2.values[0] - top2.values[1]
        B = suffix_B[dim_end] if dim_end < hidden_dim else suffix_B.new_zeros(())
        if delta > 2.0 * B:
            stop_tile = tile
            break

    return topk_summary_from_logits(partial), stop_tile, (n_tiles - stop_tile) / n_tiles


def dsd_bf16_top10(
    W: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor | None,
    weight_col_max_bf16: torch.Tensor,
    chunk_size: int,
    n_tiles: int,
) -> tuple[dict[str, Any], int, float]:
    """DSD BF16 with BF16 tile accumulation and Δ > 2B stopping."""
    hidden_dim = h.shape[0]
    vocab_size = W.shape[0]
    order, h_ord, suffix_B = prepare_ordering(h, weight_col_max_bf16)
    partial = (
        bias.clone()
        if bias is not None
        else torch.zeros(vocab_size, dtype=torch.bfloat16, device=W.device)
    )
    stop_tile = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end = min(tile * chunk_size, hidden_dim)
        idx = order[dim_start:dim_end]
        partial.add_(W[:, idx] @ h_ord[dim_start:dim_end])

        top2 = partial.topk(2)
        delta = top2.values[0] - top2.values[1]
        B = suffix_B[dim_end] if dim_end < hidden_dim else suffix_B.new_zeros(())
        if delta > 2.0 * B:
            stop_tile = tile
            break

    return topk_summary_from_logits(partial), stop_tile, (n_tiles - stop_tile) / n_tiles


def init_metric_accumulator() -> dict[str, Any]:
    return {
        "top1_matches": 0,
        "top5_set_matches": 0,
        "top10_set_matches": 0,
        "stop_tiles": [],
        "skip_rates": [],
    }


def update_metrics(
    acc: dict[str, Any],
    dense: dict[str, Any],
    dsd: dict[str, Any],
    stop_tile: int,
    skip_rate: float,
) -> None:
    acc["top1_matches"] += int(dsd["top1_id"] == dense["top1_id"])
    acc["top5_set_matches"] += int(set(dsd["top5_ids"]) == set(dense["top5_ids"]))
    acc["top10_set_matches"] += int(set(dsd["top10_ids"]) == set(dense["top10_ids"]))
    acc["stop_tiles"].append(stop_tile)
    acc["skip_rates"].append(skip_rate)


def summarize_metrics(acc: dict[str, Any], n_samples: int) -> dict[str, float]:
    return {
        "top1_agreement": acc["top1_matches"] / n_samples,
        "top5_set_agreement": acc["top5_set_matches"] / n_samples,
        "top10_set_agreement": acc["top10_set_matches"] / n_samples,
        "average_stop_tile": sum(acc["stop_tiles"]) / n_samples,
        "average_skip_rate": sum(acc["skip_rates"]) / n_samples,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = build_prompt_dataset(args.prompt_count)
    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device
    hidden_dim, vocab_size = get_model_dims(model)
    n_tiles = math.ceil(hidden_dim / args.chunk_size)

    print(f"[model] hidden_dim={hidden_dim} vocab_size={vocab_size} device={device}")
    print(f"[data] prompts={len(prompts)} batch_size={args.batch_size}")
    print(f"[dsd] chunk_size={args.chunk_size} n_tiles={n_tiles}\n")

    W, bias = get_lm_head_bf16(model)
    print(f"[prep] W={tuple(W.shape)} dtype={W.dtype} device={W.device}")

    print("[prep] computing FP32 column bounds...")
    weight_col_max_fp32 = compute_weight_col_max_fp32(W, args.col_chunk)
    print("[prep] computing BF16 column bounds...")
    weight_col_max_bf16 = compute_weight_col_max_bf16(W, args.col_chunk)
    if W.is_cuda:
        torch.cuda.synchronize(device)

    fp32_metrics = init_metric_accumulator()
    bf16_metrics = init_metric_accumulator()
    records = []
    t0 = time.time()

    for batch_start, batch_prompts in iter_batches(prompts, args.batch_size):
        h_batch = batch_hidden_states(model, tokenizer, batch_prompts)
        if W.is_cuda:
            torch.cuda.synchronize(device)

        for local_idx, prompt in enumerate(batch_prompts):
            sample_id = batch_start + local_idx
            h = h_batch[local_idx]
            dense = dense_fp32_top10(W, h, bias, args.vocab_chunk)
            dsd32, stop32, skip32 = dsd_fp32_top10(
                W, h, bias, weight_col_max_fp32, args.chunk_size, n_tiles
            )
            dsd16, stop16, skip16 = dsd_bf16_top10(
                W, h, bias, weight_col_max_bf16, args.chunk_size, n_tiles
            )

            update_metrics(fp32_metrics, dense, dsd32, stop32, skip32)
            update_metrics(bf16_metrics, dense, dsd16, stop16, skip16)

            record = {
                "sample_id": sample_id,
                "prompt": prompt,
                "dense_fp32": dense,
                "dsd_fp32": {**dsd32, "stop_tile": stop32, "skip_rate": skip32},
                "dsd_bf16": {**dsd16, "stop_tile": stop16, "skip_rate": skip16},
            }
            records.append(record)

        done = min(batch_start + len(batch_prompts), len(prompts))
        if done % max(args.batch_size, args.progress_every) == 0 or done == len(prompts):
            s32 = summarize_metrics(fp32_metrics, done)
            s16 = summarize_metrics(bf16_metrics, done)
            print(
                f"[progress] {done}/{len(prompts)} "
                f"DSD_FP32 top1={s32['top1_agreement']:.4f} "
                f"DSD_BF16 top1={s16['top1_agreement']:.4f} "
                f"elapsed={time.time() - t0:.1f}s"
            )

    summary = {
        "dense_fp32_ground_truth": {
            "description": "W.float() @ h.float(), row-chunked top10",
        },
        "dsd_fp32_vs_dense_fp32": summarize_metrics(fp32_metrics, len(prompts)),
        "dsd_bf16_vs_dense_fp32": summarize_metrics(bf16_metrics, len(prompts)),
    }

    return {
        "model": args.model,
        "prompt_count": len(prompts),
        "prompt_source": "experiments.prompts.PROMPTS cycled to requested count",
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "chunk_size": args.chunk_size,
        "n_tiles": n_tiles,
        "batch_size": args.batch_size,
        "vocab_chunk": args.vocab_chunk,
        "summary": summary,
        "records": records,
        "elapsed_s": round(time.time() - t0, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma DSD verification benchmark")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt_count", type=int, default=PROMPT_COUNT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--vocab_chunk", type=int, default=VOCAB_CHUNK)
    parser.add_argument("--col_chunk", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--output", default="gemma_dsd_verification_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = benchmark(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
