"""
experiments/analyze.py  --  Phase 1D 統計解析

120プロンプトで DSD を実行し、信頼度形成の分布を計測する。

出力ファイル:
  analysis.json     統計量 + ヒストグラムデータ
  easy_cases.json   stop_dim が小さい Top 10 (早期停止 = 高確信)
  hard_cases.json   stop_dim が大きい Top 10 (遅い停止 = 低確信)

アルゴリズム変更なし。
"""

import sys
import os
import json
import time
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsd_poc import dsd_predict

from transformers import AutoTokenizer, AutoModelForCausalLM
from experiments.prompts import PROMPTS

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"
TRAJECTORY_INTERVAL = 128


# ---------------------------------------------------------------------------
# ロード
# ---------------------------------------------------------------------------

def load_gemma(model_name: str):
    print(f"[load] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_final_hidden(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu()


def get_lm_head(model):
    lm = model.lm_head
    W = lm.weight.detach().float().cpu()
    bias = lm.bias.detach().float().cpu() if lm.bias is not None else None
    return W, bias


def dense_predict(h, W, bias=None):
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit += bias
    return int(logit.argmax())


# ---------------------------------------------------------------------------
# 統計ユーティリティ
# ---------------------------------------------------------------------------

def percentile(sorted_vals: list, p: float) -> float:
    """線形補間パーセンタイル (p: 0-100)。"""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def histogram(values: list, n_bins: int = 20) -> list[dict]:
    """等幅ビンのヒストグラムを返す。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"bin_start": lo, "bin_end": hi, "count": len(values)}]
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1)
        bins[idx] += 1
    return [
        {
            "bin_start": round(lo + i * width, 4),
            "bin_end": round(lo + (i + 1) * width, 4),
            "count": bins[i],
        }
        for i in range(n_bins)
    ]


# ---------------------------------------------------------------------------
# 1 プロンプト評価
# ---------------------------------------------------------------------------

def eval_one(prompt, tokenizer, model, W, bias, idx, total) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    h = get_final_hidden(model, input_ids)
    dense_id = dense_predict(h, W, bias)

    result = dsd_predict(h, W, bias, trajectory_interval=TRAJECTORY_INTERVAL)

    dsd_token = tokenizer.decode([result["token_id"]])
    dense_token = tokenizer.decode([dense_id])
    match = result["token_id"] == dense_id

    print(
        f"[{idx+1:3d}/{total}] match={match}  "
        f"stop={result['stop_dim']:4d}/{result['hidden_dim']}  "
        f"skip={result['skip_rate']*100:5.1f}%  "
        f"ratio={result['stopping_ratio']}  "
        f"dense={dense_token!r}  dsd={dsd_token!r}  "
        f"| {prompt[:40]!r}"
    )

    return {
        "prompt": prompt,
        "dense_token": dense_token,
        "dsd_token": dsd_token,
        "match": match,
        "stop_dim": result["stop_dim"],
        "hidden_dim": result["hidden_dim"],
        "skip_rate": result["skip_rate"],
        "max_delta": result["max_delta"],
        "final_delta": result["final_delta"],
        "stopping_ratio": result["stopping_ratio"],
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1D: DSD distribution analysis")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--analysis_output", default="analysis.json")
    parser.add_argument("--easy_output", default="easy_cases.json")
    parser.add_argument("--hard_output", default="hard_cases.json")
    parser.add_argument("--n", type=int, default=0, help="使用プロンプト数 (0=全件)")
    args = parser.parse_args()

    prompts = PROMPTS[: args.n] if args.n > 0 else PROMPTS
    total = len(prompts)

    # --- ロード ---
    tokenizer, model = load_gemma(args.model)

    cfg = model.config
    hidden_dim = cfg.hidden_size
    vocab_size = cfg.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model info]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  params={n_params/1e9:.2f}B\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)\n")

    # --- 評価ループ ---
    all_results = []
    t_start = time.time()
    for i, prompt in enumerate(prompts):
        rec = eval_one(prompt, tokenizer, model, W, bias, i, total)
        all_results.append(rec)

    elapsed = time.time() - t_start
    print(f"\n評価完了: {total} prompts / {elapsed:.1f}s ({elapsed/total:.2f}s/prompt)")

    # --- 統計 ---
    n = len(all_results)
    n_match = sum(r["match"] for r in all_results)

    stop_dims = sorted(r["stop_dim"] for r in all_results)
    skip_rates = sorted(r["skip_rate"] for r in all_results)
    max_deltas = [r["max_delta"] for r in all_results]
    stop_ratios = [r["stopping_ratio"] for r in all_results if r["stopping_ratio"] is not None]

    stats = {
        "n_prompts": n,
        "top1_match_rate": n_match / n,
        "model": args.model,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "stop_dim": {
            "mean":   round(sum(stop_dims) / n, 2),
            "median": round(percentile(stop_dims, 50), 2),
            "p10":    round(percentile(stop_dims, 10), 2),
            "p25":    round(percentile(stop_dims, 25), 2),
            "p75":    round(percentile(stop_dims, 75), 2),
            "p90":    round(percentile(stop_dims, 90), 2),
            "p95":    round(percentile(stop_dims, 95), 2),
            "min":    stop_dims[0],
            "max":    stop_dims[-1],
        },
        "skip_rate": {
            "mean":   round(sum(skip_rates) / n, 4),
            "median": round(percentile(skip_rates, 50), 4),
            "p10":    round(percentile(skip_rates, 10), 4),
            "p25":    round(percentile(skip_rates, 25), 4),
            "p75":    round(percentile(skip_rates, 75), 4),
            "p90":    round(percentile(skip_rates, 90), 4),
            "p95":    round(percentile(skip_rates, 95), 4),
            "min":    round(skip_rates[0], 4),
            "max":    round(skip_rates[-1], 4),
        },
        "max_delta": {
            "mean": round(sum(max_deltas) / n, 4),
            "min":  round(min(max_deltas), 4),
            "max":  round(max(max_deltas), 4),
        },
        "stopping_ratio": {
            "mean": round(sum(stop_ratios) / len(stop_ratios), 4) if stop_ratios else None,
            "min":  round(min(stop_ratios), 4) if stop_ratios else None,
            "max":  round(max(stop_ratios), 4) if stop_ratios else None,
        },
    }

    analysis = {
        "statistics": stats,
        "histograms": {
            "stop_dim":  histogram(stop_dims, n_bins=20),
            "skip_rate": histogram(skip_rates, n_bins=20),
        },
        "all_results": all_results,
    }

    # --- easy / hard Top 10 ---
    sorted_by_stop = sorted(all_results, key=lambda r: r["stop_dim"])
    easy_cases = sorted_by_stop[:10]
    hard_cases = sorted_by_stop[-10:][::-1]

    # --- 表示 ---
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"  prompts          : {n}")
    print(f"  top1 match       : {n_match}/{n}  ({n_match/n*100:.1f}%)")
    print(f"  stop_dim  mean   : {stats['stop_dim']['mean']:.1f} / {hidden_dim}")
    print(f"  stop_dim  median : {stats['stop_dim']['median']:.1f}")
    print(f"  stop_dim  p90    : {stats['stop_dim']['p90']:.1f}")
    print(f"  stop_dim  p95    : {stats['stop_dim']['p95']:.1f}")
    print(f"  skip_rate mean   : {stats['skip_rate']['mean']*100:.1f}%")
    print(f"  skip_rate median : {stats['skip_rate']['median']*100:.1f}%")
    print(f"  skip_rate p90    : {stats['skip_rate']['p90']*100:.1f}%")

    print("\nTop 10 EASY (smallest stop_dim):")
    for i, r in enumerate(easy_cases):
        print(f"  {i+1:2d}. stop={r['stop_dim']:4d}  skip={r['skip_rate']*100:5.1f}%  "
              f"token={r['dsd_token']!r}  | {r['prompt'][:45]!r}")

    print("\nTop 10 HARD (largest stop_dim):")
    for i, r in enumerate(hard_cases):
        print(f"  {i+1:2d}. stop={r['stop_dim']:4d}  skip={r['skip_rate']*100:5.1f}%  "
              f"token={r['dsd_token']!r}  | {r['prompt'][:45]!r}")
    print("=" * 60)

    # --- 保存 ---
    with open(args.analysis_output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.analysis_output}")

    with open(args.easy_output, "w", encoding="utf-8") as f:
        json.dump(easy_cases, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.easy_output}")

    with open(args.hard_output, "w", encoding="utf-8") as f:
        json.dump(hard_cases, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.hard_output}")


if __name__ == "__main__":
    main()
