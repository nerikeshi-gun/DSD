"""
experiments/dsd_accuracy_diag.py  --  一致率の誤差源切り分け診断

3系統を比較する:
  A. Dense FP32 : W.float() @ h.float()          ← 基準
  B. Dense BF16 : W @ h  (bfloat16 のまま)
  C. DSD   BF16 : tile-by-tile, Δ>2B で early stop

mismatch サンプルについて top1/top2 id・logit・margin を表示し、
誤差源が (1) BF16 丸め か (2) DSD 実装 かを切り分ける。

比較軸:
  A vs B : BF16 丸め誤差の影響
  A vs C : DSD 実装の影響 (BF16 丸め込み)
  B vs C : BF16 同士での DSD 実装誤差

出力:
  dsd_accuracy_diag.json
"""

import sys, os, json, math, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first_with_colmax
from experiments.dsd_runtime import (
    load_gemma, get_lm_head_gpu, batch_hidden_states_gpu,
    compute_weight_col_max,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME  = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE  = 80
N_BATCH     = 512

DEFAULT_PROMPTS = [
    "The capital of Japan is",
    "Python is a programming",
    "The largest planet in the solar system is",
    "Linux is an open source",
    "The opposite of hot is",
    "Water boils at 100 degrees",
    "The speed of light is approximately",
    "Machine learning is a subset of",
    "The first element in the periodic table is",
    "Shakespeare was born in",
    "World War II ended in",
    "DNA stands for",
    "The chemical symbol for gold is",
    "The internet was invented in",
    "A compiler converts",
    "The capital of France is",
    "The largest ocean in the world is",
    "The atomic number of carbon is",
    "HTML stands for",
    "The opposite of cold is",
]


# ---------------------------------------------------------------------------
# A. Dense FP32
# ---------------------------------------------------------------------------

def dense_fp32(W: torch.Tensor, h: torch.Tensor,
               bias: torch.Tensor | None) -> torch.Tensor:
    """W.float() @ h.float()  (一時的に float32 に昇格して計算)"""
    logit = W.float() @ h.float()
    if bias is not None:
        logit = logit + bias.float()
    return logit


# ---------------------------------------------------------------------------
# B. Dense BF16
# ---------------------------------------------------------------------------

def dense_bf16(W: torch.Tensor, h: torch.Tensor,
               bias: torch.Tensor | None) -> torch.Tensor:
    logit = W @ h
    if bias is not None:
        logit = logit + bias
    return logit


# ---------------------------------------------------------------------------
# C. DSD BF16 (tile-by-tile early stop)
# ---------------------------------------------------------------------------

def dsd_bf16(
    W: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_size: int,
    order: torch.Tensor,
    suffix_B: torch.Tensor,
    h_ord: torch.Tensor,
    n_tiles: int,
) -> tuple[torch.Tensor, int]:
    """
    Returns: (partial_logit, stop_tile)
    partial_logit は stop_tile 時点の累積ロジット (bfloat16)
    """
    H          = h_ord.shape[0]
    vocab_size = W.shape[0]

    partial_logit = (bias.clone() if bias is not None
                     else torch.zeros(vocab_size, dtype=W.dtype, device=h_ord.device))
    stop_tile = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end   = min(tile * chunk_size, H)

        idx = order[dim_start:dim_end]
        partial_logit.add_(W[:, idx] @ h_ord[dim_start:dim_end])

        top2  = partial_logit.topk(2)
        delta = top2.values[0] - top2.values[1]
        B     = suffix_B[dim_end] if dim_end < H else suffix_B.new_zeros(())

        if delta > 2.0 * B:
            stop_tile = tile
            break

    return partial_logit, stop_tile


# ---------------------------------------------------------------------------
# 1 サンプルの比較レコードを作る
# ---------------------------------------------------------------------------

def top2_info(logit: torch.Tensor) -> dict:
    top2 = logit.topk(2)
    return {
        "top1_id":     int(top2.indices[0]),
        "top2_id":     int(top2.indices[1]),
        "top1_logit":  round(float(top2.values[0]), 5),
        "top2_logit":  round(float(top2.values[1]), 5),
        "margin":      round(float(top2.values[0] - top2.values[1]), 5),
    }


def compare_sample(
    i: int,
    prompt: str,
    W: torch.Tensor,
    bias: torch.Tensor | None,
    h: torch.Tensor,               # (H,) bfloat16 GPU
    weight_col_max: torch.Tensor,  # (H,) bfloat16 GPU
    chunk_size: int,
    n_tiles: int,
) -> dict:
    order  = order_bound_first_with_colmax(h, weight_col_max)
    h_ord  = h[order]
    suffix_B = (weight_col_max[order] * h_ord.abs()).flip(0).cumsum(0).flip(0)

    logit_a = dense_fp32(W, h, bias)
    logit_b = dense_bf16(W, h, bias)
    logit_c, stop_tile = dsd_bf16(W, bias, chunk_size, order, suffix_B, h_ord, n_tiles)

    info_a = top2_info(logit_a)
    info_b = top2_info(logit_b)
    info_c = top2_info(logit_c)

    match_ab = info_a["top1_id"] == info_b["top1_id"]
    match_ac = info_a["top1_id"] == info_c["top1_id"]
    match_bc = info_b["top1_id"] == info_c["top1_id"]

    return {
        "sample_id": i,
        "prompt":    prompt,
        "stop_tile": stop_tile,
        "n_tiles":   n_tiles,
        "skip_pct":  round((n_tiles - stop_tile) / n_tiles * 100, 2),
        "match": {
            "A_vs_B": match_ab,   # BF16 丸め誤差の影響
            "A_vs_C": match_ac,   # DSD 実装の影響
            "B_vs_C": match_bc,   # BF16 同士での DSD 誤差
        },
        "A_dense_fp32": info_a,
        "B_dense_bf16": info_b,
        "C_dsd_bf16":   info_c,
    }


# ---------------------------------------------------------------------------
# mismatch の詳細表示
# ---------------------------------------------------------------------------

def print_mismatch(rec: dict, label: str):
    a = rec["A_dense_fp32"]
    b = rec["B_dense_bf16"]
    c = rec["C_dsd_bf16"]
    print(f"\n  [{label}] sample={rec['sample_id']}  prompt={rec['prompt']!r}")
    print(f"    stop_tile={rec['stop_tile']}/{rec['n_tiles']}  skip={rec['skip_pct']}%")
    print(f"    {'':12s}  {'top1_id':>10s}  {'top2_id':>10s}  "
          f"{'top1_logit':>11s}  {'top2_logit':>11s}  {'margin':>10s}")
    print(f"    {'A DenseFP32':12s}  {a['top1_id']:>10d}  {a['top2_id']:>10d}  "
          f"{a['top1_logit']:>11.4f}  {a['top2_logit']:>11.4f}  {a['margin']:>10.4f}")
    print(f"    {'B DenseBF16':12s}  {b['top1_id']:>10d}  {b['top2_id']:>10d}  "
          f"{b['top1_logit']:>11.4f}  {b['top2_logit']:>11.4f}  {b['margin']:>10.4f}")
    print(f"    {'C DSD BF16':12s}  {c['top1_id']:>10d}  {c['top2_id']:>10d}  "
          f"{c['top1_logit']:>11.4f}  {c['top2_logit']:>11.4f}  {c['margin']:>10.4f}")
    print(f"    A==B={rec['match']['A_vs_B']}  A==C={rec['match']['A_vs_C']}  "
          f"B==C={rec['match']['B_vs_C']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSD accuracy diagnostic: FP32 vs BF16 vs DSD")
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--n_batch",    type=int, default=N_BATCH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output",     default="dsd_accuracy_diag.json")
    parser.add_argument("--max_mismatch_print", type=int, default=20,
                        help="mismatch サンプルの最大表示数")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    n_batch    = args.n_batch

    base    = DEFAULT_PROMPTS
    prompts = [base[i % len(base)] for i in range(n_batch)]

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    if hasattr(cfg, "text_config"):
        hidden_dim = cfg.text_config.hidden_size
        vocab_size = cfg.text_config.vocab_size
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_tiles = math.ceil(hidden_dim / chunk_size)
    print(f"[model]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  device={device}")
    print(f"[batch]  n_users={n_batch}  chunk_size={chunk_size}  n_tiles={n_tiles}\n")

    W, bias = get_lm_head_gpu(model, dtype=torch.bfloat16)
    print(f"[prep] W: {W.shape}  {W.dtype}  {W.device}")

    print("[prep] weight_col_max をチャンク単位で計算中...")
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)
    print(f"  weight_col_max: {weight_col_max.shape}\n")

    print(f"[forward] {n_batch} ユーザーを一括 forward 中...")
    h_batch = batch_hidden_states_gpu(model, tokenizer, prompts)
    torch.cuda.synchronize(device)
    print(f"  h_batch: {tuple(h_batch.shape)}  {h_batch.dtype}\n")

    # --- 3系統を比較 ---
    print("[diag] A=DenseFP32 / B=DenseBF16 / C=DSD_BF16 を比較中...\n")
    records   = []
    t_start   = time.time()

    for i in range(n_batch):
        rec = compare_sample(
            i, prompts[i], W, bias, h_batch[i],
            weight_col_max, chunk_size, n_tiles,
        )
        records.append(rec)

        if (i + 1) % max(1, n_batch // 10) == 0 or (i + 1) == n_batch:
            n_ab = sum(r["match"]["A_vs_B"] for r in records)
            n_ac = sum(r["match"]["A_vs_C"] for r in records)
            n_bc = sum(r["match"]["B_vs_C"] for r in records)
            n    = len(records)
            print(f"  [{i+1:4d}/{n_batch}]  "
                  f"A==B: {n_ab}/{n} ({n_ab/n*100:.1f}%)  "
                  f"A==C: {n_ac}/{n} ({n_ac/n*100:.1f}%)  "
                  f"B==C: {n_bc}/{n} ({n_bc/n*100:.1f}%)  "
                  f"({time.time()-t_start:.1f}s)")

    # --- サマリ ---
    n = n_batch
    n_ab = sum(r["match"]["A_vs_B"] for r in records)
    n_ac = sum(r["match"]["A_vs_C"] for r in records)
    n_bc = sum(r["match"]["B_vs_C"] for r in records)

    # A==B かつ A!=C → DSD 固有の誤差
    dsd_only_err  = [r for r in records if     r["match"]["A_vs_B"] and not r["match"]["A_vs_C"]]
    # A!=B かつ A!=C → BF16 丸め誤差 (DSD は B と一致している可能性)
    bf16_err_only = [r for r in records if not r["match"]["A_vs_B"] and     r["match"]["B_vs_C"]]
    # A!=B かつ A!=C かつ B!=C → BF16 + DSD 両方の誤差
    both_err      = [r for r in records if not r["match"]["A_vs_B"] and not r["match"]["B_vs_C"]]

    print(f"\n{'='*65}")
    print("一致率診断サマリ")
    print(f"{'='*65}")
    print(f"  n_users = {n}")
    print(f"  A==B  (DenseFP32 vs DenseBF16) : {n_ab}/{n}  ({n_ab/n*100:.2f}%)")
    print(f"  A==C  (DenseFP32 vs DSD BF16)  : {n_ac}/{n}  ({n_ac/n*100:.2f}%)")
    print(f"  B==C  (DenseBF16 vs DSD BF16)  : {n_bc}/{n}  ({n_bc/n*100:.2f}%)")
    print()
    print(f"  DSD固有誤差  (A==B かつ A!=C) : {len(dsd_only_err)} サンプル")
    print(f"  BF16丸め誤差 (A!=B かつ B==C) : {len(bf16_err_only)} サンプル")
    print(f"  両方誤差     (A!=B かつ B!=C) : {len(both_err)} サンプル")
    print(f"{'='*65}\n")

    # --- mismatch 詳細表示 ---
    if dsd_only_err:
        print(f"[DSD固有誤差] A==B かつ A!=C  ({len(dsd_only_err)} 件, 最大 {args.max_mismatch_print} 件表示)")
        for rec in dsd_only_err[:args.max_mismatch_print]:
            print_mismatch(rec, "DSD-err")

    if bf16_err_only:
        print(f"\n[BF16丸め誤差] A!=B かつ B==C  ({len(bf16_err_only)} 件, 最大 {args.max_mismatch_print} 件表示)")
        for rec in bf16_err_only[:args.max_mismatch_print]:
            print_mismatch(rec, "BF16-err")

    if both_err:
        print(f"\n[両方誤差] A!=B かつ B!=C  ({len(both_err)} 件, 最大 {args.max_mismatch_print} 件表示)")
        for rec in both_err[:args.max_mismatch_print]:
            print_mismatch(rec, "both-err")

    # --- JSON 保存 ---
    output = {
        "model":      args.model,
        "n_users":    n_batch,
        "n_tiles":    n_tiles,
        "chunk_size": chunk_size,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "ordering":   "bound_first",
        "legend": {
            "A": "Dense FP32  (W.float() @ h.float())",
            "B": "Dense BF16  (W @ h, bfloat16)",
            "C": "DSD   BF16  (tile-by-tile, Δ>2B early stop, bfloat16)",
        },
        "summary": {
            "A_vs_B_match": round(n_ab / n, 4),
            "A_vs_C_match": round(n_ac / n, 4),
            "B_vs_C_match": round(n_bc / n, 4),
            "dsd_only_errors":  len(dsd_only_err),
            "bf16_only_errors": len(bf16_err_only),
            "both_errors":      len(both_err),
        },
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
