"""
experiments/dsd_full_trace.py  --  全タイル DSD vs Dense 完全トレース

全 48 タイルについて DSD partial_logit と Dense partial_logit を
tile-by-tile で比較し、初めて分岐するタイルを特定する。

DSD:   W[:, idx] @ h_ord[ds:de]  (チャンク GEMV)
Dense: 要素ごと sum_i W[:, idx[i]] * h_ord[i]  (スカラー積の和)

両者は数学的に同値だが BF16 の浮動小数点演算順序の違いで
異なる結果が生じる可能性がある。

出力: tile_trace_full.json  (全タイル)
"""

import sys, os, math, json, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first_with_colmax
from experiments.dsd_runtime import (
    load_gemma, get_lm_head_gpu, batch_hidden_states_gpu,
    compute_weight_col_max,
)

MODEL_NAME    = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE    = 80
TARGET_PROMPT = "The speed of light is approximately"
TOP_K         = 20


def trace_all_tiles(
    W: torch.Tensor,
    bias: torch.Tensor | None,
    h: torch.Tensor,
    weight_col_max: torch.Tensor,
    chunk_size: int,
    top_k: int,
    out_path: str,
) -> None:
    H       = h.shape[0]
    V       = W.shape[0]
    n_tiles = math.ceil(H / chunk_size)

    order    = order_bound_first_with_colmax(h, weight_col_max)
    h_ord    = h[order]

    contrib_fp32 = weight_col_max[order].float() * h_ord.abs().float()
    suffix_B_fp32 = contrib_fp32.flip(0).cumsum(0).flip(0)
    contrib_bf16  = weight_col_max[order] * h_ord.abs()
    suffix_B_bf16 = contrib_bf16.flip(0).cumsum(0).flip(0)

    # DSD:   チャンク GEMV  W[:,idx] @ h_ord[ds:de]
    # Dense: 要素ごとスカラー積  sum_i W[:,idx[i]] * h_ord[i]
    dsd_partial   = (bias.clone() if bias is not None
                     else torch.zeros(V, dtype=W.dtype, device=h.device))
    dense_partial = (bias.clone() if bias is not None
                     else torch.zeros(V, dtype=W.dtype, device=h.device))

    prev_dsd_top_ids   = set()
    prev_dense_top_ids = set()
    first_diverge_tile = None
    all_tiles          = []
    stopped            = False
    stop_tile          = n_tiles

    for tile in range(1, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]                # 元の列インデックス

        # DSD: チャンク GEMV (実装と同一)
        dsd_partial.add_(W[:, idx] @ h_ord[ds:de])

        # Dense: 要素ごとスカラー積
        for k in range(len(idx)):
            dense_partial.add_(W[:, idx[k]] * h_ord[ds + k])

        # ---- top20 ----
        dsd_top   = dsd_partial.topk(top_k)
        dense_top = dense_partial.topk(top_k)

        dsd_top_ids   = [int(i) for i in dsd_top.indices]
        dsd_top_vals  = [float(v) for v in dsd_top.values]
        dense_top_ids = [int(i) for i in dense_top.indices]
        dense_top_vals= [float(v) for v in dense_top.values]

        dsd_set   = set(dsd_top_ids)
        dense_set = set(dense_top_ids)

        # 順位差分: DSD での順位 - Dense での順位  (共通トークンのみ)
        dense_rank = {tid: r for r, tid in enumerate(dense_top_ids)}
        rank_diff  = {}
        for r, tid in enumerate(dsd_top_ids):
            if tid in dense_rank:
                rank_diff[tid] = r - dense_rank[tid]

        new_in_dsd   = sorted(dsd_set   - prev_dsd_top_ids)
        drop_dsd     = sorted(prev_dsd_top_ids - dsd_set)
        new_in_dense = sorted(dense_set  - prev_dense_top_ids)
        drop_dense   = sorted(prev_dense_top_ids - dense_set)

        prev_dsd_top_ids   = dsd_set
        prev_dense_top_ids = dense_set

        # ---- 差分統計 ----
        diff          = dsd_partial - dense_partial
        abs_diff      = diff.abs()
        max_abs_diff  = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
        argmax_token  = int(abs_diff.argmax())

        # ---- 停止条件 ----
        top2_dsd  = dsd_partial.topk(2)
        delta     = float(top2_dsd.values[0] - top2_dsd.values[1])
        B_bf16    = float(suffix_B_bf16[de]) if de < H else 0.0
        B_fp32    = float(suffix_B_fp32[de]) if de < H else 0.0
        fires     = delta > 2.0 * B_bf16

        if fires and not stopped:
            stopped   = True
            stop_tile = tile

        diverged = max_abs_diff > 0.0
        if diverged and first_diverge_tile is None:
            first_diverge_tile = tile

        # ---- コンソール出力 ----
        marker = ""
        if fires and stop_tile == tile:
            marker = "  *** STOP ***"
        if diverged and first_diverge_tile == tile:
            marker += "  <<< 初分岐 >>>"

        print(f"\ntile {tile:2d}  ds={ds}  de={de}"
              f"  max_abs_diff={max_abs_diff:.4e}"
              f"  delta={delta:.5f}  B_bf16={B_bf16:.4e}"
              f"  fires={fires}{marker}")

        # top20 並列表示 (最初の5行のみコンソール、残りはJSON)
        print(f"  {'rank':>4}  {'dsd_id':>8}  {'dsd_logit':>10}  "
              f"{'dense_id':>8}  {'dense_logit':>10}  {'rdiff':>6}")
        for r in range(min(5, top_k)):
            rdiff_str = ""
            if dsd_top_ids[r] in dense_rank:
                rdiff_str = f"{rank_diff.get(dsd_top_ids[r], '?'):+d}"
            print(f"  {r:>4}  {dsd_top_ids[r]:>8}  {dsd_top_vals[r]:>10.4f}  "
                  f"{dense_top_ids[r]:>8}  {dense_top_vals[r]:>10.4f}  {rdiff_str:>6}")
        if top_k > 5:
            print(f"  ... (残り {top_k-5} 行は JSON 参照)")

        if diverged:
            print(f"  argmax_diff: token={argmax_token}"
                  f"  dsd={float(dsd_partial[argmax_token]):.6f}"
                  f"  dense={float(dense_partial[argmax_token]):.6f}"
                  f"  diff={float(diff[argmax_token]):+.6e}")

        # ---- JSON レコード ----
        rec = {
            "tile":       tile,
            "dim_start":  ds,
            "dim_end":    de,
            "dsd_top20": [
                {"rank": r, "token_id": dsd_top_ids[r], "logit": dsd_top_vals[r]}
                for r in range(top_k)
            ],
            "dense_top20": [
                {"rank": r, "token_id": dense_top_ids[r], "logit": dense_top_vals[r]}
                for r in range(top_k)
            ],
            "rank_diff": {str(k): v for k, v in rank_diff.items()},
            "new_in_dsd_top20":    new_in_dsd,
            "drop_from_dsd_top20": drop_dsd,
            "new_in_dense_top20":    new_in_dense,
            "drop_from_dense_top20": drop_dense,
            "max_abs_diff":   round(max_abs_diff, 8),
            "mean_abs_diff":  round(mean_abs_diff, 10),
            "argmax_diff_token": argmax_token,
            "argmax_diff_dsd":   round(float(dsd_partial[argmax_token]), 6),
            "argmax_diff_dense": round(float(dense_partial[argmax_token]), 6),
            "argmax_diff_val":   round(float(diff[argmax_token]), 8),
            "current_top1": {"token_id": dsd_top_ids[0], "logit": dsd_top_vals[0]},
            "current_top2": {"token_id": dsd_top_ids[1], "logit": dsd_top_vals[1]},
            "delta":   round(delta, 8),
            "B_bf16":  round(B_bf16, 8),
            "B_fp32":  round(B_fp32, 8),
            "two_B_bf16":  round(2.0 * B_bf16, 8),
            "two_B_fp32":  round(2.0 * B_fp32, 8),
            "stop_fires":  fires,
            "first_diverge": first_diverge_tile == tile,
        }
        all_tiles.append(rec)

    # ---- サマリ ----
    print(f"\n{'='*70}")
    print(f"  n_tiles = {n_tiles}   stop_tile = {stop_tile}")
    print(f"  first_diverge_tile = {first_diverge_tile}  "
          f"(DSD と Dense が初めて異なるタイル)")
    print(f"{'='*70}")

    output = {
        "prompt":             TARGET_PROMPT,
        "n_tiles":            n_tiles,
        "chunk_size":         chunk_size,
        "hidden_dim":         H,
        "stop_tile":          stop_tile,
        "first_diverge_tile": first_diverge_tile,
        "top_k":              top_k,
        "note": {
            "dsd":   "W[:,idx] @ h_ord[ds:de]  (chunk GEMV)",
            "dense": "element-wise: sum_k W[:,idx[k]] * h_ord[ds+k]",
        },
        "tiles": all_tiles,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=MODEL_NAME)
    parser.add_argument("--prompt",      default=TARGET_PROMPT)
    parser.add_argument("--chunk_size",  type=int, default=CHUNK_SIZE)
    parser.add_argument("--top_k",       type=int, default=TOP_K)
    parser.add_argument("--output",      default="tile_trace_full.json")
    args = parser.parse_args()

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    hidden_dim = (cfg.text_config.hidden_size if hasattr(cfg, "text_config")
                  else cfg.hidden_size)
    n_tiles = math.ceil(hidden_dim / args.chunk_size)
    print(f"[model]  hidden_dim={hidden_dim}  device={device}")
    print(f"[trace]  chunk_size={args.chunk_size}  n_tiles={n_tiles}  top_k={args.top_k}")
    print(f"[prompt] {args.prompt!r}\n")

    W, bias        = get_lm_head_gpu(model, dtype=torch.bfloat16)
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)

    h_batch = batch_hidden_states_gpu(model, tokenizer, [args.prompt])
    torch.cuda.synchronize(device)

    trace_all_tiles(
        W, bias, h_batch[0], weight_col_max,
        args.chunk_size, args.top_k, args.output,
    )


if __name__ == "__main__":
    main()
