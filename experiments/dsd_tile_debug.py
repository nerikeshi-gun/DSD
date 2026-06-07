"""
experiments/dsd_tile_debug.py  --  DSD tile 単位デバッグ

dsd_accuracy_diag.py で検出された DSD 固有誤差サンプルを対象に、
tile ごとの partial_logit・delta・B の推移を追跡する。

特に確認したい項目:
  1. tile ごとの idx 範囲 (dim_start / dim_end)
  2. partial_logit の DSD と Dense の差分 (delta_logit)
  3. suffix_B の値推移
  4. 停止条件 delta > 2B の発火タイミング

使い方:
  python experiments/dsd_tile_debug.py --prompt "The speed of light is approximately"
"""

import sys, os, math, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first_with_colmax
from experiments.dsd_runtime import (
    load_gemma, get_lm_head_gpu, batch_hidden_states_gpu,
    compute_weight_col_max,
)

MODEL_NAME  = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE  = 80
TARGET_PROMPT = "The speed of light is approximately"

# 追跡対象タイル: None なら全タイル、整数なら「最後の N タイル」のみ
TAIL_TILES = None   # None = 全タイル表示


def run_tile_trace(
    W: torch.Tensor,
    bias: torch.Tensor | None,
    h: torch.Tensor,
    weight_col_max: torch.Tensor,
    chunk_size: int,
    tail_tiles: int | None = None,
) -> None:
    H       = h.shape[0]
    V       = W.shape[0]
    n_tiles = math.ceil(H / chunk_size)

    order    = order_bound_first_with_colmax(h, weight_col_max)
    h_ord    = h[order]
    suffix_B = (weight_col_max[order] * h_ord.abs()).flip(0).cumsum(0).flip(0)

    # suffix_B の最後数値を確認
    print(f"\n[suffix_B 末尾確認]")
    for i in range(max(0, H - chunk_size * 3), H + 1, chunk_size):
        if i < H:
            val = float(suffix_B[i])
            print(f"  suffix_B[{i:4d}] = {val:.6e}")
    print(f"  suffix_B の dtype : {suffix_B.dtype}")
    print(f"  suffix_B 末尾 10件 : {suffix_B[-10:].float().tolist()}")

    # Dense BF16 partial logit を tile ごとに累積 (比較用)
    dense_partial = (bias.clone() if bias is not None
                     else torch.zeros(V, dtype=W.dtype, device=h.device))
    for i in range(H):
        dense_partial.add_(W[:, i] * h[i])   # 自然順 (並び替えなし)

    # DSD: tile ごとに追跡
    dsd_partial = (bias.clone() if bias is not None
                   else torch.zeros(V, dtype=W.dtype, device=h.device))

    # Dense も bound_first 順で並べた版 (DSD と同一の次元を踏む)
    dense_ord_partial = (bias.clone() if bias is not None
                         else torch.zeros(V, dtype=W.dtype, device=h.device))

    print(f"\n[tile-by-tile 追跡]")
    print(f"  n_tiles={n_tiles}  chunk_size={chunk_size}  H={H}")

    start_tile = 1 if tail_tiles is None else max(1, n_tiles - tail_tiles + 1)

    # tail_tiles 指定時は指定タイル手前まで高速スキップ
    if start_tile > 1:
        for tile in range(1, start_tile):
            ds = (tile - 1) * chunk_size
            de = min(tile * chunk_size, H)
            idx = order[ds:de]
            dsd_partial.add_(W[:, idx] @ h_ord[ds:de])
            dense_ord_partial.add_(W[:, idx] @ h_ord[ds:de])
        print(f"  (tile 1~{start_tile-1} をスキップ)\n")

    header = (f"  {'tile':>5}  {'dim_start':>9}  {'dim_end':>7}  "
              f"{'dsd_top1':>8}  {'dsd_top2':>8}  "
              f"{'delta':>10}  {'B':>10}  {'2B':>10}  "
              f"{'delta>2B':>8}  {'match_dense':>11}  {'delta_logit_max':>15}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    stop_tile  = n_tiles
    stopped    = False

    for tile in range(1, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]

        dsd_partial.add_(W[:, idx] @ h_ord[ds:de])
        dense_ord_partial.add_(W[:, idx] @ h_ord[ds:de])

        if tile < start_tile:
            continue

        # DSD 停止判定
        top2  = dsd_partial.topk(2)
        delta = float(top2.values[0] - top2.values[1])
        B_val = float(suffix_B[de]) if de < H else 0.0
        fires = delta > 2.0 * B_val

        # Dense との差分
        diff       = (dsd_partial - dense_ord_partial).abs()
        diff_max   = float(diff.max())

        dsd_top1_id = int(top2.indices[0])
        dsd_top2_id = int(top2.indices[1])

        dense_top1  = int(dense_ord_partial.argmax())
        match_dense = dsd_top1_id == dense_top1

        flag = " ← STOP" if fires and not stopped else ""
        if fires and not stopped:
            stop_tile = tile
            stopped   = True

        if tile >= start_tile:
            print(f"  {tile:>5}  {ds:>9}  {de:>7}  "
                  f"{dsd_top1_id:>8d}  {dsd_top2_id:>8d}  "
                  f"{delta:>10.5f}  {B_val:>10.5f}  {2*B_val:>10.5f}  "
                  f"{str(fires):>8}  {str(match_dense):>11}  "
                  f"{diff_max:>15.6e}{flag}")

        if stopped and tile > stop_tile:
            break   # 停止後 1 タイル余分に表示して終わり

    # 最終比較
    print(f"\n[最終結果]")
    dense_top2   = dense_partial.topk(2)
    dsd_top2_fin = dsd_partial.topk(2)

    print(f"  Dense BF16 (自然順) : top1={int(dense_top2.indices[0])}  "
          f"top2={int(dense_top2.indices[1])}  "
          f"margin={float(dense_top2.values[0]-dense_top2.values[1]):.5f}")
    print(f"  DSD BF16            : top1={int(dsd_top2_fin.indices[0])}  "
          f"top2={int(dsd_top2_fin.indices[1])}  "
          f"margin={float(dsd_top2_fin.values[0]-dsd_top2_fin.values[1]):.5f}  "
          f"stop_tile={stop_tile}")

    # stop_tile 周辺の suffix_B と delta の値を詳しく表示
    print(f"\n[stop_tile={stop_tile} 周辺の suffix_B 精度確認]")
    print(f"  (BF16 値と float32 変換値を比較)")
    for t in range(max(1, stop_tile - 2), min(n_tiles, stop_tile + 2) + 1):
        ds = (t - 1) * chunk_size
        de = min(t * chunk_size, H)
        bf16_B  = float(suffix_B[de]) if de < H else 0.0
        # float32 で再計算
        fp32_B  = float((weight_col_max[order][de:].float() * h_ord[de:].abs().float()).sum()) if de < H else 0.0
        print(f"  tile={t}  de={de}  "
              f"suffix_B[de](bf16)={bf16_B:.6e}  "
              f"suffix_B[de](fp32)={fp32_B:.6e}  "
              f"ratio={bf16_B/fp32_B if fp32_B != 0 else 'N/A':.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSD tile-level debug trace")
    parser.add_argument("--model",       default=MODEL_NAME)
    parser.add_argument("--prompt",      default=TARGET_PROMPT)
    parser.add_argument("--chunk_size",  type=int, default=CHUNK_SIZE)
    parser.add_argument("--tail_tiles",  type=int, default=TAIL_TILES,
                        help="最後の N タイルのみ表示 (None=全タイル)")
    args = parser.parse_args()

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    if hasattr(cfg, "text_config"):
        hidden_dim = cfg.text_config.hidden_size
    else:
        hidden_dim = cfg.hidden_size
    n_tiles = math.ceil(hidden_dim / args.chunk_size)
    print(f"[model]  hidden_dim={hidden_dim}  device={device}")
    print(f"[trace]  chunk_size={args.chunk_size}  n_tiles={n_tiles}")
    print(f"[prompt] {args.prompt!r}\n")

    W, bias        = get_lm_head_gpu(model, dtype=torch.bfloat16)
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)

    h_batch = batch_hidden_states_gpu(model, tokenizer, [args.prompt])
    torch.cuda.synchronize(device)
    h = h_batch[0]

    run_tile_trace(W, bias, h, weight_col_max, args.chunk_size,
                   tail_tiles=args.tail_tiles)


if __name__ == "__main__":
    main()
