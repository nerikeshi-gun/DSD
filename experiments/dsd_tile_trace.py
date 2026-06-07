"""
experiments/dsd_tile_trace.py  --  DSD tile 単位観測

"The speed of light is approximately" で発生している DSD 固有誤差の原因追跡。
修正ではなく観測のみ行う。

対象トークン:
  tokenA = 563  (Dense top1)
  tokenB = 528  (DSD top1 = 誤り)

tile 43~48 の各タイルで:
  - partial_logit[tokenA] / [tokenB] の推移
  - DSD と Dense の partial_logit 差分
  - delta / B / 2B / 停止条件発火
"""

import sys, os, math, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first_with_colmax
from experiments.dsd_runtime import (
    load_gemma, get_lm_head_gpu, batch_hidden_states_gpu,
    compute_weight_col_max,
)

MODEL_NAME    = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE    = 80
TARGET_PROMPT = "The speed of light is approximately"
TOKEN_A       = 563
TOKEN_B       = 528
TRACE_FROM    = 43   # このタイルから表示


def sep(char="─", n=90):
    print(char * n)


def trace(
    W: torch.Tensor,
    bias: torch.Tensor | None,
    h: torch.Tensor,
    weight_col_max: torch.Tensor,
    chunk_size: int,
    token_a: int,
    token_b: int,
    trace_from: int,
) -> None:
    H       = h.shape[0]
    V       = W.shape[0]
    n_tiles = math.ceil(H / chunk_size)

    order    = order_bound_first_with_colmax(h, weight_col_max)
    h_ord    = h[order]

    # suffix_B を float32 と bfloat16 の両方で計算しておく
    contrib_bf16 = weight_col_max[order] * h_ord.abs()          # (H,) bf16
    contrib_fp32 = weight_col_max[order].float() * h_ord.abs().float()  # (H,) fp32

    suffix_B_bf16 = contrib_bf16.flip(0).cumsum(0).flip(0)      # (H,) bf16
    suffix_B_fp32 = contrib_fp32.flip(0).cumsum(0).flip(0)      # (H,) fp32

    print(f"[suffix_B 末尾確認]")
    print(f"  {'dim':>5}  {'bf16':>14}  {'fp32':>14}  {'ratio bf16/fp32':>16}")
    for i in range(max(0, H - chunk_size * 3), H + 1, chunk_size):
        if i < H:
            b = float(suffix_B_bf16[i])
            f = float(suffix_B_fp32[i])
            ratio = (b / f) if f != 0.0 else float("nan")
            print(f"  {i:>5}  {b:>14.6e}  {f:>14.6e}  {ratio:>16.6f}")
    print(f"  {'H':>5}  {'0 (end)':>14}  {'0 (end)':>14}")

    # DSD partial logit (bound_first 順で累積)
    dsd_partial = (bias.clone() if bias is not None
                   else torch.zeros(V, dtype=W.dtype, device=h.device))

    # Dense partial logit (bound_first 順で同一次元を踏む — idx は共通)
    dense_partial = (bias.clone() if bias is not None
                     else torch.zeros(V, dtype=W.dtype, device=h.device))

    # trace_from より前を高速スキップ
    for tile in range(1, trace_from):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]
        dsd_partial.add_(W[:, idx] @ h_ord[ds:de])
        dense_partial.add_(W[:, idx] @ h_ord[ds:de])

    stopped = False

    for tile in range(trace_from, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]

        # 更新前スナップショット
        before_A  = float(dsd_partial[token_a])
        before_B_ = float(dsd_partial[token_b])
        partial_before = dsd_partial.clone()

        # タイルの寄与ベクトルを別変数で取得してから加算
        chunk_contrib = W[:, idx] @ h_ord[ds:de]   # (V,) bf16
        dsd_partial.add_(chunk_contrib)
        dense_partial.add_(W[:, idx] @ h_ord[ds:de])

        # 更新後
        dsd_A   = float(dsd_partial[token_a])
        dsd_B_  = float(dsd_partial[token_b])
        den_A   = float(dense_partial[token_a])
        den_B_  = float(dense_partial[token_b])

        # 更新量の診断
        chunk_norm      = float(chunk_contrib.norm())
        partial_norm    = float(dsd_partial.norm())
        max_abs_change  = float((dsd_partial - partial_before).abs().max())
        change_A        = dsd_A  - before_A
        change_B        = dsd_B_ - before_B_

        top2    = dsd_partial.topk(2)
        top1_id = int(top2.indices[0])
        top2_id = int(top2.indices[1])
        top1_lv = float(top2.values[0])
        top2_lv = float(top2.values[1])

        delta   = top1_lv - top2_lv

        B_bf16  = float(suffix_B_bf16[de]) if de < H else 0.0
        B_fp32  = float(suffix_B_fp32[de]) if de < H else 0.0
        fires   = delta > 2.0 * B_bf16

        sep()
        print(f"tile {tile}   dim_start={ds}  dim_end={de}  "
              f"idx[0]={int(idx[0])}..idx[-1]={int(idx[-1])}")
        sep("─")

        # 更新量の診断 (実際に partial_logit が変化しているか)
        print(f"  [更新量]")
        print(f"    chunk_norm      = {chunk_norm:.6e}   "
              f"(このタイルの W[:,idx]@h_ord の L2 norm)")
        print(f"    partial_norm    = {partial_norm:.6f}   "
              f"(partial_logit 全体の L2 norm)")
        print(f"    max_abs_change  = {max_abs_change:.6e}   "
              f"(partial_logit 全要素の最大変化量)")
        print(f"    change_A        = {change_A:>+.6e}   "
              f"(logit[{token_a}] の変化量)")
        print(f"    change_B        = {change_B:>+.6e}   "
              f"(logit[{token_b}] の変化量)")
        print()

        print(f"  partial_logit[tokenA={token_a}]  dsd={dsd_A:>10.4f}   dense={den_A:>10.4f}   "
              f"delta_A(dsd-dense)={dsd_A - den_A:>+10.6f}")
        print(f"  partial_logit[tokenB={token_b}]  dsd={dsd_B_:>10.4f}   dense={den_B_:>10.4f}   "
              f"delta_B(dsd-dense)={dsd_B_ - den_B_:>+10.6f}")
        print()
        print(f"  diff = logit[A] - logit[B]"
              f"   dsd={dsd_A - dsd_B_:>+10.6f}   dense={den_A - den_B_:>+10.6f}")
        print()
        print(f"  current_top1  id={top1_id:>7d}   logit={top1_lv:>10.4f}")
        print(f"  current_top2  id={top2_id:>7d}   logit={top2_lv:>10.4f}")
        print()
        print(f"  delta (top1-top2) = {delta:>12.6f}")
        print(f"  B  (bf16)         = {B_bf16:>12.6e}   2B={2*B_bf16:>12.6e}")
        print(f"  B  (fp32)         = {B_fp32:>12.6e}   2B={2*B_fp32:>12.6e}")
        print()
        print(f"  delta > 2B (bf16) = {fires}"
              + ("  ← STOP HERE" if fires and not stopped else ""))

        if fires and not stopped:
            stopped = True
            print()
            print(f"  *** DSD stops at tile {tile} ***")
            print(f"  *** returned top1={top1_id}  (correct={token_a}) ***")

    sep("═")
    # Dense 自然順の最終 top2
    dense_nat = (bias.clone() if bias is not None
                 else torch.zeros(V, dtype=W.dtype, device=h.device))
    for i in range(H):
        dense_nat.add_(W[:, i] * h[i])
    dn_top2 = dense_nat.topk(2)
    print(f"[Dense BF16 自然順 最終]  "
          f"top1={int(dn_top2.indices[0])}  logit={float(dn_top2.values[0]):.4f}  "
          f"top2={int(dn_top2.indices[1])}  logit={float(dn_top2.values[1]):.4f}  "
          f"margin={float(dn_top2.values[0]-dn_top2.values[1]):.4f}")
    sep("═")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=MODEL_NAME)
    parser.add_argument("--prompt",      default=TARGET_PROMPT)
    parser.add_argument("--chunk_size",  type=int, default=CHUNK_SIZE)
    parser.add_argument("--token_a",     type=int, default=TOKEN_A)
    parser.add_argument("--token_b",     type=int, default=TOKEN_B)
    parser.add_argument("--trace_from",  type=int, default=TRACE_FROM)
    args = parser.parse_args()

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    hidden_dim = (cfg.text_config.hidden_size if hasattr(cfg, "text_config")
                  else cfg.hidden_size)
    n_tiles = math.ceil(hidden_dim / args.chunk_size)
    print(f"[model]  hidden_dim={hidden_dim}  device={device}")
    print(f"[trace]  chunk_size={args.chunk_size}  n_tiles={n_tiles}"
          f"  trace_from={args.trace_from}")
    print(f"[prompt] {args.prompt!r}")
    print(f"[tokens] A={args.token_a}  B={args.token_b}\n")

    W, bias        = get_lm_head_gpu(model, dtype=torch.bfloat16)
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)

    h_batch = batch_hidden_states_gpu(model, tokenizer, [args.prompt])
    torch.cuda.synchronize(device)

    trace(W, bias, h_batch[0], weight_col_max,
          args.chunk_size, args.token_a, args.token_b, args.trace_from)


if __name__ == "__main__":
    main()
