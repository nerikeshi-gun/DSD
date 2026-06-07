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


def pairwise_bound_analysis(
    W: torch.Tensor,
    h_ord: torch.Tensor,
    order: torch.Tensor,
    suffix_B_bf16: torch.Tensor,
    suffix_B_fp32: torch.Tensor,
    token_a: int,
    token_b: int,
    chunk_size: int,
    n_tiles: int,
) -> None:
    """
    トークン対 (A, B) に特化したペアワイズ残差上界を計算する。

      real_B_AB[from_dim] =
          sum_{p=from_dim}^{H-1} |W[A, order[p]] - W[B, order[p]]| * |h_ord[p]|

    全域 suffix_B は max_v|W[v,j]| を使うため real_B_AB より常に大きい (粗い)。
    もし delta > real_B_AB かつ delta < 2*B(global) なら
    DSD の停止判定は保守的すぎず、この対に限っては停止が正当化される。
    逆に delta < real_B_AB なら停止は誤り。

    また BF16 と FP32 の両方で real_B_AB を計算し、
    BF16 での精度損失がどの程度かを確認する。
    """
    H = h_ord.shape[0]

    # W の対応行を float32 で取得 (2行だけなので OOM しない)
    W_A = W[token_a, :].float()   # (H,) 元の列順
    W_B = W[token_b, :].float()   # (H,)

    # bound_first 順に並び替え
    W_A_ord = W_A[order]          # (H,) fp32
    W_B_ord = W_B[order]          # (H,) fp32
    h_ord32 = h_ord.float()       # (H,) fp32

    # 各次元の寄与: |W[A,order[p]] - W[B,order[p]]| * |h_ord[p]|
    pw_contrib_fp32 = (W_A_ord - W_B_ord).abs() * h_ord32.abs()   # (H,) fp32
    pw_contrib_bf16 = (W[token_a, order] - W[token_b, order]).abs() * h_ord.abs()  # (H,) bf16

    # suffix sum (tile 境界ごと)
    pw_suffix_fp32 = pw_contrib_fp32.flip(0).cumsum(0).flip(0)     # (H,) fp32
    pw_suffix_bf16 = pw_contrib_bf16.flip(0).cumsum(0).flip(0)     # (H,) bf16

    sep()
    print(f"[ペアワイズ残差上界]  token_A={token_a}  token_B={token_b}")
    sep("─")
    print(f"  real_B_AB[d] = sum_{{p>=d}} |W[A,order[p]] - W[B,order[p]]| * |h_ord[p]|")
    print(f"  global suffix_B[d] = sum_{{p>=d}} max_v|W[v,order[p]]| * |h_ord[p]|  (粗い上界)")
    print()
    print(f"  {'from_dim':>8}  {'tile':>5}  "
          f"{'real_B_fp32':>14}  {'real_B_bf16':>14}  "
          f"{'global_B_fp32':>15}  {'global_B_bf16':>15}  "
          f"{'ratio real/global':>18}")

    for tile in range(max(1, n_tiles - 5), n_tiles + 1):
        from_dim = (tile - 1) * chunk_size
        rB_fp32  = float(pw_suffix_fp32[from_dim]) if from_dim < H else 0.0
        rB_bf16  = float(pw_suffix_bf16[from_dim]) if from_dim < H else 0.0
        gB_fp32  = float(suffix_B_fp32[from_dim])  if from_dim < H else 0.0
        gB_bf16  = float(suffix_B_bf16[from_dim])  if from_dim < H else 0.0
        ratio    = rB_fp32 / gB_fp32 if gB_fp32 > 0 else float("nan")
        print(f"  {from_dim:>8}  {tile:>5}  "
              f"{rB_fp32:>14.6e}  {rB_bf16:>14.6e}  "
              f"{gB_fp32:>15.6e}  {gB_bf16:>15.6e}  "
              f"{ratio:>18.6f}")

    # tile 46 終端 (dim=3680) を詳しく確認
    sep("─")
    tile46_end = 46 * chunk_size   # = 3680
    if tile46_end < H:
        rB46_fp32 = float(pw_suffix_fp32[tile46_end])
        rB46_bf16 = float(pw_suffix_bf16[tile46_end])
        gB46_fp32 = float(suffix_B_fp32[tile46_end])
        gB46_bf16 = float(suffix_B_bf16[tile46_end])

        print(f"  [tile46 終端  dim={tile46_end}]")
        print(f"    real_B_AB  fp32 = {rB46_fp32:.8e}")
        print(f"    real_B_AB  bf16 = {rB46_bf16:.8e}")
        print(f"    global_B   fp32 = {gB46_fp32:.8e}")
        print(f"    global_B   bf16 = {gB46_bf16:.8e}")
        print(f"    ratio real/global (fp32) = "
              f"{rB46_fp32/gB46_fp32 if gB46_fp32 > 0 else 'N/A':.6f}")
        print()
        print(f"    寄与合計  fp32 = {float(pw_contrib_fp32[tile46_end:].sum()):.8e}")
        print(f"    寄与合計  bf16 = {float(pw_contrib_bf16[tile46_end:].sum()):.8e}")
        print()

        # tile46 以降の各次元の寄与トップ10
        remaining_contribs = pw_contrib_fp32[tile46_end:].clone()
        top10_vals, top10_pos = remaining_contribs.topk(min(10, len(remaining_contribs)))
        print(f"    tile46 以降の寄与トップ10 (ordered_dim, orig_dim, contrib):")
        for rank, (pos, val) in enumerate(zip(top10_pos.tolist(), top10_vals.tolist())):
            orig_dim = int(order[tile46_end + pos])
            print(f"      rank {rank+1:2d}: ordered_pos={tile46_end+pos:4d}  "
                  f"orig_dim={orig_dim:5d}  contrib={val:.6e}")


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

        # topk 値とテンソル直接参照が一致するか照合
        direct_top1 = float(dsd_partial[top1_id])
        direct_top2 = float(dsd_partial[top2_id])

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
        print(f"  current_top1  id={top1_id:>7d}   topk={top1_lv:>10.4f}   "
              f"dsd_partial[id]={direct_top1:>10.4f}   "
              f"match={top1_lv == direct_top1}")
        print(f"  current_top2  id={top2_id:>7d}   topk={top2_lv:>10.4f}   "
              f"dsd_partial[id]={direct_top2:>10.4f}   "
              f"match={top2_lv == direct_top2}")
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

    # ----------------------------------------------------------------
    # Dense 自然順の最終値を構築
    # ----------------------------------------------------------------
    dense_nat = (bias.clone() if bias is not None
                 else torch.zeros(V, dtype=W.dtype, device=h.device))
    for i in range(H):
        dense_nat.add_(W[:, i] * h[i])

    dn_top2 = dense_nat.topk(2)
    print(f"[Dense BF16 自然順 最終]  "
          f"top1={int(dn_top2.indices[0])}  logit={float(dn_top2.values[0]):.4f}  "
          f"top2={int(dn_top2.indices[1])}  logit={float(dn_top2.values[1]):.4f}  "
          f"margin={float(dn_top2.values[0]-dn_top2.values[1]):.4f}")

    # ----------------------------------------------------------------
    # dsd_partial (bound_first 累積) vs dense_nat (自然順累積) 差分診断
    # ----------------------------------------------------------------
    sep()
    print("[差分診断]  dsd_partial (bound_first BF16)  vs  dense_nat (natural BF16)")
    sep("─")

    def report_diff(a: torch.Tensor, b: torch.Tensor,
                    tok_a: int, tok_b: int, label: str):
        diff     = a - b
        abs_diff = diff.abs()
        print(f"  [{label}]")
        print(f"    max_abs_diff  = {float(abs_diff.max()):.6e}")
        print(f"    mean_abs_diff = {float(abs_diff.mean()):.6e}")
        print(f"    diff[token_A={tok_a}] = {float(diff[tok_a]):>+.6e}  "
              f"(dsd={float(a[tok_a]):.6f}  dense={float(b[tok_a]):.6f})")
        print(f"    diff[token_B={tok_b}] = {float(diff[tok_b]):>+.6e}  "
              f"(dsd={float(a[tok_b]):.6f}  dense={float(b[tok_b]):.6f})")
        # top1 確認
        top1_a = int(a.argmax())
        top1_b = int(b.argmax())
        print(f"    top1: dsd={top1_a}  dense={top1_b}  match={top1_a == top1_b}")
        print()

    # BF16 のまま比較
    report_diff(dsd_partial, dense_nat, token_a, token_b, "BF16")

    # float32 に昇格して再比較
    report_diff(dsd_partial.float(), dense_nat.float(), token_a, token_b, "float32 昇格後")

    # ----------------------------------------------------------------
    # float32 で全タイルを再累積して差が消えるか確認
    # ----------------------------------------------------------------
    sep("─")
    print("  [float32 で全タイルを再累積 (W は BF16 のまま chunk 昇格)]")
    dsd32 = (bias.float().clone() if bias is not None
             else torch.zeros(V, dtype=torch.float32, device=h.device))
    nat32 = (bias.float().clone() if bias is not None
             else torch.zeros(V, dtype=torch.float32, device=h.device))
    h_ord32 = h_ord.float()
    h32     = h.float()

    for tile in range(1, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]
        # W の chunk だけ float32 に昇格 (W全体を float32 化しない → OOM 回避)
        dsd32.add_(W[:, idx].float() @ h_ord32[ds:de])
        for i in range(ds, de):
            nat32.add_(W[:, order[i]].float() * h_ord32[i])

    report_diff(dsd32, nat32, token_a, token_b, "float32 再累積 (chunk 昇格)")

    sep("═")

    # ----------------------------------------------------------------
    # ペアワイズ真の残差上界: real_B_{tokenA}_{tokenB}
    #
    # 全域 B (suffix_B) は max_v|W[v,j]| を使う粗い上界。
    # トークン対 (A, B) だけに着目した真の残差上界は
    #
    #   real_B_AB[from_dim] = sum_{p=from_dim}^{H-1}
    #                           |W[tokenA, order[p]] - W[tokenB, order[p]]|
    #                           * |h_ord[p]|
    #
    # 停止条件  delta > 2*B  の 2*B は real_B_AB より常に大きい (粗い)。
    # real_B_AB < delta < 2*B の領域なら停止は誤り。
    # ----------------------------------------------------------------
    pairwise_bound_analysis(
        W=W, h_ord=h_ord, order=order,
        suffix_B_bf16=suffix_B_bf16, suffix_B_fp32=suffix_B_fp32,
        token_a=token_a, token_b=token_b,
        chunk_size=chunk_size, n_tiles=n_tiles,
    )

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
