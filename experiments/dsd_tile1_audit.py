"""
experiments/dsd_tile1_audit.py  --  tile=1 監査

目的: tile=1 時点で GEMV vs 逐次加算の差の発生源を特定する
  監査1: idx/order/h_ord/h の対応確認
  監査2: GEMV vs manual の max/mean abs diff
  監査3: torch.allclose (BF16)
  監査4: manual BF16 vs FP32
  監査5: GEMV  BF16 vs FP32

修正禁止。観測のみ。
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


def run_audit(W, h, weight_col_max, chunk_size):
    H      = h.shape[0]
    V      = W.shape[0]
    ds, de = 0, chunk_size   # tile=1

    order   = order_bound_first_with_colmax(h, weight_col_max)
    h_ord   = h[order]
    idx     = order[ds:de]   # shape: (chunk_size,)

    sep = "=" * 60

    # ----------------------------------------------------------------
    # 監査1: 添字対応
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査1: idx / order / h_ord / h[idx] 対応確認")
    print(f"{sep}")
    print(f"  order[:20] = {order[:20].tolist()}")
    print(f"  idx[:20]   = {idx[:20].tolist()}")
    print(f"  h_ord[:20] (values) = {h_ord[:20].float().tolist()}")
    print(f"  h[idx[:20]] (values) = {h[idx[:20]].float().tolist()}")

    eq_check = torch.equal(h_ord[:20], h[idx[:20]])
    print(f"\n  torch.equal(h_ord[:20], h[idx[:20]]) = {eq_check}")
    if not eq_check:
        diff_idx = (h_ord[:20] != h[idx[:20]]).nonzero(as_tuple=True)[0]
        print(f"  不一致インデックス: {diff_idx.tolist()}")
        for i in diff_idx[:5].tolist():
            print(f"    pos={i}  h_ord={float(h_ord[i]):.6f}  "
                  f"h[idx[{i}]]={float(h[idx[i]]):.6f}")

    # ----------------------------------------------------------------
    # 監査2: GEMV vs manual (BF16)
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査2: GEMV vs 逐次加算 (BF16) max/mean abs diff")
    print(f"{sep}")

    W_tile  = W[:, idx]                               # (V, chunk_size)
    h_tile  = h_ord[ds:de]                            # (chunk_size,)

    gemv_bf16   = W_tile @ h_tile                     # (V,) BF16

    manual_bf16 = torch.zeros(V, dtype=W.dtype, device=W.device)
    for k in range(len(idx)):
        manual_bf16.add_(W[:, idx[k]] * h_ord[ds + k])

    diff2        = (gemv_bf16 - manual_bf16).abs()
    max_diff2    = float(diff2.max())
    mean_diff2   = float(diff2.mean())
    argmax2      = int(diff2.argmax())

    print(f"  max_abs_diff  = {max_diff2:.6e}")
    print(f"  mean_abs_diff = {mean_diff2:.6e}")
    print(f"  argmax token  = {argmax2}")
    print(f"    gemv_bf16[argmax]   = {float(gemv_bf16[argmax2]):.6f}")
    print(f"    manual_bf16[argmax] = {float(manual_bf16[argmax2]):.6f}")

    # ----------------------------------------------------------------
    # 監査3: torch.allclose (BF16)
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査3: torch.allclose (BF16, atol=1e-2, rtol=1e-2)")
    print(f"{sep}")
    close3 = torch.allclose(gemv_bf16, manual_bf16, atol=1e-2, rtol=1e-2)
    print(f"  torch.allclose = {close3}")
    n_mismatch = int((diff2 > 1e-2).sum())
    print(f"  diff > 1e-2: {n_mismatch} / {V} tokens")

    # ----------------------------------------------------------------
    # 監査4: manual BF16 vs FP32
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査4: manual BF16 vs FP32")
    print(f"{sep}")

    manual_fp32 = torch.zeros(V, dtype=torch.float32, device=W.device)
    for k in range(len(idx)):
        manual_fp32.add_(W[:, idx[k]].float() * float(h_ord[ds + k]))

    diff4      = (manual_bf16.float() - manual_fp32).abs()
    max_diff4  = float(diff4.max())
    mean_diff4 = float(diff4.mean())
    argmax4    = int(diff4.argmax())

    print(f"  max_abs_diff  = {max_diff4:.6e}")
    print(f"  mean_abs_diff = {mean_diff4:.6e}")
    print(f"  argmax token  = {argmax4}")
    print(f"    manual_bf16[argmax] = {float(manual_bf16[argmax4]):.6f}")
    print(f"    manual_fp32[argmax] = {float(manual_fp32[argmax4]):.6f}")

    # ----------------------------------------------------------------
    # 監査5: GEMV BF16 vs FP32
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査5: GEMV BF16 vs FP32")
    print(f"{sep}")

    gemv_fp32  = W_tile.float() @ h_tile.float()     # (V,) FP32

    diff5      = (gemv_bf16.float() - gemv_fp32).abs()
    max_diff5  = float(diff5.max())
    mean_diff5 = float(diff5.mean())
    argmax5    = int(diff5.argmax())

    print(f"  max_abs_diff  = {max_diff5:.6e}")
    print(f"  mean_abs_diff = {mean_diff5:.6e}")
    print(f"  argmax token  = {argmax5}")
    print(f"    gemv_bf16[argmax] = {float(gemv_bf16[argmax5]):.6f}")
    print(f"    gemv_fp32[argmax] = {float(gemv_fp32[argmax5]):.6f}")

    # ----------------------------------------------------------------
    # 監査6: manual FP32 accumulator vs GEMV BF16
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査6: manual FP32 accumulator vs GEMV BF16")
    print(f"{sep}")

    manual_fp32_acc = torch.zeros(V, dtype=torch.float32, device=W.device)
    for k in range(len(idx)):
        manual_fp32_acc.add_(W[:, idx[k]].float() * float(h_ord[ds + k]))

    diff6      = (manual_fp32_acc - gemv_bf16.float()).abs()
    max_diff6  = float(diff6.max())
    mean_diff6 = float(diff6.mean())
    argmax6    = int(diff6.argmax())

    print(f"  max_abs_diff  = {max_diff6:.6e}")
    print(f"  mean_abs_diff = {mean_diff6:.6e}")
    print(f"  argmax token  = {argmax6}")
    print(f"    manual_fp32_acc[argmax] = {float(manual_fp32_acc[argmax6]):.6f}")
    print(f"    gemv_bf16[argmax]       = {float(gemv_bf16[argmax6]):.6f}")

    # ----------------------------------------------------------------
    # 監査7: manual FP32 accumulator vs GEMV FP32
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("監査7: manual FP32 accumulator vs GEMV FP32")
    print(f"{sep}")

    diff7      = (manual_fp32_acc - gemv_fp32).abs()
    max_diff7  = float(diff7.max())
    mean_diff7 = float(diff7.mean())
    argmax7    = int(diff7.argmax())

    print(f"  max_abs_diff  = {max_diff7:.6e}")
    print(f"  mean_abs_diff = {mean_diff7:.6e}")
    print(f"  argmax token  = {argmax7}")
    print(f"    manual_fp32_acc[argmax] = {float(manual_fp32_acc[argmax7]):.6f}")
    print(f"    gemv_fp32[argmax]       = {float(gemv_fp32[argmax7]):.6f}")

    close7 = torch.allclose(manual_fp32_acc, gemv_fp32, atol=1e-3, rtol=1e-3)
    print(f"  torch.allclose(atol=1e-3, rtol=1e-3) = {close7}")

    # ----------------------------------------------------------------
    # クロス比較サマリ
    # ----------------------------------------------------------------
    print(f"\n{sep}")
    print("サマリ")
    print(f"{sep}")
    print(f"  監査1  h_ord対応                          : {'OK' if eq_check else 'NG ← 添字バグ'}")
    print(f"  監査2  GEMV BF16    vs manual BF16  max   : {max_diff2:.6e}")
    print(f"  監査4  manual BF16  vs manual FP32  max   : {max_diff4:.6e}")
    print(f"  監査5  GEMV BF16    vs GEMV FP32    max   : {max_diff5:.6e}")
    print(f"  監査6  manual FP32  vs GEMV BF16    max   : {max_diff6:.6e}")
    print(f"  監査7  manual FP32  vs GEMV FP32    max   : {max_diff7:.6e}  allclose={close7}")

    print()
    if not eq_check:
        print("  結論D: h_ord と h[idx] が不一致 ← 添字バグ")
    elif max_diff7 < 1e-2 and max_diff6 < 0.1:
        print("  結論C: FP32同士は一致。差の原因は BF16 逐次加算誤差 (manual側)")
        print("         dsd_full_trace の Dense 側が不正確。DSD (GEMV) は正しい。")
    elif max_diff2 > 0.1:
        print("  結論B: GEMV BF16 と manual BF16 で大きな差 ← 演算順序差が主因")
    else:
        print("  結論A/?: tile=1 では有意な差なし (後タイルを調査)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--prompt",     default=TARGET_PROMPT)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    hidden_dim = (cfg.text_config.hidden_size if hasattr(cfg, "text_config")
                  else cfg.hidden_size)
    print(f"[model]  hidden_dim={hidden_dim}  device={device}")
    print(f"[prompt] {args.prompt!r}")
    print(f"[tile=1] chunk_size={args.chunk_size}")

    W, bias        = get_lm_head_gpu(model, dtype=torch.bfloat16)
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)

    h_batch = batch_hidden_states_gpu(model, tokenizer, [args.prompt])
    torch.cuda.synchronize(device)
    h = h_batch[0]

    run_audit(W, h, weight_col_max, args.chunk_size)


if __name__ == "__main__":
    main()
