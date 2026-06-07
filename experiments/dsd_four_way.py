"""
experiments/dsd_four_way.py  --  4系統の最終値だけ比較

全48タイル完走後の partial_logit を 4 経路で計算し、
token 236743 と 609 の最終値だけを出力する。
途中ログなし。経路差分 (順序 / 累積dtype / GEMV vs element-wise) を潰す目的。

  系統1  GEMV BF16
         bound_first順, chunk GEMV (W[:,idx]@h_ord), bf16 累積
         → 本番 dsd_runtime と同一

  系統2  GEMV FP32 accumulator
         bound_first順, chunk GEMV (W[:,idx].float()@h_ord32), fp32 累積

  系統3  Dense FP32
         自然順, element-wise (W[:,i].float()*h32[i]), fp32 累積 (参照真値)

  系統4  Dense BF16 element-wise
         自然順, element-wise (W[:,i]*h[i]), bf16 累積

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
TOKEN_A       = 236743
TOKEN_B       = 609


def run(W, bias, h, weight_col_max, chunk_size, tok_a, tok_b):
    H       = h.shape[0]
    V       = W.shape[0]
    n_tiles = math.ceil(H / chunk_size)

    order  = order_bound_first_with_colmax(h, weight_col_max)
    h_ord  = h[order]
    h_ord32 = h_ord.float()
    h32     = h.float()

    def zeros_bf16():
        return (bias.clone() if bias is not None
                else torch.zeros(V, dtype=W.dtype, device=h.device))

    def zeros_fp32():
        return (bias.float().clone() if bias is not None
                else torch.zeros(V, dtype=torch.float32, device=h.device))

    # ---- 系統1: GEMV BF16 (bound_first, chunk GEMV, bf16 累積) ----
    s1 = zeros_bf16()
    for tile in range(1, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]
        s1.add_(W[:, idx] @ h_ord[ds:de])

    # ---- 系統2: GEMV FP32 accumulator (bound_first, chunk GEMV, fp32 累積) ----
    s2 = zeros_fp32()
    for tile in range(1, n_tiles + 1):
        ds  = (tile - 1) * chunk_size
        de  = min(tile * chunk_size, H)
        idx = order[ds:de]
        s2.add_(W[:, idx].float() @ h_ord32[ds:de])

    # ---- 系統3: Dense FP32 (自然順, element-wise, fp32 累積) = 参照真値 ----
    s3 = zeros_fp32()
    for i in range(H):
        s3.add_(W[:, i].float() * h32[i])

    # ---- 系統4: Dense BF16 element-wise (自然順, element-wise, bf16 累積) ----
    s4 = zeros_bf16()
    for i in range(H):
        s4.add_(W[:, i] * h[i])

    def fin(t):
        top2 = t.topk(2)
        return (int(top2.indices[0]), float(top2.values[0]),
                int(top2.indices[1]), float(top2.values[1]))

    rows = [
        ("1 GEMV BF16          (bound_first, chunk GEMV, bf16 acc)", s1),
        ("2 GEMV FP32 acc      (bound_first, chunk GEMV, fp32 acc)", s2),
        ("3 Dense FP32         (natural,     elem-wise,  fp32 acc)", s3),
        ("4 Dense BF16 elem    (natural,     elem-wise,  bf16 acc)", s4),
    ]

    print(f"\n{'='*78}")
    print(f"4系統 最終値比較   tokenA={tok_a}   tokenB={tok_b}")
    print(f"{'='*78}")
    print(f"  {'系統':<54}  {'logit[A]':>10}  {'logit[B]':>10}")
    print(f"  {'-'*54}  {'-'*10}  {'-'*10}")
    for label, t in rows:
        la = float(t[tok_a])
        lb = float(t[tok_b])
        winner = "A" if la > lb else "B"
        print(f"  {label:<54}  {la:>10.5f}  {lb:>10.5f}   top(A/B)={winner}")

    print(f"\n  {'系統':<54}  {'top1_id':>8}  {'top1_lv':>9}  {'top2_id':>8}  {'top2_lv':>9}")
    print(f"  {'-'*54}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}")
    for label, t in rows:
        t1i, t1v, t2i, t2v = fin(t)
        print(f"  {label:<54}  {t1i:>8d}  {t1v:>9.4f}  {t2i:>8d}  {t2v:>9.4f}")

    # ---- ペア差分: logit[A]-logit[B] を4系統で ----
    print(f"\n  {'系統':<54}  {'logit[A]-logit[B]':>18}")
    print(f"  {'-'*54}  {'-'*18}")
    for label, t in rows:
        d = float(t[tok_a]) - float(t[tok_b])
        print(f"  {label:<54}  {d:>+18.6f}")
    print(f"{'='*78}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--prompt",     default=TARGET_PROMPT)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--token_a",    type=int, default=TOKEN_A)
    parser.add_argument("--token_b",    type=int, default=TOKEN_B)
    args = parser.parse_args()

    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

    cfg = model.config
    hidden_dim = (cfg.text_config.hidden_size if hasattr(cfg, "text_config")
                  else cfg.hidden_size)
    print(f"[model]  hidden_dim={hidden_dim}  device={device}")
    print(f"[prompt] {args.prompt!r}")
    print(f"[tokens] A={args.token_a}  B={args.token_b}")

    W, bias        = get_lm_head_gpu(model, dtype=torch.bfloat16)
    weight_col_max = compute_weight_col_max(W)
    torch.cuda.synchronize(device)

    h_batch = batch_hidden_states_gpu(model, tokenizer, [args.prompt])
    torch.cuda.synchronize(device)

    run(W, bias, h_batch[0], weight_col_max,
        args.chunk_size, args.token_a, args.token_b)


if __name__ == "__main__":
    main()
