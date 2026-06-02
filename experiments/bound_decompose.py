"""
experiments/bound_decompose.py  --  B の構成要素解析

各 checkpoint で:
  - B (remaining bound)
  - margin (top1 - top2)
  - B / margin 比率

さらに B の構成要素:
  contrib_i = max_v|W[v,i]| * |h[i]|

を降順ソートして上位100成分を出力。

目的:
  B が少数の巨大成分に支配されているか
  (= 上位 k 次元を先に処理すれば B を急減させられるか)
  それとも全体に薄く広がっているか
  (= どの順序で処理しても B はゆっくりしか減らない)

出力: bound_decompose.json
"""

import sys, os, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"

DEFAULT_PROMPTS = [
    "The capital of Japan is",
    "Python is a",
    "The largest planet in the solar system is",
    "Linux is",
    "The opposite of hot is",
]

CHECKPOINTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_COMPONENTS = 100


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_gemma(model_name):
    print(f"[load] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_final_hidden(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].detach().cpu().float()


def get_lm_head(model):
    lm = model.lm_head
    W = lm.weight.detach().cpu().float()
    bias = lm.bias.detach().cpu().float() if lm.bias is not None else None
    return W, bias


# ---------------------------------------------------------------------------
# B decomposition analysis
# ---------------------------------------------------------------------------

def analyze_bound(h: torch.Tensor, W: torch.Tensor, bias, checkpoints=CHECKPOINTS) -> dict:
    """
    全次元の contrib_i = max_v|W[v,i]| * |h[i]| を計算し、
    (1) 全体の分布統計
    (2) 降順上位 TOP_COMPONENTS 成分
    (3) 各 checkpoint での B / margin
    を返す。
    """
    H = h.shape[0]
    vocab_size = W.shape[0]

    # --- B 構成要素 ---
    weight_col_max = W.abs().max(dim=0).values   # (H,)
    h_abs          = h.abs()                     # (H,)
    contrib        = weight_col_max * h_abs       # (H,): contrib_i

    # suffix sum: suffix_B[i] = sum_{j=i}^{H-1} contrib[j] = B if we've done 0..i-1
    suffix_B = contrib.flip(0).cumsum(0).flip(0)  # (H,)
    total_B  = float(suffix_B[0])

    # --- 全体統計 ---
    contrib_sorted, sort_idx = contrib.sort(descending=True)
    contrib_sorted_list = contrib_sorted.tolist()

    # 上位 k 成分で B の何 % をカバーするか
    cumsum = contrib_sorted.cumsum(0) / total_B
    k_for_50 = int((cumsum < 0.50).sum()) + 1
    k_for_80 = int((cumsum < 0.80).sum()) + 1
    k_for_90 = int((cumsum < 0.90).sum()) + 1
    k_for_99 = int((cumsum < 0.99).sum()) + 1

    global_stats = {
        "total_B":        round(total_B, 4),
        "hidden_dim":     H,
        "mean_contrib":   round(float(contrib.mean()), 6),
        "std_contrib":    round(float(contrib.std()), 6),
        "max_contrib":    round(float(contrib_sorted[0]), 6),
        "min_contrib":    round(float(contrib_sorted[-1]), 6),
        "k_covers_50pct": k_for_50,
        "k_covers_80pct": k_for_80,
        "k_covers_90pct": k_for_90,
        "k_covers_99pct": k_for_99,
        "top1_contrib_pct": round(float(contrib_sorted[0] / total_B * 100), 3),
        "top10_contrib_pct": round(float(contrib_sorted[:10].sum() / total_B * 100), 3),
        "top100_contrib_pct": round(float(contrib_sorted[:100].sum() / total_B * 100), 3),
    }

    # 上位 TOP_COMPONENTS 成分
    top_components = [
        {
            "rank":     r + 1,
            "dim_idx":  int(sort_idx[r]),
            "contrib":  round(float(contrib_sorted[r]), 6),
            "pct_of_B": round(float(contrib_sorted[r] / total_B * 100), 4),
            "cumulative_pct": round(float(cumsum[r] * 100), 4),
        }
        for r in range(min(TOP_COMPONENTS, H))
    ]

    # --- checkpoint ごとの B / margin ---
    cp_dims = sorted(set(max(1, int(r * H)) for r in checkpoints))

    # partial_logit を累積して margin を取得
    partial_logit = bias.clone() if bias is not None else torch.zeros(vocab_size)
    cp_iter = iter(cp_dims)
    next_cp = next(cp_iter, None)
    checkpoint_records = []

    for i in range(H):
        partial_logit.add_(W[:, i] * h[i])
        dim_done = i + 1

        if next_cp is None or dim_done != next_cp:
            continue

        top2 = partial_logit.topk(2)
        top1_logit = float(top2.values[0])
        top2_logit = float(top2.values[1])
        margin     = top1_logit - top2_logit

        remaining_B = float(suffix_B[dim_done]) if dim_done < H else 0.0
        ratio = remaining_B / margin if margin > 1e-9 else None

        checkpoint_records.append({
            "observed_ratio": round(dim_done / H, 4),
            "dim_done":       dim_done,
            "remaining_B":    round(remaining_B, 4),
            "margin":         round(margin, 6),
            "B_over_margin":  round(ratio, 3) if ratio is not None else None,
            "top1_logit":     round(top1_logit, 4),
        })

        next_cp = next(cp_iter, None)
        if next_cp is None:
            break

    return {
        "global_stats":       global_stats,
        "top_components":     top_components,
        "checkpoint_records": checkpoint_records,
    }


# ---------------------------------------------------------------------------
# Single-prompt evaluation
# ---------------------------------------------------------------------------

def eval_one(prompt, tokenizer, model, W, bias) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)
    h = get_final_hidden(model, input_ids)

    t0 = time.time()
    result = analyze_bound(h, W, bias)
    elapsed = time.time() - t0

    gs = result["global_stats"]
    print(f"Prompt : {prompt!r}  ({elapsed:.1f}s)")
    print(f"  total_B       = {gs['total_B']:.2f}")
    print(f"  top-1  covers = {gs['top1_contrib_pct']:.3f}% of B")
    print(f"  top-10 covers = {gs['top10_contrib_pct']:.3f}% of B")
    print(f"  top-100 covers= {gs['top100_contrib_pct']:.3f}% of B")
    print(f"  dims to cover 50% B = {gs['k_covers_50pct']}")
    print(f"  dims to cover 80% B = {gs['k_covers_80pct']}")
    print(f"  dims to cover 90% B = {gs['k_covers_90pct']}")
    print(f"  dims to cover 99% B = {gs['k_covers_99pct']}")
    print()

    # B/margin table
    hdr = f"  {'ratio':>5}  {'remaining_B':>11}  {'margin':>8}  {'B/margin':>9}  top1_logit"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in result["checkpoint_records"]:
        bm = f"{r['B_over_margin']:9.2f}" if r["B_over_margin"] is not None else "     N/A"
        print(
            f"  {r['observed_ratio']*100:4.0f}%"
            f"  {r['remaining_B']:11.3f}"
            f"  {r['margin']:8.4f}"
            f"  {bm}"
            f"  {r['top1_logit']:10.4f}"
        )
    print()

    return {
        "prompt":  prompt,
        "elapsed_s": round(elapsed, 3),
        **result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bound decomposition analysis")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="bound_decompose.json")
    parser.add_argument("--prompts", nargs="*", default=None)
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    tokenizer, model = load_gemma(args.model)

    cfg = model.config
    if hasattr(cfg, 'text_config'):
        hidden_dim = cfg.text_config.hidden_size
        vocab_size = cfg.text_config.vocab_size
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model info]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  "
          f"params={n_params/1e9:.2f}B\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)\n")

    all_results = []
    for p in prompts:
        rec = eval_one(p, tokenizer, model, W, bias)
        all_results.append(rec)

    # Cross-prompt summary
    n = len(all_results)
    print("=" * 55)
    print("Cross-prompt summary")
    print("=" * 55)
    fields = [
        ("avg total_B",         lambda r: r["global_stats"]["total_B"]),
        ("avg top-1 % of B",    lambda r: r["global_stats"]["top1_contrib_pct"]),
        ("avg top-10 % of B",   lambda r: r["global_stats"]["top10_contrib_pct"]),
        ("avg top-100 % of B",  lambda r: r["global_stats"]["top100_contrib_pct"]),
        ("avg k for 50% B",     lambda r: r["global_stats"]["k_covers_50pct"]),
        ("avg k for 90% B",     lambda r: r["global_stats"]["k_covers_90pct"]),
        ("avg k for 99% B",     lambda r: r["global_stats"]["k_covers_99pct"]),
    ]
    for label, fn in fields:
        avg = sum(fn(r) for r in all_results) / n
        print(f"  {label:25s} = {avg:.3f}")
    print("=" * 55)

    summary = {f: round(sum(fn(r) for r in all_results)/n, 4) for f, fn in fields}

    output = {
        "model": args.model,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "summary": summary,
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
