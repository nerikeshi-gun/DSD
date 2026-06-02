"""
experiments/candidate_survival.py  --  Phase 2D: surviving candidate observation

目的:
  DSD スキャン中に「将来 top1 を逆転できる候補」が
  どのように減少するかを定量的に観測する。

定義:
  各 checkpoint (10%〜90% 観測率) において、
  候補 v が将来 top1 を逆転可能な条件:

    current_logit[v] + remaining_bound > current_top1_logit

  ここで remaining_bound はすべての候補に共通の上界:
    B = sum_{j=i+1}^{H-1} max_v|W[v,j]| * |h[j]|

  この条件を満たす v を surviving candidate と呼ぶ。

注意:
  - DSD アルゴリズム・停止条件は変更しない
  - chunk skip は実装しない
  - 観測のみ

出力: survival_curve.json
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
MAX_CANDIDATES_STORED = 200   # candidate_ids リストのサイズ上限


# ---------------------------------------------------------------------------
# Model utilities
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
    return out.hidden_states[-1][0, -1, :].detach().cpu().float()


def get_lm_head(model):
    lm = model.lm_head
    W = lm.weight.detach().cpu().float()
    bias = lm.bias.detach().cpu().float() if lm.bias is not None else None
    return W, bias


# ---------------------------------------------------------------------------
# Survival scan
# ---------------------------------------------------------------------------

def survival_scan(
    h: torch.Tensor,        # (H,)
    W: torch.Tensor,        # (vocab_size, H)
    bias,                   # (vocab_size,) or None
    checkpoints=CHECKPOINTS,
) -> list[dict]:
    """
    h を先頭から 1 次元ずつ累積しながら checkpoint ごとに
    surviving candidate を記録する。

    surviving candidate v の条件:
      partial_logit[v] + remaining_bound > partial_logit[top1]

    remaining_bound = sum_{j=i+1}^{H-1} max_v|W[v,j]| * |h[j]|
    は全候補共通の上界なので、スカラーとして計算できる。
    """
    H = h.shape[0]
    vocab_size = W.shape[0]

    # 全次元の bound 寄与を事前計算 (累積和で suffix sum)
    weight_col_max = W.abs().max(dim=0).values   # (H,)
    h_abs = h.abs()                              # (H,)
    bound_contrib = weight_col_max * h_abs       # (H,)
    # suffix_bound[i] = sum_{j=i}^{H-1} bound_contrib[j]
    suffix_bound = bound_contrib.flip(0).cumsum(0).flip(0)  # (H,)

    # checkpoint の次元インデックス
    cp_dims = sorted(set(max(1, int(r * H)) for r in checkpoints))
    cp_set  = set(cp_dims)

    # 初期ロジット
    partial_logit = bias.clone() if bias is not None else torch.zeros(vocab_size)

    records = []
    cp_iter = iter(sorted(cp_dims))
    next_cp = next(cp_iter, None)

    for i in range(H):
        partial_logit.add_(W[:, i] * h[i])

        dim_done = i + 1   # 処理済み次元数 (1-indexed)

        if next_cp is None or dim_done != next_cp:
            continue

        # --- checkpoint 処理 ---
        # remaining_bound: dim_done 以降の suffix
        remaining_B = float(suffix_bound[dim_done]) if dim_done < H else 0.0

        # top1 ロジット
        top2_res = partial_logit.topk(2)
        top1_logit = float(top2_res.values[0])
        top2_logit = float(top2_res.values[1])
        top1_id    = int(top2_res.indices[0])
        top2_id    = int(top2_res.indices[1])
        margin     = top1_logit - top2_logit

        # surviving candidates:
        #   partial_logit[v] + remaining_B > top1_logit
        #   ⟺  partial_logit[v] > top1_logit - remaining_B
        threshold  = top1_logit - remaining_B
        survivors  = (partial_logit > threshold).nonzero(as_tuple=True)[0]
        n_survivors = int(survivors.shape[0])

        # ID 一覧 (上位 MAX_CANDIDATES_STORED 件のみ保存)
        if n_survivors <= MAX_CANDIDATES_STORED:
            cand_ids = survivors.tolist()
        else:
            # ロジット降順で上位を保持
            top_surv = partial_logit[survivors].topk(MAX_CANDIDATES_STORED)
            cand_ids = survivors[top_surv.indices].tolist()

        observed_ratio = round(dim_done / H, 4)
        records.append({
            "observed_ratio":   observed_ratio,
            "dim_done":         dim_done,
            "hidden_dim":       H,
            "survivor_count":   n_survivors,
            "remaining_bound":  round(remaining_B, 6),
            "margin":           round(margin, 6),
            "top1_id":          top1_id,
            "top2_id":          top2_id,
            "candidate_ids":    cand_ids,
        })

        next_cp = next(cp_iter, None)
        if next_cp is None:
            break

    return records


# ---------------------------------------------------------------------------
# Single-prompt evaluation
# ---------------------------------------------------------------------------

def eval_one(prompt, tokenizer, model, W, bias) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)
    h = get_final_hidden(model, input_ids)

    t0 = time.time()
    records = survival_scan(h, W, bias)
    elapsed = time.time() - t0

    # Decode top tokens for display
    print(f"Prompt : {prompt!r}  ({elapsed:.1f}s)")
    print(f"  {'ratio':>6}  {'survivors':>10}  {'margin':>8}  {'top1_token':>15}")
    for r in records:
        top1_tok = tokenizer.decode([r["top1_id"]])
        print(f"  {r['observed_ratio']*100:5.0f}%  "
              f"{r['survivor_count']:>10,}  "
              f"{r['margin']:8.4f}  "
              f"{top1_tok!r:>15}")
    print()

    # Annotate candidate_ids with token text (top-20 only for readability)
    checkpoints_out = []
    for r in records:
        top1_tok = tokenizer.decode([r["top1_id"]])
        top2_tok = tokenizer.decode([r["top2_id"]])
        top_cands = [
            {"id": cid, "token": tokenizer.decode([cid])}
            for cid in r["candidate_ids"][:20]
        ]
        checkpoints_out.append({
            "observed_ratio":       r["observed_ratio"],
            "dim_done":             r["dim_done"],
            "hidden_dim":           r["hidden_dim"],
            "survivor_count":       r["survivor_count"],
            "remaining_bound":      r["remaining_bound"],
            "margin":               r["margin"],
            "top1_id":              r["top1_id"],
            "top1_token":           top1_tok,
            "top2_id":              r["top2_id"],
            "top2_token":           top2_tok,
            "top_candidates":       top_cands,    # top-20, decoded
            "all_candidate_ids":    r["candidate_ids"],
        })

    return {
        "prompt":      prompt,
        "hidden_dim":  h.shape[0],
        "elapsed_s":   round(elapsed, 3),
        "checkpoints": checkpoints_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2D: candidate survival curve")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="survival_curve.json")
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

    # Cross-prompt summary: average survivor_count per checkpoint ratio
    # Align by observed_ratio (same CHECKPOINTS for all prompts)
    n = len(all_results)
    ratio_to_counts = {}
    for rec in all_results:
        for cp in rec["checkpoints"]:
            key = cp["observed_ratio"]
            ratio_to_counts.setdefault(key, []).append(cp["survivor_count"])

    summary_rows = []
    print("=" * 55)
    print("Average surviving candidates per checkpoint")
    print("=" * 55)
    for ratio in sorted(ratio_to_counts):
        counts = ratio_to_counts[ratio]
        avg = sum(counts) / len(counts)
        pct = ratio * 100
        print(f"  {pct:4.0f}% : {avg:,.1f}")
        summary_rows.append({"observed_ratio": ratio, "avg_survivor_count": round(avg, 1)})
    print("=" * 55)

    output = {
        "model": args.model,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "summary": summary_rows,
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
