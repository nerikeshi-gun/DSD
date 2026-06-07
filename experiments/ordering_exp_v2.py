"""
experiments/ordering_exp_v2.py  --  Phase 2B: B-reduction ordering

Ordering strategies compared:
  A. Baseline     : natural order 0, 1, ..., H-1
  B. Importance   : |W[top1,i] - W[top2,i]| descending  (Phase 2A)
  C. Bound-first  : bound_contribution_i = max_v|W[v,i]| * |h[i]| descending

Rationale for C:
  Stopping condition is delta > 2B where
  B = sum_{j>=current} max_v|W[v,j]| * |h[j]|.
  Processing high bound_contribution dimensions first collapses B fastest,
  giving delta the earliest opportunity to exceed 2B.

Algorithm (dsd_predict) and stopping condition (delta > 2B) unchanged.
Only the observation order changes via column permutation of h and W.

Output: ordering_comparison_v2.json
"""

import sys, os, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsd_poc import dsd_predict

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"
TRAJECTORY_INTERVAL = 128

DEFAULT_PROMPTS = [
    "The capital of Japan is",
    "Python is a",
    "The largest planet in the solar system is",
    "Linux is",
    "The opposite of hot is",
]


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
# Ordering strategies
# ---------------------------------------------------------------------------

def order_importance(h: torch.Tensor, W: torch.Tensor, bias) -> torch.Tensor:
    """
    Phase 2A: |W[top1,i] - W[top2,i]| descending.
    Requires a dense forward to identify top1/top2.
    """
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
        top2_ids = logit.topk(2).indices
        top1_id, top2_id = int(top2_ids[0]), int(top2_ids[1])
    importance = (W[top1_id, :] - W[top2_id, :]).abs()
    return importance.argsort(descending=True)


def order_bound_first(h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Phase 2B: bound_contribution_i = max_v|W[v,i]| * |h[i]| descending.
    Maximises the rate at which the remaining bound B collapses.
    """
    weight_col_max = W.abs().max(dim=0).values   # (hidden_dim,)
    bound_contrib  = weight_col_max * h.abs()    # (hidden_dim,)
    return bound_contrib.argsort(descending=True)


def order_bound_first_with_colmax(
    h: torch.Tensor,
    weight_col_max: torch.Tensor,
) -> torch.Tensor:
    """
    weight_col_max = max_v|W[v,i]| を事前計算済みで渡す版。
    W.abs() の全体テンソル生成を回避するために使う。
    """
    bound_contrib = weight_col_max * h.abs()     # (hidden_dim,)
    return bound_contrib.argsort(descending=True)


# ---------------------------------------------------------------------------
# Single-prompt experiment
# ---------------------------------------------------------------------------

def run_one(prompt, tokenizer, model, W, bias) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)
    h = get_final_hidden(model, input_ids)

    # Precompute orders once
    ord_imp   = order_importance(h, W, bias)
    ord_bound = order_bound_first(h, W)

    def reorder(order):
        return h[order], W[:, order]

    configs = {
        "baseline":       (h,                    W             ),
        "importance":     reorder(ord_imp),
        "bound_first":    reorder(ord_bound),
    }

    records = {}
    for name, (h_, W_) in configs.items():
        t0 = time.time()
        res = dsd_predict(h_, W_, bias, trajectory_interval=TRAJECTORY_INTERVAL)
        elapsed = time.time() - t0
        token = tokenizer.decode([res["token_id"]])
        records[name] = {
            "token":          token,
            "token_id":       res["token_id"],
            "match":          res["match"],
            "stop_dim":       res["stop_dim"],
            "hidden_dim":     res["hidden_dim"],
            "skip_rate":      res["skip_rate"],
            "max_delta":      res["max_delta"],
            "final_delta":    res["final_delta"],
            "stopping_ratio": res["stopping_ratio"],
            "elapsed_s":      round(elapsed, 3),
        }
        if "trajectory" in res:
            records[name]["trajectory"] = res["trajectory"]

    # Display
    hdim = records["baseline"]["hidden_dim"]
    print(f"Prompt : {prompt!r}")
    for name, r in records.items():
        print(f"  [{name:13s}]  stop={r['stop_dim']:5d}/{hdim}  "
              f"skip={r['skip_rate']*100:5.1f}%  "
              f"max_Δ={r['max_delta']:.3f}  "
              f"ratio={r['stopping_ratio']}  "
              f"match={r['match']}  token={r['token']!r}")

    base_stop = records["baseline"]["stop_dim"]
    for name in ("importance", "bound_first"):
        diff = base_stop - records[name]["stop_dim"]
        print(f"  stop_dim reduction vs baseline  [{name:13s}] : {diff:+d} "
              f"({diff/hdim*100:+.1f}%)")
    print()

    return {"prompt": prompt, **records}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2B: B-reduction ordering")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="ordering_comparison_v2.json")
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
        rec = run_one(p, tokenizer, model, W, bias)
        all_results.append(rec)

    # Aggregate
    n = len(all_results)
    names = ("baseline", "importance", "bound_first")

    def avg(key, field):
        return round(sum(r[key][field] for r in all_results) / n, 4)

    summary = {
        "n_prompts": n,
        "hidden_dim": hidden_dim,
    }
    for name in names:
        summary[name] = {
            "avg_stop_dim":  avg(name, "stop_dim"),
            "avg_skip_rate": avg(name, "skip_rate"),
            "top1_match_rate": sum(r[name]["match"] for r in all_results) / n,
        }

    base_avg_stop = summary["baseline"]["avg_stop_dim"]
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name in names:
        s = summary[name]
        reduction = base_avg_stop - s["avg_stop_dim"]
        print(f"  [{name:13s}]  avg_stop={s['avg_stop_dim']:.1f}/{hidden_dim}  "
              f"avg_skip={s['avg_skip_rate']*100:.1f}%  "
              f"match={s['top1_match_rate']*100:.0f}%  "
              f"reduction={reduction:+.1f}")
    print("=" * 60)

    output = {"summary": summary, "results": all_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
