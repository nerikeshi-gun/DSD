"""
experiments/ordering_exp.py  --  Phase 2A: Observation ordering experiment

Experiment A (baseline):
    Scan dimensions in natural order: 0, 1, 2, ..., H-1

Experiment B (importance order):
    1. Dense forward identifies top1, top2
    2. importance_i = |W[top1, i] - W[top2, i]|
    3. Sort dimensions descending by importance_i
    4. Run identical DSD algorithm on reordered h and W columns

Reordering h[order] and W[:, order] does not change the final logit value
(W_ord @ h_ord == W @ h for any permutation), so Top1 match is preserved.

Algorithm (dsd_predict) is not modified.
Stopping condition (delta > 2B) is not modified.

Output:
    ordering_comparison.json
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
# Model utilities (same as gemma_eval.py)
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
# Importance ordering
# ---------------------------------------------------------------------------

def compute_importance_order(h: torch.Tensor, W: torch.Tensor, bias=None) -> torch.Tensor:
    """
    Dense 推論で top1/top2 を特定し、
    importance_i = |W[top1, i] - W[top2, i]| の降順インデックスを返す。
    """
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
        top2 = logit.topk(2).indices
        top1_id, top2_id = int(top2[0]), int(top2[1])

    importance = (W[top1_id, :] - W[top2_id, :]).abs()   # (hidden_dim,)
    order = importance.argsort(descending=True)            # (hidden_dim,)
    return order, top1_id, top2_id


# ---------------------------------------------------------------------------
# Single-prompt experiment
# ---------------------------------------------------------------------------

def run_experiment(prompt, tokenizer, model, W, bias) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    h = get_final_hidden(model, input_ids)

    # ------------------------------------------------------------------
    # Experiment A: baseline (natural order)
    # ------------------------------------------------------------------
    t0 = time.time()
    res_base = dsd_predict(h, W, bias, trajectory_interval=TRAJECTORY_INTERVAL)
    base_time = time.time() - t0

    # ------------------------------------------------------------------
    # Experiment B: importance order
    # Requires dense top1/top2 first, then reorder h and W columns.
    # W_ord @ h_ord == W @ h for any permutation, so full logit is identical.
    # ------------------------------------------------------------------
    order, top1_id, top2_id = compute_importance_order(h, W, bias)

    h_ord = h[order]               # (hidden_dim,)
    W_ord = W[:, order]            # (vocab_size, hidden_dim)

    t1 = time.time()
    res_imp = dsd_predict(h_ord, W_ord, bias, trajectory_interval=TRAJECTORY_INTERVAL)
    imp_time = time.time() - t1

    # Decode tokens
    base_token = tokenizer.decode([res_base["token_id"]])
    imp_token  = tokenizer.decode([res_imp["token_id"]])

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    hidden_dim = res_base["hidden_dim"]
    print(f"Prompt : {prompt!r}")
    print(f"  dense top1={tokenizer.decode([top1_id])!r} (id={top1_id})  "
          f"top2={tokenizer.decode([top2_id])!r} (id={top2_id})")
    print(f"")
    print(f"  [A] Baseline (natural order)")
    print(f"      token       = {base_token!r}  match={res_base['match']}")
    print(f"      stop_dim    = {res_base['stop_dim']:5d} / {hidden_dim}  "
          f"skip={res_base['skip_rate']*100:.1f}%")
    print(f"      max_delta   = {res_base['max_delta']:.4f}")
    print(f"      final_delta = {res_base['final_delta']:.4f}")
    print(f"      stop_ratio  = {res_base['stopping_ratio']}")
    print(f"      time        = {base_time:.2f}s")
    print(f"")
    print(f"  [B] Importance order (|W[top1,i] - W[top2,i]| desc)")
    print(f"      token       = {imp_token!r}  match={res_imp['match']}")
    print(f"      stop_dim    = {res_imp['stop_dim']:5d} / {hidden_dim}  "
          f"skip={res_imp['skip_rate']*100:.1f}%")
    print(f"      max_delta   = {res_imp['max_delta']:.4f}")
    print(f"      final_delta = {res_imp['final_delta']:.4f}")
    print(f"      stop_ratio  = {res_imp['stopping_ratio']}")
    print(f"      time        = {imp_time:.2f}s")
    print(f"")

    stop_dim_delta = res_base["stop_dim"] - res_imp["stop_dim"]
    skip_delta = res_imp["skip_rate"] - res_base["skip_rate"]
    print(f"  stop_dim reduction : {stop_dim_delta:+d}  "
          f"({stop_dim_delta/hidden_dim*100:+.1f}%)")
    print(f"  skip_rate gain     : {skip_delta*100:+.1f}pp")
    print()

    def record(res, elapsed):
        r = {k: res[k] for k in
             ("token_id", "stop_dim", "hidden_dim", "skip_rate",
              "max_delta", "final_delta", "stopping_ratio", "match")}
        r["token"] = tokenizer.decode([res["token_id"]])
        r["elapsed_s"] = round(elapsed, 3)
        if "trajectory" in res:
            r["trajectory"] = res["trajectory"]
        return r

    return {
        "prompt": prompt,
        "dense_top1_id": top1_id,
        "dense_top1_token": tokenizer.decode([top1_id]),
        "dense_top2_id": top2_id,
        "baseline": record(res_base, base_time),
        "importance_order": record(res_imp, imp_time),
        "stop_dim_reduction": stop_dim_delta,
        "skip_rate_gain": round(skip_delta, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2A: ordering experiment")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="ordering_comparison.json")
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
        rec = run_experiment(p, tokenizer, model, W, bias)
        all_results.append(rec)

    # Aggregate
    n = len(all_results)
    avg_base_stop  = sum(r["baseline"]["stop_dim"] for r in all_results) / n
    avg_imp_stop   = sum(r["importance_order"]["stop_dim"] for r in all_results) / n
    avg_base_skip  = sum(r["baseline"]["skip_rate"] for r in all_results) / n
    avg_imp_skip   = sum(r["importance_order"]["skip_rate"] for r in all_results) / n
    n_match_base   = sum(r["baseline"]["match"] for r in all_results)
    n_match_imp    = sum(r["importance_order"]["match"] for r in all_results)

    summary = {
        "n_prompts": n,
        "hidden_dim": hidden_dim,
        "baseline": {
            "top1_match_rate": n_match_base / n,
            "avg_stop_dim": round(avg_base_stop, 2),
            "avg_skip_rate": round(avg_base_skip, 4),
        },
        "importance_order": {
            "top1_match_rate": n_match_imp / n,
            "avg_stop_dim": round(avg_imp_stop, 2),
            "avg_skip_rate": round(avg_imp_skip, 4),
        },
        "avg_stop_dim_reduction": round(avg_base_stop - avg_imp_stop, 2),
        "avg_skip_rate_gain": round(avg_imp_skip - avg_base_skip, 4),
    }

    print("=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"  prompts            : {n}")
    print(f"  [A] avg stop_dim   : {avg_base_stop:.1f} / {hidden_dim}  "
          f"({avg_base_skip*100:.1f}% skip)")
    print(f"  [B] avg stop_dim   : {avg_imp_stop:.1f} / {hidden_dim}  "
          f"({avg_imp_skip*100:.1f}% skip)")
    print(f"  stop_dim reduction : {avg_base_stop - avg_imp_stop:+.1f} dims")
    print(f"  skip_rate gain     : {(avg_imp_skip - avg_base_skip)*100:+.1f}pp")
    print(f"  top1 match A / B   : {n_match_base}/{n}  /  {n_match_imp}/{n}")
    print("=" * 55)

    output = {"summary": summary, "results": all_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
