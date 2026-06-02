"""
experiments/logit_inspect.py  --  Phase 2C: logit distribution analysis

For each prompt:
  1. Run full Gemma forward pass
  2. Compute W @ h over the full vocabulary (no truncation)
  3. Extract top-50 tokens with logit and margin from top1
  4. Compute margin statistics: top1 vs top-k for k in {2,4,8,16,32,64}

Output: top50_logits.json

No DSD modifications. Observation only.
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

MARGIN_RANKS = [2, 4, 8, 16, 32, 64]
TOP_K = 50


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
# Logit inspection
# ---------------------------------------------------------------------------

def inspect_logits(prompt, tokenizer, model, W, bias, top_k=TOP_K) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    h = get_final_hidden(model, input_ids)

    with torch.no_grad():
        logits = W @ h
        if bias is not None:
            logits = logits + bias                      # (vocab_size,)

    vocab_size = logits.shape[0]

    # Top-K extraction
    topk_vals, topk_ids = logits.topk(top_k)
    topk_vals = topk_vals.tolist()
    topk_ids  = topk_ids.tolist()
    top1_logit = topk_vals[0]

    top50 = []
    for rank, (tid, logit) in enumerate(zip(topk_ids, topk_vals), start=1):
        try:
            text = tokenizer.decode([tid])
        except Exception:
            text = f"<id={tid}>"
        top50.append({
            "rank":   rank,
            "token":  text,
            "id":     tid,
            "logit":  round(logit, 4),
            "margin": round(top1_logit - logit, 4),
        })

    # Margin statistics: top1 - top_k  for k in MARGIN_RANKS
    margins = {}
    for k in MARGIN_RANKS:
        if k <= len(topk_vals):
            margins[f"top1_vs_top{k}"] = round(top1_logit - topk_vals[k - 1], 4)
        else:
            # Need to look further
            val_k = logits.topk(k).values[-1].item()
            margins[f"top1_vs_top{k}"] = round(top1_logit - val_k, 4)

    # How many tokens are "realistically competing"?
    # Define as: logit >= top1 - 1.0  (within 1 nit of top1)
    within_1 = int((logits >= top1_logit - 1.0).sum().item())
    within_2 = int((logits >= top1_logit - 2.0).sum().item())
    within_5 = int((logits >= top1_logit - 5.0).sum().item())

    # Display
    print(f"Prompt : {prompt!r}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  top1 = {top50[0]['token']!r}  logit = {top1_logit:.4f}")
    print(f"  margins:")
    for k, v in margins.items():
        print(f"    {k:18s} = {v:.4f}")
    print(f"  competing tokens:")
    print(f"    within 1.0 nit  = {within_1}")
    print(f"    within 2.0 nit  = {within_2}")
    print(f"    within 5.0 nit  = {within_5}")
    print(f"  top-5 tokens:")
    for t in top50[:5]:
        bar = "█" * max(0, int((t["logit"] - topk_vals[-1]) /
                               max(top1_logit - topk_vals[-1], 1e-6) * 20))
        print(f"    [{t['rank']:2d}] {t['token']!r:20s}  {t['logit']:8.4f}  "
              f"margin={t['margin']:7.4f}  {bar}")
    print()

    return {
        "prompt":   prompt,
        "vocab_size": vocab_size,
        "top1_logit": round(top1_logit, 4),
        "margins":  margins,
        "competing_tokens": {
            "within_1_nit": within_1,
            "within_2_nit": within_2,
            "within_5_nit": within_5,
        },
        "top50": top50,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2C: logit distribution analysis")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="top50_logits.json")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--top_k", type=int, default=TOP_K)
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

    all_records = []
    for p in prompts:
        rec = inspect_logits(p, tokenizer, model, W, bias, top_k=args.top_k)
        all_records.append(rec)

    # Cross-prompt summary
    n = len(all_records)
    print("=" * 60)
    print("Cross-prompt summary")
    print("=" * 60)
    for k in MARGIN_RANKS:
        key = f"top1_vs_top{k}"
        vals = [r["margins"][key] for r in all_records]
        avg = sum(vals) / n
        print(f"  avg {key:18s} = {avg:.4f}")
    print()
    for threshold, field in [(1.0, "within_1_nit"), (2.0, "within_2_nit"), (5.0, "within_5_nit")]:
        vals = [r["competing_tokens"][field] for r in all_records]
        avg = sum(vals) / n
        print(f"  avg tokens within {threshold:.1f} nit = {avg:.1f}")
    print("=" * 60)

    summary = {
        "n_prompts": n,
        "model": args.model,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "avg_margins": {
            f"top1_vs_top{k}": round(
                sum(r["margins"][f"top1_vs_top{k}"] for r in all_records) / n, 4
            )
            for k in MARGIN_RANKS
        },
        "avg_competing_tokens": {
            "within_1_nit": round(sum(r["competing_tokens"]["within_1_nit"] for r in all_records) / n, 1),
            "within_2_nit": round(sum(r["competing_tokens"]["within_2_nit"] for r in all_records) / n, 1),
            "within_5_nit": round(sum(r["competing_tokens"]["within_5_nit"] for r in all_records) / n, 1),
        },
    }

    output = {"summary": summary, "results": all_records}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
