"""
experiments/gemma_eval.py

Gemma 3 12B の実 hidden state で DSD を検証し、
信頼度ダイナミクス (trajectory) を観測するスクリプト。

出力:
  results.json      -- 各プロンプトの要約メトリクス + サマリ
  trajectory.json   -- 各プロンプトのチェックポイント列
                       [{dim, delta, bound, ratio}, ...]

アルゴリズムは dsd_poc.py から変更しない。
"""

import sys
import os
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsd_poc import dsd_predict  # 停止条件・アルゴリズム変更なし

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "The capital of Japan is",
    "Python is a",
    "The largest planet in the solar system is",
    "Linux is",
    "The opposite of hot is",
]

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"

# トラジェクトリのチェックポイント間隔 (次元数)
# hidden_dim=3840 に対して 128 刻みで 30 点 + 停止点
TRAJECTORY_INTERVAL = 128


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_gemma(model_name: str = MODEL_NAME):
    print(f"[load] {model_name}")
    print("  torch_dtype=bfloat16  device_map=auto  local_files_only=True")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("[load] 完了\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# hidden state / lm_head
# ---------------------------------------------------------------------------

def get_final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu()


def get_lm_head(model):
    lm_head = model.lm_head
    W = lm_head.weight.detach().float().cpu()
    bias = lm_head.bias.detach().float().cpu() if lm_head.bias is not None else None
    return W, bias


# ---------------------------------------------------------------------------
# Dense greedy (参照)
# ---------------------------------------------------------------------------

def dense_predict(h, W, bias=None) -> int:
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
    return int(logit.argmax())


# ---------------------------------------------------------------------------
# 1 プロンプト評価
# ---------------------------------------------------------------------------

def eval_one(prompt, tokenizer, model, W, bias, interval: int = TRAJECTORY_INTERVAL) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    t0 = time.time()
    h = get_final_hidden(model, input_ids)
    forward_time = time.time() - t0

    dense_id = dense_predict(h, W, bias)
    dense_token = tokenizer.decode([dense_id])

    t1 = time.time()
    result = dsd_predict(h, W, bias, trajectory_interval=interval)
    dsd_time = time.time() - t1

    dsd_token = tokenizer.decode([result["token_id"]])
    match = result["token_id"] == dense_id

    print(f"Prompt : {prompt!r}")
    print(f"  Dense        : {dense_token!r}  (id={dense_id})")
    print(f"  DSD          : {dsd_token!r}  (id={result['token_id']})")
    print(f"  match        = {match}")
    print(f"  stop_dim     = {result['stop_dim']} / {result['hidden_dim']}")
    print(f"  skip_rate    = {result['skip_rate']*100:.1f}%")
    print(f"  max_delta    = {result['max_delta']:.4f}")
    print(f"  final_delta  = {result['final_delta']:.4f}")
    print(f"  stop_ratio   = {result['stopping_ratio']}")
    print(f"  traj points  = {len(result.get('trajectory', []))}")
    print(f"  forward      = {forward_time:.2f}s  dsd_scan = {dsd_time:.2f}s")
    print()

    return {
        "prompt": prompt,
        "dense_token": dense_token,
        "dsd_token": dsd_token,
        "dense_token_id": dense_id,
        "dsd_token_id": result["token_id"],
        "match": match,
        "stop_dim": result["stop_dim"],
        "hidden_dim": result["hidden_dim"],
        "skip_rate": result["skip_rate"],
        "max_delta": result["max_delta"],
        "final_delta": result["final_delta"],
        "stopping_ratio": result["stopping_ratio"],
        "trajectory": result.get("trajectory", []),
        "forward_time_s": round(forward_time, 3),
        "dsd_scan_time_s": round(dsd_time, 3),
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma 3 12B DSD evaluation + trajectory")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--trajectory_output", default="trajectory.json")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument(
        "--interval", type=int, default=TRAJECTORY_INTERVAL,
        help="トラジェクトリ記録の次元間隔 (default: 128)",
    )
    args = parser.parse_args()

    prompts = args.prompts or EVAL_PROMPTS

    # --- ロード ---
    tokenizer, model = load_gemma(args.model)

    # --- モデル情報 ---
    cfg = model.config
    # Gemma3 は multi-modal なので text_config を参照
    if hasattr(cfg, 'text_config'):
        hidden_dim = cfg.text_config['hidden_size']
        vocab_size = cfg.text_config['vocab_size']
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print("[model info]")
    print(f"  hidden_dim  = {hidden_dim}")
    print(f"  vocab_size  = {vocab_size}")
    print(f"  parameters  = {n_params/1e9:.2f}B")
    print()

    # --- lm_head を CPU float32 に変換 ---
    print("[prep] lm_head を CPU float32 に変換中...")
    t_prep = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape = {W.shape}  dtype={W.dtype}  ({time.time()-t_prep:.1f}s)\n")

    # --- 評価ループ ---
    interval = args.interval

    all_results = []
    for prompt in prompts:
        rec = eval_one(prompt, tokenizer, model, W, bias, interval)
        all_results.append(rec)

    # --- サマリ計算 ---
    n = len(all_results)
    n_match = sum(r["match"] for r in all_results)
    skip_rates = [r["skip_rate"] for r in all_results]
    stop_dims = [r["stop_dim"] for r in all_results]

    summary = {
        "model": args.model,
        "n_prompts": n,
        "top1_match_rate": n_match / n,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "avg_stop_dim": sum(stop_dims) / n,
        "avg_skip_rate": sum(skip_rates) / n,
        "best_skip_rate": max(skip_rates),
        "worst_skip_rate": min(skip_rates),
        "best_skip_prompt": all_results[skip_rates.index(max(skip_rates))]["prompt"],
        "worst_skip_prompt": all_results[skip_rates.index(min(skip_rates))]["prompt"],
    }

    print("=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"  prompts        : {n}")
    print(f"  top1 match     : {n_match}/{n}  ({n_match/n*100:.1f}%)")
    print(f"  avg stop_dim   : {summary['avg_stop_dim']:.1f} / {hidden_dim}")
    print(f"  avg skip_rate  : {summary['avg_skip_rate']*100:.1f}%")
    print(f"  best skip_rate : {summary['best_skip_rate']*100:.1f}%  ({summary['best_skip_prompt']!r})")
    print(f"  worst skip_rate: {summary['worst_skip_rate']*100:.1f}%  ({summary['worst_skip_prompt']!r})")
    print("=" * 55)

    # --- results.json ---
    results_out = {
        "summary": summary,
        "results": [
            {k: v for k, v in r.items() if k != "trajectory"}
            for r in all_results
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.output}")

    # --- trajectory.json ---
    traj_out = [
        {
            "prompt": r["prompt"],
            "stop_dim": r["stop_dim"],
            "hidden_dim": r["hidden_dim"],
            "skip_rate": r["skip_rate"],
            "max_delta": r["max_delta"],
            "final_delta": r["final_delta"],
            "stopping_ratio": r["stopping_ratio"],
            "trajectory": r["trajectory"],
        }
        for r in all_results
    ]
    with open(args.trajectory_output, "w", encoding="utf-8") as f:
        json.dump(traj_out, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.trajectory_output}")


if __name__ == "__main__":
    main()
