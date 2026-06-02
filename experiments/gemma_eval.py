"""
experiments/gemma_eval.py

Gemma 3 12B の実 hidden state で DSD を検証するスクリプト。

処理:
  1. google/gemma-3-12b-it を bfloat16 / device_map="auto" でロード
  2. 各プロンプトの最終トークン hidden state を取得
  3. 既存の dsd_predict() をそのまま使って早期停止トークンを確定
  4. Dense (W @ h) の greedy top1 と照合
  5. results.json に出力

アルゴリズムは dsd_poc.py から変更しない。
"""

import sys
import os
import json
import time
import torch

# リポジトリルートを sys.path に追加して dsd_poc をインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsd_poc import dsd_predict  # アルゴリズムはそのまま使う

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# 評価プロンプト
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "The capital of Japan is",
    "Python is a",
    "The largest planet in the solar system is",
    "Linux is",
    "The opposite of hot is",
]

MODEL_NAME = "google/gemma-3-12b-it"


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_gemma(model_name: str = MODEL_NAME):
    print(f"[load] {model_name}")
    print("  torch_dtype=bfloat16, device_map=auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # デバイス配置を表示
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"  device_map: {model.hf_device_map}")
    else:
        devices = {next(model.parameters()).device}
    print(f"  devices: {devices}")
    print("[load] 完了\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# hidden state & lm_head の取得
# ---------------------------------------------------------------------------

def get_final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """最終層の hidden vector を float32 で返す (最後のトークン位置)。"""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    # out.hidden_states[-1]: (batch, seq_len, hidden_dim)
    h = out.hidden_states[-1][0, -1, :].float().cpu()
    return h


def get_lm_head(model):
    """lm_head の weight と bias を float32 CPU tensor で返す。"""
    lm_head = model.lm_head
    W = lm_head.weight.detach().float().cpu()   # (vocab_size, hidden_dim)
    bias = (
        lm_head.bias.detach().float().cpu()
        if lm_head.bias is not None
        else None
    )
    return W, bias


# ---------------------------------------------------------------------------
# Dense greedy 予測 (比較用)
# ---------------------------------------------------------------------------

def dense_predict(h: torch.Tensor, W: torch.Tensor, bias=None) -> int:
    """W @ h による通常の greedy argmax。参照値として使う。"""
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
    return int(logit.argmax())


# ---------------------------------------------------------------------------
# 1 プロンプト評価
# ---------------------------------------------------------------------------

def eval_one(
    prompt: str,
    tokenizer,
    model,
    W: torch.Tensor,
    bias,
) -> dict:
    # トークナイズ & forward
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    t0 = time.time()
    h = get_final_hidden(model, input_ids)
    forward_time = time.time() - t0

    # Dense top1 (参照)
    dense_id = dense_predict(h, W, bias)
    dense_token = tokenizer.decode([dense_id])

    # DSD top1 (停止アルゴリズム)
    t1 = time.time()
    result = dsd_predict(h, W, bias)
    dsd_time = time.time() - t1

    dsd_token = tokenizer.decode([result["token_id"]])
    match = result["token_id"] == dense_id

    # ログ表示
    print(f"Prompt : {prompt!r}")
    print(f"  Dense : {dense_token!r}  (token_id={dense_id})")
    print(f"  DSD   : {dsd_token!r}  (token_id={result['token_id']})")
    print(f"  match      = {match}")
    print(f"  stop_dim   = {result['stop_dim']}")
    print(f"  hidden_dim = {result['hidden_dim']}")
    print(f"  skip_rate  = {result['skip_rate'] * 100:.1f}%")
    print(f"  forward    = {forward_time:.2f}s  dsd_scan = {dsd_time:.2f}s")
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
        "forward_time_s": round(forward_time, 3),
        "dsd_scan_time_s": round(dsd_time, 3),
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma 3 12B DSD evaluation")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--output", default="results.json",
        help="結果 JSON の出力パス (デフォルト: results.json)",
    )
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="評価プロンプトを上書き",
    )
    args = parser.parse_args()

    prompts = args.prompts or EVAL_PROMPTS

    # --- ロード ---
    tokenizer, model = load_gemma(args.model)

    # lm_head は一度だけ CPU float32 に変換して使い回す
    print("[prep] lm_head を CPU float32 に変換中...")
    t_prep = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape = {W.shape}  dtype={W.dtype}  ({time.time()-t_prep:.1f}s)\n")

    # --- 評価ループ ---
    all_results = []
    for prompt in prompts:
        rec = eval_one(prompt, tokenizer, model, W, bias)
        all_results.append(rec)

    # --- サマリ ---
    n = len(all_results)
    n_match = sum(r["match"] for r in all_results)
    avg_skip = sum(r["skip_rate"] for r in all_results) / n
    avg_stop = sum(r["stop_dim"] for r in all_results) / n
    hidden_dim = all_results[0]["hidden_dim"]

    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  prompts        : {n}")
    print(f"  top1 match     : {n_match}/{n}  ({n_match/n*100:.1f}%)")
    print(f"  avg stop_dim   : {avg_stop:.1f} / {hidden_dim}")
    print(f"  avg skip_rate  : {avg_skip*100:.1f}%")
    print("=" * 50)

    # --- JSON 出力 ---
    output = {
        "model": args.model,
        "summary": {
            "n_prompts": n,
            "top1_match_rate": n_match / n,
            "avg_stop_dim": avg_stop,
            "hidden_dim": hidden_dim,
            "avg_skip_rate": avg_skip,
        },
        "results": all_results,
    }

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
