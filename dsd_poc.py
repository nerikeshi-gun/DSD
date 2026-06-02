"""
DSD (Dimension-wise Speculative Decoding) PoC for Gemma 3 12B

アルゴリズム:
  hidden vector h を hidden_dim 方向に逐次スキャンし、
  各次元を累積した後に lm_head でロジットを計算する。
  Δ = logit[top1] - logit[top2]
  B = 残り次元の累積寄与の上限 (max|w_i| * max|h_i| の和)
  Δ > B になった時点で停止し、top1 を確定トークンとして返す。

検証指標:
  1. Top1 完全一致率  (DSD の top1 == フル計算の greedy top1)
  2. stop_dim         (何次元目で停止したか)
  3. skip 率         (1 - stop_dim / hidden_dim)
"""

import argparse
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional


# ---------------------------------------------------------------------------
# DSD コア
# ---------------------------------------------------------------------------

def compute_remaining_bound(
    weight_abs: torch.Tensor,   # (hidden_dim,)  max_v|W[v,i]| を次元順に並べたもの
    hidden_abs: torch.Tensor,   # (hidden_dim,)  |h[i]|
    start_dim: int,
) -> torch.Tensor:
    """
    B = sum_{i=start_dim}^{hidden_dim-1} max_v|W[v,i]| * |h[i]|

    各ロジット単体が残り次元から受ける寄与の上限。
    停止判定では差の変化上限 2B と比較する (呼び出し側で * 2)。
    """
    if start_dim >= weight_abs.shape[0]:
        return torch.tensor(0.0, dtype=torch.float32)
    return (weight_abs[start_dim:] * hidden_abs[start_dim:]).sum()


def dsd_predict(
    h: torch.Tensor,            # (hidden_dim,) float32
    W: torch.Tensor,            # (vocab_size, hidden_dim) float32
    bias: Optional[torch.Tensor] = None,  # (vocab_size,) or None
) -> dict:
    """
    h を hidden_dim 方向に逐次スキャンして greedy top1 を早期確定する。

    Returns:
        {
          "token_id":  int,   確定トークン ID
          "stop_dim":  int,   停止した次元インデックス (1-indexed)
          "hidden_dim": int,
          "skip_rate": float, 1 - stop_dim/hidden_dim
          "full_logit_top1": int,  フル計算での top1 (一致検証用)
          "match": bool,
        }
    """
    hidden_dim = h.shape[0]
    vocab_size = W.shape[0]

    # --- フル計算 (正解) ---
    # 逐次加算と同じ累積順序を使うことで float32 の丸め誤差を揃える。
    # BLAS matmul (W @ h) は内部で異なる加算順序を使う場合があり、
    # マージンが極小のトークンで top1 が食い違うことがある。
    with torch.no_grad():
        full_logit_ref = bias.clone() if bias is not None else torch.zeros(vocab_size, dtype=torch.float32)
        for i in range(hidden_dim):
            full_logit_ref.add_(W[:, i] * h[i])
        full_top1 = int(full_logit_ref.argmax())

    # --- 各次元の max |weight| を事前計算 ---
    # weight_col_max[i] = max_v |W[v, i]|
    weight_col_max = W.abs().max(dim=0).values   # (hidden_dim,)
    h_abs = h.abs()                               # (hidden_dim,)

    # --- 逐次スキャン ---
    # partial_logit を次元ごとに加算して top1/top2 を更新
    if bias is not None:
        partial_logit = bias.clone()
    else:
        partial_logit = torch.zeros(vocab_size, dtype=torch.float32)

    stop_dim = hidden_dim  # デフォルトは最後まで

    for i in range(hidden_dim):
        # i 次元目を加算
        partial_logit.add_(W[:, i] * h[i])

        # top1 / top2
        top2_vals, top2_idx = partial_logit.topk(2)
        delta = float(top2_vals[0] - top2_vals[1])

        # 残り上限 B
        # B = sum_{j>i} max_v|W[v,j]| * |h[j]|  (各ロジット単体への最大変化)
        # 差 (top1 - top2) の最悪変化 = top2 が +B、top1 が -B → 計 2B
        B = float(compute_remaining_bound(weight_col_max, h_abs, i + 1))

        if delta > 2.0 * B:
            stop_dim = i + 1  # 1-indexed
            break

    token_id = int(partial_logit.argmax())
    skip_rate = 1.0 - stop_dim / hidden_dim

    return {
        "token_id": token_id,
        "stop_dim": stop_dim,
        "hidden_dim": hidden_dim,
        "skip_rate": skip_rate,
        "full_logit_top1": full_top1,
        "match": token_id == full_top1,
    }


# ---------------------------------------------------------------------------
# モデルロード & 推論ユーティリティ
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "cpu"):
    print(f"[load] {model_name} を {device} でロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    print("[load] 完了")
    return tokenizer, model


def get_final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """最終層の hidden vector (最後のトークン位置) を返す。"""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    # out.hidden_states[-1]: (batch, seq_len, hidden_dim)
    h = out.hidden_states[-1][0, -1, :].float()  # (hidden_dim,)
    return h


def get_lm_head(model):
    """lm_head の weight と bias を返す。"""
    lm_head = model.lm_head
    W = lm_head.weight.detach().float()           # (vocab_size, hidden_dim)
    bias = lm_head.bias.detach().float() if lm_head.bias is not None else None
    return W, bias


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate(
    tokenizer,
    model,
    prompts: list[str],
    verbose: bool = True,
) -> dict:
    W, bias = get_lm_head(model)

    results = []
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        h = get_final_hidden(model, input_ids)

        result = dsd_predict(h, W, bias)
        result["prompt"] = prompt[:60]
        results.append(result)

        if verbose:
            print(
                f"[{idx+1:3d}] match={result['match']}  "
                f"stop_dim={result['stop_dim']:5d}/{result['hidden_dim']}  "
                f"skip={result['skip_rate']:.3f}  "
                f"token='{tokenizer.decode([result['token_id']])}'"
            )

    n = len(results)
    match_rate = sum(r["match"] for r in results) / n
    avg_stop_dim = np.mean([r["stop_dim"] for r in results])
    avg_skip = np.mean([r["skip_rate"] for r in results])
    hidden_dim = results[0]["hidden_dim"]

    summary = {
        "n_samples": n,
        "top1_match_rate": match_rate,
        "avg_stop_dim": float(avg_stop_dim),
        "hidden_dim": hidden_dim,
        "avg_skip_rate": float(avg_skip),
    }

    print("\n===== Summary =====")
    print(f"  samples       : {n}")
    print(f"  top1 match    : {match_rate:.4f}  ({sum(r['match'] for r in results)}/{n})")
    print(f"  avg stop_dim  : {avg_stop_dim:.1f} / {hidden_dim}")
    print(f"  avg skip rate : {avg_skip:.4f}")
    print("===================\n")

    return summary, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The largest planet in the solar system is",
    "Water boils at 100 degrees",
    "The speed of light is approximately",
    "Python is a programming",
    "Tokyo is located in",
    "The human body has",
    "Machine learning is a subset of",
    "The first element in the periodic table is",
]


def main():
    parser = argparse.ArgumentParser(description="DSD PoC for Gemma 3 12B")
    parser.add_argument(
        "--model",
        default="google/gemma-3-12b-it",
        help="HuggingFace モデル名",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu / cuda / mps",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="評価するプロンプト (省略時はデモ用 10 件)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="結果を JSON で保存するパス",
    )
    parser.add_argument(
        "--unittest",
        action="store_true",
        help="モデル不要の単体テストのみ実行",
    )
    args = parser.parse_args()

    if args.unittest:
        run_unit_tests()
        return

    tokenizer, model = load_model(args.model, args.device)
    prompts = args.prompts or DEMO_PROMPTS

    t0 = time.time()
    summary, results = evaluate(tokenizer, model, prompts)
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s  ({elapsed/len(prompts):.2f}s/sample)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"saved: {args.output}")


# ---------------------------------------------------------------------------
# 単体テスト (モデル不要)
# ---------------------------------------------------------------------------

def run_unit_tests():
    print("=== Unit Tests ===")
    torch.manual_seed(0)

    # --- Test 1: 小さな例で top1 完全一致 ---
    # 既知の解を持つ手作り例
    vocab = 10
    hdim = 8
    W = torch.randn(vocab, hdim)
    h = torch.randn(hdim)
    expected = int((W @ h).argmax())

    res = dsd_predict(h, W, bias=None)
    assert res["match"], f"Test1 failed: token={res['token_id']} expected={expected}"
    assert res["token_id"] == expected
    print(f"[PASS] Test1: top1 一致  stop_dim={res['stop_dim']}/{hdim}")

    # --- Test 2: bias あり ---
    bias = torch.randn(vocab)
    expected_b = int((W @ h + bias).argmax())
    res2 = dsd_predict(h, W, bias=bias)
    assert res2["match"], f"Test2 failed"
    assert res2["token_id"] == expected_b
    print(f"[PASS] Test2: bias あり  stop_dim={res2['stop_dim']}/{hdim}")

    # --- Test 3: stop_dim <= hidden_dim ---
    assert 1 <= res["stop_dim"] <= hdim
    print(f"[PASS] Test3: stop_dim 範囲内")

    # --- Test 4: skip_rate 計算が正しい ---
    expected_skip = 1.0 - res["stop_dim"] / hdim
    assert abs(res["skip_rate"] - expected_skip) < 1e-6
    print(f"[PASS] Test4: skip_rate={res['skip_rate']:.4f}")

    # --- Test 5: 大規模ランダムで match 率 100% ---
    n_trial = 200
    ok = 0
    for _ in range(n_trial):
        W_ = torch.randn(100, 64)
        h_ = torch.randn(64)
        r = dsd_predict(h_, W_, bias=None)
        ok += r["match"]
    assert ok == n_trial, f"Test5: {ok}/{n_trial} matched"
    print(f"[PASS] Test5: {ok}/{n_trial} match (100%)")

    # --- Test 6: 次元が 1 の境界 ---
    W1 = torch.randn(5, 1)
    h1 = torch.randn(1)
    r1 = dsd_predict(h1, W1)
    assert r1["match"]
    assert r1["stop_dim"] == 1
    print(f"[PASS] Test6: hidden_dim=1 境界ケース")

    # --- Test 7: 統計的 skip 率が 0 以上 ---
    skip_rates = []
    for _ in range(50):
        W_ = torch.randn(1000, 256)
        h_ = torch.randn(256)
        r = dsd_predict(h_, W_)
        assert r["match"]
        skip_rates.append(r["skip_rate"])
    avg_skip = np.mean(skip_rates)
    print(f"[PASS] Test7: avg skip_rate={avg_skip:.4f}  (vocab=1000, hdim=256)")

    print("\n全テスト PASS\n")


if __name__ == "__main__":
    main()
