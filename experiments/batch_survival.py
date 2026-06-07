"""
experiments/batch_survival.py  --  Batch-level DSD survival curve

目的:
  N_BATCH ユーザーを並列に流した場合、各タイル時点で何ユーザーが
  まだ生存しているかを観測する。

  alive_users[t] = stop_tile >= t であるユーザー数

  後半タイルで生存ユーザーが急減すれば、それ以降のタイル計算は
  ほとんど不要になる → Tile-level early stop の効果を定量評価できる。

アルゴリズム:
  ordering_exp_v2.py の bound_first をそのまま使用。
  停止条件 Δ > 2B・Top1保証も変更なし。

定義:
  hidden_dim = 3840  (Gemma 3 12B)
  chunk_size = 80
  n_tiles    = 48
  stop_tile  = ceil(stop_dim / chunk_size)  (1-indexed)
  alive_users[t] = sum(stop_tile >= t) for t in 1..n_tiles

出力:
  batch_survival.json
  batch_survival_curve.png
"""

import sys, os, json, math, time, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first
from dsd_poc import dsd_predict
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE = 80
N_BATCH    = 512

DEFAULT_PROMPTS = [
    "The capital of Japan is",
    "Python is a programming",
    "The largest planet in the solar system is",
    "Linux is an open source",
    "The opposite of hot is",
    "Water boils at 100 degrees",
    "The speed of light is approximately",
    "Machine learning is a subset of",
    "The first element in the periodic table is",
    "Shakespeare was born in",
    "World War II ended in",
    "DNA stands for",
    "The chemical symbol for gold is",
    "The internet was invented in",
    "A compiler converts",
    "The capital of France is",
    "The largest ocean in the world is",
    "The atomic number of carbon is",
    "HTML stands for",
    "The opposite of cold is",
]


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


def get_lm_head(model):
    lm = model.lm_head
    W    = lm.weight.detach().cpu().float()
    bias = lm.bias.detach().cpu().float() if lm.bias is not None else None
    return W, bias


def batch_hidden_states(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """
    全プロンプトを padding=True で一括トークナイズし、
    model を 1 回だけ forward して最終層 hidden state を返す。

    Returns:
        h_batch: (N, hidden_dim) float32 CPU tensor
                 各行が対応プロンプトの最終実トークン位置の hidden vector。
    """
    device = next(model.parameters()).device

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # out.hidden_states[-1]: (N, seq_len, hidden_dim)
    last_hidden = out.hidden_states[-1]   # (N, seq_len, hidden_dim)

    # padding あり → 各サンプルの最後の実トークン位置を取得
    seq_lens = attention_mask.sum(dim=1) - 1   # (N,) 0-indexed
    h_batch = last_hidden[
        torch.arange(last_hidden.size(0), device=device), seq_lens, :
    ]   # (N, hidden_dim)

    return h_batch.detach().cpu().float()


# ---------------------------------------------------------------------------
# Single-user DSD with bound_first ordering
# ---------------------------------------------------------------------------

def run_user(h: torch.Tensor, W: torch.Tensor, bias, chunk_size: int) -> dict:
    """
    bound_first 順序で DSD を実行し、stop_tile を返す。
    ordering_exp_v2.order_bound_first をそのまま使用。
    """
    H = h.shape[0]
    n_tiles = math.ceil(H / chunk_size)

    # bound_first 並び替え (ordering_exp_v2.py と同一関数)
    order   = order_bound_first(h, W)
    h_ord   = h[order]
    W_ord   = W[:, order]

    # dsd_predict (アルゴリズム・停止条件変更なし)
    result  = dsd_predict(h_ord, W_ord, bias)

    # stop_dim (1-indexed) → stop_tile (1-indexed)
    stop_dim  = result["stop_dim"]
    stop_tile = math.ceil(stop_dim / chunk_size)

    # dense top1 (一致確認)
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
    dense_id = int(logit.argmax())

    return {
        "stop_dim":   stop_dim,
        "stop_tile":  stop_tile,
        "n_tiles":    n_tiles,
        "skip_pct":   round((n_tiles - stop_tile) / n_tiles * 100, 2),
        "top1_match": result["token_id"] == dense_id,
        "dsd_token_id":   result["token_id"],
        "dense_token_id": dense_id,
    }


# ---------------------------------------------------------------------------
# Alive curve computation
# ---------------------------------------------------------------------------

def compute_alive_curve(records: list[dict], n_tiles: int) -> list[int]:
    """
    alive_users[t] = stop_tile >= t であるユーザー数  (t: 1-indexed)
    返り値は長さ n_tiles のリスト (index=0 が tile 1)。
    """
    stop_tiles = [r["stop_tile"] for r in records]
    alive = []
    for t in range(1, n_tiles + 1):
        alive.append(sum(1 for st in stop_tiles if st >= t))
    return alive


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_survival(alive: list[int], n_users: int, n_tiles: int, out_path: str):
    tiles = list(range(1, n_tiles + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Batch DSD Survival Curve  (N={n_users}, bound_first ordering)",
                 fontsize=13, fontweight="bold")

    # --- 絶対人数 ---
    ax = axes[0]
    ax.fill_between(tiles, alive, alpha=0.25, color="#4C72B0")
    ax.plot(tiles, alive, color="#4C72B0", lw=2)
    ax.set_xlabel("Tile index")
    ax.set_ylabel("Alive users")
    ax.set_title("Alive users per tile  (absolute)")
    ax.set_xlim(1, n_tiles)
    ax.set_ylim(0, n_users * 1.05)
    ax.grid(alpha=0.3)

    # 50%・10%・1% ラインを annotate
    for pct, col in [(0.5, "#DD8452"), (0.1, "#C44E52"), (0.01, "#8C8C8C")]:
        thresh = n_users * pct
        crosses = [t for t, a in zip(tiles, alive) if a <= thresh]
        if crosses:
            t_cross = crosses[0]
            ax.axvline(t_cross, color=col, ls="--", lw=1)
            ax.text(t_cross + 0.3, thresh + n_users * 0.02,
                    f"{int(pct*100)}% @ tile {t_cross}", fontsize=8, color=col)

    # --- 割合 (%) ---
    alive_pct = [a / n_users * 100 for a in alive]
    ax2 = axes[1]
    ax2.fill_between(tiles, alive_pct, alpha=0.25, color="#55A868")
    ax2.plot(tiles, alive_pct, color="#55A868", lw=2)
    ax2.axhline(50, color="#DD8452", ls="--", lw=1, label="50%")
    ax2.axhline(10, color="#C44E52", ls="--", lw=1, label="10%")
    ax2.set_xlabel("Tile index")
    ax2.set_ylabel("Alive users (%)")
    ax2.set_title("Alive users per tile  (percentage)")
    ax2.set_xlim(1, n_tiles)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch DSD survival curve")
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--n_batch",    type=int, default=N_BATCH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output",     default="batch_survival.json")
    parser.add_argument("--plot",       default="batch_survival_curve.png")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    n_batch    = args.n_batch

    # プロンプトを繰り返して n_batch 件に
    base = DEFAULT_PROMPTS
    prompts = [base[i % len(base)] for i in range(n_batch)]

    tokenizer, model = load_gemma(args.model)

    cfg = model.config
    if hasattr(cfg, 'text_config'):
        hidden_dim = cfg.text_config.hidden_size
        vocab_size = cfg.text_config.vocab_size
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_tiles = math.ceil(hidden_dim / chunk_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model info]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  "
          f"params={n_params/1e9:.2f}B")
    print(f"[batch]  n_users={n_batch}  chunk_size={chunk_size}  n_tiles={n_tiles}\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)\n")

    # --- バッチ forward: 512 ユーザーを 1 回の model() で処理 ---
    print(f"[forward] {n_batch} ユーザーを一括 forward 中...")
    t_fwd = time.time()
    h_batch = batch_hidden_states(model, tokenizer, prompts)   # (N, hidden_dim)
    fwd_elapsed = time.time() - t_fwd
    print(f"  h_batch.shape={tuple(h_batch.shape)}  ({fwd_elapsed:.1f}s)\n")

    # --- DSD: ユーザーごとに hidden を取り出して逐次処理 ---
    print("[dsd] ユーザーごとに DSD を実行中...")
    records = []
    t_start = time.time()
    for i in range(n_batch):
        h = h_batch[i]   # (hidden_dim,)

        rec = run_user(h, W, bias, chunk_size)
        rec["prompt"] = prompts[i]
        records.append(rec)

        if (i + 1) % max(1, n_batch // 20) == 0 or (i + 1) == n_batch:
            elapsed = time.time() - t_start
            n_match = sum(r["top1_match"] for r in records)
            avg_skip = sum(r["skip_pct"] for r in records) / len(records)
            print(f"  [{i+1:4d}/{n_batch}]  "
                  f"match={n_match}/{i+1}  "
                  f"avg_skip={avg_skip:.1f}%  "
                  f"dsd_elapsed={elapsed:.1f}s")

    elapsed_total = fwd_elapsed + (time.time() - t_start)

    # --- 統計 ---
    n_match   = sum(r["top1_match"] for r in records)
    stop_tiles = [r["stop_tile"] for r in records]
    skip_pcts  = [r["skip_pct"]   for r in records]

    stop_tiles_sorted = sorted(stop_tiles)
    avg_stop   = sum(stop_tiles) / n_batch
    med_stop   = stop_tiles_sorted[n_batch // 2]
    avg_skip   = sum(skip_pcts) / n_batch

    # alive curve
    alive = compute_alive_curve(records, n_tiles)

    # --- コンソール表示 ---
    print(f"\n{'='*55}")
    print("Batch DSD Survival Curve")
    print(f"{'='*55}")
    print(f"  n_users        = {n_batch}")
    print(f"  n_tiles        = {n_tiles}  (chunk_size={chunk_size})")
    print(f"  avg stop_tile  = {avg_stop:.1f}")
    print(f"  median stop_tile = {med_stop}")
    print(f"  avg skip %     = {avg_skip:.1f}%")
    print(f"  Top1 match     = {n_match}/{n_batch}  ({n_match/n_batch*100:.1f}%)")
    print(f"  elapsed        = {elapsed_total:.1f}s\n")

    print(f"  {'Tile':>6}  {'Alive':>8}  {'%':>6}  bar")
    print("  " + "-" * 50)
    for t, a in enumerate(alive, start=1):
        pct = a / n_batch * 100
        bar = "█" * int(pct / 2)
        print(f"  Tile{t:2d}  {a:>8}  {pct:5.1f}%  {bar}")
    print(f"{'='*55}\n")

    # --- グラフ ---
    plot_survival(alive, n_batch, n_tiles, args.plot)

    # --- JSON 保存 ---
    output = {
        "model":      args.model,
        "n_users":    n_batch,
        "n_tiles":    n_tiles,
        "chunk_size": chunk_size,
        "hidden_dim": hidden_dim,
        "ordering":   "bound_first",
        "statistics": {
            "avg_stop_tile":    round(avg_stop, 2),
            "median_stop_tile": med_stop,
            "avg_skip_pct":     round(avg_skip, 2),
            "top1_match_rate":  n_match / n_batch,
            "elapsed_s":        round(elapsed_total, 2),
        },
        "alive_curve": [
            {"tile": t, "alive": a, "alive_pct": round(a / n_batch * 100, 2)}
            for t, a in enumerate(alive, start=1)
        ],
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
