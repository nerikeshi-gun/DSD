"""
experiments/dsd_runtime.py  --  DSD ランタイム実装

lm_head の W @ h をタイル単位で逐次実行し、Δ > 2B に達した時点で
ループを break する。break 以降のタイルは計算しない。

これは「停止位置の事後分析」ではない:
  - DSD ループが break した後、W の残タイルには一切アクセスしない
  - partial_logit の argmax がそのままランタイム出力 (top1 トークン)
  - 正解検証 (dense W @ h) は計測ループとは別パスで実施する

比較構造:
  [ランタイムパス]  model() → h → DSD tile loop (break あり) → token_id
  [検証パス]       token_id vs (W @ h).argmax()  ← 計測時間から除外

出力:
  dsd_runtime.json
  dsd_runtime_curve.png
"""

import sys, os, json, math, time, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME  = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE  = 80
N_BATCH     = 512

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
    lm   = model.lm_head
    W    = lm.weight.detach().cpu().float()
    bias = lm.bias.detach().cpu().float() if lm.bias is not None else None
    return W, bias


def batch_hidden_states(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    last_hidden = out.hidden_states[-1]
    seq_lens    = attention_mask.sum(dim=1) - 1
    h_batch = last_hidden[torch.arange(last_hidden.size(0), device=device), seq_lens, :]
    return h_batch.detach().cpu().float()


# ---------------------------------------------------------------------------
# DSD ランタイム: lm_head タイル単位実行 + Δ > 2B で break
# ---------------------------------------------------------------------------

def dsd_runtime_single(
    h: torch.Tensor,
    W: torch.Tensor,
    bias,
    chunk_size: int,
    suffix_B: torch.Tensor,     # 事前計算済 suffix bound (並び替え後)
    h_ord: torch.Tensor,        # h[order]
    W_ord: torch.Tensor,        # W[:, order]
    n_tiles: int,
) -> tuple[int, int, float]:
    """
    lm_head をタイル単位で実行し Δ > 2B で break する。

    break した瞬間に後続タイルへのアクセスは行わない。
    dense W @ h は呼び出し側の検証パスで別途計算する。

    Returns: (token_id, stop_tile, skip_pct)
    """
    H          = h.shape[0]
    vocab_size = W.shape[0]

    partial_logit = bias.clone() if bias is not None else torch.zeros(vocab_size)
    stop_tile     = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end   = min(tile * chunk_size, H)

        # このタイルの次元だけを読む
        for i in range(dim_start, dim_end):
            partial_logit.add_(W_ord[:, i] * h_ord[i])

        dim_done = dim_end
        top2     = partial_logit.topk(2)
        delta    = float(top2.values[0] - top2.values[1])
        B        = float(suffix_B[dim_done]) if dim_done < H else 0.0

        if delta > 2.0 * B:
            stop_tile = tile
            break   # 以降のタイルは計算しない

    token_id = int(partial_logit.argmax())
    skip_pct = round((n_tiles - stop_tile) / n_tiles * 100, 2)
    return token_id, stop_tile, skip_pct


# ---------------------------------------------------------------------------
# 事前計算: order, h_ord, W_ord, suffix_B を N ユーザー分まとめて準備
# ---------------------------------------------------------------------------

def precompute_ordering(h_batch: torch.Tensor, W: torch.Tensor):
    """
    bound_first 並び替えと suffix_B を全ユーザー分計算する。
    ランタイム計測の前に済ませておくことで計測対象から除外する。
    (実機では W の並び替えは静的スケジューラが事前に決定する)
    """
    N = h_batch.shape[0]
    prepared = []
    for i in range(N):
        h     = h_batch[i]
        order = order_bound_first(h, W)
        h_ord = h[order]
        W_ord = W[:, order]

        weight_col_max = W_ord.abs().max(dim=0).values
        h_abs          = h_ord.abs()
        suffix_B       = (weight_col_max * h_abs).flip(0).cumsum(0).flip(0)

        prepared.append((h_ord, W_ord, suffix_B))
    return prepared


# ---------------------------------------------------------------------------
# Alive curve
# ---------------------------------------------------------------------------

def compute_alive_curve(stop_tiles: list[int], n_tiles: int) -> list[int]:
    return [sum(1 for st in stop_tiles if st >= t) for t in range(1, n_tiles + 1)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_runtime(alive: list[int], speedups: list[float],
                 n_users: int, n_tiles: int, out_path: str):
    tiles = list(range(1, n_tiles + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"DSD Runtime  (N={n_users}, bound_first, lm_head tile-by-tile early stop)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.fill_between(tiles, alive, alpha=0.25, color="#4C72B0")
    ax.plot(tiles, alive, color="#4C72B0", lw=2)
    ax.set_xlabel("Tile index")
    ax.set_ylabel("Alive users")
    ax.set_title("Alive users per tile  (absolute)")
    ax.set_xlim(1, n_tiles)
    ax.set_ylim(0, n_users * 1.05)
    ax.grid(alpha=0.3)
    for pct, col in [(0.5, "#DD8452"), (0.1, "#C44E52")]:
        thresh  = n_users * pct
        crosses = [t for t, a in zip(tiles, alive) if a <= thresh]
        if crosses:
            t_cross = crosses[0]
            ax.axvline(t_cross, color=col, ls="--", lw=1)
            ax.text(t_cross + 0.2, thresh + n_users * 0.02,
                    f"{int(pct*100)}%@{t_cross}", fontsize=7, color=col)

    ax2 = axes[1]
    alive_pct = [a / n_users * 100 for a in alive]
    ax2.fill_between(tiles, alive_pct, alpha=0.25, color="#55A868")
    ax2.plot(tiles, alive_pct, color="#55A868", lw=2)
    ax2.axhline(50, color="#DD8452", ls="--", lw=0.8, label="50%")
    ax2.axhline(10, color="#C44E52", ls="--", lw=0.8, label="10%")
    ax2.set_xlabel("Tile index")
    ax2.set_ylabel("Alive users (%)")
    ax2.set_title("Alive users per tile  (percentage)")
    ax2.set_xlim(1, n_tiles)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.hist(speedups, bins=30, color="#4C72B0", alpha=0.7, edgecolor="white")
    mean_sp = sum(speedups) / len(speedups)
    ax3.axvline(mean_sp, color="#DD8452", ls="--", lw=1.5,
                label=f"mean={mean_sp:.2f}x")
    ax3.set_xlabel("Speedup (dense / dsd)")
    ax3.set_ylabel("Count")
    ax3.set_title("lm_head speedup  (dense wall time / dsd wall time)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSD runtime: lm_head tile-by-tile early stop")
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--n_batch",    type=int, default=N_BATCH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output",     default="dsd_runtime.json")
    parser.add_argument("--plot",       default="dsd_runtime_curve.png")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    n_batch    = args.n_batch

    base    = DEFAULT_PROMPTS
    prompts = [base[i % len(base)] for i in range(n_batch)]

    tokenizer, model = load_gemma(args.model)

    cfg = model.config
    if hasattr(cfg, "text_config"):
        hidden_dim = cfg.text_config.hidden_size
        vocab_size = cfg.text_config.vocab_size
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_tiles  = math.ceil(hidden_dim / chunk_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  "
          f"params={n_params/1e9:.2f}B")
    print(f"[batch]  n_users={n_batch}  chunk_size={chunk_size}  n_tiles={n_tiles}\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)\n")

    # --- Step 1: transformer forward (h を得るために全実行; DSD 対象外) ---
    print(f"[step1/forward] {n_batch} ユーザーを一括 forward 中...")
    t_fwd = time.time()
    h_batch = batch_hidden_states(model, tokenizer, prompts)
    fwd_elapsed = time.time() - t_fwd
    print(f"  h_batch.shape={tuple(h_batch.shape)}  ({fwd_elapsed:.1f}s)\n")

    # --- Step 2: 事前計算 (並び替え + suffix_B; 計測対象外) ---
    print("[step2/precompute] bound_first 並び替えと suffix_B を計算中...")
    t_pre = time.time()
    prepared = precompute_ordering(h_batch, W)
    print(f"  ({time.time()-t_pre:.1f}s)\n")

    # --- Step 3: DSD ランタイム計測 ---
    # lm_head を tile-by-tile で実行し Δ > 2B で break する。
    # dense W @ h はここでは計算しない。
    print("[step3/dsd-runtime] lm_head tile-by-tile early stop ...")
    print("  Δ > 2B に達した時点で break; 後続タイルは計算しない\n")

    dsd_token_ids = []
    stop_tiles    = []
    skip_pcts     = []
    dsd_times     = []

    t_loop = time.time()
    for i in range(n_batch):
        h_ord, W_ord, suffix_B = prepared[i]

        t0 = time.perf_counter()
        token_id, stop_tile, skip_pct = dsd_runtime_single(
            h_batch[i], W, bias, chunk_size,
            suffix_B, h_ord, W_ord, n_tiles,
        )
        elapsed_i = time.perf_counter() - t0

        dsd_token_ids.append(token_id)
        stop_tiles.append(stop_tile)
        skip_pcts.append(skip_pct)
        dsd_times.append(elapsed_i)

        if (i + 1) % max(1, n_batch // 20) == 0 or (i + 1) == n_batch:
            avg_skip = sum(skip_pcts) / len(skip_pcts)
            avg_dsd  = sum(dsd_times) / len(dsd_times)
            print(f"  [{i+1:4d}/{n_batch}]  "
                  f"avg_skip={avg_skip:.1f}%  "
                  f"avg_dsd_time={avg_dsd*1000:.2f}ms  "
                  f"elapsed={time.time()-t_loop:.1f}s")

    dsd_wall = time.time() - t_loop

    # --- Step 4: dense 計測 (検証・比較用; ランタイム計測とは別) ---
    print("\n[step4/dense] dense W @ h を計測 (検証用, ランタイムとは別パス) ...")
    dense_token_ids = []
    dense_times     = []
    for i in range(n_batch):
        t0 = time.perf_counter()
        logit = W @ h_batch[i]
        if bias is not None:
            logit = logit + bias
        dense_times.append(time.perf_counter() - t0)
        dense_token_ids.append(int(logit.argmax()))

    # --- 統計 ---
    n_match  = sum(d == r for d, r in zip(dsd_token_ids, dense_token_ids))
    speedups = [d / s for d, s in zip(dense_times, dsd_times) if s > 0]

    avg_stop  = sum(stop_tiles) / n_batch
    med_stop  = sorted(stop_tiles)[n_batch // 2]
    avg_skip  = sum(skip_pcts)  / n_batch
    avg_spdup = sum(speedups)   / len(speedups)
    med_spdup = sorted(speedups)[len(speedups) // 2]
    alive     = compute_alive_curve(stop_tiles, n_tiles)

    elapsed_total = fwd_elapsed + dsd_wall

    print(f"\n{'='*60}")
    print("DSD Runtime  (lm_head tile-by-tile early stop)")
    print(f"{'='*60}")
    print(f"  n_users           = {n_batch}")
    print(f"  n_tiles           = {n_tiles}  (chunk_size={chunk_size})")
    print(f"  avg stop_tile     = {avg_stop:.1f}")
    print(f"  median stop_tile  = {med_stop}")
    print(f"  avg skip %        = {avg_skip:.1f}%")
    print(f"  Top1 match        = {n_match}/{n_batch}  ({n_match/n_batch*100:.1f}%)")
    print(f"  lm_head speedup   = {avg_spdup:.2f}x (avg)  {med_spdup:.2f}x (median)")
    print(f"  fwd elapsed       = {fwd_elapsed:.1f}s")
    print(f"  dsd loop elapsed  = {dsd_wall:.1f}s\n")

    print(f"  {'Tile':>6}  {'Alive':>8}  {'%':>6}  bar")
    print("  " + "-" * 52)
    for t, a in enumerate(alive, start=1):
        pct = a / n_batch * 100
        bar = "█" * int(pct / 2)
        print(f"  Tile{t:2d}  {a:>8}  {pct:5.1f}%  {bar}")
    print(f"{'='*60}\n")

    plot_runtime(alive, speedups, n_batch, n_tiles, args.plot)

    records = [
        {
            "prompt":       prompts[i],
            "stop_tile":    stop_tiles[i],
            "n_tiles":      n_tiles,
            "skip_pct":     skip_pcts[i],
            "top1_match":   dsd_token_ids[i] == dense_token_ids[i],
            "dsd_token_id":   dsd_token_ids[i],
            "dense_token_id": dense_token_ids[i],
            "dsd_elapsed_s":   dsd_times[i],
            "dense_elapsed_s": dense_times[i],
        }
        for i in range(n_batch)
    ]

    output = {
        "model":      args.model,
        "n_users":    n_batch,
        "n_tiles":    n_tiles,
        "chunk_size": chunk_size,
        "hidden_dim": hidden_dim,
        "ordering":   "bound_first",
        "statistics": {
            "avg_stop_tile":         round(avg_stop, 2),
            "median_stop_tile":      med_stop,
            "avg_skip_pct":          round(avg_skip, 2),
            "top1_match_rate":       round(n_match / n_batch, 4),
            "avg_lmhead_speedup":    round(avg_spdup, 3),
            "median_lmhead_speedup": round(med_spdup, 3),
            "fwd_elapsed_s":         round(fwd_elapsed, 2),
            "dsd_loop_elapsed_s":    round(dsd_wall, 2),
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
