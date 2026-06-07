"""
experiments/batch_survival_phase1.py  --  遅延判定 DSD の生存曲線

仮説:
  「Tile 40 以前では停止がほぼ発生しない」

検証:
  Tile 1 〜 START_CHECK_TILE (デフォルト 40):
      Δ の累積のみ実行。停止判定を行わない。
  Tile START_CHECK_TILE+1 〜 n_tiles:
      通常通り Δ > 2B で停止判定。

比較対象 (batch_survival.py の結果):
  - Top1 match rate
  - avg skip %
  - avg stop tile
  - alive_users(tile)

出力:
  batch_survival_phase1.json
  batch_survival_phase1_curve.png

元ファイル (batch_survival.py) は変更しない。
"""

import sys, os, json, math, time, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ordering_exp_v2 import order_bound_first
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME      = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE      = 80
N_BATCH         = 512
START_CHECK_TILE = 40     # このタイル以前は停止判定しない

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
# Model utilities (batch_survival.py と同一)
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
    seq_lens = attention_mask.sum(dim=1) - 1
    h_batch = last_hidden[torch.arange(last_hidden.size(0), device=device), seq_lens, :]
    return h_batch.detach().cpu().float()


# ---------------------------------------------------------------------------
# Phase-1 DSD: 遅延停止判定
# ---------------------------------------------------------------------------

def run_user_phase1(
    h: torch.Tensor,
    W: torch.Tensor,
    bias,
    chunk_size: int,
    start_check_tile: int,
) -> dict:
    """
    bound_first 順序で DSD を実行する。

    Tile 1 〜 start_check_tile:
        部分ロジットを累積するが Δ > 2B 判定を行わない。
    Tile start_check_tile+1 〜 n_tiles:
        通常通り Δ > 2B で停止判定。

    Top1 一致は dense (W @ h) と比較して検証する。
    """
    H          = h.shape[0]
    vocab_size = W.shape[0]
    n_tiles    = math.ceil(H / chunk_size)

    # bound_first 並び替え
    order = order_bound_first(h, W)
    h_ord = h[order]
    W_ord = W[:, order]

    # B の suffix sum を事前計算
    weight_col_max = W_ord.abs().max(dim=0).values   # (H,)
    h_abs          = h_ord.abs()
    suffix_B       = (weight_col_max * h_abs).flip(0).cumsum(0).flip(0)  # (H,)

    partial_logit  = bias.clone() if bias is not None else torch.zeros(vocab_size)
    stop_tile      = n_tiles   # デフォルト: 最後まで

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end   = min(tile * chunk_size, H)

        # このタイルを読む
        for i in range(dim_start, dim_end):
            partial_logit.add_(W_ord[:, i] * h_ord[i])

        # --- 停止判定: start_check_tile より前はスキップ ---
        if tile < start_check_tile:
            continue

        dim_done = dim_end
        top2     = partial_logit.topk(2)
        delta    = float(top2.values[0] - top2.values[1])
        B        = float(suffix_B[dim_done]) if dim_done < H else 0.0

        if delta > 2.0 * B:
            stop_tile = tile
            break

    token_id = int(partial_logit.argmax())

    # dense top1
    with torch.no_grad():
        logit = W @ h
        if bias is not None:
            logit = logit + bias
    dense_id = int(logit.argmax())

    return {
        "stop_dim":   stop_tile * chunk_size,   # 近似 (タイル単位)
        "stop_tile":  stop_tile,
        "n_tiles":    n_tiles,
        "skip_pct":   round((n_tiles - stop_tile) / n_tiles * 100, 2),
        "top1_match": token_id == dense_id,
        "dsd_token_id":   token_id,
        "dense_token_id": dense_id,
    }


# ---------------------------------------------------------------------------
# Alive curve
# ---------------------------------------------------------------------------

def compute_alive_curve(records: list[dict], n_tiles: int) -> list[int]:
    stop_tiles = [r["stop_tile"] for r in records]
    return [sum(1 for st in stop_tiles if st >= t) for t in range(1, n_tiles + 1)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_survival(alive: list[int], n_users: int, n_tiles: int,
                  start_check_tile: int, out_path: str):
    tiles = list(range(1, n_tiles + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Batch DSD Phase-1 Survival Curve  "
        f"(N={n_users}, bound_first, check from tile {start_check_tile+1})",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, (data, ylabel, title, color) in enumerate([
        (alive,
         "Alive users",
         "Alive users per tile  (absolute)",
         "#4C72B0"),
        ([a / n_users * 100 for a in alive],
         "Alive users (%)",
         "Alive users per tile  (percentage)",
         "#55A868"),
    ]):
        ax = axes[ax_idx]
        ax.fill_between(tiles, data, alpha=0.25, color=color)
        ax.plot(tiles, data, color=color, lw=2, label="Phase-1")

        # START_CHECK_TILE の垂直線
        ax.axvline(start_check_tile, color="#DD8452", ls="--", lw=1.2,
                   label=f"check starts @ tile {start_check_tile+1}")

        ax.set_xlabel("Tile index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(1, n_tiles)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        if ax_idx == 0:
            ax.set_ylim(0, n_users * 1.05)
            for pct, col in [(0.5, "#DD8452"), (0.1, "#C44E52")]:
                thresh = n_users * pct
                crosses = [t for t, a in zip(tiles, alive) if a <= thresh]
                if crosses:
                    t_cross = crosses[0]
                    ax.axvline(t_cross, color=col, ls=":", lw=1)
                    ax.text(t_cross + 0.2, thresh + n_users * 0.02,
                            f"{int(pct*100)}%@{t_cross}", fontsize=7, color=col)
        else:
            ax.set_ylim(0, 105)
            ax.axhline(50, color="#DD8452", ls="--", lw=0.8)
            ax.axhline(10, color="#C44E52", ls="--", lw=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch DSD Phase-1 survival curve")
    parser.add_argument("--model",            default=MODEL_NAME)
    parser.add_argument("--n_batch",          type=int, default=N_BATCH)
    parser.add_argument("--chunk_size",       type=int, default=CHUNK_SIZE)
    parser.add_argument("--start_check_tile", type=int, default=START_CHECK_TILE,
                        help="このタイル番号より前は停止判定しない (1-indexed, default=40)")
    parser.add_argument("--output", default="batch_survival_phase1.json")
    parser.add_argument("--plot",   default="batch_survival_phase1_curve.png")
    args = parser.parse_args()

    chunk_size       = args.chunk_size
    n_batch          = args.n_batch
    start_check_tile = args.start_check_tile

    base    = DEFAULT_PROMPTS
    prompts = [base[i % len(base)] for i in range(n_batch)]

    tokenizer, model = load_gemma(args.model)

    cfg = model.config
    if hasattr(cfg, 'text_config'):
        hidden_dim = cfg.text_config.hidden_size
        vocab_size = cfg.text_config.vocab_size
    else:
        hidden_dim = cfg.hidden_size
        vocab_size = cfg.vocab_size
    n_tiles  = math.ceil(hidden_dim / chunk_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model info]  hidden_dim={hidden_dim}  vocab_size={vocab_size}  "
          f"params={n_params/1e9:.2f}B")
    print(f"[batch]  n_users={n_batch}  chunk_size={chunk_size}  n_tiles={n_tiles}")
    print(f"[phase1] 停止判定: tile {start_check_tile+1}〜{n_tiles} のみ\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)\n")

    # --- バッチ forward ---
    print(f"[forward] {n_batch} ユーザーを一括 forward 中...")
    t_fwd = time.time()
    h_batch = batch_hidden_states(model, tokenizer, prompts)
    fwd_elapsed = time.time() - t_fwd
    print(f"  h_batch.shape={tuple(h_batch.shape)}  ({fwd_elapsed:.1f}s)\n")

    # --- Phase-1 DSD ---
    print("[dsd/phase1] 遅延停止判定を実行中...")
    records = []
    t_start = time.time()
    for i in range(n_batch):
        rec = run_user_phase1(
            h_batch[i], W, bias,
            chunk_size=chunk_size,
            start_check_tile=start_check_tile,
        )
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

    dsd_elapsed   = time.time() - t_start
    elapsed_total = fwd_elapsed + dsd_elapsed

    # --- 統計 ---
    n_match    = sum(r["top1_match"] for r in records)
    stop_tiles = [r["stop_tile"] for r in records]
    skip_pcts  = [r["skip_pct"]  for r in records]
    avg_stop   = sum(stop_tiles) / n_batch
    med_stop   = sorted(stop_tiles)[n_batch // 2]
    avg_skip   = sum(skip_pcts) / n_batch
    alive      = compute_alive_curve(records, n_tiles)

    # --- コンソール表示 ---
    print(f"\n{'='*58}")
    print(f"Batch DSD Phase-1 Survival Curve  "
          f"(check from tile {start_check_tile+1})")
    print(f"{'='*58}")
    print(f"  n_users           = {n_batch}")
    print(f"  n_tiles           = {n_tiles}  (chunk_size={chunk_size})")
    print(f"  start_check_tile  = {start_check_tile+1}  "
          f"(tiles 1〜{start_check_tile} は判定スキップ)")
    print(f"  avg stop_tile     = {avg_stop:.1f}")
    print(f"  median stop_tile  = {med_stop}")
    print(f"  avg skip %        = {avg_skip:.1f}%")
    print(f"  Top1 match        = {n_match}/{n_batch}  ({n_match/n_batch*100:.1f}%)")
    print(f"  elapsed           = {elapsed_total:.1f}s  "
          f"(fwd={fwd_elapsed:.1f}s  dsd={dsd_elapsed:.1f}s)\n")

    print(f"  {'Tile':>6}  {'Alive':>8}  {'%':>6}  bar")
    print("  " + "-" * 52)
    for t, a in enumerate(alive, start=1):
        pct    = a / n_batch * 100
        bar    = "█" * int(pct / 2)
        marker = " ← check start" if t == start_check_tile + 1 else ""
        print(f"  Tile{t:2d}  {a:>8}  {pct:5.1f}%  {bar}{marker}")
    print(f"{'='*58}\n")

    plot_survival(alive, n_batch, n_tiles, start_check_tile, args.plot)

    # --- JSON 保存 ---
    output = {
        "model":             args.model,
        "n_users":           n_batch,
        "n_tiles":           n_tiles,
        "chunk_size":        chunk_size,
        "hidden_dim":        hidden_dim,
        "ordering":          "bound_first",
        "start_check_tile":  start_check_tile,
        "statistics": {
            "avg_stop_tile":    round(avg_stop, 2),
            "median_stop_tile": med_stop,
            "avg_skip_pct":     round(avg_skip, 2),
            "top1_match_rate":  n_match / n_batch,
            "elapsed_s":        round(elapsed_total, 2),
            "fwd_elapsed_s":    round(fwd_elapsed, 2),
            "dsd_elapsed_s":    round(dsd_elapsed, 2),
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
