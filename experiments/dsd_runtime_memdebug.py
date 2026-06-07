"""
experiments/dsd_runtime_memdebug.py  --  DSD ランタイム GPU 版 メモリ診断

目的:
  GPU 版移行後に W[:, order] 等がどれだけ VRAM を消費しているか確認する。
  最適化は行わない。どこでメモリを食っているかを計測することが目的。

診断対象:
  ・W[:, order]  (V×H float32 の並び替えコピー)
  ・suffix_B     (H float32 ベクトル)
  ・topk         (vocab_size に対する topk)
  ・実際の VRAM 推移

出力:
  dsd_runtime_memdebug.json  (メモリ推移をすべて記録)
"""

import sys, os, json, math, time, torch

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


# ------------------------------------------------------------------
# helper
# ------------------------------------------------------------------

def gpu_mem_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 3)

def gpu_reserved_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / (1024 ** 3)

def print_gpu_mem(tag: str):
    print(
        f"[GPU] {tag:<25s} "
        f"alloc={gpu_mem_gb():.2f} GB  "
        f"reserved={gpu_reserved_gb():.2f} GB"
    )


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


def get_lm_head_gpu(model, dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor | None]:
    """W と bias を GPU 上で指定 dtype のまま返す。デフォルト bfloat16 (OOM 回避)。"""
    lm     = model.lm_head
    device = next(model.parameters()).device
    W    = lm.weight.detach().to(device=device, dtype=dtype)
    bias = (lm.bias.detach().to(device=device, dtype=dtype)
            if lm.bias is not None else None)
    return W, bias


def batch_hidden_states_gpu(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """最終層 hidden state を GPU bfloat16 のまま返す。"""
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
    return h_batch.detach()   # bfloat16 GPU


# ---------------------------------------------------------------------------
# 事前計算 (メモリ診断あり)
# ---------------------------------------------------------------------------

def precompute_ordering_gpu(
    h_batch: torch.Tensor,
    W: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    N = h_batch.shape[0]
    prepared = []

    # W_ord サイズ推定 (1 ユーザー分のコピーコスト)
    bytes_per_elem = W.element_size()          # bfloat16=2, float32=4
    bytes_est = W.shape[0] * W.shape[1] * bytes_per_elem
    print(
        f"[estimate] W_ord size = "
        f"{bytes_est / 1024**3:.2f} GB  "
        f"(per user, {N} users → "
        f"{bytes_est * N / 1024**3:.2f} GB if all retained)"
    )

    for i in range(N):
        h     = h_batch[i]
        order = order_bound_first(h, W)
        h_ord = h[order]
        W_ord = W[:, order]                              # ← 疑惑の主犯

        weight_col_max = W_ord.abs().max(dim=0).values
        suffix_B = (weight_col_max * h_ord.abs()).flip(0).cumsum(0).flip(0)

        prepared.append((h_ord, W_ord, suffix_B))

        if (i + 1) % 8 == 0:
            print_gpu_mem(f"precompute {i+1}")

    return prepared


# ---------------------------------------------------------------------------
# DSD ランタイム (GPU tile-by-tile)
# ---------------------------------------------------------------------------

def dsd_runtime_single_gpu(
    bias: torch.Tensor | None,
    chunk_size: int,
    suffix_B: torch.Tensor,
    h_ord: torch.Tensor,
    W_ord: torch.Tensor,
    n_tiles: int,
) -> tuple[int, int, float]:
    H          = h_ord.shape[0]
    vocab_size = W_ord.shape[0]

    partial_logit = (bias.clone() if bias is not None
                     else torch.zeros(vocab_size, dtype=W_ord.dtype, device=h_ord.device))
    stop_tile = n_tiles

    for tile in range(1, n_tiles + 1):
        dim_start = (tile - 1) * chunk_size
        dim_end   = min(tile * chunk_size, H)

        partial_logit.add_(
            W_ord[:, dim_start:dim_end] @ h_ord[dim_start:dim_end]
        )

        top2  = partial_logit.topk(2)
        delta = top2.values[0] - top2.values[1]
        B     = suffix_B[dim_end] if dim_end < H else suffix_B.new_zeros(())

        if delta > 2.0 * B:
            stop_tile = tile
            break

    token_id = int(partial_logit.argmax())
    skip_pct = round((n_tiles - stop_tile) / n_tiles * 100, 2)
    return token_id, stop_tile, skip_pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSD runtime GPU memory debug")
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--n_batch",    type=int, default=N_BATCH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output",     default="dsd_runtime_memdebug.json")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    n_batch    = args.n_batch

    base    = DEFAULT_PROMPTS
    prompts = [base[i % len(base)] for i in range(n_batch)]

    # ------------------------------------------------------------------
    # 1. モデルロード
    # ------------------------------------------------------------------
    tokenizer, model = load_gemma(args.model)
    device = next(model.parameters()).device

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
          f"params={n_params/1e9:.2f}B  device={device}")
    print(f"[batch]  n_users={n_batch}  chunk_size={chunk_size}  n_tiles={n_tiles}\n")

    print_gpu_mem("after model load")        # ← 計測点 1
    mem_log = {}

    # ------------------------------------------------------------------
    # 2. lm_head → GPU bfloat16 (float32 コピーは OOM の原因のため避ける)
    # ------------------------------------------------------------------
    print("\n[prep] lm_head を GPU bfloat16 で保持中...")
    t0 = time.time()
    W, bias = get_lm_head_gpu(model, dtype=torch.bfloat16)
    print(f"  W.shape={W.shape}  device={W.device}  dtype={W.dtype}  ({time.time()-t0:.1f}s)")

    print_gpu_mem("after lm_head")           # ← 計測点 2
    mem_log["after_model_load"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                                   "reserved_gb": round(gpu_reserved_gb(), 3)}

    # ------------------------------------------------------------------
    # 3. transformer forward
    # ------------------------------------------------------------------
    print(f"\n[step1/forward] {n_batch} ユーザーを一括 forward 中...")
    torch.cuda.synchronize(device)
    t_fwd = time.time()
    h_batch = batch_hidden_states_gpu(model, tokenizer, prompts)
    torch.cuda.synchronize(device)
    fwd_elapsed = time.time() - t_fwd
    print(f"  h_batch: shape={tuple(h_batch.shape)}  device={h_batch.device}  "
          f"dtype={h_batch.dtype}  ({fwd_elapsed:.1f}s)")

    print_gpu_mem("after hidden")            # ← 計測点 3
    mem_log["after_lm_head"]  = {"alloc_gb": round(gpu_mem_gb(), 3),
                                  "reserved_gb": round(gpu_reserved_gb(), 3)}

    # ------------------------------------------------------------------
    # 4. 事前計算
    # ------------------------------------------------------------------
    print("\n[step2/precompute] bound_first 並び替え + suffix_B (CUDA) ...")
    print_gpu_mem("before precompute")       # ← 計測点 4
    mem_log["before_precompute"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                                     "reserved_gb": round(gpu_reserved_gb(), 3)}

    torch.cuda.synchronize(device)
    t_pre = time.time()
    prepared = precompute_ordering_gpu(h_batch, W)    # ← ループ内で計測点 5
    torch.cuda.synchronize(device)
    print(f"  ({time.time()-t_pre:.1f}s)")

    print_gpu_mem("after precompute")        # ← 計測点 6
    mem_log["after_precompute"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                                    "reserved_gb": round(gpu_reserved_gb(), 3)}

    # ------------------------------------------------------------------
    # 5. DSD ランタイム
    # ------------------------------------------------------------------
    print("\n[step3/dsd] lm_head tile-by-tile CUDA early stop ...")
    print_gpu_mem("before dsd")              # ← 計測点 7
    mem_log["before_dsd"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                              "reserved_gb": round(gpu_reserved_gb(), 3)}

    dsd_token_ids = []
    stop_tiles    = []
    skip_pcts_    = []
    dsd_times     = []

    torch.cuda.synchronize(device)
    t_loop = time.time()
    for i in range(n_batch):
        h_ord, W_ord, suffix_B = prepared[i]

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        token_id, stop_tile, skip_pct = dsd_runtime_single_gpu(
            bias, chunk_size, suffix_B, h_ord, W_ord, n_tiles,
        )
        torch.cuda.synchronize(device)
        elapsed_i = time.perf_counter() - t0

        dsd_token_ids.append(token_id)
        stop_tiles.append(stop_tile)
        skip_pcts_.append(skip_pct)
        dsd_times.append(elapsed_i)

        if (i + 1) % max(1, n_batch // 20) == 0 or (i + 1) == n_batch:
            avg_skip = sum(skip_pcts_) / len(skip_pcts_)
            avg_dsd  = sum(dsd_times)  / len(dsd_times)
            print(f"  [{i+1:4d}/{n_batch}]  "
                  f"avg_skip={avg_skip:.1f}%  "
                  f"avg_dsd={avg_dsd*1000:.2f}ms  "
                  f"elapsed={time.time()-t_loop:.1f}s")

    torch.cuda.synchronize(device)
    dsd_wall = time.time() - t_loop

    print_gpu_mem("after dsd")               # ← 計測点 8
    mem_log["after_dsd"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                             "reserved_gb": round(gpu_reserved_gb(), 3)}

    # ------------------------------------------------------------------
    # 6. dense 計測 (検証用)
    # ------------------------------------------------------------------
    print("\n[step4/dense] dense W @ h 計測 (検証用, 別パス) ...")
    dense_token_ids = []
    dense_times     = []
    for i in range(n_batch):
        h = h_batch[i]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logit = W @ h
        if bias is not None:
            logit = logit + bias
        torch.cuda.synchronize(device)
        dense_times.append(time.perf_counter() - t0)
        dense_token_ids.append(int(logit.argmax()))

    print_gpu_mem("after dense")             # ← 計測点 9
    mem_log["after_dense"] = {"alloc_gb": round(gpu_mem_gb(), 3),
                               "reserved_gb": round(gpu_reserved_gb(), 3)}

    # ------------------------------------------------------------------
    # 統計
    # ------------------------------------------------------------------
    n_match  = sum(d == r for d, r in zip(dsd_token_ids, dense_token_ids))
    speedups = [d / s for d, s in zip(dense_times, dsd_times) if s > 0]

    avg_stop  = sum(stop_tiles) / n_batch
    avg_skip  = sum(skip_pcts_) / n_batch
    avg_spdup = sum(speedups)   / len(speedups)

    # W_ord 理論サイズ (element_size() は dtype に応じて自動: bf16=2, fp32=4)
    bytes_W_ord_single = W.element_size() * W.shape[0] * W.shape[1]
    bytes_W_ord_all    = bytes_W_ord_single * n_batch

    print(f"\n{'='*62}")
    print("DSD Runtime GPU  メモリ診断サマリ")
    print(f"{'='*62}")
    print(f"  W (original)         : {W.element_size() * W.numel() / 1024**3:.2f} GB  "
          f"({W.dtype}, {tuple(W.shape)})")
    print(f"  W_ord (1 user copy)  : {bytes_W_ord_single / 1024**3:.2f} GB")
    print(f"  W_ord (N={n_batch} users) : {bytes_W_ord_all / 1024**3:.2f} GB  (if all retained)")
    print(f"  h_batch              : {h_batch.element_size() * h_batch.numel() / 1024**2:.1f} MB  "
          f"({h_batch.dtype}, {tuple(h_batch.shape)})")
    print()
    print(f"  {'Phase':<20s}  {'alloc':>8s}  {'reserved':>10s}")
    print(f"  {'-'*42}")
    for phase, vals in mem_log.items():
        print(f"  {phase:<20s}  {vals['alloc_gb']:>7.2f}GB  {vals['reserved_gb']:>9.2f}GB")
    print()
    print(f"  avg stop_tile  = {avg_stop:.1f}  avg skip = {avg_skip:.1f}%")
    print(f"  Top1 match     = {n_match}/{n_batch}  ({n_match/n_batch*100:.1f}%)")
    print(f"  lm_head speedup= {avg_spdup:.2f}x (avg)")
    print(f"{'='*62}\n")

    records = [
        {
            "prompt":         prompts[i],
            "stop_tile":      stop_tiles[i],
            "skip_pct":       skip_pcts_[i],
            "top1_match":     dsd_token_ids[i] == dense_token_ids[i],
            "dsd_elapsed_ms":   round(dsd_times[i]   * 1000, 4),
            "dense_elapsed_ms": round(dense_times[i] * 1000, 4),
        }
        for i in range(n_batch)
    ]

    output = {
        "model":      args.model,
        "n_users":    n_batch,
        "n_tiles":    n_tiles,
        "chunk_size": chunk_size,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "ordering":   "bound_first",
        "compute":    f"CUDA {W.dtype} (no CPU transfer)",
        "W_dtype":    str(W.dtype),
        "memory_sizes_gb": {
            "W_original":        round(W.element_size() * W.numel() / 1024**3, 3),
            "W_ord_per_user":    round(bytes_W_ord_single / 1024**3, 3),
            "W_ord_all_users":   round(bytes_W_ord_all    / 1024**3, 3),
            "h_batch_mb":        round(h_batch.element_size() * h_batch.numel() / 1024**2, 2),
        },
        "memory_log": mem_log,
        "statistics": {
            "avg_stop_tile":      round(avg_stop, 2),
            "avg_skip_pct":       round(avg_skip, 2),
            "top1_match_rate":    round(n_match / n_batch, 4),
            "avg_lmhead_speedup": round(avg_spdup, 3),
            "fwd_elapsed_s":      round(fwd_elapsed, 2),
            "dsd_loop_elapsed_s": round(dsd_wall, 2),
        },
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
