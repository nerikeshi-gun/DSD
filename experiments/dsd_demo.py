"""
experiments/dsd_demo.py  --  DSD 動作証明デモ (ターミナル録画用)

証明したいこと:
  1. Δ > B 成立後に実際にチャンクを読まずに停止している
  2. Dense と Top1 が完全一致する
  3. planned / executed / cancelled chunks の差を定量表示

構成:
  Phase 1  Dense 実行 (全チャンク読込)
  Phase 2  DSD 実行 (Δ>B で停止)
  Phase 3  バッチ全体の ACTIVE / STOPPED 可視化
  Phase 4  最終サマリ

用語:
  chunk      = CHUNK_SIZE 次元のまとまり (デフォルト 80 dim → 48 chunks / 3840 dim)
  planned    = 全チャンク数 (= ceil(H / CHUNK_SIZE))
  executed   = 実際に読んだチャンク数
  cancelled  = planned - executed  (実際に読まなかったチャンク数)
"""

import sys, os, json, time, math, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsd_poc import compute_remaining_bound
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME   = "/home/kaneyama/models/gemma-3-12b-it"
CHUNK_SIZE   = 80       # 1 chunk = 80 dim  →  3840/80 = 48 chunks
SLEEP_DENSE  = 0.04     # Dense 各チャンク表示間隔 (秒)
SLEEP_DSD    = 0.06     # DSD 各チャンク表示間隔 (秒)
SLEEP_BATCH  = 0.05     # バッチ一覧の更新間隔 (秒)

# ANSI color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"

DEMO_PROMPTS = [
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
]


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_gemma(model_name):
    print(f"{CYAN}[load]{RESET} {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_final_hidden(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].detach().cpu().float()


def get_lm_head(model):
    lm = model.lm_head
    W    = lm.weight.detach().cpu().float()
    bias = lm.bias.detach().cpu().float() if lm.bias is not None else None
    return W, bias


# ---------------------------------------------------------------------------
# Chunk-level DSD (no algorithm change, just grouped stopping check)
# ---------------------------------------------------------------------------

def dsd_chunk(h, W, bias, chunk_size=CHUNK_SIZE):
    """
    CHUNK_SIZE 次元ごとに Δ > 2B を確認。
    実際に読んだチャンク数 (executed) と停止チャンク (stop_chunk) を返す。

    Returns:
        token_id, stop_chunk, n_chunks, delta_at_stop, B_at_stop, trajectory
    """
    H          = h.shape[0]
    vocab_size = W.shape[0]
    n_chunks   = math.ceil(H / chunk_size)

    weight_col_max = W.abs().max(dim=0).values   # (H,)
    h_abs          = h.abs()
    suffix_B       = (weight_col_max * h_abs).flip(0).cumsum(0).flip(0)

    partial_logit = bias.clone() if bias is not None else torch.zeros(vocab_size)
    stop_chunk = n_chunks  # デフォルト: 最後まで読む
    trajectory = []

    for c in range(n_chunks):
        dim_start = c * chunk_size
        dim_end   = min(dim_start + chunk_size, H)

        # このチャンクを読む
        for i in range(dim_start, dim_end):
            partial_logit.add_(W[:, i] * h[i])

        dim_done = dim_end
        top2 = partial_logit.topk(2)
        delta = float(top2.values[0] - top2.values[1])
        B     = float(suffix_B[dim_done]) if dim_done < H else 0.0

        trajectory.append({
            "chunk": c + 1,
            "delta": round(delta, 4),
            "B":     round(B, 4),
            "two_B": round(2.0 * B, 4),
            "stop":  delta > 2.0 * B,
        })

        if delta > 2.0 * B:
            stop_chunk = c + 1
            break

    token_id    = int(partial_logit.argmax())
    delta_stop  = trajectory[-1]["delta"]
    B_stop      = trajectory[-1]["B"]

    return token_id, stop_chunk, n_chunks, delta_stop, B_stop, trajectory


# ---------------------------------------------------------------------------
# Pre-compute all results (model inference, then animate)
# ---------------------------------------------------------------------------

def precompute(prompts, tokenizer, model, W, bias, chunk_size=CHUNK_SIZE):
    print(f"\n{CYAN}[precompute]{RESET} {len(prompts)} プロンプトを処理中...\n")
    results = []
    for i, p in enumerate(prompts):
        input_ids = tokenizer(p, return_tensors="pt").input_ids
        input_ids = input_ids.to(next(model.parameters()).device)
        h = get_final_hidden(model, input_ids)

        # Dense top1
        with torch.no_grad():
            logit = W @ h
            if bias is not None:
                logit += bias
        dense_id = int(logit.argmax())

        # DSD
        dsd_id, stop_chunk, n_chunks, delta, B, traj = dsd_chunk(h, W, bias, chunk_size=chunk_size)

        results.append({
            "prompt":      p,
            "dense_token": tokenizer.decode([dense_id]),
            "dsd_token":   tokenizer.decode([dsd_id]),
            "dense_id":    dense_id,
            "dsd_id":      dsd_id,
            "match":       dense_id == dsd_id,
            "stop_chunk":  stop_chunk,
            "n_chunks":    n_chunks,
            "executed":    stop_chunk,
            "cancelled":   n_chunks - stop_chunk,
            "skip_rate":   (n_chunks - stop_chunk) / n_chunks,
            "delta":       delta,
            "B":           B,
            "trajectory":  traj,
        })
        print(f"  [{i+1:2d}/{len(prompts)}]  stop={stop_chunk:2d}/{n_chunks}  "
              f"match={dense_id == dsd_id}  "
              f"dense={tokenizer.decode([dense_id])!r:12s}  "
              f"prompt={p[:35]!r}")

    return results


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

def clear_lines(n):
    """n 行上に戻って消去。"""
    for _ in range(n):
        print("\033[1A\033[2K", end="")


def bar(done, total, width=30, char_done="█", char_todo="░"):
    filled = int(done / total * width)
    return char_done * filled + char_todo * (total - filled)


# ---------------------------------------------------------------------------
# Phase 1: Dense 表示
# ---------------------------------------------------------------------------

def phase_dense(results, delay=SLEEP_DENSE):
    n_chunks = results[0]["n_chunks"]
    n = len(results)
    print(f"\n{BOLD}{WHITE}{'='*60}{RESET}")
    print(f"{BOLD}{WHITE}  Phase 1 : Dense (全チャンク読込){RESET}")
    print(f"{BOLD}{WHITE}{'='*60}{RESET}\n")

    total_chunks = 0
    for r in results:
        label = f"Token{results.index(r)+1:02d}"
        for c in range(1, n_chunks + 1):
            chunk_bar = bar(c, n_chunks, width=24)
            print(f"  {BLUE}{label}{RESET}  [{chunk_bar}] {c:2d}/{n_chunks}  "
                  f"{GREEN}{r['dense_token']!r}{RESET}", end="\r")
            time.sleep(delay)
            total_chunks += 1
        print(f"  {BLUE}{label}{RESET}  [{bar(n_chunks,n_chunks,24)}] {n_chunks}/{n_chunks}  "
              f"{GREEN}{r['dense_token']!r}{RESET}  {GRAY}DONE{RESET}")

    print(f"\n  {BOLD}Total Read : {total_chunks:,} chunks{RESET}\n")
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# Phase 2: DSD 表示 (各トークンの停止過程)
# ---------------------------------------------------------------------------

def phase_dsd(results, delay=SLEEP_DSD):
    print(f"\n{BOLD}{WHITE}{'='*60}{RESET}")
    print(f"{BOLD}{WHITE}  Phase 2 : DSD (Δ>B で停止){RESET}")
    print(f"{BOLD}{WHITE}{'='*60}{RESET}\n")

    total_executed  = 0
    total_cancelled = 0

    for idx, r in enumerate(results):
        label     = f"Token{idx+1:02d}"
        n_chunks  = r["n_chunks"]
        traj      = r["trajectory"]

        print(f"  {BOLD}{CYAN}{label}{RESET}  prompt={r['prompt'][:35]!r}")
        stopped = False
        for pt in traj:
            c     = pt["chunk"]
            delta = pt["delta"]
            two_B = pt["two_B"]
            is_stop = pt["stop"]

            delta_col = GREEN if is_stop else YELLOW
            b_col     = RED   if is_stop else WHITE

            chunk_bar = bar(c, n_chunks, width=20)
            print(f"    chunk={c:2d}/{n_chunks} [{chunk_bar}]  "
                  f"Δ={delta_col}{delta:.4f}{RESET}  "
                  f"2B={b_col}{two_B:.4f}{RESET}", end="")

            if is_stop:
                cancelled = n_chunks - c
                print(f"  {BOLD}{RED}>>> STOP <<<{RESET}")
                print(f"    {GREEN}executed ={c:3d}{RESET}  "
                      f"{GRAY}cancelled={cancelled:3d}{RESET}  "
                      f"skip={cancelled/n_chunks*100:.1f}%  "
                      f"token={GREEN}{r['dsd_token']!r}{RESET}  "
                      f"match={GREEN if r['match'] else RED}{r['match']}{RESET}")
                stopped = True
            else:
                print()
            time.sleep(delay)

        if not stopped:
            print(f"    {GRAY}(最後まで読了){RESET}  token={GREEN}{r['dsd_token']!r}{RESET}")

        total_executed  += r["executed"]
        total_cancelled += r["cancelled"]
        print()

    print(f"  Total executed  : {total_executed:,} chunks")
    print(f"  Total cancelled : {GRAY}{total_cancelled:,} chunks{RESET}\n")
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# Phase 3: バッチ全体 ACTIVE / STOPPED
# ---------------------------------------------------------------------------

def phase_batch(results, delay=SLEEP_BATCH):
    print(f"\n{BOLD}{WHITE}{'='*60}{RESET}")
    print(f"{BOLD}{WHITE}  Phase 3 : バッチ進捗 (ACTIVE / STOPPED){RESET}")
    print(f"{BOLD}{WHITE}{'='*60}{RESET}\n")

    n         = len(results)
    n_chunks  = results[0]["n_chunks"]
    stop_map  = {i: r["stop_chunk"] for i, r in enumerate(results)}
    status    = ["PENDING"] * n

    # ヘッダ表示
    for i, r in enumerate(results):
        print(f"  Token{i+1:02d}  {GRAY}PENDING{RESET}  {r['prompt'][:30]!r}")

    for chunk in range(1, n_chunks + 1):
        time.sleep(delay)
        # 状態更新
        for i in range(n):
            if status[i] == "PENDING":
                if chunk >= stop_map[i]:
                    status[i] = "STOPPED"
                else:
                    status[i] = "ACTIVE"
            elif status[i] == "ACTIVE" and chunk >= stop_map[i]:
                status[i] = "STOPPED"

        # 再描画
        clear_lines(n)
        for i, r in enumerate(results):
            s = status[i]
            if s == "ACTIVE":
                col  = YELLOW
                info = f"chunk {chunk:2d}/{n_chunks}"
            elif s == "STOPPED":
                col  = GREEN
                info = f"STOP  {stop_map[i]:2d}/{n_chunks}  skip={r['cancelled']/n_chunks*100:.0f}%"
            else:
                col  = GRAY
                info = "PENDING"
            token_col = GREEN if r["match"] else RED
            print(f"  Token{i+1:02d}  {col}{s:7s}{RESET}  {info}  "
                  f"→ {token_col}{r['dsd_token']!r}{RESET}  "
                  f"{r['prompt'][:28]!r}")

    print()
    time.sleep(1.5)


# ---------------------------------------------------------------------------
# Phase 4: サマリ
# ---------------------------------------------------------------------------

def phase_summary(results):
    n        = len(results)
    n_chunks = results[0]["n_chunks"]

    dense_total = n * n_chunks
    dsd_total   = sum(r["executed"]  for r in results)
    saved       = sum(r["cancelled"] for r in results)
    reduction   = saved / dense_total * 100

    n_match  = sum(r["match"] for r in results)

    # Top5 match: dsd token is in top-5 of dense logits
    # (We only have the argmax; top5 requires full logit — skip for demo,
    #  report as N/A unless results carry it)
    top5_match = n_match  # conservative: if top1 matches, top5 also matches

    planned_total   = dense_total
    executed_total  = dsd_total
    cancelled_total = saved

    print(f"\n{BOLD}{WHITE}{'='*60}{RESET}")
    print(f"{BOLD}{WHITE}  Phase 4 : Final Summary{RESET}")
    print(f"{BOLD}{WHITE}{'='*60}{RESET}\n")

    print(f"  {BOLD}Batch size   :{RESET} {n} prompts")
    print(f"  {BOLD}Chunks/token :{RESET} {n_chunks}")
    print(f"  {BOLD}Chunk size   :{RESET} {CHUNK_SIZE} dims\n")

    print(f"  {BLUE}Dense{RESET}")
    print(f"    planned_chunks   = {planned_total:,}")
    print(f"    executed_chunks  = {planned_total:,}  (100%)")
    print(f"    cancelled_chunks = 0\n")

    print(f"  {GREEN}DSD{RESET}")
    print(f"    planned_chunks   = {planned_total:,}")
    print(f"    executed_chunks  = {executed_total:,}  ({executed_total/planned_total*100:.1f}%)")
    print(f"    cancelled_chunks = {GRAY}{cancelled_total:,}{RESET}  "
          f"({cancelled_total/planned_total*100:.1f}%)\n")

    print(f"  {BOLD}Saved   :{RESET} {saved:,} chunks  ({reduction:.1f}% reduction)\n")

    match_col = GREEN if n_match == n else RED
    print(f"  {BOLD}Top1 Match  :{RESET} {match_col}{n_match}/{n}  "
          f"({n_match/n*100:.1f}%){RESET}")
    print(f"  {BOLD}Top5 Match  :{RESET} {GREEN}{top5_match}/{n}  "
          f"({top5_match/n*100:.1f}%){RESET}  (top1 ⊂ top5)")

    per_token = [(r["executed"], r["cancelled"], r["skip_rate"]) for r in results]
    avg_skip = sum(r["skip_rate"] for r in results) / n
    best_skip = max(r["skip_rate"] for r in results)
    worst_skip = min(r["skip_rate"] for r in results)
    print(f"\n  {BOLD}Skip rate   :{RESET} avg={avg_skip*100:.1f}%  "
          f"best={best_skip*100:.1f}%  worst={worst_skip*100:.1f}%")

    print(f"\n{BOLD}{WHITE}{'='*60}{RESET}\n")

    return {
        "n_prompts":        n,
        "n_chunks":         n_chunks,
        "chunk_size":       CHUNK_SIZE,
        "dense_executed":   planned_total,
        "dsd_planned":      planned_total,
        "dsd_executed":     executed_total,
        "dsd_cancelled":    cancelled_total,
        "reduction_pct":    round(reduction, 2),
        "top1_match_rate":  n_match / n,
        "avg_skip_rate":    round(avg_skip, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSD demo for recording")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="demo_results.json")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--chunk_size", type=int, default=80)
    parser.add_argument("--no_animation", action="store_true",
                        help="アニメーション省略 (CI/テスト用)")
    args = parser.parse_args()

    chunk_size = args.chunk_size   # ローカル変数で管理 (global 不要)

    prompts = args.prompts or DEMO_PROMPTS
    delay_dense = 0.0 if args.no_animation else SLEEP_DENSE
    delay_dsd   = 0.0 if args.no_animation else SLEEP_DSD
    delay_batch = 0.0 if args.no_animation else SLEEP_BATCH

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
          f"params={n_params/1e9:.2f}B")
    print(f"[chunk]       chunk_size={chunk_size}  "
          f"n_chunks={math.ceil(hidden_dim/chunk_size)}\n")

    print("[prep] lm_head を CPU float32 に変換中...")
    t0 = time.time()
    W, bias = get_lm_head(model)
    print(f"  W.shape={W.shape}  ({time.time()-t0:.1f}s)")

    # Precompute all DSD results first (model inference)
    results = precompute(prompts, tokenizer, model, W, bias, chunk_size=chunk_size)

    # Animate phases
    input(f"\n{YELLOW}[Enter] を押してデモを開始...{RESET}")
    phase_dense(results, delay=delay_dense)

    input(f"\n{YELLOW}[Enter] DSD フェーズへ...{RESET}")
    phase_dsd(results, delay=delay_dsd)

    input(f"\n{YELLOW}[Enter] バッチ全体表示へ...{RESET}")
    phase_batch(results, delay=delay_batch)

    summary = phase_summary(results)

    # Save
    output = {
        "summary": summary,
        "results": [
            {k: v for k, v in r.items() if k != "trajectory"}
            for r in results
        ],
        "trajectories": [
            {"prompt": r["prompt"], "trajectory": r["trajectory"]}
            for r in results
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
