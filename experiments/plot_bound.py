"""
experiments/plot_bound.py  --  B 構造の可視化

bound_decompose.json を読み込み、3種類のグラフを生成する:

  1. contrib_profile.png
       各次元の contrib_i = max_v|W[v,i]| * |h[i]| の降順プロット
       → 少数成分に支配されているか / 均一か を一目で確認

  2. cumulative_coverage.png
       上位 k 次元で B の何 % をカバーできるかの累積グラフ
       → 「上位 X 次元を先に読めば B の Y% を消せる」を証明

  3. b_margin_ratio.png
       各 checkpoint での B / margin の推移
       → B が margin に対してどれだけ大きいかを時系列で確認

入力:  bound_decompose.json  (bound_decompose.py の出力)
出力:  contrib_profile.png
       cumulative_coverage.png
       b_margin_ratio.png
"""

import sys, os, json, argparse
import matplotlib
matplotlib.use("Agg")           # ヘッドレス環境対応
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

INPUT_FILE = "bound_decompose.json"
DPI        = 150
FIGSIZE    = (10, 5)

COLORS = {
    "main":   "#4C72B0",
    "accent": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "gray":   "#8C8C8C",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph 1: contrib profile (降順)
# ---------------------------------------------------------------------------

def plot_contrib_profile(data: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("B Contribution Profile per Dimension (sorted descending)",
                 fontsize=13, fontweight="bold")

    for result in data["results"]:
        gs    = result["global_stats"]
        comps = result["top_components"]
        prompt = result["prompt"][:40]

        ranks   = [c["rank"] for c in comps]
        contribs = [c["contrib"] for c in comps]

        ax = axes[0]
        ax.plot(ranks, contribs, alpha=0.7, label=prompt)

    axes[0].set_xlabel("Rank (sorted by contrib, 1=largest)")
    axes[0].set_ylabel("contrib_i  =  max|W[:,i]| × |h[i]|")
    axes[0].set_title("Top-100 components (linear scale)")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    # log scale
    for result in data["results"]:
        comps  = result["top_components"]
        ranks  = [c["rank"] for c in comps]
        contribs = [c["contrib"] for c in comps]
        axes[1].plot(ranks, contribs, alpha=0.7)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("contrib_i  (log scale)")
    axes[1].set_title("Top-100 components (log scale)")
    axes[1].grid(alpha=0.3, which="both")

    # annotation from summary
    gs0 = data["results"][0]["global_stats"]
    ann = (f"top-1  covers {gs0['top1_contrib_pct']:.2f}% of B\n"
           f"top-10 covers {gs0['top10_contrib_pct']:.2f}% of B\n"
           f"top-100 covers {gs0['top100_contrib_pct']:.2f}% of B")
    axes[0].text(0.97, 0.97, ann, transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(out_dir, "contrib_profile.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Graph 2: cumulative coverage
# ---------------------------------------------------------------------------

def plot_cumulative_coverage(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Cumulative B Coverage by Top-k Dimensions",
                 fontsize=13, fontweight="bold")

    for result in data["results"]:
        comps  = result["top_components"]
        gs     = result["global_stats"]
        prompt = result["prompt"][:40]

        ranks    = [c["rank"] for c in comps]
        cum_pcts = [c["cumulative_pct"] for c in comps]

        ax.plot(ranks, cum_pcts, alpha=0.8, label=prompt)

    # Reference lines
    ax.axhline(50, color=COLORS["gray"], ls="--", lw=0.8, label="50% coverage")
    ax.axhline(80, color=COLORS["accent"], ls="--", lw=0.8, label="80% coverage")
    ax.axhline(90, color=COLORS["red"], ls="--", lw=0.8, label="90% coverage")

    # Annotate k_covers_* from first result
    gs0 = data["results"][0]["global_stats"]
    for pct, k in [(50, gs0["k_covers_50pct"]),
                   (80, gs0["k_covers_80pct"]),
                   (90, gs0["k_covers_90pct"])]:
        if k <= 100:
            ax.axvline(k, color=COLORS["gray"], ls=":", lw=0.6)
            ax.text(k + 0.5, pct - 3, f"k={k}", fontsize=7, color=COLORS["gray"])

    ax.set_xlabel("Number of top-k dimensions included")
    ax.set_ylabel("Cumulative % of total B")
    ax.set_ylim(0, 105)
    ax.set_xlim(1, 100)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "cumulative_coverage.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Graph 3: B / margin over checkpoints
# ---------------------------------------------------------------------------

def plot_b_margin_ratio(data: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Remaining Bound B vs Margin over Observation Progress",
                 fontsize=13, fontweight="bold")

    ax_ratio = axes[0]
    ax_abs   = axes[1]

    for result in data["results"]:
        cps    = result["checkpoint_records"]
        prompt = result["prompt"][:40]

        ratios = [cp["observed_ratio"] * 100 for cp in cps]
        bom    = [cp["B_over_margin"] for cp in cps if cp["B_over_margin"] is not None]
        ratios_valid = [cp["observed_ratio"] * 100
                        for cp in cps if cp["B_over_margin"] is not None]

        Bs      = [cp["remaining_B"] for cp in cps]
        margins = [cp["margin"]      for cp in cps]

        ax_ratio.plot(ratios_valid, bom, marker="o", markersize=3,
                      alpha=0.8, label=prompt)
        ax_abs.plot(ratios, Bs,      marker="o", markersize=3, alpha=0.8,
                    label=f"B ({prompt})")
        ax_abs.plot(ratios, margins, marker="s", markersize=3, alpha=0.6,
                    ls="--")

    ax_ratio.axhline(1.0, color=COLORS["red"], ls="--", lw=1.2, label="B/margin = 1 (stop threshold)")
    ax_ratio.set_xlabel("Observed ratio (%)")
    ax_ratio.set_ylabel("B / margin  (log scale)")
    ax_ratio.set_yscale("log")
    ax_ratio.set_title("B/margin ratio  (stop when < 1)")
    ax_ratio.legend(fontsize=7)
    ax_ratio.grid(alpha=0.3, which="both")

    ax_abs.set_xlabel("Observed ratio (%)")
    ax_abs.set_ylabel("Value")
    ax_abs.set_title("Remaining B (solid) vs margin (dashed)")
    ax_abs.set_yscale("log")
    ax_abs.legend(fontsize=6)
    ax_abs.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(out_dir, "b_margin_ratio.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot bound decomposition analysis")
    parser.add_argument("--input",   default=INPUT_FILE)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("先に bound_decompose.py を実行してください:")
        print("  python experiments/bound_decompose.py --output bound_decompose.json")
        sys.exit(1)

    data = load_data(args.input)
    n    = len(data["results"])
    print(f"[load] {args.input}  ({n} prompts)")
    print(f"[info] model={data.get('model','?')}  "
          f"hidden_dim={data.get('hidden_dim','?')}  "
          f"vocab_size={data.get('vocab_size','?')}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_contrib_profile(data, args.out_dir)
    plot_cumulative_coverage(data, args.out_dir)
    plot_b_margin_ratio(data, args.out_dir)

    # Print key numbers
    print("\n=== Key Statistics ===")
    for r in data["results"]:
        gs = r["global_stats"]
        print(f"  {r['prompt'][:40]!r}")
        print(f"    total_B        = {gs['total_B']}")
        print(f"    top-1  % of B  = {gs['top1_contrib_pct']:.3f}%")
        print(f"    top-10 % of B  = {gs['top10_contrib_pct']:.3f}%")
        print(f"    top-100% of B  = {gs['top100_contrib_pct']:.3f}%")
        print(f"    k → 50% B      = {gs['k_covers_50pct']} dims")
        print(f"    k → 90% B      = {gs['k_covers_90pct']} dims")
        print(f"    k → 99% B      = {gs['k_covers_99pct']} dims")
        print()


if __name__ == "__main__":
    main()
