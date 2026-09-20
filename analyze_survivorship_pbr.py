"""生存者バイアスがPBRの結論（4.4-65/67）を覆しうるかを最悪ケースで確かめる。

⚠️ 上場廃止銘柄の株価はyfinanceで取得できない（4.4-9で6銘柄、
   2026-09-20に無作為60銘柄で再確認＝取得できたのは1件のみ）。
   **欠落している銘柄を実際に混ぜることはできない。**
   そこで「最も結論に不利な形で欠落していたら」を仮定して、
   それでも結論が保つかを見る（感度分析）。

最悪ケースの置き方:
  ・倒産（株価ゼロ）した欠落トレードは**すべて割安側(Q1)**にあったと仮定
  ・買収プレミアムを得た欠落トレードは**すべて割高側(Q4)**にあったと仮定
  （実際は割安株ほど買収されやすいので、これは現実より不利な仮定）

使い方: ./venv/bin/python analyze_survivorship_pbr.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent

# 4.4-9 の実測（JPX上場廃止一覧 2017-2025、643件）
DELIST_RATE_Y = 0.0187      # 年間の上場廃止率
BANKRUPT_SHARE = 0.017      # うち経営破綻
ACQUIRE_SHARE = 0.956       # うち買収・MBO・株式併合
HOLD_DAYS = 51              # 採用ルールの平均保有日数
TOB_PREMIUM = 30.0          # 買収プレミアム（%）。日本のTOBは30%前後が目安


def main() -> None:
    u = pd.read_csv(B / "output" / "trades_with_cost.csv")
    u["entry_date"] = pd.to_datetime(u["entry_date"])
    f = pd.read_csv(B / "output" / "segment_features.csv")
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    u = u.merge(f[["code", "entry_date", "PBR"]], on=["code", "entry_date"], how="left")
    p = u[u["PBR"].notna() & (u["PBR"] > 0)].copy()
    p["実質"] = p["return_pct"] - (p["スプレッド%"] + p["インパクト%"])
    p["Q"] = pd.qcut(p["PBR"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    n = len(p)
    per_trade = DELIST_RATE_Y * HOLD_DAYS / 365      # 1トレードが廃止に遭う確率
    n_bank = n * per_trade * BANKRUPT_SHARE
    n_acq = n * per_trade * ACQUIRE_SHARE
    print(f"PBR検証の対象 {n:,}トレード（{p['entry_date'].dt.year.min()}"
          f"〜{p['entry_date'].dt.year.max()}年）")
    print(f"1トレードが上場廃止に遭う確率 {per_trade*100:.4f}%"
          f"（年率{DELIST_RATE_Y*100:.2f}% × 保有{HOLD_DAYS}日）\n")
    print(f"欠落していたはずの件数（推定）:")
    print(f"  経営破綻 {n_bank:.1f}件 / 買収 {n_acq:.1f}件 "
          f"（合計 {n_bank+n_acq:.1f}件＝全体の {(n_bank+n_acq)/n*100:.2f}%）\n")

    q1 = p[p["Q"] == "Q1"]["実質"]
    q4 = p[p["Q"] == "Q4"]["実質"]
    obs = q1.median() - q4.median()
    print(f"現状の Q1−Q4（実質・中央値）: {obs:+.2f}pt")
    print(f"  Q1 {q1.median():+.2f}%（{len(q1)}件） / Q4 {q4.median():+.2f}%（{len(q4)}件）\n")

    # --- 最悪ケース：破綻はすべてQ1、買収プレミアムはすべてQ4 ---
    print("=== 最悪ケース：破綻をすべて割安側に、買収益をすべて割高側に寄せる ===")
    q1w = np.concatenate([q1.values, np.full(int(round(n_bank)), -100.0)])
    q4w = np.concatenate([q4.values, np.full(int(round(n_acq)), TOB_PREMIUM)])
    worst = np.median(q1w) - np.median(q4w)
    print(f"  Q1に破綻{int(round(n_bank))}件(-100%)を追加 → 中央値 {np.median(q1w):+.2f}%")
    print(f"  Q4に買収{int(round(n_acq))}件(+{TOB_PREMIUM:.0f}%)を追加 → 中央値 {np.median(q4w):+.2f}%")
    print(f"  Q1−Q4 = {worst:+.2f}pt（現状 {obs:+.2f}pt）")
    print(f"  → 結論は{'**保つ**' if worst > 0 else '**覆る**'}")

    # --- どこまで積めば覆るか ---
    print("\n=== 何件積めば結論が覆るか（Q1に-100%を足し続ける）===")
    for k in [10, 25, 50, 100, 200, 340]:
        v = np.concatenate([q1.values, np.full(k, -100.0)])
        d = np.median(v) - q4.median()
        flag = "→ **ここで逆転**" if d <= 0 else ""
        print(f"  Q1に{k:>4}件(-100%)追加: Q1中央値 {np.median(v):+.2f}% / "
              f"Q1−Q4 {d:+.2f}pt {flag}")
    print(f"\n  ※推定される実際の破綻欠落は {n_bank:.1f}件。"
          f"上の表と比べて桁が合っているかを見ること。")

    # --- 2017年より前（JPXアーカイブが無い時代）の比率 ---
    pre = (p["entry_date"].dt.year < 2017).mean() * 100
    print(f"\n⚠️ PBR検証のうち2017年より前（廃止理由が未検証の時代）: {pre:.1f}%")


if __name__ == "__main__":
    main()
