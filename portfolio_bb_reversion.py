"""BB逆張りを、予算の制約を入れたポートフォリオで検証する。

⚠️ **トレード単位の成績は実行不可能な前提に立っている。**
   -2σだと年5,277件＝1日20件以上のシグナルが出るが、予算100万円では
   同時に10銘柄程度しか持てない。**大半のシグナルは実際には買えない。**
   4.4-71 でPBRがトレード単位では全関門を通りながらポートフォリオで
   落ちたのと同じ構図なので、こちらで判定する。

⚠️ 経路依存が大きいので複数シードの幅で比べる（6.6-2）。

使い方: ./venv/bin/python portfolio_bb_reversion.py [σ] [シード数]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
CAPITAL, LOT = 1_000_000, 100
MAX_POS = 4          # 同時に持つ銘柄数（予算100万円で現実的な上限）
TIME_STOP, STOP_LOSS = 60, -10.0
IMPACT_C, ORDER_YEN = 0.5, 1_000_000


def tick_size(p): return 1.0 if p < 3000 else (5.0 if p < 5000 else (10.0 if p < 30000 else 50.0))


def main() -> None:
    ns = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    t = pd.read_csv(B / "output" / f"bb_reversion_{ns}sigma.csv")
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    # 実質%は既にコスト控除済み。約定価格は entry_price 相当が無いので
    # リターンだけで資金を回す（保有中は資金が拘束される点を再現する）
    print(f"-{ns}σ / {len(t):,}トレード / シード{nseed}本")

    days = sorted(set(t["entry_date"]) | set(t["exit_date"]))
    by_entry = {d: g for d, g in t.groupby("entry_date")}
    results = []
    for seed in range(nseed):
        rng = np.random.default_rng(seed)
        cash, pos, peak, dd = CAPITAL, [], CAPITAL, 0.0
        for d in days:
            # 手仕舞い
            still = []
            for p in pos:
                if p["exit_date"] <= d:
                    cash += p["amount"] * (1 + p["ret"] / 100)
                else:
                    still.append(p)
            pos = still
            # 新規（同日に複数出たらランダムに選ぶ＝どれを選ぶかの恣意を避ける）
            g = by_entry.get(d)
            if g is not None and len(g):
                idx = rng.permutation(len(g))
                for k in idx:
                    if len(pos) >= MAX_POS:
                        break
                    r = g.iloc[k]
                    # ⚠️ 投入額を固定にすると、資産が下回った時点で二度と
                    #    約定しなくなる（2026-09-22に実際に踏んだ。2009年に
                    #    止まって17年間現金のままという結果が出ていた）。
                    #    現在の資産に対する割合で決める。
                    equity_now = cash + sum(p["amount"] for p in pos)
                    amt = equity_now / MAX_POS
                    if cash < amt or amt < 10_000:
                        break
                    cash -= amt
                    pos.append({"exit_date": r["exit_date"], "amount": amt,
                                "ret": r["実質%"]})
            equity = cash + sum(p["amount"] for p in pos)
            peak = max(peak, equity)
            dd = min(dd, equity / peak - 1)
        final = cash + sum(p["amount"] for p in pos)
        results.append({"seed": seed, "最終資産": final,
                        "リターン%": (final / CAPITAL - 1) * 100, "最大DD%": dd * 100})
    r = pd.DataFrame(results)
    print(f"\n=== 26年後の最終リターン ===")
    print(f"  中央値 {r['リターン%'].median():+,.1f}% / 最小 {r['リターン%'].min():+,.1f}% "
          f"/ 最大 {r['リターン%'].max():+,.1f}% / 幅 {r['リターン%'].max()-r['リターン%'].min():,.1f}pt")
    print(f"  最大ドローダウン 中央値 {r['最大DD%'].median():.1f}%")
    print(f"\n⚠️ 同じ期間の本命ルールは 4.4-71 の測定で中央値+94.1%（シード20本）。")
    print("   ただし選択ルールも建玉数も違うので厳密な比較ではない。")


if __name__ == "__main__":
    main()
