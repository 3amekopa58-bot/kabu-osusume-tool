"""月別の季節性（上がりやすい月・下がりやすい月）を検証する。

⚠️ 月別は多重検定の罠が大きい。12ヶ月×複数指標なら偶然でも「良い月」が
   いくつも出る。採用基準どおり**重複しない3期間すべてで同じ向き**かを
   確認しないと、過去をまとめて見ただけの結果論になる（4.4-5 の業種と同じ型）。

使い方: ./venv/bin/python analyze_month_seasonality.py
出力  : output/month_seasonality.csv
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent
PERIODS = [("第1期 2000-2008", 2000, 2008),
           ("第2期 2009-2017", 2009, 2017),
           ("第3期 2018-2026", 2018, 2026)]


def nikkei_monthly() -> pd.DataFrame:
    """日経平均の月次騰落率。"""
    h = yf.Ticker("^N225").history(start="1999-12-01", end="2026-09-30")
    m = h["Close"].resample("ME").last()
    d = (m.pct_change() * 100).dropna().to_frame("騰落%")
    d["年"] = d.index.year
    d["月"] = d.index.month
    return d


def main() -> None:
    d = nikkei_monthly()
    print(f"日経平均 月次 {len(d)}ヶ月分（{d.index[0]:%Y-%m}〜{d.index[-1]:%Y-%m}）\n")

    print("=== 月別の平均騰落率（全期間）===")
    allm = d.groupby("月")["騰落%"].agg(["mean", "median", "count"])
    allm["勝率%"] = d.groupby("月")["騰落%"].apply(lambda s: (s > 0).mean() * 100)
    print(allm.to_string(float_format="%.2f"))

    print("\n=== 重複しない3期間で向きが揃うか ===")
    rows = {}
    for lab, a, b in PERIODS:
        s = d[(d["年"] >= a) & (d["年"] <= b)]
        rows[lab] = s.groupby("月")["騰落%"].mean()
    t = pd.DataFrame(rows)
    t["全期間"] = allm["mean"]
    t["3期間とも+"] = (t[[p[0] for p in PERIODS]] > 0).all(axis=1)
    t["3期間とも-"] = (t[[p[0] for p in PERIODS]] < 0).all(axis=1)
    print(t.to_string(float_format="%.2f"))

    up = list(t[t["3期間とも+"]].index)
    dn = list(t[t["3期間とも-"]].index)
    print(f"\n3期間すべてプラス: {up if up else 'なし'}")
    print(f"3期間すべてマイナス: {dn if dn else 'なし'}")
    # ⚠️ 帰無仮説を「符号は五分五分」にしてはいけない。26年の日経は上昇基調で
    #    プラスの月がもともと多いため、偶然でも揃いやすい。各期間の実際の
    #    プラス率を使って期待個数を出す。
    ps = [(t[lab] > 0).mean() for lab, _, _ in PERIODS]
    exp_up = 12 * (ps[0] * ps[1] * ps[2])
    exp_dn = 12 * ((1 - ps[0]) * (1 - ps[1]) * (1 - ps[2]))
    print("\n=== 偶然でもどれだけ揃うか（各期間の実際のプラス率から）===")
    for (lab, _, _), q in zip(PERIODS, ps):
        print(f"  {lab}: 12ヶ月中プラスが {q*12:.0f}ヶ月（{q*100:.0f}%）")
    print(f"  → 3期間ともプラスが揃う月の期待個数 {exp_up:.1f}個（観測 {len(up)}個）")
    print(f"  → 3期間ともマイナスが揃う月の期待個数 {exp_dn:.1f}個（観測 {len(dn)}個）")
    if len(up) <= exp_up + 1:
        print("\n  **観測は期待値とほぼ同じ＝偶然と区別がつかない。**")

    t.to_csv(B / "output" / "month_seasonality.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
