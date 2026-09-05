"""
決算発表をまたいで持つことの是非を検証する

現行の手仕舞いは**保有60日**。四半期決算は3か月ごとなので、
**どのトレードもほぼ必ず1回は決算発表をまたぐ**。決算はサプライズで
株価が大きく飛ぶ場面なのに、この点は一度も検証していなかった。

4.4-45 で PEAD（決算後のドリフト）が不採用になった理由は
「現行ルールがすでに同じもの（上がっている銘柄）を見ていたから」だった。
そこで今回は**現行ルールが見ていない軸**＝保有期間の設計そのものを狙う。

検証する3つ:
  A 現行           : 60日 or 損切り-10%（そのまま）
  B 決算前に手仕舞う : 保有中に決算発表が来たら、その前営業日の終値で降りる
  C 実態の把握      : 決算をまたいだトレードと、またがなかったトレードの比較

⚠️ 後知恵について：決算発表日そのものは事前に会社が予告するので、
   「次の決算がいつか」はエントリー時点で実務上わかる。ただしここで
   使っているのは J-Quants の**実際の開示日**なので、予定が動いた場合の
   ずれは再現できていない。厳密には僅かに後知恵が入る。

⚠️ J-Quants Standard は10年ぶんなので、26年13,976件のうち
   2016-09以降の分しか検証できない。

使い方:
    python3 analyze_earnings_hold.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
SUMMARY = BASE_DIR / "data" / "jquants_summary.json"
TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"
# 現行の保有上限。backtest.py は (当日 - エントリー日).days で数えるので暦日
HOLDING_LIMIT_DAYS = 60

SUBPERIODS = [
    ("全期間", "2016-09-01", "2030-01-01"),
    ("第1期 2016-09〜2020-01", "2016-09-01", "2020-01-31"),
    ("第2期 2020-02〜2023-05", "2020-02-01", "2023-05-31"),
    ("第3期 2023-06〜2026-09", "2023-06-01", "2026-09-30"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else float("inf")


def summarize(x):
    return {"件数": len(x), "勝率%": round((x > 0).mean() * 100, 1),
            "平均%": round(x.mean(), 2), "PF": round(pf(x), 2)}


def main():
    data = json.load(open(SUMMARY, encoding="utf-8"))["data"]
    codes = sorted(data.keys())
    hist = fetch_histories(codes, period="max")

    # 銘柄ごとの終値と、決算開示日の一覧
    closes, discs = {}, {}
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 300:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        closes[code] = c
        ds = sorted({r["DiscDate"] for r in data[code] if r.get("DiscDate")})
        discs[code] = pd.DatetimeIndex([pd.Timestamp(d) for d in ds])

    tr = load_trades(TRADES)
    tr = tr[tr["entry_date"] >= pd.Timestamp("2016-09-05")]

    rows = []
    for _, t in tr.iterrows():
        code = t["code"]
        c = closes.get(code)
        d = discs.get(code)
        if c is None or d is None or len(d) == 0:
            continue
        # ⚠️ 「またいだか」を**実際の手仕舞い日**で判定してはいけない。
        # 損切りで早く終わった悪いトレードは決算に到達しにくいので、
        # 「またいだ＝長く生き残った＝良かった」を測るだけになる（因果が逆）。
        # エントリー時点で決まる「今後60日（保有上限）以内に決算があるか」
        # で判定する。決算日程は事前に会社が予告するので実務上わかる。
        limit = t["entry_date"] + pd.Timedelta(days=HOLDING_LIMIT_DAYS)
        inside = d[(d > t["entry_date"]) & (d <= limit)]
        crossed = len(inside) > 0
        alt = np.nan
        # 実際の手仕舞いが決算より前なら（損切り等）、Bでも結果は変わらない
        if crossed and inside[0] < t["exit_date"]:
            # その前営業日の終値で降りる
            pos = c.index.searchsorted(inside[0])
            if pos >= 1:
                px = float(c.iloc[pos - 1])
                if np.isfinite(px) and px > 0:
                    alt = (px - t["entry_price"]) / t["entry_price"] * 100
        rows.append({
            "entry_date": t["entry_date"], "現行%": t["return_pct"],
            "またいだ": crossed, "決算前に降りた%": alt,
            "決算まで日数": (inside[0] - t["entry_date"]).days if crossed else np.nan,
            "保有日数": t["holding_days"],
        })

    df = pd.DataFrame(rows)
    n_cross = int(df["またいだ"].sum())
    print(f"検証対象のトレード: {len(df):,}件（2016-09以降）")
    print(f"  決算をまたいだ: {n_cross:,}件（{n_cross/len(df)*100:.1f}%）")
    print(f"  またがなかった: {len(df)-n_cross:,}件")
    print(f"  エントリーから決算までの日数の中央値: "
          f"{df['決算まで日数'].median():.0f}日")
    print(f"  ※「またいだ」は**エントリー時点で今後{HOLDING_LIMIT_DAYS}日以内に"
          f"決算があるか**で判定（実際の手仕舞い日では判定しない）\n")
    # 交絡が消えているかの確認：保有日数が両群で偏っていないか
    for flag, name in [(True, "決算あり"), (False, "決算なし")]:
        g = df[df["またいだ"] == flag]
        print(f"    {name}: 平均保有 {g['保有日数'].mean():.1f}日 / "
              f"損切り率 {(g['現行%'] <= -9.9).mean()*100:.1f}%")
    print()

    for label, lo, hi in SUBPERIODS:
        s = df[(df["entry_date"] >= pd.Timestamp(lo))
               & (df["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 200:
            print(f"=== {label} === 件数不足（{len(s)}件）\n")
            continue

        print(f"=== {label}（{len(s):,}件）===")

        # C 実態の把握：またいだ／またがなかったで分ける
        tbl = []
        for flag, name in [(True, "決算をまたいだ"), (False, "またがなかった")]:
            g = s[s["またいだ"] == flag]["現行%"]
            if len(g):
                tbl.append({"区分": name, **summarize(g)})
        tbl.append({"区分": "全体（現行）", **summarize(s["現行%"])})

        # B 決算前に手仕舞う：またいだ分だけ差し替え、またがない分はそのまま
        alt = s["現行%"].copy()
        m = s["またいだ"] & s["決算前に降りた%"].notna()
        alt[m] = s.loc[m, "決算前に降りた%"]
        tbl.append({"区分": "B 決算前に手仕舞う", **summarize(alt)})

        print(pd.DataFrame(tbl).set_index("区分").to_string())
        # 決算またぎがリスクを上げているか（損切りに当たる割合）
        for flag, name in [(True, "決算あり"), (False, "決算なし")]:
            g = s[s["またいだ"] == flag]["現行%"]
            if len(g):
                print(f"    {name}の損切り率: {(g <= -9.9).mean()*100:.1f}%",
                      end="   ")
        print("\n")

    print("⚠️ 重複しない3期間すべてで同じ向きに出ない限り採用しない。")
    print("   Bは手仕舞いを早めるので、保有日数が短くなる分も効いている点に注意。")


if __name__ == "__main__":
    main()
