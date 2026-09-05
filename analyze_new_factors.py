"""
未検証だった4つの軸を、現行ルールのトレードの中で検証する

4.4-45 の棚卸しで「データは手元にあるのに未検証」と分かった4つ：

  ① 長期モメンタム（12か月）
     現行の相対力は**50日**だけ。古典的な12か月モメンタムは見ていない
  ② 移動平均からの乖離率
     `ものわかれ` は乖離の**縮小**を見ており、乖離の**大きさ**は別物
  ③ ボラティリティ（低ボラ効果）
     ボラは暴落の予兆としてしか測っていない。銘柄選択の軸としては未検証
  ④ 売買代金（流動性）
     ユニバースの絞り込みには使っているが、ファクターとしては未検証

⚠️ 4.4-45 の教訓に従い、**最初から現行ルールが出したトレードの中で**測る。
   全銘柄で有意でも、現行ルール通過銘柄の中で効かなければ足す意味がない
   （PEADはそれで不採用になった）。

⚠️ J-Quantsを使わないので**26年13,976件すべて**が対象。
   決算まわりの検証（10年）より土台が厚い。

後知恵の排除：どの指標もエントリー日**まで**の株価から作る。

⚠️ **株価の水準そのものを指標にしてはいけない（2026-09-05に判明）。**
   `price_cache.py` は `auto_adjust=True` で取得しており、株価は分割で
   **遡及調整**される。分割は株価が上がった後に行われるので、
   「過去の調整後株価が安い」＝「その後大きく上がった」となる。
   実測：調整後株価が下位25%のトレードは**73.8%**がエントリー後に分割
   （上位25%は23.2%）。低位株効果はこの後知恵の産物だった。
   一方 **売買代金＝株価×出来高 は分割倍率が打ち消し合って不変**なので
   安全（分割日をまたいだ比が約1.00であることを実データで確認済み）。

使い方:
    python3 analyze_new_factors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"

# 重複しない3期間（他の節と同じ区切り）
SUBPERIODS = [
    ("全期間", "1999-01-01", "2030-01-01"),
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]

FACTORS = [
    ("mom12", "① 12か月モメンタム"),
    ("dev25", "② 25日線からの乖離率"),
    ("vol60", "③ ボラティリティ(60日)"),
    ("turnover", "④ 売買代金(20日平均)"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else float("inf")


def main():
    tr = load_trades(TRADES)
    codes = sorted(tr["code"].unique())
    print(f"トレード: {len(tr):,}件 / {len(codes)}銘柄")

    hist = fetch_histories(codes, period="max")

    feats = {}
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 300:
            continue
        c, v = h["Close"], h["Volume"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
            v = pd.Series(v.values, index=idx)
        ma25 = c.rolling(25).mean()
        ret = c.pct_change()
        feats[code] = pd.DataFrame({
            # 12か月モメンタム（250営業日）
            "mom12": c / c.shift(250) - 1,
            # 25日線からの乖離率（％）
            "dev25": (c - ma25) / ma25 * 100,
            # 年率ボラティリティ（％）
            "vol60": ret.rolling(60).std() * np.sqrt(252) * 100,
            # 売買代金（億円）
            "turnover": (c * v).rolling(20).mean() / 1e8,
        }, index=idx)

    rows = []
    for _, t in tr.iterrows():
        f = feats.get(t["code"])
        if f is None:
            continue
        pos = f.index.searchsorted(t["entry_date"])
        if pos >= len(f.index):
            continue
        # エントリー日「まで」の値を使う（当日を含む）
        r = f.iloc[pos]
        rows.append({"entry_date": t["entry_date"],
                     "return_pct": t["return_pct"],
                     **{k: r[k] for k, _ in FACTORS}})
    d = pd.DataFrame(rows)
    print(f"指標を作れたトレード: {len(d):,}件\n")

    for key, label in FACTORS:
        print(f"===== {label} =====")
        for plabel, lo, hi in SUBPERIODS:
            s = d[(d["entry_date"] >= pd.Timestamp(lo))
                  & (d["entry_date"] <= pd.Timestamp(hi))].dropna(subset=[key])
            if len(s) < 300:
                print(f"--- {plabel} --- 件数不足（{len(s)}件）")
                continue
            s = s.copy()
            s["帯"] = pd.qcut(s[key], 4, labels=["最小", "小", "大", "最大"],
                             duplicates="drop")
            out = []
            for b, g in s.groupby("帯", observed=True):
                out.append({"帯": b, "件数": len(g),
                            "中央値": round(g[key].median(), 2),
                            "勝率%": round((g["return_pct"] > 0).mean() * 100, 1),
                            "平均%": round(g["return_pct"].mean(), 2),
                            "PF": round(pf(g["return_pct"]), 2)})
            df_out = pd.DataFrame(out).set_index("帯")
            print(f"--- {plabel}（{len(s):,}件・条件なしPF "
                  f"{pf(s['return_pct']):.2f}）---")
            print(df_out.to_string())
        print()

    print("⚠️ 重複しない3期間すべてで同じ向きに出ない限り採用しない。")
    print("   『全期間だけ効く』は入れ子の期間で見ているのと同じで根拠にならない。")


if __name__ == "__main__":
    main()
