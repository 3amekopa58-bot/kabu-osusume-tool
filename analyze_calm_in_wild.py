"""
「普段は荒い銘柄を、たまたま静かなときに買う」は効くか

4.4-51 で2つのことが分かった：
  ・銘柄について**持続する性質はボラティリティだけ**（順位相関+0.70/+0.64）
  ・**荒い銘柄ほど現行ルールの成績が良い**（時価総額とは独立）
一方 4.4-49 では
  ・**エントリー時点**の直近ボラは、低いほうが良い（ただし第2期で逆転）

この2つは変数が違うので矛盾しない。そこから
**「普段は荒い銘柄を、たまたま静かなときに買う」が最も良いのでは**
という仮説が出た。検証する。

⚠️ 4.4-51 の弱点は「性質→成績の引き継ぎが構造上2回しか作れず、
   その2回とも上昇相場だった」ことだった。ここでは
   **1トレードごとにエントリー時点で両方の指標を作る**ので、
   期間の引き継ぎに頼らず**26年・重複しない3期間**で確かめられる。
   下落相場（第1期 2000-2010、日経-41.6%）も入る。

指標の作り方（どちらもエントリー日までの株価だけで作る＝後知恵なし）:
  ・平常時の荒さ = エントリーの**60日前**を終点とする250日ボラ
      直近60日と重ならないようにずらす（重なると同じものを2回測る）
  ・直近の荒さ   = エントリー日までの60日ボラ
  ・静けさ比     = 直近 ÷ 平常時（1未満なら普段より静か）

使い方:
    python3 analyze_calm_in_wild.py
"""

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

CALM_WINDOW = 60      # 直近の荒さを測る日数
BASE_WINDOW = 250     # 平常時の荒さを測る日数
GAP = 60              # 平常時の窓を直近から何日ずらすか

SUBPERIODS = [
    ("全期間", "1999-01-01", "2030-01-01"),
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def main():
    tr = load_trades()
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")

    feats = {}
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < BASE_WINDOW + GAP + 50:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        r = c.pct_change()
        recent = r.rolling(CALM_WINDOW).std() * np.sqrt(252) * 100
        # 平常時は直近と重ならないように GAP 日ずらす
        base = (r.rolling(BASE_WINDOW).std() * np.sqrt(252) * 100).shift(GAP)
        feats[code] = pd.DataFrame({"直近": recent, "平常": base}, index=idx)

    rows = []
    for _, t in tr.iterrows():
        f = feats.get(t["code"])
        if f is None:
            continue
        pos = f.index.searchsorted(t["entry_date"])
        if pos >= len(f.index):
            continue
        r = f.iloc[pos]
        if not np.isfinite(r["直近"]) or not np.isfinite(r["平常"]) or r["平常"] <= 0:
            continue
        rows.append({"entry_date": t["entry_date"], "ret": t["return_pct"],
                     "平常": r["平常"], "直近": r["直近"],
                     "静けさ比": r["直近"] / r["平常"]})
    d = pd.DataFrame(rows)
    print(f"両方の指標を作れたトレード: {len(d):,}件 / 全{len(tr):,}件")
    print(f"静けさ比の中央値: {d['静けさ比'].median():.2f}"
          f"（1未満＝普段より静か）\n")
    print(f"平常の荒さ × 静けさ比 の順位相関: "
          f"{d['平常'].rank().corr(d['静けさ比'].rank()):+.3f}"
          f"（低いほど別の情報）\n")

    for plabel, lo, hi in SUBPERIODS:
        s = d[(d["entry_date"] >= pd.Timestamp(lo))
              & (d["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 500:
            continue
        s = s.copy()
        s["平常帯"] = pd.qcut(s["平常"], 2, labels=["穏やかな銘柄", "荒い銘柄"])
        s["静けさ帯"] = pd.qcut(s["静けさ比"], 2, labels=["普段より静か", "普段より荒い"])
        piv = s.pivot_table(index="平常帯", columns="静けさ帯", values="ret",
                            aggfunc=pf, observed=True).round(2)
        cnt = s.pivot_table(index="平常帯", columns="静けさ帯", values="ret",
                            aggfunc="size", observed=True)
        print(f"=== {plabel}（{len(s):,}件・条件なしPF {pf(s['ret']):.2f}）===")
        print(piv.to_string())
        print(f"  件数: {cnt.values.min()}〜{cnt.values.max()}")
        # 仮説の対象＝「荒い銘柄 × 普段より静か」
        cell = s[(s["平常帯"] == "荒い銘柄") & (s["静けさ帯"] == "普段より静か")]["ret"]
        print(f"  仮説の枠（荒い銘柄×普段より静か）: {len(cell)}件 "
              f"PF {pf(cell):.2f} / 勝率 {(cell>0).mean()*100:.1f}% "
              f"/ 平均 {cell.mean():+.2f}%")
        print()

    print("⚠️ 重複しない3期間すべてで同じマスが最良にならない限り採用しない。")


if __name__ == "__main__":
    main()
