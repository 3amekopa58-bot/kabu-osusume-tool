"""
売買代金は「時価総額（小型株）」の言い換えかを確かめる

4.4-49 で、売買代金が小さいほど成績が良いことが
**3期間すべてで単調**に出た。単調・後知恵なし・現実的なコストに耐える、
という条件のそろった候補だが、**採用は保留した**。理由は

  4.4-25 で採用済みの【小型】印（時価総額300億円未満）と
  同じものを見ている可能性があるから

4.4-45 で PEAD が不採用になったのは「現行ルールがすでに同じもの
（上がっている銘柄）を見ていたから」だった。**独立性を確かめずに
足すと同じ轍を踏む。**

#### 時価総額の復元のしかた（分割への対処）

株価は分割で遡及調整されているので、当時の発行済株式数を掛けると
**分割した銘柄の時価総額を過小評価する**。正しくは

    時価総額(t) ＝ 調整後株価(t) × **現在の**発行済株式数

分割は株価を1/Sにし株式数をS倍にするので、両者が打ち消し合って
当時の時価総額が正しく出る（4.4-49 で低位株効果が後知恵だと分かった
のと同じ理屈を、逆向きに使う）。

⚠️ ただし**増資・自社株買いのぶんは先読みが残る**。後から大量に増資した
   会社は過去の時価総額が過大に出る。独立性を見るには十分だが、
   これを実運用のルールに使ってはいけない。

使い方:
    python3 analyze_turnover_vs_size.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
FUNDAMENTALS = BASE_DIR / "data" / "fundamental_history.json"
EDINET = BASE_DIR / "data" / "edinet_financials.json"

SUBPERIODS = [
    ("全期間", "1999-01-01", "2030-01-01"),
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else float("inf")


def latest_shares() -> dict:
    """銘柄ごとの最新の発行済株式数。2つのデータ源を突き合わせる。"""
    out = {}
    for path, getter in [
        (FUNDAMENTALS, lambda v: v),
        (EDINET, lambda v: list(v.values())),
    ]:
        if not path.exists():
            continue
        raw = json.load(open(path, encoding="utf-8"))
        data = raw.get("data", raw)
        for code, recs in data.items():
            try:
                rows = getter(recs)
                rows = [r for r in rows if r.get("shares")]
                if not rows:
                    continue
                rows.sort(key=lambda r: r.get("period_end", ""))
                out.setdefault(code, float(rows[-1]["shares"]))
            except Exception:
                continue
    return out


def main():
    shares = latest_shares()
    print(f"発行済株式数が取れた銘柄: {len(shares)}")

    tr = load_trades()
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")

    turn = {}
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
        turn[code] = pd.DataFrame(
            {"turnover": (c * v).rolling(20).mean() / 1e8}, index=idx)

    rows = []
    for _, t in tr.iterrows():
        f = turn.get(t["code"])
        sh = shares.get(t["code"])
        if f is None or not sh:
            continue
        pos = f.index.searchsorted(t["entry_date"])
        if pos >= len(f.index):
            continue
        rows.append({
            "entry_date": t["entry_date"], "ret": t["return_pct"],
            "turnover": float(f.iloc[pos]["turnover"]),
            # 時価総額（億円）＝ 調整後株価 × 現在の株式数
            "cap": t["entry_price"] * sh / 1e8,
        })
    d = pd.DataFrame(rows).dropna()
    print(f"両方そろったトレード: {len(d):,}件 "
          f"（全{len(tr):,}件）\n")
    # どちらも分布が極端に歪んでいるので、素の相関は外れ値に支配される。
    # 順位に直してから測る（scipyを入れずに済むよう rank + 相関で計算）
    rank_corr = d["turnover"].rank().corr(d["cap"].rank())
    print(f"売買代金と時価総額の相関: 素の値 {d['turnover'].corr(d['cap']):+.3f}"
          f" / **順位相関 {rank_corr:+.3f}**")
    print("  （分布が歪むので順位相関のほうが実態に近い）\n")

    print("=== 時価総額の帯ごとに、売買代金で4分割したPF ===")
    print("どの時価総額帯でも売買代金が効くなら、別々の情報を持っている\n")
    for plabel, lo, hi in SUBPERIODS:
        s = d[(d["entry_date"] >= pd.Timestamp(lo))
              & (d["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 400:
            continue
        s = s.copy()
        s["時価総額帯"] = pd.qcut(s["cap"], 4,
                              labels=["小", "やや小", "やや大", "大"])
        s["代金帯"] = pd.qcut(s["turnover"], 4,
                           labels=["最小", "小", "大", "最大"])
        piv = s.pivot_table(index="時価総額帯", columns="代金帯",
                            values="ret", aggfunc=pf, observed=True).round(2)
        cnt = s.pivot_table(index="時価総額帯", columns="代金帯",
                            values="ret", aggfunc="size", observed=True)
        print(f"--- {plabel}（{len(s):,}件・条件なしPF {pf(s['ret']):.2f}）---")
        print(piv.to_string())
        print(f"  各マスの件数: 最小 {int(cnt.min().min())}〜最大 "
              f"{int(cnt.max().max())}件")
        print()

    print("=== それぞれ単独で見たPF（全期間・最小→最大）===")
    for col, name in [("turnover", "売買代金"), ("cap", "時価総額")]:
        b = pd.qcut(d[col], 4, labels=["最小", "小", "大", "最大"])
        vals = [pf(d[b == k]["ret"]) for k in ["最小", "小", "大", "最大"]]
        print(f"  {name:<8} " + " ".join(f"{v:5.2f}" for v in vals))
    print(f"  条件なし: {pf(d['ret']):.2f}")


if __name__ == "__main__":
    main()
