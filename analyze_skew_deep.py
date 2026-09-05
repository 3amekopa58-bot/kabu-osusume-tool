"""
「歪度」だけが4回の引き継ぎすべてで条件なしを上回った理由を精査する

`analyze_stock_habits_full.py` で14種類の性質を4回の引き継ぎで調べたところ、
4回すべてで向きが一致したのは ボラ／自己相関1／ジャンプ率／歪度／尖度 の5個。
しかしそのうち
  ・ジャンプ率はボラと順位相関 +0.98 ＝ 同じもの
  ・尖度・自己相関1は「最上位帯が条件なしを下回る回」がある
  ・ボラは最後の引き継ぎで最上位帯が条件なしと同じ
**歪度だけが、4回すべてで「最上位帯＞条件なし」かつ「最下位帯＜条件なし」**
だった。

⚠️ **多重検定**：14個も試せば偶然でも1〜2個は一致する（期待値1.75個）。
   歪度が本物かを確かめるには、偶然でどの程度起きるかを直接測るしかない。

⚠️ **歪度は持続性が +0.174 しかない**（ボラは+0.670）。
   性質として安定していないのに次の期間を予測できるのは不自然で、
   ノイズを拾っている可能性がある。これも確かめる。

調べること:
  ① 順列検定 … 銘柄ラベルをシャッフルしても同じことが起きるか
  ② 単調性   … 帯の切り方に依存しない形で効いているか
  ③ 独立性   … ボラ・時価総額で層別しても残るか（4.4-50の教訓）
  ④ 解釈     … 正の歪度＝たまに大きく上がる、が現行ルールと合うのか

使い方:
    python3 analyze_skew_deep.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
WINDOWS = [
    ("2000-2005", "2000-01-01", "2004-12-31"),
    ("2005-2010", "2005-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2014-12-31"),
    ("2015-2020", "2015-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
]
MIN_DAYS = 400
N_PERM = 500


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def rc(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank()) if m.sum() >= 30 else np.nan


def main():
    tr = load_trades()
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")

    raw = json.load(open(BASE_DIR / "data" / "fundamental_history.json",
                         encoding="utf-8"))
    fdata = raw.get("data", raw)
    shares = {}
    for code, recs in fdata.items():
        rows = [r for r in recs if r.get("shares")]
        if rows:
            rows.sort(key=lambda r: r.get("period_end", ""))
            shares[code] = float(rows[-1]["shares"])

    px = {}
    for code in codes:
        d = hist.get(code)
        if d is None or len(d) < MIN_DAYS:
            continue
        idx = d.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        px[code] = pd.Series(d["Close"].values, index=idx)

    # 窓ごとに 歪度・ボラ・時価総額
    T = {}
    for lab, lo, hi in WINDOWS:
        rows = {}
        for code, c in px.items():
            w = c[(c.index >= pd.Timestamp(lo)) & (c.index <= pd.Timestamp(hi))]
            r = w.pct_change().dropna()
            if len(r) < MIN_DAYS:
                continue
            sh = shares.get(code)
            rows[code] = {
                "歪度": r.skew(),
                "ボラ": r.std() * np.sqrt(252) * 100,
                "時価総額": float(w.iloc[-1]) * sh / 1e8 if sh else np.nan,
            }
        T[lab] = pd.DataFrame(rows).T
    trans = [(WINDOWS[i][0], WINDOWS[i+1][0], WINDOWS[i+1][1], WINDOWS[i+1][2])
             for i in range(len(WINDOWS) - 1)]

    # ---------- ① 順列検定 ----------
    print("=" * 72)
    print("① 順列検定：銘柄ラベルをシャッフルしても同じことが起きるか")
    print("=" * 72)
    print("  指標＝『最上位帯のPF − 最下位帯のPF』を4回ぶん合計した値\n")

    def spread_sum(assign):
        """assign: 窓ラベル -> (銘柄 -> 帯) の割り当て"""
        tot = 0.0
        for a, b, lo, hi in trans:
            band = assign[a]
            nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                     & (tr["entry_date"] <= pd.Timestamp(hi))]
            hi_ = nxt[nxt["code"].isin(set(band[band == "最大"].index))]["return_pct"]
            lo_ = nxt[nxt["code"].isin(set(band[band == "最小"].index))]["return_pct"]
            if len(hi_) < 40 or len(lo_) < 40:
                return np.nan
            tot += pf(hi_) - pf(lo_)
        return tot

    real = {}
    for a, _, _, _ in trans:
        s = T[a]["歪度"].dropna()
        real[a] = pd.qcut(s, 4, labels=["最小", "小", "大", "最大"])
    obs = spread_sum(real)

    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        shuf = {}
        for a, _, _, _ in trans:
            b = real[a]
            vals = b.values.copy()
            rng.shuffle(vals)
            shuf[a] = pd.Series(vals, index=b.index)
        v = spread_sum(shuf)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    p = (np.abs(null) >= abs(obs)).mean()
    print(f"  実測: {obs:+.3f}")
    print(f"  偶然（{len(null)}回シャッフル）: 平均{null.mean():+.3f} / "
          f"標準偏差{null.std():.3f} / 95%が {np.percentile(null,2.5):+.3f}〜"
          f"{np.percentile(null,97.5):+.3f}")
    print(f"  **p = {p:.3f}**（偶然でこれ以上の差がつく確率）")
    print(f"  → {'偶然では説明しにくい' if p < 0.05 else '**偶然の範囲**'}")
    print(f"\n  ⚠️ 14個の性質を試したので、p<0.05 でも "
          f"14×{p:.3f}≈{min(1,14*p):.2f} の確率で偶然に出る（多重検定）\n")

    # ---------- ② 単調性 ----------
    print("=" * 72)
    print("② 帯の切り方に依存しない形で効いているか（連続値の順位相関）")
    print("=" * 72)
    print("  前の窓の歪度の順位 と 次の窓のその銘柄の平均リターン の相関\n")
    for a, b, lo, hi in trans:
        s = T[a]["歪度"].dropna()
        nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                 & (tr["entry_date"] <= pd.Timestamp(hi))]
        g = nxt.groupby("code")["return_pct"].agg(["mean", "count"])
        g = g[g["count"] >= 3]
        j = pd.DataFrame({"歪度": s}).join(g, how="inner")
        print(f"  {a}→{b}: 銘柄{len(j):>4} / 順位相関 "
              f"{rc(j['歪度'], j['mean']):+.3f}")

    # ---------- ③ 独立性 ----------
    print("\n" + "=" * 72)
    print("③ ボラ・時価総額で層別しても残るか（4.4-50 の教訓）")
    print("=" * 72)
    for a, _, _, _ in trans:
        d = T[a][["歪度", "ボラ", "時価総額"]].dropna()
        print(f"  {a}: 歪度×ボラ {rc(d['歪度'], d['ボラ']):+.3f} / "
              f"歪度×時価総額 {rc(d['歪度'], d['時価総額']):+.3f}")
    print()
    for ctrl in ["ボラ", "時価総額"]:
        print(f"  --- {ctrl}で2分割した中での「歪度 最大帯 − 最小帯」のPF差 ---")
        print(f"    {'引き継ぎ':<24}{'小さい側':>10}{'大きい側':>10}")
        for a, b, lo, hi in trans:
            d = T[a][["歪度", ctrl]].dropna()
            d = d.copy()
            d["g"] = pd.qcut(d[ctrl], 2, labels=["小", "大"])
            nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                     & (tr["entry_date"] <= pd.Timestamp(hi))]
            out = []
            for gl in ["小", "大"]:
                sub = d[d["g"] == gl]
                bd = pd.qcut(sub["歪度"], 2, labels=["低", "高"])
                hi_ = nxt[nxt["code"].isin(set(bd[bd == "高"].index))]["return_pct"]
                lo_ = nxt[nxt["code"].isin(set(bd[bd == "低"].index))]["return_pct"]
                out.append(pf(hi_) - pf(lo_) if len(hi_) >= 40 and len(lo_) >= 40
                           else np.nan)
            print(f"    {a}→{b:<14}" + "".join(
                f"{v:>+10.2f}" if np.isfinite(v) else f"{'-':>10}" for v in out))
        print("    ※両側とも正なら、その要因とは独立に効いている\n")


if __name__ == "__main__":
    main()
