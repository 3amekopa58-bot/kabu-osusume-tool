"""
銘柄ごとの「チャートのくせ」は実在するか、そして使えるか

うねり取り（第3章）は銘柄選定について、かなり具体的に書いている：

  ・**大化け（急騰）する銘柄は避ける。むしろ困る。**
    同じ価格帯で上がったり下がったりを繰り返す銘柄でないとできない
  ・ニュース・材料に反応しない、**自律的な周期性の高い銘柄**がよい
  ・複数銘柄に手を広げず、**1銘柄に絞って専門化する**

つまり「銘柄ごとにクセがあり、それは持続する」という前提に立っている。
このツールは全銘柄を機械的に同じ扱いにしているので、この前提が正しければ
**銘柄を選ぶ余地がある**ことになる。

⚠️ **4.4-5 に先例がある。** 業種レベルでは「クセは実在するが将来には
   使えない」と判明した（年代間の順位相関が -0.30 / -0.14 と**逆相関**。
   ある年代で良かった業種は次の年代でむしろ悪くなる）。
   銘柄レベルでも同じ結末になる可能性が高いが、未検証なので確かめる。

2段階で調べる:

  ① **過去の成績**に持続性はあるか
     前半で良かった銘柄は後半も良いか。4.4-5 の銘柄版。
     ⚠️ 1銘柄あたり26年で約15トレードしかなく、半分に割ると約7件。
        偶然でどの程度ばらつくかを**順列検定**で出して比べる。

  ② **チャートの性質**に持続性はあるか、そしてそれは成績を予測するか
     成績そのものではなく、うねり取りが言う「値動きの質」を測る:
       ・効率比 = |期間の正味変化| ÷ |日々の変化の合計|
         低いほど「行ったり来たり」＝うねり取り向き
       ・1日リターンの自己相関（負なら平均回帰＝押し目が効きやすい）
       ・ボラティリティ
     性質が持続しても成績を予測しなければ使えない。両方見る。

使い方:
    python3 analyze_stock_habits.py
"""

import numpy as np
import pandas as pd

from pathlib import Path

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
SPLIT = pd.Timestamp("2013-06-30")   # 26年をほぼ半分に割る
MIN_TRADES = 5                       # 片側でこの件数未満の銘柄は使わない
N_PERM = 200                         # 順列検定の試行回数


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def rank_corr(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    tr = load_trades()
    first = tr[tr["entry_date"] < SPLIT]
    second = tr[tr["entry_date"] >= SPLIT]
    print(f"前半 {first['entry_date'].min().date()}〜{SPLIT.date()}: "
          f"{len(first):,}件")
    print(f"後半 {SPLIT.date()}〜{second['entry_date'].max().date()}: "
          f"{len(second):,}件\n")

    a = first.groupby("code")["return_pct"].agg(["mean", "count"])
    b = second.groupby("code")["return_pct"].agg(["mean", "count"])
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j["count_1"] >= MIN_TRADES) & (j["count_2"] >= MIN_TRADES)]
    print(f"=== ① 過去の成績に持続性はあるか ===")
    print(f"前後半とも{MIN_TRADES}件以上ある銘柄: {len(j)}銘柄")
    print(f"  1銘柄あたり 前半{j['count_1'].median():.0f}件 / "
          f"後半{j['count_2'].median():.0f}件（中央値）\n")

    obs = rank_corr(j["mean_1"], j["mean_2"])
    # 順列検定：後半の成績を銘柄間でシャッフルしたときの相関の分布
    rng = np.random.default_rng(0)
    null = []
    v2 = j["mean_2"].values.copy()
    for _ in range(N_PERM):
        rng.shuffle(v2)
        null.append(rank_corr(j["mean_1"].values, v2))
    null = np.array(null)
    print(f"  前半平均リターン → 後半平均リターン の順位相関: **{obs:+.3f}**")
    print(f"  偶然だとどうなるか（{N_PERM}回シャッフル）: "
          f"平均{null.mean():+.3f} / 95%が {np.percentile(null,2.5):+.3f}〜"
          f"{np.percentile(null,97.5):+.3f} の範囲")
    verdict = "偶然の範囲" if abs(obs) <= np.percentile(np.abs(null), 95) else "偶然では説明できない"
    print(f"  → **{verdict}**\n")

    # 前半の成績で4分割 → 後半の成績
    j["帯"] = pd.qcut(j["mean_1"], 4, labels=["最下位", "下位", "上位", "最上位"])
    rows = []
    for band, g in j.groupby("帯", observed=True):
        codes = set(g.index)
        s2 = second[second["code"].isin(codes)]["return_pct"]
        rows.append({"前半の成績": band, "銘柄数": len(g),
                     "前半の平均%": round(g["mean_1"].mean(), 2),
                     "後半の平均%": round(s2.mean(), 2),
                     "後半のPF": round(pf(s2), 2), "後半の件数": len(s2)})
    print("  前半の成績で分けて、後半どうだったか：")
    print(pd.DataFrame(rows).set_index("前半の成績").to_string())
    print(f"  （後半の全体: 平均{second['return_pct'].mean():+.2f}% / "
          f"PF{pf(second['return_pct']):.2f}）\n")

    # ---------- ② チャートの性質 ----------
    print("=== ② チャートの性質は持続するか、成績を予測するか ===\n")
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")

    def traits(c):
        """効率比・自己相関・ボラティリティ"""
        r = c.pct_change().dropna()
        if len(r) < 200:
            return None
        net = abs(c.iloc[-1] - c.iloc[0])
        path = c.diff().abs().sum()
        return {
            "効率比": net / path if path else np.nan,
            "自己相関": r.autocorr(lag=1),
            "ボラ": r.std() * np.sqrt(252) * 100,
        }

    rows = []
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 500:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        t1 = traits(c[c.index < SPLIT])
        t2 = traits(c[c.index >= SPLIT])
        if not t1 or not t2:
            continue
        row = {"code": code}
        row.update({f"{k}_1": v for k, v in t1.items()})
        row.update({f"{k}_2": v for k, v in t2.items()})
        rows.append(row)
    t = pd.DataFrame(rows).set_index("code").dropna()
    print(f"  性質を測れた銘柄: {len(t)}\n")
    print("  性質そのものの持続性（前半 → 後半の順位相関）:")
    for k in ["効率比", "自己相関", "ボラ"]:
        print(f"    {k:<8} {rank_corr(t[f'{k}_1'], t[f'{k}_2']):+.3f}")
    print("    ※高ければ「その銘柄の性質」として安定している\n")

    print("  前半の性質で分けて、後半の成績はどうだったか：")
    for k in ["効率比", "自己相関", "ボラ"]:
        t2 = t.copy()
        t2["帯"] = pd.qcut(t2[f"{k}_1"], 4, labels=["最小", "小", "大", "最大"])
        out = []
        for band, g in t2.groupby("帯", observed=True):
            s2 = second[second["code"].isin(set(g.index))]["return_pct"]
            if len(s2) < 50:
                continue
            out.append({"帯": band, "銘柄数": len(g), "後半の件数": len(s2),
                        "後半の平均%": round(s2.mean(), 2),
                        "後半のPF": round(pf(s2), 2)})
        print(f"\n    --- 前半の{k} ---")
        print(pd.DataFrame(out).set_index("帯").to_string())
    print(f"\n  （後半の全体: 平均{second['return_pct'].mean():+.2f}% / "
          f"PF{pf(second['return_pct']):.2f}）")
    print("\n⚠️ 前半の性質で分けて後半の成績に差がつかなければ、"
          "『くせ』は観測できても使えない（4.4-5 の業種と同じ結末）。\n")

    multi_period(tr, cl_from(hist, codes))


def cl_from(hist, codes):
    out = {}
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 500:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        out[code] = c
    return out


def multi_period(tr, cl):
    """
    1回の分割では偶然の可能性が残るので、重複しない3期間の
    「前の期間の性質 → 次の期間の成績」を2回ぶん見る。
    さらに、4.4-50 の教訓（売買代金は時価総額の言い換えだった）を踏まえ、
    **時価総額で層別しても効果が残るか**まで確かめる。
    """
    import json
    P = [("第1期", "2000-01-01", "2010-03-31"),
         ("第2期", "2010-04-01", "2018-01-31"),
         ("第3期", "2018-02-01", "2026-12-31")]

    raw = json.load(open(BASE_DIR / "data" / "fundamental_history.json",
                         encoding="utf-8"))
    data = raw.get("data", raw)
    shares = {}
    for code, recs in data.items():
        rows = [r for r in recs if r.get("shares")]
        if rows:
            rows.sort(key=lambda r: r.get("period_end", ""))
            shares[code] = float(rows[-1]["shares"])

    def period_traits(lo, hi):
        rows = {}
        for code, c in cl.items():
            s = c[(c.index >= pd.Timestamp(lo)) & (c.index <= pd.Timestamp(hi))]
            r = s.pct_change().dropna()
            if len(r) < 200:
                continue
            net = abs(s.iloc[-1] - s.iloc[0])
            path = s.diff().abs().sum()
            sh = shares.get(code)
            rows[code] = {
                "効率比": net / path if path else np.nan,
                "ボラ": r.std() * np.sqrt(252) * 100,
                # 時価総額＝調整後株価×現在の株式数（分割が打ち消し合う。4.4-50）
                "時価総額": float(s.iloc[-1]) * sh / 1e8 if sh else np.nan,
            }
        return pd.DataFrame(rows).T

    T = {lab: period_traits(lo, hi) for lab, lo, hi in P}

    print("=== ③ 3期間で確かめる（1回の分割では偶然が残るため）===\n")
    print("  性質の持続性（連続する2期間の順位相関）")
    print(f"    {'指標':<8}{'第1期→第2期':>14}{'第2期→第3期':>14}")
    for k in ["効率比", "ボラ"]:
        v = []
        for a, b in [("第1期", "第2期"), ("第2期", "第3期")]:
            j = T[a][[k]].join(T[b][[k]], lsuffix="_a", rsuffix="_b").dropna()
            v.append(rank_corr(j[f"{k}_a"], j[f"{k}_b"]))
        print(f"    {k:<8}{v[0]:>+14.3f}{v[1]:>+14.3f}")

    print("\n  前の期間の性質で分けて、次の期間のPF")
    for k in ["効率比", "ボラ"]:
        print(f"\n    --- {k} ---")
        print(f"    {'引き継ぎ':<14}{'最小':>7}{'小':>7}{'大':>7}{'最大':>7}{'条件なし':>9}")
        for a, b, lo, hi in [("第1期", "第2期", "2010-04-01", "2018-01-31"),
                             ("第2期", "第3期", "2018-02-01", "2026-12-31")]:
            src = T[a][k].dropna()
            band = pd.qcut(src, 4, labels=["最小", "小", "大", "最大"])
            nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                     & (tr["entry_date"] <= pd.Timestamp(hi))]
            line = f"    {a}→{b:<9}"
            for lab2 in ["最小", "小", "大", "最大"]:
                s = nxt[nxt["code"].isin(set(band[band == lab2].index))]["return_pct"]
                line += f"{pf(s):>7.2f}" if len(s) >= 50 else f"{'-':>7}"
            print(line + f"{pf(nxt['return_pct']):>9.2f}")

    print("\n  ⚠️ 時価総額の言い換えでないかの確認（4.4-50の教訓）")
    for a, b, lo, hi in [("第1期", "第2期", "2010-04-01", "2018-01-31"),
                         ("第2期", "第3期", "2018-02-01", "2026-12-31")]:
        d = T[a][["ボラ", "時価総額"]].dropna()
        nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                 & (tr["entry_date"] <= pd.Timestamp(hi))]
        print(f"\n    {a}→{b}  ボラ×時価総額の順位相関 "
              f"{rank_corr(d['ボラ'], d['時価総額']):+.3f}"
              f" / 次期間の条件なしPF {pf(nxt['return_pct']):.2f}")
        d = d.copy()
        d["cap"] = pd.qcut(d["時価総額"], 2, labels=["小型", "大型"])
        d["vol"] = pd.qcut(d["ボラ"], 2, labels=["穏やか", "荒い"])
        print(f"      {'':<8}{'穏やか':>8}{'荒い':>8}")
        for capb in ["小型", "大型"]:
            line = f"      {capb:<8}"
            for volb in ["穏やか", "荒い"]:
                cs = set(d[(d["cap"] == capb) & (d["vol"] == volb)].index)
                s = nxt[nxt["code"].isin(cs)]["return_pct"]
                line += f"{pf(s):>8.2f}" if len(s) >= 50 else f"{'-':>8}"
            print(line)
    print("\n  ⚠️ どちらのサイズでも同じ向きに出れば、時価総額とは別の情報。")


if __name__ == "__main__":
    main()
