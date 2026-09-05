"""
銘柄ごとの「チャートのくせ」を徹底的に調べる

4.4-51／4.4-52 で3つの性質（効率比・自己相関・ボラ）を調べたが、
**構造的な弱点が2つ**残っていた：

  ① 引き継ぎが2回しか作れず、その2回とも上昇相場だった
  ② 調べた性質が3つだけだった

ここでは
  ① **5年窓で4回の引き継ぎ**を作る。うち起点2つ・対象1つが下落局面
       2000-2005 -39.5% / 2005-2010 -8.4% / 2010-2015 +63.8%
       2015-2020 +35.9% / 2020-2026 +185.4%
  ② 性質を **14種類** に広げる（材料反応・平均回帰速度・個別要因など）

各性質について**3点セット**で測る：
  (a) 持続性  … 前の窓と次の窓で順位相関（性質として安定しているか）
  (b) 予測力  … 前の窓の性質で4分割 → 次の窓のPF
  (c) 重複    … 他の性質・時価総額・売買代金との順位相関
                 （4.4-50 の教訓。素の相関ではなく順位相関で見る）

⚠️ 後知恵の排除：性質は**前の窓の株価だけ**から作り、成績は**次の窓**で
   測るので、判断時点で知り得ない情報は入らない。

⚠️ 採用基準：**4回の引き継ぎすべてで同じ向き**に出ること。
   1回でも逆なら不採用（4.4-52 で、上昇相場だけで見ると逆の結論が出ると
   分かったため）。

使い方:
    python3 analyze_stock_habits_full.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fetch_nikkei_close
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

MIN_DAYS = 400        # 窓の中でこの日数未満の銘柄は性質を作らない
MIN_TRADES_CELL = 40  # 帯ごとのトレードがこれ未満なら判定に使わない

TRAITS = [
    ("ボラ",       "年率ボラティリティ"),
    ("効率比",     "|正味変化|÷|日々の変化の合計|（低い＝行ったり来たり）"),
    ("自己相関1",  "1日リターンのラグ1自己相関（負＝短期反転）"),
    ("自己相関5",  "同ラグ5"),
    ("分散比20",   "20日分散÷(20×1日分散)。1超＝トレンド性/1未満＝平均回帰"),
    ("回帰半減期", "20日線からの乖離が半分に戻るまでの日数"),
    ("ジャンプ率", "1日±5%超の日の割合（材料に反応しやすいか）"),
    ("歪度",       "日次リターンの歪み（正＝たまに大きく上がる）"),
    ("尖度",       "とがり具合（大きい＝ふだん静かでたまに飛ぶ）"),
    ("日中値幅",   "(高値−安値)÷終値 の平均"),
    ("出来高変動", "出来高の標準偏差÷平均"),
    ("日経β",     "日経に対する感応度"),
    ("個別要因",   "1−R²（日経で説明できない部分。高い＝独自の動き）"),
    ("上昇連続",   "連続して上がる日数の平均（ランレングス）"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def rc(a, b):
    """順位相関（4.4-50：金額系は素の相関だと外れ値に支配される）"""
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def compute_traits(o, h, l, c, v, nk) -> dict:
    r = c.pct_change().dropna()
    if len(r) < MIN_DAYS:
        return None
    ann = np.sqrt(252)

    # 分散比：k日リターンの分散 ÷ (k × 1日分散)
    k = 20
    # 株価に0や欠損があると log が警告を出すので正の値だけで計算する
    cp = c[c > 0]
    rk = np.log(cp).diff(k).dropna() if len(cp) > k else pd.Series(dtype=float)
    vr = (rk.var() / (k * r.var())) if (len(rk) > 10 and r.var() > 0) else np.nan

    # 20日線からの乖離の平均回帰の速さ
    ma20 = c.rolling(20).mean()
    dev = ((c - ma20) / ma20).dropna()
    rho = dev.autocorr(lag=1) if len(dev) > 60 else np.nan
    half = (-np.log(2) / np.log(rho)) if (rho is not None and 0 < rho < 1) else np.nan

    # 日経との関係
    nka = nk.reindex(c.index, method="ffill")
    nr = nka.pct_change()
    both = pd.concat([r, nr], axis=1).dropna()
    beta = idio = np.nan
    if len(both) > 100 and both.iloc[:, 1].var() > 0:
        beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
        corr = both.iloc[:, 0].corr(both.iloc[:, 1])
        idio = 1 - corr ** 2 if pd.notna(corr) else np.nan

    # 連続上昇の平均長
    up = (r > 0).astype(int).values
    runs, cur = [], 0
    for x in up:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    net = abs(c.iloc[-1] - c.iloc[0])
    path = c.diff().abs().sum()

    return {
        "ボラ": r.std() * ann * 100,
        "効率比": net / path if path else np.nan,
        "自己相関1": r.autocorr(lag=1),
        "自己相関5": r.autocorr(lag=5),
        "分散比20": vr,
        "回帰半減期": half,
        "ジャンプ率": (r.abs() > 0.05).mean() * 100,
        "歪度": r.skew(),
        "尖度": r.kurt(),
        "日中値幅": ((h - l) / c).mean() * 100,
        "出来高変動": v.std() / v.mean() if v.mean() else np.nan,
        "日経β": beta,
        "個別要因": idio,
        "上昇連続": float(np.mean(runs)) if runs else np.nan,
    }


def main():
    tr = load_trades()
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")
    nk = fetch_nikkei_close("max")
    if getattr(nk.index, "tz", None) is not None:
        nk.index = nk.index.tz_localize(None)

    # 銘柄ごとの整形済みデータ
    px = {}
    for code in codes:
        d = hist.get(code)
        if d is None or len(d) < MIN_DAYS:
            continue
        idx = d.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        px[code] = pd.DataFrame(
            {"o": d["Open"].values, "h": d["High"].values, "l": d["Low"].values,
             "c": d["Close"].values, "v": d["Volume"].values}, index=idx)
    print(f"対象銘柄: {len(px)} / トレード: {len(tr):,}件\n")

    # 窓ごとの相場環境
    print("=== 5年窓と相場環境 ===")
    regime = {}
    for lab, lo, hi in WINDOWS:
        s = nk[(nk.index >= pd.Timestamp(lo)) & (nk.index <= pd.Timestamp(hi))]
        ret = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100
        regime[lab] = ret
        n = len(tr[(tr["entry_date"] >= pd.Timestamp(lo))
                   & (tr["entry_date"] <= pd.Timestamp(hi))])
        print(f"  {lab}  日経 {ret:+7.1f}%（{'上昇' if ret > 0 else '下落'}）"
              f" / トレード {n:,}件")
    print()

    # 窓ごとの性質
    T = {}
    for lab, lo, hi in WINDOWS:
        rows = {}
        for code, d in px.items():
            w = d[(d.index >= pd.Timestamp(lo)) & (d.index <= pd.Timestamp(hi))]
            if len(w) < MIN_DAYS:
                continue
            t = compute_traits(w["o"], w["h"], w["l"], w["c"], w["v"], nk)
            if t:
                rows[code] = t
        T[lab] = pd.DataFrame(rows).T
        print(f"  {lab}: 性質を作れた銘柄 {len(T[lab])}")
    print()

    trans = [(WINDOWS[i][0], WINDOWS[i + 1][0], WINDOWS[i + 1][1], WINDOWS[i + 1][2])
             for i in range(len(WINDOWS) - 1)]

    # ---------- (a) 持続性 ----------
    print("=" * 74)
    print("(a) 性質としての持続性（前の窓 → 次の窓 の順位相関）")
    print("=" * 74)
    hdr = f"{'性質':<12}" + "".join(f"{a[:4]}→{b[:4]:>5}" for a, b, _, _ in trans) + f"{'平均':>8}"
    print(hdr)
    persist = {}
    for name, _ in TRAITS:
        vals = []
        for a, b, _, _ in trans:
            j = T[a][[name]].join(T[b][[name]], lsuffix="_a", rsuffix="_b").dropna()
            vals.append(rc(j[f"{name}_a"], j[f"{name}_b"]))
        persist[name] = np.nanmean(vals)
        print(f"{name:<12}" + "".join(f"{v:>10.3f}" for v in vals)
              + f"{persist[name]:>8.3f}")
    print("\n  ※順位相関が高い＝『その銘柄の性質』として安定している")
    print("    低い（0付近や負）＝ 期間ごとに入れ替わる＝くせとして存在しない\n")

    # ---------- (b) 予測力 ----------
    print("=" * 74)
    print("(b) 予測力（前の窓の性質で4分割 → 次の窓のPF）")
    print("=" * 74)
    verdicts = {}
    for name, desc in TRAITS:
        lines, dirs = [], []
        for a, b, lo, hi in trans:
            src = T[a][name].dropna()
            if len(src) < 100:
                lines.append((f"{a}→{b}", [np.nan] * 4, np.nan))
                continue
            try:
                band = pd.qcut(src, 4, labels=["最小", "小", "大", "最大"],
                               duplicates="drop")
            except ValueError:
                lines.append((f"{a}→{b}", [np.nan] * 4, np.nan))
                continue
            nxt = tr[(tr["entry_date"] >= pd.Timestamp(lo))
                     & (tr["entry_date"] <= pd.Timestamp(hi))]
            vals = []
            for lab2 in ["最小", "小", "大", "最大"]:
                cs = set(band[band == lab2].index)
                s = nxt[nxt["code"].isin(cs)]["return_pct"]
                vals.append(pf(s) if len(s) >= MIN_TRADES_CELL else np.nan)
            base = pf(nxt["return_pct"])
            lines.append((f"{a}→{b}", vals, base))
            if np.isfinite(vals[0]) and np.isfinite(vals[3]):
                dirs.append(np.sign(vals[0] - vals[3]))
        ok = len(dirs) == len(trans) and len(set(dirs)) == 1 and dirs[0] != 0
        verdicts[name] = ok
        mark = "★一貫" if ok else " 不一致"
        print(f"\n--- {name}（{desc}）  持続性{persist[name]:+.3f}  {mark} ---")
        print(f"    {'引き継ぎ':<22}{'最小':>8}{'小':>8}{'大':>8}{'最大':>8}{'条件なし':>10}")
        for lbl, vals, base in lines:
            print(f"    {lbl:<22}" + "".join(
                f"{v:>8.2f}" if np.isfinite(v) else f"{'-':>8}" for v in vals)
                + (f"{base:>10.2f}" if np.isfinite(base) else f"{'-':>10}"))

    print("\n" + "=" * 74)
    print("★一貫 = 4回の引き継ぎすべてで『最小帯 − 最大帯』の符号が同じ")
    print("=" * 74)
    good = [n for n, ok in verdicts.items() if ok]
    print(f"  4回すべてで向きが一致した性質: "
          f"{('、'.join(good)) if good else '**なし**'}\n")

    # ---------- (c) 重複 ----------
    print("=" * 74)
    print("(c) 性質どうしの重複（全窓をまとめた順位相関）")
    print("=" * 74)
    allT = pd.concat([T[w[0]] for w in WINDOWS])
    names = [n for n, _ in TRAITS]
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            M.loc[i, j] = rc(allT[i], allT[j])
    print(M.round(2).to_string())
    print("\n  |順位相関| >= 0.7 の組み合わせ（実質同じものを見ている）:")
    dup = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if pd.notna(M.loc[a, b]) and abs(M.loc[a, b]) >= 0.7:
                dup.append(f"    {a} × {b}: {M.loc[a, b]:+.2f}")
    print("\n".join(dup) if dup else "    なし")


if __name__ == "__main__":
    main()
