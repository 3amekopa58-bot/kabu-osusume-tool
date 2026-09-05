"""
「独自に動く銘柄」を集めるとドローダウンは浅くなるか

4.4-53 で、14種類の性質のうち**最も持続するのは個別要因**だった
（1−R²＝日経で説明できない部分。順位相関 0.754）。
4.4-54 の実例でも、任天堂は26年を通じて個別要因が高く（0.82→0.84）、
トヨタは一貫して低い（0.68→0.59、下位5%）。
**「独自に動くかどうか」は銘柄に固有で安定した性質**である。

これまで性質はすべて**トレード単位のPF**で評価してきた。
個別要因は4回の引き継ぎのうち2回で強く効いたが一貫せず不採用だった。

しかし**個別要因の本来の意味はリターンではなくリスク**にある。
日経と連動しない銘柄を集めれば、**同時に落ちにくい＝分散が効く**はず。
これはトレード単位では絶対に測れない（1件ずつ独立に数えるため）。
**ポートフォリオ単位でしか測れない、未検証の問い。**

⚠️ 4.4-42 で「余剰資金を日経ETFで運用するのは日経に賭けているだけ」と
   分かった。今回はその逆で、**日経から離れることの価値**を測る。

⚠️ `portfolio_sim.py` は経路依存が極めて強いので、株価に±0.01%の誤差を
   乗せた複数試行の**幅**で比べる（4.4-38 以降の標準）。

設計:
  ・5年窓で銘柄の個別要因を測り、**次の5年**のシミュレーションに使う
    （後知恵なし）。4回の引き継ぎ。
  ・各回、ユニバースを個別要因の上位半分／下位半分に分け、それぞれで
    ポートフォリオを回す（銘柄数を揃えて公平に比べる）
  ・リターンだけでなく**最大ドローダウン**を見る

使い方:
    python3 analyze_idio_diversification.py [試行回数]
"""

import sys

import numpy as np
import pandas as pd

import portfolio_sim as ps
from price_cache import fetch_histories
from analyze_sensitivity import perturb, NOISE_PCT

WINDOWS = [
    ("2000-2005", "2000-01-01", "2004-12-31"),
    ("2005-2010", "2005-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2014-12-31"),
    ("2015-2020", "2015-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
]
MIN_DAYS = 400


def market_stats(c: pd.Series, nk: pd.Series):
    """個別要因（1−R²）と日経βを返す"""
    r = c.pct_change()
    nr = nk.reindex(c.index, method="ffill").pct_change()
    both = pd.concat([r, nr], axis=1).dropna()
    if len(both) < 100 or both.iloc[:, 1].var() <= 0:
        return np.nan, np.nan
    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
    idio = 1 - corr ** 2 if pd.notna(corr) else np.nan
    return idio, beta


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tickers = ps.load_tickers("universe.csv")
    codes = [t["code"] for t in tickers]
    names = {t["code"]: t["name"] for t in tickers}

    print(f"対象 {len(codes)}銘柄 / 各設定を{trials}通りの株価誤差"
          f"（±{NOISE_PCT}%）で試す")
    print("（経路依存が強いので点推定ではなく幅で比べる）\n")

    nk = ps.fetch_nikkei_close("max")
    if getattr(nk.index, "tz", None) is not None:
        nk.index = nk.index.tz_localize(None)
    regime_raw = ps.fetch_market_regime_adx("max")
    fetched = fetch_histories(codes, period="max")

    base_hist = {}
    for t in tickers:
        h = fetched.get(t["code"])
        if h is None or len(h) < MIN_DAYS:
            continue
        if (h["Close"].pct_change().abs() > ps.MAX_PLAUSIBLE_DAILY_MOVE).any():
            continue
        idx = h.index
        if getattr(idx, "tz", None) is not None:
            h = h.copy()
            h.index = idx.tz_localize(None)
        base_hist[t["code"]] = h
    print(f"  {len(base_hist)}銘柄を対象にします\n")

    # 窓ごとの個別要因と日経β
    idio, beta = {}, {}
    for lab, lo, hi in WINDOWS:
        iv, bv = {}, {}
        for code, h in base_hist.items():
            c = h["Close"]
            w = c[(c.index >= pd.Timestamp(lo)) & (c.index <= pd.Timestamp(hi))]
            if len(w) < MIN_DAYS:
                continue
            i_, b_ = market_stats(w, nk)
            if np.isfinite(i_) and np.isfinite(b_):
                iv[code], bv[code] = i_, b_
        idio[lab], beta[lab] = pd.Series(iv), pd.Series(bv)
        print(f"  {lab}: 測れた銘柄 {len(iv)} / 個別要因の中央値 "
              f"{np.median(list(iv.values())):.3f} / βの中央値 "
              f"{np.median(list(bv.values())):.2f} / 両者の順位相関 "
              f"{pd.Series(iv).rank().corr(pd.Series(bv).rank()):+.3f}")
    print()

    trans = [(WINDOWS[i][0], WINDOWS[i+1][0], WINDOWS[i+1][1], WINDOWS[i+1][2])
             for i in range(len(WINDOWS) - 1)]

    # ⚠️ 個別要因とβは順位相関 -0.58〜-0.81 と強く相関する。
    # βで直接分けても同じ効果が出るなら、個別要因は何も足していない
    # （4.4-50 で売買代金が時価総額の言い換えだったのと同じ構図）。
    # そこで両方の分け方で回して比べる。
    for axis in ("個別要因", "日経β"):
        src = idio if axis == "個別要因" else beta
        # 個別要因は「高いほうが独自」、βは「低いほうが市場から離れている」
        run_split(axis, src, trans, base_hist, names, nk, regime_raw, trials)


def run_split(axis, src, trans, base_hist, names, nk, regime_raw, trials):
    print("=" * 78)
    print(f"【{axis}】で上位半分／下位半分に分け、次の窓を回す")
    print("=" * 78)

    summary = []
    for a, b, lo, hi in trans:
        s = src[a].dropna().sort_values()
        half = len(s) // 2
        if axis == "個別要因":
            groups = {"市場寄り（連動）": list(s.index[:half]),
                      "市場から離れる（独自）": list(s.index[half:])}
        else:
            # βは小さいほうが市場から離れている
            groups = {"市場から離れる（低β）": list(s.index[:half]),
                      "市場寄り（高β）": list(s.index[half:])}
        nkw = nk[(nk.index >= pd.Timestamp(lo)) & (nk.index <= pd.Timestamp(hi))]
        nk_ret = (nkw.iloc[-1] - nkw.iloc[0]) / nkw.iloc[0] * 100
        print(f"\n--- {a} の個別要因 → {b} の運用"
              f"（日経 {nk_ret:+.1f}%）---")
        print(f"    {'グループ':<24}{'リターン中央値':>14}{'幅':>8}"
              f"{'最大DD中央値':>14}{'幅':>8}{'取引':>7}")
        row = {"引き継ぎ": f"{a}→{b}", "日経%": round(nk_ret, 1)}
        for gname, gcodes in groups.items():
            rets, dds, ns = [], [], []
            for trial in range(trials):
                rng = np.random.default_rng(trial)
                sig_map, name_map = {}, {}
                for code in gcodes:
                    h = base_hist.get(code)
                    if h is None:
                        continue
                    hh = h if trial == 0 else perturb(h, rng)
                    sig_map[code] = ps.build_signals(hh, nk)
                    name_map[code] = names[code]
                if len(sig_map) < 50:
                    continue
                cal = sig_map[next(iter(sig_map))].index
                for df in sig_map.values():
                    cal = cal.union(df.index)
                cal = cal[(cal >= pd.Timestamp(lo)) & (cal <= pd.Timestamp(hi))]
                if len(cal) < 250:
                    continue
                reg = regime_raw.reindex(cal, method="ffill").fillna(False)
                r = ps.simulate(sig_map, name_map, reg, cal, "volume",
                                apply_tax=True)
                rets.append(r["total_return_pct"])
                dds.append(r["max_drawdown_pct"])
                ns.append(r["n_trades"])
            if not rets:
                continue
            print(f"    {gname:<24}{np.median(rets):>14.1f}"
                  f"{max(rets)-min(rets):>8.1f}"
                  f"{np.median(dds):>14.1f}{max(dds)-min(dds):>8.1f}"
                  f"{int(np.median(ns)):>7}")
            key = "独自" if "離れる" in gname else "連動"
            row[f"{key}_ret"] = round(float(np.median(rets)), 1)
            row[f"{key}_dd"] = round(float(np.median(dds)), 1)
            row[f"{key}_ret幅"] = round(max(rets) - min(rets), 1)
            row[f"{key}_dd幅"] = round(max(dds) - min(dds), 1)
        summary.append(row)

    print("\n" + "-" * 78)
    print(f"まとめ【{axis}】（市場から離れる − 市場寄り）")
    print("-" * 78)
    df = pd.DataFrame(summary)
    if len(df) and "独自_ret" in df.columns:
        df["リターン差"] = (df["独自_ret"] - df["連動_ret"]).round(1)
        # DDは負の値なので「大きい＝浅い」
        df["DD差(正=浅い)"] = (df["独自_dd"] - df["連動_dd"]).round(1)
        print(df[["引き継ぎ", "日経%", "独自_ret", "連動_ret", "リターン差",
                  "独自_dd", "連動_dd", "DD差(正=浅い)"]].to_string(index=False))
        print("\n⚠️ 各設定の『幅』より小さい差は株価0.01%の誤差でも生じる。")
        print("   4回すべてで同じ向きに出ない限り採用しない。")
    print()


if __name__ == "__main__":
    main()
