"""
ポジション管理（同時保有数・資金稼働率の上限）を検証する

ツールは「何を買うか」しか答えておらず、**いくらで何銘柄持つか**は
未検証だった。書籍には具体的な数値規定がある：

  ・うねり取り（第6章）：**資金稼働率の上限50%**
      3分割して満玉になっても投入は総資金の50%以内。
      「相場の予測は半分当たって半分外れるのが前提」で、
      連敗に耐える余力を常に残す
  ・片山晃（PART 2）：**1〜2銘柄に集中**
      ⚠️「予算100万円なら4銘柄程度」は**このツール側の現状**であって
        片山晃の主張ではない（片山晃_ルール.md の比較表の左列）。
        著者は「分散すると1銘柄あたりの監視の労力と時間も分散される」として
        集中を選んでいる。

  なお資金は100万円（INITIAL_CAPITAL）が前提で、100株単位なので
  株価2,500円なら1銘柄25万円＝**理論上も最大4銘柄程度**という制約が
  もともとある。

⚠️ `portfolio_sim.py` は**経路依存が極めて強い**。株価0.01%の誤差で
   26年リターンが20pt動くことを実測済み（analyze_sensitivity.py）。
   点推定に意味がないので、**株価に微小誤差を乗せた複数試行の幅**を出し、
   その幅より大きい差だけを有意とみなす。

⚠️ `portfolio_sim.load_tickers()` の既定は **tickers.csv（日経225）**。
   現行ルールのユニバースは **universe.csv（944銘柄）** なので、
   明示的に渡さないと**違う母集団で測ってしまう**（2026-09-04に実際に踏んだ）。

使い方:
    python3 analyze_position_limits.py [期間] [試行回数] [--tickers パス]
      例: python3 analyze_position_limits.py max 5
      銘柄CSV 省略時は universe.csv
"""

import sys

import numpy as np
import pandas as pd

import portfolio_sim as ps
from price_cache import fetch_histories
from analyze_sensitivity import perturb, NOISE_PCT

# 検証する設定（書籍の規定と、その前後）
VARIANTS = [
    ("制限なし（現状）", None, None),
    ("4銘柄まで（現状の実質上限）", 4, None),
    ("2銘柄まで（片山流の集中）", 2, None),
    ("1銘柄まで（極端な集中）", 1, None),
    ("稼働率50%まで（うねり取り）", None, 50.0),
    ("稼働率50%＋4銘柄", 4, 50.0),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    period = args[0] if args else "max"
    trials = int(args[1]) if len(args) > 1 else 5

    tickers_path = (sys.argv[sys.argv.index("--tickers") + 1]
                    if "--tickers" in sys.argv else "universe.csv")
    tickers = ps.load_tickers(tickers_path)
    print(f"対象 {len(tickers)}銘柄（{tickers_path}）/ 過去{period} / "
          f"各設定を{trials}通りの"
          f"株価誤差（±{NOISE_PCT}%）で試す")
    print("（経路依存が強いので、点推定ではなく幅で比べる）\n")

    regime_raw = ps.fetch_market_regime_adx(period)
    nikkei = ps.fetch_nikkei_close(period)
    fetched = fetch_histories([t["code"] for t in tickers], period=period)

    base_hist, name_map = {}, {}
    for t in tickers:
        h = fetched.get(t["code"])
        if h is None or len(h) < 120:
            continue
        if (h["Close"].pct_change().abs() > ps.MAX_PLAUSIBLE_DAILY_MOVE).any():
            continue
        base_hist[t["code"]] = h
        name_map[t["code"]] = t["name"]
    print(f"  {len(base_hist)}銘柄を対象にします\n")

    results = {}
    for label, maxpos, maxdep in VARIANTS:
        rets, dds, ntr = [], [], []
        for trial in range(trials):
            rng = np.random.default_rng(trial)
            sig_map = {
                code: ps.build_signals(h if trial == 0 else perturb(h, rng), nikkei)
                for code, h in base_hist.items()
            }
            calendar = sig_map[next(iter(sig_map))].index
            for df in sig_map.values():
                calendar = calendar.union(df.index)
            regime = regime_raw.reindex(calendar, method="ffill").fillna(False)
            r = ps.simulate(sig_map, name_map, regime, calendar, "volume",
                            park_cash_in_index=nikkei, apply_tax=True,
                            max_positions=maxpos, max_deployed_pct=maxdep)
            rets.append(r["total_return_pct"])
            dds.append(r["max_drawdown_pct"])
            ntr.append(r["n_trades"])
        results[label] = {
            "リターン中央値%": round(float(np.median(rets)), 1),
            "最小%": round(min(rets), 1), "最大%": round(max(rets), 1),
            "幅pt": round(max(rets) - min(rets), 1),
            "最大DD%": round(float(np.median(dds)), 1),
            "取引数": int(np.median(ntr)),
        }
        print(f"  {label:<28} 中央値 {results[label]['リターン中央値%']:>8.1f}%  "
              f"幅 {results[label]['幅pt']:>6.1f}pt  "
              f"DD {results[label]['最大DD%']:>6.1f}%  "
              f"取引 {results[label]['取引数']:>5}件")

    print("\n=== まとめ ===")
    print(pd.DataFrame(results).T.to_string())
    nk = nikkei
    print(f"\n（参考）同期間の日経平均を買い持ち・納税後: "
          f"{(nk.iloc[-1] - nk.iloc[0]) / nk.iloc[0] * 100 * (1 - ps.TAX_RATE):+.1f}%")
    print("\n⚠️ 各設定の『幅』より小さい差は、株価0.01%の誤差でも生じる。")
    print("   幅を超えて差がついた設定だけを『違う』と判断すること。")


if __name__ == "__main__":
    main()
