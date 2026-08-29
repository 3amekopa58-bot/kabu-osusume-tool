"""
ポートフォリオシミュレーションのノイズ幅を実測する

portfolio_sim.py は資金を日々回しながら「その日どの銘柄を買うか」を順に
決めていくため、経路依存が強い。ある日わずかな差でシグナルの成否が反転
すると、そこから先の保有銘柄・資金・売買タイミングがすべて変わる。

実際、株価データを yf.Ticker().history() から yf.download() に変えた
だけで（差は0.0003円程度）、27年のリターンが +142.7% → +122.7% と
20pt動いた。この規模のブレがあるなら、「戦略が日経を+22.5pt上回った」の
ような細かい差は意味を持たない。

そこでこのスクリプトは、株価にごく小さなランダム誤差（既定±0.01%）を
乗せた世界を何通りも作り、結果がどれだけ散らばるかを測る。得られた
散らばりが「この差より小さい違いは論じても無意味」という下限になる。

使い方:
    python3 analyze_sensitivity.py [期間] [試行回数] [--tickers パス]
      例: python3 analyze_sensitivity.py max 8
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_sim as ps
from price_cache import fetch_histories

# 株価に乗せる相対誤差の大きさ。取引所の呼値より細かく、経済的には
# 完全に無意味な差（1,000円の株で0.1円）である点が重要
NOISE_PCT = 0.01


def perturb(hist: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """OHLCに±NOISE_PCT%のランダム誤差を乗せる（出来高は触らない）"""
    out = hist.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            factor = 1 + rng.uniform(-NOISE_PCT / 100, NOISE_PCT / 100, len(out))
            out[col] = out[col].values * factor
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    period = args[0] if args else "max"
    trials = int(args[1]) if len(args) > 1 else 8
    tickers_path = (sys.argv[sys.argv.index("--tickers") + 1]
                    if "--tickers" in sys.argv else None)

    tickers = ps.load_tickers(tickers_path)
    print(f"対象銘柄: {len(tickers)}件"
          f"（{Path(tickers_path).name if tickers_path else 'tickers.csv'}）")
    print(f"株価に±{NOISE_PCT}%のランダム誤差を乗せた世界を{trials}通り作り、"
          f"過去{period}のシミュレーション結果の散らばりを測ります…\n")

    regime_raw = ps.fetch_market_regime_adx(period)
    nikkei_close = ps.fetch_nikkei_close(period)
    fetched = fetch_histories([t["code"] for t in tickers], period=period)

    base_hist, name_map = {}, {}
    for t in tickers:
        code, name = t["code"], t["name"]
        hist = fetched.get(code)
        if hist is None or len(hist) < 120:
            continue
        if (hist["Close"].pct_change().abs() > ps.MAX_PLAUSIBLE_DAILY_MOVE).any():
            continue
        base_hist[code] = hist
        name_map[code] = name
    print(f"  → {len(base_hist)}銘柄を対象にします")

    results = []
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        # trial 0 は誤差なし（＝通常のシミュレーションと同じ結果）
        sig_map = {
            code: ps.build_signals(h if trial == 0 else perturb(h, rng), nikkei_close)
            for code, h in base_hist.items()
        }
        calendar = sig_map[next(iter(sig_map))].index
        for df in sig_map.values():
            calendar = calendar.union(df.index)
        regime = regime_raw.reindex(calendar, method="ffill").fillna(False)

        r = ps.simulate(sig_map, name_map, regime, calendar, "volume",
                        park_cash_in_index=nikkei_close, apply_tax=True)
        results.append(r)
        label = "誤差なし" if trial == 0 else f"誤差あり#{trial}"
        print(f"  [{trial + 1}/{trials}] {label}: "
              f"{r['total_return_pct']:+.1f}% "
              f"(DD {r['max_drawdown_pct']:.1f}% / {r['n_trades']}件)")

    rets = [r["total_return_pct"] for r in results]
    noisy = rets[1:]  # 誤差を乗せた試行だけで散らばりを測る
    nk = nikkei_close.loc[(nikkei_close.index >= calendar[0])
                          & (nikkei_close.index <= calendar[-1])]
    nikkei_after_tax = (nk.iloc[-1] - nk.iloc[0]) / nk.iloc[0] * 100 * (1 - ps.TAX_RATE)

    print(f"\n=== ノイズ幅の実測（過去{period}・±{NOISE_PCT}%の株価誤差）===")
    print(f"誤差なしの結果        : {rets[0]:+.1f}%")
    if noisy:
        spread = max(noisy) - min(noisy)
        print(f"誤差ありの結果        : 最小 {min(noisy):+.1f}% / "
              f"最大 {max(noisy):+.1f}% / 平均 {sum(noisy)/len(noisy):+.1f}%")
        print(f"散らばり（最大-最小）  : {spread:.1f}pt")
        print(f"標準偏差              : {pd.Series(noisy).std():.1f}pt")
        print(f"\n→ この期間では、{spread:.0f}pt程度の差は株価0.01%の誤差でも生じる。"
              f"\n   これより小さい差を「勝った/負けた」と論じてはいけない。")
    print(f"\n（参考）同期間の日経平均を買い持ち・納税後: {nikkei_after_tax:+.1f}%")


if __name__ == "__main__":
    main()
