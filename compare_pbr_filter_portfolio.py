"""PBRフィルターをポートフォリオ水準で検証する（4.4-71 の最終関門）。

⚠️ **トレード単位で改善しても、ポートフォリオで改善するとは限らない。**
   4.4-5 の業種フィルターは、件数が減って資金が遊び、成績が悪化して
   不採用になった。同じ土俵で確かめる。

⚠️ portfolio_sim は経路依存が極端（株価0.01%の誤差で26年リターンが
   20pt動く＝6.6-2）。単発の比較では判断できないので、
   **複数シードの幅**で比べる。

使い方: ./venv/bin/python compare_pbr_filter_portfolio.py [期間]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_sim as ps

SEEDS = list(range(20))
THRESHOLDS = [None, 1.0, 0.8]


def main() -> None:
    period = sys.argv[1] if len(sys.argv) > 1 else "10y"
    tickers = ps.load_tickers(str(Path(ps.BASE_DIR) / "universe.csv"))
    print(f"対象 {len(tickers)}銘柄 / 期間 {period} / シード {SEEDS}")

    print("日経平均と株価を用意中…")
    regime = ps.fetch_market_regime_adx(period)
    nikkei = ps.fetch_nikkei_close(period)
    fetched = ps.fetch_histories([t["code"] for t in tickers], period=period)
    sig_map, name_map = {}, {}
    for t in tickers:
        h = fetched.get(t["code"])
        if h is None or len(h) < 120:
            continue
        if (h["Close"].pct_change().abs() > ps.MAX_PLAUSIBLE_DAILY_MOVE).any():
            continue          # 汚染データ（本体と同じ基準で除外）
        sig_map[t["code"]] = ps.build_signals(h, nikkei)
        name_map[t["code"]] = t["name"]
    calendar = sorted(set().union(*[d.index for d in sig_map.values()]))
    print(f"  → {len(sig_map)}銘柄")
    bps = ps.load_bps_map()
    print(f"BPSのある銘柄: {len(bps)}")

    # ⚠️ 設定ごとの分布を比べるだけでは、経路依存のノイズに埋もれる。
    #    **同じシードで対にして差を取る**と、共通のノイズが打ち消えて
    #    フィルターの効果だけが残る（対応のある比較）。
    res = {}
    for t in THRESHOLDS:
        res[t] = [ps.simulate(sig_map, name_map, regime, calendar, "random",
                              seed=sd, max_pbr=t, bps_map=bps)["total_return_pct"]
                  for sd in SEEDS]
    rows = []
    for t in THRESHOLDS:
        v = np.array(res[t])
        rows.append({"しきい値": "なし" if t is None else f"PBR<{t}",
                     "中央値%": np.median(v), "最小%": v.min(),
                     "最大%": v.max(), "幅pt": v.max() - v.min()})
    print(f"\n=== 最終リターン（シード{len(SEEDS)}本）===")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f"))

    print("\n=== 同一シードでの対比較（フィルターあり − なし）===")
    base = np.array(res[None])
    for t in THRESHOLDS:
        if t is None:
            continue
        d = np.array(res[t]) - base
        win = (d > 0).sum()
        print(f"  PBR<{t}: 差の中央値 {np.median(d):+.1f}pt / "
              f"平均 {d.mean():+.1f}pt / 範囲 {d.min():+.1f}〜{d.max():+.1f}pt")
        print(f"          {len(SEEDS)}本中 {win}本で改善"
              f"（偶然なら約{len(SEEDS)/2:.0f}本）")
    print("\n⚠️ 対比較でも過半数が改善し、差の範囲がゼロを跨がないことを確認する。")


if __name__ == "__main__":
    main()
