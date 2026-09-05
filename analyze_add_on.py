"""
すでに持っている銘柄に買い増すのは得か損かを検証する

4.4-35 で「同じ銘柄が複数日にわたって推奨される」ことを実測し
（通知の約30%が重複）、通知に注意書きを足した：

    ※すでに持っている銘柄が再び出ることがあります（実測で通知の約3割）。
    　検証は1銘柄1回を前提にしているので、買い増しには当てはまりません

しかしこの注意書きは「検証していない」と言っているだけで、
**買い増しが得なのか損なのかは測っていなかった**。
実際にお金を動かす判断なので測る。

`backtest.py` は `in_position` フラグで保有中の再エントリーを禁じており、
26年13,633トレードのうち再エントリーは0件。`portfolio_sim.py` も
同様だったので、`allow_add_on` を足して買い増しを許した場合と比べる。

⚠️ 買い増しは「同じ銘柄に賭け金を増やす」ことなので、リターンだけでなく
   **1銘柄への集中度（分散の効かなさ）**も見る必要がある。
   最大ドローダウンで見る。

⚠️ `portfolio_sim.py` は経路依存が極めて強い（株価0.01%の誤差で26年
   リターンが20pt動く）。微小誤差を乗せた複数試行の**幅**で比べる。

⚠️ 結論が**予算100万円に固有**でないかを確かめるため、`--capital` で
   初期資金を変えられるようにしてある。予算が増えれば100株単位の制約が
   緩み、買い増しが起きやすくなるので、結論が変わりうる。

使い方:
    python3 analyze_add_on.py [期間] [試行回数] [--tickers パス] [--capital 円]
"""

import sys

import numpy as np
import pandas as pd

import portfolio_sim as ps
from price_cache import fetch_histories
from analyze_sensitivity import perturb, NOISE_PCT

SUBPERIODS = [
    ("全期間", None, None),
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]

# allow_add_on: 同じ銘柄に何回まで買い増すか（0＝買い増さない＝現行）
VARIANTS = [
    ("買い増さない（現行）", 0),
    ("1回まで買い増す", 1),
    ("2回まで買い増す", 2),
    ("制限なく買い増す", 999),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    period = args[0] if args else "max"
    trials = int(args[1]) if len(args) > 1 else 5
    tickers_path = (sys.argv[sys.argv.index("--tickers") + 1]
                    if "--tickers" in sys.argv else "universe.csv")
    if "--capital" in sys.argv:
        # simulate() は呼び出し時にモジュール変数を読むので差し替えれば効く
        ps.INITIAL_CAPITAL = int(sys.argv[sys.argv.index("--capital") + 1])
    print(f"初期資金: {ps.INITIAL_CAPITAL:,}円")

    tickers = ps.load_tickers(tickers_path)
    print(f"対象 {len(tickers)}銘柄（{tickers_path}）/ 過去{period} / "
          f"各設定を{trials}通りの株価誤差（±{NOISE_PCT}%）で試す")
    print("（待機資金は日経ETF・課税込み。ほかの条件は現行のまま）\n")

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

    acc = {}
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        sig_map = {
            code: ps.build_signals(h if trial == 0 else perturb(h, rng), nikkei)
            for code, h in base_hist.items()
        }
        cal_all = sig_map[next(iter(sig_map))].index
        for df in sig_map.values():
            cal_all = cal_all.union(df.index)

        for plabel, lo, hi in SUBPERIODS:
            calendar = cal_all
            if lo:
                calendar = calendar[(calendar >= pd.Timestamp(lo))
                                    & (calendar <= pd.Timestamp(hi))]
            if len(calendar) < 250:
                continue
            regime = regime_raw.reindex(calendar, method="ffill").fillna(False)
            for vlabel, addon in VARIANTS:
                r = ps.simulate(sig_map, name_map, regime, calendar, "volume",
                                park_cash_in_index=nikkei, apply_tax=True,
                                allow_add_on=addon)
                d = acc.setdefault((plabel, vlabel),
                                   {"ret": [], "dd": [], "n": [], "win": []})
                d["ret"].append(r["total_return_pct"])
                d["dd"].append(r["max_drawdown_pct"])
                d["n"].append(r["n_trades"])
                d["win"].append(r["win_rate"])

    for plabel, _, _ in SUBPERIODS:
        rows = []
        for vlabel, _ in VARIANTS:
            d = acc.get((plabel, vlabel))
            if not d:
                continue
            rows.append({
                "設定": vlabel,
                "リターン中央値%": round(float(np.median(d["ret"])), 1),
                "最小%": round(min(d["ret"]), 1),
                "最大%": round(max(d["ret"]), 1),
                "幅pt": round(max(d["ret"]) - min(d["ret"]), 1),
                "最大DD%": round(float(np.median(d["dd"])), 1),
                "勝率%": round(float(np.median(d["win"])), 1),
                "取引数": int(np.median(d["n"])),
            })
        if not rows:
            continue
        print(f"=== {plabel} ===")
        print(pd.DataFrame(rows).set_index("設定").to_string())
        print()

    print("⚠️ 各設定の『幅』より小さい差は、株価0.01%の誤差でも生じる。")
    print("   重複しない3期間すべてで同じ向きに出ない限り採用しない。")


if __name__ == "__main__":
    main()
