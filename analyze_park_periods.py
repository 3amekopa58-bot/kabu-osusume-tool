"""
コア・サテライト（余剰資金を日経ETFで運用）を重複しない3期間で再検証する

4.4-4 で「余剰資金を日経ETFで運用すると結果が逆転する」と結論したが、
確認したのは **5年 / 10年 / 27年 という入れ子の期間だけ** だった。
入れ子の期間は同じ相場を重複して数えるので、「どの期間でも成り立つか」の
確認にはならない。4.4-41 の副産物で、重複しない3期間に割ると
**日経が下げ続けた2000-2010では現金待機のほうが大差で良い**ことが
見えたため、条件を揃えて測り直す。

⚠️ 4.4-4 の母集団は **tickers.csv（日経225）**。現行ルールのユニバースは
   universe.csv（944銘柄）なので、両方で回して母集団の影響も分ける。

⚠️ `portfolio_sim.py` は経路依存が極めて強い（株価0.01%の誤差で26年
   リターンが20pt動く）。点推定に意味がないので、微小誤差を乗せた
   複数試行の**幅**を出し、その幅を超えた差だけを有意とみなす。

比較するのは「待機資金の置き場」だけ。相場環境の門（ADX20超）は
現行のまま両方に掛けてあるので、4.4-41 の「休む/買う」とは別の問い。

使い方:
    python3 analyze_park_periods.py [期間] [試行回数] [--tickers パス]
      例: python3 analyze_park_periods.py max 5 --tickers tickers.csv
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

VARIANTS = [
    ("現金で待つ", None),
    ("日経ETFで運用", "nikkei"),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    period = args[0] if args else "max"
    trials = int(args[1]) if len(args) > 1 else 5
    tickers_path = (sys.argv[sys.argv.index("--tickers") + 1]
                    if "--tickers" in sys.argv else "tickers.csv")

    tickers = ps.load_tickers(tickers_path)
    print(f"対象 {len(tickers)}銘柄（{tickers_path}）/ 過去{period} / "
          f"各設定を{trials}通りの株価誤差（±{NOISE_PCT}%）で試す")
    print("（相場環境の門＝ADX20超は現行のまま。比べるのは待機資金の置き場だけ）\n")

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

    # 試行ごとにシグナルを作り直し、その中で各期間・各設定を回す
    acc = {}   # (期間, 設定) -> {"ret": [...], "dd": [...]}
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        sig_map = {
            code: ps.build_signals(h if trial == 0 else perturb(h, rng), nikkei)
            for code, h in base_hist.items()
        }
        cal_all = sig_map[next(iter(sig_map))].index
        for df in sig_map.values():
            cal_all = cal_all.union(df.index)
        sim_start, sim_end = cal_all[0], cal_all[-1]

        for plabel, lo, hi in SUBPERIODS:
            calendar = cal_all
            if lo:
                calendar = calendar[(calendar >= pd.Timestamp(lo))
                                    & (calendar <= pd.Timestamp(hi))]
            if len(calendar) < 250:
                continue
            regime = regime_raw.reindex(calendar, method="ffill").fillna(False)
            for vlabel, park in VARIANTS:
                r = ps.simulate(
                    sig_map, name_map, regime, calendar, "volume",
                    park_cash_in_index=nikkei if park else None, apply_tax=True)
                d = acc.setdefault((plabel, vlabel), {"ret": [], "dd": []})
                d["ret"].append(r["total_return_pct"])
                d["dd"].append(r["max_drawdown_pct"])

    # 日経を買い持ちした場合（最後に売って納税）を期間ごとに出す。
    # ⚠️ ^N225 の period="max" は1965年まで遡る。全期間の比較で
    # スライスを忘れると61年分の指数と27年分の戦略を比べることになる
    # （4.4-4 で修正済みのバグを、このスクリプトで一度再現してしまった）。
    for plabel, lo, hi in SUBPERIODS:
        lo_ts = pd.Timestamp(lo) if lo else sim_start
        hi_ts = pd.Timestamp(hi) if hi else sim_end
        nk = nikkei[(nikkei.index >= lo_ts) & (nikkei.index <= hi_ts)]
        if len(nk) < 2:
            continue
        raw = (nk.iloc[-1] - nk.iloc[0]) / nk.iloc[0] * 100
        after = raw * (1 - ps.TAX_RATE) if raw > 0 else raw
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
            })
        print(f"=== {plabel} ===")
        print(pd.DataFrame(rows).set_index("設定").to_string())
        print(f"（参考）日経を買い持ち・納税後: {after:+.1f}%\n")

    print("⚠️ 各設定の『幅』より小さい差は、株価0.01%の誤差でも生じる。")
    print("   重複しない3期間すべてで同じ向きに出ない限り採用しない。")


if __name__ == "__main__":
    main()
