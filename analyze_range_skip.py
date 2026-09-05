"""
「レンジ相場では見送る」のが正しいかを検証する

4.4-26 で、日経のADXが20を割ると**全銘柄が一律に失格**して推奨がゼロに
なることを確認した。平均は3.4営業日に1度でも、出ない時期は3週間以上出ない。
このとき通知は◇参考枠（＝ADX20未満などで条件を一部だけ満たす銘柄）を
出し続けていた。

トレード単位の実測では、ADX20未満でも**勝率50.0%・平均+1.83%・PF1.44**と
プラスではある。だから「レンジでも買ったほうが得では？」という疑問が残り、
REQUIREMENTS には**見送りが正しいかは未検証**と書いてあった。

⚠️ トレード単位のPFではこの問いに答えられない。
   「休む」は資金を寝かせる代わりに、**次の本命に全額を振り向けられる**
   という利益がある。資金の取り合いを含めた**ポートフォリオ単位**でしか
   優劣は測れない。

⚠️ `portfolio_sim.py` は経路依存が極めて強い（株価0.01%の誤差で26年
   リターンが20pt動く）。点推定に意味がないので、微小誤差を乗せた
   複数試行の**幅**を出し、その幅を超えた差だけを有意とみなす。

検証する4通り:
   ・休む（現行）× 待機資金は現金
   ・レンジでも買う × 待機資金は現金
   ・休む（現行）× 待機資金は日経ETF
   ・レンジでも買う × 待機資金は日経ETF

  「待機資金の置き場」で結論が変わりうる。休むことの損は資金を遊ばせる
  ことなので、ETFで運用していればその損は小さくなるはずで、
  どちらの前提かを明示しないと比較にならない。

使い方:
    python3 analyze_range_skip.py [期間] [試行回数] [--tickers パス]
      例: python3 analyze_range_skip.py max 5
      銘柄CSV 省略時は universe.csv
"""

import sys

import numpy as np
import pandas as pd

import portfolio_sim as ps
from price_cache import fetch_histories
from analyze_sensitivity import perturb, NOISE_PCT

# 重複しない3期間（4.4-26 と同じ区切り）。ネストした5y/10y/26y とは別の
# 「別々のデータで再現するか」を見るための分割。
SUBPERIODS = [
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]


def run_variants(base_hist, name_map, nikkei, regime_raw, trials,
                 calendar_slice=None):
    """
    休む／買う × 現金／ETF の4通りを、株価に微小誤差を乗せて trials 回ずつ回す。
    calendar_slice に (開始, 終了) を渡すとその期間だけを回す（指標の助走は
    全期間のデータで計算済みなので、カレンダーを切るだけでよい）。
    """
    acc = {}
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        sig_map = {
            code: ps.build_signals(h if trial == 0 else perturb(h, rng), nikkei)
            for code, h in base_hist.items()
        }
        calendar = sig_map[next(iter(sig_map))].index
        for df in sig_map.values():
            calendar = calendar.union(df.index)
        if calendar_slice:
            lo, hi = calendar_slice
            calendar = calendar[(calendar >= lo) & (calendar <= hi)]
        if len(calendar) < 250:
            continue

        regime_adx = regime_raw.reindex(calendar, method="ffill").fillna(False)
        # 「レンジでも買う」＝ 相場環境の門を開けっぱなしにする。
        # 銘柄側の条件（PPP3/4以上・100日線上・出来高・相対力）はそのまま。
        regime_always = pd.Series(True, index=calendar)

        for label, regime, park in [
            ("休む（現行）× 現金", regime_adx, None),
            ("レンジでも買う × 現金", regime_always, None),
            ("休む（現行）× 日経ETF", regime_adx, nikkei),
            ("レンジでも買う × 日経ETF", regime_always, nikkei),
        ]:
            r = ps.simulate(sig_map, name_map, regime, calendar, "volume",
                            park_cash_in_index=park, apply_tax=True)
            d = acc.setdefault(label, {"ret": [], "dd": [], "n": [], "dep": []})
            d["ret"].append(r["total_return_pct"])
            d["dd"].append(r["max_drawdown_pct"])
            d["n"].append(r["n_trades"])
            d["dep"].append(r["avg_deployed_pct"])
    return acc


def summarize(acc):
    out = {}
    for label, d in acc.items():
        out[label] = {
            "リターン中央値%": round(float(np.median(d["ret"])), 1),
            "最小%": round(min(d["ret"]), 1),
            "最大%": round(max(d["ret"]), 1),
            "幅pt": round(max(d["ret"]) - min(d["ret"]), 1),
            "最大DD%": round(float(np.median(d["dd"])), 1),
            "取引数": int(np.median(d["n"])),
            "投入率%": round(float(np.median(d["dep"])), 1),
        }
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    period = args[0] if args else "max"
    trials = int(args[1]) if len(args) > 1 else 5
    tickers_path = (sys.argv[sys.argv.index("--tickers") + 1]
                    if "--tickers" in sys.argv else "universe.csv")

    tickers = ps.load_tickers(tickers_path)
    print(f"対象 {len(tickers)}銘柄（{tickers_path}）/ 過去{period} / "
          f"各設定を{trials}通りの株価誤差（±{NOISE_PCT}%）で試す")
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
    print(f"  {len(base_hist)}銘柄を対象にします")

    # 参考：ADXが20超だった日の割合（＝どれだけ休むことになるか）
    adx_days = regime_raw.mean() * 100
    print(f"  日経ADX20超の日: 全体の{adx_days:.1f}%（残りは休む日）\n")

    print("=== 全期間 ===")
    full = summarize(run_variants(base_hist, name_map, nikkei, regime_raw, trials))
    print(pd.DataFrame(full).T.to_string())

    for label, lo, hi in SUBPERIODS:
        print(f"\n=== {label} ===")
        acc = run_variants(base_hist, name_map, nikkei, regime_raw, trials,
                           calendar_slice=(pd.Timestamp(lo), pd.Timestamp(hi)))
        if not acc:
            print("  データ不足のため実行できません")
            continue
        print(pd.DataFrame(summarize(acc)).T.to_string())

    print("\n⚠️ 各設定の『幅』より小さい差は、株価0.01%の誤差でも生じる。")
    print("   幅を超えて差がついた設定だけを『違う』と判断すること。")
    print("   重複しない3期間すべてで同じ向きに出ない限り採用しない。")


if __name__ == "__main__":
    main()
