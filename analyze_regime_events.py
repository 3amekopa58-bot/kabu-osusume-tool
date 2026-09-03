"""
暴落局面・季節性・周期性と、このルールの成績の関係を調べる

これまでの検証はすべて「銘柄の条件」だったが、
**いつ買うか（相場環境）**との関係は、日経平均のADXフィルターしか見ていない。
暴落時にどう振る舞うのか、季節性があるのかは未検証だった。

やること：
  ① 実際に起きた危機イベントの前後で、成績がどう変わるか
  ② 月・曜日の季節性
  ③ 日経平均のドローダウン水準（高値からの下落率）別の成績
  ④ 日経平均の年間騰落との関係

⚠️ 危機イベントは**後知恵で日付を指定している**。「暴落が来たら避ける」
   という運用はできない（事前には分からない）。ここで測るのは
   「暴落局面で何が起きたか」であって、予測ルールではない。

⚠️ ①は該当件数が少なくなりやすい。件数を必ず併記して読むこと。

使い方:
    python3 analyze_regime_events.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
SUSPICIOUS_RETURN_THRESHOLD = 500.0
DEFAULT_TRADES = ("output/backtest_trades_timesl10d60_either_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")

# 後知恵で指定した主要な危機イベント（開始日〜おおむね底を打つまで）
CRISES = [
    ("ITバブル崩壊", "2000-04-01", "2003-04-30"),
    ("ライブドア・ショック", "2006-01-16", "2006-06-30"),
    ("リーマン・ショック", "2008-09-01", "2009-03-31"),
    ("東日本大震災", "2011-03-11", "2011-06-30"),
    ("チャイナ・ショック", "2015-08-01", "2016-02-29"),
    ("コロナ・ショック", "2020-02-20", "2020-04-30"),
    ("2022年の下落（利上げ）", "2022-01-01", "2022-10-31"),
    ("2024年8月の急落", "2024-07-11", "2024-08-16"),
    ("2025年4月の急落", "2025-04-01", "2025-05-13"),
]


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    w = sub[sub["return_pct"] > 0]["return_pct"].sum()
    l = abs(sub[sub["return_pct"] <= 0]["return_pct"].sum())
    return {"件数": len(sub), "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
            "平均%": round(sub["return_pct"].mean(), 2),
            "PF": round(w / l, 2) if l else float("inf")}


def main():
    from price_cache import fetch_histories
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / DEFAULT_TRADES
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    nk = fetch_histories(["^N225"], period="max", verbose=False).get("^N225")
    c = nk["Close"]
    if c.index.tz is not None:
        c.index = c.index.tz_localize(None)
    dd = (c / c.cummax() - 1) * 100          # 高値からの下落率
    ret1y = c.pct_change(250) * 100          # 過去1年の騰落率

    df["dd"] = dd.reindex(df["entry_date"], method="ffill").values
    df["nk1y"] = ret1y.reindex(df["entry_date"], method="ffill").values

    base = stats(df)
    print(f"全体: {base['件数']:,}件 / 勝率{base['勝率%']}% / "
          f"平均{base['平均%']}% / PF{base['PF']}")
    print(f"期間: {df.entry_date.min().date()} 〜 {df.entry_date.max().date()}\n")

    print("=== ① 危機イベント中にエントリーしたトレード ===")
    print("⚠️ 後知恵で日付を指定している。予測ルールではない")
    out = {"全体（平常時含む）": base}
    allc = pd.Series(False, index=df.index)
    for name, s, e in CRISES:
        m = (df.entry_date >= s) & (df.entry_date <= e)
        allc |= m
        st = stats(df[m])
        if st and st["件数"] >= 10:
            out[f"{name}"] = st
    out["危機中すべて"] = stats(df[allc])
    out["危機以外"] = stats(df[~allc])
    print(pd.DataFrame(out).T.to_string())
    print()

    print("=== ② 日経平均のドローダウン別（エントリー日時点、高値からの下落率）===")
    bands = [(-100, -20, "-20%以下"), (-20, -10, "-20〜-10%"),
             (-10, -5, "-10〜-5%"), (-5, 0.1, "-5%〜高値圏")]
    out = {}
    for lo, hi, lab in bands:
        st = stats(df[(df.dd > lo) & (df.dd <= hi)])
        if st and st["件数"] >= 30:
            out[lab] = st
    print(pd.DataFrame(out).T.to_string())
    print()

    print("=== ③ 日経平均の過去1年騰落率別 ===")
    bands = [(-100, -10, "-10%以下"), (-10, 0, "-10〜0%"),
             (0, 10, "0〜+10%"), (10, 25, "+10〜+25%"), (25, 1e9, "+25%以上")]
    out = {}
    for lo, hi, lab in bands:
        st = stats(df[(df.nk1y > lo) & (df.nk1y <= hi)])
        if st and st["件数"] >= 30:
            out[lab] = st
    print(pd.DataFrame(out).T.to_string())
    print()

    print("=== ④ 月別（季節性）===")
    df["月"] = df.entry_date.dt.month
    out = {f"{m}月": stats(df[df["月"] == m]) for m in range(1, 13)}
    print(pd.DataFrame({k: v for k, v in out.items() if v}).T.to_string())
    print()

    print("=== ⑤ 曜日別 ===")
    names = ["月", "火", "水", "木", "金"]
    df["曜日"] = df.entry_date.dt.dayofweek
    out = {names[d]: stats(df[df["曜日"] == d]) for d in range(5)}
    print(pd.DataFrame({k: v for k, v in out.items() if v}).T.to_string())


if __name__ == "__main__":
    main()
