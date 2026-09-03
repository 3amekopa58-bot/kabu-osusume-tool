"""
「いつ買うか」＝平時の相場環境と成績の関係を検証する

4.4-30 で暴落の予測は行き止まりと分かった。残るのは
**平時の相場環境によって成績が変わるか**という問い。

見る指標（すべてエントリー日時点で計算でき、後知恵ではない）：
  ① 日経平均の過去1年騰落率（相場の過熱度）
  ② 直近1年高値からの下落率（押し目の深さ）
  ③ 月（季節性）

⚠️ 季節性は過剰適合の典型。**重複しない3期間で一貫しなければ採らない**。
   月別は12通りあるので、偶然どれかが良く見えるのは当然。

使い方:
    python3 analyze_market_context.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
SUSPICIOUS_RETURN_THRESHOLD = 500.0
DEFAULT_TRADES = ("output/backtest_trades_timesl10d60_either_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")


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
    df["ret1y"] = (c.pct_change(250) * 100).reindex(df["entry_date"],
                                                    method="ffill").values
    df["dd"] = ((c / c.rolling(250, min_periods=20).max() - 1) * 100
                ).reindex(df["entry_date"], method="ffill").values
    df["月"] = df["entry_date"].dt.month

    edges = [df["entry_date"].quantile(x) for x in (1 / 3, 2 / 3)]
    df["era"] = df["entry_date"].apply(
        lambda d: 0 if d <= edges[0] else (1 if d <= edges[1] else 2))
    base = stats(df)
    print(f"全体 {base['件数']:,}件 / 勝率{base['勝率%']}% / PF{base['PF']}")
    print(f"期間 {df.entry_date.min().date()} 〜 {df.entry_date.max().date()}\n")

    def check(col, bands, title, fmt="{}"):
        print(f"=== {title} ===")
        rows = {}
        for lo, hi, lab in bands:
            m = (df[col] > lo) & (df[col] <= hi)
            st = stats(df[m])
            if not st or st["件数"] < 100:
                continue
            per = []
            for i in range(3):
                e = stats(df[m & (df.era == i)])
                per.append(e["PF"] if e and e["件数"] >= 30 else None)
            rows[lab] = {**st,
                         "第1期PF": per[0], "第2期PF": per[1], "第3期PF": per[2]}
        out = pd.DataFrame(rows).T
        print(out.to_string())
        # 全期間で全体を上回る帯を探す
        b = [stats(df[df.era == i])["PF"] for i in range(3)]
        print(f"  （比較用）全体の各期PF: {b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f}")
        good = [k for k, v in rows.items()
                if all(v[f"第{i+1}期PF"] is not None and v[f"第{i+1}期PF"] > b[i]
                       for i in range(3))]
        print(f"  → 3期間すべてで全体を上回る帯: {good if good else 'なし'}")
        print()

    check("ret1y", [(-1e9, -10, "-10%以下"), (-10, 0, "-10〜0%"), (0, 10, "0〜+10%"),
                    (10, 25, "+10〜+25%"), (25, 1e9, "+25%以上")],
          "① 日経平均の過去1年騰落率")
    check("dd", [(-1e9, -15, "-15%以下"), (-15, -8, "-15〜-8%"),
                 (-8, -3, "-8〜-3%"), (-3, 0.1, "高値圏(-3%〜)")],
          "② 直近1年高値からの下落率")
    check("月", [(m - 0.5, m + 0.5, f"{m}月") for m in range(1, 13)],
          "③ 月（季節性）")


if __name__ == "__main__":
    main()
