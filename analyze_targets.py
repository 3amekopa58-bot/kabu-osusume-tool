"""
利確目標の検証：チャートから引いた目標が実際に到達するかを測る

通知に「どこまで上がる見込みか」を出すにあたり、全銘柄一律の固定%ではなく
チャート分析から銘柄ごとの目標を出したい。ただし目標は「予測」を名乗る以上、
実際に到達するのかを検証してからでないと通知に載せられない。

このスクリプトは、バックテストの各トレードについてエントリー時点で
各方式の目標株価を計算し、保有期間中の高値が実際にそこへ到達したかを集計する。

検証する方式:
  A) 直近高値      … 過去60日の最高値（節目・抵抗線。株価チャート大全由来）
  B) ATR×3        … 値動きの荒さに比例させた目標（ボラティリティ調整）
  C) フィボナッチ  … 直近スイングの値幅を127.2%投影（株価チャート大全由来）
  D) 固定+10.3%    … 現行の通知が使っている値（比較用のベースライン）

使い方:
    python3 analyze_targets.py [トレード明細CSV]
      省略時は output/_universe_max_trades.csv（現行ルール・944銘柄・26年）。
      無ければ次で作る:
        python3 backtest.py timesl either trend marketadx volume rs sl10 max \
                --tickers universe.csv
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from trade_data import load_trades

from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
# ⚠️ 2026-09-05まで、ここは **日経225・5年・536件** の古いファイル
# （backtest_trades_..._5y_20260829.csv）を指していた。
# 一方 notify.py の TARGET_HIT_RATE_BANDS / DAYS_TO_TARGET_* は
# **944銘柄ユニバース・26年**で取り直した値だったため、
# REQUIREMENTS の「取り直し方」に従って `python3 analyze_targets.py` を
# 実行すると、**古い母集団の数値が出て定数が退化する**状態だった。
# 現行ルールの正規のトレード明細を既定にする（4.4-56）。
DEFAULT_TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"

SUSPICIOUS_RETURN_THRESHOLD = 500.0
SWING_LOOKBACK = 60      # 直近高値・スイング判定に使う期間
ATR_MULTIPLE = 3.0
FIB_EXPANSION = 1.272    # フィボナッチ・エクスパンション127.2%
FIXED_TARGET_PCT = 10.3  # 現行の通知が使っている固定値


def calc_atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = hist["Close"].shift(1)
    tr = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - prev_close).abs(),
        (hist["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def targets_at(hist: pd.DataFrame, atr: pd.Series, i: int, entry: float) -> dict:
    """エントリー日（位置i）時点で計算できる各方式の目標株価を返す。"""
    win_hi = hist["High"].iloc[max(0, i - SWING_LOOKBACK):i + 1]
    win_lo = hist["Low"].iloc[max(0, i - SWING_LOOKBACK):i + 1]
    swing_high = float(win_hi.max())
    swing_low = float(win_lo.min())

    out = {}
    # A) 直近高値。すでに高値を更新している場合は目標にならないので除外
    out["直近高値"] = swing_high if swing_high > entry else None
    # B) ATR×3
    a = float(atr.iloc[i])
    out["ATR×3"] = entry + ATR_MULTIPLE * a if a == a and a > 0 else None
    # C) フィボナッチ・エクスパンション（安値→高値の値幅を127.2%投影）
    rng = swing_high - swing_low
    out["フィボ127.2%"] = swing_low + rng * FIB_EXPANSION if rng > 0 else None
    # D) 固定%
    out["固定+10.3%"] = entry * (1 + FIXED_TARGET_PCT / 100)
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRADES
    # 汚染データの検知つきで読む（4.4-48）。あり得ない値なら例外で止まる
    df = load_trades(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD]

    print(f"対象: {path.name} / {len(df)}トレード")
    print("各トレードについてエントリー時点の目標を計算し、保有期間中の高値が")
    print("そこへ到達したかを判定します。銘柄ごとに株価データを取得します…\n")

    rows = []
    codes = sorted(df["code"].unique())
    fetched = fetch_histories(codes, period="6y")
    for n, code in enumerate(codes, 1):
        try:
            hist = fetched.get(code)
            if hist is None or len(hist) < SWING_LOOKBACK + 10:
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            atr = calc_atr(hist)
            idx = hist.index
            for _, t in df[df["code"] == code].iterrows():
                pos = idx.searchsorted(t["entry_date"])
                if pos >= len(idx) or pos < SWING_LOOKBACK:
                    continue
                entry = float(t["entry_price"])
                tg = targets_at(hist, atr, pos, entry)
                # 保有期間中の高値
                end = idx.searchsorted(t["exit_date"])
                hi = float(hist["High"].iloc[pos:end + 1].max()) if end >= pos else entry
                rec = {"code": code, "return_pct": t["return_pct"], "max_high": hi,
                       "entry": entry}
                for k, v in tg.items():
                    rec[k] = v
                    rec[f"hit_{k}"] = (v is not None and hi >= v)
                    rec[f"pct_{k}"] = ((v - entry) / entry * 100) if v is not None else None
                    # 目標に何営業日で到達したか（到達しなかった場合はNone）
                    days = None
                    if v is not None and end >= pos:
                        window = hist["High"].iloc[pos:end + 1]
                        reached = window[window >= v]
                        if len(reached):
                            days = int(idx.searchsorted(reached.index[0]) - pos)
                    rec[f"days_{k}"] = days
                rows.append(rec)
        except Exception:
            pass
        if n % 50 == 0:
            print(f"  {n}/{len(codes)}銘柄")

    res = pd.DataFrame(rows)
    if res.empty:
        print("集計できるトレードがありませんでした。")
        return

    # 再集計できるよう明細を保存する（株価の再取得は数分かかるため）
    detail_path = BASE_DIR / "output" / "target_analysis_detail.csv"
    res.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"明細を保存しました: {detail_path}\n")

    print(f"\n集計対象: {len(res)}トレード\n")
    print("=" * 74)
    print("各方式の目標の「高さ」と「到達率」")
    print("=" * 74)
    print(f'{"方式":<16}{"目標の高さ":>12}{"到達率":>9}{"算出可能":>10}')
    for k in ["直近高値", "ATR×3", "フィボ127.2%", "固定+10.3%"]:
        sub = res[res[k].notna()]
        if sub.empty:
            continue
        print(f'{k:<16}{sub[f"pct_{k}"].median():>10.1f}%{sub[f"hit_{k}"].mean()*100:>8.1f}%'
              f'{len(sub)/len(res)*100:>9.0f}%')
    print("※目標の高さ＝エントリー価格から何%上かの中央値。到達率＝保有期間中に")
    print("  高値がそこへ届いた割合。算出可能＝その方式で目標を出せたトレードの割合")

    print("\n" + "=" * 74)
    print("目標の高さ別の到達率（方式によらず、目標が高いほど届きにくいはず）")
    print("=" * 74)
    band = pd.concat([
        pd.DataFrame({"pct": res[f"pct_{k}"], "hit": res[f"hit_{k}"], "方式": k})
        for k in ["直近高値", "ATR×3", "フィボ127.2%"]
    ]).dropna()
    band["帯"] = pd.cut(band["pct"], [0, 5, 10, 15, 20, 30, 1000],
                        labels=["〜5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%〜"])
    tbl = band.groupby("帯", observed=True).agg(件数=("hit", "size"), 到達率=("hit", "mean"))
    tbl["到達率"] = (tbl["到達率"] * 100).round(1)
    print(tbl.to_string())

    print("\n" + "=" * 74)
    print("採用方式（ATR×3）の目標の高さ別・到達率 ※通知に埋め込む値")
    print("=" * 74)
    a = res[["pct_ATR×3", "hit_ATR×3"]].dropna()
    a.columns = ["pct", "hit"]
    a["帯"] = pd.cut(a["pct"], [0, 4, 6, 8, 10, 1000],
                     labels=["〜4%", "4-6%", "6-8%", "8-10%", "10%〜"])
    t2 = a.groupby("帯", observed=True).agg(件数=("hit", "size"), 到達率=("hit", "mean"))
    t2["到達率"] = (t2["到達率"] * 100).round(1)
    print(t2.to_string())
    print("※通知では銘柄ごとの目標の高さに応じてこの到達率を出し分ける")

    print("\n" + "=" * 74)
    print("目標に到達するまでの営業日数（到達したトレードのみ） ※通知に埋め込む値")
    print("=" * 74)
    d = res[["pct_ATR×3", "days_ATR×3"]].dropna()
    d.columns = ["pct", "days"]
    d["帯"] = pd.cut(d["pct"], [0, 6, 8, 10, 1000],
                     labels=["〜6%", "6-8%", "8-10%", "10%〜"])
    t3 = d.groupby("帯", observed=True).agg(
        件数=("days", "size"), 中央値=("days", "median"),
        平均=("days", "mean"), 上位25=("days", lambda s: s.quantile(0.25)),
        上位75=("days", lambda s: s.quantile(0.75)))
    print(t3.round(1).to_string())
    print(f"\n全体: 中央値{d['days'].median():.0f}営業日 / 平均{d['days'].mean():.1f}営業日")
    print("※到達しなかったトレードは含まない（含めると「到達しない」が混ざり意味が変わる）")


if __name__ == "__main__":
    main()
