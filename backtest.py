"""
バックテスト：「下半身」エントリー×「5日線割れ」エグジットの有効性検証

tickers.csv の銘柄について過去5年分のデータを取得し、
  買い: 下半身シグナル点灯（5日線が上向き＋陽線で5日線を上抜け）
  売り: 終値が5日線を割り込む
というルールを機械的に繰り返した場合の成績を集計する。
screen.py が「買いタイミング／売りタイミング」として提示しているルールと同一。

使い方:
    python backtest.py
出力:
    output/backtest_trades_YYYYMMDD.csv（全トレード明細）
    標準出力に集計サマリー
"""

import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
OUTPUT_DIR = BASE_DIR / "output"

HISTORY_PERIOD = "5y"
MIN_WARMUP_DAYS = 10  # 5日線の傾き判定に必要な最低本数


def load_tickers():
    with open(TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def backtest_ticker(
    code: str,
    name: str,
    hist: pd.DataFrame,
    exit_period: int = 5,
    trend_filter: bool = False,
    min_ppp_matches: int = 3,
    exit_mode: str = "ma",
) -> list[dict]:
    """
    エントリー: 下半身シグナル（5日線が上向き＋陽線で5日線を上抜け）で固定。
    trend_filter=True の場合、さらに「PPP一致度がmin_ppp_matches以上」
    「株価が100日線より上」の強いトレンド条件を満たす下半身だけを採用する。

    エグジット:
      exit_mode="ma"        終値が exit_period 日線を割り込んだら手仕舞い
      exit_mode="ppp_break" 5日線が20日線を下抜けたら手仕舞い（PPPの並びが崩れる＝一番緩い条件）
    """
    close = hist["Close"]
    open_ = hist["Open"]
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma_exit = close.rolling(exit_period).mean()

    ma_periods = (5, 10, 20, 50, 100)
    sma = {n: close.rolling(n).mean() for n in ma_periods} if trend_filter else None

    trades = []
    in_position = False
    entry_price = entry_date = None

    for i in range(MIN_WARMUP_DAYS, len(close)):
        if pd.isna(sma5.iloc[i]) or pd.isna(sma5.iloc[i - 4]) or pd.isna(sma_exit.iloc[i]) or pd.isna(sma20.iloc[i]):
            continue
        c, o = close.iloc[i], open_.iloc[i]
        prev_c, prev_s = close.iloc[i - 1], sma5.iloc[i - 1]

        if not in_position:
            crossed_above = prev_c <= prev_s and c > sma5.iloc[i]
            is_bullish = c > o
            slope_up = sma5.iloc[i] > sma5.iloc[i - 4]
            signal = crossed_above and is_bullish and slope_up

            if signal and trend_filter:
                if pd.isna(sma[100].iloc[i]):
                    signal = False
                else:
                    ppp_matches = sum(
                        sma[ma_periods[j]].iloc[i] > sma[ma_periods[j + 1]].iloc[i]
                        for j in range(len(ma_periods) - 1)
                    )
                    signal = ppp_matches >= min_ppp_matches and c > sma[100].iloc[i]

            if signal:
                in_position = True
                entry_price = float(c)
                entry_date = close.index[i]
        else:
            should_exit = (sma5.iloc[i] < sma20.iloc[i]) if exit_mode == "ppp_break" else (c < sma_exit.iloc[i])
            if should_exit:
                exit_price = float(c)
                exit_date = close.index[i]
                ret_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "code": code,
                    "name": name,
                    "entry_date": entry_date.date(),
                    "exit_date": exit_date.date(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "holding_days": (exit_date - entry_date).days,
                })
                in_position = False

    return trades


def buy_and_hold_return(hist: pd.DataFrame) -> float:
    close = hist["Close"]
    return (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100


def main():
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "5"
    exit_mode = "ppp_break" if arg1 == "ppp" else "ma"
    exit_period = 20 if exit_mode == "ppp_break" else int(arg1)
    trend_filter = len(sys.argv) > 2 and sys.argv[2] == "trend"

    tickers = load_tickers()
    filter_desc = "PPP3/4以上+100日線上のみ" if trend_filter else "フィルターなし"
    exit_desc = "5日線が20日線を下抜け（PPP崩れ）" if exit_mode == "ppp_break" else f"{exit_period}日線割れ"
    print(f"{len(tickers)}銘柄で下半身バックテストを実行します（過去{HISTORY_PERIOD}、エグジット={exit_desc}、エントリー条件={filter_desc}）…")

    all_trades = []
    bh_returns = []

    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            hist = yf.Ticker(code).history(period=HISTORY_PERIOD)
            if len(hist) < 120:
                print(f"  [{i}/{len(tickers)}] {name} ({code}) データ不足のためスキップ")
                continue
            trades = backtest_ticker(code, name, hist, exit_period=exit_period, trend_filter=trend_filter, exit_mode=exit_mode)
            all_trades.extend(trades)
            bh_returns.append(buy_and_hold_return(hist))
            print(f"  [{i}/{len(tickers)}] {name} ({code}) トレード数: {len(trades)}")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {name} ({code}) 取得失敗: {e}")

    if not all_trades:
        print("トレードが1件も発生しませんでした。")
        return

    df = pd.DataFrame(all_trades)

    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = "_trend" if trend_filter else ""
    tag = "ppp" if exit_mode == "ppp_break" else f"exit{exit_period}"
    out_path = OUTPUT_DIR / f"backtest_trades_{tag}{suffix}_{dt.date.today():%Y%m%d}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    win_rate = (df["return_pct"] > 0).mean() * 100
    avg_return = df["return_pct"].mean()
    median_return = df["return_pct"].median()
    avg_holding = df["holding_days"].mean()
    avg_bh = sum(bh_returns) / len(bh_returns) if bh_returns else float("nan")

    print(f"\n完了: {out_path}")
    print(f"\n=== 下半身シグナル バックテスト結果（エグジット={exit_desc}） ===")
    print(f"総トレード数: {len(df)}件（対象{len(bh_returns)}銘柄）")
    print(f"勝率: {win_rate:.1f}%")
    print(f"平均リターン: {avg_return:+.2f}% / トレード")
    print(f"リターン中央値: {median_return:+.2f}% / トレード")
    print(f"平均保有日数: {avg_holding:.1f}日")
    print(f"（参考）対象期間の単純バイ&ホールド平均リターン: {avg_bh:+.2f}%")

    print("\n--- リターン上位10トレード ---")
    print(df.sort_values("return_pct", ascending=False).head(10)[
        ["name", "code", "entry_date", "exit_date", "return_pct", "holding_days"]
    ].to_string(index=False))

    print("\n--- リターン下位10トレード ---")
    print(df.sort_values("return_pct", ascending=True).head(10)[
        ["name", "code", "entry_date", "exit_date", "return_pct", "holding_days"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
