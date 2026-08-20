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


def fetch_market_regime(period: str = "5y") -> pd.Series:
    """
    日経平均自体のトレンド状態（マーケットレジームフィルター）。
    指数の終値が100日線より上なら「地合いが良い（上昇トレンド）」と判定する。
    出典: market regime filter は複数の情報源で「戦略の成否はシグナルより
    どんな相場環境で使うか次第」として有効性が示されている手法。
    """
    idx = yf.Ticker("^N225").history(period=period)
    close = idx["Close"]
    sma100 = close.rolling(100).mean()
    return close > sma100


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX（Average Directional Index）。トレンドの向きではなく「強さ」を
    0〜100で示す指標（一般に20〜25以上でトレンドが強い、それ未満はレンジ相場
    とされる）。Wilderの平滑化（EWMA, alpha=1/period）で近似計算する。
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def fetch_market_regime_adx(period: str = "5y", adx_threshold: float = 20.0) -> pd.Series:
    """
    マーケットレジームフィルターのADX版。「終値が100日線より上」（方向）に
    加えて、ADXが閾値以上（トレンドに十分な勢いがある＝レンジ相場でない）
    の日だけを「地合いが良い」と判定する。方向感の乏しい年（レンジ相場）を
    除外する狙い（PPP版では効果が薄かったため、別角度として試す）。
    """
    idx = yf.Ticker("^N225").history(period=period)
    close = idx["Close"]
    sma100 = close.rolling(100).mean()
    is_uptrend = close > sma100
    adx = calc_adx(idx["High"], idx["Low"], close)
    is_strong_trend = adx > adx_threshold
    return is_uptrend & is_strong_trend


def fetch_market_regime_ppp(period: str = "5y", min_matches: int = 3) -> pd.Series:
    """
    マーケットレジームフィルターの強化版。「終値が100日線より上」という
    緩い基準ではなく、日経平均自体にPPP（5>10>20>50>100日線の並び）判定を
    適用し、min_matches組以上揃っている＝方向感の強い上昇トレンドの日だけを
    「地合いが良い」と判定する。方向感の乏しい年（レンジ相場）を除外する狙い。
    """
    idx = yf.Ticker("^N225").history(period=period)
    close = idx["Close"]
    ma_periods = (5, 10, 20, 50, 100)
    sma = {n: close.rolling(n).mean() for n in ma_periods}
    matches = sum(
        (sma[ma_periods[i]] > sma[ma_periods[i + 1]]).astype(int)
        for i in range(len(ma_periods) - 1)
    )
    return matches >= min_matches


def backtest_ticker(
    code: str,
    name: str,
    hist: pd.DataFrame,
    exit_period: int = 5,
    trend_filter: bool = False,
    min_ppp_matches: int = 3,
    exit_mode: str = "ma",
    market_regime: pd.Series = None,
    volume_filter: bool = False,
    volume_multiple: float = 1.5,
) -> list[dict]:
    """
    エントリー: 下半身シグナル（5日線が上向き＋陽線で5日線を上抜け）で固定。
    trend_filter=True の場合、さらに「PPP一致度がmin_ppp_matches以上」
    「株価が100日線より上」の強いトレンド条件を満たす下半身だけを採用する。
    market_regime を渡した場合、日経平均自体が上昇トレンドの日だけ新規エントリーを許可する。
    volume_filter=True の場合、下半身当日の出来高が直近20日平均の
    volume_multiple倍以上（出来高を伴ったブレイク）の場合だけ採用する。

    エグジット:
      exit_mode="ma"        終値が exit_period 日線を割り込んだら手仕舞い
      exit_mode="ppp_break" 5日線が20日線を下抜けたら手仕舞い（PPPの並びが崩れる＝一番緩い条件）
    """
    close = hist["Close"]
    open_ = hist["Open"]
    volume = hist["Volume"]
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma_exit = close.rolling(exit_period).mean()
    vol_avg20 = volume.rolling(20).mean()

    ma_periods = (5, 10, 20, 50, 100)
    sma = {n: close.rolling(n).mean() for n in ma_periods} if trend_filter else None

    regime_aligned = market_regime.reindex(close.index, method="ffill") if market_regime is not None else None

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

            if signal and regime_aligned is not None:
                regime_ok = regime_aligned.iloc[i]
                signal = bool(regime_ok) if pd.notna(regime_ok) else False

            if signal and volume_filter:
                if pd.isna(vol_avg20.iloc[i]) or vol_avg20.iloc[i] == 0:
                    signal = False
                else:
                    signal = volume.iloc[i] >= vol_avg20.iloc[i] * volume_multiple

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
    trend_filter = "trend" in sys.argv[2:]
    use_market_regime_ppp = "marketppp" in sys.argv[2:]
    use_market_regime_adx = "marketadx" in sys.argv[2:]
    use_market_regime = "market" in sys.argv[2:] or use_market_regime_ppp or use_market_regime_adx
    use_volume_filter = "volume" in sys.argv[2:]
    period_args = [a for a in sys.argv[2:] if a == "max" or (a.endswith("y") and a[:-1].isdigit())]
    history_period = period_args[0] if period_args else HISTORY_PERIOD

    tickers = load_tickers()
    filter_desc = "PPP3/4以上+100日線上のみ" if trend_filter else "フィルターなし"
    if use_market_regime_ppp:
        filter_desc += "+日経平均自体がPPP3/4以上の強いトレンドの日のみ"
    elif use_market_regime_adx:
        filter_desc += "+日経平均がADX20超の強いトレンドの日のみ"
    elif use_market_regime:
        filter_desc += "+日経平均が上昇トレンドの日のみ"
    if use_volume_filter:
        filter_desc += "+出来高が20日平均の1.5倍以上"
    exit_desc = "5日線が20日線を下抜け（PPP崩れ）" if exit_mode == "ppp_break" else f"{exit_period}日線割れ"
    print(f"{len(tickers)}銘柄で下半身バックテストを実行します（過去{history_period}、エグジット={exit_desc}、エントリー条件={filter_desc}）…")

    market_regime = None
    if use_market_regime_ppp:
        print("日経平均のデータを取得中（PPP判定）…")
        market_regime = fetch_market_regime_ppp(history_period)
    elif use_market_regime_adx:
        print("日経平均のデータを取得中（ADX判定）…")
        market_regime = fetch_market_regime_adx(history_period)
    elif use_market_regime:
        print("日経平均のデータを取得中…")
        market_regime = fetch_market_regime(history_period)

    all_trades = []
    bh_returns = []

    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            hist = yf.Ticker(code).history(period=history_period)
            if len(hist) < 120:
                print(f"  [{i}/{len(tickers)}] {name} ({code}) データ不足のためスキップ")
                continue
            trades = backtest_ticker(
                code, name, hist, exit_period=exit_period, trend_filter=trend_filter,
                exit_mode=exit_mode, market_regime=market_regime, volume_filter=use_volume_filter,
            )
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
    if use_market_regime_ppp:
        market_suffix = "_marketppp"
    elif use_market_regime_adx:
        market_suffix = "_marketadx"
    elif use_market_regime:
        market_suffix = "_market"
    else:
        market_suffix = ""
    suffix = ("_trend" if trend_filter else "") + market_suffix + ("_volume" if use_volume_filter else "")
    tag = "ppp" if exit_mode == "ppp_break" else f"exit{exit_period}"
    out_path = OUTPUT_DIR / f"backtest_trades_{tag}{suffix}_{history_period}_{dt.date.today():%Y%m%d}.csv"
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
