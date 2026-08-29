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

from price_cache import fetch_histories, fetch_history

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
OUTPUT_DIR = BASE_DIR / "output"

HISTORY_PERIOD = "5y"
MIN_WARMUP_DAYS = 10  # 5日線の傾き判定に必要な最低本数

# 1トレードのリターンがこれを超えたら警告を出す（yfinanceの株式分割データ
# 不整合等による異常値を検知するため。2026-08-22、東京海上HD(8766)の
# 2005年データで実際に約18,700%という不正な値を検出した経緯あり）
SUSPICIOUS_RETURN_THRESHOLD = 500.0

# 1日でこれを超える値動きは実在しないので、株式分割データの不整合とみなして
# その銘柄を検証対象から外す（東京海上HD/8766は初値が-0.18円、日本航空/9201は
# 1日で191,400%と記録されている）。portfolio_sim.py もこの値を使う
MAX_PLAUSIBLE_DAILY_MOVE = 0.8


def load_tickers(path=None):
    """既定は日経225（tickers.csv）。universe.csv 等を渡せば対象を差し替えられる。"""
    with open(path or TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fetch_market_regime(period: str = "5y") -> pd.Series:
    """
    日経平均自体のトレンド状態（マーケットレジームフィルター）。
    指数の終値が100日線より上なら「地合いが良い（上昇トレンド）」と判定する。
    出典: market regime filter は複数の情報源で「戦略の成否はシグナルより
    どんな相場環境で使うか次第」として有効性が示されている手法。
    """
    idx = fetch_history("^N225", period=period)
    close = idx["Close"]
    sma100 = close.rolling(100).mean()
    return close > sma100


def fetch_sector(code: str) -> str:
    """業種（GICSセクター、yfinance由来）。取得できない場合は'Unknown'"""
    try:
        return yf.Ticker(code).info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def build_sector_close(hist_map: dict, sector_map: dict) -> dict:
    """
    セクターごとに、そのセクター内銘柄の正規化株価（開始日を100とする）の
    単純平均を合成し、自前の「セクター指数」として使う（公式指数ではなく
    tickers.csvの225銘柄内での近似）。
    """
    sector_series = {}
    for sector in set(sector_map.values()):
        normalized = []
        for code, s in sector_map.items():
            if s != sector:
                continue
            hist = hist_map.get(code)
            if hist is None or hist.empty:
                continue
            close = hist["Close"]
            normalized.append(close / close.iloc[0])
        if len(normalized) < 3:
            continue  # 銘柄数が少なすぎるセクターは指数として不安定なため除外
        combined = pd.concat(normalized, axis=1)
        sector_series[sector] = combined.mean(axis=1)
    return sector_series


def build_sector_regime(sector_close_map: dict, nikkei_close: pd.Series, period: int = 50) -> dict:
    """セクター指数÷日経平均の比率が自身のperiod日移動平均より上か（セクター全体が市場をアウトパフォーム中か）"""
    regime_map = {}
    for sector, sector_close in sector_close_map.items():
        nikkei_aligned = nikkei_close.reindex(sector_close.index, method="ffill")
        ratio = sector_close / nikkei_aligned
        sma = ratio.rolling(period).mean()
        regime_map[sector] = ratio > sma
    return regime_map


def fetch_earnings_dates(code: str) -> set:
    """
    決算発表日の集合（date型）を取得する。yfinanceの決算日データは
    過去5年分程度しか遡れないため、このフィルターは5年バックテストのみで
    検証する（10年・26年での頑健性チェックはできない）。
    """
    try:
        ed = yf.Ticker(code).earnings_dates
        if ed is None or ed.empty:
            return set()
        return {d.date() for d in ed.index}
    except Exception:
        return set()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR（Average True Range）。Wilderの平滑化で計算する、値動きの荒さの指標"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def fetch_nikkei_close(period: str = "5y") -> pd.Series:
    """レラティブストレングス計算用に日経平均の終値だけを取得する"""
    return fetch_history("^N225", period=period)["Close"]


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


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（Relative Strength Index）。Wilderの平滑化（EWMA, alpha=1/period）で
    計算する、0〜100の過熱感指標。株価チャート大全（戸松信博監修）に
    従い、標準期間14日・70%以上を「買われすぎ」の目安とする。
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_takuri_daki_confirmed(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 10
) -> pd.Series:
    """
    株価チャート大全（戸松信博監修）の「たくり線」（下ヒゲの長い小陽線/
    小陰線＝底値圏での押し目買い意識を示す）に続けて「抱き線」（直後の
    包み足＝反転をより明確に確定させる）が出現したかを検出する。
    直近lookback日以内にこの組み合わせが1回でも発生していればTrueとする
    （下半身エントリーの信頼度を補強する候補シグナルとして検証）。
    """
    body = (close - open_).abs()
    total_range = (high - low).replace(0, pd.NA)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)

    is_takuri = (
        (lower_shadow >= total_range * 0.6)
        & (body <= total_range * 0.3)
        & (upper_shadow <= total_range * 0.15)
    ).fillna(False)

    body_lo = pd.concat([open_, close], axis=1).min(axis=1)
    body_hi = pd.concat([open_, close], axis=1).max(axis=1)
    prev_body_lo = body_lo.shift(1)
    prev_body_hi = body_hi.shift(1)
    is_daki = (
        (body_lo <= prev_body_lo) & (body_hi >= prev_body_hi) & (body > body.shift(1))
    ).fillna(False)

    # 「たくり線の翌日に抱き線」が起きた日を1つの複合シグナル日として立てる
    combo_day = (is_takuri.shift(1).fillna(False)) & is_daki
    return combo_day.rolling(lookback, min_periods=1).max().astype(bool)


def calc_fib_retracement_pct(
    close: pd.Series, swing_lookback: int = 60, pullback_lookback: int = 15
) -> pd.Series:
    """
    株価チャート大全のフィボナッチ・リトレースメント（38.2%/50%/61.8%）を
    踏まえ、直近swing_lookback日の高値（スイングハイの近似）からの、
    直近pullback_lookback日の安値までの押しの深さを%で返す簡易近似。
    退行率 = (スイングハイ - 直近安値) / スイングハイ × 100
    """
    swing_high = close.rolling(swing_lookback).max()
    pullback_low = close.rolling(pullback_lookback).min()
    return (swing_high - pullback_low) / swing_high * 100


def calc_ichimoku_sanyaku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    一目均衡表の「三役好転」（株価チャート大全 由来）。以下3条件が揃った日を
    強い買いサインとする。転換した瞬間（前日は不成立）だけTrueを返す。
      ①転換線（9日）が基準線（26日）を上抜け
      ②遅行線（終値を26日前にずらした線）が26日前の株価を上抜け
      ③株価が雲（先行スパン1と2の間）を上抜け
    """
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)

    cond = (
        (tenkan > kijun)
        & (close > close.shift(26))   # 遅行線が26日前の株価より上
        & (close > cloud_top)
    ).fillna(False)
    return (cond & ~cond.shift(1).fillna(False)).astype(bool)


def calc_macd_golden_cross(close: pd.Series) -> pd.Series:
    """
    MACD（12日EMA−26日EMA）がシグナル線（MACDの9日EMA）を上抜けた日
    ＝ゴールデンクロス（株価チャート大全 由来）。
    """
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd.ewm(span=9, adjust=False).mean()
    above = macd > signal_line
    return (above & ~above.shift(1).fillna(False)).fillna(False).astype(bool)


def calc_bandwalk_start(close: pd.Series, period: int = 20) -> pd.Series:
    """
    ボリンジャーバンドの「バンドウォーク」入り（株価チャート大全 由来）。
    終値が+1σを上抜けた日＝強いトレンドに乗り始めたサインとして扱う
    （±3σの逆張りとは逆に、バンドウォークは順張りに使う、と同書にある）。
    """
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper1 = ma + sd
    above = close > upper1
    return (above & ~above.shift(1).fillna(False)).fillna(False).astype(bool)


def fetch_market_regime_adx(period: str = "5y", adx_threshold: float = 20.0) -> pd.Series:
    """
    マーケットレジームフィルターのADX版。「終値が100日線より上」（方向）に
    加えて、ADXが閾値以上（トレンドに十分な勢いがある＝レンジ相場でない）
    の日だけを「地合いが良い」と判定する。方向感の乏しい年（レンジ相場）を
    除外する狙い（PPP版では効果が薄かったため、別角度として試す）。
    """
    idx = fetch_history("^N225", period=period)
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
    idx = fetch_history("^N225", period=period)
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
    nikkei_close: pd.Series = None,
    rs_period: int = 50,
    atr_multiple: float = 2.5,
    earnings_dates: set = None,
    earnings_avoid_days: int = 2,
    sector_regime: pd.Series = None,
    time_stop_days: int = 60,
    rsi_filter: bool = False,
    rsi_threshold: float = 70.0,
    dev_filter: bool = False,
    dev_threshold: float = 20.0,
    dev_ma_period: int = 25,
    exit_dev_threshold: float = 20.0,
    candle_filter: bool = False,
    candle_lookback: int = 10,
    fib_filter: bool = False,
    fib_low: float = 25.0,
    fib_high: float = 65.0,
    profit_target_pct: float = 10.0,
    stop_loss_pct: float = 10.0,
    entry_mode: str = "kahanshin",
    pullback_tolerance: float = 2.0,
    cost_pct: float = 0.0,
    side: str = "long",
) -> list[dict]:
    """
    エントリー:
      entry_mode="kahanshin"（既定）下半身シグナル（5日線が上向き＋陽線で
        5日線を上抜け）。ブレイクした瞬間を買う順張り型。
      entry_mode="either" 下半身と押し目買いのどちらかが点灯したら買う。
        両方式の良いところを取れるか（＝シグナル件数を増やしつつ勝率を
        保てるか）を検証するための併用モード。
      entry_mode="pullback" 押し目買い型（グランビルの法則②③由来）。
        上昇トレンド中に株価が20日線まで押し（安値が20日線＋
        pullback_tolerance%以内まで下落）、当日陽線で20日線より上に
        戻して反発したところを買う。下半身とは「ブレイクを買うか、
        押し目を買うか」という根本的に逆の発想。
    trend_filter=True の場合、さらに「PPP一致度がmin_ppp_matches以上」
    「株価が100日線より上」の強いトレンド条件を満たす下半身だけを採用する。
    market_regime を渡した場合、日経平均自体が上昇トレンドの日だけ新規エントリーを許可する。
    volume_filter=True の場合、下半身当日の出来高が直近20日平均の
    volume_multiple倍以上（出来高を伴ったブレイク）の場合だけ採用する。
    nikkei_close を渡した場合、レラティブストレングス（個別銘柄÷日経平均の
    比率線）が自身のrs_period日移動平均より上（＝直近で日経平均をアウト
    パフォームしている）の場合だけ採用する。
    earnings_dates を渡した場合、決算発表日の前後earnings_avoid_days営業日は
    新規エントリーを見送る（決算ギャップによる値飛びを避ける狙い）。
    sector_regime を渡した場合、その銘柄が属するセクター全体が日経平均を
    アウトパフォーム中の日だけ新規エントリーを許可する。
    rsi_filter=True の場合、RSI（14日）が rsi_threshold 以上（買われすぎ）の
    日は新規エントリーを見送る（株価チャート大全のRSI「70〜80%以上で
    買われすぎ」を踏まえたフィルター）。
    dev_filter=True の場合、株価が dev_ma_period 日線から dev_threshold %
    以上上方乖離している日は新規エントリーを見送る（同書のかい離率・
    グランビルの法則④を踏まえた「追いかけ買い」回避フィルター）。
    candle_filter=True の場合、直近candle_lookback日以内に「たくり線→
    抱き線」の複合シグナル（同書由来）が出現していない下半身は見送る。
    fib_filter=True の場合、フィボナッチ・リトレースメント（同書由来）を
    踏まえ、直近の押しの深さが fib_low〜fib_high %の範囲に収まっていない
    （浅すぎる・深すぎる押し目の）下半身は見送る。

    エグジット:
      exit_mode="ma"         終値が exit_period 日線を割り込んだら手仕舞い
      exit_mode="ppp_break"  5日線が20日線を下抜けたら手仕舞い（PPPの並びが崩れる＝一番緩い条件）
      exit_mode="atr_trail"  保有中の最高値から atr_multiple×ATR 下落したら手仕舞い（トレーリングストップ）
      exit_mode="ppp_or_atr" 上記2つのどちらか早い方で手仕舞い
      exit_mode="time_stop"  保有日数が time_stop_days に達したら手仕舞い（純粋なタイムストップ）
      exit_mode="ppp_or_time" PPP崩れ or タイムストップの早い方で手仕舞い
      exit_mode="dev_exit"   株価が dev_ma_period 日線から exit_dev_threshold %
                              以上上方乖離したら手仕舞い（グランビルの法則④の
                              逆張り利確を踏まえた早期利確エグジット）
      exit_mode="ppp_or_dev" PPP崩れ or 上記乖離エグジットの早い方で手仕舞い
      exit_mode="time_and_dev" タイムストップ or 乖離エグジットの早い方
                              （PPP崩れは見ない＝一時的な下げで振り落とされない）
      exit_mode="ppp_or_time_or_dev" PPP崩れ or タイムストップ or 乖離エグジット
                              の3つのうち最も早いもので手仕舞い
      exit_mode="profit_target" 含み益が profit_target_pct % に達したら利確
                              （純粋な固定利確目標。損切りはしないので
                              負けポジションは期間末まで持ち越される点に注意）
      exit_mode="ppp_or_target" PPP崩れ or 固定利確目標の早い方で手仕舞い
      exit_mode="time_or_sl" タイムストップ or 損切り（含み損が stop_loss_pct %
                              に達したら手仕舞い）の早い方。タイムストップ単独は
                              損切りが一切効かず平均損失が約2倍になるため、
                              その弱点を補う折衷案
      exit_mode="time_dev_sl" タイムストップ or 乖離利確 or 損切りの最も早いもの
    """
    close = hist["Close"]
    open_ = hist["Open"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma_exit = close.rolling(exit_period).mean()
    vol_avg20 = volume.rolling(20).mean()
    atr = calc_atr(high, low, close) if exit_mode in ("atr_trail", "ppp_or_atr") else None
    rsi = calc_rsi(close) if rsi_filter else None
    sma_dev = (
        close.rolling(dev_ma_period).mean()
        if dev_filter or exit_mode in (
            "dev_exit", "ppp_or_dev", "time_and_dev", "ppp_or_time_or_dev", "time_dev_sl"
        )
        else None
    )
    candle_confirmed = (
        calc_takuri_daki_confirmed(open_, high, low, close, candle_lookback)
        if candle_filter else None
    )
    ichimoku_sig = calc_ichimoku_sanyaku(high, low, close) if "ichimoku" in entry_mode else None
    macd_sig = calc_macd_golden_cross(close) if "macd" in entry_mode else None
    bandwalk_sig = calc_bandwalk_start(close) if "bandwalk" in entry_mode else None
    fib_retracement = calc_fib_retracement_pct(close) if fib_filter else None

    ma_periods = (5, 10, 20, 50, 100)
    sma = {n: close.rolling(n).mean() for n in ma_periods} if trend_filter else None

    regime_aligned = market_regime.reindex(close.index, method="ffill") if market_regime is not None else None
    sector_regime_aligned = sector_regime.reindex(close.index, method="ffill") if sector_regime is not None else None

    rs_signal = None
    if nikkei_close is not None:
        nikkei_aligned = nikkei_close.reindex(close.index, method="ffill")
        rs_ratio = close / nikkei_aligned
        rs_sma = rs_ratio.rolling(rs_period).mean()
        rs_signal = rs_ratio > rs_sma

    trades = []
    in_position = False
    entry_price = entry_date = None
    highest_since_entry = None

    for i in range(MIN_WARMUP_DAYS, len(close)):
        if pd.isna(sma5.iloc[i]) or pd.isna(sma5.iloc[i - 4]) or pd.isna(sma_exit.iloc[i]) or pd.isna(sma20.iloc[i]):
            continue
        c, o = close.iloc[i], open_.iloc[i]
        prev_c, prev_s = close.iloc[i - 1], sma5.iloc[i - 1]

        if not in_position:
            if side == "short":
                # 空売り側は買い側を上下反転させた条件にする。
                # 逆下半身：5日線が下向き＋陰線で5日線を下抜けた日に売る
                kahanshin_sig = bool(
                    prev_c >= prev_s and c < sma5.iloc[i]
                    and c < o and sma5.iloc[i] < sma5.iloc[i - 4]
                )
                # 戻り売り：下降トレンド中に20日線まで戻して陰線で反落した日
                pullback_sig = bool(
                    high.iloc[i] >= sma20.iloc[i] * (1 - pullback_tolerance / 100)
                    and c < o and c < sma20.iloc[i]
                )
            else:
                # 押し目買い：上昇トレンド中に20日線まで押して反発した日を買う
                pullback_sig = bool(
                    low.iloc[i] <= sma20.iloc[i] * (1 + pullback_tolerance / 100)
                    and c > o and c > sma20.iloc[i]
                )
                # 下半身：5日線が上向き＋陽線で5日線を上抜けた日を買う
                kahanshin_sig = bool(
                    prev_c <= prev_s and c > sma5.iloc[i]
                    and c > o and sma5.iloc[i] > sma5.iloc[i - 4]
                )
            # entry_mode はシグナル名を "+" で連結した集合として扱う
            # （例 "either" = 下半身+押し目買い、"either+macd" = さらにMACD追加）
            if entry_mode == "kahanshin":
                signal = kahanshin_sig
            elif entry_mode == "pullback":
                signal = pullback_sig
            else:
                parts = entry_mode.split("+")
                signal = False
                if "either" in parts or "kahanshin" in parts:
                    signal = signal or kahanshin_sig
                if "either" in parts or "pullback" in parts:
                    signal = signal or pullback_sig
                if "ichimoku" in parts:
                    signal = signal or bool(ichimoku_sig.iloc[i])
                if "macd" in parts:
                    signal = signal or bool(macd_sig.iloc[i])
                if "bandwalk" in parts:
                    signal = signal or bool(bandwalk_sig.iloc[i])

            if signal and trend_filter:
                if pd.isna(sma[100].iloc[i]):
                    signal = False
                elif side == "short":
                    # 空売り側は逆PPP（短期線ほど下）＋100日線より下を条件にする
                    ppp_matches = sum(
                        sma[ma_periods[j]].iloc[i] < sma[ma_periods[j + 1]].iloc[i]
                        for j in range(len(ma_periods) - 1)
                    )
                    signal = ppp_matches >= min_ppp_matches and c < sma[100].iloc[i]
                else:
                    ppp_matches = sum(
                        sma[ma_periods[j]].iloc[i] > sma[ma_periods[j + 1]].iloc[i]
                        for j in range(len(ma_periods) - 1)
                    )
                    signal = ppp_matches >= min_ppp_matches and c > sma[100].iloc[i]

            if signal and regime_aligned is not None:
                regime_ok = regime_aligned.iloc[i]
                # 空売りは地合いが悪い（下降トレンド）ときに仕掛けるので反転させる
                if side == "short":
                    signal = (not bool(regime_ok)) if pd.notna(regime_ok) else False
                else:
                    signal = bool(regime_ok) if pd.notna(regime_ok) else False

            if signal and volume_filter:
                if pd.isna(vol_avg20.iloc[i]) or vol_avg20.iloc[i] == 0:
                    signal = False
                else:
                    signal = volume.iloc[i] >= vol_avg20.iloc[i] * volume_multiple

            if signal and rs_signal is not None:
                rs_ok = rs_signal.iloc[i]
                # 空売りは日経平均をアンダーパフォームしている銘柄を狙う
                if side == "short":
                    signal = (not bool(rs_ok)) if pd.notna(rs_ok) else False
                else:
                    signal = bool(rs_ok) if pd.notna(rs_ok) else False

            if signal and earnings_dates:
                today = close.index[i].date()
                near_earnings = any(
                    abs((today - ed).days) <= earnings_avoid_days for ed in earnings_dates
                )
                signal = not near_earnings

            if signal and sector_regime_aligned is not None:
                sector_ok = sector_regime_aligned.iloc[i]
                signal = bool(sector_ok) if pd.notna(sector_ok) else False

            if signal and rsi_filter:
                if pd.isna(rsi.iloc[i]):
                    signal = False
                else:
                    signal = rsi.iloc[i] < rsi_threshold

            if signal and dev_filter:
                if pd.isna(sma_dev.iloc[i]) or sma_dev.iloc[i] == 0:
                    signal = False
                else:
                    dev_pct = (c - sma_dev.iloc[i]) / sma_dev.iloc[i] * 100
                    signal = dev_pct < dev_threshold

            if signal and candle_filter:
                signal = bool(candle_confirmed.iloc[i])

            if signal and fib_filter:
                fib_pct = fib_retracement.iloc[i]
                signal = pd.notna(fib_pct) and fib_low <= fib_pct <= fib_high

            if signal:
                in_position = True
                entry_price = float(c)
                entry_date = close.index[i]
                highest_since_entry = float(c)
        else:
            highest_since_entry = max(highest_since_entry, float(c))
            ppp_break_hit = sma5.iloc[i] < sma20.iloc[i]
            atr_hit = (
                pd.notna(atr.iloc[i]) and c < highest_since_entry - atr_multiple * atr.iloc[i]
                if atr is not None else False
            )
            time_hit = (close.index[i] - entry_date).days >= time_stop_days
            dev_hit = (
                pd.notna(sma_dev.iloc[i]) and sma_dev.iloc[i] != 0
                and (c - sma_dev.iloc[i]) / sma_dev.iloc[i] * 100 >= exit_dev_threshold
                if sma_dev is not None else False
            )
            # 空売りは値下がりが利益なので、損益の符号を反転させて判定する
            pnl_pct = (
                (entry_price - c) / entry_price * 100 if side == "short"
                else (c - entry_price) / entry_price * 100
            )
            target_hit = pnl_pct >= profit_target_pct
            sl_hit = pnl_pct <= -stop_loss_pct
            if exit_mode == "ppp_break":
                should_exit = ppp_break_hit
            elif exit_mode == "atr_trail":
                should_exit = atr_hit
            elif exit_mode == "ppp_or_atr":
                should_exit = ppp_break_hit or atr_hit
            elif exit_mode == "time_stop":
                should_exit = time_hit
            elif exit_mode == "ppp_or_time":
                should_exit = ppp_break_hit or time_hit
            elif exit_mode == "dev_exit":
                should_exit = dev_hit
            elif exit_mode == "ppp_or_dev":
                should_exit = ppp_break_hit or dev_hit
            elif exit_mode == "time_and_dev":
                should_exit = time_hit or dev_hit
            elif exit_mode == "ppp_or_time_or_dev":
                should_exit = ppp_break_hit or time_hit or dev_hit
            elif exit_mode == "profit_target":
                should_exit = target_hit
            elif exit_mode == "ppp_or_target":
                should_exit = ppp_break_hit or target_hit
            elif exit_mode == "time_or_sl":
                should_exit = time_hit or sl_hit
            elif exit_mode == "time_dev_sl":
                should_exit = time_hit or dev_hit or sl_hit
            else:
                should_exit = c < sma_exit.iloc[i]
            if should_exit:
                exit_price = float(c)
                exit_date = close.index[i]
                # 往復の取引コスト（手数料＋スリッページ）を控除した実質リターン
                if side == "short":
                    ret_pct = (entry_price - exit_price) / entry_price * 100 - cost_pct
                else:
                    ret_pct = (exit_price - entry_price) / entry_price * 100 - cost_pct
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


def parse_config(args) -> dict:
    """
    コマンドライン引数（`sys.argv[1:]` 相当）を設定dictに変換する。
    compare.py が複数条件を1プロセスで回すために、main()から切り出したもの。
    """
    arg1 = args[0] if args else "5"
    rest = args[1:]
    exit_mode_map = {
        "ppp": "ppp_break", "atrtrail": "atr_trail", "pporatr": "ppp_or_atr",
        "timestop": "time_stop", "pportime": "ppp_or_time",
        "devexit": "dev_exit", "ppordev": "ppp_or_dev",
        "timeanddev": "time_and_dev", "pportimeordev": "ppp_or_time_or_dev",
        "target": "profit_target", "pportarget": "ppp_or_target",
        "timesl": "time_or_sl", "timedevsl": "time_dev_sl",
    }
    exit_mode = exit_mode_map.get(arg1, "ma")
    exit_period = 20 if exit_mode != "ma" else int(arg1)
    trend_filter = "trend" in rest
    use_market_regime_ppp = "marketppp" in rest
    use_market_regime_adx = "marketadx" in rest
    use_market_regime = "market" in rest or use_market_regime_ppp or use_market_regime_adx
    use_volume_filter = "volume" in rest
    use_rs_filter = "rs" in rest
    use_earnings_filter = "earnings" in rest
    use_sector_filter = "sector" in rest
    use_rsi_filter = "rsi" in rest
    use_dev_filter = "dev" in rest
    use_candle_filter = "candle" in rest
    use_fib_filter = "fib" in rest
    # 損切り幅は "sl10" / "sl15" のような引数で上書きできる（既定は10%）
    sl_args = [a for a in rest if a.startswith("sl") and a[2:].isdigit()]
    stop_loss_pct = float(sl_args[0][2:]) if sl_args else 10.0
    # タイムストップの日数は "ts30" / "ts90" のような引数で上書きできる（既定は60日）
    ts_args = [a for a in rest if a.startswith("ts") and a[2:].isdigit()]
    time_stop_days = int(ts_args[0][2:]) if ts_args else 60
    # エントリー方式は引数の組み合わせで決まる。指定できるシグナル名は
    # kahanshin / pullback / either（＝前2つ）/ ichimoku / macd / bandwalk。
    # 複数指定すると「どれか点灯で買い」になる（例: either macd）
    entry_parts = [
        a for a in rest
        if a in ("kahanshin", "pullback", "either", "ichimoku", "macd", "bandwalk")
    ]
    entry_mode = "+".join(entry_parts) if entry_parts else "kahanshin"
    # 往復の取引コスト（手数料＋スリッページ）。"cost20" で0.20%を意味する
    # （既定は0＝コスト無視。過去の検証結果と数値を比較できるようにするため）
    cost_args = [a for a in rest if a.startswith("cost") and a[4:].isdigit()]
    cost_pct = float(cost_args[0][4:]) / 100 if cost_args else 0.0
    # "short" を付けると空売り側（条件をすべて上下反転）を検証する
    side = "short" if "short" in rest else "long"
    period_args = [a for a in rest if a == "max" or (a.endswith("y") and a[:-1].isdigit())]
    history_period = period_args[0] if period_args else HISTORY_PERIOD

    filter_desc = "PPP3/4以上+100日線上のみ" if trend_filter else "フィルターなし"
    if use_market_regime_ppp:
        filter_desc += "+日経平均自体がPPP3/4以上の強いトレンドの日のみ"
    elif use_market_regime_adx:
        filter_desc += "+日経平均がADX20超の強いトレンドの日のみ"
    elif use_market_regime:
        filter_desc += "+日経平均が上昇トレンドの日のみ"
    if use_volume_filter:
        filter_desc += "+出来高が20日平均の1.5倍以上"
    if use_rs_filter:
        filter_desc += "+レラティブストレングスが50日平均より上（日経平均をアウトパフォーム中）"
    if use_earnings_filter:
        filter_desc += "+決算発表日の前後2営業日は見送り"
    if use_sector_filter:
        filter_desc += "+所属セクター全体が日経平均をアウトパフォーム中のみ"
    if use_rsi_filter:
        filter_desc += "+RSI(14日)が70未満（買われすぎ回避）"
    if use_dev_filter:
        filter_desc += "+株価が25日線から20%未満の上方乖離（追いかけ買い回避）"
    if use_candle_filter:
        filter_desc += "+直近10日以内にたくり線→抱き線の複合シグナルあり"
    if use_fib_filter:
        filter_desc += "+押しの深さがフィボナッチ25〜65%の範囲内"
    if cost_pct:
        filter_desc += f"／往復コスト{cost_pct:.2f}%を控除"
    exit_desc = {
        "ppp_break": "5日線が20日線を下抜け（PPP崩れ）",
        "atr_trail": "保有中の最高値から2.5×ATR下落（トレーリングストップ）",
        "ppp_or_atr": "PPP崩れ or トレーリングストップの早い方",
        "time_stop": "保有60日で強制手仕舞い（タイムストップ）",
        "ppp_or_time": "PPP崩れ or タイムストップの早い方",
        "dev_exit": "株価が25日線から20%以上上方乖離（グランビル法則④の早期利確）",
        "ppp_or_dev": "PPP崩れ or 乖離エグジットの早い方",
        "time_and_dev": "タイムストップ60日 or 乖離20%の早い方（PPP崩れは見ない）",
        "ppp_or_time_or_dev": "PPP崩れ or タイムストップ60日 or 乖離20%の最も早いもの",
        "profit_target": "含み益が10%に達したら利確（固定利確目標・損切りなし）",
        "ppp_or_target": "PPP崩れ or 固定利確目標10%の早い方",
        "time_or_sl": f"タイムストップ{time_stop_days}日 or 損切り-{stop_loss_pct:.0f}%の早い方",
        "time_dev_sl": f"タイムストップ{time_stop_days}日 or 乖離20%利確 or 損切り-{stop_loss_pct:.0f}%の最も早いもの",
    }.get(exit_mode, f"{exit_period}日線割れ")
    _names = {
        "kahanshin": "下半身", "pullback": "押し目買い", "either": "下半身/押し目買い",
        "ichimoku": "一目三役好転", "macd": "MACD GC", "bandwalk": "バンドウォーク",
    }
    entry_desc = ("【空売り】" if side == "short" else "") + " or ".join(_names.get(p, p) for p in entry_mode.split("+"))

    return {
        "exit_mode": exit_mode, "exit_period": exit_period, "trend_filter": trend_filter,
        "use_market_regime_ppp": use_market_regime_ppp, "use_market_regime_adx": use_market_regime_adx,
        "use_market_regime": use_market_regime, "use_volume_filter": use_volume_filter,
        "use_rs_filter": use_rs_filter, "use_earnings_filter": use_earnings_filter,
        "use_sector_filter": use_sector_filter, "use_rsi_filter": use_rsi_filter,
        "use_dev_filter": use_dev_filter, "use_candle_filter": use_candle_filter,
        "use_fib_filter": use_fib_filter, "stop_loss_pct": stop_loss_pct,
        "time_stop_days": time_stop_days, "entry_mode": entry_mode, "cost_pct": cost_pct,
        "side": side, "history_period": history_period,
        "filter_desc": filter_desc, "exit_desc": exit_desc, "entry_desc": entry_desc,
    }


def run_backtest(cfg: dict, hist_map: dict, name_map: dict, market_regime=None,
                 nikkei_close=None, sector_map=None, sector_regime_map=None,
                 verbose: bool = True):
    """
    取得済みの株価データに対して1条件ぶんのバックテストを回し、
    (トレードのDataFrame, バイ&ホールドの平均リターン) を返す。
    株価取得と切り離してあるので、compare.py は同じ hist_map を使い回して
    複数条件を1プロセスで検証できる。
    """
    sector_map = sector_map or {}
    sector_regime_map = sector_regime_map or {}
    all_trades, bh_returns = [], []

    for i, code in enumerate(hist_map, 1):
        name = name_map[code]
        hist = hist_map[code]
        try:
            earnings_dates = fetch_earnings_dates(code) if cfg["use_earnings_filter"] else None
            sector_regime = (sector_regime_map.get(sector_map.get(code))
                             if cfg["use_sector_filter"] else None)
            trades = backtest_ticker(
                code, name, hist, exit_period=cfg["exit_period"],
                trend_filter=cfg["trend_filter"], exit_mode=cfg["exit_mode"],
                market_regime=market_regime, volume_filter=cfg["use_volume_filter"],
                nikkei_close=nikkei_close if cfg["use_rs_filter"] else None,
                earnings_dates=earnings_dates, sector_regime=sector_regime,
                rsi_filter=cfg["use_rsi_filter"], dev_filter=cfg["use_dev_filter"],
                candle_filter=cfg["use_candle_filter"], fib_filter=cfg["use_fib_filter"],
                stop_loss_pct=cfg["stop_loss_pct"], time_stop_days=cfg["time_stop_days"],
                entry_mode=cfg["entry_mode"], cost_pct=cfg["cost_pct"], side=cfg["side"],
            )
            all_trades.extend(trades)
            bh_returns.append(buy_and_hold_return(hist))
            if verbose:
                print(f"  [{i}/{len(hist_map)}] {name} ({code}) トレード数: {len(trades)}")
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(hist_map)}] {name} ({code}) バックテスト失敗: {e}")

    df = pd.DataFrame(all_trades)
    avg_bh = sum(bh_returns) / len(bh_returns) if bh_returns else float("nan")
    return df, avg_bh


def load_price_data(tickers, history_period: str, use_sector_filter: bool = False):
    """全銘柄の株価（＋必要ならセクター）をキャッシュ経由でまとめて用意する。"""
    # 以前は1銘柄ずつ yf.Ticker().history() を呼んでおり、同じデータを
    # 毎回ダウンロードするため225銘柄で5〜15分かかっていた
    print(f"{len(tickers)}銘柄の株価データを用意中…")
    fetched = fetch_histories([t["code"] for t in tickers], period=history_period)
    hist_map, name_map, sector_map, short_data = {}, {}, {}, 0
    excluded = []
    for t in tickers:
        code, name = t["code"], t["name"]
        hist = fetched.get(code)
        if hist is None or len(hist) < 120:
            short_data += 1
            continue
        # 汚染データの除外。yfinanceの株式分割データ不整合により、実際には
        # あり得ない値動きが記録されている銘柄がある（東京海上HD/8766は
        # 初値が-0.18円、日本航空/9201は1日で191,400%）。以前は
        # portfolio_sim.py だけがこれを除外し、backtest.py は警告を出すだけ
        # だったため、26年バックテストの平均リターンとPFが過大に出ていた
        daily = hist["Close"].pct_change().abs()
        if (daily > MAX_PLAUSIBLE_DAILY_MOVE).any():
            excluded.append(f"{name}({code}) {daily.idxmax().date()} に{daily.max()*100:,.0f}%変動")
            continue
        hist_map[code] = hist
        name_map[code] = name
        if use_sector_filter:
            sector_map[code] = fetch_sector(code)
    if short_data:
        print(f"  データ不足で{short_data}銘柄をスキップしました")
    if excluded:
        print(f"  ⚠️ 分割データ不整合の疑いで{len(excluded)}銘柄を除外しました：")
        for e in excluded[:10]:
            print(f"     {e}")
        if len(excluded) > 10:
            print(f"     …ほか{len(excluded) - 10}銘柄")
    return hist_map, name_map, sector_map


def main():
    cfg = parse_config(sys.argv[1:])
    exit_mode = cfg["exit_mode"]
    exit_period = cfg["exit_period"]
    trend_filter = cfg["trend_filter"]
    use_market_regime_ppp = cfg["use_market_regime_ppp"]
    use_market_regime_adx = cfg["use_market_regime_adx"]
    use_market_regime = cfg["use_market_regime"]
    use_volume_filter = cfg["use_volume_filter"]
    use_rs_filter = cfg["use_rs_filter"]
    use_earnings_filter = cfg["use_earnings_filter"]
    use_sector_filter = cfg["use_sector_filter"]
    use_rsi_filter = cfg["use_rsi_filter"]
    use_dev_filter = cfg["use_dev_filter"]
    use_candle_filter = cfg["use_candle_filter"]
    use_fib_filter = cfg["use_fib_filter"]
    stop_loss_pct = cfg["stop_loss_pct"]
    time_stop_days = cfg["time_stop_days"]
    entry_mode = cfg["entry_mode"]
    cost_pct = cfg["cost_pct"]
    side = cfg["side"]
    history_period = cfg["history_period"]
    exit_desc, entry_desc, filter_desc = cfg["exit_desc"], cfg["entry_desc"], cfg["filter_desc"]

    tickers = load_tickers()
    print(f"{len(tickers)}銘柄で{entry_desc}バックテストを実行します（過去{history_period}、エグジット={exit_desc}、エントリー条件={filter_desc}）…")

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

    nikkei_close = None
    if use_rs_filter or use_sector_filter:
        print("レラティブストレングス計算用に日経平均のデータを取得中…")
        nikkei_close = fetch_nikkei_close(history_period)

    # セクターフィルターは全銘柄の株価をまず集めてセクター指数を合成する
    # 必要があるため、先に全銘柄のヒストリー（＋セクター）を取得しておく
    hist_map, name_map, sector_map = load_price_data(tickers, history_period, use_sector_filter)

    sector_regime_map = {}
    if use_sector_filter:
        print("セクター指数を合成中…")
        sector_close_map = build_sector_close(hist_map, sector_map)
        sector_regime_map = build_sector_regime(sector_close_map, nikkei_close)

    df, avg_bh = run_backtest(
        cfg, hist_map, name_map, market_regime=market_regime, nikkei_close=nikkei_close,
        sector_map=sector_map, sector_regime_map=sector_regime_map,
    )

    if df.empty:
        print("トレードが1件も発生しませんでした。")
        return

    suspicious = df[df["return_pct"].abs() > SUSPICIOUS_RETURN_THRESHOLD]
    if not suspicious.empty:
        print(f"\n⚠️  リターンが{SUSPICIOUS_RETURN_THRESHOLD:.0f}%を超えるトレードが{len(suspicious)}件あります。"
              "yfinanceの株式分割データ不整合等による異常値の可能性があるため、手動で確認してください：")
        for _, row in suspicious.iterrows():
            print(f"   {row['code']} {row['name']} {row['entry_date']}→{row['exit_date']} "
                  f"return={row['return_pct']:.1f}% (entry={row['entry_price']:.2f}, exit={row['exit_price']:.2f})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    if use_market_regime_ppp:
        market_suffix = "_marketppp"
    elif use_market_regime_adx:
        market_suffix = "_marketadx"
    elif use_market_regime:
        market_suffix = "_market"
    else:
        market_suffix = ""
    # side を必ずファイル名に含める（含めないと空売りの結果が買いの結果を
    # 上書きしてしまう。2026-08-29に実際に上書き事故を起こしたため明示）
    suffix = ("_short" if side == "short" else "") + (f"_{entry_mode}" if entry_mode != "kahanshin" else "") + (f"_cost{cost_pct*100:.0f}" if cost_pct else "") + ("_trend" if trend_filter else "") + market_suffix + ("_volume" if use_volume_filter else "") + ("_rs" if use_rs_filter else "") + ("_earnings" if use_earnings_filter else "") + ("_sector" if use_sector_filter else "") + ("_rsi" if use_rsi_filter else "") + ("_dev" if use_dev_filter else "") + ("_candle" if use_candle_filter else "") + ("_fib" if use_fib_filter else "")
    tag = {
        "ppp_break": "ppp", "atr_trail": "atrtrail", "ppp_or_atr": "pporatr",
        "time_stop": "timestop", "ppp_or_time": "pportime",
        "dev_exit": "devexit", "ppp_or_dev": "ppordev",
        "time_and_dev": "timeanddev", "ppp_or_time_or_dev": "pportimeordev",
        "profit_target": "target", "ppp_or_target": "pportarget",
        "time_or_sl": f"timesl{stop_loss_pct:.0f}d{time_stop_days}",
        "time_dev_sl": f"timedevsl{stop_loss_pct:.0f}d{time_stop_days}",
    }.get(exit_mode, f"exit{exit_period}")
    out_path = OUTPUT_DIR / f"backtest_trades_{tag}{suffix}_{history_period}_{dt.date.today():%Y%m%d}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    win_rate = (df["return_pct"] > 0).mean() * 100
    avg_return = df["return_pct"].mean()
    median_return = df["return_pct"].median()
    avg_holding = df["holding_days"].mean()

    print(f"\n完了: {out_path}")
    print(f"\n=== {entry_desc}シグナル バックテスト結果（エグジット={exit_desc}） ===")
    print(f"総トレード数: {len(df)}件（対象{len(hist_map)}銘柄）")
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
