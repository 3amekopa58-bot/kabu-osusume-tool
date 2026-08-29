"""
おすすめ株スクリーニングツール（MVP）

tickers.csv に書かれた銘柄について、
  - 割安度（PER・PBR・配当利回り）
  - トレンド（「相場の赤本」の PPP／下半身／9の法則 に基づく分析）
をそれぞれスコア化し、合算した「おすすめ度」でランキングしたCSVを出力する。

トレンド分析のロジックは 相場の赤本_ルール.md を参照。

使い方:
    python screen.py
出力:
    output/recommend_YYYYMMDD.csv
"""

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
OUTPUT_DIR = BASE_DIR / "output"
EDINET_CACHE_PATH = BASE_DIR / "data" / "edinet_valuation_diff.json"

FUNDAMENTAL_WEIGHT = 0.5
TECHNICAL_WEIGHT = 0.5

BUDGET = 1_000_000  # 予算（円）。日本株は基本100株単位（単元株）での購入となる
LOT_SIZE = 100

MA_PERIODS = (5, 10, 20, 50, 100)
MIN_HISTORY_DAYS = 105  # 100日線 + スイングカウント用のバッファ

# 早期利確ライン（グランビルの法則④・株価チャート大全 由来）。
# 25日線からの上方かい離率がこの閾値以上で利確を検討する目安を表示する。
# バックテストで確認済み：5年・10年・26年の全期間で勝率が一貫して改善
# （52.5%→55.4%、43.7%→45.7%、42.8%→45.2%）した一方、PFは一貫して悪化
# （3.13→2.55、2.25→1.87、2.05→1.80）。大きく伸びるトレードの利益を
# 早期に刈り取る分、勝つ頻度は上がるが1件あたりの利益幅は縮小する
# トレードオフ。2026-08-29、勝率を優先する方針で採用（screen.pyでは
# 目安表示のみ、実際の売却判断は利用者が行う）
DEV_MA_PERIOD = 25
EXIT_DEV_THRESHOLD = 20.0


def load_tickers():
    with open(TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_edinet_cache() -> dict:
    """
    かぶ1000氏の考え方に基づく参考指標（その他有価証券評価差額金の増減）。
    scripts/build_edinet_cache.py で事前生成したJSONを読むだけで、
    毎回EDINET APIを叩かない（年1回しか更新されないデータのため）。
    有価証券報告書の「個別（親会社単体）」財務諸表から取得（連結決算が
    IFRSの企業でも、個別は日本基準で開示するのが通例のため収録対象になる）。
    それでも個別財務諸表でこの項目自体を開示していない企業は対象外。
    """
    if not EDINET_CACHE_PATH.exists():
        return {}
    with open(EDINET_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f).get("data", {})


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


def fetch_nikkei_close(period: str = "3y") -> pd.Series:
    """レラティブストレングス計算用に日経平均の終値だけを取得する"""
    return yf.Ticker("^N225").history(period=period)["Close"]


def fetch_market_regime(adx_threshold: float = 20.0) -> bool:
    """
    マーケットレジームフィルター：日経平均自体が上昇トレンド（終値が100日線より上）
    かつ、ADXがadx_threshold超（トレンドに十分な勢いがある＝レンジ相場でない）か。
    バックテストで確認済み：5年・10年・26年のすべての期間で、単純な「100日線より
    上」だけの判定より、ADXでトレンド強度も見る方が勝率・平均リターン・
    プロフィットファクターが一貫して改善した（26年PF 1.67→2.02）。
    """
    idx = yf.Ticker("^N225").history(period="1y")
    close = idx["Close"]
    sma100 = close.rolling(100).mean()
    if pd.isna(sma100.iloc[-1]):
        return True  # データ不足時は制限しない
    is_uptrend = close.iloc[-1] > sma100.iloc[-1]
    adx = calc_adx(idx["High"], idx["Low"], close)
    is_strong_trend = pd.notna(adx.iloc[-1]) and adx.iloc[-1] > adx_threshold
    return bool(is_uptrend and is_strong_trend)


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def swing_leg_count(close: pd.Series, sma5: pd.Series) -> tuple[str, int]:
    """
    「9の法則」の起点定義：上昇開始直前の最安値（下落開始直前の最高値）の
    ローソク足を起点に、終値が5日線を越える/割り込むまでの本数を数える。
    途中で陰線・陽線が混じってもカウントはリセットしない（終値が5日線の
    反対側に確定して初めてレグが切り替わる）。
    戻り値: (現在のレグの向き 'up'/'down', そのレグの継続本数)
    """
    valid = sma5.first_valid_index()
    if valid is None:
        return None, None
    start_i = close.index.get_loc(valid)
    if start_i >= len(close) - 1:
        return None, None

    phase = "up" if close.iloc[start_i] >= sma5.iloc[start_i] else "down"
    leg_start = start_i

    for i in range(start_i + 1, len(close)):
        c, s = close.iloc[i], sma5.iloc[i]
        if phase == "up":
            if c < s:
                peak_offset = close.iloc[leg_start:i].values.argmax()
                leg_start = leg_start + peak_offset
                phase = "down"
        else:
            if c > s:
                trough_offset = close.iloc[leg_start:i].values.argmin()
                leg_start = leg_start + trough_offset
                phase = "up"

    count = (len(close) - 1) - leg_start + 1
    return phase, count


def ma_pair_direction(close: pd.Series, short_n: int = 5, long_n: int = 20, lookback: int = 3):
    """短期線・長期線が両方とも同じ方向（上/下）に傾いているか。'up'/'down'/None。"""
    if len(close) < long_n + lookback:
        return None
    short_sma = close.rolling(short_n).mean()
    long_sma = close.rolling(long_n).mean()
    if pd.isna(short_sma.iloc[-1]) or pd.isna(long_sma.iloc[-1]):
        return None
    short_up = short_sma.iloc[-1] > short_sma.iloc[-1 - lookback]
    long_up = long_sma.iloc[-1] > long_sma.iloc[-1 - lookback]
    short_down = short_sma.iloc[-1] < short_sma.iloc[-1 - lookback]
    long_down = long_sma.iloc[-1] < long_sma.iloc[-1 - lookback]
    if short_up and long_up:
        return "up"
    if short_down and long_down:
        return "down"
    return None


def find_recent_cross(close: pd.Series, short_n: int = 5, long_n: int = 20, max_age: int = 6):
    """直近max_age本以内の5日線と20日線のクロスを探す。戻り値: (何本前か, 'up'/'down')。"""
    short_sma = close.rolling(short_n).mean()
    long_sma = close.rolling(long_n).mean()
    diff = (short_sma - long_sma).dropna()
    n = len(diff)
    if n < 2:
        return None, None
    for age in range(0, min(max_age, n - 1)):
        i = n - 1 - age
        if diff.iloc[i] > 0 and diff.iloc[i - 1] <= 0:
            return age, "up"
        if diff.iloc[i] < 0 and diff.iloc[i - 1] >= 0:
            return age, "down"
    return None, None


def detect_kuchibashi(close: pd.Series) -> dict:
    """
    「くちばし」／「逆くちばし」（相場の赤本 第7章）。
    5日線と20日線が同じ方向に傾きながらクロスし、週足・月足も同方向で、
    前の高値/安値付近でなく、急騰・急落中でもない場合に成立する。
    """
    age, direction = find_recent_cross(close, 5, 20, max_age=6)
    if direction is None:
        return {"signal": None, "label": "くちばし: シグナルなし"}

    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    cross_i = len(close) - 1 - age
    lookback = 3
    if cross_i - lookback < 0:
        return {"signal": None, "label": "くちばし: シグナルなし"}

    # 条件①：クロス時点で5日線・20日線が同じ方向
    if direction == "up":
        same_direction = (
            sma5.iloc[cross_i] > sma5.iloc[cross_i - lookback]
            and sma20.iloc[cross_i] > sma20.iloc[cross_i - lookback]
        )
    else:
        same_direction = (
            sma5.iloc[cross_i] < sma5.iloc[cross_i - lookback]
            and sma20.iloc[cross_i] < sma20.iloc[cross_i - lookback]
        )
    if not same_direction:
        return {"signal": None, "label": "くちばし: 単なるクロス（方向不一致）"}

    # 条件④：前の高値・安値付近（直近60営業日の高安の3%以内）を避ける
    window = close.iloc[max(0, cross_i - 60):cross_i + 1]
    recent_high, recent_low = window.max(), window.min()
    price_at_cross = close.iloc[cross_i]
    near_extreme = (
        abs(price_at_cross - recent_high) / recent_high < 0.03
        or abs(price_at_cross - recent_low) / recent_low < 0.03
    )
    if near_extreme:
        return {"signal": None, "label": "くちばし: 前の高値/安値付近のため見送り"}

    # 条件⑤：急騰・急落中（クロス前後5営業日に前日比7%超の変化）を避ける
    recent_window = close.iloc[max(0, cross_i - 5):cross_i + 1]
    if (recent_window.pct_change().abs() > 0.07).any():
        return {"signal": None, "label": "くちばし: 急騰・急落中のため見送り"}

    # 条件②：週足・月足も同方向
    weekly_close = close.resample("W-FRI").last().dropna()
    monthly_close = close.resample("ME").last().dropna()
    weekly_dir = ma_pair_direction(weekly_close)
    monthly_dir = ma_pair_direction(monthly_close)

    if weekly_dir != direction or monthly_dir != direction:
        return {"signal": None, "label": "くちばし: 日足のみ成立（週足/月足は不一致）"}

    label = "くちばし成立（日足・週足・月足すべて上向き一致）" if direction == "up" \
        else "逆くちばし成立（日足・週足・月足すべて下向き一致）"
    return {"signal": direction, "fresh_days_ago": age, "label": label}


def detect_monowakare(close: pd.Series, sma: dict, lookback: int = 20, min_recent: int = 8) -> dict:
    """
    「ものわかれ」（＝相場師朗氏がYouTubeで解説する「黒い縁取り」からの抜け）。
    5日線と10日/20日/50日線のいずれかとの乖離幅（価格比）が、直近lookback営業日
    以内に最小（＝収束）をつけたあと、直近min_recent日以内に再拡大に転じて
    いるかを判定する。下半身の信頼度を上げる補助シグナルとして使う。
    """
    price = close.iloc[-1]
    sma5 = sma[5]

    for period in (10, 20, 50):
        other = sma[period]
        spread = (sma5 - other).abs() / price
        recent_window = spread.iloc[-(lookback + 1):]
        if recent_window.isna().any():
            continue

        min_pos = recent_window.values.argmin()
        days_since_min = len(recent_window) - 1 - min_pos
        if days_since_min > min_recent:
            continue

        min_spread_value = recent_window.iloc[min_pos]
        current_spread = spread.iloc[-1]
        if min_spread_value <= 0 or current_spread <= min_spread_value * 1.15:
            continue

        direction = "up" if sma5.iloc[-1] > other.iloc[-1] else "down"
        arrow = "上抜け" if direction == "up" else "下抜け"
        return {
            "signal": direction,
            "period": period,
            "label": f"ものわかれ: 5日線と{period}日線の収束（黒い縁取り）から{arrow}",
        }

    return {"signal": None, "period": None, "label": "ものわかれ: シグナルなし"}


def nearest_round_levels(price: float) -> tuple[float, float]:
    """価格帯に応じた「キリのいい株価」の刻み幅で、直近の節目（下・上）を返す。"""
    if price < 500:
        step = 50
    elif price < 2000:
        step = 100
    elif price < 5000:
        step = 250
    elif price < 20000:
        step = 500
    else:
        step = 1000
    lower = (price // step) * step
    upper = lower + step
    return lower, upper


def detect_fushime(close: pd.Series, lookback: int = 60, recent_days: int = 5) -> dict:
    """
    「節目」（相場の赤本 第2章）。キリのいい株価と直近lookback営業日の高値・安値を
    節目候補とし、直近recent_days以内にそれを上抜けたか（勢いづきやすい局面）、
    今まさに節目付近（2%以内）にいるか（足踏みしやすい局面）を判定する。
    """
    price = close.iloc[-1]
    round_lower, round_upper = nearest_round_levels(price)

    window = close.iloc[-(lookback + 1):-1]
    prior_high = window.max()
    prior_low = window.min()

    levels_above = sorted(set(lvl for lvl in [round_upper, prior_high] if lvl > 0))
    recent_window = close.iloc[-(recent_days + 1):]

    breakout_level = None
    for lvl in levels_above:
        if recent_window.iloc[0] <= lvl < recent_window.iloc[-1]:
            breakout_level = lvl
            break

    near_level = None
    for lvl in [round_lower, round_upper, prior_high, prior_low]:
        if lvl > 0 and abs(price - lvl) / lvl < 0.02:
            near_level = lvl
            break

    if breakout_level is not None:
        label = f"節目（{breakout_level:,.0f}円）を突破直後 → 勢いづきやすい局面"
    elif near_level is not None:
        label = f"節目（{near_level:,.0f}円）に接近中 → 一旦足踏みしやすい局面"
    else:
        label = "節目: 特になし"

    return {"breakout_level": breakout_level, "near_level": near_level, "label": label}


def fetch_one(code: str, name: str, edinet_cache: Optional[dict] = None, nikkei_close: Optional[pd.Series] = None) -> dict:
    ticker = yf.Ticker(code)
    info = ticker.info
    hist = ticker.history(period="3y")

    edinet_info = (edinet_cache or {}).get(code, {})

    row = {
        "code": code,
        "name": name,
        "price": info.get("currentPrice"),
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        # ベンジャミン・グレアムの財務健全性基準（流動比率は高いほど、
        # 負債比率は低いほど良い）。yfinanceのinfoからそのまま取得。
        "current_ratio": info.get("currentRatio"),
        "debt_to_equity": info.get("debtToEquity"),
        # かぶ1000氏の参考指標：保有有価証券の含み益の増減（個別財務諸表ベース）
        # （収録企業は225銘柄中171銘柄・スコアには未使用、参考表示のみ）
        "securities_valuation_diff_change_yen": edinet_info.get("valuation_diff_change_yen"),
    }

    if len(hist) >= MIN_HISTORY_DAYS:
        close = hist["Close"]
        open_ = hist["Open"]
        volume = hist["Volume"]
        sma = {n: close.rolling(n).mean() for n in MA_PERIODS}

        row["last_close"] = close.iloc[-1]
        row["last_open"] = open_.iloc[-1]
        row["rsi14"] = calc_rsi(close)
        for n in MA_PERIODS:
            row[f"sma{n}"] = sma[n].iloc[-1]

        # 25日線からの上方かい離率（早期利確ラインの判定用）
        sma25 = close.rolling(DEV_MA_PERIOD).mean().iloc[-1]
        row["dev_from_sma25_pct"] = (
            (row["last_close"] - sma25) / sma25 * 100 if pd.notna(sma25) and sma25 != 0 else None
        )

        # 出来高フィルター：本日の出来高が直近20日平均の1.5倍以上か
        # バックテストで確認済み：出来高を伴った下半身はPF 1.80→2.60に大幅改善
        vol_avg20 = volume.rolling(20).mean().iloc[-1]
        row["volume_ratio"] = (volume.iloc[-1] / vol_avg20) if vol_avg20 and vol_avg20 > 0 else None
        row["volume_confirmed"] = bool(row["volume_ratio"] is not None and row["volume_ratio"] >= 1.5)

        # PPP: 5日線>10日線>20日線>50日線>100日線 の並びがどこまで揃っているか（0〜4）
        up_matches = sum(
            sma[MA_PERIODS[i]].iloc[-1] > sma[MA_PERIODS[i + 1]].iloc[-1]
            for i in range(len(MA_PERIODS) - 1)
        )
        row["ppp_matches"] = up_matches
        row["ppp_up"] = up_matches == 4
        row["ppp_down"] = up_matches == 0

        # バックテストで確認済みの「強いトレンド」条件（下半身の信頼度フィルター）
        row["trend_filter_pass"] = bool(up_matches >= 3 and row["last_close"] > sma[100].iloc[-1]) \
            if not pd.isna(sma[100].iloc[-1]) else False

        # 5日線自体の向き（直近3営業日での傾き）
        sma5_series = sma[5]
        sma5_prev = sma5_series.iloc[-4]
        row["sma5_slope_up"] = sma5_series.iloc[-1] > sma5_prev
        row["sma5_slope_down"] = sma5_series.iloc[-1] < sma5_prev

        # 下半身／逆下半身：直近1本が5日線をまたいだか＋ローソク足の色＋5日線自体の向き
        prev_close, prev_sma5 = close.iloc[-2], sma5_series.iloc[-2]
        crossed_above = prev_close <= prev_sma5 and row["last_close"] > row["sma5"]
        crossed_below = prev_close >= prev_sma5 and row["last_close"] < row["sma5"]
        is_bullish = row["last_close"] > row["last_open"]
        is_bearish = row["last_close"] < row["last_open"]

        row["kahanshin"] = bool(crossed_above and is_bullish and row["sma5_slope_up"])
        row["gyaku_kahanshin"] = bool(crossed_below and is_bearish and row["sma5_slope_down"])

        # 9の法則（スイング起点版）
        phase, count = swing_leg_count(close, sma5_series)
        row["td_buy"] = count if phase == "down" else 0
        row["td_sell"] = count if phase == "up" else 0

        # くちばし／逆くちばし
        kb = detect_kuchibashi(close)
        row["kuchibashi_signal"] = kb["signal"]
        row["kuchibashi_label"] = kb["label"]

        # ものわかれ（黒い縁取りからの抜け）
        mw = detect_monowakare(close, sma)
        row["monowakare_signal"] = mw["signal"]
        row["monowakare_label"] = mw["label"]

        # 節目（キリのいい株価・前の高値安値）
        fs = detect_fushime(close)
        row["fushime_breakout_level"] = fs["breakout_level"]
        row["fushime_label"] = fs["label"]

        # レラティブストレングス：個別銘柄÷日経平均の比率が自身の50日移動
        # 平均より上＝直近で日経平均をアウトパフォーム中。バックテストで
        # 確認済み：5年・10年・26年の全期間で勝率・PFが一貫して改善
        # （26年PF 2.02→2.06、勝率42.2%→42.9%）。2026-08-21採用
        if nikkei_close is not None and len(nikkei_close) > 0:
            nikkei_aligned = nikkei_close.reindex(close.index, method="ffill")
            rs_ratio = close / nikkei_aligned
            rs_sma = rs_ratio.rolling(50).mean()
            row["relative_strength_confirmed"] = bool(
                pd.notna(rs_sma.iloc[-1]) and rs_ratio.iloc[-1] > rs_sma.iloc[-1]
            )
        else:
            row["relative_strength_confirmed"] = False
    else:
        row["last_close"] = row["last_open"] = row["rsi14"] = None
        for n in MA_PERIODS:
            row[f"sma{n}"] = None
        row["dev_from_sma25_pct"] = None
        row["ppp_matches"] = row["ppp_up"] = row["ppp_down"] = None
        row["trend_filter_pass"] = None
        row["sma5_slope_up"] = row["sma5_slope_down"] = None
        row["kahanshin"] = row["gyaku_kahanshin"] = None
        row["td_buy"] = row["td_sell"] = None
        row["kuchibashi_signal"] = None
        row["kuchibashi_label"] = "データ不足"
        row["monowakare_signal"] = None
        row["monowakare_label"] = "データ不足"
        row["fushime_breakout_level"] = None
        row["fushime_label"] = "データ不足"
        row["volume_ratio"] = None
        row["volume_confirmed"] = None
        row["relative_strength_confirmed"] = None

    return row


def td_label(row) -> str:
    buy, sell = row["td_buy"], row["td_sell"]
    if pd.isna(buy):
        return "データ不足"
    for n in (23, 17, 9):
        if buy == n:
            return f"9の法則: 下落{n}本目（利確/反発を強く意識する節目）"
        if sell == n:
            return f"9の法則: 上昇{n}本目（利確/反落を強く意識する節目）"
    if buy >= 6:
        return f"9の法則: 下落{buy}本目（底打ち接近）"
    if sell >= 6:
        return f"9の法則: 上昇{sell}本目（天井接近）"
    return "9の法則: シグナルなし"


def trend_label(row) -> str:
    """PPP／下半身に基づくトレンドの様相ラベル。"""
    if pd.isna(row["ppp_matches"]):
        return "データ不足"

    if row["ppp_up"]:
        base = "PPP（完璧な上昇トレンド）"
    elif row["ppp_down"]:
        base = "逆PPP（完璧な下降トレンド）"
    elif row["ppp_matches"] >= 2:
        base = f"上昇トレンド気味（PPP {row['ppp_matches']}/4）"
    else:
        base = f"下降トレンド気味（PPP {row['ppp_matches']}/4）"

    if row["kahanshin"]:
        base += "・本日「下半身」シグナル点灯"
    if row["gyaku_kahanshin"]:
        base += "・本日「逆下半身」シグナル点灯"
    if row.get("monowakare_signal") == "up":
        base += "・ものわかれ（黒い縁取り）からの上抜けあり"
    if row.get("volume_confirmed"):
        base += "・出来高急増を伴う"

    return base


def buy_timing(row, market_regime_up: bool = True) -> str:
    """本日が買いタイミングかどうか、いくらで買うことになるかを示す。"""
    if pd.isna(row["price"]):
        return "データ不足"

    signals = []
    if row.get("kahanshin"):
        signals.append("下半身")
    if row.get("kuchibashi_signal") == "up":
        signals.append("くちばし")

    if not signals:
        return "本日は買いシグナルなし（様子見）"

    notes = []
    if not row.get("trend_filter_pass"):
        notes.append("トレンドやや弱め")
    if not market_regime_up:
        notes.append("日経平均の地合いが弱い（トレンド不足）")
    if row.get("kahanshin") and not row.get("volume_confirmed"):
        notes.append("出来高の伴いが弱い")
    confidence = f"（{'・'.join(notes)}・慎重に）" if notes else ""
    return (
        f"買いタイミング（{'・'.join(signals)}点灯）{confidence}: "
        f"{row['price']:,.0f}円 × 100株 = {row['lot_cost']:,.0f}円"
    )


def sell_timing(row) -> str:
    """
    利益確定・手仕舞いの目安を単一の目安価格として示す。
    本来のルールは「5日線が20日線を下抜けたら」だが、これは実質的に
    「株価が20日線あたりまで下がってきたら」とほぼ同義（価格が下がると
    反応の速い5日線が先に下がり、20日線を下抜ける）。バックテストで
    比較検証した「5日線が20日線を下抜け」を、日々の目安価格として
    20日線の現在値そのもので近似して提示する。
    空売りの新規シグナルではなく、現物買いした株を手仕舞う（売却する）
    タイミングの目安。購入価格より上で手仕舞えれば利益確定、下なら
    損切りになる（結果は状況次第）。
    """
    if pd.isna(row["sma20"]):
        return "データ不足"

    base = f"売り目安: {row['sma20']:,.0f}円（終値がこの価格を下回ったら手仕舞い＝保有株の売却を検討）"

    buy = row.get("td_buy")
    if buy in (9, 17, 23):
        base += f" ／ 9の法則が{buy}本目＝手仕舞いを強く意識する節目に到達"
    elif pd.notna(buy) and buy >= 6:
        next_checkpoint = 9 if buy < 9 else (17 if buy < 17 else 23)
        base += f" ／ 9の法則は現在{buy}本目（次の節目は{next_checkpoint}本目）"

    dev = row.get("dev_from_sma25_pct")
    if pd.notna(dev) and dev >= EXIT_DEV_THRESHOLD:
        base += f" ／ 25日線から{dev:.0f}%上方乖離＝早期利確ライン到達（勝率は上がるがPFは下がるトレードオフあり、利確を検討）"

    return base


def technical_score(row, market_regime_up: bool = True) -> float:
    """
    0〜1超のトレンド健全度スコア。
    PPPの完成度・下半身シグナル・RSI・9の法則（下落レグの継続本数）を加点する。
    market_regime_up: 日経平均自体が上昇トレンドかどうか（マーケットレジームフィルター）。
    """
    if pd.isna(row["ppp_matches"]) or pd.isna(row["rsi14"]) or pd.isna(row["td_buy"]):
        return 0.0

    score = 0.0

    # PPPの完成度（5>10>20>50>100 が何組成立しているか）
    score += (row["ppp_matches"] / 4) * 0.25

    # 下半身シグナルが本日点灯＝号砲。バックテストで確認済みの各条件を
    # 加算方式で評価する（PF実績: 基本1.29 → トレンド強め1.44 → +地合い1.80 →
    # +出来高2.60）。条件が重なるほど過去の実績上、信頼度が高い。
    if row["kahanshin"]:
        bonus = 0.10  # 下半身が出ただけでも号砲として加点
        if row.get("trend_filter_pass"):
            bonus += 0.08  # PPP3/4以上＋100日線より上
        if market_regime_up:
            bonus += 0.05  # 日経平均自体が上昇トレンド
        if row.get("volume_confirmed"):
            bonus += 0.10  # 出来高が20日平均の1.5倍以上（バックテストで最も効果が大きかった条件）
        if row.get("monowakare_signal") == "up":
            bonus += 0.05  # ものわかれ（黒い縁取り）からの上抜けと重なる
        if row.get("relative_strength_confirmed"):
            bonus += 0.05  # 日経平均をアウトパフォーム中（レラティブストレングス）
        score += bonus

    # RSIが40〜70の健全な上昇域
    if 40 <= row["rsi14"] <= 70:
        score += 0.20

    # 9の法則：下落レグが伸びるほど底打ち・反発の確度が上がる（9→17→23の節目）
    buy = row["td_buy"]
    if buy == 23:
        score += 0.35
    elif buy == 17:
        score += 0.25
    elif buy == 9:
        score += 0.20
    elif buy in (7, 8):
        score += 0.10

    # くちばし成立（日足・週足・月足すべて一致した強い上昇転換シグナル）
    if row.get("kuchibashi_signal") == "up":
        score += 0.30

    # 節目（キリのいい株価・前の高値）を直近で突破＝勢いづきやすい局面
    # （DataFrame化でNoneがNaNになるため is not None ではなく pd.notna で判定する）
    if pd.notna(row.get("fushime_breakout_level")):
        score += 0.15

    return score


def rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    パーセンタイル順位を0〜1のスコアに変換する。
    データ欠損（NaN）は「最良」でも「最悪」でもなく中立（0.5点）として扱う。
    （pandasのrank(na_option="bottom")はNaNを最大値として扱うため、そのまま
    使うと「データが無い銘柄が満点になる」という誤った結果になることがあった。
    2026-08-18に発見・修正。）
    """
    pct = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        pct = 1 - pct
    return pct.fillna(0.5)


def build_ranking(rows: list[dict], budget: int = BUDGET, market_regime_up: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    df["lot_cost"] = df["price"] * LOT_SIZE
    df["affordable"] = df["lot_cost"] <= budget

    # 割安度スコアは「予算内で買える銘柄同士」で比較する
    # （買えない大型株が混ざるとランキングの相対評価が歪むため）
    pool = df[df["affordable"]].copy()
    excluded = df[~df["affordable"]].copy()

    pool["per_score"] = rank_score(pool["per"], higher_is_better=False)
    pool["pbr_score"] = rank_score(pool["pbr"], higher_is_better=False)
    pool["dividend_score"] = rank_score(pool["dividend_yield"], higher_is_better=True)

    # ベンジャミン・グレアムの基準を追加
    # ①グレアム・ナンバー的な考え方：PER×PBRが低いほど良い（22.5以下が目安）
    pool["graham_number"] = pool["per"] * pool["pbr"]
    pool["graham_score"] = rank_score(pool["graham_number"], higher_is_better=False)
    # ②財務健全性：流動比率は高いほど良い、負債比率は低いほど良い
    pool["current_ratio_score"] = rank_score(pool["current_ratio"], higher_is_better=True)
    pool["debt_score"] = rank_score(pool["debt_to_equity"], higher_is_better=False)

    pool["fundamental_score"] = pool[[
        "per_score", "pbr_score", "dividend_score",
        "graham_score", "current_ratio_score", "debt_score",
    ]].mean(axis=1)

    pool["technical_score"] = pool.apply(lambda r: technical_score(r, market_regime_up), axis=1)
    pool["trend_label"] = pool.apply(trend_label, axis=1)
    pool["td_label"] = pool.apply(td_label, axis=1)
    pool["buy_timing"] = pool.apply(lambda r: buy_timing(r, market_regime_up), axis=1)
    pool["sell_timing"] = pool.apply(sell_timing, axis=1)

    pool["total_score"] = (
        pool["fundamental_score"] * FUNDAMENTAL_WEIGHT
        + pool["technical_score"] * TECHNICAL_WEIGHT
    )

    pool = pool.sort_values("total_score", ascending=False)
    return pool, excluded


def main():
    print("日経平均のトレンド（マーケットレジーム）を確認します…")
    try:
        market_regime_up = fetch_market_regime()
    except Exception as e:
        print(f"  取得失敗（フィルターなしで続行）: {e}")
        market_regime_up = True
    print(f"  日経平均は{'上昇トレンド・ADXも強い' if market_regime_up else '上昇トレンドでない、またはADXが弱い（レンジ相場）'}と判定")

    tickers = load_tickers()
    edinet_cache = load_edinet_cache()
    try:
        nikkei_close = fetch_nikkei_close()
    except Exception as e:
        print(f"日経平均データ取得失敗（レラティブストレングスなしで続行）: {e}")
        nikkei_close = None
    print(f"{len(tickers)}銘柄のデータを取得します…")

    rows = []
    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            rows.append(fetch_one(code, name, edinet_cache, nikkei_close))
            print(f"  [{i}/{len(tickers)}] {name} ({code}) 取得OK")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {name} ({code}) 取得失敗: {e}")

    df, excluded = build_ranking(rows, market_regime_up=market_regime_up)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"recommend_{dt.date.today():%Y%m%d}.csv"

    cols = [
        "code", "name", "price", "lot_cost", "per", "pbr", "dividend_yield",
        "graham_number", "current_ratio", "debt_to_equity", "securities_valuation_diff_change_yen",
        "sma5", "sma10", "sma20", "sma50", "sma100", "ppp_matches", "trend_filter_pass",
        "volume_ratio", "volume_confirmed", "relative_strength_confirmed", "dev_from_sma25_pct",
        "rsi14", "trend_label", "td_buy", "td_sell", "td_label", "kuchibashi_label",
        "monowakare_label", "fushime_label",
        "buy_timing", "sell_timing",
        "fundamental_score", "technical_score", "total_score",
    ]
    df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n予算 {BUDGET:,}円（{LOT_SIZE}株単位）で買える銘柄: {len(df)}/{len(rows)}")
    if not excluded.empty:
        over = ", ".join(f"{r['name']}({r['lot_cost']:,.0f}円)" for _, r in excluded.iterrows())
        print(f"予算オーバーで除外: {over}")

    print(f"\n完了: {out_path}")
    print("\n--- 上位10銘柄 ---")
    print(df[["name", "code", "buy_timing", "sell_timing", "total_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
