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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from backtest import calc_cup_with_handle
from price_cache import fetch_histories, fetch_history

BASE_DIR = Path(__file__).parent
# 対象銘柄は日経225ではなく、全上場から「予算内で買えて流動性がある」で
# 絞った944銘柄（scripts/build_universe.py が生成）。予算100万円では大型株
# ほど買えず資金が遊ぶのが最大の敗因だったため（REQUIREMENTS 4.4-8）
TICKERS_CSV = BASE_DIR / "universe.csv"
# info（財務データ）の取得は1銘柄ずつHTTPリクエストが飛ぶため、944銘柄では
# 直列だと数十分かかる。IO待ちなのでスレッドで並列化する
INFO_WORKERS = 8
# 財務データを取りに行く銘柄数の上限。テクニカルスコア上位のみに絞る
# （944銘柄すべてに info を投げるとレート制限で全滅するため）
FUNDAMENTAL_POOL_SIZE = 120
# レート制限に当たったときの待ち時間（秒）。試行ごとに倍にして待つ
RATE_LIMIT_WAIT_SEC = 5
# 財務データの取得がこの割合を超えて失敗したら、割安度スコアが機能しないので
# 警告する（欠損は一律0.5点として扱われるため、黙っていると
# 「テクニカルだけで選んだ結果」が通常の推奨に見えてしまう）
FUNDAMENTAL_FAILURE_WARN_RATIO = 0.3

# --- 片山流「新高値ブレイク投資」（別系統。片山晃_ルール.md / REQUIREMENTS 4.4-14）---
# 現行ルールとは買う位置が正反対（押し目 vs 新高値）なので、同じスコアに混ぜず
# 別枠で出す。14年5,768件の検証で、勝率はほぼ同じままPFが1.83倍・平均リターンが
# 1.85倍になった条件を使う（件数は13分の1）
NEW_HIGH_PERIOD = 250          # 52週高値。著者が現在使っている期間
# 時価総額の上限（億円）。著者は「時価総額50億円〜数百億円台の中小型株」を
# 対象にしている（PART 6 条件①）。26年のトレードを時価総額バンド別に見たところ、
# **300億円未満が重複しない3期間すべてで最良**だった（REQUIREMENTS 4.4-25）。
# ⚠️ 必須条件にはしない。時価総額を出せるのはEDINETに株数がある729/944銘柄
# だけで、必須にすると判定不能な215銘柄を落としてしまうため。印として出す。
KATAYAMA_SMALL_CAP_OKU = 300
KATAYAMA_STOP_LOSS_PCT = 8.0   # 損切り-8%（現行ルールの-10%より厳しい）

# 条件は2種類を併記する。書籍に明記された条件と、このツールの検証で最も
# 成績が良かった条件がPERで食い違うため、どちらが実際に機能するかを
# 運用しながら見るのが目的（REQUIREMENTS 4.4-15）。
#
#   書籍版：増収10%↑・増益20%↑・ROE10%↑・PER39倍以下
#           （著者は「PER30倍台まで買い、50倍超で売り」「PER15倍未満を
#            対象にしていてはチャンスを逃す」と明言している）
#   検証版：増収10%↑・増益30%↑・PER20倍未満
#           （14年5,768件の検証ではPER30倍台がPF0.50と最悪だった。
#            著者の対象はグロース市場の中小型株、こちらは全上場から
#            予算内・流動性で絞った944銘柄なので母集団が違う）
KATAYAMA_BOOK = {
    "label": "書籍版", "min_rev": 10.0, "min_profit": 20.0,
    "min_roe": 10.0, "max_per": 39.0,
}
KATAYAMA_TESTED = {
    "label": "検証版", "min_rev": 10.0, "min_profit": 30.0,
    "min_roe": None, "max_per": 20.0,
}
# 長期版：PART 6「中小型株の中長期投資」の条件。著者は「利益は伸びるのも
# 落ち込むのもどちらもOK」「増収減益はむしろ先行投資のチャンス」と書いており、
# 売上高の伸びだけを見る。増益を要求しない点が上の2つと決定的に違う。
#
# ⚠️ **長期保有が前提**。上の2つ（短期のエグジット＝PPP崩れ or -8%、平均保有
# 30日）では増益条件を付けたほうが明確に良い。逆に500日フォワードリターンで
# 見ると増収重視が全期間で最良になる（REQUIREMENTS 4.4-16）。
# PER39倍以下（＝著者の「PER30倍台まで買い」）は3期間すべてで勝率が改善したので
# 付ける。付けないとPER213倍・ROE0.3%のような銘柄が通ってしまい、PART 6 が
# 前提にしている「成長余地のある会社」と別物になる
KATAYAMA_LONG = {
    "label": "長期版", "min_rev": 10.0, "min_profit": None,
    "min_roe": None, "max_per": 39.0,
}
EDINET_FINANCIALS_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
# 片山晃 PART 7 のOKポイント①②「上場から5年以内／10年以内」の判定用。
# 成長余地（伸びしろ）が大きく残っている会社を見分ける指標として使う。
# ⚠️ NGポイント②「上場5年以内に下方修正2回以上」は**実装していない**。
# 下方修正の履歴はTDnetの適時開示にしかなく、TDnetは直近1か月ぶんしか
# 公開していない（2026-08-30に実測。1年前の日付は404）。上場年数だけで
# 除外すると著者の意図（下方修正を連発する会社を避ける）と別物になるため、
# 片方だけの実装はしない
LISTING_DATES_PATH = BASE_DIR / "data" / "listing_dates.json"

# 片山晃 PART 7 のNGポイント②「下方修正を連発する会社は避ける」。
# J-Quants（JPX公式）の財務サマリーから会社予想の推移を追って数える。
# 2026-09-02にStandardプランへ移行し、120件/分・遅延なし・10年分になった。
# 944銘柄すべてに回すと8分かかるので、**片山流の候補だけ**に絞る方針は維持する
# （下方修正のチェックが要るのは片山流の判定だけなので、これで足りる）。
# 書籍の条件は**「上場5年以内に下方修正2回以上」**で、期間が限定されている。
# ⚠️ 2026-09-02にStandardプランへ移行して10年分が見えるようになったところ、
# 期間を切らずに数えていたため古い会社ほど回数が積み上がり、候補がほぼ全滅した
# （エレコム2回・ハピネット3回・不二製油5回）。書籍どおり窓を切ること。
KATAYAMA_MAX_DOWNWARD_REVISIONS = 2
KATAYAMA_REVISION_WINDOW_YEARS = 5    # 上場から何年以内を見るか（書籍の規定）
# 1回の実行でJ-Quantsに問い合わせる上限。Standardは120件/分なので
# 60件でも約36秒しかかからない（無料プランのときは12件で2.6分かかっていた）
JQUANTS_MAX_LOOKUPS = 60
OUTPUT_DIR = BASE_DIR / "output"
EDINET_CACHE_PATH = BASE_DIR / "data" / "edinet_valuation_diff.json"

FUNDAMENTAL_WEIGHT = 0.5
TECHNICAL_WEIGHT = 0.5

BUDGET = 1_000_000  # 予算（円）。日本株は基本100株単位（単元株）での購入となる
LOT_SIZE = 100

MA_PERIODS = (5, 10, 20, 50, 100)
MIN_HISTORY_DAYS = 105  # 100日線 + スイングカウント用のバッファ

# 25日線からのかい離率（参考指標）。当初は「20%以上の上方乖離で早期利確」
# というエグジットルールとして採用したが、2026-08-29の追加検証で下記の
# タイムストップ＋損切りに置き換えたため、現在はCSVの参考列としてのみ使う
DEV_MA_PERIOD = 25

# 手仕舞いルール（2026-08-29採用）：「保有60日で手仕舞い」＋「買値-10%で損切り」
# バックテストで5年・10年・26年の全期間で勝率・平均リターンがともに一貫して
# 改善したため採用：
#   勝率      55.4%→59.5%（5年）、45.7%→51.5%（10年）、45.2%→50.2%（26年）
#   平均R    +3.68%→+5.63%、+2.11%→+3.26%、+2.01%→+2.87%
# 「損切りなしのタイムストップ単独」は勝率54.4%（26年）とさらに高かったが、
# 下げ続ける銘柄を60日持ち切るため-20%超の大損が2件→50件に激増したので不採用。
# 損切り-10%を併用することで大損を6件に抑えつつ、勝率と平均リターンの
# 改善分を確保している（PFは1.80→1.64とやや低下するトレードオフあり）
HOLDING_DAYS_LIMIT = 60
STOP_LOSS_PCT = 10.0

# 押し目買いシグナル（グランビルの法則②③由来、2026-08-29採用）。
# 上昇トレンド中に安値が20日線＋この％以内まで押し、当日陽線で20日線より
# 上に戻して反発した日を買いとする。既存の下半身（ブレイクを買う）とは
# 逆の発想で、「下半身 or 押し目買い」のどちらかが点灯したら買う運用。
# バックテストで5年・10年・26年の全期間で勝率が一貫改善し、しかも
# PFは同等以上を維持した（今回検証した全候補で唯一、収益性を犠牲に
# しなかった）：
#   勝率 59.5→60.6%(5年)、51.5→53.3%(10年)、50.2→52.0%(26年)
#   PF   2.55→2.65、1.81→1.81、1.64→1.69
#   件数 274→536、592→1155、1640→3139（約2倍。大損の発生率は0.37→0.32%）
PULLBACK_TOLERANCE_PCT = 2.0


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
    # 個別株と同じ経路（tzなしindex）で取らないと、レラティブストレングスの
    # 計算で tz付き/なし の比較になり全銘柄が失敗する
    return fetch_history("^N225", period=period, stale_days=0)["Close"]


def fetch_market_regime(adx_threshold: float = 20.0) -> bool:
    """
    マーケットレジームフィルター：日経平均自体が上昇トレンド（終値が100日線より上）
    かつ、ADXがadx_threshold超（トレンドに十分な勢いがある＝レンジ相場でない）か。
    バックテストで確認済み：5年・10年・26年のすべての期間で、単純な「100日線より
    上」だけの判定より、ADXでトレンド強度も見る方が勝率・平均リターン・
    プロフィットファクターが一貫して改善した（26年PF 1.67→2.02）。
    """
    idx = fetch_history("^N225", period="1y", stale_days=0)
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


def fetch_fundamentals(code: str, edinet_cache: Optional[dict] = None,
                       retries: int = 2) -> dict:
    """
    財務データ（yfinanceのinfo）を取る。1銘柄につきHTTPリクエストが1本飛ぶため、
    944銘柄すべてに対して呼ぶとYahooのレート制限（Too Many Requests）に当たる。
    テクニカル評価で上位に入った銘柄にだけ呼ぶこと（main参照）。
    レート制限は時間を置けば解けるので、少し待って数回だけ試し直す。
    """
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(code).info
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(RATE_LIMIT_WAIT_SEC * (attempt + 1))
    else:  # pragma: no cover
        info = {}
    edinet_info = (edinet_cache or {}).get(code, {})
    return {
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


def fetch_one(code: str, name: str, edinet_cache: Optional[dict] = None,
              nikkei_close: Optional[pd.Series] = None,
              hist: Optional[pd.DataFrame] = None,
              with_fundamentals: bool = True) -> dict:
    """
    1銘柄ぶんの評価行を作る。with_fundamentals=False なら株価データだけで
    計算できるテクニカル指標のみを埋める（ネットワーク不要）。
    """
    if hist is None:
        # 一括取得から漏れた銘柄のフォールバック。index の tz を落として
        # キャッシュ経由のデータと揃える（混在すると日経平均との比較で落ちる）
        hist = yf.Ticker(code).history(period="3y")
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

    row = {"code": code, "name": name}
    if with_fundamentals:
        row.update(fetch_fundamentals(code, edinet_cache))
    else:
        row.update({k: None for k in (
            "price", "per", "pbr", "dividend_yield", "current_ratio",
            "debt_to_equity", "securities_valuation_diff_change_yen")})

    if len(hist) >= MIN_HISTORY_DAYS:
        close = hist["Close"]
        open_ = hist["Open"]
        low = hist["Low"]
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

        # 5日線自体の向き（4営業日前と比較した傾き）。
        # 2026-08-29修正：以前は iloc[-4]（＝3営業日前）と比較していたが、
        # backtest.py / portfolio_sim.py は4営業日前と比較しており、
        # 検証したルールと実際の判定がズレていた（下半身シグナルの約19%で
        # 判定が分かれ、screen.py側が1割ほど取りこぼしていた）。
        # バックテストで検証済みのルールに合わせる。
        sma5_series = sma[5]
        sma5_prev = sma5_series.iloc[-5]
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

        # 押し目買い：安値が20日線＋2%以内まで押し、陽線で20日線より上に戻した
        sma20_now = sma[20].iloc[-1]
        # 新高値ブレイク（片山流）。過去NEW_HIGH_PERIOD営業日の最高値を更新したか
        if len(close) > NEW_HIGH_PERIOD:
            prev_max = close.iloc[-(NEW_HIGH_PERIOD + 1):-1].max()
            row["new_high"] = bool(row["last_close"] > prev_max)
            row["new_high_ref"] = float(prev_max)
        else:
            row["new_high"] = False
            row["new_high_ref"] = None

        # カップ・ウィズ・ハンドル（片山晃_ルール.md PART 4 補足／原典はオニール）。
        # 新高値のうちこの形が完成しているものは、重複しない3期間すべてで
        # PFが改善した（REQUIREMENTS 4.4-17）。件数が6分の1に減るので
        # 必須条件にはせず、印として付けて優先順位の判断に使う
        row["cup_with_handle"] = bool(
            row["new_high"] and calc_cup_with_handle(close, NEW_HIGH_PERIOD).iloc[-1]
        )

        row["pullback"] = bool(
            pd.notna(sma20_now)
            and low.iloc[-1] <= sma20_now * (1 + PULLBACK_TOLERANCE_PCT / 100)
            and is_bullish
            and row["last_close"] > sma20_now
        )

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
        row["kahanshin"] = row["gyaku_kahanshin"] = row["pullback"] = None
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
    if row.get("pullback"):
        base += "・本日「押し目買い」シグナル点灯（20日線まで押して反発）"
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
    if row.get("pullback"):
        signals.append("押し目買い")
    if row.get("kuchibashi_signal") == "up":
        signals.append("くちばし")

    if not signals:
        return "本日は買いシグナルなし（様子見）"

    notes = []
    if not row.get("trend_filter_pass"):
        notes.append("トレンドやや弱め")
    if not market_regime_up:
        notes.append("日経平均の地合いが弱い（トレンド不足）")
    if (row.get("kahanshin") or row.get("pullback")) and not row.get("volume_confirmed"):
        notes.append("出来高の伴いが弱い")
    confidence = f"（{'・'.join(notes)}・慎重に）" if notes else ""
    return (
        f"買いタイミング（{'・'.join(signals)}点灯）{confidence}: "
        f"{row['price']:,.0f}円 × 100株 = {row['lot_cost']:,.0f}円"
    )


def sell_timing(row) -> str:
    """
    手仕舞い（保有株の売却）の目安を示す。2026-08-29のバックテスト結果に
    基づき、ルールを「5日線が20日線を下抜けたら（PPP崩れ）」から
    「保有60日で手仕舞い ＋ 買値-10%で損切り（早い方）」に変更した。
    本日この銘柄を買った場合を前提に、損切り価格と手仕舞い期限を具体的に示す。
    空売りの新規シグナルではなく、現物買いした株を売却するタイミングの目安。
    """
    price = row.get("price")
    if pd.isna(price):
        return "データ不足"

    stop_price = price * (1 - STOP_LOSS_PCT / 100)
    deadline = dt.date.today() + dt.timedelta(days=HOLDING_DAYS_LIMIT)
    base = (
        f"売り目安: 損切り{stop_price:,.0f}円（買値-{STOP_LOSS_PCT:.0f}%）"
        f"／{deadline:%Y-%m-%d}頃まで（保有{HOLDING_DAYS_LIMIT}日）に手仕舞い"
        f"。※本日の価格で買った場合"
    )

    buy = row.get("td_buy")
    if buy in (9, 17, 23):
        base += f" ／ 9の法則が{int(buy)}本目＝手仕舞いを強く意識する節目に到達"
    elif pd.notna(buy) and buy >= 6:
        next_checkpoint = 9 if buy < 9 else (17 if buy < 17 else 23)
        base += f" ／ 9の法則は現在{int(buy)}本目（次の節目は{next_checkpoint}本目）"

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

    # 買いシグナル（下半身 or 押し目買い）が本日点灯＝号砲。バックテストで
    # 確認済みの各条件を加算方式で評価する（PF実績: 基本1.29 → トレンド強め
    # 1.44 → +地合い1.80 → +出来高2.60）。条件が重なるほど信頼度が高い。
    # 押し目買いは2026-08-29に追加（下半身と同格の独立シグナルとして扱う。
    # 併用で勝率が全期間改善しPFも同等以上を維持したため、加点も同じ扱い）
    if row["kahanshin"] or row.get("pullback"):
        bonus = 0.10  # 買いシグナルが出ただけでも号砲として加点
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


def _as_number(series: pd.Series) -> pd.Series:
    """
    数値として扱うべき列を確実に数値にする。yfinanceのinfoは銘柄によって
    同じ項目を文字列で返すことがあり、混ざると rank() が
    「'<' not supported between instances of 'str' and 'float'」で落ちる
    （2026-08-30にPER列で実際に発生）。数値にできない値はNaN＝欠損扱い。
    """
    return pd.to_numeric(series, errors="coerce")


def rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    パーセンタイル順位を0〜1のスコアに変換する。
    データ欠損（NaN）は「最良」でも「最悪」でもなく中立（0.5点）として扱う。
    （pandasのrank(na_option="bottom")はNaNを最大値として扱うため、そのまま
    使うと「データが無い銘柄が満点になる」という誤った結果になることがあった。
    2026-08-18に発見・修正。）
    """
    pct = _as_number(series).rank(pct=True, na_option="keep")
    if not higher_is_better:
        pct = 1 - pct
    return pct.fillna(0.5)


def load_growth() -> dict:
    """
    EDINETの決算データから、銘柄ごとの直近の増収率・増益率を出す。
    通信不要（scripts/build_edinet_financials.py が作ったファイルを読むだけ）。

    ⚠️ available_from（決算期末+92日＝有報の提出期限）を過ぎた決算だけを使う。
    まだ公表されていない決算を使うと、実運用でも再現できない推奨になる。
    """
    if not EDINET_FINANCIALS_PATH.exists():
        return {}
    data = json.loads(EDINET_FINANCIALS_PATH.read_text(encoding="utf-8"))["data"]
    today = dt.date.today().isoformat()
    out = {}
    for code, periods in data.items():
        recs = sorted((r for r in periods.values() if r.get("available_from", "") <= today),
                      key=lambda r: r["period_end"])
        if len(recs) < 2:
            continue
        cur, prev = recs[-1], recs[-2]
        # EDINET版は営業利益を持たず経常利益（IFRSは税引前利益）。
        # 「利益の伸び」を見る目的なのであるほうを使う
        pkey = "operating_income" if cur.get("operating_income") is not None else "ordinary_income"

        def rate(key):
            a, b = cur.get(key), prev.get(key)
            if a is None or b is None or not b or b <= 0:
                return None
            return (a - b) / b * 100

        # ROE＝当期純利益÷自己資本×100（著者がPART 4で追加した条件）
        ni, na = cur.get("net_income"), cur.get("net_assets")
        out[code] = {
            "revenue_growth": rate("revenue"),
            "profit_growth": rate(pkey),
            "roe": (ni / na * 100) if (ni is not None and na and na > 0) else None,
            "period_end": cur["period_end"],
            "eps": cur.get("eps"),
            "shares": cur.get("shares"),
        }
    return out


def load_listing_dates() -> dict:
    """
    銘柄ごとの上場日（"YYYY-MM-DD"）を返す。通信不要。
    2001年以前から上場している銘柄（old_listing）は、yfinanceのデータ開始日が
    本当の上場日ではないので除く。片山流NGポイント②の「上場5年以内」という
    窓を切るのに使う。
    """
    if not LISTING_DATES_PATH.exists():
        return {}
    data = json.loads(LISTING_DATES_PATH.read_text(encoding="utf-8"))["data"]
    return {code: r["first_trade_date"] for code, r in data.items()
            if not r.get("old_listing") and r.get("first_trade_date")}


def load_listing_years() -> dict:
    """
    銘柄ごとの上場からの年数を返す（通信不要）。
    2001年以前から上場している銘柄は yfinance のデータ開始日しか分からないため
    None を返す（「上場から25年」などと誤解しないように）。
    """
    if not LISTING_DATES_PATH.exists():
        return {}
    data = json.loads(LISTING_DATES_PATH.read_text(encoding="utf-8"))["data"]
    today = dt.date.today()
    out = {}
    for code, r in data.items():
        if r.get("old_listing"):
            out[code] = None
            continue
        try:
            d = dt.date.fromisoformat(r["first_trade_date"])
        except (ValueError, KeyError):
            continue
        out[code] = round((today - d).days / 365.25, 1)
    return out


def fetch_jquants_facts(codes: list, listing_dates: dict = None) -> dict:
    """
    J-Quants から片山流の候補について2つを引く（1銘柄1コールで両方まかなう）。

      down     … 会社予想の下方修正回数（PART 7 NGポイント②）
      quarter  … 直近四半期の前年同期比 増収率（PART 6「四半期ごとに10%増」）
      progress … 通期の会社予想に対する進捗率（PART 5。四半期25%ずつが目安）

    このツールのEDINETデータは**有価証券報告書＝年次**なので、四半期の
    前年同期比は年次からは出せない。同じ `summary()` の応答に四半期累計の
    売上（Sales）が入っているので、**APIコールを増やさずに**両方を得る。

    APIキーが無い・APIが落ちている場合は静かに空を返す＝除外しない。
    ここで落ちて推奨全体が出せなくなるほうが困るため。
    """
    listing_dates = listing_dates or {}
    if not codes:
        return {}
    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from jquants import (JQuantsClient, count_downward_revisions,
                             progress_rate, quarterly_revenue_growth)
        cli = JQuantsClient()
    except Exception as e:
        print(f"  J-Quantsを使えないので下方修正と四半期増収率は省略します: {e}")
        return {}

    out = {}
    print(f"  片山流の候補{len(codes)}銘柄をJ-Quantsで確認します"
          f"（下方修正＋四半期の前年同期比＋進捗率）…")
    for code in codes:
        try:
            rows = cli.summary(code)
            # NGポイント②は「上場5年以内」の話なので、その窓だけを数える。
            # 上場から5年を過ぎた会社にはこの条件を適用しない（Noneを返す）
            listed = listing_dates.get(code)
            down = None
            if listed:
                since = listed
                end = (dt.date.fromisoformat(listed)
                       + dt.timedelta(days=365 * KATAYAMA_REVISION_WINDOW_YEARS))
                if dt.date.today() <= end:
                    down = count_downward_revisions(rows, since=since)["down"]
            out[code] = {
                "down": down,
                "quarter": quarterly_revenue_growth(rows),
                "progress": progress_rate(rows),
            }
        except Exception as e:
            print(f"    {code}: 取得失敗のためチェックせず通す ({str(e)[:60]})")
    return out


def build_ranking(rows: list[dict], budget: int = BUDGET, market_regime_up: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # yfinanceのinfoは銘柄によって同じ項目を文字列で返すことがある。
    # 個別に直すと必ず漏れるので、数値として扱う列は入口でまとめて変換する
    # （2026-08-30、PER列の文字列混入で rank() が落ちた。その修正後も
    #  graham_number = per × pbr の pbr で同じ型エラーが再発した）
    for col in ("price", "per", "pbr", "dividend_yield", "current_ratio",
                "debt_to_equity", "securities_valuation_diff_change_yen",
                "revenue_growth", "profit_growth", "roe", "years_since_listing",
                "rsi14", "volume_ratio", "dev_from_sma25_pct", "ppp_matches",
                "td_buy", "td_sell", "last_close", "last_open",
                "sma5", "sma10", "sma20", "sma50", "sma100"):
        if col in df.columns:
            df[col] = _as_number(df[col])

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

    # 採用ルール（backtest.py で検証した条件一式）を満たしているかを列に落とす。
    # 通知側で buy_timing の文字列を解析すると、文言を変えたときに静かに壊れる
    pool["cond_signal"] = pool["kahanshin"] | pool.get("pullback", False)
    pool["cond_trend"] = pool["trend_filter_pass"].fillna(False).astype(bool)
    pool["cond_volume"] = pool["volume_confirmed"].fillna(False).astype(bool)
    pool["cond_rs"] = pool["relative_strength_confirmed"].fillna(False).astype(bool)
    pool["cond_regime"] = bool(market_regime_up)
    cond_cols = ["cond_signal", "cond_trend", "cond_volume", "cond_rs", "cond_regime"]
    pool["conditions_met"] = pool[cond_cols].sum(axis=1)
    # 全条件を満たす行だけが、バックテストの成績（勝率60.6%/PF2.65）の前提に合う
    pool["conditions_all"] = pool["conditions_met"] == len(cond_cols)

    pool["technical_score"] = pool.apply(lambda r: technical_score(r, market_regime_up), axis=1)
    pool["trend_label"] = pool.apply(trend_label, axis=1)
    pool["td_label"] = pool.apply(td_label, axis=1)
    pool["buy_timing"] = pool.apply(lambda r: buy_timing(r, market_regime_up), axis=1)
    pool["sell_timing"] = pool.apply(sell_timing, axis=1)

    # --- 片山流（別系統・REQUIREMENTS 4.4-14/15）---
    # 現行スコアとは混ぜない。書籍版と検証版の2条件をそれぞれ判定する
    per = pool["per"]
    nh = pool["new_high"].fillna(False).astype(bool)

    def _match(spec):
        m = nh & (pool["revenue_growth"].fillna(-999) >= spec["min_rev"])
        # 長期版は増益を条件にしない（PART 6：利益は増えても減ってもOK）
        if spec["min_profit"] is not None:
            m &= pool["profit_growth"].fillna(-999) >= spec["min_profit"]
        if spec["max_per"] is not None:
            m &= per.notna() & (per > 0) & (per <= spec["max_per"])
        if spec["min_roe"] is not None:
            m &= pool["roe"].fillna(-999) >= spec["min_roe"]
        return m

    pool["katayama_book"] = _match(KATAYAMA_BOOK)
    pool["katayama_tested"] = _match(KATAYAMA_TESTED)
    pool["katayama_long"] = _match(KATAYAMA_LONG)
    # 小型株の印（必須条件ではない）。時価総額が分からない銘柄は False のまま
    cap = pd.to_numeric(pool.get("market_cap_oku"), errors="coerce")
    pool["small_cap"] = (cap.notna() & (cap < KATAYAMA_SMALL_CAP_OKU))
    pool["katayama"] = (pool["katayama_book"] | pool["katayama_tested"]
                        | pool["katayama_long"])

    # NGポイント②：下方修正を繰り返す会社を外す。候補にだけJ-Quantsを引く
    pool["downward_revisions"] = None
    pool["q_revenue_growth"] = None       # 四半期累計の前年同期比
    pool["q_revenue_growth_sa"] = None    # その四半期"単独"の前年同期比
    pool["q_period"] = None
    pool["progress_sales"] = None    # 通期予想に対する売上の進捗率
    pool["progress_op"] = None       # 同・営業利益
    pool["progress_expected"] = None # 目安（25%×四半期数）
    # ⚠️ J-Quants無料プランは約4か月遅れなので、いつ時点の数字かを必ず持つ
    pool["jq_disc_date"] = None
    cand = pool.index[pool["katayama"]].tolist()[:JQUANTS_MAX_LOOKUPS]
    if cand:
        facts = fetch_jquants_facts([str(pool.at[i, "code"]) for i in cand],
                                    load_listing_dates())
        for i in cand:
            f = facts.get(str(pool.at[i, "code"])) or {}
            q = f.get("quarter") or {}
            pool.at[i, "q_revenue_growth"] = q.get("cumulative")
            pool.at[i, "q_revenue_growth_sa"] = q.get("standalone")
            pool.at[i, "q_period"] = q.get("period")
            pr = f.get("progress") or {}
            pool.at[i, "progress_sales"] = pr.get("sales")
            pool.at[i, "progress_op"] = pr.get("op")
            pool.at[i, "progress_expected"] = pr.get("expected")
            pool.at[i, "jq_disc_date"] = (pr.get("disc_date")
                                          or q.get("disc_date"))
            n = f.get("down")
            pool.at[i, "downward_revisions"] = n
            if n is not None and n >= KATAYAMA_MAX_DOWNWARD_REVISIONS:
                # 下方修正が多い会社は片山流の対象から外す（NGポイント②）
                pool.at[i, "katayama_book"] = False
                pool.at[i, "katayama_tested"] = False
                pool.at[i, "katayama_long"] = False
                pool.at[i, "katayama"] = False

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

    # 株価は一括で取る。当日の推奨を出すので stale_days=0 を指定して
    # 必ず最新の足まで取り直す（古いキャッシュで推奨を出すと静かに狂う）
    hist_map = fetch_histories([t["code"] for t in tickers], period="3y", stale_days=0)

    # 第1段階：株価だけで計算できるテクニカル指標を全銘柄ぶん作る（通信なし）
    rows, failures = [], []
    for t in tickers:
        code, name = t["code"], t["name"]
        try:
            rows.append(fetch_one(code, name, edinet_cache, nikkei_close,
                                  hist=hist_map.get(code), with_fundamentals=False))
        except Exception as e:
            failures.append(f"{name} ({code}) 評価失敗: {e}")
    print(f"  テクニカル評価: {len(rows)}件（失敗{len(failures)}件）")

    # 第2段階：財務データはテクニカル上位の銘柄にだけ取りに行く。
    # info は1銘柄1リクエストで、944銘柄ぶん投げるとYahooに
    # 「Too Many Requests」で弾かれ全滅する（2026-08-29に実際に発生）。
    # 圏外の銘柄は総合スコアで上位に来る余地がないため、実質の損失はない。
    # 決算データ（通信不要）を各行に載せる。片山流の判定に使う
    growth = load_growth()
    listing_years = load_listing_years()
    for r in rows:
        r["years_since_listing"] = listing_years.get(r["code"])
        g = growth.get(r["code"], {})
        r["revenue_growth"] = g.get("revenue_growth")
        r["profit_growth"] = g.get("profit_growth")
        # 時価総額（億円）＝株価 × 発行済株式数。EDINETに株数がある銘柄だけ。
        # ⚠️ ここは第2段階（info取得）より前なので "price" はまだ入っていない。
        # 第1段階で入る "last_close"（終値）を使う
        sh, px = g.get("shares"), r.get("last_close")
        try:
            r["market_cap_oku"] = float(px) * sh / 1e8 if sh and px is not None else None
        except (TypeError, ValueError):
            r["market_cap_oku"] = None
        r["roe"] = g.get("roe")

    def _tech(r):
        # 株価データが短い銘柄は指標が揃わない。順位付けの前段なので0点扱いにする
        try:
            return technical_score(r, market_regime_up)
        except (KeyError, TypeError):
            return 0.0

    rows.sort(key=_tech, reverse=True)
    targets, rest = rows[:FUNDAMENTAL_POOL_SIZE], rows[FUNDAMENTAL_POOL_SIZE:]

    # 片山流の候補は「新高値＋高成長」で、押し目狙いのテクニカルスコアでは
    # 上位に来ないことが多い。圏外にいる候補も財務データの取得対象に加える
    # （加えないとPERが埋まらず、片山流の判定が常に不成立になる）
    picked = {id(r) for r in targets}
    # 長期版は増益を条件にしないので、増収だけで拾う（利益で足切りしない）
    _min_rev = min(v["min_rev"] for v in
                   (KATAYAMA_BOOK, KATAYAMA_TESTED, KATAYAMA_LONG))
    extra = [r for r in rest
             if r.get("new_high")
             and (r.get("revenue_growth") or -999) >= _min_rev
             and id(r) not in picked]
    if extra:
        targets = targets + extra
        rest = [r for r in rest if id(r) not in {id(x) for x in extra}]
    print(f"  財務データを取得する{len(targets)}銘柄を選定しました"
          f"（テクニカル上位{FUNDAMENTAL_POOL_SIZE} ＋ 片山流候補{len(extra)}）…")

    def _fund(r):
        try:
            return r["code"], fetch_fundamentals(r["code"], edinet_cache), None
        except Exception as e:
            return r["code"], None, f"{r['name']} ({r['code']}) 財務データ取得失敗: {e}"

    fund_failures = []
    with ThreadPoolExecutor(max_workers=INFO_WORKERS) as pool:
        for i, (code, fund, err) in enumerate(pool.map(_fund, targets), 1):
            if fund is not None:
                next(r for r in targets if r["code"] == code).update(fund)
            else:
                fund_failures.append(err)
            if i % 25 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] 財務データ取得済み"
                      f"{i - len(fund_failures)}件 / 失敗{len(fund_failures)}件")

    # 財務データが取れなかった銘柄は、株価が無くランキングに乗せられない。
    # 終値で代用して「予算内で買えるか」の判定だけは通す
    for r in targets:
        if r.get("price") is None:
            r["price"] = r.get("last_close")
    rows = [r for r in targets if r.get("price") is not None]
    failures.extend(fund_failures)
    for f in failures[:10]:
        print(f"  {f}")
    if len(failures) > 10:
        print(f"  …ほか{len(failures) - 10}件")
    print(f"  → {len(rows)}銘柄をランキング対象にします"
          f"（テクニカル圏外{len(rest)}銘柄は財務データを取らず除外）")

    # 財務データの欠損は rank_score で一律0.5点になる＝割安度が効かなくなる。
    # 黙って進むと「テクニカルだけで選んだ結果」が通常の推奨に見えるため明示する
    if targets and len(fund_failures) / len(targets) > FUNDAMENTAL_FAILURE_WARN_RATIO:
        print(f"\n⚠️  財務データが{len(fund_failures)}/{len(targets)}銘柄で取得できませんでした"
              "（Yahooのレート制限の可能性）。\n"
              "    欠損は中立（0.5点）扱いになるため、この結果は実質"
              "テクニカルのみのランキングです。\n"
              "    時間を置いて再実行してください。")

    df, excluded = build_ranking(rows, market_regime_up=market_regime_up)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"recommend_{dt.date.today():%Y%m%d}.csv"

    cols = [
        "code", "name", "price", "lot_cost", "per", "pbr", "dividend_yield",
        "graham_number", "current_ratio", "debt_to_equity", "securities_valuation_diff_change_yen",
        "sma5", "sma10", "sma20", "sma50", "sma100", "ppp_matches", "trend_filter_pass",
        "volume_ratio", "volume_confirmed", "relative_strength_confirmed", "dev_from_sma25_pct",
        "kahanshin", "pullback",
        "cond_signal", "cond_trend", "cond_volume", "cond_rs", "cond_regime",
        "conditions_met", "conditions_all",
        "new_high", "cup_with_handle", "market_cap_oku", "small_cap", "revenue_growth", "profit_growth", "roe", "years_since_listing",
        "katayama", "katayama_book", "katayama_tested", "katayama_long",
        "downward_revisions", "q_revenue_growth", "q_revenue_growth_sa",
        "q_period", "progress_sales", "progress_op", "progress_expected", "jq_disc_date",
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
