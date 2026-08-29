"""
ポートフォリオシミュレーション：予算制約を入れた現実的な成績検証

backtest.py は「シグナルが出た全銘柄を売買できた」前提でトレード単位に
集計しているが、実際には予算（既定100万円・100株単位）の範囲でしか
買えず、シグナルが同時に多数出た日はどれかを選ぶ必要がある。
このスクリプトは資金を実際に回しながら日次でシミュレーションし、
「実際にいくらになるのか」を測る。

エントリー: 下半身 or 押し目買い（screen.py / backtest.py の採用ルール）
            ＋ PPP3/4以上・100日線上・日経ADXレジーム・出来高・相対力
エグジット: 保有60日 or 買値-10%の損切り（早い方）

使い方:
    python portfolio_sim.py [期間] [選択ルール]
      期間     : 5y / 10y / max（既定 5y）
      選択ルール: volume（出来高が強い順・既定）/ first（銘柄コード順）
                 / random（ランダム。seed違いで複数回試行して平均を出す）
出力:
    標準出力に資産推移サマリー
"""

import csv
import datetime as dt
import random
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from backtest import (
    SUSPICIOUS_RETURN_THRESHOLD,
    fetch_market_regime_adx,
    fetch_nikkei_close,
)
from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"

INITIAL_CAPITAL = 1_000_000
LOT_SIZE = 100
HOLDING_DAYS_LIMIT = 60
STOP_LOSS_PCT = 10.0
PULLBACK_TOLERANCE_PCT = 2.0
COST_PCT = 0.2  # 往復の取引コスト（手数料＋スリッページ）
RANDOM_TRIALS = 5

# 1日の終値がこの倍率を超えて動いた銘柄は、yfinanceの株式分割データ不整合に
# よる汚染データとみなして対象から除外する（東京海上HD/8766の2006-09-29に
# 誤った分割比率500が記録されており、それ以前の株価が1/500になっている等）。
# 日経225の大型株が1日で+80%動くことは実質ないため、誤検知の心配は小さい。
# 資産推移そのものが壊れるため、集計時の除外ではなく読み込み時に落とす。
MAX_PLAUSIBLE_DAILY_MOVE = 0.8

# 余剰資金を日経ETFで運用する場合（parkindex）のコスト。
# 個別株を買うときはその分だけETFを売り、売ったときはETFを買い戻すため、
# 個別株の売買のたびに片道分のETF売買コストがかかる（27年で876トレード
# ＝月2.7回程度なので、日々売買するような非現実的な頻度にはならない）。
ETF_TRADE_COST_PCT = 0.05   # ETFの片道売買コスト（スプレッド＋手数料）
ETF_EXPENSE_RATIO = 0.0015  # 信託報酬 年0.15%
TRADING_DAYS_PER_YEAR = 252

# 譲渡益課税（所得税15%＋復興特別所得税0.315%＋住民税5%）。
# その年の実現損益を通算し、プラスなら年末に課税する（損益通算は再現するが、
# 3年間の繰越控除は再現していない＝実際よりやや不利に出る）。
TAX_RATE = 0.20315


def load_tickers(path=None):
    """既定は日経225（tickers.csv）。universe.csv 等を渡せば対象を差し替えられる。"""
    with open(path or TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_signals(hist: pd.DataFrame, nikkei_close: pd.Series) -> pd.DataFrame:
    """
    1銘柄について、日次のエントリーシグナルと補助情報をまとめたDataFrameを返す。
    backtest.py の entry_mode="either" ＋ trend/volume/rs フィルターと同じ条件。
    """
    close, open_, low, volume = hist["Close"], hist["Open"], hist["Low"], hist["Volume"]
    ma = {n: close.rolling(n).mean() for n in (5, 10, 20, 50, 100)}

    is_bullish = close > open_
    # 下半身：5日線が上向き＋陽線で5日線を上抜け
    kahanshin = (
        (close.shift(1) <= ma[5].shift(1)) & (close > ma[5])
        & is_bullish & (ma[5] > ma[5].shift(4))
    )
    # 押し目買い：20日線まで押して陽線で反発
    pullback = (
        (low <= ma[20] * (1 + PULLBACK_TOLERANCE_PCT / 100))
        & is_bullish & (close > ma[20])
    )

    ppp_matches = sum(
        (ma[a] > ma[b]).astype(int)
        for a, b in ((5, 10), (10, 20), (20, 50), (50, 100))
    )
    trend_ok = (ppp_matches >= 3) & (close > ma[100])

    vol_avg20 = volume.rolling(20).mean()
    volume_ratio = volume / vol_avg20
    volume_ok = volume_ratio >= 1.5

    nikkei_aligned = nikkei_close.reindex(close.index, method="ffill")
    rs_ratio = close / nikkei_aligned
    rs_ok = rs_ratio > rs_ratio.rolling(50).mean()

    return pd.DataFrame({
        "close": close,
        "signal": (kahanshin | pullback) & trend_ok & volume_ok & rs_ok,
        "volume_ratio": volume_ratio,
    })


def simulate(sig_map: dict, name_map: dict, regime: pd.Series,
             calendar: pd.DatetimeIndex, rule: str, seed: int = 0,
             park_cash_in_index: pd.Series = None, apply_tax: bool = False) -> dict:
    """
    資金を実際に回しながら日次でシミュレーションする。
    park_cash_in_index に日経平均の終値を渡すと、個別株を買っていない
    余剰資金を日経平均のETFで運用しているものとして日次で増減させる
    （「シグナルが出ていない間は現金」という構造的弱点への対処案の検証用）。
    """
    rng = random.Random(seed)
    capital = float(INITIAL_CAPITAL)
    positions = {}   # code -> dict(entry_price, entry_date, shares)
    trades = []
    equity_curve = []
    deployed_ratios = []

    index_ret = (
        park_cash_in_index.reindex(calendar, method="ffill").pct_change().fillna(0.0)
        if park_cash_in_index is not None else None
    )
    # ETFで運用している資金の取得原価（譲渡益課税の計算に使う）
    etf_basis = float(INITIAL_CAPITAL)
    realized_gain_this_year = 0.0
    total_tax_paid = 0.0
    current_year = calendar[0].year
    stock_cost_one_way = COST_PCT / 2 / 100  # 往復コストの片道分

    for day in calendar:
        # --- 0) 年が変わったら、前年の実現益に課税する ---
        if apply_tax and day.year != current_year:
            if realized_gain_this_year > 0:
                tax = realized_gain_this_year * TAX_RATE
                capital -= tax
                total_tax_paid += tax
                if index_ret is not None:
                    # 納税分はETFを取り崩して払う＝その分だけ原価も減る
                    etf_basis = max(0.0, etf_basis - tax)
            realized_gain_this_year = 0.0
            current_year = day.year

        # 余剰資金をインデックスで運用する場合、日々その分だけ増減させ、
        # 信託報酬を日割りで差し引く
        if index_ret is not None:
            capital *= (1 + float(index_ret.get(day, 0.0)))
            capital *= (1 - ETF_EXPENSE_RATIO / TRADING_DAYS_PER_YEAR)
        # --- 1) 手仕舞い判定（保有60日 or 買値-10%） ---
        for code in list(positions.keys()):
            pos = positions[code]
            df = sig_map[code]
            if day not in df.index:
                continue
            price = float(df.at[day, "close"])
            held_days = (day - pos["entry_date"]).days
            loss_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            if held_days >= HOLDING_DAYS_LIMIT or loss_pct <= -STOP_LOSS_PCT:
                # 個別株の売却コストを引いた手取り
                proceeds = price * pos["shares"] * (1 - stock_cost_one_way)
                realized_gain_this_year += proceeds - pos["cost_basis"]
                # 売却代金をETFに戻す際の買付コスト
                if index_ret is not None:
                    proceeds -= proceeds * ETF_TRADE_COST_PCT / 100
                    etf_basis += proceeds
                capital += proceeds
                trades.append({
                    "code": code, "name": name_map[code],
                    "entry_date": pos["entry_date"].date(), "exit_date": day.date(),
                    "return_pct": loss_pct - COST_PCT,
                    "holding_days": held_days,
                })
                del positions[code]

        # --- 2) 新規エントリー（地合いが良い日のみ、予算の範囲で） ---
        regime_ok = bool(regime.get(day, False))
        if regime_ok:
            candidates = []
            for code, df in sig_map.items():
                if code in positions or day not in df.index:
                    continue
                row = df.loc[day]
                if bool(row["signal"]):
                    candidates.append((code, float(row["close"]),
                                       float(row["volume_ratio"] or 0)))
            if rule == "volume":
                candidates.sort(key=lambda x: -x[2])
            elif rule == "random":
                rng.shuffle(candidates)
            else:  # first = 銘柄コード順
                candidates.sort(key=lambda x: x[0])

            for code, price, _ in candidates:
                # 個別株の購入コスト（買付手数料込み）
                stock_outlay = price * LOT_SIZE * (1 + stock_cost_one_way)
                cost = stock_outlay
                # 原資はETFを売って作るため、その売却コストも要る
                if index_ret is not None:
                    cost += price * LOT_SIZE * ETF_TRADE_COST_PCT / 100
                if cost <= capital:
                    if index_ret is not None and capital > 0:
                        # ETFを取り崩した分だけ含み益が実現する
                        gain_ratio = 1 - (etf_basis / capital) if capital > 0 else 0
                        realized_gain_this_year += cost * gain_ratio
                        etf_basis -= cost * (etf_basis / capital)
                    capital -= cost
                    positions[code] = {
                        "entry_price": price, "entry_date": day, "shares": LOT_SIZE,
                        "cost_basis": stock_outlay,
                    }

        # --- 3) 時価評価 ---
        holdings_value = 0.0
        for code, pos in positions.items():
            df = sig_map[code]
            if day in df.index:
                holdings_value += float(df.at[day, "close"]) * pos["shares"]
            else:
                holdings_value += pos["entry_price"] * pos["shares"]
        total = capital + holdings_value
        equity_curve.append(total)
        deployed_ratios.append(holdings_value / total if total > 0 else 0.0)

    eq = pd.Series(equity_curve, index=calendar)
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() * 100
    tdf = pd.DataFrame(trades)
    return {
        "final": eq.iloc[-1],
        "total_return_pct": (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "max_drawdown_pct": max_dd,
        "n_trades": len(tdf),
        "win_rate": (tdf["return_pct"] > 0).mean() * 100 if len(tdf) else float("nan"),
        "avg_return_pct": tdf["return_pct"].mean() if len(tdf) else float("nan"),
        "max_trade_return": tdf["return_pct"].max() if len(tdf) else float("nan"),
        "avg_deployed_pct": sum(deployed_ratios) / len(deployed_ratios) * 100,
        "total_tax_paid": total_tax_paid,
        "equity": eq,
    }


def main():
    # "--tickers <パス>" は位置引数として数えない（rule に紛れ込ませないため）
    positional = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--tickers":
            skip = True
            continue
        positional.append(a)
    period = positional[0] if positional else "5y"
    rule = positional[1] if len(positional) > 1 else "volume"
    # "parkindex" を付けると、個別株を買っていない余剰資金を日経ETFで運用する
    park_index = "parkindex" in sys.argv[2:]
    # "tax" を付けると譲渡益課税（年ごとの損益通算後に20.315%）を考慮する
    apply_tax = "tax" in sys.argv[2:]

    # "--tickers <パス>" で対象銘柄を差し替えられる（例: universe.csv）
    tickers_path = sys.argv[sys.argv.index("--tickers") + 1] if "--tickers" in sys.argv else None
    tickers = load_tickers(tickers_path)
    print(f"対象銘柄: {len(tickers)}件"
          f"（{Path(tickers_path).name if tickers_path else 'tickers.csv'}）")
    print(f"{len(tickers)}銘柄・過去{period}・選択ルール={rule} でポートフォリオ"
          f"シミュレーションを実行します（初期資金{INITIAL_CAPITAL:,}円・"
          f"{LOT_SIZE}株単位・往復コスト{COST_PCT}%）…")

    print("日経平均のデータを取得中…")
    regime_raw = fetch_market_regime_adx(period)
    nikkei_close = fetch_nikkei_close(period)

    sig_map, name_map = {}, {}
    excluded = []
    print(f"{len(tickers)}銘柄の株価データを用意中…")
    fetched = fetch_histories([t["code"] for t in tickers], period=period)
    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            hist = fetched.get(code)
            if hist is None or len(hist) < 120:
                continue
            # 汚染データの検出（1日で±80%超の値動き＝分割データ不整合の疑い）
            daily = hist["Close"].pct_change().abs()
            if (daily > MAX_PLAUSIBLE_DAILY_MOVE).any():
                worst = daily.idxmax()
                excluded.append(f"{name}({code}) {worst.date()} に{daily.max()*100:.0f}%変動")
                continue
            sig_map[code] = build_signals(hist, nikkei_close)
            name_map[code] = name
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {name} ({code}) 取得失敗: {e}")

    if excluded:
        print(f"\n⚠️  株価データに不自然な急変（1日±{MAX_PLAUSIBLE_DAILY_MOVE*100:.0f}%超）が"
              f"あり、分割データ不整合の疑いがあるため{len(excluded)}銘柄を除外しました：")
        for e in excluded:
            print(f"   {e}")
    print(f"  → {len(sig_map)}銘柄を対象にシミュレーションします")

    # 全銘柄の営業日を統合したカレンダー
    calendar = sorted(set().union(*[df.index for df in sig_map.values()]))
    calendar = pd.DatetimeIndex(calendar)
    regime = regime_raw.reindex(calendar, method="ffill").fillna(False)

    if rule == "random":
        results = [simulate(sig_map, name_map, regime, calendar, rule, seed=s)
                   for s in range(RANDOM_TRIALS)]
        print(f"\n=== 結果（ランダム選択・{RANDOM_TRIALS}回試行の平均）===")
        for key, label in [("total_return_pct", "トータルリターン"),
                           ("max_drawdown_pct", "最大ドローダウン"),
                           ("win_rate", "勝率"), ("n_trades", "トレード数")]:
            vals = [r[key] for r in results]
            print(f"{label}: 平均 {sum(vals)/len(vals):+.1f}"
                  f"（最小 {min(vals):+.1f} / 最大 {max(vals):+.1f}）")
        return

    r = simulate(sig_map, name_map, regime, calendar, rule,
                 park_cash_in_index=nikkei_close if park_index else None,
                 apply_tax=apply_tax)

    # 日経平均との比較は「シミュレーションと同一期間」に揃える。
    # period="max" の ^N225 は1965年まで遡るため、そのまま比較すると
    # 61年分の指数リターンと26年分の戦略リターンを比べることになり無意味
    nk = nikkei_close.loc[(nikkei_close.index >= calendar[0])
                          & (nikkei_close.index <= calendar[-1])]
    nikkei_ret = (nk.iloc[-1] - nk.iloc[0]) / nk.iloc[0] * 100

    mode = (f"・余剰資金は日経ETFで運用（ETF売買{ETF_TRADE_COST_PCT}%片道・"
            f"信託報酬年{ETF_EXPENSE_RATIO*100:.2f}%込み）") if park_index else ""
    if apply_tax:
        mode += f"・譲渡益課税{TAX_RATE*100:.3f}%込み"
    print(f"\n=== ポートフォリオシミュレーション結果（{rule}{mode}）===")
    print(f"対象期間: {calendar[0].date()} 〜 {calendar[-1].date()}")
    print(f"初期資金: {INITIAL_CAPITAL:,}円 → 最終資産: {r['final']:,.0f}円")
    print(f"トータルリターン: {r['total_return_pct']:+.1f}%")
    print(f"最大ドローダウン: {r['max_drawdown_pct']:.1f}%")
    print(f"個別株への平均投入率: {r['avg_deployed_pct']:.1f}%"
          f"（残りは{'日経ETF' if park_index else '現金'}）")
    print(f"トレード数: {r['n_trades']}件 / 勝率: {r['win_rate']:.1f}% "
          f"/ 平均リターン: {r['avg_return_pct']:+.2f}%")
    if apply_tax:
        print(f"支払った税金の累計: {r['total_tax_paid']:,.0f}円")
        # 買い持ちは売るまで課税されないため、最後に一度だけ課税して比較する
        nikkei_after_tax = nikkei_ret * (1 - TAX_RATE)
        print(f"（参考）同期間の日経平均を持ち切った場合: {nikkei_ret:+.1f}%"
              f" → 最後に売却して納税後 {nikkei_after_tax:+.1f}%")
    else:
        print(f"（参考）同期間の日経平均を持ち切った場合: {nikkei_ret:+.1f}%")
    if pd.notna(r["max_trade_return"]) and r["max_trade_return"] > SUSPICIOUS_RETURN_THRESHOLD:
        print(f"\n⚠️  1トレードで{r['max_trade_return']:.0f}%という異常なリターンが"
              "含まれています。データ不整合の可能性があるため結果を鵜呑みにしないでください。")


if __name__ == "__main__":
    main()
