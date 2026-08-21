"""
かぶ1000氏の参考指標（その他有価証券評価差額金の前期比増減）が、
翌年の株価リターンと関係があるかを検証するバックテスト。

data/edinet_valuation_diff_history.json（過去5年度分、build_edinet_history.py
で作成済み）を使い、各銘柄・各年度について
  「評価差額金が増えた/減った」→「その後1年間の株価リターン」
を集計する。

有価証券報告書の提出は5〜7月に集中するため、開示が確実に行き渡っている
タイミングとして8月1日を「シグナル日」とみなす（実際の提出日は分からない
ため、保守的に月末近くを取る＝ルックアヘッドバイアスを避ける）。
2026年度分は1年後のリターンがまだ観測できない（未来のデータが無い）ため、
本バックテストの対象から除外する。

使い方:
    python3 backtest_edinet.py
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "data" / "edinet_valuation_diff_history.json"

FORWARD_DAYS = 365
SIGNAL_MONTH_DAY = (8, 1)  # 8月1日を開示行き渡り後の基準日とする


def load_history() -> dict:
    with HISTORY_PATH.open(encoding="utf-8") as f:
        return json.load(f)["data"]


def nearest_price_on_or_after(hist: pd.DataFrame, target: pd.Timestamp):
    if hist.index.tz is not None and target.tz is None:
        target = target.tz_localize(hist.index.tz)
    idx = hist.index[hist.index >= target]
    if len(idx) == 0:
        return None, None
    d = idx[0]
    return d, float(hist.loc[d, "Close"])


def main():
    history = load_history()
    today = pd.Timestamp(dt.date.today())

    # バックテスト用のレコードを組み立てる（2026年度分は将来データが無いため除外）
    records = []
    for ticker, entries in history.items():
        for e in entries:
            fy = e["fiscal_year"]
            if fy >= 2026:
                continue
            current_v, prior_v = e.get("current_yen"), e.get("prior_yen")
            if current_v is None or prior_v is None:
                continue
            change = current_v - prior_v
            if change == 0:
                continue  # 変化なしは増減の判定材料にならないため除外
            signal_date = pd.Timestamp(dt.date(fy, *SIGNAL_MONTH_DAY))
            forward_date = signal_date + pd.Timedelta(days=FORWARD_DAYS)
            if forward_date > today:
                continue
            records.append({
                "ticker": ticker,
                "filer_name": e.get("filer_name"),
                "fiscal_year": fy,
                "prior_yen": prior_v,
                "change_yen": change,
                "direction": "増加" if change > 0 else "減少",
                "signal_date": signal_date,
                "forward_date": forward_date,
            })

    print(f"対象レコード数: {len(records)}件（銘柄×年度）")

    tickers = sorted({r["ticker"] for r in records})
    print(f"価格データを取得します（{len(tickers)}銘柄）…")

    price_cache = {}
    for i, t in enumerate(tickers, 1):
        try:
            hist = yf.Ticker(t).history(start="2017-06-01")
            price_cache[t] = hist
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t} 取得失敗: {e}")
        if i % 30 == 0:
            print(f"  進捗: {i}/{len(tickers)}")

    results = []
    for r in records:
        hist = price_cache.get(r["ticker"])
        if hist is None or hist.empty:
            continue
        entry_date, entry_price = nearest_price_on_or_after(hist, r["signal_date"])
        exit_date, exit_price = nearest_price_on_or_after(hist, r["forward_date"])
        if entry_price is None or exit_price is None:
            continue
        ret_pct = (exit_price - entry_price) / entry_price * 100
        results.append({**r, "entry_price": entry_price, "exit_price": exit_price, "return_pct": ret_pct})

    print(f"\nリターン計算できたレコード数: {len(results)}件\n")

    df = pd.DataFrame(results)
    if df.empty:
        print("データが無いため集計できません")
        return

    def summarize(sub: pd.DataFrame, label: str):
        n = len(sub)
        if n == 0:
            print(f"  {label}: データなし")
            return
        win_rate = (sub["return_pct"] > 0).mean() * 100
        avg = sub["return_pct"].mean()
        median = sub["return_pct"].median()
        print(f"  {label}: 件数={n}, 勝率={win_rate:.1f}%, 平均リターン={avg:+.2f}%, 中央値={median:+.2f}%")

    print("=== 全体（ベンチマーク：増減問わず全レコード） ===")
    summarize(df, "全体")

    print("\n=== 評価差額金の増減別 ===")
    summarize(df[df["direction"] == "増加"], "評価差額金 増加")
    summarize(df[df["direction"] == "減少"], "評価差額金 減少")

    print("\n=== 年度別（評価差額金 増加 vs 減少） ===")
    for fy in sorted(df["fiscal_year"].unique()):
        sub_fy = df[df["fiscal_year"] == fy]
        print(f" {fy}年度:")
        summarize(sub_fy[sub_fy["direction"] == "増加"], "  増加")
        summarize(sub_fy[sub_fy["direction"] == "減少"], "  減少")

    # 変化幅の大きさとリターンの関係も確認（変化率ベース。前期の絶対値が
    # 小さすぎる銘柄は率が暴れるため、前期1億円未満は対象外）
    sizable = df[df["prior_yen"].abs() >= 1e8].copy()
    sizable["change_pct"] = sizable["change_yen"] / sizable["prior_yen"].abs() * 100
    top_q = sizable["change_pct"].quantile(0.75)
    bottom_q = sizable["change_pct"].quantile(0.25)
    print(f"\n=== 変化率の上位25%・下位25%（前期1億円以上の銘柄のみ、対象{len(sizable)}件） ===")
    summarize(sizable[sizable["change_pct"] >= top_q], f"上位25%（変化率{top_q:+.1f}%以上）")
    summarize(sizable[sizable["change_pct"] <= bottom_q], f"下位25%（変化率{bottom_q:+.1f}%以下）")

    out_path = BASE_DIR / "output" / "edinet_backtest_detail.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n詳細データ: {out_path}")


if __name__ == "__main__":
    main()
