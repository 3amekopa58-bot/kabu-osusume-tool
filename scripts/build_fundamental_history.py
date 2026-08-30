"""
過去の決算データ（EPS・BPS）を集めて data/fundamental_history.json に保存する

screen.py は総合スコアの50%を割安度（PER・PBR・配当利回り等）に配分している
のに、backtest.py はファンダメンタルを一切見ていない。つまり推奨順位の半分は
一度も検証されていない。それを検証できるようにするためのデータを用意する。

⚠️ 先読みバイアスへの対処が要点：
  - yfinance の info が返す PER/PBR は「今日の値」なので、過去のトレードに
    当てはめると完全な先読みになる。必ず「当時のEPS/BPS」から計算し直す
  - 決算期末（例:2026-03-31）の数字は、その日には公表されていない。
    日本の会社は期末から3か月以内に有価証券報告書を出すので、
    backtest_edinet.py と同じ考え方で **期末+3か月** を情報が使えるように
    なった日（available_from）として記録する

yfinance の財務データは5期分しか遡れないため、検証できるのは実質4年ぶん。
より長い期間が必要なら EDINET から取る必要がある（scripts/build_edinet_history.py 参照）。

使い方:
    python3 scripts/build_fundamental_history.py [銘柄CSV] [--limit N]
      銘柄CSV 省略時は universe.csv
"""

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "data" / "fundamental_history.json"

# info は1銘柄1リクエスト。944銘柄を一気に投げるとYahooに弾かれるので
# 控えめな並列数にし、失敗したら間を置いて数回だけ試し直す（screen.py と同じ方針）
WORKERS = 4
RETRIES = 2
RETRY_WAIT_SEC = 5
# 決算期末からこの日数が経つまでは、その決算の数字を「使えない」扱いにする
# （有価証券報告書の提出期限が期末から3か月以内のため）
DISCLOSURE_LAG_DAYS = 92


def _pick(df: pd.DataFrame, *names):
    """財務諸表の項目名は銘柄・年度でぶれるので、候補を順に探す"""
    for n in names:
        if df is not None and n in df.index:
            return df.loc[n]
    return None


def fetch_one(code: str) -> list:
    """1銘柄ぶんの決算期ごとのEPS・BPSを返す。取れなければ空リスト。"""
    for attempt in range(RETRIES + 1):
        try:
            t = yf.Ticker(code)
            fin, bs = t.financials, t.balance_sheet
            break
        except Exception:
            if attempt == RETRIES:
                return []
            time.sleep(RETRY_WAIT_SEC * (attempt + 1))
    if fin is None or fin.empty or bs is None or bs.empty:
        return []

    net_income = _pick(fin, "Net Income", "Net Income Common Stockholders",
                       "Net Income From Continuing Operation Net Minority Interest")
    revenue = _pick(fin, "Total Revenue", "Operating Revenue")
    op_income = _pick(fin, "Operating Income", "Total Operating Income As Reported")
    shares = _pick(bs, "Ordinary Shares Number", "Share Issued")
    equity = _pick(bs, "Stockholders Equity", "Common Stock Equity")

    rows = []
    for col in fin.columns:
        try:
            n_sh = float(shares[col])
            if not n_sh or n_sh != n_sh:
                continue
            period_end = pd.Timestamp(col)
            rec = {
                "period_end": period_end.strftime("%Y-%m-%d"),
                # この日以降なら、その決算の数字を知っていたはず（先読み防止）
                "available_from": (period_end + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
                                   ).strftime("%Y-%m-%d"),
                "shares": n_sh,
            }
            for key, series in (("net_income", net_income), ("revenue", revenue),
                                ("operating_income", op_income), ("equity", equity)):
                v = float(series[col]) if series is not None and col in series else float("nan")
                rec[key] = None if v != v else v
            if rec["net_income"] is not None:
                rec["eps"] = rec["net_income"] / n_sh
            if rec["equity"] is not None:
                rec["bps"] = rec["equity"] / n_sh
            if "eps" in rec or "bps" in rec:
                rows.append(rec)
        except Exception:
            continue
    return sorted(rows, key=lambda r: r["period_end"])


def main():
    # "--limit N" の N を銘柄ファイル名と取り違えないよう、値ごと取り除く
    args, limit, skip = [], None, False
    for a in sys.argv[1:]:
        if skip:
            limit, skip = int(a), False
            continue
        if a == "--limit":
            skip = True
            continue
        if not a.startswith("--"):
            args.append(a)
    path = Path(args[0]) if args else BASE_DIR / "universe.csv"

    tickers = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if limit:
        tickers = tickers[:limit]
    codes = [t["code"] for t in tickers]
    print(f"{len(codes)}銘柄の決算データを取得します（{path.name}）…")

    result, failed = {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (code, rows) in enumerate(zip(codes, pool.map(fetch_one, codes)), 1):
            if rows:
                result[code] = rows
            else:
                failed.append(code)
            if i % 50 == 0 or i == len(codes):
                print(f"  [{i}/{len(codes)}] 取得{len(result)}銘柄 / 失敗{len(failed)}銘柄")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "note": "決算期ごとのEPS/BPS等。available_from 以降でのみ使うこと（先読み防止）",
        "disclosure_lag_days": DISCLOSURE_LAG_DAYS,
        "data": result,
    }, ensure_ascii=False), encoding="utf-8")

    periods = [len(v) for v in result.values()]
    print(f"\n保存: {OUT_PATH}")
    print(f"  取得できた銘柄: {len(result)} / {len(codes)}")
    if periods:
        print(f"  1銘柄あたりの決算期数: 中央値{sorted(periods)[len(periods)//2]}期")
        alld = sorted(r["period_end"] for v in result.values() for r in v)
        print(f"  期間: {alld[0]} 〜 {alld[-1]}")
    if failed:
        print(f"  失敗した銘柄の例: {failed[:5]}")


if __name__ == "__main__":
    main()
