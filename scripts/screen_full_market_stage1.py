"""
かぶ1000流スクリーニングの全市場版・ステージ1（yfinanceのみ、EDINET不使用）。

東証上場全銘柄（約3,822社、EDINETコード一覧から抽出）についてPER・PBR・
グレアム指数・配当利回りを取得し、かぶ1000流の3段階基準で分類する。
225銘柄版（find_kabu1000_candidates.py）と違い、実質PBR・ネットネット指数は
まだ計算しない（EDINETへの負荷を抑えるため、この段階では見送る）。

このステージで1つ以上の基準に該当した銘柄だけを
/tmp/kabu1000_full_market_stage1_candidates.json に出力し、
ステージ2（scripts/screen_full_market_stage2.py）でEDINETの
実質PBR・ネットネット指数を深掘りする対象として使う。

件数が多いため、100銘柄ごとに /tmp/kabu1000_full_market_stage1_progress.json
へ途中経過を保存する（中断しても再開しやすくするため）。

使い方:
    python3 scripts/screen_full_market_stage1.py
"""

import json
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).parent.parent
TICKERS_PATH = Path("/tmp/all_listed_tickers.json")
PROGRESS_PATH = Path("/tmp/kabu1000_full_market_stage1_progress.json")
CANDIDATES_PATH = Path("/tmp/kabu1000_full_market_stage1_candidates.json")

PER_TIERS = [(6, "激安"), (8, "超割安"), (10, "割安")]
PBR_TIERS = [(0.3, "激安"), (0.4, "超割安"), (0.5, "割安")]
GRAHAM_TIERS = [(5.0, "激安"), (8.0, "超割安"), (10.0, "割安")]
DIVIDEND_TIERS = [(5.0, "激安"), (4.0, "超割安"), (3.0, "割安")]


def classify_low_is_good(value, tiers):
    if value is None:
        return None
    for threshold, label in tiers:
        if value < threshold:
            return label
    return None


def classify_high_is_good(value, tiers):
    if value is None:
        return None
    for threshold, label in tiers:
        if value >= threshold:
            return label
    return None


def main():
    with TICKERS_PATH.open(encoding="utf-8") as f:
        tickers = json.load(f)
    print(f"{len(tickers)}銘柄を取得します（ステージ1：yfinanceのみ）…")

    all_rows = []
    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            info = yf.Ticker(code).info
            per = info.get("forwardPE") or info.get("trailingPE")
            if per is not None and per <= 0:
                per = None
            pbr = info.get("priceToBook")
            dividend_yield = info.get("dividendYield")
            price = info.get("currentPrice")
            market_cap = info.get("marketCap")

            graham = (per * pbr) if (per and pbr) else None
            dividend_pct = dividend_yield if dividend_yield else None

            row = {
                "code": code, "name": name, "price": price, "market_cap": market_cap,
                "per": per, "pbr": pbr, "graham": graham, "dividend_pct": dividend_pct,
                "per_tier": classify_low_is_good(per, PER_TIERS),
                "pbr_tier": classify_low_is_good(pbr, PBR_TIERS),
                "graham_tier": classify_low_is_good(graham, GRAHAM_TIERS),
                "dividend_tier": classify_high_is_good(dividend_pct, DIVIDEND_TIERS),
            }
            all_rows.append(row)
        except Exception as e:
            all_rows.append({"code": code, "name": name, "error": str(e)})

        if i % 100 == 0:
            print(f"  進捗: {i}/{len(tickers)}")
            with PROGRESS_PATH.open("w", encoding="utf-8") as f:
                json.dump(all_rows, f, ensure_ascii=False)

    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False)

    candidates = [
        r for r in all_rows
        if any(r.get(k) for k in ("per_tier", "pbr_tier", "graham_tier", "dividend_tier"))
    ]
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)

    print(f"\n取得完了: {len(all_rows)}銘柄")
    print(f"1つ以上の基準に該当: {len(candidates)}銘柄")
    print(f"候補リスト保存先: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
