"""
銘柄の上場日（に相当する日付）を集めて data/listing_dates.json に保存する

片山晃『5年で1億貯める株式投資』PART 7 の
  OKポイント① 上場から5年以内
  OKポイント② 上場から10年以内
  NGポイント② 上場5年以内に下方修正2回以上（→ 上場年数の部分のみ）
を判定するために使う（片山晃_ルール.md 参照）。

⚠️ **これは厳密な「上場日」ではない。** yfinance の firstTradeDate は
Yahooが株価を持っている最初の日で、日本株では概ね1999〜2001年が下限。
そのためトヨタ(7203)は1949年上場なのに 1999-05-06 と返る。

  - firstTradeDate が **2002年以降** → 実質的に上場日とみなせる
  - firstTradeDate が **2001年以前** → 「それ以前から上場している」以上の
    ことは分からない（old_listing フラグを立てる）

「上場5年以内／10年以内」の判定には前者しか使わないので実用上は足りるが、
2001年以前の銘柄を「上場から25年」などと解釈してはいけない。

使い方:
    python3 scripts/build_listing_dates.py [銘柄CSV]
      省略時は universe.csv
"""

import csv
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "data" / "listing_dates.json"

WORKERS = 6
RETRIES = 2
RETRY_WAIT_SEC = 5
# yfinanceの日本株データはおおむねこの年より前に遡れない。
# これ以前の日付は「データの開始日」であって上場日ではない
DATA_FLOOR_YEAR = 2002


def fetch_one(code: str):
    for attempt in range(RETRIES + 1):
        try:
            info = yf.Ticker(code).info
            break
        except Exception:
            if attempt == RETRIES:
                return code, None
            time.sleep(RETRY_WAIT_SEC * (attempt + 1))
    ep = info.get("firstTradeDateEpochUtc") or info.get("firstTradeDateMilliseconds")
    if not ep:
        return code, None
    try:
        d = dt.datetime.utcfromtimestamp(ep if ep < 1e11 else ep / 1000).date()
    except (ValueError, OSError, OverflowError):
        return code, None
    return code, {
        "first_trade_date": d.isoformat(),
        # True なら「実際の上場日はもっと前」という意味で、年数を計算してはいけない
        "old_listing": d.year < DATA_FLOOR_YEAR,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else BASE_DIR / "universe.csv"
    codes = [t["code"] for t in csv.DictReader(open(path, encoding="utf-8-sig"))]
    print(f"{len(codes)}銘柄（{path.name}）の上場日を取得します…")

    result, failed = {}, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (code, rec) in enumerate(pool.map(fetch_one, codes), 1):
            if rec:
                result[code] = rec
            else:
                failed += 1
            if i % 100 == 0 or i == len(codes):
                print(f"  [{i}/{len(codes)}] 取得{len(result)}件 / 失敗{failed}件")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "note": "yfinanceのfirstTradeDate。old_listing=True の銘柄は"
                "実際の上場日がもっと前なので上場年数を計算しないこと",
        "data_floor_year": DATA_FLOOR_YEAR,
        "data": result,
    }, ensure_ascii=False), encoding="utf-8")

    today = dt.date.today()
    recent5 = recent10 = old = 0
    for r in result.values():
        if r["old_listing"]:
            old += 1
            continue
        d = dt.date.fromisoformat(r["first_trade_date"])
        years = (today - d).days / 365.25
        if years <= 5:
            recent5 += 1
        if years <= 10:
            recent10 += 1

    print(f"\n保存: {OUT_PATH}")
    print(f"  取得できた銘柄       : {len(result)} / {len(codes)}")
    print(f"  上場5年以内          : {recent5}銘柄")
    print(f"  上場10年以内         : {recent10}銘柄")
    print(f"  2001年以前から上場   : {old}銘柄（年数の計算はできない）")


if __name__ == "__main__":
    main()
