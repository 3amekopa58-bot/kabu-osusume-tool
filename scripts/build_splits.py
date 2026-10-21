"""全銘柄の株式分割履歴を一括取得して data/splits.json に保存する。

⚠️ **なぜ必要か。** yfinance の株価は常に分割調整済みで、EDINET の
   BPS・EPS・株式数はその決算期の名目値。両者を素で割ると
   **分割した銘柄が分割倍率のぶん割安に見える**。
   2026-10-21、10倍株スクリーナーで134銘柄中22銘柄がこれに当たり、
   さらに基準値を測った analyze_tenbagger_profile.py も同じ誤りだった（4.4-80）。

   対策は**1株当たりの値を使わず、合計値で比べる**こと：
     時価総額 = 株価 × 現在株式数（分割が打ち消し合う）
     PBR = 時価総額 ÷ 純資産
     PER = 時価総額 ÷ 純利益
   現在株式数 = EDINETの株式数 × その決算期より後の分割倍率。
   この分割倍率をここで作る。

⚠️ 1件ずつ Ticker.splits を叩くと1,557銘柄で2時間かかる。
   yf.download(actions=True) は一括で分割を返すのでそれを使う。

使い方: ./venv/bin/python scripts/build_splits.py [銘柄CSV]
"""
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent.parent
OUT = B / "data" / "splits.json"
BATCH, WAIT = 100, 1.0


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "data" / "all_listed_tickers.json"
    if path.suffix == ".json":
        codes = [t["code"] for t in json.loads(path.read_text(encoding="utf-8"))]
    else:
        codes = [r["code"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))]
    print(f"{len(codes)}銘柄の分割履歴を取得します（{path.name}）")

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8")).get("data", {})
        print(f"  既存 {len(out)}銘柄を引き継ぎます")

    for i in range(0, len(codes), BATCH):
        b = [c for c in codes[i:i + BATCH]]
        try:
            d = yf.download(b, period="max", group_by="ticker", actions=True,
                            auto_adjust=False, progress=False, threads=True)
        except Exception:
            continue
        time.sleep(WAIT)
        for c in b:
            try:
                s = d[c]["Stock Splits"].dropna()
                s = s[s > 0]
                out[c] = {str(k.date()): float(v) for k, v in s.items()}
            except Exception:
                pass
        if (i // BATCH) % 5 == 0:
            print(f"  {min(i + BATCH, len(codes))}/{len(codes)} → {len(out)}銘柄", flush=True)
            OUT.write_text(json.dumps({"note": "yfinanceの分割履歴。日付→倍率",
                                       "data": out}, ensure_ascii=False), encoding="utf-8")

    OUT.write_text(json.dumps({"note": "yfinanceの分割履歴。日付→倍率",
                               "data": out}, ensure_ascii=False), encoding="utf-8")
    n = sum(1 for v in out.values() if v)
    print(f"\n完了: {OUT}")
    print(f"  {len(out)}銘柄 / そのうち分割履歴あり {n}銘柄")


if __name__ == "__main__":
    main()
