"""
東証上場全銘柄のティッカー一覧を、EDINETコード一覧（Edinetcode.zip）から
作成し、/tmp/all_listed_tickers.json に保存する。

screen_full_market_stage1.py・stage2.py の前段として実行する（全市場版の
かぶ1000流スクリーニング用。tickers.csvの225銘柄とは別の、より広い探索用）。

使い方:
    python3 scripts/build_all_listed_tickers.py
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import requests

EDINET_CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
CODELIST_LOCAL_DIR = Path("/tmp/edinetcodelist")
OUTPUT_PATH = Path("/tmp/all_listed_tickers.json")


def main():
    CODELIST_LOCAL_DIR.mkdir(exist_ok=True)
    codelist_path = CODELIST_LOCAL_DIR / "EdinetcodeDlInfo.csv"
    if not codelist_path.exists():
        print("EDINETコード一覧をダウンロード中…")
        resp = requests.get(EDINET_CODELIST_URL, timeout=30)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        zf.extractall(CODELIST_LOCAL_DIR)

    with codelist_path.open(encoding="cp932") as f:
        lines = f.readlines()
    reader = csv.DictReader(lines[1:])  # 1行目は説明行なのでスキップ
    rows = list(reader)

    tickers = []
    seen = set()
    for r in rows:
        if r["上場区分"] != "上場":
            continue
        sc = r["証券コード"].strip()
        if not sc or len(sc) < 4:
            continue
        code = sc[:4] + ".T"
        if code in seen:
            continue
        seen.add(code)
        tickers.append({"code": code, "name": r["提出者名"]})

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(tickers, f, ensure_ascii=False, indent=1)

    print(f"上場銘柄数: {len(tickers)}")
    print(f"保存先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
