"""
tickers.csv の銘柄名を日本語に統一するための対応表を作る。

yfinanceは日本株でも英語名しか返さないため（例: "Subaru Corporation"）、
通知やCSVの銘柄名が英語と日本語で混在してしまう。EDINETのコード一覧には
日本語の社名（提出者名）が入っているので、そこから証券コード→日本語名の
対応表を作り data/japanese_names.json に保存する。

使い方:
    python3 scripts/build_japanese_names.py
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
OUTPUT_PATH = BASE_DIR / "data" / "japanese_names.json"

EDINET_CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
CODELIST_LOCAL_DIR = Path("/tmp/edinetcodelist")


def load_edinet_names() -> dict:
    """EDINETコード一覧から「証券コード(4桁) -> 日本語社名」の辞書を作る。"""
    CODELIST_LOCAL_DIR.mkdir(exist_ok=True)
    codelist_path = CODELIST_LOCAL_DIR / "EdinetcodeDlInfo.csv"
    if not codelist_path.exists():
        print("EDINETコード一覧をダウンロード中…")
        resp = requests.get(EDINET_CODELIST_URL, timeout=30)
        resp.raise_for_status()
        zipfile.ZipFile(io.BytesIO(resp.content)).extractall(CODELIST_LOCAL_DIR)

    names = {}
    with codelist_path.open(encoding="cp932") as f:
        # 1行目はヘッダーではない説明行なので読み飛ばす
        f.readline()
        for r in csv.DictReader(f):
            sc = (r.get("証券コード") or "").strip()
            name = (r.get("提出者名") or "").strip()
            if len(sc) == 5 and sc.endswith("0"):
                sc = sc[:4]  # EDINETは末尾0付きの5桁で持っている
            if sc and name:
                names[sc] = name
    return names


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    edinet = load_edinet_names()
    print(f"EDINETから{len(edinet)}社の日本語名を読み込みました")

    with TICKERS_CSV.open(encoding="utf-8-sig") as f:
        tickers = list(csv.DictReader(f))

    mapping, missing = {}, []
    for t in tickers:
        code = t["code"]                    # 例 "7270.T"
        num = code.replace(".T", "")
        ja = edinet.get(num)
        if ja:
            mapping[code] = ja
        else:
            missing.append(f"{t['name']}({code})")

    OUTPUT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"{len(mapping)}/{len(tickers)}銘柄の日本語名を {OUTPUT_PATH} に保存しました")
    if missing:
        print(f"\n日本語名が見つからなかった{len(missing)}銘柄（英語名のまま使われます）:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
