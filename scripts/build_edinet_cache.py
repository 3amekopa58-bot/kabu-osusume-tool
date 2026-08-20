"""
EDINETから「その他有価証券評価差額金」（かぶ1000氏が重視する、保有有価証券の
含み益の増減）を取得し、data/edinet_valuation_diff.json に保存する。

有価証券報告書の「個別（親会社単体）」財務諸表から取得する。連結決算が
IFRSの企業でも個別は日本基準で開示するのが通例のため、連結会計基準に
かかわらず幅広い企業をカバーできる（225銘柄中171銘柄で取得成功、2026年
8月時点）。個別財務諸表でもこの項目自体を開示していない企業のみ対象外。

有価証券報告書は年1回しか更新されないため、このキャッシュも都度手動で
（例：毎年6〜7月頃に）再実行すればよい。screen.py は毎回このスクリプトを
実行せず、生成済みのJSONを読むだけ。

実行方法:
    python3 scripts/build_edinet_cache.py
"""

import csv
import datetime as dt
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"
TICKERS_PATH = ROOT / "tickers.csv"
OUTPUT_PATH = ROOT / "data" / "edinet_valuation_diff.json"

EDINET_CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
EDINET_DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
EDINET_DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"

VALUATION_DIFF_TAG = "jppfs_cor:ValuationDifferenceOnAvailableForSaleSecurities"

# 検索対象期間：3月決算企業の有報提出が集中する時期を広めにカバー
SEARCH_START = dt.date.today().replace(month=5, day=1)
SEARCH_END = dt.date.today().replace(month=7, day=31)


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".envにEDINET_API_KEYが見つかりません")


def load_ticker_to_edinet_code() -> dict[str, str]:
    resp = requests.get(EDINET_CODELIST_URL, timeout=30)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    with zf.open("EdinetcodeDlInfo.csv") as f:
        text = f.read().decode("cp932")
    lines = text.splitlines()
    reader = csv.DictReader(lines[1:])  # 1行目は説明行なのでスキップ
    code_map = {}
    for row in reader:
        sc = row["証券コード"].strip()
        if sc:
            code_map[sc[:4]] = row["ＥＤＩＮＥＴコード"]

    with TICKERS_PATH.open(encoding="utf-8-sig") as f:
        tickers = [row["code"] for row in csv.DictReader(f)]

    return {t: code_map.get(t.split(".")[0]) for t in tickers}


def find_doc_ids(edinet_codes: set[str], api_key: str) -> dict[str, str]:
    found = {}
    d = SEARCH_START
    while d <= SEARCH_END:
        resp = requests.get(
            EDINET_DOCUMENTS_URL,
            params={"date": d.isoformat(), "type": 2, "Subscription-Key": api_key},
            timeout=20,
        )
        if resp.status_code == 200:
            for doc in resp.json().get("results", []):
                ec = doc.get("edinetCode")
                if ec in edinet_codes and doc.get("docTypeCode") == "120":
                    found[ec] = doc.get("docID")
        d += dt.timedelta(days=1)
        time.sleep(0.1)
    return found


def extract_valuation_diff(doc_id: str, api_key: str) -> Optional[dict]:
    resp = requests.get(
        EDINET_DOCUMENT_URL.format(doc_id=doc_id),
        params={"type": 5, "Subscription-Key": api_key},
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    main_files = [n for n in zf.namelist() if "jpcrp030000-asr-001" in n]
    if not main_files:
        return None
    with zf.open(main_files[0]) as f:
        content = f.read().decode("utf-16")
    reader = csv.reader(content.splitlines(), delimiter="\t")
    rows = list(reader)

    prior = current = None
    filer_name = None
    accounting_standard = None
    for row in rows[1:]:
        if len(row) < 9:
            continue
        if row[0] == VALUATION_DIFF_TAG:
            if row[2] == "Prior1YearInstant_NonConsolidatedMember":
                prior = row[8]
            elif row[2] == "CurrentYearInstant_NonConsolidatedMember":
                current = row[8]
        elif row[0] == "jpdei_cor:FilerNameInJapaneseDEI":
            filer_name = row[8]
        elif row[0] == "jpdei_cor:AccountingStandardsDEI":
            accounting_standard = row[8]

    if current is None:
        return None  # IFRS・米国基準など、このタグが無い

    def to_num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    current_v = to_num(current)
    prior_v = to_num(prior)
    change = (current_v - prior_v) if (current_v is not None and prior_v is not None) else None

    return {
        "filer_name": filer_name,
        "accounting_standard": accounting_standard,
        "valuation_diff_current_yen": current_v,
        "valuation_diff_prior_yen": prior_v,
        "valuation_diff_change_yen": change,
    }


def main():
    api_key = load_api_key()
    print("EDINETコード一覧を取得・225銘柄とマッピング中…")
    ticker_map = load_ticker_to_edinet_code()
    matched = {t: ec for t, ec in ticker_map.items() if ec}
    print(f"  {len(matched)}/{len(ticker_map)} 銘柄でEDINETコードが一致")

    print(f"有価証券報告書の書類IDを検索中（{SEARCH_START}〜{SEARCH_END}）…")
    doc_ids = find_doc_ids(set(matched.values()), api_key)
    print(f"  {len(doc_ids)}/{len(matched)} 銘柄で書類が見つかりました")

    result = {}
    skipped_ifrs = 0
    for i, (ticker, ec) in enumerate(matched.items(), 1):
        doc_id = doc_ids.get(ec)
        if not doc_id:
            continue
        info = extract_valuation_diff(doc_id, api_key)
        if info:
            result[ticker] = info
        else:
            skipped_ifrs += 1
        if i % 30 == 0:
            print(f"  進捗: {i}/{len(matched)}")
        time.sleep(0.25)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": dt.date.today().isoformat(),
                "note": "その他有価証券評価差額金（日本基準開示企業のみ・IFRS企業は対象外）。かぶ1000氏の考え方に基づく参考指標。",
                "data": result,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )

    print(f"\n取得できた銘柄: {len(result)}/{len(matched)}（IFRS等で取得不可: {skipped_ifrs}）")
    print(f"保存先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
