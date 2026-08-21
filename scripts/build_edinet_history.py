"""
EDINETから「その他有価証券評価差額金」の複数年分の時系列データを取得し、
data/edinet_valuation_diff_history.json に保存する（バックテスト用）。

各年の有価証券報告書には「当期」「前期」の2時点分の値が入っているため、
1年おきに5回（当期＝2026,2024,2022,2020,2018年提出分）取得すれば、
実質10年弱（2017〜2026年3月期）の時系列を効率よく集められる。

data/edinet_valuation_diff.json（build_edinet_cache.pyが作る単年キャッシュ）
とは別物。screen.pyはこのhistoryファイルを読まない（バックテスト専用）。

実行方法:
    python3 scripts/build_edinet_history.py
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
OUTPUT_PATH = ROOT / "data" / "edinet_valuation_diff_history.json"

EDINET_CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
EDINET_DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
EDINET_DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"

VALUATION_DIFF_TAG = "jppfs_cor:ValuationDifferenceOnAvailableForSaleSecurities"

# 1年おきに5回。それぞれ「当期・前期」の2時点が取れるので実質10年弱をカバー
TARGET_FISCAL_YEARS = [2026, 2024, 2022, 2020, 2018]


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".envにEDINET_API_KEYが見つかりません")


def load_ticker_to_edinet_code() -> dict:
    resp = requests.get(EDINET_CODELIST_URL, timeout=30)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    with zf.open("EdinetcodeDlInfo.csv") as f:
        text = f.read().decode("cp932")
    lines = text.splitlines()
    reader = csv.DictReader(lines[1:])
    code_map = {}
    for row in reader:
        sc = row["証券コード"].strip()
        if sc:
            code_map[sc[:4]] = row["ＥＤＩＮＥＴコード"]

    with TICKERS_PATH.open(encoding="utf-8-sig") as f:
        tickers = [row["code"] for row in csv.DictReader(f)]

    return {t: code_map.get(t.split(".")[0]) for t in tickers}


def find_doc_ids_for_year(edinet_codes: set, fiscal_year: int, api_key: str) -> dict:
    """指定した会計年度（3月期決算、当年5〜7月提出分）の有報docIDを検索する"""
    start = dt.date(fiscal_year, 5, 1)
    end = dt.date(fiscal_year, 7, 31)
    found = {}
    d = start
    while d <= end:
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
    fiscal_year_end = None
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
        elif row[0] == "jpdei_cor:CurrentFiscalYearEndDateDEI":
            fiscal_year_end = row[8]

    if current is None:
        return None

    def to_num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    current_v = to_num(current)
    prior_v = to_num(prior)

    return {
        "filer_name": filer_name,
        "fiscal_year_end": fiscal_year_end,
        "current_yen": current_v,
        "prior_yen": prior_v,
    }


def main():
    api_key = load_api_key()
    print("EDINETコード一覧を取得・225銘柄とマッピング中…")
    ticker_map = load_ticker_to_edinet_code()
    matched = {t: ec for t, ec in ticker_map.items() if ec}
    print(f"  {len(matched)}/{len(ticker_map)} 銘柄でEDINETコードが一致")
    edinet_codes = set(matched.values())
    ec_to_ticker = {ec: t for t, ec in matched.items()}

    # ticker -> [{fiscal_year, current_yen, prior_yen, filer_name}, ...]
    history: dict = {t: [] for t in matched}

    for fy in TARGET_FISCAL_YEARS:
        print(f"\n=== {fy}年度分の書類を検索中（{fy}-05-01〜{fy}-07-31）===")
        doc_ids = find_doc_ids_for_year(edinet_codes, fy, api_key)
        print(f"  {len(doc_ids)}/{len(edinet_codes)} 銘柄で書類が見つかりました")

        found_count = 0
        for i, (ec, doc_id) in enumerate(doc_ids.items(), 1):
            ticker = ec_to_ticker[ec]
            info = extract_valuation_diff(doc_id, api_key)
            if info:
                info["fiscal_year"] = fy
                history[ticker].append(info)
                found_count += 1
            if i % 30 == 0:
                print(f"    進捗: {i}/{len(doc_ids)}")
            time.sleep(0.25)
        print(f"  抽出成功: {found_count}/{len(doc_ids)}")

        # 年ごとに途中保存（長時間かかるため、途中で止まってもここまでのデータは残る）
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": dt.date.today().isoformat(),
                    "target_fiscal_years": TARGET_FISCAL_YEARS,
                    "note": "その他有価証券評価差額金の複数年分（個別財務諸表ベース）。バックテスト用。",
                    "data": {t: v for t, v in history.items() if v},
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print(f"  中間保存: {OUTPUT_PATH}")

    total_points = sum(len(v) for v in history.values())
    covered_tickers = sum(1 for v in history.values() if v)
    print(f"\n完了。データ点数合計: {total_points}件、収録銘柄数: {covered_tickers}/{len(matched)}")
    print(f"保存先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
