"""
かぶ1000流スクリーニングの全市場版・ステージ2（EDINET深掘り）。

ステージ1（screen_full_market_stage1.py）でPER・PBR・グレアム指数・配当
利回りのうち3つ以上が「割安」以上に該当した銘柄（/tmp/kabu1000_stage2_targets.json、
500銘柄）について、EDINETから実質PBR・ネットネット指数を計算するための
データを取得する。抽出ロジックは build_edinet_cache.py のものを再利用する。

結果は /tmp/kabu1000_stage2_result.json に保存する。

使い方:
    python3 scripts/screen_full_market_stage2.py
"""

import json
import sys
import time
from pathlib import Path

import requests
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import build_edinet_cache as edinet  # noqa: E402

TARGETS_PATH = Path("/tmp/kabu1000_stage2_targets.json")
CODELIST_PATH = Path("/tmp/edinetcodelist/EdinetcodeDlInfo.csv")
RESULT_PATH = Path("/tmp/kabu1000_stage2_result.json")

PER_TIERS = [(6, "激安"), (8, "超割安"), (10, "割安")]
PBR_TIERS = [(0.3, "激安"), (0.4, "超割安"), (0.5, "割安")]
NET_NET_TIERS = [(0.5, "激安"), (0.66, "超割安"), (1.0, "割安")]
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


def load_edinet_code_map(codes_needed: set) -> dict:
    import csv
    with CODELIST_PATH.open(encoding="cp932") as f:
        lines = f.readlines()
    reader = csv.DictReader(lines[1:])
    code_map = {}
    for row in reader:
        sc = row["証券コード"].strip()
        if sc and (sc[:4] + ".T") in codes_needed:
            code_map[sc[:4] + ".T"] = row["ＥＤＩＮＥＴコード"]
    return code_map


def main():
    with TARGETS_PATH.open(encoding="utf-8") as f:
        targets = json.load(f)
    ticker_codes = {t["code"] for t in targets}
    print(f"ステージ2対象: {len(targets)}銘柄")

    api_key = edinet.load_api_key()
    ticker_to_ec = load_edinet_code_map(ticker_codes)
    print(f"  EDINETコード一致: {len(ticker_to_ec)}/{len(targets)}")

    ec_set = set(ticker_to_ec.values())
    print(f"有価証券報告書の書類IDを検索中（{edinet.SEARCH_START}〜{edinet.SEARCH_END}）…")
    doc_ids = edinet.find_doc_ids(ec_set, api_key)
    print(f"  {len(doc_ids)}/{len(ec_set)} 銘柄で書類が見つかりました")

    ec_to_ticker = {ec: t for t, ec in ticker_to_ec.items()}
    results = {}
    for i, (ec, doc_id) in enumerate(doc_ids.items(), 1):
        ticker = ec_to_ticker[ec]
        info = edinet.extract_valuation_diff(doc_id, api_key)
        if info:
            results[ticker] = info
        if i % 50 == 0:
            print(f"  進捗: {i}/{len(doc_ids)}")
            with RESULT_PATH.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        time.sleep(0.25)

    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"EDINETデータ取得完了: {len(results)}/{len(targets)}")

    # yfinanceからbookValue・sharesOutstandingを追加取得（実質PBR計算用）
    stage1_by_code = {t["code"]: t for t in targets}
    final_rows = []
    print("実質PBR・ネットネット指数を計算中…")
    for i, t in enumerate(targets, 1):
        code = t["code"]
        try:
            info = yf.Ticker(code).info
            book_value = info.get("bookValue")
            shares_out = info.get("sharesOutstanding")
            price = t.get("price")
            market_cap = t.get("market_cap")
            pbr = t.get("pbr")

            edinet_info = results.get(code)
            real_pbr = pbr
            real_pbr_note = "PBRで代用（評価差額金データなし）"
            net_net = None
            if edinet_info and book_value and shares_out and price:
                diff_yen = edinet_info.get("valuation_diff_current_yen")
                if diff_yen is not None:
                    adjusted_bps = book_value + (diff_yen / shares_out)
                    if adjusted_bps > 0:
                        real_pbr = price / adjusted_bps
                        real_pbr_note = "評価差額金のみ反映（不動産含み益は未反映）"
            if edinet_info and market_cap:
                denom = edinet_info.get("net_net_denominator_yen")
                if denom is not None and denom > 0:
                    net_net = market_cap / denom

            row = dict(t)
            row["real_pbr"] = real_pbr
            row["real_pbr_note"] = real_pbr_note
            row["net_net"] = net_net
            row["real_pbr_tier"] = classify_low_is_good(real_pbr, PBR_TIERS)
            row["net_net_tier"] = classify_low_is_good(net_net, NET_NET_TIERS)
            final_rows.append(row)
        except Exception as e:
            row = dict(t)
            row["error2"] = str(e)
            final_rows.append(row)
        if i % 100 == 0:
            print(f"  進捗: {i}/{len(targets)}")

    with open("/tmp/kabu1000_full_market_final.json", "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=1)
    print(f"最終結果保存先: /tmp/kabu1000_full_market_final.json（{len(final_rows)}件）")


if __name__ == "__main__":
    main()
