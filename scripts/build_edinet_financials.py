"""
EDINETから業績の時系列（売上高・経常利益・純利益・純資産・発行済株式数）を集める

yfinance の財務データは5期分しか遡れず、割安度・成長性の検証が
2022年以降の2割弱のトレードにしか当てられなかった（REQUIREMENTS 4.4-10/11/12）。
特に片山流の検証は実質2年分しかなく、成績がほぼ2025年だけで作られていた
ため判定を保留している。期間を伸ばすためにEDINETから直接集める。

有価証券報告書の「主要な経営指標等の推移」は**5期分**を掲載する規定なので、
1社につき5年おきに取得すれば、1回あたり5年分をカバーできる。
⚠️ ただしEDINET APIは保存期間が約10年で、2017年より前の日付には応答しない
（実測：2016-06-29 は status=404）。そのため遡れるのは**2013年度まで**で、
当初想定した2008年までは取得できない。

⚠️ **株式併合・分割**：有報のBPS/EPSは「その時点の株数」ベースだが、yfinanceの
株価は併合・分割を遡って調整済み。そのまま組み合わせるとPER/PBRが桁でずれる
（古河電気工業(5801)は2024年の10株→1株併合で、EDINETのBPSがyfinance比10倍に
なることを実測）。**必ず scripts/adjust_edinet_split.py で調整してから使うこと。**

⚠️ 先読みバイアス：決算期末の数字はその日には公表されていない。
有報の提出期限は期末から3か月以内なので、期末+92日を available_from として
記録し、バックテスト側はその日以降でのみ使う（build_fundamental_history.py と同じ）。

使い方:
    python3 scripts/build_edinet_financials.py [銘柄CSV] [--years 2026,2021,2016,2011]
      銘柄CSV 省略時は universe.csv
"""

import csv
import datetime as dt
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "data" / "edinet_financials.json"

CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"

# 「主要な経営指標等の推移」は5期分載るので、5年刻みで取れば重複なく遡れる
# ⚠️ EDINET APIは書類の保存期間が約10年で、2017年より前の日付は
# status=404 を返す（2026-08-30に実測。2016-06-29 → Not Found）。
# したがって遡れる上限は2017年提出分＝2013年度まで。
# 「主要な経営指標等の推移」は5期分なので 2017 と 2022 と 2026 で
# 2013〜2026年度をカバーできる
DEFAULT_YEARS = [2026, 2022, 2017]
# 有報の提出は3月期決算なら5〜7月に集中する
SEARCH_MONTHS = [(5, 1), (8, 15)]
DISCLOSURE_LAG_DAYS = 92

# 会計基準によってタグが違うので候補を順に探す（日本基準/IFRS/米国基準）
# タグ名は実際の有報から確認したもの（推測で書くとIFRS企業を静かに取りこぼす。
# 2026-08-30、"RevenuesIFRS"（複数形）と書いていて村田製作所・ソフトバンクG等の
# 売上高が全欠損した経緯あり。正しくは "RevenueIFRS"（単数形））
TAGS = {
    "revenue": ["jpcrp_cor:NetSalesSummaryOfBusinessResults",
                "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",
                "jpcrp_cor:NetSalesIFRSSummaryOfBusinessResults",
                "jpcrp_cor:TotalRevenuesUSGAAPSummaryOfBusinessResults",
                "jpcrp_cor:OperatingRevenue1SummaryOfBusinessResults",
                "jpcrp_cor:OperatingRevenueSummaryOfBusinessResults"],
    # ⚠️ IFRSに「経常利益」は無いので税引前利益が入る。日本基準の経常利益とは
    # 別物なので、増減率の比較には使えても水準の横並び比較には使わないこと
    "ordinary_income": ["jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults",
                        "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults",
                        "jpcrp_cor:IncomeBeforeIncomeTaxesUSGAAPSummaryOfBusinessResults"],
    "net_income": ["jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults",
                   "jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
                   "jpcrp_cor:NetIncomeLossUSGAAPSummaryOfBusinessResults"],
    "net_assets": ["jpcrp_cor:NetAssetsSummaryOfBusinessResults",
                   "jpcrp_cor:EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
                   "jpcrp_cor:TotalEquityIFRSSummaryOfBusinessResults",
                   "jpcrp_cor:TotalShareholdersEquityUSGAAPSummaryOfBusinessResults"],
    # IFRS様式には発行済株式数のタグが無い。無い場合は純資産÷BPSで逆算する
    "shares": ["jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults"],
    # IFRS様式では "EquityToAssetRatioIFRS..." が1株当たり親会社所有者帰属持分
    # （名前に反してBPS。村田製作所で1,493.58＝yfinance実績と0.0%一致を確認済み）
    "bps": ["jpcrp_cor:NetAssetsPerShareSummaryOfBusinessResults",
            "jpcrp_cor:EquityToAssetRatioIFRSSummaryOfBusinessResults"],
    # 日本基準は "BasicEarningsLossPerShare"（Loss入り）。"BasicEarningsPerShare"
    # と書いていて日本基準企業のEPSが全欠損した（2026-08-30）
    "eps": ["jpcrp_cor:BasicEarningsLossPerShareSummaryOfBusinessResults",
            "jpcrp_cor:BasicEarningsPerShareSummaryOfBusinessResults",
            "jpcrp_cor:BasicEarningsLossPerShareIFRSSummaryOfBusinessResults"],
}
# 「主要な経営指標等の推移」は当期＋過去4期。連結（Memberが付かない方）を使う
CONTEXTS = ["Prior4Year", "Prior3Year", "Prior2Year", "Prior1Year", "CurrentYear"]


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".env に EDINET_API_KEY がありません")


def load_seccode_map(codes: list) -> dict:
    """証券コード（4桁）→ EDINETコード の対応表"""
    resp = requests.get(CODELIST_URL, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    text = zf.open("EdinetcodeDlInfo.csv").read().decode("cp932")
    lines = text.splitlines()
    m = {}
    for row in csv.DictReader(lines[1:]):
        sc = row["証券コード"].strip()
        if sc:
            m[sc[:4]] = row["ＥＤＩＮＥＴコード"]
    return {c: m.get(c.split(".")[0]) for c in codes}


def find_docs(year: int, wanted: set, api_key: str) -> dict:
    """その年に提出された有価証券報告書のdocIDを、EDINETコード別に集める"""
    found = {}
    d = dt.date(year, *SEARCH_MONTHS[0])
    end = dt.date(year, *SEARCH_MONTHS[1])
    days = 0
    while d <= end:
        try:
            resp = requests.get(DOCUMENTS_URL,
                                params={"date": d.isoformat(), "type": 2,
                                        "Subscription-Key": api_key}, timeout=30)
            if resp.status_code == 200:
                for doc in resp.json().get("results", []):
                    ec = doc.get("edinetCode")
                    if doc.get("docTypeCode") == "120" and ec in wanted and ec not in found:
                        found[ec] = doc.get("docID")
        except Exception:
            pass
        d += dt.timedelta(days=1)
        days += 1
        if days % 20 == 0:
            print(f"    {year}年 {d.isoformat()} まで走査 → {len(found)}社ぶん発見")
        time.sleep(0.05)
    return found


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def extract(doc_id: str, api_key: str) -> list:
    """1つの有報から、5期分の業績レコードを取り出す"""
    try:
        resp = requests.get(DOCUMENT_URL.format(doc_id=doc_id),
                            params={"type": 5, "Subscription-Key": api_key}, timeout=60)
        if resp.status_code != 200:
            return []
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        main = [n for n in zf.namelist() if "jpcrp030000-asr-001" in n]
        if not main:
            return []
        rows = list(csv.reader(
            zf.open(main[0]).read().decode("utf-16").splitlines(), delimiter="\t"))
    except Exception:
        return []

    # tag -> context -> value（連結のみ。NonConsolidatedMember が付く行は使わない）
    table, fy_end, filer = {}, None, None
    for row in rows[1:]:
        if len(row) < 9:
            continue
        tag, ctx, val = row[0], row[2], row[8]
        if tag == "jpdei_cor:CurrentFiscalYearEndDateDEI":
            fy_end = val
        elif tag == "jpdei_cor:FilerNameInJapaneseDEI":
            filer = val
        if "NonConsolidatedMember" in ctx:
            continue
        base = ctx.split("_")[0]
        for key, cands in TAGS.items():
            if tag in cands:
                table.setdefault(key, {}).setdefault(base, val)

    if not fy_end:
        return []
    try:
        end_date = dt.date.fromisoformat(fy_end)
    except ValueError:
        return []

    out = []
    for offset, ctx_prefix in enumerate(CONTEXTS):
        back = len(CONTEXTS) - 1 - offset  # Prior4 → 4年前
        try:
            period_end = end_date.replace(year=end_date.year - back)
        except ValueError:  # 2/29 など
            period_end = end_date.replace(year=end_date.year - back, day=28)
        # Duration（期間）とInstant（時点）でコンテキスト名が違う
        rec = {"period_end": period_end.isoformat(),
               "available_from": (period_end + dt.timedelta(days=DISCLOSURE_LAG_DAYS)).isoformat(),
               "filer_name": filer}
        got = False
        for key in TAGS:
            for suffix in ("Duration", "Instant"):
                v = table.get(key, {}).get(ctx_prefix + suffix)
                if v is not None:
                    rec[key] = _num(v)
                    got = got or rec[key] is not None
                    break
            else:
                rec[key] = None
        # IFRS様式は発行済株式数を載せないので、純資産÷BPSで補う
        if rec.get("shares") is None and rec.get("net_assets") and rec.get("bps"):
            try:
                rec["shares"] = rec["net_assets"] / rec["bps"]
                rec["shares_estimated"] = True
            except ZeroDivisionError:
                pass
        if got:
            out.append(rec)
    return out


def main():
    args, years, skip = [], DEFAULT_YEARS, False
    for a in sys.argv[1:]:
        if skip:
            years, skip = [int(x) for x in a.split(",")], False
            continue
        if a == "--years":
            skip = True
            continue
        if not a.startswith("--"):
            args.append(a)
    path = Path(args[0]) if args else ROOT / "universe.csv"

    api_key = load_api_key()
    codes = [t["code"] for t in csv.DictReader(open(path, encoding="utf-8-sig"))]
    print(f"{len(codes)}銘柄（{path.name}）のEDINETコードを引き当て中…")
    ec_map = load_seccode_map(codes)
    matched = {c: e for c, e in ec_map.items() if e}
    print(f"  {len(matched)}/{len(codes)} 銘柄で一致")
    ec_to_code = {e: c for c, e in matched.items()}

    # 既存の結果があれば引き継ぐ（途中で止まっても再開できるように）
    result = {}
    if OUT_PATH.exists():
        try:
            result = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("data", {})
            print(f"  既存データ {len(result)}銘柄ぶんを引き継ぎます")
        except Exception:
            result = {}

    for year in years:
        print(f"\n=== {year}年提出分（{year-4}〜{year}年度をカバー）===")
        docs = find_docs(year, set(ec_to_code), api_key)
        print(f"  有報 {len(docs)}社ぶんを取得します…")
        for i, (ec, doc_id) in enumerate(docs.items(), 1):
            code = ec_to_code[ec]
            for rec in extract(doc_id, api_key):
                result.setdefault(code, {})[rec["period_end"]] = rec
            if i % 100 == 0 or i == len(docs):
                print(f"    [{i}/{len(docs)}] 累計 {len(result)}銘柄")
            time.sleep(0.03)
        # 年ごとに保存しておく（長時間走るので途中で失っても無駄にしない）
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "note": "EDINETの有報「主要な経営指標等の推移」から取得。"
                    "available_from 以降でのみ使うこと（先読み防止）",
            "disclosure_lag_days": DISCLOSURE_LAG_DAYS,
            "data": result,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"  途中保存: {OUT_PATH}")

    n_periods = [len(v) for v in result.values()]
    print(f"\n完了: {OUT_PATH}")
    print(f"  銘柄数: {len(result)}")
    if n_periods:
        print(f"  1銘柄あたりの決算期数: 中央値 {sorted(n_periods)[len(n_periods)//2]}期")
        alld = sorted(p for v in result.values() for p in v)
        print(f"  期間: {alld[0]} 〜 {alld[-1]}")


if __name__ == "__main__":
    main()
