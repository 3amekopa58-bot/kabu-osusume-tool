"""
EDINETから四半期・半期の業績（売上高・経常利益）を集める

片山晃『5年で1億貯める株式投資』PART 6 は増収率を
**「四半期決算ごとに前年同期比『売上高10%増』が目安」**と書いている。
このツールの `revenue_growth` は有価証券報告書＝**年次**なので粒度が違う
（REQUIREMENTS 4.4-18）。J-Quantsで補ったが、無料プランは約2年しか
遡れず**長期検証ができない**ため、判定には使えていなかった。

EDINETなら無料でここを埋められる：

  四半期報告書（docTypeCode=140）… 2017年〜2024年3月（2024年4月に制度廃止）
  半期報告書  （docTypeCode=160）… 2024年4月〜現在

**どちらも1件の中に「当期」と「前年同期」が両方入っている**ので、
1件取れば前年同期比が完結する（有報のように5期分を組み立てる必要がない）。

  140: 相対年度 = 「当四半期累計期間」「前年度同四半期累計期間」
  160: 相対年度 = 「当中間期」「前中間期」

⚠️ 先読みバイアス：`available_from` には**実際の提出日**（documents.jsonの
submitDateTime）を入れる。有報ビルダーは提出日が分からないので
「期末+92日」で近似していたが、こちらは本物の日付が使えるので近似不要。

⚠️ 連結優先。`NonConsolidatedMember` が付く行（個別）は使わない。

使い方:
    python3 scripts/build_edinet_quarterly.py [銘柄CSV] [--from 2017] [--to 2026]
      銘柄CSV 省略時は universe.csv
      **中断しても再実行すれば続きから進む**（取得済みdocIDは飛ばす）
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
OUT_PATH = ROOT / "data" / "edinet_quarterly.json"

CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"

WANTED_TYPES = {"140": "quarter", "160": "half"}

# 四半期報告書は期末から45日以内、半期報告書は3か月以内が提出期限。
# 決算期は3月・12月・9月などに散らばるので、提出が集中する月の
# 1〜25日を走査する（全日走査は8年で約2,900回になり時間がかかりすぎる）
SEARCH_WINDOWS = [(m, 1, 25) for m in (2, 5, 6, 7, 8, 9, 11, 12)]

# 売上高・経常利益のタグ。「経営指標等」版と本表版の両方が入るので順に探す
REVENUE_TAGS = [
    "jpcrp_cor:NetSalesSummaryOfBusinessResults",
    "jpcrp_cor:NetSalesSummaryOfBusinessRes",
    "jppfs_cor:NetSales",
    "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",
    "jpigp_cor:RevenueIFRS",
]
PROFIT_TAGS = [
    "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults",
    "jpcrp_cor:OrdinaryIncomeLossSummaryOfB",
    "jppfs_cor:OrdinaryIncome",
    "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults",
    "jpigp_cor:ProfitLossBeforeTaxIFRS",
]
# 「相対年度」の表記 → 当期/前年同期
CUR_LABELS = {"当四半期累計期間", "当中間期", "当中間会計期間", "当第2四半期累計期間"}
PRIOR_LABELS = {"前年度同四半期累計期間", "前中間期", "前中間会計期間",
                "前年度同中間期", "前第2四半期累計期間"}


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".env に EDINET_API_KEY がありません")


def load_tickers(path: Path) -> list:
    with path.open(encoding="utf-8-sig") as f:
        return [r["code"] for r in csv.DictReader(f) if r.get("code")]


def load_seccode_map(codes: list) -> dict:
    """証券コード（4桁）→ EDINETコード"""
    resp = requests.get(CODELIST_URL, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    lines = zf.open("EdinetcodeDlInfo.csv").read().decode("cp932").splitlines()
    m = {}
    for row in csv.DictReader(lines[1:]):
        sc = (row.get("証券コード") or "").strip()
        if sc:
            m[sc[:4]] = row["ＥＤＩＮＥＴコード"]
    return {c: m.get(c.split(".")[0]) for c in codes}


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def find_docs(year: int, wanted: set, api_key: str, seen: set) -> list:
    """その年の四半期・半期報告書を探す。戻り値は取得すべき書類のメタ情報"""
    out = []
    for month, d0, d1 in SEARCH_WINDOWS:
        for day in range(d0, d1 + 1):
            try:
                date = dt.date(year, month, day)
            except ValueError:
                continue
            if date > dt.date.today():
                continue
            try:
                resp = requests.get(DOCUMENTS_URL,
                                    params={"date": date.isoformat(), "type": 2,
                                            "Subscription-Key": api_key}, timeout=30)
                if resp.status_code != 200:
                    continue
                for doc in resp.json().get("results", []):
                    ec, tc, did = (doc.get("edinetCode"), doc.get("docTypeCode"),
                                   doc.get("docID"))
                    if tc in WANTED_TYPES and ec in wanted and did not in seen:
                        out.append({"doc_id": did, "edinet_code": ec,
                                    "kind": WANTED_TYPES[tc],
                                    "submitted": (doc.get("submitDateTime") or "")[:10],
                                    "period_end": doc.get("periodEnd"),
                                    "desc": doc.get("docDescription") or ""})
            except Exception:
                pass
            time.sleep(0.05)
        print(f"    {year}年{month}月 走査済 → 累計{len(out)}件")
    return out


def extract(doc_id: str, api_key: str) -> dict:
    """1件の四半期／半期報告書から、当期と前年同期の売上高・経常利益を取り出す"""
    try:
        resp = requests.get(DOCUMENT_URL.format(doc_id=doc_id),
                            params={"type": 5, "Subscription-Key": api_key}, timeout=60)
        if resp.status_code != 200:
            return {}
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        # 監査報告書（jpaud）ではなく本体を読む
        main = [n for n in zf.namelist()
                if n.endswith(".csv") and "jpaud" not in n]
        if not main:
            return {}
        rows = list(csv.reader(
            zf.open(main[0]).read().decode("utf-16").splitlines(), delimiter="\t"))
    except Exception:
        return {}

    got = {}
    for row in rows[1:]:
        if len(row) < 9:
            continue
        tag, ctx, rel, val = row[0], row[2], row[3], row[8]
        if "NonConsolidatedMember" in ctx:      # 個別は使わない（連結優先）
            continue
        if rel in CUR_LABELS:
            when = "cur"
        elif rel in PRIOR_LABELS:
            when = "prior"
        else:
            continue
        if tag in REVENUE_TAGS:
            got.setdefault(f"revenue_{when}", _num(val))
        elif tag in PROFIT_TAGS:
            got.setdefault(f"profit_{when}", _num(val))
    return {k: v for k, v in got.items() if v is not None}


def main():
    argv = sys.argv[1:]
    # --from / --to の「値」を位置引数と取り違えないように、フラグとその値を除く
    skip = set()
    for i, a in enumerate(argv):
        if a in ("--from", "--to"):
            skip.update({i, i + 1})
    args = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("--")]
    tickers_path = Path(args[0]) if args else ROOT / "universe.csv"
    y_from = int(argv[argv.index("--from") + 1]) if "--from" in argv else 2017
    y_to = int(argv[argv.index("--to") + 1]) if "--to" in argv else dt.date.today().year

    api_key = load_api_key()
    codes = load_tickers(tickers_path)
    print(f"対象: {len(codes)}銘柄（{tickers_path.name}）/ {y_from}〜{y_to}年")

    # 途中から再開できるように、既存の結果を読み込む
    store = {"data": {}, "fetched_doc_ids": []}
    if OUT_PATH.exists():
        store = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        store.setdefault("fetched_doc_ids", [])
        print(f"  既存データを読み込みました: {len(store['data'])}銘柄 / "
              f"取得済み書類{len(store['fetched_doc_ids'])}件")
    seen = set(store["fetched_doc_ids"])

    print("  証券コード→EDINETコードの対応表を取得中…")
    sec2ed = load_seccode_map(codes)
    ed2sec = {e: c for c, e in sec2ed.items() if e}
    wanted = set(ed2sec)
    print(f"  EDINETコードが分かったのは {len(wanted)}銘柄")

    def save():
        store["fetched_doc_ids"] = sorted(seen)
        store["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        OUT_PATH.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

    for year in range(y_from, y_to + 1):
        print(f"\n[{year}年] 書類を探しています…")
        docs = find_docs(year, wanted, api_key, seen)
        print(f"  未取得の書類: {len(docs)}件。中身を取得します…")
        ok = 0
        for i, d in enumerate(docs, 1):
            rec = extract(d["doc_id"], api_key)
            seen.add(d["doc_id"])
            if rec.get("revenue_cur") is not None and rec.get("revenue_prior") is not None:
                code = ed2sec[d["edinet_code"]]
                rec.update({"kind": d["kind"], "period_end": d["period_end"],
                            "available_from": d["submitted"], "desc": d["desc"][:60]})
                store["data"].setdefault(code, {})[d["doc_id"]] = rec
                ok += 1
            if i % 100 == 0:
                print(f"    {i}/{len(docs)} 取得済（有効 {ok}件）")
                save()
            time.sleep(0.05)
        save()
        print(f"  [{year}年] 完了: 有効レコード {ok}件 / 累計{len(store['data'])}銘柄")

    n = sum(len(v) for v in store["data"].values())
    print(f"\n完了: {OUT_PATH}")
    print(f"  {len(store['data'])}銘柄 / 合計{n}レコード")


if __name__ == "__main__":
    main()
