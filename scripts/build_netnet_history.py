"""
EDINETから「ネットネット指数の分母」の履歴を集める（かぶ1000流の検証用）

かぶ1000『貯金40万円が株式投資で4億円』第3章：

  換金性が高い流動資産 ＝ 現金及び預金 ＋ 受取手形及び売掛金
                        ＋ 有価証券 ＋ 投資有価証券 － 貸倒引当金
  ネットネット指数 ＝ 時価総額 ÷（換金性が高い流動資産 － 総負債）
  0.66未満＝超割安 / 0.5未満＝激安

既存の `build_edinet_cache.py` は**日経225・最新年のみ**で履歴が無く、
過去に遡った検証ができなかった。こちらは**944銘柄・複数年**を集める。

⚠️ 個別（親会社単体）財務諸表から取る。連結がIFRSの企業でも個別は
   日本基準で開示するのが通例なので、幅広くカバーできる。
   タグ定義は `build_edinet_cache.py` から読み込んで共用する
   （書き写すと本体とずれたときに気づけないため）。

⚠️ 先読みバイアス：`available_from` には**実際の提出日**を入れる。

使い方:
    python3 scripts/build_netnet_history.py [銘柄CSV] [--from 2017] [--to 2026]
      **中断しても再実行すれば続きから進む**
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
sys.path.insert(0, str(ROOT / "scripts"))
import build_edinet_cache as bc   # タグ定義を共用する

ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "data" / "netnet_history.json"
DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"

# 有報の提出は3月期決算なら5〜7月に集中する（12月期は3月）
SEARCH_WINDOWS = [(3, 1, 31), (5, 1, 31), (6, 1, 30), (7, 1, 15)]


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".env に EDINET_API_KEY がありません")


def load_tickers(path: Path) -> list:
    with path.open(encoding="utf-8-sig") as f:
        return [r["code"] for r in csv.DictReader(f) if r.get("code")]


def load_seccode_map(codes: list) -> dict:
    resp = requests.get(bc.EDINET_CODELIST_URL, timeout=60)
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


def extract(doc_id: str, api_key: str) -> dict:
    """1つの有報から、換金性が高い流動資産の内訳と総負債を取り出す"""
    try:
        resp = requests.get(DOCUMENT_URL.format(doc_id=doc_id),
                            params={"type": 5, "Subscription-Key": api_key}, timeout=60)
        if resp.status_code != 200:
            return {}
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        main = [n for n in zf.namelist() if n.endswith(".csv") and "jpaud" not in n]
        if not main:
            return {}
        rows = list(csv.reader(
            zf.open(main[0]).read().decode("utf-16").splitlines(), delimiter="\t"))
    except Exception:
        return {}

    groups = {
        "cash": bc.CASH_TAGS, "receivables": bc.RECEIVABLES_TAGS,
        "securities": bc.SECURITIES_TAGS,
        "investment_securities": bc.INVESTMENT_SECURITIES_TAGS,
        "allowance": bc.ALLOWANCE_TAGS, "liabilities": bc.LIABILITIES_TAGS,
    }
    got = {k: 0.0 for k in groups}
    found = {k: False for k in groups}
    for row in rows[1:]:
        if len(row) < 9:
            continue
        tag, ctx, val = row[0], row[2], row[8]
        # 個別（NonConsolidated）の当期時点のみ。前期（Prior）は使わない
        if "NonConsolidated" not in ctx or "Prior" in ctx:
            continue
        v = _num(val)
        if v is None:
            continue
        for key, tags in groups.items():
            if tag in tags:
                got[key] += v
                found[key] = True

    # 現金と総負債が取れていなければ、その企業は個別で開示していないとみなす
    if not (found["cash"] and found["liabilities"]):
        return {}
    liquid = (got["cash"] + got["receivables"] + got["securities"]
              + got["investment_securities"] + got["allowance"])  # 引当金は負の値
    return {"liquid_assets_yen": liquid, "liabilities_yen": got["liabilities"],
            "denominator_yen": liquid - got["liabilities"]}


def main():
    argv = sys.argv[1:]
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
    print(f"対象 {len(codes)}銘柄（{tickers_path.name}）/ {y_from}〜{y_to}年")

    store = {"data": {}, "fetched_doc_ids": []}
    if OUT_PATH.exists():
        store = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        store.setdefault("fetched_doc_ids", [])
        print(f"  既存: {len(store['data'])}銘柄 / 取得済み{len(store['fetched_doc_ids'])}件")
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
        found = []
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
                    if resp.status_code == 200:
                        for doc in resp.json().get("results", []):
                            ec, did = doc.get("edinetCode"), doc.get("docID")
                            if (doc.get("docTypeCode") == "120" and ec in wanted
                                    and did not in seen):
                                found.append({"doc_id": did, "edinet_code": ec,
                                              "submitted": (doc.get("submitDateTime") or "")[:10],
                                              "period_end": doc.get("periodEnd")})
                except Exception:
                    pass
                time.sleep(0.05)
        print(f"[{year}年] 未取得の有報 {len(found)}件。中身を取得します…")
        ok = 0
        for i, d in enumerate(found, 1):
            rec = extract(d["doc_id"], api_key)
            seen.add(d["doc_id"])
            if rec:
                rec.update({"period_end": d["period_end"],
                            "available_from": d["submitted"]})
                store["data"].setdefault(ed2sec[d["edinet_code"]], {})[d["doc_id"]] = rec
                ok += 1
            if i % 100 == 0:
                print(f"    {i}/{len(found)}（有効 {ok}件）")
                save()
        save()
        print(f"  [{year}年] 完了: 有効{ok}件 / 累計{len(store['data'])}銘柄")

    n = sum(len(v) for v in store["data"].values())
    print(f"\n完了: {OUT_PATH}\n  {len(store['data'])}銘柄 / 合計{n}レコード")


if __name__ == "__main__":
    main()
