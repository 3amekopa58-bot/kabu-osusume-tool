#!/bin/bash
# JPXの上場廃止銘柄一覧を取得して data/jpx_delisted.csv を作る
#
# ページはJavaScriptでバックナンバーを切り替える作りだが、各年のHTMLは
# 固定URL（archives-01〜09）で取れる。表はHTMLに直接埋まっている。
# 取得できるのは2017年以降のみ（JPXがそれ以前を公開していない）。
set -euo pipefail
cd "$(dirname "$0")/.."

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "JPXから上場廃止銘柄一覧を取得します…"
curl -sfL -A "Mozilla/5.0" \
  "https://www.jpx.co.jp/listing/stocks/delisted/index.html" -o "$TMP/cur.html"
for i in 01 02 03 04 05 06 07 08 09; do
  curl -sfL -A "Mozilla/5.0" \
    "https://www.jpx.co.jp/listing/stocks/delisted/archives-${i}.html" \
    -o "$TMP/arch_${i}.html"
  sleep 1  # 連続アクセスを避ける
done

python3 - "$TMP" <<'PYEOF'
import csv, glob, re, sys
from pathlib import Path

tmp = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"{tmp}/*.html")):
    s = Path(path).read_text(encoding="utf-8")
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", s, re.S):
        tds = [re.sub(r"<[^>]+>", "", td).strip().replace("　", " ")
               for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) >= 5 and re.match(r"^\d{4}/\d{2}/\d{2}$", tds[0]):
            rows.append({"date": tds[0], "name": tds[1], "code": tds[2],
                         "market": tds[3], "reason": tds[4]})

uniq = {(r["date"], r["code"]): r for r in rows}
rows = sorted(uniq.values(), key=lambda r: r["date"])
out = Path("data/jpx_delisted.csv")
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date", "name", "code", "market", "reason"])
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)}件を {out} に保存しました（{rows[0]['date']}〜{rows[-1]['date']}）")
PYEOF
