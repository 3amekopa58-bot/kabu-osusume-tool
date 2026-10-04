"""紹介専用ブログの静的サイト生成。標準ライブラリのみ。

使い方:  python3 asp/build.py   → asp/public/ にHTMLが出力される
商品の追加・編集は asp/products.csv だけで行う（site.json でサイト名等を変更）。
"""
import csv
import json
import shutil
from html import escape as e
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "public"
SITE = json.loads((ROOT / "site.json").read_text(encoding="utf-8"))
BASE = SITE.get("base_path", "").rstrip("/")  # GitHub Pages用: "/リポジトリ名/asp" など

CSS = """
:root{--c:#2b6cb0;--bg:#f7f8fa;--t:#222}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,"Hiragino Sans","Noto Sans JP",sans-serif;background:var(--bg);color:var(--t);line-height:1.8}
header,footer{background:#fff;padding:14px 16px;border-bottom:1px solid #e3e6ea}footer{border:0;border-top:1px solid #e3e6ea;margin-top:40px;font-size:.85rem;color:#666}
.w{max-width:760px;margin:0 auto;padding:0 16px}header a{color:var(--t);text-decoration:none;font-weight:700;font-size:1.2rem}
.pr{background:#fff8e1;border:1px solid #f0d68a;padding:8px 12px;font-size:.85rem;border-radius:6px;margin:16px 0}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px;margin:14px 0}.card h2{margin:0 0 4px;font-size:1.1rem}
.tag{display:inline-block;background:#e6f0fa;color:var(--c);font-size:.75rem;padding:2px 8px;border-radius:99px}
.btn{display:block;text-align:center;background:#e8590c;color:#fff!important;padding:14px;border-radius:8px;text-decoration:none;font-weight:700;margin:20px 0}
.pro li::marker{content:"○ "}.con li::marker{content:"△ "}table{border-collapse:collapse;width:100%}td,th{border:1px solid #e3e6ea;padding:8px;background:#fff;text-align:left}
img{max-width:100%;height:auto}a{color:var(--c)}
"""


def page(title, body, desc="", path=""):
    name = e(SITE["name"])
    canon = f'<link rel="canonical" href="{e(SITE["url"].rstrip("/") + path)}">' if SITE.get("url") else ""
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}｜{name}</title><meta name="description" content="{e(desc)}">{canon}
<link rel="stylesheet" href="{BASE}/style.css"></head><body>
<header><div class="w"><a href="{BASE}/">{name}</a></div></header>
<main class="w">
<p class="pr">【広告】本ページはプロモーション（アフィリエイト広告）を含みます。</p>
{body}</main>
<footer><div class="w"><a href="{BASE}/about.html">運営者情報・広告について</a>　<a href="{BASE}/privacy.html">プライバシーポリシー</a><p>© {name}</p></div></footer>
</body></html>"""


def load():
    with open(ROOT / "products.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("slug")]
    slugs = [r["slug"] for r in rows]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("products.csv: slug が重複しています")
    return rows


def paras(text):
    text = text.replace("\\n", "\n")
    return "".join(f"<p>{e(p.strip())}</p>" for p in text.split("\n\n") if p.strip())


def lis(s, cls):
    items = [x.strip() for x in s.split("|") if x.strip()]
    return f'<ul class="{cls}">' + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul>" if items else ""


def link(r, label="公式サイトで詳細を見る"):
    url = r["affiliate_url"].strip()
    if not url.startswith("https://"):
        raise SystemExit(f"{r['slug']}: affiliate_url が https:// で始まっていません")
    return f'<a class="btn" href="{e(url)}" rel="nofollow sponsored noopener" target="_blank">{label} ›</a>'


def card(r):
    return (f'<div class="card"><span class="tag">{e(r["category"])}</span>'
            f'<h2><a href="{BASE}/p/{e(r["slug"])}.html">{e(r["name"])}</a></h2>'
            f'<p>{e(r["summary"])}</p></div>')


def write(path, html):
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")


def main():
    rows = load()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    write("style.css", CSS)

    cats = sorted({r["category"] for r in rows})
    nav = " ".join(f'<a class="tag" href="{BASE}/c/{e(c)}.html">{e(c)}</a>' for c in cats)
    write("index.html", page(SITE["name"], f"<h1>{e(SITE['tagline'])}</h1><p>{nav}</p>" + "".join(card(r) for r in rows),
                             SITE["tagline"], "/"))
    for c in cats:
        write(f"c/{c}.html", page(c, f"<h1>{e(c)}</h1>" + "".join(card(r) for r in rows if r["category"] == c)))

    for r in rows:
        body = (f'<span class="tag">{e(r["category"])}</span><h1>{e(r["name"])}</h1><p>{e(r["summary"])}</p>'
                f'<table><tr><th>価格</th><td>{e(r["price"])}</td></tr><tr><th>おすすめの人</th><td>{e(r["target"])}</td></tr></table>'
                f'{link(r)}<h2>良いところ</h2>{lis(r["pros"], "pro")}<h2>気になるところ</h2>{lis(r["cons"], "con")}'
                f'<h2>使ってみた感想</h2>{paras(r["review"])}{link(r)}')
        write(f"p/{r['slug']}.html", page(r["name"], body, r["summary"], f"/p/{r['slug']}.html"))

    write("about.html", page("運営者情報・広告について",
          f"<h1>運営者情報・広告について</h1><p>運営者：{e(SITE['owner'])}</p>"
          "<p>当サイトはA8.netなどのアフィリエイトプログラムに参加しており、記事内のリンクから商品を購入・申込みされた場合、"
          "運営者が報酬を受け取ることがあります。ただし、掲載内容は運営者自身の体験と判断に基づいており、報酬の有無・多寡で評価を変えることはありません。</p>"))
    write("privacy.html", page("プライバシーポリシー",
          "<h1>プライバシーポリシー</h1><p>当サイトでは、広告配信およびアクセス解析のためCookieを使用する場合があります。"
          "Cookieはブラウザの設定で無効にできます。取得した情報は、これらの目的以外には利用しません。</p>"))
    print(f"{len(rows)}商品 → {OUT} に出力しました")


if __name__ == "__main__":
    main()
