# A8.net 紹介専用ブログ

株ツールとは独立。標準ライブラリのみで動く静的サイトジェネレーター。

## 使い方
1. `site.json` にサイト名・運営者名を入れる
2. A8.netで提携した商品を `products.csv` に1行ずつ追加（`affiliate_url` にA8の提携リンクのURL）
3. `python3 asp/build.py` → `asp/public/` にサイトが出力される
4. `asp/public/` を公開（GitHub Pages / Cloudflare Pages / Netlify など）

## ルール
- 全ページ上部に「【広告】」表記が自動で入る（ステマ規制対応）。消さないこと
- 感想は自分が実際に使った体験を書く。体験していないものを「使ってみた」と書かない
- 効果の断定（「必ず上達する」等）は避ける
- A8の提携リンクは `rel="nofollow sponsored"` で出力される
- A8の規約で禁止される広告表現・自己アフィリエイトは各プログラムの注意事項を確認
