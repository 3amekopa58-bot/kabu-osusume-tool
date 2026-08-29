"""
生存者バイアスの大きさを見積もる

universe.csv は「今日買えて今日流動性がある」銘柄で作っているため、
過去に上場廃止になった銘柄が1つも入っていない。この欠落が結果を
どちら向きにどれだけ歪めるかを、JPXの上場廃止銘柄一覧から見積もる。

⚠️ 上場廃止は投資家から見て2種類あり、バイアスの向きが逆になる：
  - 買収・MBO・株式併合 → プレミアム付きで買い取られる（株主は利益）
    これが欠落する＝**良い結果を取り除いている**＝戦略に不利なバイアス
  - 経営破綻          → 株価はほぼゼロ（株主は損失）
    これが欠落する＝**悪い結果を取り除いている**＝戦略に有利なバイアス

上場廃止銘柄の株価そのものは yfinance が返さない（2026-08-30に東芝・
ベネッセHD等6銘柄で確認。"possibly delisted; no timezone found"）ため、
件数と理由の分布から間接的に見積もる。

データ元: https://www.jpx.co.jp/listing/stocks/delisted/
  スクレイプ結果は data/jpx_delisted.csv に保存済み（scripts/fetch_jpx_delisted.sh）

使い方:
    python3 scripts/analyze_survivorship.py
"""

import collections
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DELISTED_CSV = BASE_DIR / "data" / "jpx_delisted.csv"

# JPXのアーカイブは2017年以降しか公開されていない
PERIOD_FROM, PERIOD_TO = "2017/01/01", "2025/12/31"
YEARS = 9
LISTED = 3822        # 全上場銘柄数（scripts/build_all_listed_tickers.py 実測）
HOLD_DAYS = 51       # 採用ルールの平均保有日数
TRADES_26Y = 13633   # 944銘柄・26年のトレード数
EXPECTED_PCT = 3.04  # 現在の期待値（notify.py と同じ値）


def classify(reason: str) -> str:
    """上場廃止理由を、株を持っていた投資家から見た帰結で分ける。"""
    if any(k in reason for k in ("破産", "民事再生", "会社更生", "銀行取引停止",
                                 "債務超過", "解散")):
        return "経営破綻"
    if any(k in reason for k in ("買収", "ＭＢＯ", "MBO", "売渡請求", "株式の併合",
                                 "株式併合", "完全子会社", "株式交換", "株式移転", "合併")):
        return "買収・非公開化"
    return "その他"


def main():
    if not DELISTED_CSV.exists():
        print(f"{DELISTED_CSV} がありません。"
              "先に scripts/fetch_jpx_delisted.sh を実行してください。")
        return

    rows = [r for r in csv.DictReader(open(DELISTED_CSV, encoding="utf-8-sig"))
            if PERIOD_FROM <= r["date"] <= PERIOD_TO]
    n = len(rows)
    cat = collections.Counter(classify(r["reason"]) for r in rows)
    per_year = n / YEARS

    print(f"=== 上場廃止の実績（{PERIOD_FROM[:4]}〜{PERIOD_TO[:4]}年・JPX公式）===")
    print(f"  総件数: {n}件 / {YEARS}年 = 年平均{per_year:.0f}件"
          f"（上場{LISTED}社の{per_year / LISTED * 100:.2f}%/年）\n")
    for k, c in cat.most_common():
        print(f"  {c:5d}件 ({c / n * 100:5.1f}%)  {k}")

    hold_frac = HOLD_DAYS / 365
    p = {k: per_year * (c / n) / LISTED * hold_frac for k, c in cat.items()}

    print(f"\n=== 1トレード（平均{HOLD_DAYS}日保有）が上場廃止に遭う確率 ===")
    for k in cat:
        print(f"  {k:8s}: {p[k] * 100:.4f}%  "
              f"→ 26年{TRADES_26Y:,}トレード中 {TRADES_26Y * p[k]:.2f}件")

    # 最悪ケース：破綻に遭ったトレードがすべて -100% だったとみなす
    n_bank = TRADES_26Y * p.get("経営破綻", 0)
    impact = n_bank * (-100 - EXPECTED_PCT) / TRADES_26Y
    print(f"\n=== 最悪ケース（破綻トレードがすべて-100%）===")
    print(f"  平均リターンへの影響: {impact:+.4f}pt")
    print(f"  期待値 {EXPECTED_PCT:+.2f}% → {EXPECTED_PCT + impact:+.2f}%")

    print(f"\n=== 結論 ===")
    print(f"  欠落銘柄の{cat['買収・非公開化'] / n * 100:.1f}%は買収・非公開化で、"
          "株主はプレミアムを受け取っている。")
    print("  つまりこの期間の生存者バイアスは、良い結果を取り除く方向＝")
    print("  **戦略に不利**に働いており、実測値はむしろ過小評価の可能性がある。")
    print(f"  破綻による過大評価は{abs(impact):.4f}ptで、無視できる規模。")
    print("\n  ⚠️ ただしJPXのアーカイブは2017年以降のみ。26年バックテストが含む")
    print("     2001-2003（ITバブル崩壊）・2008-2009（金融危機）は倒産率が")
    print("     はるかに高く、この期間のバイアスは未検証。")
    print("     → 採用判断は 5年・10年 の結果を主の根拠にすること。")


if __name__ == "__main__":
    main()
