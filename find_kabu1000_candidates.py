"""
かぶ1000氏の3段階評価基準（激安・超割安・割安）に基づいて、225銘柄から
候補を探す（かぶ1000_ルール.md「第5章」節の基準表を参照）。

使える基準（データが揃っているもの）:
  - 予想PER（forwardPE。無ければtrailingPEで代用。赤字＝PERがマイナスの
    銘柄は「割安」ではなく「業績不振」なので対象外扱いにする）
  - 実質PBR（EDINETキャッシュに評価差額金がある銘柄のみ近似計算。
    無い銘柄は通常のPBRで代用し、その旨を明記する）
  - グレアム指数（PER×PBR）
  - 配当利回り
  - ネットネット指数（時価総額÷（換金性が高い流動資産－総負債））。
    EDINETキャッシュに分母データがある銘柄のみ計算（scripts/build_edinet_cache.py
    が取得する net_net_denominator_yen を使用）。分母がマイナス（負債が
    換金性資産を上回る、大半の企業がこれに該当）の場合は計算不能として除外

使えない基準（今回は対象外。理由も表示）:
  - EBITDA倍率：EV/EBITDAの分子（企業価値）の定義が未確認

技術分析（下半身・PPP等）は見ない。あくまでファンダメンタルズの
スクリーニングのみ（screen.pyのスコアリングとは別軸の参考情報）。

使い方:
    python3 find_kabu1000_candidates.py
"""

import csv
import json
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
EDINET_CACHE_PATH = BASE_DIR / "data" / "edinet_valuation_diff.json"

# かぶ1000流の3段階基準（かぶ1000_ルール.md 第5章より）
PER_TIERS = [(6, "激安"), (8, "超割安"), (10, "割安")]
PBR_TIERS = [(0.3, "激安"), (0.4, "超割安"), (0.5, "割安")]
NET_NET_TIERS = [(0.5, "激安"), (0.66, "超割安"), (1.0, "割安")]
GRAHAM_TIERS = [(5.0, "激安"), (8.0, "超割安"), (10.0, "割安")]
DIVIDEND_TIERS = [(5.0, "激安"), (4.0, "超割安"), (3.0, "割安")]  # 高いほど良いので降順評価


def load_tickers():
    with open(TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_edinet_cache() -> dict:
    if not EDINET_CACHE_PATH.exists():
        return {}
    with open(EDINET_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f).get("data", {})


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


def main():
    tickers = load_tickers()
    edinet_cache = load_edinet_cache()
    print(f"{len(tickers)}銘柄を取得します…")

    rows = []
    for i, t in enumerate(tickers, 1):
        code, name = t["code"], t["name"]
        try:
            info = yf.Ticker(code).info
            per = info.get("forwardPE") or info.get("trailingPE")
            pbr = info.get("priceToBook")
            dividend_yield = info.get("dividendYield")
            book_value = info.get("bookValue")
            shares_out = info.get("sharesOutstanding")
            price = info.get("currentPrice")
            market_cap = info.get("marketCap")

            # 赤字（PERがマイナス）は「割安」ではなく「業績不振」なので
            # スクリーニング対象から外す（激安と誤判定しないように）
            if per is not None and per <= 0:
                per = None

            # 実質PBR近似：EDINETの評価差額金（有価証券含み益）だけを反映。
            # 不動産等の含み益は含まれないため、実際の実質PBRより高め
            # （割高寄り）に出る点に注意
            real_pbr = pbr
            real_pbr_note = "PBRで代用（評価差額金データなし）"
            edinet_info = edinet_cache.get(code)
            if edinet_info and book_value and shares_out and price:
                diff_yen = edinet_info.get("valuation_diff_current_yen")
                if diff_yen is not None:
                    adjusted_bps = book_value + (diff_yen / shares_out)
                    if adjusted_bps > 0:
                        real_pbr = price / adjusted_bps
                        real_pbr_note = "評価差額金のみ反映（不動産含み益は未反映）"

            # ネットネット指数＝時価総額÷（換金性が高い流動資産－総負債）。
            # 分母がマイナス（大半の企業）の場合は指数として意味を成さない
            # ため計算しない
            net_net = None
            if edinet_info and market_cap:
                denom = edinet_info.get("net_net_denominator_yen")
                if denom is not None and denom > 0:
                    net_net = market_cap / denom

            graham = (per * pbr) if (per and pbr) else None
            # yfinanceのdividendYieldは既に「%」単位の数値（例:4.02 = 4.02%）で
            # 返ってくる（2026年時点の仕様。フラクション表記ではない）ため、
            # そのまま使う（100倍すると誤って桁が2つ多くなる）
            dividend_pct = dividend_yield if dividend_yield else None

            rows.append({
                "code": code, "name": name, "price": price,
                "per": per, "pbr": pbr, "real_pbr": real_pbr,
                "real_pbr_note": real_pbr_note,
                "graham": graham, "dividend_pct": dividend_pct,
                "net_net": net_net,
                "per_tier": classify_low_is_good(per, PER_TIERS),
                "real_pbr_tier": classify_low_is_good(real_pbr, PBR_TIERS),
                "graham_tier": classify_low_is_good(graham, GRAHAM_TIERS),
                "dividend_tier": classify_high_is_good(dividend_pct, DIVIDEND_TIERS),
                "net_net_tier": classify_low_is_good(net_net, NET_NET_TIERS),
            })
            if i % 30 == 0:
                print(f"  進捗: {i}/{len(tickers)}")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {name} ({code}) 取得失敗: {e}")

    tier_score = {"激安": 3, "超割安": 2, "割安": 1, None: 0}
    tier_keys = ("per_tier", "real_pbr_tier", "graham_tier", "dividend_tier", "net_net_tier")
    for r in rows:
        r["hit_count"] = sum(1 for k in tier_keys if r[k])
        r["tier_total"] = sum(tier_score[r[k]] for k in tier_keys)

    rows.sort(key=lambda r: (-r["hit_count"], -r["tier_total"]))

    print("\n" + "=" * 100)
    print("かぶ1000流：5指標のうち2つ以上で「割安」以上に該当する候補")
    print("=" * 100)
    candidates = [r for r in rows if r["hit_count"] >= 2]
    for r in candidates:
        print(f"\n[{r['code']}] {r['name']}  株価{r['price']}円")
        print(
            (f"  PER {r['per']:.1f}倍" + (f"（{r['per_tier']}）" if r['per_tier'] else "")) if r['per'] else "  PER 不明",
            end="  ",
        )
        print(
            (f"実質PBR {r['real_pbr']:.2f}倍" + (f"（{r['real_pbr_tier']}）" if r['real_pbr_tier'] else "")) if r['real_pbr'] else "実質PBR 不明",
            end="  ",
        )
        print(
            (f"グレアム指数 {r['graham']:.1f}" + (f"（{r['graham_tier']}）" if r['graham_tier'] else "")) if r['graham'] else "グレアム指数 不明",
            end="  ",
        )
        print(
            (f"配当利回り {r['dividend_pct']:.1f}%" + (f"（{r['dividend_tier']}）" if r['dividend_tier'] else "")) if r['dividend_pct'] else "配当利回り 不明",
            end="  ",
        )
        print(
            (f"ネットネット指数 {r['net_net']:.2f}" + (f"（{r['net_net_tier']}）" if r['net_net_tier'] else "")) if r['net_net'] else "ネットネット指数 計算不能（分母マイナス or データなし）",
        )
        print(f"  実質PBRの算出方法: {r['real_pbr_note']}")

    print(f"\n該当銘柄数: {len(candidates)}/{len(rows)}")
    print("\n※EBITDA倍率は今回のデータでは計算していません")
    print("※実質PBRは有価証券の含み益のみ反映（不動産含み益は含まれないため、実際より割高寄りに出ます）")
    print("※ネットネット指数は分母（換金性が高い流動資産－総負債）がプラスの銘柄のみ計算（大半はマイナスのため対象外）")


if __name__ == "__main__":
    main()
