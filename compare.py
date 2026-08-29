"""
複数の売買ルールを1プロセスでまとめて比較する

backtest.py を条件ごとに起動すると、そのたびに株価を読み直し、結果を1つずつ
目で見比べることになる。このスクリプトは株価を一度だけ用意して全条件を回し、
勝率・PF・平均リターンを1枚の表にして出す。

使い方:
    # ファイルに書いた条件リストを比較（1行1条件、# はコメント）
    python3 compare.py variants.txt

    # 条件をその場で並べる（条件どうしは ";" で区切る）
    python3 compare.py "timesl 5y trend either ; timesl 5y trend kahanshin"

    # 全条件を複数期間で回す（採用基準＝5年・10年・26年で一貫して改善するか）
    python3 compare.py variants.txt --periods 5y,10y,max

各条件の書式は backtest.py の引数と同じ（例: `timesl trend marketadx volume rs
either sl10 ts60 5y`）。条件に期間を書かず --periods を指定した場合は、
指定した全期間について回す。
"""

import sys
from pathlib import Path

import pandas as pd

import backtest as bt

BASE_DIR = Path(__file__).parent


def profit_factor(returns: pd.Series) -> float:
    """PF＝勝ちトレードの合計利益 ÷ 負けトレードの合計損失（絶対値）"""
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    return float(wins / abs(losses)) if losses else float("inf")


def parse_variants(arg: str) -> list:
    """ファイルパスか、";" 区切りの条件文字列を条件リストに変換する"""
    path = Path(arg)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        specs = [ln.split("#")[0].strip() for ln in lines]
    else:
        specs = [s.strip() for s in arg.split(";")]
    return [s for s in specs if s]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    specs = parse_variants(sys.argv[1])
    periods = None
    if "--periods" in sys.argv:
        periods = sys.argv[sys.argv.index("--periods") + 1].split(",")
    tickers_path = None
    if "--tickers" in sys.argv:
        tickers_path = sys.argv[sys.argv.index("--tickers") + 1]

    tickers = bt.load_tickers(tickers_path)
    print(f"対象銘柄: {len(tickers)}件"
          f"（{Path(tickers_path).name if tickers_path else 'tickers.csv'}）")
    rows = []

    # 期間ごとに株価を1回だけ用意し、その期間の全条件を回す
    for period in (periods or [None]):
        variants = []
        for spec in specs:
            cfg = bt.parse_config(spec.split())
            if period:
                cfg["history_period"] = period
            variants.append((spec, cfg))

        # 同じ期間なら株価も日経平均も共有できる
        target_period = period or variants[0][1]["history_period"]
        mixed = {c["history_period"] for _, c in variants}
        if len(mixed) > 1:
            print(f"⚠️ 条件ごとに期間が違います（{mixed}）。"
                  "株価の共有ができないので --periods で揃えることを推奨します。")

        print(f"\n===== 期間 {target_period} =====")
        hist_map, name_map, _ = bt.load_price_data(tickers, target_period)

        regime_cache, nikkei_cache = {}, {}
        for spec, cfg in variants:
            if cfg["use_market_regime_ppp"]:
                key = "ppp"
            elif cfg["use_market_regime_adx"]:
                key = "adx"
            elif cfg["use_market_regime"]:
                key = "sma"
            else:
                key = None
            if key and key not in regime_cache:
                regime_cache[key] = {
                    "ppp": bt.fetch_market_regime_ppp,
                    "adx": bt.fetch_market_regime_adx,
                    "sma": bt.fetch_market_regime,
                }[key](target_period)
            if cfg["use_rs_filter"] and "n" not in nikkei_cache:
                nikkei_cache["n"] = bt.fetch_nikkei_close(target_period)

            print(f"  検証中: {spec}")
            df, avg_bh = bt.run_backtest(
                cfg, hist_map, name_map,
                market_regime=regime_cache.get(key),
                nikkei_close=nikkei_cache.get("n"),
                verbose=False,
            )
            if df.empty:
                rows.append({"期間": target_period, "条件": spec, "件数": 0})
                continue
            r = df["return_pct"]
            rows.append({
                "期間": target_period,
                "条件": spec,
                "件数": len(df),
                "勝率%": round((r > 0).mean() * 100, 1),
                "PF": round(profit_factor(r), 2),
                "平均%": round(r.mean(), 2),
                "中央値%": round(r.median(), 2),
                "大負け件数": int((r <= -20).sum()),
                "保有日数": round(df["holding_days"].mean(), 1),
            })

    table = pd.DataFrame(rows)
    print("\n=== 比較結果 ===")
    print(table.to_string(index=False))

    out = BASE_DIR / "output" / "compare_result.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
