"""
売買可能な銘柄ユニバースを作る（日経225からの拡大用）

tickers.csv は日経225の大型株だけだが、予算100万円・100株単位という制約では
むしろ中小型株のほうが買いやすい。一方で小型株には流動性の問題があり、
バックテストが前提にしている「終値で100株買える」「往復コスト0.2%」という
仮定が崩れる。そこで全上場3,822銘柄から以下の条件で絞り込む：

  ①買える      : 株価×100株 が予算以内
  ②流動性がある: 直近の平均売買代金が閾値以上
                  （1単元の売買が1日の出来高に占める割合が小さいこと）
  ③データがある: 十分な期間の株価データが取得できる
  ④汚染がない  : 1日で±80%超の異常な値動きがない（分割データ不整合の疑い）

yfinanceの一括ダウンロード（yf.download）を使うため、3,822銘柄でも
個別取得より大幅に速い。

使い方:
    python3 scripts/build_universe.py [出力先CSV]
      省略時は universe.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
ALL_TICKERS_JSON = Path("/tmp/all_listed_tickers.json")
DEFAULT_OUTPUT = BASE_DIR / "universe.csv"

BUDGET = 1_000_000
LOT_SIZE = 100
# 1日の平均売買代金がこれ未満の銘柄は、100株の売買でも値が動いてしまうため除外。
# 1単元（数十万円）が1日の売買代金の1%未満に収まる目安として1億円とした。
MIN_DAILY_TURNOVER = 100_000_000
MIN_DAYS = 120                  # スクリーニングに必要な最低営業日数
MAX_PLAUSIBLE_DAILY_MOVE = 0.8  # 1日で±80%超は分割データ不整合の疑い
BATCH_SIZE = 200


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    if not ALL_TICKERS_JSON.exists():
        print(f"{ALL_TICKERS_JSON} がありません。先に "
              "python3 scripts/build_all_listed_tickers.py を実行してください。")
        return

    listed = json.loads(ALL_TICKERS_JSON.read_text(encoding="utf-8"))
    codes = [t["code"] for t in listed]
    names = {t["code"]: t["name"] for t in listed}
    print(f"上場{len(codes)}銘柄について、直近6ヶ月の株価を一括取得します…")

    kept, stats = [], {"データ不足": 0, "予算オーバー": 0, "流動性不足": 0, "データ汚染": 0}
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        try:
            data = yf.download(batch, period="6mo", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print(f"  バッチ{i // BATCH_SIZE + 1} 取得失敗: {e}")
            continue

        for code in batch:
            try:
                d = data[code] if len(batch) > 1 else data
                d = d.dropna(subset=["Close"])
                if len(d) < MIN_DAYS:
                    stats["データ不足"] += 1
                    continue
                price = float(d["Close"].iloc[-1])
                if price * LOT_SIZE > BUDGET:
                    stats["予算オーバー"] += 1
                    continue
                turnover = float((d["Close"] * d["Volume"]).tail(20).mean())
                if turnover < MIN_DAILY_TURNOVER:
                    stats["流動性不足"] += 1
                    continue
                if (d["Close"].pct_change().abs() > MAX_PLAUSIBLE_DAILY_MOVE).any():
                    stats["データ汚染"] += 1
                    continue
                kept.append({"code": code, "name": names.get(code, code),
                             "price": round(price, 1),
                             "turnover_oku": round(turnover / 1e8, 2)})
            except Exception:
                stats["データ不足"] += 1

        print(f"  {min(i + BATCH_SIZE, len(codes))}/{len(codes)}銘柄 "
              f"→ 通過{len(kept)}銘柄")

    df = pd.DataFrame(kept).sort_values("turnover_oku", ascending=False)
    df[["code", "name"]].to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 絞り込み結果 ===")
    print(f"上場全銘柄        : {len(codes)}")
    for k, v in stats.items():
        print(f"  {k}で除外       : {v}")
    print(f"売買可能なユニバース: {len(df)}銘柄 → {out_path}")
    if len(df):
        print(f"\n売買代金の分布（億円/日）:")
        q = df["turnover_oku"].describe(percentiles=[0.25, 0.5, 0.75])
        print(f"  中央値 {q['50%']:.1f} / 上位25% {q['75%']:.1f} / 下位25% {q['25%']:.1f}")
        print(f"\n株価の分布（円）:")
        p = pd.DataFrame(kept)["price"].describe(percentiles=[0.5])
        print(f"  中央値 {p['50%']:,.0f}円（100株で{p['50%']*100:,.0f}円）")


if __name__ == "__main__":
    main()
