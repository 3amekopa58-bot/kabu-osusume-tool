"""本番 universe.csv の各銘柄を、現在の基準で判定し直して脱落理由を出す。

2026-09-13／09-19 の再構築で「前回入っていた銘柄が脱落する」現象が
続いたため、脱落が**基準割れ**なのか**取得できていないだけ**なのかを
銘柄ごとに切り分ける道具。build_universe.py と同じ基準を使う。

使い方: ./venv/bin/python diagnose_universe_drops.py
出力  : output/universe_drop_reason.csv
"""
import csv
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent
MIN_DAYS, BUDGET, MIN_TURNOVER_OKU, MAX_MOVE = 120, 1_000_000, 1.0, 0.8


def main() -> None:
    prev = list(csv.DictReader(open(B / "universe.csv", encoding="utf-8-sig")))
    codes = [r["code"] for r in prev]
    nm = {r["code"]: r["name"] for r in prev}
    print(f"本番 {len(codes)}銘柄を現在の基準で判定し直します")

    res = {}
    for i in range(0, len(codes), 100):
        b = codes[i:i + 100]
        try:
            d = yf.download(b, period="1y", group_by="ticker",
                            auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        time.sleep(1)
        for c in b:
            try:
                x = d[c].dropna(subset=["Close"])
                if len(x) == 0:
                    res[c] = ("取得失敗", 0, 0.0, 0.0)
                    continue
                px = float(x["Close"].iloc[-1])
                tv = float((x["Close"] * x["Volume"]).tail(20).mean()) / 1e8
                if len(x) < MIN_DAYS:
                    r = "データ不足"
                elif px * 100 > BUDGET:
                    r = "予算オーバー"
                elif tv < MIN_TURNOVER_OKU:
                    r = "流動性不足"
                elif (x["Close"].pct_change().abs() > MAX_MOVE).any():
                    r = "データ汚染"
                else:
                    r = "OK"
                res[c] = (r, len(x), px, tv)
            except Exception:
                res[c] = ("取得失敗", 0, 0.0, 0.0)
        print(f"  {min(i + 100, len(codes))}/{len(codes)}", flush=True)

    df = pd.DataFrame([{"code": c, "name": nm[c], "理由": v[0], "日数": v[1],
                        "株価": v[2], "売買代金億": v[3]} for c, v in res.items()])
    out = B / "output" / "universe_drop_reason.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n=== 理由の内訳 ===")
    print(df["理由"].value_counts().to_string())
    ng = df[df["理由"] != "OK"].sort_values(["理由", "売買代金億"])
    print(f"\n=== 基準を外れた {len(ng)}銘柄 ===")
    print(ng.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
