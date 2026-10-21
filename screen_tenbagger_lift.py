"""10倍株が生まれやすい場所を、lift で確認できた軸だけで絞る（4.4-81）。

⚠️ **これは予測ではない。** lift（基準率に対する倍率）が約3倍の場所を示すだけ。
   基準率1.4%の3倍＝**4.2%**。100銘柄選んで96銘柄は10倍にならない。
   さらに生存者バイアス未反映なので実際はこれより低い（4.4-9）。

4.4-72 の条件（PBR<0.5・PER<15）は**使わない**。
分割の補正後にPBR・PERの判別力が消えたため（4.4-80）。

使う軸（`analyze_tenbagger_lift.py` で3窓すべて lift>1 を確認）:

| 軸 | 向き | 単独lift中央値 |
|---|---|---|
| 売買代金 | 細い | 3.3倍 |
| 高値からの下落 | 高値から離れている | 3.3倍 |
| ボラ | 荒い | 2.5倍 |

⚠️ 1年騰落は「高値からの下落」と順位相関0.66で同じものなので入れない。
⚠️ 軸を足してもliftは3倍前後で止まる（4.4-81）。多くても3軸まで。
⚠️ 時価総額は売買代金と順位相関+0.851で同じもの（4.4-50）。予算条件で代替する。

使い方: ./venv/bin/python screen_tenbagger_lift.py [各軸の分位(既定35)]
出力  : output/tenbagger_lift_candidates.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent
TICKERS = B / "data" / "all_listed_tickers.json"
OUT = B / "output" / "tenbagger_lift_candidates.csv"
BUDGET = 1_000_000
MIN_TURNOVER = 0.03      # 億円。これ未満は売買が成立しない
BATCH, WAIT = 100, 1.0


def fetch(batch, tries=3):
    for _ in range(tries):
        try:
            return yf.download(batch, period="2y", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            time.sleep(20)
    return None


def main() -> None:
    q = (float(sys.argv[1]) if len(sys.argv) > 1 else 35.0) / 100
    tick = json.loads(TICKERS.read_text(encoding="utf-8"))
    codes = [t["code"] for t in tick]
    names = {t["code"]: t["name"] for t in tick}
    print(f"全上場{len(codes)}銘柄。各軸の下位/上位{q*100:.0f}%で絞ります")
    print("⚠️ 予測ではない。lift約3倍＝確率4.2%。96%は10倍にならない\n")

    rows = []
    for i in range(0, len(codes), BATCH):
        b = codes[i:i + BATCH]
        d = fetch(b)
        if d is None:
            continue
        time.sleep(WAIT)
        for c in b:
            try:
                x = d[c].dropna(subset=["Close"])
                if len(x) < 260:
                    continue
                px = float(x["Close"].iloc[-1])
                if px * 100 > BUDGET:
                    continue
                tv = float((x["Close"] * x["Volume"]).tail(60).mean()) / 1e8
                if tv < MIN_TURNOVER:
                    continue
                hi = float(x["Close"].tail(250).max())
                v60 = float(x["Volume"].tail(60).mean())
                vp = float(x["Volume"].iloc[-260:-60].mean())
                rows.append({
                    "code": c, "name": names.get(c, c), "株価": round(px, 1),
                    "売買代金": round(tv, 3),
                    "高値からの下落%": round((px / hi - 1) * 100, 1),
                    "ボラ%": round(float(x["Close"].pct_change().tail(60).std() * 100), 2),
                    "出来高比": round(v60 / vp, 2) if vp else None,
                    "1年騰落%": round((px / float(x["Close"].iloc[-245]) - 1) * 100, 1),
                })
            except Exception:
                pass
        if (i // BATCH) % 8 == 0:
            print(f"  {min(i+BATCH, len(codes))}/{len(codes)} → {len(rows)}銘柄")

    df = pd.DataFrame(rows)
    if df.empty:
        print("該当なし")
        return
    print(f"\n株価・予算・流動性で絞った母集団 {len(df):,}銘柄")

    # 3軸で絞る。しきい値は母集団の分位点（固定値だと市場全体の変動で意味が変わる）
    th_tv = df["売買代金"].quantile(q)
    th_dd = df["高値からの下落%"].quantile(q)         # 下位＝より下落
    th_vo = df["ボラ%"].quantile(1 - q)               # 上位＝より荒い
    print(f"  しきい値: 売買代金≤{th_tv:.3f}億 / 高値からの下落≤{th_dd:.1f}% / "
          f"ボラ≥{th_vo:.2f}%")
    m = ((df["売買代金"] <= th_tv) & (df["高値からの下落%"] <= th_dd)
         & (df["ボラ%"] >= th_vo))
    out = df[m].copy()
    # 3軸の順位を平均して並べる（どれか1つに偏らないよう順位で揃える）
    r = (out["売買代金"].rank(pct=True)
         + out["高値からの下落%"].rank(pct=True)
         + (1 - out["ボラ%"].rank(pct=True))) / 3
    out["合致度"] = r.round(3)
    out = out.sort_values("合致度")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\n=== 該当 {len(out)}銘柄（3軸すべてを満たす・合致度の高い順）===")
    print(out.head(20).to_string(index=False))
    print(f"\n保存: {OUT}")
    print(f"\n⚠️ 該当{len(out)}銘柄のうち、過去の実績から期待できる10倍株は"
          f"5年で約{len(out)*0.042:.0f}銘柄（lift3倍・基準率1.4%として）。")
    print("   残りは10倍になりません。生存者バイアスを考えるとさらに少ない。")


if __name__ == "__main__":
    main()
