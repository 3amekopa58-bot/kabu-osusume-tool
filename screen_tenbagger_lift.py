"""10倍株が生まれやすい場所を、lift で確認できた軸だけで絞る（4.4-81）。

⚠️ **これは予測ではない。** lift（基準率に対する倍率）が約3倍の場所を示すだけ。
   基準率1.4%の3倍＝**4.2%**。100銘柄選んで96銘柄は10倍にならない。
   さらに生存者バイアス未反映なので実際はこれより低い（4.4-9）。

4.4-72 の条件（PBR<0.5・PER<15）は**使わない**。
分割の補正後にPBR・PERの判別力が消えたため（4.4-80）。

使う軸は **売買代金が細いことだけ**（4.4-84 で1軸に減らした）。

⚠️ **2026-10-21に3軸から1軸へ変更した。** 以前は売買代金＋高値からの下落＋
   ボラの3軸だったが、後者2本は**暴落のliftも上げる**軸だった。
   暴落側を測らずに条件を作っていたため、リスクが増えていることに
   気づいていなかった（4.4-84）。

| 条件 | 10倍lift（3窓） | 崩壊70%lift（3窓） |
|---|---|---|
| 旧3軸（各35%） | 1.63 / 7.87 / 3.70 | 0.00 / 0.67 / **3.46** |
| 売買代金のみ 下位35% | 1.77 / 2.85 / 1.51 | 0.95 / 0.51 / 0.65 |
| **売買代金のみ 下位10%（既定）** | **3.32 / 3.32 / 1.94** | **0.00 / 0.36 / 0.45** |
| 売買代金のみ 下位5% | 4.74 / 4.94 / 2.75 | 0.00 / 0.71 / 0.00 |

**売買代金のみが、3窓すべてで「10倍lift>1 かつ 崩壊lift<1」を満たす唯一の軸。**

⚠️ **しきい値は厳しいほど良い。** 4.4-84 では35%しか試しておらず、
   **最も効果の薄い値を選んでいた**（10倍lift 1.77倍）。
   10%なら3.32倍、5%なら4.74倍。既定を**10%**にした
   （5%は窓あたり65銘柄まで絞られ、売買代金が極端に細くなって
   実際に売買できるか怪しいため。10%で5%の7割の効果が得られる）。

⚠️ PBR・PERを足すと**10倍株が消える**（W3期で10倍lift 0.47〜0.00）。
   割安株は壊れにくいが10倍にもならない＝**暴落回避と10倍株探しは両立しない**。
⚠️ 時価総額は売買代金と順位相関+0.851で同じもの（4.4-50）。予算条件で代替する。
⚠️ 高値からの下落・ボラ・出来高比は**表示するが絞り込みには使わない**
   （暴落liftを上げるため）。

使い方: ./venv/bin/python screen_tenbagger_lift.py [売買代金の下位何%(既定10)]
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
    q = (float(sys.argv[1]) if len(sys.argv) > 1 else 10.0) / 100
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

    # 絞るのは売買代金だけ（4.4-84）。しきい値は母集団の分位点
    # （固定値だと市場全体の変動で意味が変わる）
    th_tv = df["売買代金"].quantile(q)
    print(f"  しきい値: 売買代金 ≤ {th_tv:.3f}億（下位{q*100:.0f}%）")
    out = df[df["売買代金"] <= th_tv].copy()
    out = out.sort_values("売買代金")   # 細い順
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\n=== 該当 {len(out)}銘柄（売買代金の細い順）===")
    print("※高値からの下落・ボラ・出来高比は参考表示。絞り込みには使っていない")
    print(out.head(20).to_string(index=False))
    print(f"\n保存: {OUT}")
    print(f"\n⚠️ 該当{len(out)}銘柄のうち、過去の実績から期待できる10倍株は"
          f"5年で約{len(out)*0.014*3.3:.0f}銘柄（lift3.3倍・基準率1.4%として）。")
    print(f"   同じ期間に70%超の崩壊は約{len(out)*0.038*0.4:.0f}銘柄"
          f"（崩壊lift0.4倍＝基準率より少ない）。")
    print("   残りは10倍になりません。生存者バイアスを考えると当たりはさらに少ない。")
    print("   ⚠️ 1銘柄あたり数万円で20〜30銘柄に分散する前提の道具です。")


if __name__ == "__main__":
    main()
