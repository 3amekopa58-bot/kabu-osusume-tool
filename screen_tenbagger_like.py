"""過去の10倍株が上昇する前の姿に、今どれだけ似ているかで並べる。

⚠️ **これは予測ではない。** 過去34件（2014-2026、3窓）の10倍株が
   上昇前に持っていた数値の型に近い銘柄を機械的に並べるだけ。
   基準率は窓あたり0.2〜1.3%で、**大半は10倍にならない**。
   しかも上場廃止組が欠けた生存者バイアス込みなので実際はもっと低い（4.4-9）。

`analyze_tenbagger_profile.py fund` で測った判別力に基づく:

| 軸 | 3窓の順位相関 | 扱い |
|---|---|---|
| 時価総額 | -0.388 / -0.189 / -0.326 | **採用**（小さいほど良い） |
| PBR | -0.314 / -0.154 / -0.633 | **採用** |
| PER | -0.235 / -0.226 / -0.445 | **採用** |
| 売買代金 | -0.331 / -0.058 / -0.294 | **採用**（ただし4.4-63の低位株交絡あり） |
| ROE・利益率・増収率 | 向きが揃わない | **使わない**（表示のみ） |
| テクニカル各種 | 揃っても相関0.02〜0.18 | **使わない**（表示のみ） |

⚠️ **業績で絞ってはいけない。** 10倍株の上昇前は ROE 2.79%・経常利益率1.03%・
   増収率 -11.65% と、平均的な銘柄（6.61% / 6.12% / -1.67%）より**悪かった**。
   「good な会社を探す」という直感とは逆になる。

使い方: ./venv/bin/python screen_tenbagger_like.py [並列数]
出力  : output/tenbagger_like.csv
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
OUT = B / "output" / "tenbagger_like.csv"

# 予備選抜。ここは緩めに取り、最終的な並びは類似度スコアで決める
MAX_CAP_OKU, MAX_PBR, MAX_PER, MIN_TURNOVER = 300.0, 0.5, 15.0, 0.05
BUDGET = 1_000_000

# 10倍株が上昇前に持っていた値（analyze_tenbagger_profile.py fund の実測中央値）
TARGET = {"時価総額億": 104.8, "PBR": 0.15, "PER": 4.24, "売買代金": 0.36}
REF_OTHER = {"ROE%": 6.61, "経常利益率%": 6.12, "増収率%": -1.67,
             "200日線乖離%": 5.71, "高値からの下落%": -9.90, "出来高比": 0.92}
REF_TEN = {"ROE%": 2.79, "経常利益率%": 1.03, "増収率%": -11.65,
           "200日線乖離%": 1.87, "高値からの下落%": -19.74, "出来高比": 0.95}

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
    tick = json.loads(TICKERS.read_text(encoding="utf-8"))
    codes = [t["code"] for t in tick]
    names = {t["code"]: t["name"] for t in tick}
    print(f"全上場{len(codes)}銘柄から、過去の10倍株の『上昇前の姿』に似た銘柄を探します")
    print(f"⚠️ 予測ではない。基準率は窓あたり0.2〜1.3%＝大半は10倍にならない\n")

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
                v60 = float(x["Volume"].tail(60).mean())
                vp = float(x["Volume"].iloc[-260:-60].mean())
                sma200 = float(x["Close"].tail(200).mean())
                hi = float(x["Close"].tail(250).max())
                rows.append({
                    "code": c, "name": names.get(c, c), "株価": px, "売買代金": tv,
                    "200日線乖離%": (px / sma200 - 1) * 100,
                    "高値からの下落%": (px / hi - 1) * 100,
                    "出来高比": v60 / vp if vp else np.nan,
                    "1年騰落%": (px / float(x["Close"].iloc[-245]) - 1) * 100,
                })
            except Exception:
                pass
        if (i // BATCH) % 8 == 0:
            print(f"  {min(i+BATCH, len(codes))}/{len(codes)} → 候補{len(rows)}")

    df = pd.DataFrame(rows)
    print(f"\n株価で絞った候補 {len(df)}銘柄。財務を照会します…")

    recs = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        try:
            info = yf.Ticker(r["code"]).info
            cap, pbr, per = info.get("marketCap"), info.get("priceToBook"), info.get("trailingPE")
            if not cap or not pbr or pbr <= 0:
                continue
            cap_oku = cap / 1e8
            if cap_oku > MAX_CAP_OKU or pbr > MAX_PBR:
                continue
            if per is not None and (per <= 0 or per > MAX_PER):
                continue
            roe = info.get("returnOnEquity")
            recs.append({**r.to_dict(), "時価総額億": round(cap_oku, 0),
                         "PBR": round(pbr, 2), "PER": round(per, 1) if per else None,
                         "ROE%": round(roe * 100, 1) if roe else None,
                         "経常利益率%": round(info.get("operatingMargins", 0) * 100, 1)
                         if info.get("operatingMargins") else None,
                         "増収率%": round(info.get("revenueGrowth", 0) * 100, 1)
                         if info.get("revenueGrowth") is not None else None,
                         "業種": info.get("sector")})
        except Exception:
            pass
        if n % 50 == 0:
            print(f"  {n}/{len(df)} 照会済 → 該当{len(recs)}")
        time.sleep(0.3)

    out = pd.DataFrame(recs)
    if out.empty:
        print("該当なし")
        return

    # --- 類似度スコア：判別力のある4軸だけ、順位で距離を取る ---
    # 生の値の差だと単位も分布も違うので順位に直してから比べる
    score = np.zeros(len(out))
    for col, tgt in [("時価総額億", TARGET["時価総額億"]), ("PBR", TARGET["PBR"]),
                     ("PER", TARGET["PER"]), ("売買代金", TARGET["売買代金"])]:
        s = out[col].astype(float)
        both = pd.concat([s, pd.Series([tgt])], ignore_index=True)
        r = both.rank(pct=True)
        score += (r.iloc[:-1].values - r.iloc[-1]) ** 2
    out["類似度"] = np.sqrt(score / 4)          # 0に近いほど似ている
    out = out.sort_values("類似度")

    cols = ["code", "name", "株価", "時価総額億", "PBR", "PER", "売買代金", "類似度",
            "ROE%", "経常利益率%", "増収率%", "200日線乖離%", "高値からの下落%",
            "出来高比", "1年騰落%", "業種"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n=== 該当 {len(out)}銘柄（類似度の高い順）===")
    print(out[cols[:8]].head(25).to_string(index=False, float_format="%.2f"))
    print(f"\n保存: {OUT}")
    print("\n【10倍株の上昇前 vs 平均的な銘柄】※絞り込みには使っていない")
    for k in REF_TEN:
        print(f"  {k:<14} 10倍株 {REF_TEN[k]:>7.2f} / 平均的 {REF_OTHER[k]:>7.2f}")
    print("\n⚠️ 業績で絞っていないのは、10倍株の上昇前の業績が**平均より悪かった**ため。")
    print("⚠️ テクニカルで絞っていないのは、3窓の順位相関が0.02〜0.18で実用外のため。")


if __name__ == "__main__":
    main()
