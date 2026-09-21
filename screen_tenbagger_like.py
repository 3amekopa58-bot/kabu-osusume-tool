"""過去の10倍株が上昇する前の姿に、今どれだけ似ているかで全市場を並べる。

⚠️ **これは予測ではない。** `analyze_tenbagger_profile.py fund` で測った
   34件（2014-2026、3窓）の10倍株が上昇前に持っていた数値の型に
   近い銘柄を機械的に並べるだけ。基準率は窓あたり0.2〜1.3%で
   **大半は10倍にならない**。上場廃止組が欠けた生存者バイアス込みなので
   実際はもっと低い（4.4-9）。

⚠️ **yfinance の `.info` を使わない。** 2026-09-21 に全市場で試したところ
   レート制限で51分に50件も進まなかった（4.4-72）。
   PBR・PER・時価総額は **EDINET の BPS・EPS・株式数 × 現在株価** で出す。
   プロファイルの測定も EDINET なので**データ源が揃う**利点もある。

判別力に基づく扱い（4.4-72）:

| 軸 | 3窓の順位相関 | 扱い |
|---|---|---|
| 時価総額 / PBR / PER / 売買代金 | -0.15 〜 -0.63 | **類似度に使う** |
| ROE・利益率・増収率 | 向きが揃わない | **表示のみ** |
| テクニカル各種 | 揃っても 0.02〜0.18 | **表示のみ** |

⚠️ **業績で絞ってはいけない。** 10倍株の上昇前は ROE 2.79%・経常利益率1.03%・
   増収率 -11.65% と、平均的な銘柄より**悪かった**（4.4-72）。

使い方: ./venv/bin/python screen_tenbagger_like.py
出力  : output/tenbagger_like.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent
TICKERS = B / "data" / "all_listed_tickers.json"
FIN = B / "data" / "edinet_financials.json"
JQ = B / "data" / "jquants_summary.json"
OUT = B / "output" / "tenbagger_like.csv"

MAX_CAP_OKU, MAX_PBR, MAX_PER, MIN_TURNOVER = 300.0, 0.5, 15.0, 0.05
BUDGET = 1_000_000
# 10倍株が上昇前に持っていた値（analyze_tenbagger_profile.py fund の実測中央値）
TARGET = {"時価総額億": 104.8, "PBR": 0.15, "PER": 4.24, "売買代金": 0.36}
BATCH, WAIT = 100, 1.0


def load_fin() -> dict:
    """{銘柄: 最新の決算（開示済み）} を作る。開示日は実開示日を優先（4.4-69）。"""
    fin = json.loads(FIN.read_text(encoding="utf-8"))["data"]
    disc = {}
    if JQ.exists():
        for code, recs in json.loads(JQ.read_text(encoding="utf-8"))["data"].items():
            for r in recs:
                if r.get("CurPerType") != "FY":
                    continue
                d, e = r.get("DiscDate"), r.get("CurFYEn")
                if d and e and pd.Timestamp(d) >= pd.Timestamp(e):
                    disc.setdefault((code, pd.Timestamp(e).date()), pd.Timestamp(d))
    now = pd.Timestamp.now()
    out = {}
    for code, hist in fin.items():
        best = None
        for v in sorted(hist.values(), key=lambda x: x.get("period_end") or ""):
            if not v.get("available_from"):
                continue
            av = pd.Timestamp(v["available_from"])
            pe = v.get("period_end")
            if pe:
                base = pd.Timestamp(pe).date()
                for off in range(11):
                    hit = False
                    for sg in (1, -1):
                        k = (code, base + pd.Timedelta(days=off * sg))
                        if k in disc:
                            av, hit = disc[k], True
                            break
                    if hit:
                        break
            if av <= now:
                best = v
        if best:
            out[code] = best
    return out


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
    fin = load_fin()
    print(f"全上場{len(codes)}銘柄 / 財務のある銘柄 {len(fin)}")
    print("⚠️ 予測ではない。基準率は窓あたり0.2〜1.3%＝大半は10倍にならない\n")

    rows, no_fin = [], 0
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
                f = fin.get(c)
                if not f:
                    no_fin += 1
                    continue
                bps, eps, sh = f.get("bps"), f.get("eps"), f.get("shares")
                if not bps or bps <= 0 or not sh:
                    continue
                pbr = px / bps
                per = px / eps if eps and eps > 0 else None
                cap = px * sh / 1e8
                if pbr > MAX_PBR or cap > MAX_CAP_OKU:
                    continue
                if per is not None and per > MAX_PER:
                    continue
                if per is None:      # 赤字＝PERが出ない。10倍株にも多いので残す
                    pass
                ni, na = f.get("net_income"), f.get("net_assets")
                rev, oi = f.get("revenue"), f.get("ordinary_income")
                v60 = float(x["Volume"].tail(60).mean())
                vp = float(x["Volume"].iloc[-260:-60].mean())
                sma200 = float(x["Close"].tail(200).mean())
                hi = float(x["Close"].tail(250).max())
                rows.append({
                    "code": c, "name": names.get(c, c), "株価": round(px, 1),
                    "時価総額億": round(cap, 0), "PBR": round(pbr, 2),
                    "PER": round(per, 1) if per else None,
                    "売買代金": round(tv, 2),
                    "ROE%": round(ni / na * 100, 1) if ni and na and na > 0 else None,
                    "経常利益率%": round(oi / rev * 100, 1) if oi and rev and rev > 0 else None,
                    "200日線乖離%": round((px / sma200 - 1) * 100, 1),
                    "高値からの下落%": round((px / hi - 1) * 100, 1),
                    "出来高比": round(v60 / vp, 2) if vp else None,
                    "1年騰落%": round((px / float(x["Close"].iloc[-245]) - 1) * 100, 1),
                    "決算期": f.get("period_end"),
                })
            except Exception:
                pass
        if (i // BATCH) % 8 == 0:
            print(f"  {min(i+BATCH, len(codes))}/{len(codes)} → 該当{len(rows)}")

    out = pd.DataFrame(rows)
    if out.empty:
        print("該当なし")
        return
    print(f"\n財務が無くて判定できなかった銘柄: {no_fin}")

    # 類似度：判別力のある4軸だけ、順位での距離（単位も分布も違うので生値は使わない）
    sc = np.zeros(len(out))
    for col, tgt in [("時価総額億", TARGET["時価総額億"]), ("PBR", TARGET["PBR"]),
                     ("PER", TARGET["PER"]), ("売買代金", TARGET["売買代金"])]:
        s = out[col].astype(float)
        r = pd.concat([s, pd.Series([tgt])], ignore_index=True).rank(pct=True)
        sc += (r.iloc[:-1].values - r.iloc[-1]) ** 2
    out["類似度"] = (np.sqrt(sc / 4)).round(3)
    out = out.sort_values("類似度")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\n=== 該当 {len(out)}銘柄（10倍株の上昇前に近い順）===")
    print(out.head(25)[["code", "name", "株価", "時価総額億", "PBR", "PER",
                        "売買代金", "類似度"]].to_string(index=False))
    print(f"\n保存: {OUT}")
    print("\n⚠️ 業績・テクニカルは列にあるが**絞り込みに使っていない**。")
    print("   10倍株の上昇前の業績は平均より悪く（ROE2.79% vs 6.61%）、")
    print("   テクニカルの順位相関は0.02〜0.18で実用外だったため（4.4-72）。")


if __name__ == "__main__":
    main()
