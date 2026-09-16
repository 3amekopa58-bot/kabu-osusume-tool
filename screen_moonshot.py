"""
過去に10倍以上になった銘柄と「同じプロファイル」の銘柄を全市場から拾う

⚠️ **これは予測ではない。** 過去の10倍株が上昇を始める前に持っていた
   特徴を実測し（4.4-59）、それに今当てはまる銘柄を機械的に並べるだけ。
   **当てはまっても大半は10倍にならない。**

実測した10倍株の姿（2015-09時点／その後10年で10倍以上になった57銘柄の中央値）:
   時価総額 92億 ／ PBR 0.24 ／ PER 4.8 ／ 売買代金 0.34億
   ROE 5.7% ／ 経常利益率 4.0%（どちらも全群で最低）
   出来高比 0.87（直近60日が過去200日より細い）／ 1年騰落 +6.8%

   ⚠️ **収益性が低いことが特徴**という点に注意。「良い会社」ではなく
      「安く放置されていた会社」が10倍になっている。
   ⚠️ チャート形状に判別力は無かった（順位相関すべて-0.15未満）ので
      テクニカルは補助的な絞り込みにしか使わない。

⚠️ 生存者バイアス：実測の母集団は「今も上場している銘柄」。
   同じ特徴で上場廃止になった銘柄は数えられていないので、
   **10倍になる確率は実測値より確実に低い。**

⚠️ 流動性：10倍株の売買代金の中央値0.34億は、このツールのユニバース基準
   （1億円）を下回る。**日次の推奨対象とは別物**で、終値で買える前提も
   成り立ちにくい。

使い方:
    python3 screen_moonshot.py [上位何件]
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

B = Path(__file__).parent
TICKERS = Path("/tmp/all_listed_tickers.json")
OUT = B / "output" / "moonshot_candidates.csv"

# 実測した10倍株の中央値を基準にする（4.4-59）
MAX_CAP_OKU   = 300      # 時価総額（10倍株の中央値92億・3〜10倍組303億）
MAX_PBR       = 0.5      # 10倍株の中央値0.24
MAX_PER       = 10.0     # 10倍株の中央値4.8
MIN_TURNOVER  = 0.05     # 売買代金の下限（億円）。これ未満は売買が成立しない
BATCH = 100
WAIT = 1.0

def fetch(batch, tries=3):
    for a in range(tries):
        try:
            return yf.download(batch, period="2y", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            if a == tries-1: return None
            time.sleep(20*(a+1))
    return None

def main():
    top = int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 40
    listed = json.loads(TICKERS.read_text(encoding="utf-8"))
    codes = [t["code"] for t in listed]; names = {t["code"]: t["name"] for t in listed}
    print(f"全上場{len(codes)}銘柄から、過去の10倍株と同じ特徴を持つ銘柄を探します")
    print(f"条件: 時価総額<{MAX_CAP_OKU}億 / PBR<{MAX_PBR} / PER<{MAX_PER} / 売買代金>{MIN_TURNOVER}億\n")

    rows, failed = [], []
    for i in range(0, len(codes), BATCH):
        b = codes[i:i+BATCH]
        d = fetch(b)
        if d is None:
            failed.extend(b); continue
        time.sleep(WAIT)
        for c in b:
            try:
                x = d[c].dropna(subset=["Close"])
                if len(x) < 120: continue
                px = float(x["Close"].iloc[-1])
                tv = float((x["Close"]*x["Volume"]).tail(60).mean())/1e8
                if tv < MIN_TURNOVER: continue
                v60 = float(x["Volume"].tail(60).mean())
                vprev = float(x["Volume"].iloc[-260:-60].mean()) if len(x)>260 else np.nan
                base = float(x["Close"].iloc[-245]) if len(x) >= 245 else float(x["Close"].iloc[0])
                r1 = (px/base-1)*100
                rows.append({"code":c,"name":names.get(c,c),"株価":px,"売買代金":tv,
                             "出来高比": v60/vprev if vprev else np.nan, "1年騰落%": r1})
            except Exception:
                pass
        if (i//BATCH) % 8 == 0:
            print(f"  {min(i+BATCH,len(codes))}/{len(codes)}銘柄 → 候補{len(rows)}")
    print(f"\n株価データ取得: {len(rows)}銘柄 / 失敗{len(failed)}\n")

    # PBR/PER は yfinance の info を、絞り込んだ銘柄にだけ問い合わせる
    df = pd.DataFrame(rows)
    df = df[df["株価"]*100 <= 1_000_000]      # 100株が予算内
    print(f"予算100万円以内: {len(df)}銘柄。財務を照会します…")
    recs=[]
    for n,(_,r) in enumerate(df.iterrows(),1):
        try:
            info = yf.Ticker(r["code"]).info
            cap = info.get("marketCap"); pbr = info.get("priceToBook")
            per = info.get("trailingPE")
            if not cap or not pbr: continue
            cap_oku = cap/1e8
            if pbr <= 0: continue              # 債務超過は除外
            if cap_oku > MAX_CAP_OKU or pbr > MAX_PBR: continue
            if per is not None and (per <= 0 or per > MAX_PER): continue
            recs.append({**r.to_dict(), "時価総額億":round(cap_oku,0),
                         "PBR":round(pbr,2), "PER":round(per,1) if per else None,
                         "ROE%":round(info.get("returnOnEquity",0)*100,1) if info.get("returnOnEquity") else None,
                         "業種":info.get("sector")})
        except Exception:
            pass
        if n % 50 == 0: print(f"  {n}/{len(df)} 照会済 → 該当{len(recs)}")
        time.sleep(0.3)

    out = pd.DataFrame(recs)
    if len(out)==0:
        print("該当なし"); return
    out = out.sort_values(["PBR","時価総額億"])
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n=== 該当 {len(out)}銘柄（PBRの低い順）===")
    print(out.head(top).to_string(index=False))
    print(f"\n保存: {OUT}")
    print("\n⚠️ これは予測ではなく『過去の10倍株と同じ特徴の銘柄』の一覧。")
    print("   大半は10倍にならず、生存者バイアスを考えると確率は実測より低い。")

if __name__ == "__main__":
    main()
