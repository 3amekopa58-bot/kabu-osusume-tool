"""
株価が上昇する前の会社の特徴を、多角的に実測する

各年末（2014〜2023）を判定時点とし、**その時点で開示済み**の財務
（EDINET の available_from 以降）と株価指標だけを使って、
**その後2年の騰落**との関係を見る。

⚠️ 後知恵の排除:
   ・財務は available_from（開示日）以降のものだけ
   ・株価水準そのものは使わない（分割調整で汚染。4.4-49）
   ・時価総額は 調整後株価×現在株式数 で復元（分割が打ち消し合う。4.4-50）
⚠️ 生存者バイアス: 途中で消えた銘柄は入っていない。上昇率は過大に出る。
"""
import csv, json
from pathlib import Path
import numpy as np, pandas as pd
from price_cache import fetch_histories

B=Path(__file__).parent
HOLD_Y=2          # 何年後の騰落を見るか
COHORTS=list(range(2014, 2024))

def main():
    tk=list(csv.DictReader(open(B/"universe.csv",encoding="utf-8-sig")))
    codes=[t["code"] for t in tk]; names={t["code"]:t["name"] for t in tk}
    hist=fetch_histories(codes, period="max")
    fin=json.load(open(B/"data"/"edinet_financials.json",encoding="utf-8"))["data"]
    fh=json.load(open(B/"data"/"fundamental_history.json",encoding="utf-8"))
    fh=fh.get("data",fh)
    shares={}
    for c,r in fh.items():
        x=[v for v in r if v.get("shares")]
        if x: shares[c]=float(sorted(x,key=lambda z:z.get("period_end",""))[-1]["shares"])

    px={}
    for c in codes:
        d=hist.get(c)
        if d is None or len(d)<500: continue
        i=d.index
        if getattr(i,"tz",None) is not None: i=i.tz_localize(None)
        px[c]=pd.DataFrame({"c":d["Close"].values,"v":d["Volume"].values},index=i)

    rows=[]
    for c in px:
        recs=fin.get(c)
        if not recs: continue
        rs=sorted(recs.values(), key=lambda r: r.get("period_end",""))
        s=px[c]
        for y in COHORTS:
            t=pd.Timestamp(f"{y}-12-31"); fwd=pd.Timestamp(f"{y+HOLD_Y}-12-31")
            if s.index[-1] < fwd or s.index[0] > t - pd.Timedelta(days=1200): continue
            i0=s.index.searchsorted(t); i1=s.index.searchsorted(fwd)
            if i0>=len(s) or i1>=len(s): continue
            p0=float(s["c"].iloc[i0]); p1=float(s["c"].iloc[i1])
            if p0<=0: continue
            # その時点で開示済みの最新決算（と3年前）
            avail=[r for r in rs if r.get("available_from") and pd.Timestamp(r["available_from"])<=t]
            if len(avail)<4: continue
            cur, old = avail[-1], avail[-4]
            sh=shares.get(c)
            def g(r,k):
                v=r.get(k)
                return float(v) if v not in (None,"") else None
            bps,eps=g(cur,"bps"),g(cur,"eps")
            rev,oi,ni,na=g(cur,"revenue"),g(cur,"ordinary_income"),g(cur,"net_income"),g(cur,"net_assets")
            rev0,ni0=g(old,"revenue"),g(old,"net_income")
            w=s.iloc[max(0,i0-250):i0+1]
            r1=w["c"].pct_change().dropna()
            rows.append({
                "code":c,"name":names[c],"年":y,
                "騰落%":(p1/p0-1)*100,
                "時価総額":p0*sh/1e8 if sh else np.nan,
                "PBR":p0/bps if bps and bps>0 else np.nan,
                "PER":p0/eps if eps and eps>0 else np.nan,
                "ROE%":ni/na*100 if ni is not None and na else np.nan,
                "経常利益率%":oi/rev*100 if oi is not None and rev else np.nan,
                "3年増収%":(rev/rev0-1)*100 if rev and rev0 and rev0>0 else np.nan,
                "3年増益%":(ni/ni0-1)*100 if ni and ni0 and ni0>0 else np.nan,
                "過去1年騰落%":(p0/float(s["c"].iloc[max(0,i0-250)])-1)*100,
                "ボラ%":r1.std()*np.sqrt(252)*100 if len(r1)>100 else np.nan,
                "売買代金":float((w["c"]*w["v"]).tail(60).mean())/1e8,
            })
    d=pd.DataFrame(rows)
    print(f"対象: {len(d):,}件（{d['code'].nunique()}銘柄 × {len(COHORTS)}コホート）")
    print(f"{HOLD_Y}年後の騰落: 中央値{d['騰落%'].median():+.1f}% / "
          f"2倍以上 {(d['騰落%']>=100).mean()*100:.1f}%\n")

    d["区分"]=pd.cut(d["騰落%"],[-100,-30,0,50,100,1e9],
                    labels=["-30%以下","下落","0〜50%","50〜100%","**2倍以上**"])
    cols=["時価総額","PBR","PER","ROE%","経常利益率%","3年増収%","3年増益%",
          "過去1年騰落%","ボラ%","売買代金"]
    agg=d.groupby("区分",observed=True)[cols].median().round(2)
    agg.insert(0,"件数",d.groupby("区分",observed=True).size())
    print("=== 2年後の騰落で分けた、判定時点の特徴（中央値）===")
    print(agg.to_string())

    print("\n=== 各指標の単調性（最下位区分 → 最上位区分）===")
    for k in cols:
        v=agg[k].tolist()
        mono = "単調↑" if all(v[i]<=v[i+1] for i in range(len(v)-1)) else \
               ("単調↓" if all(v[i]>=v[i+1] for i in range(len(v)-1)) else "—")
        print(f"  {k:<12} {' → '.join(f'{x:>8.1f}' for x in v)}   {mono}")

    print("\n=== 順位相関（指標 vs 2年後の騰落）===")
    for k in cols:
        s=d[[k,"騰落%"]].dropna()
        print(f"  {k:<12} {s[k].rank().corr(s['騰落%'].rank()):+.3f}  ({len(s):,}件)")

if __name__=="__main__":
    main()
