"""
株価が倍になった銘柄の特徴を実測する

⚠️ 後知恵の排除（4.4-49 の教訓）:
   ・**株価の水準は使わない。** 分割で遡及調整されるため、
     「過去に安かった」＝「その後分割した」＝「その後上がった」になる
   ・時価総額は「調整後株価 × 現在の株式数」で復元（分割が打ち消し合う。4.4-50）
   ・特徴は**前の5年窓**で測り、**次の5年窓**の騰落を見る（判定時点で既知の情報だけ）

⚠️ 生存者バイアス: universe.csv は「今日存在する銘柄」なので、
   途中で上場廃止になった銘柄は入っていない。倍増率は過大に出る。

使い方: python3 analyze_multibagger.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import fetch_nikkei_close
from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
WINDOWS = [("2000-2005","2000-01-01","2004-12-31"),("2005-2010","2005-01-01","2009-12-31"),
           ("2010-2015","2010-01-01","2014-12-31"),("2015-2020","2015-01-01","2019-12-31"),
           ("2020-2026","2020-01-01","2026-12-31")]
MIN_DAYS = 400

def main():
    import csv
    tk=list(csv.DictReader(open(BASE_DIR/"universe.csv",encoding="utf-8-sig")))
    codes=[t["code"] for t in tk]; names={t["code"]:t["name"] for t in tk}
    hist=fetch_histories(codes, period="max")
    nk=fetch_nikkei_close("max")
    if getattr(nk.index,"tz",None) is not None: nk.index=nk.index.tz_localize(None)
    raw=json.load(open(BASE_DIR/"data"/"fundamental_history.json",encoding="utf-8"))
    fd=raw.get("data",raw); shares={}
    for c,recs in fd.items():
        r=[x for x in recs if x.get("shares")]
        if r: shares[c]=float(sorted(r,key=lambda z:z.get("period_end",""))[-1]["shares"])

    px={}
    for c in codes:
        d=hist.get(c)
        if d is None or len(d)<MIN_DAYS: continue
        idx=d.index
        if getattr(idx,"tz",None) is not None: idx=idx.tz_localize(None)
        px[c]=pd.DataFrame({"c":d["Close"].values,"v":d["Volume"].values}, index=idx)

    def traits(c, lo, hi):
        w=px[c][(px[c].index>=pd.Timestamp(lo))&(px[c].index<=pd.Timestamp(hi))]
        if len(w)<MIN_DAYS: return None
        r=w["c"].pct_change().dropna()
        sh=shares.get(c)
        nr=nk.reindex(w.index,method="ffill").pct_change()
        b=pd.concat([r,nr],axis=1).dropna()
        beta=idio=np.nan
        if len(b)>100 and b.iloc[:,1].var()>0:
            beta=b.iloc[:,0].cov(b.iloc[:,1])/b.iloc[:,1].var()
            cr=b.iloc[:,0].corr(b.iloc[:,1]); idio=1-cr**2 if pd.notna(cr) else np.nan
        return {"時価総額":float(w["c"].iloc[-1])*sh/1e8 if sh else np.nan,
                "売買代金":float((w["c"]*w["v"]).tail(60).mean())/1e8,
                "ボラ":r.std()*np.sqrt(252)*100, "日経β":beta, "個別要因":idio,
                "5年騰落":(float(w["c"].iloc[-1])/float(w["c"].iloc[0])-1)*100}

    T={lab:{c:t for c in px if (t:=traits(c,lo,hi))} for lab,lo,hi in WINDOWS}
    rows=[]
    for i in range(len(WINDOWS)-1):
        a,b=WINDOWS[i][0],WINDOWS[i+1][0]
        for c in T[a]:
            if c not in T[b]: continue
            r={"code":c,"name":names[c],"窓":b,"次の5年騰落":T[b][c]["5年騰落"]}
            for k in ["時価総額","売買代金","ボラ","日経β","個別要因"]:
                r[k]=T[a][c][k]
            r["前の5年騰落"]=T[a][c]["5年騰落"]
            rows.append(r)
    d=pd.DataFrame(rows).dropna(subset=["次の5年騰落"])
    print(f"対象: {len(d):,}件（銘柄×引き継ぎ4回）\n")

    d["区分"]=pd.cut(d["次の5年騰落"], [-100,-50,0,100,300,1e9],
                    labels=["半値以下","下落","2倍未満","2〜4倍","**4倍超**"])
    out=[]
    for g,s in d.groupby("区分",observed=True):
        row={"区分":g,"件数":len(s),"割合%":round(len(s)/len(d)*100,1)}
        for k in ["時価総額","売買代金","ボラ","日経β","個別要因","前の5年騰落"]:
            row[k]=round(float(s[k].median()),2)
        out.append(row)
    print("=== 次の5年の騰落で分けたときの、開始時点の特徴（中央値）===")
    print(pd.DataFrame(out).set_index("区分").to_string())
    print("\n※時価総額=億円 / 売買代金=億円・日 / ボラ=年率% / 前の5年騰落=%")

    print("\n=== 2倍以上になった銘柄の特徴（下位25%〜上位25%）===")
    win=d[d["次の5年騰落"]>=100]
    for k in ["時価総額","売買代金","ボラ","日経β","個別要因"]:
        q=win[k].quantile([.25,.5,.75])
        a=d[k].median()
        print(f"  {k:<8} {q[.25]:>8.2f} 〜 {q[.75]:>8.2f}（中央値{q[.5]:>7.2f}） 全体は{a:>7.2f}")

    print(f"\n=== 実際に4倍超になった例（直近の窓）===")
    ex=d[(d["窓"]=="2020-2026")&(d["次の5年騰落"]>=300)].nlargest(10,"次の5年騰落")
    for _,r in ex.iterrows():
        print(f"  {r['code']:<8}{r['name'].replace('株式会社','')[:20]:<22}"
              f"{r['次の5年騰落']:>8,.0f}%  開始時 時価総額{r['時価総額']:>7,.0f}億 "
              f"ボラ{r['ボラ']:>5.1f}%")

if __name__ == "__main__":
    main()
