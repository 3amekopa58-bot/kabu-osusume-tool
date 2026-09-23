"""BB逆張りに「大型株のみ」「悪材料を除外」の条件を足して検証する（4.4-76）。

4.4-75 で -3σ逆張りは**トレード単位では優秀・ポートフォリオでは-98%**だった。
原因は「利益は暴落時の集中日にあるが、4枠では取れない」。
利用者の指定で2つ絞り込む：

**大型株のみ**
  売買代金（20日平均）で判定する。時価総額との順位相関は +0.851 で
  実質同じものを測っており（4.4-50）、毎日・全期間で取れるので
  先読みも欠損も起きない。

**悪材料の除外**（直接は分からないので代理指標）
  ① 窓開け下落：前日終値から始値が -GAP% 超で飛んだ
     → 決算・事故・下方修正はたいてい窓を開ける
  ② 出来高急増：当日の出来高が20日平均の VOL倍 超
     → 材料が出ると出来高が跳ねる
  ③ 決算直後：J-Quantsの実開示日から EARN営業日以内
  ⚠️ いずれも**当日までの情報**だけで判定する（先読みなし）。
  ⚠️ 代理指標であって「悪材料の有無」そのものではない。
     良い材料での急騰も①②に引っかかるが、逆張りの買い場面では
     下落を伴うので実害は小さいと判断した。

使い方:
    ./venv/bin/python backtest_bb_reversion2.py [σ] [売買代金億] [gap%] [vol倍] [決算除外日数]
    例: ./venv/bin/python backtest_bb_reversion2.py 3 10 5 3 5
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories

B = Path(__file__).resolve().parent
TIME_STOP, STOP_LOSS = 60, -10.0
IMPACT_C, ORDER_YEN = 0.5, 1_000_000
PERIODS = [("第1期 2000-2008", 2000, 2008), ("第2期 2009-2017", 2009, 2017),
           ("第3期 2018-2026", 2018, 2026)]


def tick_size(p): return 1.0 if p < 3000 else (5.0 if p < 5000 else (10.0 if p < 30000 else 50.0))
def pf(s):
    w = s[s > 0].sum(); l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def load_disc() -> dict:
    """{銘柄: [開示日...]} 実際の決算発表日（J-Quants、2016年〜）。"""
    p = B / "data" / "jquants_summary.json"
    if not p.exists():
        return {}
    out = {}
    for code, recs in json.loads(p.read_text(encoding="utf-8"))["data"].items():
        ds = sorted({pd.Timestamp(r["DiscDate"]) for r in recs if r.get("DiscDate")})
        if ds:
            out[code] = ds
    return out


def run(hist, names, disc, ns, min_tv_oku, gap_pct, vol_mult, earn_days):
    trades = []
    for code, h in hist.items():
        if h is None or len(h) < 250:
            continue
        h = h.copy(); h.index = pd.to_datetime(h.index).tz_localize(None)
        if (h["Close"].pct_change().abs() > 0.8).any():
            continue
        c, o, v = h["Close"], h["Open"], h["Volume"]
        sma = c.rolling(20).mean(); sd = c.rolling(20).std()
        lower = sma - ns * sd
        vol20 = c.pct_change().rolling(20).std() * 100
        tv20 = (c * v).rolling(20).mean()
        vavg = v.rolling(20).mean()
        gap = (o / c.shift(1) - 1) * 100
        ed = disc.get(code, [])
        idx = c.index
        i, n = 20, len(c)
        while i < n - 1:
            if pd.isna(lower.iloc[i]) or c.iloc[i] >= lower.iloc[i]:
                i += 1; continue
            # --- 大型株フィルター ---
            if min_tv_oku and (pd.isna(tv20.iloc[i]) or tv20.iloc[i] / 1e8 < min_tv_oku):
                i += 1; continue
            # --- 悪材料の代理指標で除外 ---
            if gap_pct and pd.notna(gap.iloc[i]) and gap.iloc[i] <= -gap_pct:
                i += 1; continue
            if vol_mult and pd.notna(vavg.iloc[i]) and vavg.iloc[i] > 0 \
               and v.iloc[i] / vavg.iloc[i] >= vol_mult:
                i += 1; continue
            if earn_days and ed:
                d0 = idx[i]
                if any(0 <= (d0 - x).days <= earn_days * 1.5 for x in ed):
                    i += 1; continue
            e = i + 1; ep = float(o.iloc[e])
            if ep <= 0:
                i += 1; continue
            j, exit_p = e, None
            while j < n - 1:
                j += 1
                if float(c.iloc[j]) / ep - 1 <= STOP_LOSS / 100:
                    exit_p = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]); break
                if pd.notna(sma.iloc[j]) and float(c.iloc[j]) >= float(sma.iloc[j]):
                    exit_p = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]); break
                if j - e >= TIME_STOP:
                    exit_p = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]); break
            if exit_p is None:
                break
            ret = (exit_p / ep - 1) * 100
            sp = tick_size(ep) / ep * 100
            adv = float(tv20.iloc[i]) if pd.notna(tv20.iloc[i]) and tv20.iloc[i] > 0 else np.nan
            vv = float(vol20.iloc[i]) if pd.notna(vol20.iloc[i]) else 2.0
            part = min(ORDER_YEN / adv, 1.0) if adv == adv else 1.0
            cost = sp + IMPACT_C * vv * np.sqrt(part) * 2
            trades.append({"code": code, "name": names.get(code, code),
                           "entry_date": idx[e], "exit_date": idx[j], "保有日数": j - e,
                           "リターン%": ret, "コスト%": cost, "実質%": ret - cost,
                           "年": idx[e].year})
            i = j + 1
    return pd.DataFrame(trades)


def main() -> None:
    a = sys.argv[1:]
    ns = float(a[0]) if len(a) > 0 else 3.0
    min_tv = float(a[1]) if len(a) > 1 else 10.0
    gap = float(a[2]) if len(a) > 2 else 5.0
    vol = float(a[3]) if len(a) > 3 else 3.0
    earn = int(a[4]) if len(a) > 4 else 5
    tk = list(csv.DictReader(open(B / "universe.csv", encoding="utf-8-sig")))
    codes = [t["code"] for t in tk]; names = {t["code"]: t["name"] for t in tk}
    print(f"{len(codes)}銘柄の株価を用意中…")
    hist = fetch_histories(codes, period="max")
    disc = load_disc()
    print(f"決算開示日のある銘柄 {len(disc)}（2016年〜。それ以前は決算除外が効かない）")
    print(f"\n条件: -{ns}σ / 売買代金{min_tv}億以上 / 窓開け-{gap}%超を除外 / "
          f"出来高{vol}倍超を除外 / 決算後{earn}営業日を除外\n")
    d = run(hist, names, disc, ns, min_tv, gap, vol, earn)
    if len(d) == 0:
        print("トレード0件"); return
    # ⚠️ ファイル名に全条件を入れる。売買代金だけだと、悪材料フィルタの
    #    有無で別物なのに同じ名前になり、**上書きされる**（2026-09-22に実踏）。
    tag = f"{ns}s_tv{min_tv:g}_gap{gap:g}_vol{vol:g}_earn{earn}"
    out = B / "output" / f"bb_rev2_{tag}.csv"
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  → {out.name}")
    print(f"{len(d):,}件 / 年平均 {len(d)/27:.0f}件 / 平均保有 {d['保有日数'].mean():.1f}日")
    print(f"  コスト前: 中央値 {d['リターン%'].median():+.2f}% / 平均 {d['リターン%'].mean():+.2f}%")
    print(f"  実質: 中央値 {d['実質%'].median():+.2f}% / 平均 {d['実質%'].mean():+.2f}% "
          f"/ 勝率 {(d['実質%']>0).mean()*100:.1f}% / PF {pf(d['実質%']):.2f}"
          f"（往復コスト中央値 {d['コスト%'].median():.2f}%）")
    per = []
    for lab, x, y in PERIODS:
        s = d[(d["年"] >= x) & (d["年"] <= y)]
        per.append(f"{lab}: {s['実質%'].median():+.2f}%({len(s):,})" if len(s) >= 30 else f"{lab}: 件数不足({len(s)})")
    print(f"  期間別: " + " / ".join(per))
    g = d.groupby("entry_date").size()
    print(f"\n  シグナルの出る日 {len(g):,}日 / 1日あたり中央値{g.median():.0f}件 最大{g.max()}件")
    big = g[g >= 5].index
    if len(big):
        print(f"  1日5件以上の日 {len(big):,}日（全体の{d[d.entry_date.isin(big)].shape[0]/len(d)*100:.0f}%）"
              f" 平均{d[d.entry_date.isin(big)]['実質%'].mean():+.2f}% / "
              f"それ以外 {d[~d.entry_date.isin(big)]['実質%'].mean():+.2f}%")


if __name__ == "__main__":
    main()
