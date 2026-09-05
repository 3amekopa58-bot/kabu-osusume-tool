"""
印（小型・割安・押し目帯）の数ごとの成績を再現する

⚠️ **なぜ作ったか（2026-09-05）**
   `notify.py` の `MARK_STATS`（0個 PF1.31 / 1個 1.92 / 2個 2.36 / 3個 8.47）は
   4.4-33 でアドホックに計算されており、**再現用のスクリプトが無かった**。
   ⭐︎（条件充足数）と違い、印には**実測成績を通知に併記している**
   （PF8.47まで表示する）。**通知で最も強い主張をしている数字が
   取り直せない**のは危ういので、再現できるようにする（4.4-56 の棚卸し）。

印の定義（`notify.py` の `count_marks` と `screen.py` に合わせる）:
  ・小型     … 時価総額 300億円未満（KATAYAMA_SMALL_CAP_OKU）
  ・割安     … PBR 0.5未満（KABU1000_PBR_TIERS の最も緩い帯）
  ・押し目帯 … 日経が直近250日高値から -15〜-8% **かつ** ADXが強い
               （⚠️ ADX条件込みが `market_dip_band` の定義。4.4-31）
  ⚠️ カップは含めない（新高値と組み合わせたときだけ効く。4.4-19）

後知恵の排除:
  ・PBR・時価総額は EDINET の `available_from`（開示日）以降のものだけ使う
  ・時価総額 ＝ **調整後株価 × その時点の発行済株式数 × 累積分割倍率**
    ではなく、`調整後株価 × 現在の株式数` で近似する（分割が打ち消し合う。
    4.4-50/4.4-55 と同じ扱い）。⚠️ 増資・自社株買いのぶんは残る

使い方:
    python3 analyze_marks.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (DIP_BAND_LOW, DIP_BAND_HIGH, DIP_PEAK_WINDOW,
                      fetch_market_regime_adx, fetch_nikkei_close)
from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
SMALL_CAP_OKU = 300      # screen.py の KATAYAMA_SMALL_CAP_OKU
PBR_CHEAP = 0.5          # screen.py の KABU1000_PBR_TIERS の最も緩い帯

SUBPERIODS = [
    ("第1期 2000-01〜2010-03", "2000-01-01", "2010-03-31"),
    ("第2期 2010-03〜2018-01", "2010-04-01", "2018-01-31"),
    ("第3期 2018-01〜2026-08", "2018-02-01", "2026-12-31"),
]


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def summarize(x):
    return {"件数": len(x), "勝率%": round((x > 0).mean() * 100, 1),
            "平均%": round(x.mean(), 2), "PF": round(pf(x), 2)}


def load_fundamentals():
    """銘柄 -> [(available_from, bps)] と 銘柄 -> 現在の発行済株式数"""
    bps_hist, shares = {}, {}
    for fn in ("fundamental_history.json", "edinet_financials.json"):
        p = BASE_DIR / "data" / fn
        if not p.exists():
            continue
        raw = json.load(open(p, encoding="utf-8"))
        data = raw.get("data", raw)
        for code, recs in data.items():
            rows = list(recs.values()) if isinstance(recs, dict) else recs
            got = []
            for r in rows:
                af, b = r.get("available_from"), r.get("bps")
                if af and b:
                    got.append((pd.Timestamp(af), float(b)))
                if r.get("shares"):
                    shares.setdefault(code, [])
                    shares[code].append((r.get("period_end", ""),
                                         float(r["shares"])))
            if got:
                bps_hist.setdefault(code, []).extend(got)
    for code in bps_hist:
        bps_hist[code] = sorted(set(bps_hist[code]))
    latest_shares = {c: sorted(v)[-1][1] for c, v in shares.items() if v}
    return bps_hist, latest_shares


def main():
    tr = load_trades()
    codes = sorted(tr["code"].unique())
    hist = fetch_histories(codes, period="max")
    bps_hist, shares = load_fundamentals()
    print(f"BPSの履歴がある銘柄 {len(bps_hist)} / 株式数がある銘柄 {len(shares)}")

    nk = fetch_nikkei_close("max")
    adx = fetch_market_regime_adx("max")
    for s in (nk, adx):
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    # 押し目帯：日経が直近250日高値から -15〜-8%（backtest.py と同じ式）
    nk_dd = (nk / nk.rolling(DIP_PEAK_WINDOW, min_periods=20).max() - 1) * 100
    dip = (nk_dd > DIP_BAND_LOW) & (nk_dd <= DIP_BAND_HIGH)
    # ⚠️ market_dip_band は「押し目帯 かつ ADXが強い」（4.4-31）
    dip_band = dip & adx.reindex(dip.index, method="ffill").fillna(False)

    closes = {}
    for code in codes:
        h = hist.get(code)
        if h is None:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        closes[code] = c

    rows = []
    for _, t in tr.iterrows():
        code, ed = t["code"], t["entry_date"]
        c = closes.get(code)
        bh = bps_hist.get(code)
        sh = shares.get(code)
        if c is None or not bh or not sh:
            continue
        # エントリー日**より前**に開示された最新のBPSだけ使う（後知恵の排除）
        avail = [b for d, b in bh if d <= ed]
        if not avail or avail[-1] <= 0:
            continue
        pos = c.index.searchsorted(ed)
        if pos >= len(c.index):
            continue
        px = float(c.iloc[pos])
        pbr = px / avail[-1]
        cap_oku = px * sh / 1e8
        n = int(cap_oku < SMALL_CAP_OKU) + int(pbr < PBR_CHEAP) \
            + int(bool(dip_band.reindex([ed], method="ffill").fillna(False).iloc[0]))
        rows.append({"entry_date": ed, "ret": t["return_pct"], "印の数": n,
                     "小型": cap_oku < SMALL_CAP_OKU, "割安": pbr < PBR_CHEAP,
                     "押し目帯": bool(dip_band.reindex([ed], method="ffill")
                                  .fillna(False).iloc[0])})
    d = pd.DataFrame(rows)
    print(f"印を再現できたトレード: {len(d):,}件"
          f"（全{len(tr):,}件。BPS・株式数がある銘柄に限られる）")
    print(f"※4.4-33 の記録では 5,768件\n")

    print("=== 印の数ごとの成績（notify.py の MARK_STATS の出どころ）===")
    out = []
    for n, g in d.groupby("印の数"):
        out.append({"印の数": f"{n}個", **summarize(g["ret"])})
    print(pd.DataFrame(out).set_index("印の数").to_string())
    print(f"  全体: {summarize(d['ret'])}")
    print("  ※記録: 0個 3,032件 49.1% PF1.31 / 1個 1,639件 53.4% 1.92 /"
          " 2個 966件 59.3% 2.36 / 3個 128件 77.3% 8.47\n")

    print("=== 単独効果 ===")
    out = []
    for k in ["小型", "割安", "押し目帯"]:
        out.append({"印": k, **summarize(d[d[k]]["ret"])})
    print(pd.DataFrame(out).set_index("印").to_string())
    print("  ※記録: 押し目帯 573件 75.2% 6.16 / 割安 1,375件 57.6% 2.12 /"
          " 小型 1,736件 54.5% 2.06\n")

    print("=== ⚠️ 重複しない3期間（印に成績を併記する根拠が保てているか）===")
    out = []
    for label, lo, hi in SUBPERIODS:
        s = d[(d["entry_date"] >= pd.Timestamp(lo))
              & (d["entry_date"] <= pd.Timestamp(hi))]
        row = {"期間": label}
        for n in range(4):
            g = s[s["印の数"] == n]["ret"]
            row[f"{n}個"] = round(pf(g), 2) if len(g) >= 30 else None
            row[f"{n}個件数"] = len(g)
        out.append(row)
    print(pd.DataFrame(out).set_index("期間").to_string())
    print("\n  ⚠️ 3期間すべてで単調でなければ、"
          "⭐︎と同じく成績の併記をやめる根拠になる（4.4-37 と同じ判断）")


if __name__ == "__main__":
    main()
