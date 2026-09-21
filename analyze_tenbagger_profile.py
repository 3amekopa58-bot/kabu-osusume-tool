"""10倍株が上昇する前の姿を、3つの軸（規模/割安・業績・テクニカル）で測る。

⚠️ 後知恵の排除:
   ・**株価の水準は使わない**（分割調整で「昔安い＝その後分割＝上がった」。4.4-49）
   ・時価総額は 調整後株価×現在株式数 で復元（分割が打ち消し合う。4.4-50）
   ・財務は `available_from`（実開示日。4.4-69）以降のものだけ
   ・特徴は**窓の開始時点**で測り、**その窓の騰落**を見る

⚠️ 生存者バイアス: 途中で消えた銘柄は入っていない。10倍率は過大に出る（4.4-9）。

⚠️ 判別力の判定は**重複しない複数の窓すべてで同じ向き**を要求する。
   1つの窓で相関が出ても採用しない（4.4-63 と同じ規律）。

使い方: ./venv/bin/python analyze_tenbagger_profile.py
出力  : output/tenbagger_profile.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories

B = Path(__file__).resolve().parent
# 重複しない5年窓。各窓の開始時点の姿 → その窓の騰落
WINDOWS = [("W1 2006-2010", "2006-01-01", "2010-12-31"),
           ("W2 2011-2015", "2011-01-01", "2015-12-31"),
           ("W3 2016-2020", "2016-01-01", "2020-12-31"),
           ("W4 2021-2026", "2021-01-01", "2026-08-31")]
MIN_PRE_DAYS = 250      # 開始時点までに必要な履歴

# ⚠️ ファンダは2013年より前が存在しない（4.4-64）。上のW1・W2では
#    時価総額/PBR/PER が測れず「2窓しか無い＝判定不能」になる。
#    財務のある範囲を3等分した窓も用意し、規模・割安を3窓で判定する。
#    引数 fund を付けるとこちらを使う。
WINDOWS_FUND = [("F1 2014-2017", "2014-01-01", "2017-12-31"),
                ("F2 2018-2021", "2018-01-01", "2021-12-31"),
                ("F3 2022-2026", "2022-01-01", "2026-08-31")]


def load_fundamentals() -> tuple:
    fin = json.loads((B / "data" / "edinet_financials.json").read_text(encoding="utf-8"))["data"]
    disc = {}
    jq = B / "data" / "jquants_summary.json"
    if jq.exists():
        for code, recs in json.loads(jq.read_text(encoding="utf-8"))["data"].items():
            for r in recs:
                if r.get("CurPerType") != "FY":
                    continue
                d, e = r.get("DiscDate"), r.get("CurFYEn")
                if d and e and pd.Timestamp(d) >= pd.Timestamp(e):
                    disc.setdefault((code, pd.Timestamp(e).date()), pd.Timestamp(d))
    return fin, disc


def avail(rec: dict, disc: dict, code: str) -> pd.Timestamp:
    pe = rec.get("period_end")
    if pe:
        base = pd.Timestamp(pe).date()
        for off in range(11):
            for sg in (1, -1):
                k = (code, base + pd.Timedelta(days=off * sg))
                if k in disc:
                    return disc[k]
    return pd.Timestamp(rec["available_from"])


def fin_asof(hist: dict, when: pd.Timestamp, disc: dict, code: str) -> list:
    """when 時点で開示済みの決算を古い順に返す（増収率などに前期が要る）。"""
    if not hist:
        return []
    items = sorted(hist.values(), key=lambda x: x.get("period_end") or "")
    return [h for h in items if h.get("available_from") and avail(h, disc, code) <= when]


def main() -> None:
    import sys
    global WINDOWS
    if "fund" in sys.argv[1:]:
        WINDOWS = WINDOWS_FUND
        print("財務のある範囲を3等分した窓で測ります\n")
    tk = list(csv.DictReader(open(B / "universe.csv", encoding="utf-8-sig")))
    codes = [t["code"] for t in tk]
    names = {t["code"]: t["name"] for t in tk}
    print(f"{len(codes)}銘柄の株価を用意中…")
    hist = fetch_histories(codes, period="max")
    fin, disc = load_fundamentals()
    print(f"財務のある銘柄 {len(fin)}\n")

    rows = []
    for wlab, ws, we in WINDOWS:
        ws_t, we_t = pd.Timestamp(ws), pd.Timestamp(we)
        for code in codes:
            h = hist.get(code)
            if h is None or len(h) == 0:
                continue
            h = h.copy()
            h.index = pd.to_datetime(h.index).tz_localize(None)
            c = h["Close"]
            pre = c.index[c.index < ws_t]
            post = c.index[(c.index >= ws_t) & (c.index <= we_t)]
            if len(pre) < MIN_PRE_DAYS or len(post) < 200:
                continue
            p0 = pre[-1]
            px = float(c.loc[p0])
            ret = (float(c.loc[post[-1]]) / float(c.loc[post[0]]) - 1) * 100

            # --- テクニカル（窓開始時点）---
            sma200 = float(c.loc[:p0].tail(200).mean())
            v = h["Volume"]
            v60 = float(v.loc[:p0].tail(60).mean())
            vprev = float(v.loc[:p0].tail(260).head(200).mean()) if len(pre) >= 260 else np.nan
            hi = float(c.loc[:p0].tail(250).max())
            r = {
                "窓": wlab, "code": code, "name": names.get(code, code),
                "リターン%": ret, "10倍以上": ret >= 900,
                "200日線乖離%": (px / sma200 - 1) * 100 if sma200 else np.nan,
                "1年騰落%": (px / float(c.loc[pre[-245]]) - 1) * 100 if len(pre) >= 245 else np.nan,
                "ボラ%": float(c.loc[:p0].pct_change().tail(60).std() * 100),
                "出来高比": v60 / vprev if vprev else np.nan,
                "高値からの下落%": (px / hi - 1) * 100 if hi else np.nan,
                "売買代金億": float((c * v).loc[:p0].tail(60).mean()) / 1e8,
            }
            # --- 規模/割安・業績（開示済みのみ）---
            fh = fin_asof(fin.get(code), ws_t, disc, code)
            if fh:
                f0 = fh[-1]
                bps, eps, sh = f0.get("bps"), f0.get("eps"), f0.get("shares")
                r["時価総額億"] = px * sh / 1e8 if sh else np.nan
                r["PBR"] = px / bps if bps and bps > 0 else np.nan
                r["PER"] = px / eps if eps and eps > 0 else np.nan
                ni, na = f0.get("net_income"), f0.get("net_assets")
                rev, oi = f0.get("revenue"), f0.get("ordinary_income")
                r["ROE%"] = ni / na * 100 if ni and na and na > 0 else np.nan
                r["経常利益率%"] = oi / rev * 100 if oi and rev and rev > 0 else np.nan
                if len(fh) >= 2:
                    p1 = fh[-2]
                    pr, pi = p1.get("revenue"), p1.get("ordinary_income")
                    r["増収率%"] = (rev / pr - 1) * 100 if rev and pr and pr > 0 else np.nan
                    r["増益率%"] = (oi / pi - 1) * 100 if oi and pi and pi > 0 else np.nan
            rows.append(r)

    d = pd.DataFrame(rows)
    tag = "_fund" if WINDOWS is WINDOWS_FUND else ""
    d.to_csv(B / "output" / f"tenbagger_profile{tag}.csv", index=False,
             encoding="utf-8-sig")
    print(f"{len(d):,}件（銘柄×窓）\n")

    print("=== 10倍以上になった件数 ===")
    g = d.groupby("窓")["10倍以上"].agg(["sum", "size"])
    g["率%"] = g["sum"] / g["size"] * 100
    print(g.to_string(float_format="%.2f"))

    AXES = {
        "規模・割安": ["時価総額億", "PBR", "PER", "売買代金億"],
        "業績": ["ROE%", "経常利益率%", "増収率%", "増益率%"],
        "テクニカル": ["200日線乖離%", "1年騰落%", "ボラ%", "出来高比", "高値からの下落%"],
    }
    print("\n=== その後の騰落との順位相関（窓ごと）===")
    print("※重複しない全窓で同じ向きでなければ判別力なしと扱う\n")
    for axis, cols in AXES.items():
        print(f"--- {axis} ---")
        for c in cols:
            if c not in d:
                continue
            vals = []
            for wlab, _, _ in WINDOWS:
                s = d[(d["窓"] == wlab) & d[c].notna()]
                if len(s) < 100:
                    vals.append(np.nan); continue
                vals.append(s[c].rank().corr(s["リターン%"].rank()))
            ok = all(v == v for v in vals) and (all(v > 0 for v in vals) or all(v < 0 for v in vals))
            mark = f"**{len(vals)}窓とも同じ向き**" if ok else "向きが揃わない"
            body = " / ".join("  n/a" if v != v else f"{v:+.3f}" for v in vals)
            print(f"  {c:<14} {body}   {mark}")
        print()


if __name__ == "__main__":
    main()
