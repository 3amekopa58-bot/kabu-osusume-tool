"""暴落を避けられるかを lift で検証する（4.4-84）。

4.4-81 で「裾の事象は順位相関では見えない。liftで測れ」と分かった。
同じ道具を**反対側の裾＝暴落**に当てる。

⚠️ **上場廃止は測れない。** 廃止銘柄の株価はyfinanceに存在せず（4.4-9で
   無作為60銘柄で再確認）、ユニバースは現存銘柄のみ。
   「廃止を避ける条件」は原理的に求められない。暴落で代替する。

⚠️ **最も重要な問い**：10倍株の条件（売買代金が細い・高値から離れている・
   荒い）は**暴落の条件でもあるのか**。同じなら「宝くじと地雷が同じ場所」で、
   10倍株リストは期待値の話では済まなくなる。

使い方: ./venv/bin/python analyze_crash_lift.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
MIN_N = 80
LEVELS = [(-30, "下落30%超"), (-50, "暴落50%超"), (-70, "崩壊70%超")]


def lift(d: pd.DataFrame, col: str, q: float, low: bool, tgt: str) -> list:
    out = []
    for lab, g in d.groupby("窓"):
        g = g[g[col].notna()]
        base = g[tgt].mean()
        if base == 0 or len(g) < 200:
            out.append(np.nan); continue
        th = g[col].quantile(q if low else 1 - q)
        s = g[g[col] <= th] if low else g[g[col] >= th]
        if len(s) < MIN_N:
            out.append(np.nan); continue
        out.append(s[tgt].mean() / base)
    return out


def main() -> None:
    d = pd.read_csv(B / "output" / "tenbagger_profile.csv")
    for lv, _ in LEVELS:
        d[f"暴落{lv}"] = d["リターン%"] <= lv
    print(f"対象 {len(d):,}件（銘柄×窓・5年窓）\n")
    print("=== 事象の基準率 ===")
    print(f"  10倍以上      {d['10倍以上'].mean()*100:>6.2f}%（{int(d['10倍以上'].sum())}件）")
    for lv, nm in LEVELS:
        c = d[f"暴落{lv}"]
        print(f"  {nm:<12}{c.mean()*100:>6.2f}%（{int(c.sum())}件）")

    AX = {"売買代金億": True, "高値からの下落%": True, "ボラ%": False,
          "1年騰落%": True, "出来高比": True, "時価総額億": True,
          "PBR": True, "PER": True, "ROE%": True, "経常利益率%": True}
    print("\n" + "=" * 86)
    print("=== 同じ軸で『10倍』と『暴落』の lift を並べる（各10%で切る）===")
    print("=" * 86)
    print(f"{'軸':<16}{'向き':<6}{'10倍のlift':<20}{'暴落50%超のlift':<22}{'判定'}")
    print("-" * 86)
    for col, low in AX.items():
        if col not in d.columns:
            continue
        up = [x for x in lift(d, col, 0.10, low, "10倍以上") if x == x]
        dn = [x for x in lift(d, col, 0.10, low, "暴落-50") if x == x]
        if len(up) < 2 or len(dn) < 2:
            continue
        mu, md = np.median(up), np.median(dn)
        side = "下位" if low else "上位"
        if mu > 1.3 and md > 1.3:
            v = "**両方上がる＝宝くじと地雷が同じ場所**"
        elif mu > 1.3 and md < 1.0:
            v = "**10倍だけ上がる＝good**"
        elif mu < 1.0 and md > 1.3:
            v = "避けるべき側"
        else:
            v = "はっきりしない"
        print(f"{col:<16}{side:<6}{' / '.join(f'{x:.1f}' for x in up):<20}"
              f"{' / '.join(f'{x:.1f}' for x in dn):<22}{v}")
        print(f"{'':<22}中央値{mu:.1f}倍{'':<12}中央値{md:.1f}倍")

    print("\n" + "=" * 86)
    print("=== 10倍株スクリーナーの3軸をすべて満たす場合（4.4-82の条件）===")
    print("=" * 86)
    d2 = d[d["窓"] != "W1 2006-2010"]
    for q in (0.25, 0.35):
        rows = []
        for lab, g in d2.groupby("窓"):
            m = ((g["売買代金億"] <= g["売買代金億"].quantile(q))
                 & (g["高値からの下落%"] <= g["高値からの下落%"].quantile(q))
                 & (g["ボラ%"] >= g["ボラ%"].quantile(1 - q)))
            s = g[m]
            if len(s) < 25:
                continue
            r = {"窓": lab, "該当": len(s)}
            for tgt, nm in [("10倍以上", "10倍"), ("暴落-50", "暴落50%超"),
                            ("暴落-70", "崩壊70%超")]:
                b = g[tgt].mean()
                r[f"{nm}率%"] = s[tgt].mean() * 100
                r[f"{nm}lift"] = s[tgt].mean() / b if b else np.nan
            rows.append(r)
        if rows:
            t = pd.DataFrame(rows)
            print(f"\n各軸{int(q*100)}%で絞った場合:")
            print(t.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
