"""4.4-64 で残った PBR帯 を、採用してよいかまで詰める。

区分検証エンジン（segment_lab.py）を通過しただけでは採用しない。
通過は「否定されなかった」であって「確かめられた」ではないため、
以下を追加で確認する:

  ① 期間の切り方を変えても向きが変わらないか（切り方への頑健性）
  ② 各期間の中でも単調か（全期間でならしただけではないか）
  ③ 他の交絡で説明できないか（時価総額・業種・流動性で層別）
  ④ 効果がノイズ幅を超えているか（ブートストラップ）

⚠️ **比較は中央値で行う（4.4-65）。** 平均だと 8105.T(+1,189%) の1件で
   「小型株では割安が不利」という逆の結論が出る。平均は参考として併記。

使い方: ./venv/bin/python verify_pbr.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


def load() -> pd.DataFrame:
    d = pd.read_csv(B / "output" / "_universe_max_trades.csv")
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    f = pd.read_csv(B / "output" / "segment_features.csv")
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    m = d.merge(f, on=["code", "entry_date"], how="inner")
    m = m[m["PBR"].notna() & (m["PBR"] > 0)].copy()
    m["年"] = m["entry_date"].dt.year
    sec = json.loads((B / "data" / "sectors.json").read_text(encoding="utf-8"))
    m["業種"] = m["code"].map(sec).fillna("Unknown")
    return m


def pf(s):
    w = s[s > 0].sum(); l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def q4(m, col="PBR"):
    return pd.qcut(m[col], 4, labels=["Q1", "Q2", "Q3", "Q4"])


def main() -> None:
    m = load()
    m["PBR帯4"] = q4(m)
    print(f"対象 {len(m):,}トレード / {m['code'].nunique()}銘柄 "
          f"/ {m['年'].min()}〜{m['年'].max()}年\n")

    # ① 期間の切り方への頑健性 -------------------------------------
    print("=" * 66)
    print("① 期間の切り方を変える（Q1−Q4 の差＝中央値。プラスなら割安が有利）")
    print("=" * 66)
    splits = {
        "3分割 13-17/18-21/22-26": [(2013, 2017), (2018, 2021), (2022, 2026)],
        "3分割 13-16/17-20/21-26": [(2013, 2016), (2017, 2020), (2021, 2026)],
        "2分割 13-19/20-26":      [(2013, 2019), (2020, 2026)],
        "4分割 13-15/16-18/19-22/23-26":
            [(2013, 2015), (2016, 2018), (2019, 2022), (2023, 2026)],
    }
    for name, ps in splits.items():
        out = []
        for a, b in ps:
            s = m[(m["年"] >= a) & (m["年"] <= b)]
            if len(s) < 100:
                out.append("件数不足"); continue
            g = s.groupby(q4(s), observed=True)["return_pct"].median()
            out.append(f"{g.get('Q1', np.nan) - g.get('Q4', np.nan):+.2f}")
        print(f"  {name:<32} {' / '.join(out)}")

    # ② 各期間の中での単調性 ---------------------------------------
    print("\n" + "=" * 66)
    print("② 各期間の中でも単調か（中央値。全期間でならしただけではないか）")
    print("=" * 66)
    for lab, a, b in [("F1 2013-2017", 2013, 2017), ("F2 2018-2021", 2018, 2021),
                      ("F3 2022-2026", 2022, 2026)]:
        s = m[(m["年"] >= a) & (m["年"] <= b)]
        g = s.groupby(q4(s), observed=True)["return_pct"].agg(["median", "size"])
        v = list(g["median"])
        mono = "単調" if all(x > y for x, y in zip(v, v[1:])) else "**単調でない**"
        print(f"  {lab}: " + " / ".join(f"{q}{x:+.2f}({n})" for q, x, n
                                        in zip(g.index, g["median"], g["size"]))
              + f"  → {mono}")

    # ③ 他の交絡で層別 ---------------------------------------------
    print("\n" + "=" * 66)
    print("③ 他の交絡で層別しても残るか（各層での Q1−Q4＝中央値）")
    print("=" * 66)
    for col, label in [("時価総額億", "時価総額"), ("売買代金億", "売買代金"),
                       ("ボラ%", "ボラティリティ")]:
        sub = m[m[col].notna()]
        if len(sub) < 400:
            print(f"  {label}: 件数不足"); continue
        sub = sub.assign(層=pd.qcut(sub[col], 4, labels=["最小", "小", "大", "最大"]))
        parts = []
        for lv, g in sub.groupby("層", observed=True):
            if len(g) < 100:
                parts.append(f"{lv}:件数不足"); continue
            q = g.groupby(q4(g), observed=True)["return_pct"].median()
            parts.append(f"{lv}:{q.get('Q1', np.nan) - q.get('Q4', np.nan):+.2f}")
        print(f"  {label}で層別  " + " / ".join(parts))

    print("\n  業種別（件数300以上）:")
    for s_, g in m.groupby("業種"):
        if len(g) < 300:
            continue
        q = g.groupby(q4(g), observed=True)["return_pct"].median()
        print(f"    {s_:<22} {q.get('Q1', np.nan) - q.get('Q4', np.nan):+6.2f}pt "
              f"({len(g)}件)")

    # ④ ノイズ幅との比較 --------------------------------------------
    print("\n" + "=" * 66)
    print("④ 効果がノイズ幅を超えているか（ブートストラップ2,000回）")
    print("=" * 66)
    a_ = m[m["PBR帯4"] == "Q1"]["return_pct"].values
    b_ = m[m["PBR帯4"] == "Q4"]["return_pct"].values
    obs = np.median(a_) - np.median(b_)
    diffs = [np.median(rng.choice(a_, len(a_), True)) -
             np.median(rng.choice(b_, len(b_), True)) for _ in range(2000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # 帰無分布：ラベルをシャッフルして同じ差を作る
    pool = np.concatenate([a_, b_])
    null = []
    for _ in range(2000):
        p = rng.permutation(pool)
        null.append(np.median(p[:len(a_)]) - np.median(p[len(a_):]))
    pval = (np.abs(null) >= abs(obs)).mean()
    print(f"  Q1−Q4 の実測差: {obs:+.2f}pt")
    print(f"  95%信頼区間: {lo:+.2f} 〜 {hi:+.2f}pt")
    print(f"  ラベルを入れ替えた帰無分布でこれ以上の差が出る確率: p={pval:.3f}")
    print(f"  → {'**ゼロを跨がない**' if lo > 0 else '**ゼロを跨ぐ＝差があると言えない**'}")


if __name__ == "__main__":
    main()
