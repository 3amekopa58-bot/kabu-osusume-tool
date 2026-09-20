"""PBRを「フィルター」にしたとき実際に成績が上がるかを確かめる。

⚠️ **集計して差があることと、フィルターにして効くことは別。**
   4.4-5 では業種で「集計すると差がある」→「フィルターにすると悪化」
   となって不採用になった。4.4-65〜70 でPBRの集計上の差は確認したが、
   それは**採用の根拠にならない**。本節がその最終関門。

見るべき点:
  ① 期間ごとに成績が上がるか（3期間すべてで改善しなければ不採用）
  ② 件数がどれだけ減るか（減りすぎると実運用で機会が無くなる）
  ③ コスト控除後でも改善するか

⚠️ PBRが取れないトレードは判定から外す（58%欠損）。
   実運用の screen.py は yfinance の info からPBRを取るので
   欠損はほぼ無い。ここでの欠損は過去データの制約であって運用上の制約ではない。

使い方: ./venv/bin/python verify_pbr_as_filter.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
PERIODS = [("F1 2013-2017", 2013, 2017), ("F2 2018-2021", 2018, 2021),
           ("F3 2022-2026", 2022, 2026)]


def pf(s):
    w = s[s > 0].sum(); l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def load() -> pd.DataFrame:
    u = pd.read_csv(B / "output" / "trades_with_cost.csv")
    u["entry_date"] = pd.to_datetime(u["entry_date"])
    f = pd.read_csv(B / "output" / "segment_features.csv")
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    u = u.merge(f[["code", "entry_date", "PBR"]], on=["code", "entry_date"], how="left")
    u["年"] = u["entry_date"].dt.year
    u["実質"] = u["return_pct"] - (u["スプレッド%"] + u["インパクト%"])
    return u[u["PBR"].notna() & (u["PBR"] > 0)].copy()


def stats(s: pd.Series) -> dict:
    return {"件数": len(s), "中央値%": s.median(), "平均%": s.mean(),
            "勝率%": (s > 0).mean() * 100, "PF": pf(s)}


def main() -> None:
    d = load()
    print(f"PBRが取れたトレード {len(d):,}件（{d['年'].min()}〜{d['年'].max()}年）")
    print("※実質＝コスト控除後。判定は中央値\n")

    thresholds = [None, 2.0, 1.5, 1.0, 0.8, 0.5]
    print("=" * 78)
    print("① 全期間：しきい値ごとの成績（実質）")
    print("=" * 78)
    rows = []
    for t in thresholds:
        s = d if t is None else d[d["PBR"] < t]
        r = {"しきい値": "なし" if t is None else f"PBR<{t}"}
        r.update(stats(s["実質"]))
        r["残存率%"] = len(s) / len(d) * 100
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.2f"))

    print("\n" + "=" * 78)
    print("② 期間ごと：3期間すべてで改善しなければ不採用")
    print("=" * 78)
    for t in [2.0, 1.5, 1.0, 0.8, 0.5]:
        line, ok = [], True
        for lab, a, b in PERIODS:
            base = d[(d["年"] >= a) & (d["年"] <= b)]
            filt = base[base["PBR"] < t]
            if len(filt) < 50:
                line.append(f"{lab}:件数不足({len(filt)})"); ok = False; continue
            diff = filt["実質"].median() - base["実質"].median()
            line.append(f"{lab}:{diff:+.2f}pt({len(filt)}件)")
            ok &= diff > 0
        print(f"  PBR<{t}: " + " / ".join(line)
              + ("  → **3期間とも改善**" if ok else "  → 改善しない期間あり"))

    print("\n" + "=" * 78)
    print("③ 機会損失：フィルターで何件失うか")
    print("=" * 78)
    per_year = d.groupby("年").size()
    print(f"  フィルターなし: 年平均 {per_year.mean():.0f}件")
    for t in [1.0, 0.8, 0.5]:
        s = d[d["PBR"] < t].groupby("年").size()
        print(f"  PBR<{t}: 年平均 {s.mean():.0f}件"
              f"（{s.mean()/per_year.mean()*100:.0f}%に減る）")
    print("\n  ⚠️ 4.4-5 の業種フィルターは件数が276→176に減ったうえで")
    print("     成績も悪化して不採用になった。件数の減少は実運用では")
    print("     「買う日が無い」＝資金が遊ぶことを意味する。")


if __name__ == "__main__":
    main()
