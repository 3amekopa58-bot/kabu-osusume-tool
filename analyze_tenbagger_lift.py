"""10倍株の特徴を「基準率に対する倍率（lift）」で探す。

⚠️ **なぜ順位相関をやめたか。** 4.4-72〜81 は順位相関で測っていたが、
   あれは「全3,000銘柄のリターンを説明できるか」を測る指標で、
   10倍株（年0.2〜1.3%の裾の事象）は**大多数の平凡な銘柄に薄められて消える**。
   実際、売買代金は順位相関 -0.33 で「弱い」と見えたが、
   lift で測ると**最細10%で基準率の4.4倍**、10倍株34件中15件がそこに集中していた。
   **問いは「リターンを説明できるか」ではなく
   「この条件の銘柄が10倍になる確率は何倍か」。**

判定:
  ① 測定可能な全窓で lift > 1（1つでも下回れば不採用）
  ② 該当件数が十分（MIN_N未満は判定しない）
  ③ 偶然の期待個数と比べる（指標を増やせば偶然の当たりが出る）

使い方: ./venv/bin/python analyze_tenbagger_lift.py [fund|price|both]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
MIN_N = 80          # 1窓あたりの最低該当件数
QS = [0.10, 0.25]   # 下位/上位 何%で切るか


def lift_by_window(d: pd.DataFrame, col: str, q: float, low: bool) -> tuple:
    """窓ごとの lift を返す。low=True なら下位q、False なら上位q。"""
    out = []
    for lab, g in d.groupby("窓"):
        g = g[g[col].notna()]
        base = g["10倍以上"].mean()
        if base == 0 or len(g) < 200:
            out.append((lab, np.nan, 0, 0))
            continue
        th = g[col].quantile(q if low else 1 - q)
        s = g[g[col] <= th] if low else g[g[col] >= th]
        if len(s) < MIN_N:
            out.append((lab, np.nan, len(s), 0))
            continue
        out.append((lab, s["10倍以上"].mean() / base, len(s),
                    int(s["10倍以上"].sum())))
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    files = []
    if which in ("fund", "both"):
        files.append(("ファンダ窓(3窓)", B / "output" / "tenbagger_profile_fund.csv"))
    if which in ("price", "both"):
        files.append(("株価窓(4窓)", B / "output" / "tenbagger_profile.csv"))

    for tag, f in files:
        if not f.exists():
            continue
        d = pd.read_csv(f)
        cols = [c for c in d.columns
                if c not in ("窓", "code", "name", "リターン%", "10倍以上")]
        print("=" * 76)
        print(f"{tag}  {len(d):,}件 / 10倍 {int(d['10倍以上'].sum())}件 "
              f"/ 基準率 {d['10倍以上'].mean()*100:.2f}%")
        print("=" * 76)
        hits = []
        for c in cols:
            for q in QS:
                for low in (True, False):
                    res = lift_by_window(d, c, q, low)
                    vals = [x[1] for x in res if x[1] == x[1]]
                    if len(vals) < 2:
                        continue
                    side = "下位" if low else "上位"
                    ok = all(v > 1 for v in vals)
                    body = " / ".join(f"{v:.1f}" for v in vals)
                    tot = sum(x[3] for x in res)
                    if ok:
                        hits.append((c, side, q, np.median(vals), tot, len(vals)))
                        print(f"  {c:<14}{side}{int(q*100):>3}%  lift {body}"
                              f"  中央値{np.median(vals):.1f}倍  10倍株{tot}件"
                              f"  **全{len(vals)}窓で>1**")
        print(f"\n  → 全窓でlift>1だった組み合わせ: {len(hits)}個 "
              f"/ 試した {len(cols)*len(QS)*2}個")
        # 偶然の期待：liftが1を超える確率を0.5と仮定（保守的）
        import math
        exp = sum(0.5 ** n for _, _, _, _, _, n in hits) if hits else 0
        print(f"  ⚠️ 指標を{len(cols)*len(QS)*2}通り試している。偶然でも"
              f"全窓一致するものは出るので、**lift の大きさと件数**で選ぶこと。")
        if hits:
            print(f"\n  lift中央値の大きい順:")
            for c, side, q, m, tot, n in sorted(hits, key=lambda x: -x[3])[:8]:
                print(f"    {c:<14}{side}{int(q*100):>3}%  中央値{m:>4.1f}倍"
                      f"  10倍株{tot:>3}件  {n}窓")
        print()


if __name__ == "__main__":
    main()
