"""取引コストを流動性に応じて見積もり、実質リターンを測る。

⚠️ **なぜ一律の%ではいけないか。** 4.4-66 で、成績が最も良い売買代金帯は
   当時の商いが1日1,400万円しかないと判明した。そこでのスリッページは
   1日17億円の銘柄とは桁が違う。一律コストだと、**最も費用のかかる帯を
   最も有望に見せてしまう**。

コストの内訳（往復）:
  ① 手数料      : 現在は現物ゼロの証券会社が多い。可変にする
  ② スプレッド  : 呼値（ティック）1つぶん。株価が安いほど%では大きい
  ③ 市場インパクト: c × 日次ボラ × sqrt(注文額 ÷ 1日の売買代金)
                    平方根則。注文が商いに対して大きいほど効く

⚠️ ②③は**推定**であって実測ではない。係数を変えた感度も出す。
   「正確なコスト」ではなく「コストを無視した場合との差」を見るための道具。

使い方: ./venv/bin/python analyze_trading_costs.py
出力  : output/trades_with_cost.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
ORDER_YEN = 1_000_000      # 1銘柄あたりの注文額（利用者の予算）
IMPACT_C = 0.5             # 平方根則の係数。0.3〜1.0で感度を見る


def tick_size(price: float) -> float:
    """東証の呼値（標準）。株価が安いほど%で見たスプレッドは大きい。"""
    if price < 3000:
        return 1.0
    if price < 5000:
        return 5.0
    if price < 30000:
        return 10.0
    return 50.0


def pf(s: pd.Series) -> float:
    w = s[s > 0].sum(); l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def build() -> pd.DataFrame:
    d = pd.read_csv(B / "output" / "_universe_max_trades.csv")
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    f = pd.read_csv(B / "output" / "segment_features.csv")
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    m = d.merge(f[["code", "entry_date", "売買代金億", "ボラ%"]],
                on=["code", "entry_date"], how="left")

    px = m["entry_price"]
    # ② スプレッド：往復で呼値1つぶん（片道は半値幅）
    m["スプレッド%"] = m["entry_price"].map(tick_size) / px * 100
    # ③ インパクト：平方根則
    adv_yen = m["売買代金億"] * 1e8
    part = np.clip(ORDER_YEN / adv_yen, 0, 1.0)      # 注文が商いに占める割合
    m["インパクト%"] = IMPACT_C * m["ボラ%"] * np.sqrt(part) * 2   # 往復
    return m


def report(m: pd.DataFrame, fee_pct: float, impact_c: float) -> dict:
    cost = fee_pct + m["スプレッド%"] + m["インパクト%"] * (impact_c / IMPACT_C)
    net = m["return_pct"] - cost
    return {"手数料%": fee_pct, "係数c": impact_c,
            "コスト中央値%": cost.median(), "コスト平均%": cost.mean(),
            "実質中央値%": net.median(), "実質平均%": net.mean(),
            "勝率%": (net > 0).mean() * 100, "PF": pf(net)}


def main() -> None:
    m = build()
    use = m.dropna(subset=["売買代金億", "ボラ%"])
    print(f"対象 {len(use):,}件（特徴量のある分。全{len(m):,}件）")
    print(f"注文額 {ORDER_YEN:,}円 / 平方根則の係数 c={IMPACT_C}\n")

    print("=== コストの内訳（中央値）===")
    print(f"  スプレッド   {use['スプレッド%'].median():.3f}%"
          f"（株価中央値 {use['entry_price'].median():,.0f}円）")
    print(f"  インパクト   {use['インパクト%'].median():.3f}%"
          f"（売買代金中央値 {use['売買代金億'].median():.2f}億）")
    print(f"  合計（手数料0） {(use['スプレッド%']+use['インパクト%']).median():.3f}%")

    print("\n=== 感度：仮定を変えたときの実質成績 ===")
    rows = [report(use, fee, c) for fee in (0.0, 0.1) for c in (0.3, 0.5, 1.0)]
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.3f"))
    print(f"\n（参考）コストを引かない場合: 中央値 {use['return_pct'].median():+.2f}% / "
          f"平均 {use['return_pct'].mean():+.2f}% / PF {pf(use['return_pct']):.2f}")

    print("\n=== 売買代金帯ごと（手数料0・c=0.5）===")
    cost = use["スプレッド%"] + use["インパクト%"]
    u = use.assign(コスト=cost, 実質=use["return_pct"] - cost,
                   帯=pd.qcut(use["売買代金億"], 4,
                              labels=["Q1少", "Q2", "Q3", "Q4多"]))
    g = u.groupby("帯", observed=True).agg(
        件数=("実質", "size"), 売買代金中央=("売買代金億", "median"),
        コスト中央=("コスト", "median"),
        コスト前中央=("return_pct", "median"), 実質中央=("実質", "median"),
        実質平均=("実質", "mean"))
    print(g.to_string(float_format="%.2f"))
    print("\n⚠️ コスト前で最も良かった帯が、コスト後にどうなるかを見ること。")

    u.to_csv(B / "output" / "trades_with_cost.csv", index=False,
             encoding="utf-8-sig")


if __name__ == "__main__":
    main()
