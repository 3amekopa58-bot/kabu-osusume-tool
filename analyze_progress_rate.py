"""
進捗率がトレード成績に効くのかを検証する

片山晃『5年で1億貯める株式投資』PART 5 より：

  進捗率 = 四半期決算の業績が通期予想の何%を達成しているか。
  四半期ごとに25%ずつ達成すれば通期100%が目安。
  「まあまあ好決算」＝第2四半期で50%をわずかに下回る49%のような
  **中途半端な数字**。著者はこれを空売りの対象とみなす。

  > このケースにおいては、売上高や営業利益より「進捗率」で見たほうが
  > わかりやすい（期初予想への期待はすでに株価に織り込まれているため）

つまり著者の主張は「**進捗率が目安を下回る銘柄は買うべきでない**」。
これを買い側のフィルターとして検証する。

⚠️ 先読みバイアス：各トレードのエントリー日時点で**すでに開示されていた**
   決算だけを使う（J-Quantsの DiscDate ＝ 実際の開示日）。

⚠️ 5年・10年に加えて**重複しない期間**でも一貫するかを見る。
   四半期増収率は入れ子の期間では良く見えたのに、重複しない期間で
   崩れた前例がある（REQUIREMENTS 4.4-21）。

使い方:
    python3 analyze_progress_rate.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
SUMMARY_PATH = BASE_DIR / "data" / "jquants_summary.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0
DEFAULT_TRADES = ("output/backtest_trades_pppsl8_newhigh_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")

_ORDER = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        print("J-Quantsのデータがありません。先に "
              "python3 scripts/build_jquants_financials.py を実行してください。")
        sys.exit(1)
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["data"]
    out = {}
    for code, rows in data.items():
        rs = [r for r in rows if r.get("DiscDate")]
        out[code] = sorted(rs, key=lambda r: r["DiscDate"])
    return out


def progress_at(rows: list, as_of: pd.Timestamp):
    """
    as_of 時点で開示済みの最新の四半期決算から進捗率を返す。

    通期決算（FY）が最新なら、その年度は終わっているので進捗率に意味は無い
    （None を返す）。1Q〜3Q のときだけ計算する。
    """
    usable = [r for r in rows if pd.Timestamp(r["DiscDate"]) <= as_of]
    if not usable:
        return None
    r = usable[-1]
    t = r.get("CurPerType")
    if t not in ("1Q", "2Q", "3Q"):
        return None
    expected = 25.0 * _ORDER[t]

    def rate(a_key, f_key):
        a, f = _num(r.get(a_key)), _num(r.get(f_key))
        if a is None or not f or f <= 0:
            return None
        return a / f * 100

    sales, op = rate("Sales", "FSales"), rate("OP", "FOP")
    if sales is None and op is None:
        return None
    return {"sales": sales, "op": op, "expected": expected, "period": t}


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    w = sub[sub["return_pct"] > 0]["return_pct"].sum()
    l = abs(sub[sub["return_pct"] <= 0]["return_pct"].sum())
    return {"件数": len(sub),
            "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
            "平均%": round(sub["return_pct"].mean(), 2),
            "PF": round(w / l, 2) if l else float("inf")}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / DEFAULT_TRADES
    summ = load_summary()
    print(f"J-Quantsデータ: {len(summ)}銘柄 / "
          f"{sum(len(v) for v in summ.values()):,}レコード")

    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    rows = []
    for _, t in df.iterrows():
        pr = progress_at(summ.get(t["code"], []), t["entry_date"])
        if pr is None:
            continue
        rows.append({"return_pct": t["return_pct"],
                     "sales": pr["sales"], "op": pr["op"],
                     "expected": pr["expected"], "period": pr["period"],
                     "entry_date": t["entry_date"], "code": t["code"]})
    r = pd.DataFrame(rows)
    if r.empty:
        print("判定できるトレードがありません")
        return
    # 目安との差（プラスなら進捗が先行、マイナスなら遅れ）
    r["gap_sales"] = r["sales"] - r["expected"]
    r["gap_op"] = r["op"] - r["expected"]
    print(f"進捗率を出せたトレード: {len(r):,}件"
          f"（全{len(df):,}件中）")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    def report(sub: pd.DataFrame, title: str):
        variants = {
            "条件なし": sub,
            "営業利益 進捗が目安以上": sub[sub["gap_op"] >= 0],
            "営業利益 進捗が目安未満": sub[sub["gap_op"] < 0],
            "　└ 目安を5pt以上下回る": sub[sub["gap_op"] < -5],
            "売上 進捗が目安以上": sub[sub["gap_sales"] >= 0],
            "売上 進捗が目安未満": sub[sub["gap_sales"] < 0],
        }
        out = {}
        for k, v in variants.items():
            st = stats(v.dropna(subset=["gap_op"]) if "営業利益" in k or "目安を5pt" in k
                       else v)
            if st and st["件数"] >= 20:
                out[k] = st
        if out:
            print(f"=== {title} ===")
            print(pd.DataFrame(out).T.to_string())
            print()

    report(r, f"全期間（{len(r):,}件）")

    print("=== 四半期別（1Qはノイズが大きい可能性がある）===")
    for t in ("1Q", "2Q", "3Q"):
        part = r[r["period"] == t]
        if len(part) < 50:
            continue
        a = stats(part[part["gap_op"] >= 0])
        b = stats(part[part["gap_op"] < 0])
        if a and b:
            print(f"  {t}: 目安以上 {a['件数']:>4}件 PF{a['PF']:.2f} 勝率{a['勝率%']:.1f}%"
                  f"  /  目安未満 {b['件数']:>4}件 PF{b['PF']:.2f} 勝率{b['勝率%']:.1f}%")
    print()

    print("=== 重複しない3期間 ===")
    edges = [r["entry_date"].quantile(x) for x in (1/3, 2/3)]
    def era(d):
        return 0 if d <= edges[0] else (1 if d <= edges[1] else 2)
    r = r.copy()
    r["era"] = r["entry_date"].apply(era)
    for i in range(3):
        part = r[r["era"] == i]
        if len(part) < 100:
            print(f"  第{i+1}期: サンプル不足（{len(part)}件）")
            continue
        span = f"{part['entry_date'].min().date()}〜{part['entry_date'].max().date()}"
        report(part, f"第{i+1}期 {span}")

    print("採用基準：進捗率で絞った組が条件なしを**全期間で**上回って初めて採用する。")
    print("（1期間だけの改善では不採用。REQUIREMENTS の採否基準）")


if __name__ == "__main__":
    main()
