"""
「四半期の増収率」と「年次の増収率」のどちらが効くのかを検証する

片山晃『5年で1億貯める株式投資』PART 6 は
**「四半期決算ごとに前年同期比『売上高10%増』が目安」**と書いている。
このツールの片山流モードは有価証券報告書＝**年次**の増収率で判定しており、
粒度が違っていた（REQUIREMENTS 4.4-18）。

J-Quants無料プランは約2年しか遡れず検証できなかったが、
`scripts/build_edinet_quarterly.py` で四半期報告書（2017〜2024年3月）と
半期報告書（2024年4月〜）から履歴を作ったので、ここで比べる。

⚠️ 先読みバイアス：四半期データの `available_from` は**実際の提出日**。
   エントリー日より後に提出された決算は使わない。

⚠️ 5年・10年・26年に加え、**重複しない期間**でも一貫するかを見る
   （入れ子の期間だけでは直近の相場に効いただけのものを弾けない）。

使い方:
    python3 analyze_quarterly_growth.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
QUARTERLY_PATH = BASE_DIR / "data" / "edinet_quarterly.json"
ANNUAL_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0

DEFAULT_TRADES = ("output/backtest_trades_pppsl8_newhigh_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")


def load_quarterly() -> dict:
    if not QUARTERLY_PATH.exists():
        print("四半期データがありません。先に "
              "python3 scripts/build_edinet_quarterly.py を実行してください。")
        sys.exit(1)
    data = json.loads(QUARTERLY_PATH.read_text(encoding="utf-8"))["data"]
    out = {}
    for code, docs in data.items():
        recs = [r for r in docs.values()
                if r.get("revenue_cur") and r.get("revenue_prior")]
        out[code] = sorted(recs, key=lambda r: r.get("available_from") or "")
    return out


def load_annual() -> dict:
    data = json.loads(ANNUAL_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted((list(v.values()) if isinstance(v, dict) else v),
                      key=lambda r: r["period_end"])
            for c, v in data.items()}


def quarterly_growth_at(recs: list, as_of: pd.Timestamp):
    """as_of 時点で提出済みの最新の四半期決算から、前年同期比の増収率を返す"""
    usable = [r for r in recs
              if r.get("available_from") and pd.Timestamp(r["available_from"]) <= as_of]
    if not usable:
        return None
    r = usable[-1]
    prev = r["revenue_prior"]
    if not prev or prev <= 0:
        return None
    return (r["revenue_cur"] - prev) / prev * 100


def annual_growth_at(recs: list, as_of: pd.Timestamp):
    usable = [r for r in recs if pd.Timestamp(r["available_from"]) <= as_of]
    if len(usable) < 2:
        return None
    a, b = usable[-1].get("revenue"), usable[-2].get("revenue")
    if a is None or not b or b <= 0:
        return None
    return (a - b) / b * 100


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
    q, a = load_quarterly(), load_annual()
    print(f"四半期データ: {len(q)}銘柄 / "
          f"{sum(len(v) for v in q.values()):,}レコード")

    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    rows = []
    for _, t in df.iterrows():
        qg = quarterly_growth_at(q.get(t["code"], []), t["entry_date"])
        ag = annual_growth_at(a.get(t["code"], []), t["entry_date"])
        if qg is None and ag is None:
            continue
        rows.append({"return_pct": t["return_pct"], "q": qg, "a": ag,
                     "entry_date": t["entry_date"], "code": t["code"]})
    r = pd.DataFrame(rows)
    both = r[r["q"].notna() & r["a"].notna()]
    print(f"対象トレード: {len(r):,}件（うち四半期・年次の両方が揃う {len(both):,}件）")
    if both.empty:
        print("両方揃うトレードがありません。クロールの完了を待ってください。")
        return
    print(f"期間: {both['entry_date'].min().date()} 〜 {both['entry_date'].max().date()}\n")

    variants = [
        ("条件なし", None, None),
        ("年次 増収10%以上", "a", 10.0),
        ("四半期 増収10%以上", "q", 10.0),
    ]

    def report(sub: pd.DataFrame, title: str):
        print(f"=== {title} ===")
        out = {}
        for label, col, thr in variants:
            s = sub if col is None else sub[sub[col] >= thr]
            st = stats(s)
            if st:
                out[label] = st
        print(pd.DataFrame(out).T.to_string())
        print()
        return pd.DataFrame(out).T

    report(both, f"全期間（{len(both):,}件）")

    print("=== 一致度 ===")
    agree = ((both["a"] >= 10) == (both["q"] >= 10)).mean() * 100
    print(f"「10%以上か」の判定が一致した割合: {agree:.1f}%")
    print(f"  年次○→四半期✗: {int(((both['a']>=10)&(both['q']<10)).sum()):,}件")
    print(f"  年次✗→四半期○: {int(((both['a']<10)&(both['q']>=10)).sum()):,}件")
    print(f"相関: {both['a'].corr(both['q']):.2f}\n")

    print("=== 重複しない3期間 ===")
    edges = [both["entry_date"].quantile(x) for x in (1/3, 2/3)]
    def era(d):
        return 0 if d <= edges[0] else (1 if d <= edges[1] else 2)
    both = both.copy()
    both["era"] = both["entry_date"].apply(era)
    for i in range(3):
        part = both[both["era"] == i]
        if len(part) < 50:
            print(f"  第{i+1}期: サンプル不足（{len(part)}件）")
            continue
        span = f"{part['entry_date'].min().date()}〜{part['entry_date'].max().date()}"
        report(part, f"第{i+1}期 {span}")

    print("採用基準：四半期が年次を**全期間で**上回って初めて置き換える。")
    print("（1期間だけの改善では不採用。REQUIREMENTS の採否基準）")


if __name__ == "__main__":
    main()
