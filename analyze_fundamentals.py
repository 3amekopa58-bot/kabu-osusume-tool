"""
割安度（PER・PBR）がトレード成績に寄与しているかを検証する

screen.py は総合スコアの50%を割安度スコアに配分しているのに、backtest.py は
ファンダメンタルを一切見ていない。つまり**推奨順位の半分は一度も検証されて
いない**。この状態を解消するためのスクリプト。

⚠️ 先読みバイアスの回避が要点：
  - yfinance の info が返す PER/PBR は「今日の値」。過去のトレードに当てはめる
    のは完全な先読みなので使わない
  - 代わりに data/fundamental_history.json（scripts/build_fundamental_history.py
    が作る）の「当時のEPS/BPS」を使い、エントリー日時点で**すでに公表されていた**
    決算だけからPER・PBRを計算する（available_from を過ぎた決算のみ採用）

やっていること：
  backtest.py が出したトレード明細の各行について、エントリー日時点のPER・PBRを
  求め、その水準ごとにリターン・勝率を集計する。割安な組が明確に良ければ
  割安度スコアには意味があり、差がなければ配分を見直す根拠になる。

使い方:
    python3 analyze_fundamentals.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "data" / "fundamental_history.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        print(f"{HISTORY_PATH} がありません。"
              "先に python3 scripts/build_fundamental_history.py を実行してください。")
        sys.exit(1)
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))["data"]


def latest_available(records: list, as_of: pd.Timestamp) -> dict:
    """as_of 時点で公表済みの決算のうち、最も新しいものを返す（無ければNone）"""
    usable = [r for r in records if pd.Timestamp(r["available_from"]) <= as_of]
    return usable[-1] if usable else None


def band(value, edges, labels):
    for e, l in zip(edges, labels):
        if value < e:
            return l
    return labels[-1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is None:
        cands = sorted(BASE_DIR.glob("output/backtest_trades_*universe*.csv"))
        if not cands:
            print("トレード明細CSVを指定してください（output/backtest_trades_*.csv）")
            return
        path = cands[-1]

    hist = load_history()
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    print(f"対象: {path.name} / {len(df):,}トレード")
    print(f"決算データ: {len(hist)}銘柄\n")

    rows = []
    no_data = 0
    for _, t in df.iterrows():
        recs = hist.get(t["code"])
        if not recs:
            no_data += 1
            continue
        rec = latest_available(recs, t["entry_date"])
        if rec is None:
            no_data += 1
            continue
        price = float(t["entry_price"])
        per = price / rec["eps"] if rec.get("eps") else None
        pbr = price / rec["bps"] if rec.get("bps") else None
        rows.append({
            "return_pct": t["return_pct"], "per": per, "pbr": pbr,
            "entry_date": t["entry_date"], "code": t["code"],
        })

    if not rows:
        print("判定できるトレードがありませんでした（決算データの期間が短すぎます）")
        return
    r = pd.DataFrame(rows)
    print(f"判定できたトレード: {len(r):,}件"
          f"（決算データ無しで除外 {no_data:,}件）")
    print(f"対象期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    def report(col, edges, labels, title):
        sub = r[r[col].notna() & (r[col] > 0)].copy()
        if sub.empty:
            print(f"=== {title}: 判定できるデータなし ===\n")
            return
        sub["帯"] = sub[col].apply(lambda v: band(v, edges, labels))
        g = sub.groupby("帯", observed=True)["return_pct"]
        out = pd.DataFrame({
            "件数": g.size(),
            "勝率%": (g.apply(lambda s: (s > 0).mean() * 100)).round(1),
            "平均%": g.mean().round(2),
            "中央値%": g.median().round(2),
        }).reindex(labels).dropna(how="all")
        print(f"=== {title} ===")
        print(out.to_string())
        # 割安側と割高側で差があるか
        if len(out) >= 2:
            lo, hi = out.iloc[0], out.iloc[-1]
            print(f"  → 最も割安な帯と最も割高な帯の差: "
                  f"勝率 {lo['勝率%'] - hi['勝率%']:+.1f}pt / "
                  f"平均リターン {lo['平均%'] - hi['平均%']:+.2f}pt")
        print()

    report("per", [10, 15, 20, 30, float("inf")],
           ["〜10倍", "10-15倍", "15-20倍", "20-30倍", "30倍〜"],
           "PER別のトレード成績（低いほど割安）")
    report("pbr", [0.8, 1.2, 2.0, 3.0, float("inf")],
           ["〜0.8倍", "0.8-1.2倍", "1.2-2.0倍", "2.0-3.0倍", "3.0倍〜"],
           "PBR別のトレード成績（低いほど割安）")

    # 期間を前半・後半に割って、同じ傾向が続くかを見る。
    # 業種別の偏りは集計では見えたのにアウトオブサンプルで消えた前例があるため
    # （REQUIREMENTS 4.4-5）、単一期間の集計だけで採用を決めてはいけない
    mid = r["entry_date"].quantile(0.5)
    print("=== 持続性チェック（期間を前半・後半に分割）===")
    for col, edges, labels, name in [
        ("per", [15, float("inf")], ["割安(PER15倍未満)", "割高(PER15倍以上)"], "PER"),
        ("pbr", [1.2, float("inf")], ["割安(PBR1.2倍未満)", "割高(PBR1.2倍以上)"], "PBR"),
    ]:
        sub = r[r[col].notna() & (r[col] > 0)].copy()
        if sub.empty:
            continue
        sub["帯"] = sub[col].apply(lambda v: band(v, edges, labels))
        print(f"\n  【{name}】")
        for era, part in (("前半", sub[sub["entry_date"] <= mid]),
                          ("後半", sub[sub["entry_date"] > mid])):
            if part.empty:
                continue
            g = part.groupby("帯", observed=True)["return_pct"]
            wr = (g.apply(lambda x: (x > 0).mean() * 100)).reindex(labels)
            n = g.size().reindex(labels)
            gap = wr.iloc[0] - wr.iloc[1] if wr.notna().all() else float("nan")
            span = f"{part['entry_date'].min().date()}〜{part['entry_date'].max().date()}"
            print(f"    {era}({span}): "
                  f"割安 {wr.iloc[0]:.1f}%({int(n.iloc[0])}件) / "
                  f"割高 {wr.iloc[1]:.1f}%({int(n.iloc[1])}件) → 差 {gap:+.1f}pt")

    print()
    print("=== 判定 ===")
    print("割安な帯の成績が一貫して良ければ、割安度スコアには意味がある。")
    print("差が小さい／逆転しているなら、screen.py の割安度50%という配分は")
    print("根拠がないことになるので見直す（REQUIREMENTS 4.4 に記録すること）。")


if __name__ == "__main__":
    main()
