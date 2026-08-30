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
# EDINET由来（2013年〜・株式分割調整済み）を優先し、無ければyfinance由来（5期分）
HISTORY_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
FALLBACK_HISTORY_PATH = BASE_DIR / "data" / "fundamental_history.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0


def load_history() -> dict:
    """
    決算データを {code: [決算期の古い順のレコード]} で返す。
    EDINET版は {code: {period_end: rec}} という形なのでリストに直す。
    """
    path = HISTORY_PATH if HISTORY_PATH.exists() else FALLBACK_HISTORY_PATH
    if not path.exists():
        print("決算データがありません。先に "
              "python3 scripts/build_edinet_financials.py を実行してください。")
        sys.exit(1)
    print(f"決算データ: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    out = {}
    for code, v in data.items():
        recs = list(v.values()) if isinstance(v, dict) else v
        out[code] = sorted(recs, key=lambda r: r["period_end"])
    return out


def latest_available(records: list, as_of: pd.Timestamp) -> dict:
    """as_of 時点で公表済みの決算のうち、最も新しいものを返す（無ければNone）"""
    usable = [r for r in records if pd.Timestamp(r["available_from"]) <= as_of]
    return usable[-1] if usable else None


def growth_at(records: list, as_of: pd.Timestamp):
    """
    as_of 時点で公表済みの直近2期から、増収率・増益率を返す。
    片山晃『5年で1億貯める株式投資』の新高値ブレイク投資が見ている
    「増収増益」を機械的に判定するため（片山晃_ルール.md 参照）。
    """
    usable = [r for r in records if pd.Timestamp(r["available_from"]) <= as_of]
    if len(usable) < 2:
        return None, None
    cur, prev = usable[-1], usable[-2]
    def rate(key):
        a, b = cur.get(key), prev.get(key)
        if a is None or b is None or not b or b <= 0:
            return None
        return (a - b) / b * 100
    # EDINET由来は operating_income を持たず ordinary_income（IFRSは税引前利益）。
    # どちらでも「利益の伸び」を見る目的には使えるので、あるほうを使う
    profit_key = "operating_income" if cur.get("operating_income") is not None else "ordinary_income"
    return rate("revenue"), rate(profit_key)


def band(value, edges, labels):
    """edges は昇順、labels も「低い側から」の順で渡すこと（先頭が境界未満）"""
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
        rev_g, op_g = growth_at(recs, t["entry_date"])
        rows.append({
            "return_pct": t["return_pct"], "per": per, "pbr": pbr,
            "revenue_growth": rev_g, "op_growth": op_g,
            "entry_date": t["entry_date"], "code": t["code"],
        })

    if not rows:
        print("判定できるトレードがありませんでした（決算データの期間が短すぎます）")
        return
    r = pd.DataFrame(rows)
    print(f"判定できたトレード: {len(r):,}件"
          f"（決算データ無しで除外 {no_data:,}件）")
    print(f"対象期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    def report(col, edges, labels, title, positive_only=False):
        # PER/PBR は負の値（赤字・債務超過）に意味がないので除外するが、
        # 増収率・増益率のマイナスは「減収・減益」という重要な情報なので残す
        sub = r[r[col].notna()].copy()
        if positive_only:
            sub = sub[sub[col] > 0]
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
            print(f"  → 「{out.index[0]}」と「{out.index[-1]}」の差: "
                  f"勝率 {lo['勝率%'] - hi['勝率%']:+.1f}pt / "
                  f"平均リターン {lo['平均%'] - hi['平均%']:+.2f}pt")
        print()

    report("per", [10, 15, 20, 30, float("inf")],
           ["〜10倍", "10-15倍", "15-20倍", "20-30倍", "30倍〜"],
           "PER別のトレード成績（低いほど割安）", positive_only=True)
    report("pbr", [0.8, 1.2, 2.0, 3.0, float("inf")],
           ["〜0.8倍", "0.8-1.2倍", "1.2-2.0倍", "2.0-3.0倍", "3.0倍〜"],
           "PBR別のトレード成績（低いほど割安）", positive_only=True)

    report("revenue_growth", [0, 5, 10, 20, float("inf")],
           ["減収", "0-5%増", "5-10%増", "10-20%増", "20%〜増"],
           "増収率別のトレード成績（片山流：業績の伸びを見る）")
    report("op_growth", [0, 10, 30, 60, float("inf")],
           ["減益", "0-10%増", "10-30%増", "30-60%増", "60%〜増"],
           "営業増益率別のトレード成績（片山流）")

    # 片山流の中核＝「増収増益」を1つの条件として見る
    sub = r[r["revenue_growth"].notna() & r["op_growth"].notna()].copy()
    if not sub.empty:
        sub["区分"] = sub.apply(
            lambda x: "増収増益" if x["revenue_growth"] > 0 and x["op_growth"] > 0
            else ("減収減益" if x["revenue_growth"] <= 0 and x["op_growth"] <= 0 else "どちらか一方"),
            axis=1)
        g = sub.groupby("区分", observed=True)["return_pct"]
        out = pd.DataFrame({
            "件数": g.size(),
            "勝率%": (g.apply(lambda x: (x > 0).mean() * 100)).round(1),
            "平均%": g.mean().round(2),
            "中央値%": g.median().round(2),
        }).reindex(["増収増益", "どちらか一方", "減収減益"]).dropna(how="all")
        print("=== 増収増益かどうか（片山流の中核条件）===")
        print(out.to_string())
        if {"増収増益", "減収減益"} <= set(out.index):
            print(f"  → 増収増益と減収減益の差: "
                  f"勝率 {out.loc['増収増益','勝率%'] - out.loc['減収減益','勝率%']:+.1f}pt / "
                  f"平均 {out.loc['増収増益','平均%'] - out.loc['減収減益','平均%']:+.2f}pt")
        print()

    # 期間を前半・後半に割って、同じ傾向が続くかを見る。
    # 業種別の偏りは集計では見えたのにアウトオブサンプルで消えた前例があるため
    # （REQUIREMENTS 4.4-5）、単一期間の集計だけで採用を決めてはいけない
    mid = r["entry_date"].quantile(0.5)
    print("=== 持続性チェック（期間を前半・後半に分割）===")
    for col, edges, labels, name in [
        ("per", [15, float("inf")], ["PER15倍未満", "PER15倍以上"], "PER"),
        ("pbr", [1.2, float("inf")], ["PBR1.2倍未満", "PBR1.2倍以上"], "PBR"),
        ("op_growth", [30, float("inf")], ["営業増益30%未満", "営業増益30%以上"], "営業増益率"),
        ("revenue_growth", [10, float("inf")], ["増収10%未満", "増収10%以上"], "増収率"),
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
            av = g.mean().reindex(labels)
            n = g.size().reindex(labels)
            if not wr.notna().all():
                continue
            span = f"{part['entry_date'].min().date()}〜{part['entry_date'].max().date()}"
            print(f"    {era}({span}) {int(n.iloc[0])}件 vs {int(n.iloc[1])}件")
            print(f"      勝率      : {labels[0]} {wr.iloc[0]:.1f}% / "
                  f"{labels[1]} {wr.iloc[1]:.1f}% → 差 {wr.iloc[0] - wr.iloc[1]:+.1f}pt")
            print(f"      平均リターン: {labels[0]} {av.iloc[0]:+.2f}% / "
                  f"{labels[1]} {av.iloc[1]:+.2f}% → 差 {av.iloc[0] - av.iloc[1]:+.2f}pt")

    print()
    print("=== 判定 ===")
    print("割安な帯の成績が一貫して良ければ、割安度スコアには意味がある。")
    print("差が小さい／逆転しているなら、screen.py の割安度50%という配分は")
    print("根拠がないことになるので見直す（REQUIREMENTS 4.4 に記録すること）。")


if __name__ == "__main__":
    main()
