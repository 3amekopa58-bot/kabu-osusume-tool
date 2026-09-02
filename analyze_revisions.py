"""
業績予想の修正（上方／下方）がトレード成績に効くのかを検証する

片山晃『5年で1億貯める株式投資』PART 7 OKポイント⑤：

  **過去にたびたび下方修正していない**
  上方修正より「**下方修正をしないこと**」のほうが重要。
  楽観的な見通しを出して下方修正を繰り返す企業は避ける。
  たびたび上方修正を出すのは「予想が保守的」という意味で株価にポジティブ。

著者は「下方修正しないこと > 上方修正すること」と優劣まで述べているので、
両方を測って効果量を比べる。

⚠️ 数える窓は**エントリー日より前の3年間**に固定した。
   全期間で数えると古い会社ほど回数が積み上がり、比較にならない
   （REQUIREMENTS 4.4-22 で実際にこの罠を踏んだ）。
   3年は「3決算期ぶん見れば体質が分かる」という理由で**先に決めた値**で、
   成績を見ながら動かしていない。

⚠️ 先読みバイアス：DiscDate（実際の開示日）がエントリー日以前の開示だけを使う。

使い方:
    python3 analyze_revisions.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
SUMMARY_PATH = BASE_DIR / "data" / "jquants_summary.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0
LOOKBACK_YEARS = 3
DEFAULT_TRADES = ("output/backtest_trades_pppsl8_newhigh_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        print("J-Quantsのデータがありません。先に "
              "python3 scripts/build_jquants_financials.py を実行してください。\n"
              "（規約によりデータ本体はコミットしていない。約20分で再取得できる）")
        sys.exit(1)
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted([r for r in rows if r.get("DiscDate")],
                      key=lambda r: r["DiscDate"])
            for c, rows in data.items()}


def revisions_before(rows: list, as_of: pd.Timestamp, key="FOP") -> dict:
    """
    as_of より前の LOOKBACK_YEARS 年間に出た会社予想の上方／下方修正を数える。
    同じ決算期（CurFYEn）の中で、当期予想が前回より下がれば下方修正。
    """
    start = as_of - pd.DateOffset(years=LOOKBACK_YEARS)
    by_fy = {}
    for r in rows:
        d = pd.Timestamp(r["DiscDate"])
        if not (start <= d <= as_of):
            continue
        fy, v = r.get("CurFYEn"), _num(r.get(key))
        if not fy or v is None:
            continue
        by_fy.setdefault(fy, []).append((r["DiscDate"], v))

    down = up = 0
    last_dir = None      # 直近の修正の向き（イベント型の検証用）
    events = []
    for fy, items in by_fy.items():
        items.sort()
        for (d0, v0), (d1, v1) in zip(items, items[1:]):
            if v1 < v0:
                down += 1; events.append((d1, "down"))
            elif v1 > v0:
                up += 1; events.append((d1, "up"))
    if events:
        events.sort()
        last_dir = events[-1][1]
    return {"down": down, "up": up, "last": last_dir,
            "periods": len(by_fy)}


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
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    rows = []
    for _, t in df.iterrows():
        rs = summ.get(t["code"])
        if not rs:
            continue
        rv = revisions_before(rs, t["entry_date"])
        # 判定できる決算期が2つ以上ないと「たびたび」を測れない
        if rv["periods"] < 2:
            continue
        rows.append({"return_pct": t["return_pct"], "down": rv["down"],
                     "up": rv["up"], "last": rv["last"],
                     "entry_date": t["entry_date"], "code": t["code"]})
    r = pd.DataFrame(rows)
    if r.empty:
        print("判定できるトレードがありません")
        return
    print(f"判定できたトレード: {len(r):,}件（全{len(df):,}件中）")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}")
    print(f"窓: エントリー前{LOOKBACK_YEARS}年\n")

    def report(sub: pd.DataFrame, title: str):
        variants = {
            "条件なし": sub,
            "下方修正 0回": sub[sub["down"] == 0],
            "下方修正 1回": sub[sub["down"] == 1],
            "下方修正 2回以上": sub[sub["down"] >= 2],
            "上方修正 0回": sub[sub["up"] == 0],
            "上方修正 1回以上": sub[sub["up"] >= 1],
            "上方修正 2回以上": sub[sub["up"] >= 2],
            "直近の修正が上方": sub[sub["last"] == "up"],
            "直近の修正が下方": sub[sub["last"] == "down"],
        }
        out = {k: stats(v) for k, v in variants.items()}
        out = {k: v for k, v in out.items() if v and v["件数"] >= 30}
        if out:
            print(f"=== {title} ===")
            print(pd.DataFrame(out).T.to_string())
            print()

    report(r, f"全期間（{len(r):,}件）")

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

    print("採用基準：条件なしを**重複しない全期間で**上回って初めて採用する。")
    print("著者は「下方修正しないこと > 上方修正すること」と述べているので、")
    print("その優劣が実際に再現されるかも見る。")


if __name__ == "__main__":
    main()
