"""
かぶ1000流のPBR階級（激安/超割安/割安）が実際に効くのかを検証する

かぶ1000『貯金40万円が株式投資で4億円』第2章：
  **PBR 0.4以上0.5未満＝割安 / 0.3以上0.4未満＝超割安 / 0.3未満＝激安**
  ただし「PBRが低くても換金性の低い資産（在庫・工場）が多い場合は
  本当の割安とは言えない」とも書かれている。

⚠️ 4.4-10 で PBR を検証済みだが、あちらは
   「〜0.8 / 0.8-1.2 / 1.2-2.0 / 2.0-3.0 / 3.0〜」という**このツール独自の帯**。
   **かぶ1000氏の階級（0.3/0.4/0.5）はもっと低い水準**で、
   その刻みで見たことはまだ無い。

⚠️ バリュー株は保有期間が長いので、素のフォワードリターンでも測る。

使い方:
    python3 analyze_pbr_tiers.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
FIN_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0
DEFAULT_TRADES = ("output/backtest_trades_timesl10d60_either_trend_marketadx_"
                  "volume_rs_universe_max_20260830.csv")
FORWARD_HORIZONS = (250, 500)

TIERS = [(0, 0.3, "激安（0.3未満）"), (0.3, 0.4, "超割安（0.3-0.4）"),
         (0.4, 0.5, "割安（0.4-0.5）"), (0.5, 1.0, "0.5-1.0"),
         (1.0, 2.0, "1.0-2.0"), (2.0, 1e9, "2.0以上")]


def load_fin() -> dict:
    data = json.loads(FIN_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted(v.values(), key=lambda r: r["period_end"])
            for c, v in data.items()}


def bps_at(recs: list, as_of: pd.Timestamp):
    ok = [r for r in recs
          if pd.Timestamp(r["available_from"]) <= as_of and r.get("bps")]
    return ok[-1]["bps"] if ok else None


def stats(sub: pd.DataFrame, col="return_pct") -> dict:
    s = sub[col].dropna()
    if s.empty:
        return {}
    w, l = s[s > 0].sum(), abs(s[s <= 0].sum())
    return {"件数": len(s), "勝率%": round((s > 0).mean() * 100, 1),
            "平均%": round(s.mean(), 2), "PF": round(w / l, 2) if l else float("inf")}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / DEFAULT_TRADES
    fin = load_fin()
    from price_cache import fetch_histories

    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    prices = fetch_histories(sorted(df["code"].unique()), period="max", verbose=False)

    rows = []
    for _, t in df.iterrows():
        b = bps_at(fin.get(t["code"], []), t["entry_date"])
        if not b or b <= 0:
            continue
        rec = {"return_pct": t["return_pct"], "entry_date": t["entry_date"],
               "code": t["code"], "pbr": float(t["entry_price"]) / b}
        h = prices.get(t["code"])
        if h is not None and not h.empty:
            idx, e = h.index, t["entry_date"]
            if idx.tz is not None and e.tz is None:
                e = e.tz_localize(idx.tz)
            pos = idx.searchsorted(e)
            if pos < len(idx):
                p0 = float(h["Close"].iloc[pos])
                for d in FORWARD_HORIZONS:
                    j = pos + d
                    if p0 and j < len(idx):
                        rec[f"fwd{d}"] = (float(h["Close"].iloc[j]) - p0) / p0 * 100
        rows.append(rec)

    r = pd.DataFrame(rows)
    print(f"PBRを出せたトレード: {len(r):,}件（全{len(df):,}件中）")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    def report(sub, title, col="return_pct"):
        out = {"全体": stats(sub, col)}
        for lo, hi, lab in TIERS:
            s = stats(sub[(sub["pbr"] >= lo) & (sub["pbr"] < hi)], col)
            if s and s["件数"] >= 30:
                out[lab] = s
        print(f"=== {title} ===")
        print(pd.DataFrame(out).T.to_string())
        print()

    report(r, "採用ルールのエグジット込み（平均保有60日）")
    for d in FORWARD_HORIZONS:
        if f"fwd{d}" in r.columns:
            report(r, f"素のフォワードリターン {d}営業日（買い持ち）", f"fwd{d}")

    print("=== 重複しない3期間（PBR0.5未満＝かぶ1000流の割安帯）===")
    edges = [r["entry_date"].quantile(x) for x in (1 / 3, 2 / 3)]
    r["era"] = r["entry_date"].apply(
        lambda d: 0 if d <= edges[0] else (1 if d <= edges[1] else 2))
    for i in range(3):
        p = r[r["era"] == i]
        a, b = stats(p), stats(p[p["pbr"] < 0.5])
        if a and b and b["件数"] >= 30:
            print(f"  {p.entry_date.min().date()}〜{p.entry_date.max().date()}  "
                  f"全体 PF{a['PF']:.2f}({a['件数']:,}件) → "
                  f"0.5未満 PF{b['PF']:.2f}({b['件数']}件)  "
                  f"勝率 {a['勝率%']:.1f}%→{b['勝率%']:.1f}%")
        else:
            print(f"  第{i+1}期: サンプル不足（{b.get('件数', 0)}件）")
    print("\n採用基準：重複しない全期間で全体を上回って初めて採用する。")


if __name__ == "__main__":
    main()
