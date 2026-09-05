"""
かぶ1000流「ネットネット株」が実際に効くのかを検証する

かぶ1000『貯金40万円が株式投資で4億円』第3章：

  換金性が高い流動資産 ＝ 現金及び預金 ＋ 受取手形及び売掛金
                        ＋ 有価証券 ＋ 投資有価証券 － 貸倒引当金
  **ネットネット指数 ＝ 時価総額 ÷（換金性が高い流動資産 － 総負債）**
  （小さいほど割安）**0.66未満＝超割安 / 0.5未満＝激安**

⚠️ 分母がマイナスの企業が大半（負債のほうが大きい）。指数として意味を
   成さないので対象外にする。これは元の定義どおりの扱い。

⚠️ バリュー株投資は保有期間が長い。エグジット条件つきのバックテスト
   （平均保有30〜60日）では手法の前提と合わないので、
   **エントリー日からの素のフォワードリターン**でも測る
   （片山流の長期版を検証したときと同じ考え方＝REQUIREMENTS 4.4-16）。

⚠️ 先読みバイアス：有報の**実際の提出日**（available_from）以降でのみ使う。

使い方:
    python3 analyze_netnet.py [トレード明細CSV]
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
NETNET_PATH = BASE_DIR / "data" / "netnet_history.json"
FIN_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0
# ⚠️ 中間ファイルを名指しで固定すると、コードや母集団を変えたときに
# 追随せず、古い成果物で測り続けることになる（2026-09-05に
# analyze_targets.py と analyze_sectors.py で実際に起きた＝4.4-56）。
# 現行ルールの正規のトレード明細を指す。無ければ次で作る:
#   python3 backtest.py timesl either trend marketadx volume rs sl10 max \
#           --tickers universe.csv
DEFAULT_TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"
FORWARD_HORIZONS = (60, 250, 500)


def load_netnet() -> dict:
    if not NETNET_PATH.exists():
        print("ネットネットの履歴がありません。先に "
              "python3 scripts/build_netnet_history.py を実行してください。")
        sys.exit(1)
    data = json.loads(NETNET_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted(v.values(), key=lambda r: r.get("available_from") or "")
            for c, v in data.items()}


def load_shares() -> dict:
    data = json.loads(FIN_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted(v.values(), key=lambda r: r["period_end"])
            for c, v in data.items()}


def value_at(recs: list, as_of: pd.Timestamp, key: str):
    """as_of 時点で提出済みの最新レコードから値を取る"""
    ok = [r for r in recs
          if r.get("available_from") and pd.Timestamp(r["available_from"]) <= as_of]
    return ok[-1].get(key) if ok else None


def stats(sub: pd.DataFrame, col="return_pct") -> dict:
    s = sub[col].dropna()
    if s.empty:
        return {}
    w = s[s > 0].sum()
    l = abs(s[s <= 0].sum())
    return {"件数": len(s), "勝率%": round((s > 0).mean() * 100, 1),
            "平均%": round(s.mean(), 2), "PF": round(w / l, 2) if l else float("inf")}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / DEFAULT_TRADES
    nn, fin = load_netnet(), load_shares()
    print(f"ネットネットの履歴: {len(nn)}銘柄 / "
          f"{sum(len(v) for v in nn.values()):,}レコード")

    from price_cache import fetch_histories
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    prices = fetch_histories(sorted(df["code"].unique()), period="max", verbose=False)

    rows = []
    for _, t in df.iterrows():
        denom = value_at(nn.get(t["code"], []), t["entry_date"], "denominator_yen")
        if denom is None or denom <= 0:      # 分母マイナスは対象外（定義どおり）
            continue
        sh = value_at([{**r, "available_from": r["available_from"]}
                       for r in fin.get(t["code"], [])], t["entry_date"], "shares")
        if not sh:
            continue
        cap = float(t["entry_price"]) * sh
        rec = {"return_pct": t["return_pct"], "entry_date": t["entry_date"],
               "code": t["code"], "netnet": cap / denom}
        # 素のフォワードリターン（保有期間の前提を外して測る）
        h = prices.get(t["code"])
        if h is not None and not h.empty:
            idx = h.index
            e = t["entry_date"]
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
    if r.empty:
        print("判定できるトレードがありません（分母がプラスの銘柄が少ない可能性）")
        return
    print(f"ネットネット指数を出せたトレード: {len(r):,}件（全{len(df):,}件中）")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    bands = [(0, 0.5, "激安（0.5未満）"), (0.5, 0.66, "超割安（0.5-0.66）"),
             (0.66, 1.0, "0.66-1.0"), (1.0, 2.0, "1.0-2.0"), (2.0, 1e9, "2.0以上")]

    def report(sub, title, col="return_pct"):
        out = {"全体": stats(sub, col)}
        for lo, hi, lab in bands:
            s = stats(sub[(sub["netnet"] >= lo) & (sub["netnet"] < hi)], col)
            if s and s["件数"] >= 20:
                out[lab] = s
        if len(out) > 1:
            print(f"=== {title} ===")
            print(pd.DataFrame(out).T.to_string())
            print()

    report(r, "採用ルールのエグジット込み（平均保有60日）")
    for d in FORWARD_HORIZONS:
        if f"fwd{d}" in r.columns:
            report(r, f"素のフォワードリターン {d}営業日（買い持ち）", f"fwd{d}")

    print("=== 重複しない3期間（激安＋超割安 = 0.66未満）===")
    edges = [r["entry_date"].quantile(x) for x in (1 / 3, 2 / 3)]
    def era(d):
        return 0 if d <= edges[0] else (1 if d <= edges[1] else 2)
    r["era"] = r["entry_date"].apply(era)
    for i in range(3):
        p = r[r["era"] == i]
        a, b = stats(p), stats(p[p["netnet"] < 0.66])
        if a and b and b["件数"] >= 20:
            print(f"  {p.entry_date.min().date()}〜{p.entry_date.max().date()}  "
                  f"全体 PF{a['PF']:.2f}({a['件数']:,}件) → "
                  f"0.66未満 PF{b['PF']:.2f}({b['件数']}件)  "
                  f"勝率 {a['勝率%']:.1f}%→{b['勝率%']:.1f}%")
        else:
            print(f"  第{i+1}期: サンプル不足（{b.get('件数', 0)}件）")
    print("\n採用基準：重複しない全期間で全体を上回って初めて採用する。")


if __name__ == "__main__":
    main()
