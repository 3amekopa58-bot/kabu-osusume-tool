"""
片山流モードの銘柄条件バリアントを、過去のトレード実績で比較する

背景（片山晃_ルール.md）：
  PART 4 の「新高値ブレイク×ROE投資」は **増収10%・増益20%・ROE10%** を
  明示している。一方 PART 6 の「中小型株の中長期投資」では
  **「利益は伸びるのも落ち込むのもどちらもOK」**、増収減益はむしろ
  先行投資のチャンスだと書かれている。同じ著者だが手法が違うため、
  どちらの条件が実際に効いているのかを実データで確かめる必要がある。

やっていること：
  backtest.py が出した新高値ブレイクのトレード明細に対し、
  エントリー日時点で**すでに公表されていた**決算（EDINET・分割調整済み）から
  増収率・増益率を求め、バリアントごとにトレードを絞って成績を比べる。

  ⚠️ 先読みバイアスは available_from（決算期末＋92日）で回避している。
  ⚠️ 5年・10年・26年の3期間すべてで一貫して良くなければ採用しない
     （REQUIREMENTS の採否基準）。

使い方:
    python3 analyze_katayama_variants.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "data" / "edinet_financials_adjusted.json"
SUSPICIOUS_RETURN_THRESHOLD = 500.0

# 比較する銘柄条件。min_profit=None は「利益を条件にしない」＝PART 6 の考え方
VARIANTS = [
    ("条件なし（新高値のみ）",      None, None, None),
    ("PART6 増収重視（利益不問）",  10.0, None, None),
    # PER上限は書籍に根拠のある2水準だけ試す（恣意的な探索を避けるため）。
    # 「PER30倍台まで買い」＝39倍以下、「PER50倍を超えたら売り」＝50倍以下
    ("　└ +PER50倍以下",           10.0, None, 50.0),
    ("　└ +PER39倍以下",           10.0, None, 39.0),
    ("PART4 書籍版（増収10増益20）", 10.0, 20.0, 39.0),
    ("検証版（増収10増益30）",      10.0, 30.0, 20.0),
    ("増収10%以上かつ減益",         10.0, "down", None),
]


def load_history() -> dict:
    data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))["data"]
    return {c: sorted((list(v.values()) if isinstance(v, dict) else v),
                      key=lambda r: r["period_end"])
            for c, v in data.items()}


def growth_at(records: list, as_of: pd.Timestamp):
    """as_of 時点で公表済みの直近2期から増収率・増益率(%)を返す"""
    usable = [r for r in records if pd.Timestamp(r["available_from"]) <= as_of]
    if len(usable) < 2:
        return None, None
    cur, prev = usable[-1], usable[-2]

    def rate(key):
        a, b = cur.get(key), prev.get(key)
        if a is None or b is None or not b or b <= 0:
            return None
        return (a - b) / b * 100

    profit_key = ("operating_income" if cur.get("operating_income") is not None
                  else "ordinary_income")
    return rate("revenue"), rate(profit_key)


def enrich(path: Path, hist: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    rows = []
    for _, t in df.iterrows():
        recs = hist.get(t["code"])
        if not recs:
            continue
        rev_g, op_g = growth_at(recs, t["entry_date"])
        if rev_g is None or op_g is None:
            continue
        # エントリー日時点で公表済みの決算のEPSからPERを出す（先読み回避）
        usable = [x for x in recs
                  if pd.Timestamp(x["available_from"]) <= t["entry_date"]]
        eps = usable[-1].get("eps") if usable else None
        per = float(t["entry_price"]) / eps if eps else None
        rows.append({"return_pct": t["return_pct"], "rev": rev_g, "op": op_g,
                     "per": per,
                     "entry_date": t["entry_date"], "code": t["code"]})
    return pd.DataFrame(rows)


def select(r: pd.DataFrame, min_rev, min_profit, max_per=None) -> pd.DataFrame:
    sub = r
    if min_rev is not None:
        sub = sub[sub["rev"] >= min_rev]
    if min_profit == "down":
        sub = sub[sub["op"] < 0]
    elif min_profit is not None:
        sub = sub[sub["op"] >= min_profit]
    if max_per is not None:
        sub = sub[sub["per"].notna() & (sub["per"] > 0) & (sub["per"] <= max_per)]
    return sub


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    wins = sub[sub["return_pct"] > 0]["return_pct"]
    losses = sub[sub["return_pct"] <= 0]["return_pct"]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")
    return {
        "件数": len(sub),
        "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
        "平均%": round(sub["return_pct"].mean(), 2),
        "中央値%": round(sub["return_pct"].median(), 2),
        "PF": round(pf, 2),
    }


def forward_returns(r: pd.DataFrame, horizons=(60, 250, 500)) -> pd.DataFrame:
    """
    エントリー日から N営業日後の株価リターンを付ける。

    PART 6 の「中長期投資」は保有期間が年単位なのに、このバックテストの
    平均保有日数は30日程度しかない。エグジット条件（PPP崩れ or -8%）を
    外して素のフォワードリターンで見ないと、PART 6 の主張を検証したことに
    ならないため、別軸として用意する。
    """
    codes = sorted(r["code"].unique())
    prices = fetch_histories(codes, period="max", verbose=False)
    out = r.copy()
    for h in horizons:
        out[f"fwd{h}"] = None
    for i, t in out.iterrows():
        hist = prices.get(t["code"])
        if hist is None or hist.empty:
            continue
        idx = hist.index
        entry = t["entry_date"]
        if idx.tz is not None and entry.tz is None:
            entry = entry.tz_localize(idx.tz)
        pos = idx.searchsorted(entry)
        if pos >= len(idx):
            continue
        p0 = float(hist["Close"].iloc[pos])
        if not p0:
            continue
        for h in horizons:
            j = pos + h
            if j < len(idx):
                out.at[i, f"fwd{h}"] = (float(hist["Close"].iloc[j]) - p0) / p0 * 100
    for h in horizons:
        out[f"fwd{h}"] = pd.to_numeric(out[f"fwd{h}"], errors="coerce")
    return out


def report_forward(r: pd.DataFrame, era: str, horizons=(60, 250, 500)):
    print(f"=== {era}：エントリー日からの素のフォワードリターン"
          f"（エグジット条件なし・買い持ち）===")
    rows = {}
    for label, min_rev, min_profit, max_per in VARIANTS:
        sub = select(r, min_rev, min_profit, max_per)
        d = {}
        for h in horizons:
            v = sub[f"fwd{h}"].dropna()
            if len(v) < 30:
                continue
            d[f"{h}日 件数"] = len(v)
            d[f"{h}日 勝率%"] = round((v > 0).mean() * 100, 1)
            d[f"{h}日 平均%"] = round(v.mean(), 2)
        if d:
            rows[label] = d
    if rows:
        print(pd.DataFrame(rows).T.to_string())
    print()


def persistence_forward(r: pd.DataFrame, horizon=500, n_split=3):
    """
    フォワードリターンの優劣が期間をまたいで続くかを見る。

    500日リターンは窓が大きく重なるため、見かけの件数ほど独立した
    サンプルは無い。単一期間の集計だけで採用を決めてはいけない
    （REQUIREMENTS 4.4-5 の前例）。
    """
    col = f"fwd{horizon}"
    sub = r[r[col].notna()].copy()
    if sub.empty:
        return
    sub = sub.sort_values("entry_date")
    edges = [sub["entry_date"].quantile(i / n_split) for i in range(1, n_split)]
    def era_of(d):
        for i, e in enumerate(edges):
            if d <= e:
                return i
        return n_split - 1
    sub["era"] = sub["entry_date"].apply(era_of)

    print(f"=== 持続性チェック：{horizon}日フォワードリターンを"
          f"エントリー時期で{n_split}分割 ===")
    for i in range(n_split):
        part = sub[sub["era"] == i]
        if part.empty:
            continue
        span = f"{part['entry_date'].min().date()}〜{part['entry_date'].max().date()}"
        print(f"\n  【第{i+1}期 {span}】")
        for label, min_rev, min_profit, max_per in VARIANTS:
            v = select(part, min_rev, min_profit, max_per)[col].dropna()
            if len(v) < 20:
                print(f"    {label:<26} 件数{len(v):>5}  （サンプル不足）")
                continue
            print(f"    {label:<26} 件数{len(v):>5}  "
                  f"勝率 {(v > 0).mean() * 100:5.1f}%  平均 {v.mean():+7.2f}%")
    print()


def main():
    hist = load_history()
    files = {
        "5年":  "backtest_trades_pppsl8_newhigh_trend_marketadx_volume_rs_universe_5y_20260901.csv",
        "10年": "backtest_trades_pppsl8_newhigh_trend_marketadx_volume_rs_universe_10y_20260901.csv",
        "26年": "backtest_trades_pppsl8_newhigh_trend_marketadx_volume_rs_universe_max_20260830.csv",
    }
    print("片山流モード：銘柄条件バリアントの比較")
    print("（新高値ブレイク＋PPP崩れ or -8%損切り／944銘柄ユニバース）\n")

    summary = {}
    for era, fname in files.items():
        path = BASE_DIR / "output" / fname
        if not path.exists():
            print(f"⚠️  {era}: {fname} がありません。スキップします")
            continue
        r = enrich(path, hist)
        if r.empty:
            print(f"⚠️  {era}: 決算データと突合できるトレードがありませんでした")
            continue
        span = f"{r['entry_date'].min().date()}〜{r['entry_date'].max().date()}"
        print(f"=== {era}（{span}） 突合できたトレード {len(r):,}件 ===")
        rows = {}
        for label, min_rev, min_profit, max_per in VARIANTS:
            s = stats(select(r, min_rev, min_profit, max_per))
            if s:
                rows[label] = s
        out = pd.DataFrame(rows).T
        print(out.to_string())
        print()
        summary[era] = out
        if era == "26年":
            fr = forward_returns(r)
            report_forward(fr, era)
            persistence_forward(fr)

    if len(summary) >= 2:
        print("=== 一貫性チェック（採否の判断材料）===")
        base = "条件なし（新高値のみ）"
        for label, _, _, _ in VARIANTS:
            if label == base:
                continue
            line = []
            ok = True
            for era, out in summary.items():
                if label not in out.index or base not in out.index:
                    continue
                d_pf = out.loc[label, "PF"] - out.loc[base, "PF"]
                d_wr = out.loc[label, "勝率%"] - out.loc[base, "勝率%"]
                n = int(out.loc[label, "件数"])
                line.append(f"{era}: PF{d_pf:+.2f} 勝率{d_wr:+.1f}pt ({n}件)")
                if d_pf <= 0:
                    ok = False
            mark = "✅ 全期間で改善" if ok and len(line) == len(summary) else "❌ 一貫しない"
            print(f"  {label}")
            print(f"    {' / '.join(line)}")
            print(f"    → {mark}")
        print()
        print("採用基準：5年・10年・26年すべてでPFが改善して初めて採用する")
        print("（1期間だけの改善は不採用。REQUIREMENTS の採否基準）")


if __name__ == "__main__":
    main()
