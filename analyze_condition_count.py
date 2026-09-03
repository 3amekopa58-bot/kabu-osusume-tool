"""
採用条件を何個満たしているかで成績がどう変わるかを実測する

通知は参考枠に「条件 4/5」と出しているが、**4/5がどの程度の成績なのかは
示していなかった**。⭐︎で表示するなら根拠が要るので測る。

5条件：
  ① cond_signal  下半身 or 押し目買いが点灯
  ② cond_trend   PPP3/4以上 ＋ 100日線より上
  ③ cond_volume  出来高が20日平均の1.5倍以上
  ④ cond_rs      日経をアウトパフォーム中
  ⑤ cond_regime  日経がADX20超 ＋ 100日線より上

⚠️ フィルターなしの明細（シグナルが点灯した全トレード）を母集団にする。
   ①は全件で成立しているので、実質の範囲は1〜5個。

⚠️ 銘柄ごとに条件の時系列をまとめて作り、エントリー日で引く
   （1件ずつ履歴を切ると12万件では終わらない）。

使い方:
    python3 analyze_condition_count.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
SUSPICIOUS_RETURN_THRESHOLD = 500.0
DEFAULT_TRADES = ("output/backtest_trades_timesl10d60_either_"
                  "universe_max_20260903.csv")
MA_PERIODS = (5, 10, 20, 50, 100)


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    w = sub[sub["return_pct"] > 0]["return_pct"].sum()
    l = abs(sub[sub["return_pct"] <= 0]["return_pct"].sum())
    return {"件数": len(sub), "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
            "平均%": round(sub["return_pct"].mean(), 2),
            "PF": round(w / l, 2) if l else float("inf")}


def main():
    sys.path.insert(0, str(BASE_DIR))
    import screen
    from price_cache import fetch_histories

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / DEFAULT_TRADES
    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    print(f"対象: {len(df):,}トレード（{path.name}）")

    codes = sorted(df["code"].unique())
    hist = fetch_histories(codes + ["^N225"], period="max", verbose=False)

    # ⑤ 相場環境は全銘柄共通なので先に1本だけ作る
    nk = hist.get("^N225")
    nc, nh, nl = nk["Close"], nk["High"], nk["Low"]
    for x in (nc, nh, nl):
        if x.index.tz is not None:
            x.index = x.index.tz_localize(None)
    regime = ((nc > nc.rolling(100).mean())
              & (screen.calc_adx(nh, nl, nc) > 20)).fillna(False)

    rows = []
    for i, code in enumerate(codes, 1):
        h = hist.get(code)
        if h is None or h.empty:
            continue
        c, v = h["Close"], h["Volume"]
        idx = c.index
        if idx.tz is not None:
            c, v = c.copy(), v.copy()
            c.index = v.index = idx.tz_localize(None)
        sma = {n: c.rolling(n).mean() for n in MA_PERIODS}
        up = sum((sma[MA_PERIODS[j]] > sma[MA_PERIODS[j + 1]]).astype(int)
                 for j in range(len(MA_PERIODS) - 1))
        cond_trend = ((up >= 3) & (c > sma[100])).fillna(False)
        cond_volume = (v >= v.rolling(20).mean() * 1.5).fillna(False)
        rel = c / nc.reindex(c.index, method="ffill")
        cond_rs = (rel > rel.rolling(50).mean()).fillna(False)

        sub = df[df["code"] == code]
        for _, t in sub.iterrows():
            d = t["entry_date"]
            if d not in cond_trend.index:
                pos = cond_trend.index.searchsorted(d)
                if pos >= len(cond_trend.index):
                    continue
                d = cond_trend.index[pos]
            n = 1  # cond_signal は全件成立
            n += int(bool(cond_trend.get(d, False)))
            n += int(bool(cond_volume.get(d, False)))
            n += int(bool(cond_rs.get(d, False)))
            n += int(bool(regime.reindex([d], method="ffill").iloc[0]))
            rows.append({"return_pct": t["return_pct"], "条件数": n,
                         "entry_date": t["entry_date"]})
        if i % 200 == 0:
            print(f"  {i}/{len(codes)}銘柄")

    r = pd.DataFrame(rows)
    print(f"\n再現できた: {len(r):,}件")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    print("=== 条件の充足数ごとの成績 ===")
    out = {}
    for n in range(1, 6):
        st = stats(r[r["条件数"] == n])
        if st and st["件数"] >= 50:
            out[f"{n}/5"] = st
    print(pd.DataFrame(out).T.to_string())

    print("\n=== 重複しない3期間 ===")
    edges = [r["entry_date"].quantile(x) for x in (1 / 3, 2 / 3)]
    r["era"] = r["entry_date"].apply(
        lambda d: 0 if d <= edges[0] else (1 if d <= edges[1] else 2))
    for n in range(1, 6):
        line = f"  {n}/5: "
        for i in range(3):
            st = stats(r[(r["条件数"] == n) & (r["era"] == i)])
            line += f"{('PF%.2f' % st['PF']) if st and st['件数'] >= 30 else '—':>9}"
        print(line)
    print("\n→ 充足数が増えるほど良くなり、3期間で一貫していれば⭐︎表示の根拠になる")


if __name__ == "__main__":
    main()
