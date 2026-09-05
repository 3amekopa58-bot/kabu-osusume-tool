"""
決算発表後のドリフト（PEAD）を検証する

これまでの44節で「決算の細かい読み方」は**ことごとく効かなかった**：
四半期の増収率（4.4-21）・進捗率（4.4-23）・上方/下方修正（4.4-24）。
しかしそのすべては「決算の**中身の数字**」を見ていた。

**決算発表という「出来事」そのもの**＝発表直後に市場がどう反応したか、
そしてその後どう動くかは、一度も測っていない。

海外では PEAD（Post-Earnings Announcement Drift）＝
「サプライズの大きかった銘柄は発表後もしばらく同じ方向に動き続ける」
という現象が最も頑健なアノマリーの一つとして知られる。
片山晃の「決算モメンタム投資」もこの発想に近い。

⚠️ 会社の予想と実績の差（サプライズ）を正確に作るのは難しいので、
   まず**市場の反応そのもの**を代理変数として使う。
   発表翌日の値動きは「市場がその決算をどう評価したか」の要約であり、
   後知恵ではない（発表後の公開情報）。

後知恵の排除：
   - エントリーは発表翌日の**終値**。反応を見てから買う想定なので、
     その時点で知り得ない情報は使っていない
   - DiscDate は J-Quants の実際の開示日（決算期末ではない）

使い方:
    python3 analyze_pead.py [保有日数...]
      例: python3 analyze_pead.py 20 60
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
SUMMARY = BASE_DIR / "data" / "jquants_summary.json"

# 重複しない3期間（J-Quants Standard は10年ぶんなので2016年以降で割る）
SUBPERIODS = [
    ("第1期 2016-09〜2020-01", "2016-09-01", "2020-01-31"),
    ("第2期 2020-02〜2023-05", "2020-02-01", "2023-05-31"),
    ("第3期 2023-06〜2026-09", "2023-06-01", "2026-09-30"),
]


def main():
    horizons = [int(a) for a in sys.argv[1:] if a.isdigit()] or [20, 60]

    data = json.load(open(SUMMARY, encoding="utf-8"))["data"]
    codes = sorted(data.keys())
    print(f"決算イベント: {len(codes)}銘柄 / "
          f"{sum(len(v) for v in data.values()):,}レコード")
    print(f"保有日数: {horizons}\n")

    # 銘柄キーは "5803.T" 形式で、price_cache のキーと同じ。剥がすとキャッシュを外す
    hist = fetch_histories(codes, period="max")

    rows = []
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 300:
            continue
        close = h["Close"]
        idx = close.index
        # タイムゾーン付きだと比較でつまずくので落とす
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            close = pd.Series(close.values, index=idx)

        for rec in data[code]:
            d = rec.get("DiscDate")
            if not d:
                continue
            t = pd.Timestamp(d)
            # 発表日以降で最初の営業日を t0 とし、その翌営業日の終値で入る
            pos = idx.searchsorted(t)
            if pos + 1 >= len(idx):
                continue
            p_before = float(close.iloc[pos - 1]) if pos > 0 else np.nan
            p_react = float(close.iloc[pos + 1])   # 反応が出た日の終値＝買値
            if not np.isfinite(p_before) or p_before <= 0:
                continue
            # 市場の反応（発表前日の終値 → 反応日の終値）
            reaction = (p_react - p_before) / p_before * 100
            row = {"code": code, "date": idx[pos + 1], "reaction": reaction}
            for hz in horizons:
                q = pos + 1 + hz
                row[f"fwd{hz}"] = ((float(close.iloc[q]) - p_react) / p_react * 100
                                   if q < len(idx) else np.nan)
            rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["reaction"])
    print(f"突き合わせできたイベント: {len(df):,}件")
    print(f"期間: {df['date'].min().date()} 〜 {df['date'].max().date()}\n")

    def report(sub, label):
        if len(sub) < 500:
            print(f"=== {label} === 件数不足（{len(sub)}件）\n")
            return
        # 反応の大きさで5等分
        sub = sub.copy()
        sub["帯"] = pd.qcut(sub["reaction"], 5,
                           labels=["最も下げた", "下げ", "中位", "上げ", "最も上げた"])
        out = []
        for band, g in sub.groupby("帯", observed=True):
            r = {"帯": band, "件数": len(g),
                 "反応中央値%": round(g["reaction"].median(), 2)}
            for hz in horizons:
                col = g[f"fwd{hz}"].dropna()
                if len(col):
                    r[f"{hz}日後の平均%"] = round(col.mean(), 2)
                    r[f"{hz}日勝率%"] = round((col > 0).mean() * 100, 1)
            out.append(r)
        print(f"=== {label}（{len(sub):,}件）===")
        print(pd.DataFrame(out).set_index("帯").to_string())
        # 全体平均（比較の基準）
        base = {hz: sub[f"fwd{hz}"].dropna().mean() for hz in horizons}
        print("  基準（全イベント平均）: " +
              " / ".join(f"{hz}日 {base[hz]:+.2f}%" for hz in horizons))
        print()

    report(df, "全期間")
    for label, lo, hi in SUBPERIODS:
        sub = df[(df["date"] >= pd.Timestamp(lo)) & (df["date"] <= pd.Timestamp(hi))]
        report(sub, label)

    print("⚠️ 上は『全銘柄・全イベント』の生の集計であって、"
          "現行ルールを通過した銘柄の話ではない。\n"
          "   採否を決めるのは下の『現行ルールの中で効くか』のほう。\n")

    within_rule(df)

    print("⚠️ 重複しない3期間すべてで同じ向きに出ない限り採用しない。")
    print("   ここでの『反応』は市場の評価であって会社のサプライズではない。")


def _pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else float("inf")


def within_rule(events: pd.DataFrame):
    """
    現行ルールが実際に出したトレード（output/_universe_max_trades.csv）に、
    エントリー直前の決算反応をひも付けて、反応の大小で成績が変わるかを見る。

    ⚠️ これが本番の問い。生のイベントスタディで効いて見えても、
       現行ルール（PPP・100日線上・出来高・相対力・ADX）を通過した銘柄の
       中で効かなければ、足す意味がない。
    後知恵の排除：エントリー日より**前**の開示だけを使う。
    """
    path = BASE_DIR / "output" / "_universe_max_trades.csv"
    if not path.exists():
        print(f"（{path.name} が無いので現行ルール内の検証はスキップ）")
        return
    tr = load_trades(path)

    by_code = {c: g.sort_values("date") for c, g in events.groupby("code")}
    out = []
    for _, t in tr.iterrows():
        e = by_code.get(t["code"])
        if e is None:
            continue
        prior = e[(e["date"] < t["entry_date"])
                  & (e["date"] >= t["entry_date"] - pd.Timedelta(days=90))]
        if prior.empty:
            continue
        out.append({"entry_date": t["entry_date"], "return_pct": t["return_pct"],
                    "reaction": prior.iloc[-1]["reaction"]})
    d = pd.DataFrame(out)
    print(f"=== 現行ルールの中で効くか（直近90日以内に決算があった"
          f"{len(d):,}件）===\n")

    for label, lo, hi in [("全期間", "2016-09-01", "2030-01-01")] + SUBPERIODS:
        s = d[(d["entry_date"] >= pd.Timestamp(lo))
              & (d["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 200:
            print(f"--- {label} --- 件数不足（{len(s)}件）\n")
            continue
        s = s.copy()
        s["帯"] = pd.qcut(s["reaction"], 4,
                         labels=["最も下げた", "下げ", "上げ", "最も上げた"])
        rows = [{"帯": b, "件数": len(g),
                 "勝率%": round((g["return_pct"] > 0).mean() * 100, 1),
                 "平均%": round(g["return_pct"].mean(), 2),
                 "PF": round(_pf(g["return_pct"]), 2)}
                for b, g in s.groupby("帯", observed=True)]
        print(f"--- {label}（{len(s):,}件）---")
        print(pd.DataFrame(rows).set_index("帯").to_string())
        print(f"  条件なし: 勝率{(s['return_pct'] > 0).mean() * 100:.1f}% / "
              f"平均{s['return_pct'].mean():+.2f}% / PF{_pf(s['return_pct']):.2f}\n")


if __name__ == "__main__":
    main()
