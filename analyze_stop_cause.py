"""
損切りの主因は決算というイベントなのかを調べる

4.4-46 で「決算をまたぐトレードは損切り率が21%→31%に上がる」ことを
3期間一貫して確認した。ここから
**「このツールの損切りの主因はボラティリティではなくイベントではないか」**
という仮説が出た。これが正しければ、損切り幅をいくら動かしても改善
しなかった理由（4.4-39/4.4-40）が説明できる。

調べる2つ:
  ① 損切りは決算の直後に集中しているか
     （損切り日が決算発表の直後N日以内である割合を、全保有日の
       基準割合と比べる）
  ② 決算で飛ばされた損切りは、その後戻っているか
     （損切り後20日・60日の株価。戻るなら「降りるのが早すぎた」ことになる）

⚠️ ②で戻っていたとしても、それだけでルールを変える根拠にはならない。
   「戻る前にさらに下がる」経路を通れば実運用では耐えられない。
   ここで見るのはあくまで**仮説が立つかどうか**まで。

使い方:
    python3 analyze_stop_cause.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
SUMMARY = BASE_DIR / "data" / "jquants_summary.json"
TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"

STOP_THRESHOLD = -9.9   # これ以下なら損切りで終わったとみなす
NEAR_DAYS = 5           # 決算発表から何営業日以内を「決算直後」とするか

SUBPERIODS = [
    ("全期間", "2016-09-01", "2030-01-01"),
    ("第1期 2016-09〜2020-01", "2016-09-01", "2020-01-31"),
    ("第2期 2020-02〜2023-05", "2020-02-01", "2023-05-31"),
    ("第3期 2023-06〜2026-09", "2023-06-01", "2026-09-30"),
]


def main():
    data = json.load(open(SUMMARY, encoding="utf-8"))["data"]
    codes = sorted(data.keys())
    hist = fetch_histories(codes, period="max")

    closes, discs = {}, {}
    for code in codes:
        h = hist.get(code)
        if h is None or len(h) < 300:
            continue
        c = h["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            c = pd.Series(c.values, index=idx)
        closes[code] = c
        ds = sorted({r["DiscDate"] for r in data[code] if r.get("DiscDate")})
        discs[code] = pd.DatetimeIndex([pd.Timestamp(d) for d in ds])

    tr = pd.read_csv(TRADES, encoding="utf-8-sig")
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr = tr[tr["entry_date"] >= pd.Timestamp("2016-09-05")]

    rows = []
    for _, t in tr.iterrows():
        code = t["code"]
        c, d = closes.get(code), discs.get(code)
        if c is None or d is None or len(d) == 0:
            continue
        stopped = t["return_pct"] <= STOP_THRESHOLD

        # 手仕舞い日が決算発表の直後NEAR_DAYS営業日以内か
        ei = c.index.searchsorted(t["exit_date"])
        near = False
        if ei < len(c.index):
            lo = c.index[max(0, ei - NEAR_DAYS)]
            prev = d[(d <= c.index[min(ei, len(c.index) - 1)]) & (d >= lo)]
            near = len(prev) > 0

        # 手仕舞い後に戻ったか
        fwd = {}
        for hz in (20, 60):
            q = ei + hz
            fwd[hz] = ((float(c.iloc[q]) - t["exit_price"]) / t["exit_price"] * 100
                       if q < len(c.index) else np.nan)

        rows.append({"entry_date": t["entry_date"], "損切り": stopped,
                     "決算直後": near, "fwd20": fwd[20], "fwd60": fwd[60],
                     "保有日数": t["holding_days"]})

    df = pd.DataFrame(rows)
    print(f"対象トレード: {len(df):,}件（2016-09以降）")
    print(f"損切りで終わった: {int(df['損切り'].sum()):,}件"
          f"（{df['損切り'].mean()*100:.1f}%）\n")

    print(f"=== ① 損切りは決算の直後（{NEAR_DAYS}営業日以内）に集中しているか ===")
    out = []
    for label, lo, hi in SUBPERIODS:
        s = df[(df["entry_date"] >= pd.Timestamp(lo))
               & (df["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 200:
            continue
        stop = s[s["損切り"]]
        other = s[~s["損切り"]]
        out.append({
            "期間": label,
            "損切りが決算直後%": round(stop["決算直後"].mean() * 100, 1),
            "それ以外の手仕舞いが決算直後%": round(other["決算直後"].mean() * 100, 1),
            "損切り件数": len(stop),
        })
    print(pd.DataFrame(out).set_index("期間").to_string())
    print("  ※右の列が『基準』。損切りだけ高ければイベントが原因という傍証\n")

    print("=== ② 損切りのあと戻っているか（手仕舞い値からの騰落）===")
    out = []
    for label, lo, hi in SUBPERIODS:
        s = df[(df["entry_date"] >= pd.Timestamp(lo))
               & (df["entry_date"] <= pd.Timestamp(hi))]
        if len(s) < 200:
            continue
        for flag, name in [(True, "決算直後の損切り"), (False, "それ以外の損切り")]:
            g = s[s["損切り"] & (s["決算直後"] == flag)]
            if len(g) < 30:
                continue
            out.append({
                "期間": label, "区分": name, "件数": len(g),
                "20日後の平均%": round(g["fwd20"].dropna().mean(), 2),
                "20日後プラス%": round((g["fwd20"].dropna() > 0).mean() * 100, 1),
                "60日後の平均%": round(g["fwd60"].dropna().mean(), 2),
                "60日後プラス%": round((g["fwd60"].dropna() > 0).mean() * 100, 1),
            })
    print(pd.DataFrame(out).set_index(["期間", "区分"]).to_string())
    print("\n⚠️ 戻っていても、それだけでルール変更の根拠にはならない。")
    print("   途中でさらに下がる経路を通れば実運用では耐えられない。")


if __name__ == "__main__":
    main()
