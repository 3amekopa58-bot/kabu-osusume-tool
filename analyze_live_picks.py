"""
実際に推奨した銘柄がその後どうなったかを測る（バックテストとの照合）

バックテストは「勝率51.8%・期待値+3.04%」と出しているが、
**実際に通知した銘柄がどうなったか**は誰も測っていなかった。
`output/recommend_YYYYMMDD.csv` を後追いして実測を出す。

⚠️ 2026-09-02まで、日次のGitHub Actionsは推奨CSVをジョブ終了時に
   破棄していた（4.4-26で修正）。手元にあるのは手動実行の残りだけで、
   **日次の連続記録ではない**。本格的な照合は artifact が溜まってから。

⚠️ 通知に出たのは上位PICK_COUNT件だけ。ここでは
   「通知に出た銘柄」＝候補をスコア順に並べた上位N件、として再現する。

判定：バックテストと同じルールで出口を決める
   ・利確：ATR×3の目標に到達（高値ベース）
   ・損切り：-10%に到達（安値ベース）
   ・期限：60営業日
   まだ決着していないものは「保有中」として現在値で評価する。

使い方:
    python3 analyze_live_picks.py [--top N]
"""

import glob
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
STOP_PCT = -10.0
HOLD_DAYS = 60
ATR_MULTIPLE = 3.0


def load_picks(top_n: int) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(BASE_DIR / "output" / "recommend_*.csv"))):
        name = Path(f).stem
        if "_before" in name:              # 同日の実験用ファイルは除く
            continue
        date = pd.to_datetime(name.replace("recommend_", ""), format="%Y%m%d")
        d = pd.read_csv(f)
        if "buy_timing" not in d.columns:
            continue
        cand = d[d["buy_timing"].astype(str).str.contains("買いタイミング")].copy()
        if cand.empty:
            continue
        # 通知は「全条件を満たす銘柄」を優先し、無ければ一部充足を出す
        if "conditions_all" in cand.columns and cand["conditions_all"].any():
            cand = cand[cand["conditions_all"]]
            枠 = "本命"
        else:
            枠 = "参考"
            if "conditions_met" in cand.columns:
                cand = cand.sort_values("conditions_met", ascending=False)
        if "total_score" in cand.columns:
            cand = cand.sort_values("total_score", ascending=False)
        for _, r in cand.head(top_n).iterrows():
            rows.append({"推奨日": date, "枠": 枠, "code": r["code"],
                         "name": r.get("name"), "推奨価格": r.get("price")})
    return pd.DataFrame(rows)


def main():
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 3
    from price_cache import fetch_histories

    picks = load_picks(top_n)
    if picks.empty:
        print("推奨CSVが見つかりません")
        return
    picks = picks[picks["推奨価格"].notna()]
    print(f"照合する推奨: {len(picks)}件"
          f"（{picks['推奨日'].min().date()} 〜 {picks['推奨日'].max().date()}・"
          f"1日あたり上位{top_n}件）")
    print(f"内訳: {picks['枠'].value_counts().to_dict()}\n")

    hist = fetch_histories(sorted(picks["code"].unique()), period="1y",
                           verbose=False, stale_days=0)
    out = []
    for _, p in picks.iterrows():
        h = hist.get(p["code"])
        if h is None or h.empty:
            continue
        idx = h.index
        d = p["推奨日"]
        if idx.tz is not None and d.tz is None:
            d = d.tz_localize(idx.tz)
        pos = idx.searchsorted(d)
        if pos >= len(idx) - 1:
            continue
        entry = float(p["推奨価格"])
        # ATRから利確目標を出す（backtest/notify と同じ考え方）
        tr = pd.concat([h["High"] - h["Low"],
                        (h["High"] - h["Close"].shift()).abs(),
                        (h["Low"] - h["Close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[pos]
        target = entry + ATR_MULTIPLE * atr if pd.notna(atr) else None
        stop = entry * (1 + STOP_PCT / 100)

        fwd = h.iloc[pos + 1: pos + 1 + HOLD_DAYS]
        result, days, ret = "保有中", len(fwd), None
        for i, (dt_, row) in enumerate(fwd.iterrows(), 1):
            if row["Low"] <= stop:
                result, days, ret = "損切り", i, STOP_PCT
                break
            if target is not None and row["High"] >= target:
                result, days, ret = "利確", i, (target - entry) / entry * 100
                break
        if ret is None:
            last = float(fwd["Close"].iloc[-1]) if len(fwd) else entry
            ret = (last - entry) / entry * 100
            if len(fwd) >= HOLD_DAYS:
                result = "期限切れ"
        out.append({**p.to_dict(), "結果": result, "日数": days,
                    "リターン%": round(ret, 2)})

    r = pd.DataFrame(out)
    print("=== 結果の内訳 ===")
    print(r["結果"].value_counts().to_string())
    print()
    decided = r[r["結果"] != "保有中"]
    print(f"=== 決着したもの（{len(decided)}件）===")
    if len(decided):
        wr = (decided["リターン%"] > 0).mean() * 100
        w = decided[decided["リターン%"] > 0]["リターン%"].sum()
        l = abs(decided[decided["リターン%"] <= 0]["リターン%"].sum())
        print(f"  勝率 {wr:.1f}% / 平均 {decided['リターン%'].mean():+.2f}% / "
              f"PF {w / l if l else float('inf'):.2f}")
        print(f"  （バックテストの期待値：勝率51.8% / 平均+3.04% / PF1.71）")
    print()
    print("=== 全明細 ===")
    show = r.copy()
    show["推奨日"] = show["推奨日"].dt.date
    print(show[["推奨日", "枠", "code", "name", "推奨価格", "結果",
                "日数", "リターン%"]].to_string(index=False))
    print("\n⚠️ 手元のCSVは手動実行の残りで日次の連続記録ではない。")
    print("   件数も少なく、この数字で判断してはいけない。")
    print("   2026-09-03以降はartifactに残るので、数か月後に本格的な照合ができる。")


if __name__ == "__main__":
    main()
