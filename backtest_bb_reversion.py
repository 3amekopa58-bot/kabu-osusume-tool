"""ボリンジャーバンドの逆張り平均回帰を検証する。

ルール:
  エントリー: 終値が -Nσ（20日移動平均 − N×20日標準偏差）を下回った翌日の始値
  エグジット : 終値が中心線（20日移動平均）に戻った翌日の始値
  保険       : 期限切れ（既定60日）と損切り（既定-10%）。
               これが無いと「戻らない株」を永遠に持つことになり、
               実運用と懸け離れる。

⚠️ 先読みの排除:
   ・シグナルは**その日の終値**で判定し、売買は**翌日の始値**で行う
     （終値で買えたことにすると、その日の値動きを知って売買したことになる）
   ・σ・移動平均はすべて当日までの20日で計算

⚠️ コストは 4.4-67 の模型（呼値のスプレッド＋平方根則のインパクト）で控除。
   逆張りは安い銘柄・荒い銘柄で発火しやすく、**コストが効きやすい**。

⚠️ 判定は中央値。重複しない3期間すべてで同じ向きでなければ採用しない。

使い方: ./venv/bin/python backtest_bb_reversion.py [σ...]
   例 : ./venv/bin/python backtest_bb_reversion.py 2 2.5 3 4
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from price_cache import fetch_histories

B = Path(__file__).resolve().parent
TIME_STOP = 60          # 営業日
STOP_LOSS = -10.0       # %
ORDER_YEN = 1_000_000
IMPACT_C = 0.5
PERIODS = [("第1期 2000-2008", 2000, 2008), ("第2期 2009-2017", 2009, 2017),
           ("第3期 2018-2026", 2018, 2026)]


def tick_size(p: float) -> float:
    return 1.0 if p < 3000 else (5.0 if p < 5000 else (10.0 if p < 30000 else 50.0))


def pf(s: pd.Series) -> float:
    w = s[s > 0].sum(); l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def run(hist: dict, names: dict, n_sigma: float) -> pd.DataFrame:
    trades = []
    for code, h in hist.items():
        if h is None or len(h) < 250:
            continue
        h = h.copy()
        h.index = pd.to_datetime(h.index).tz_localize(None)
        if (h["Close"].pct_change().abs() > 0.8).any():
            continue                       # 汚染データ（本体と同じ基準）
        c, o = h["Close"], h["Open"]
        sma = c.rolling(20).mean()
        sd = c.rolling(20).std()
        lower = sma - n_sigma * sd
        vol20 = c.pct_change().rolling(20).std() * 100
        tv20 = (c * h["Volume"]).rolling(20).mean()
        idx = c.index
        i, n = 20, len(c)
        while i < n - 1:
            if pd.isna(lower.iloc[i]) or c.iloc[i] >= lower.iloc[i]:
                i += 1
                continue
            # 翌日の始値で買う
            e = i + 1
            ep = float(o.iloc[e])
            if ep <= 0:
                i += 1
                continue
            j, exit_p, why = e, None, None
            while j < n - 1:
                j += 1
                if float(c.iloc[j]) / ep - 1 <= STOP_LOSS / 100:
                    exit_p, why = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]), "損切り"
                    break
                if pd.notna(sma.iloc[j]) and float(c.iloc[j]) >= float(sma.iloc[j]):
                    exit_p, why = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]), "中心線"
                    break
                if j - e >= TIME_STOP:
                    exit_p, why = float(o.iloc[j + 1]) if j + 1 < n else float(c.iloc[j]), "期限"
                    break
            if exit_p is None:
                break
            ret = (exit_p / ep - 1) * 100
            # コスト
            sp = tick_size(ep) / ep * 100
            adv = float(tv20.iloc[i]) if pd.notna(tv20.iloc[i]) and tv20.iloc[i] > 0 else np.nan
            v = float(vol20.iloc[i]) if pd.notna(vol20.iloc[i]) else 2.0
            part = min(ORDER_YEN / adv, 1.0) if adv == adv else 1.0
            cost = sp + IMPACT_C * v * np.sqrt(part) * 2
            trades.append({"code": code, "name": names.get(code, code),
                           "entry_date": idx[e], "exit_date": idx[j],
                           "保有日数": j - e, "理由": why,
                           "リターン%": ret, "コスト%": cost, "実質%": ret - cost,
                           "年": idx[e].year})
            i = j + 1
    return pd.DataFrame(trades)


def main() -> None:
    sigmas = [float(a) for a in sys.argv[1:]] or [2.0, 2.5, 3.0, 4.0]
    tk = list(csv.DictReader(open(B / "universe.csv", encoding="utf-8-sig")))
    codes = [t["code"] for t in tk]
    names = {t["code"]: t["name"] for t in tk}
    print(f"{len(codes)}銘柄の株価を用意中…")
    hist = fetch_histories(codes, period="max")
    print(f"エグジット: 中心線(20日線)到達 / 期限{TIME_STOP}日 / 損切り{STOP_LOSS}%")
    print("※シグナルは終値判定、売買は翌日始値。コストは4.4-67の模型で控除\n")

    for ns in sigmas:
        d = run(hist, names, ns)
        if len(d) == 0:
            print(f"=== -{ns}σ: トレード0件 ===\n")
            continue
        print(f"=== -{ns}σ で買い、中心線で売る ===")
        print(f"  {len(d):,}件 / 年平均 {len(d)/27:.0f}件 / 平均保有 {d['保有日数'].mean():.1f}日")
        print(f"  コスト前: 中央値 {d['リターン%'].median():+.2f}% / 平均 {d['リターン%'].mean():+.2f}% "
              f"/ 勝率 {(d['リターン%']>0).mean()*100:.1f}% / PF {pf(d['リターン%']):.2f}")
        print(f"  コスト控除後: 中央値 {d['実質%'].median():+.2f}% / 平均 {d['実質%'].mean():+.2f}% "
              f"/ 勝率 {(d['実質%']>0).mean()*100:.1f}% / PF {pf(d['実質%']):.2f}"
              f"  （往復コスト中央値 {d['コスト%'].median():.2f}%）")
        per = []
        for lab, a, b in PERIODS:
            s = d[(d["年"] >= a) & (d["年"] <= b)]
            per.append(f"{lab}: {s['実質%'].median():+.2f}%({len(s):,}件)" if len(s) >= 50 else f"{lab}: 件数不足")
        ok = all("件数不足" not in x for x in per) and all(
            d[(d["年"] >= a) & (d["年"] <= b)]["実質%"].median() > 0 for _, a, b in PERIODS)
        print(f"  期間別（実質）: " + " / ".join(per))
        print(f"  → {'**3期間ともプラス**' if ok else '3期間で揃わない'}")
        print(f"  手仕舞いの内訳: " + " / ".join(
            f"{k} {v}件({v/len(d)*100:.0f}%)" for k, v in d["理由"].value_counts().items()))
        d.to_csv(B / "output" / f"bb_reversion_{ns}sigma.csv", index=False, encoding="utf-8-sig")
        print()


if __name__ == "__main__":
    main()
