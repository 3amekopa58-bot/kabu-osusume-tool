"""
暴落の「予兆」が事前に観測できるかを検証する

`analyze_crash_cycles.py` で周期性は否定された（変動係数0.86＝ほぼランダム）。
では**いつ来るかは読めなくても、直前に何か兆候はあるのか**を調べる。

⚠️ 使うのは**その日までに観測できる指標だけ**。
   「暴落の3か月前は…」のような後知恵の指標は使わない。

見る指標（すべて日経平均から、その日までのデータで計算できる）：
  ・過去1年の騰落率（過熱度）
  ・直近1年高値からの下落率（ドローダウン）
  ・ボラティリティ（20日／250日の標準偏差と、その比＝ボラの急変）
  ・200日移動平均からの乖離
  ・ADX（トレンドの強さ。screen.py が既に使っている指標）

判定：各指標の水準ごとに、**その後3か月以内に暴落（直近1年高値から-20%）が
      始まった割合**を出す。全体の発生率より明確に高い水準があれば予兆になる。

使い方:
    python3 analyze_crash_warning.py
"""

import numpy as np
import pandas as pd

from price_cache import fetch_histories
from analyze_crash_cycles import find_crashes

HORIZON_DAYS = 60          # 「その後3か月以内」＝60営業日
CRASH_THRESHOLD = -20.0
PEAK_WINDOW = 250


def main():
    nk = fetch_histories(["^N225"], period="max", verbose=False).get("^N225")
    c, h, l = nk["Close"], nk["High"], nk["Low"]
    for s in (c, h, l):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)

    ev = find_crashes(c, CRASH_THRESHOLD, PEAK_WINDOW)
    starts = pd.DatetimeIndex(ev["開始"])
    print(f"日経平均 {c.index[0].date()} 〜 {c.index[-1].date()}")
    print(f"暴落: {len(ev)}件（直近{PEAK_WINDOW}日高値から{CRASH_THRESHOLD}%）\n")

    df = pd.DataFrame(index=c.index)
    df["ret1y"] = c.pct_change(250) * 100
    df["dd"] = (c / c.rolling(PEAK_WINDOW, min_periods=20).max() - 1) * 100
    r = c.pct_change()
    df["vol20"] = r.rolling(20).std() * np.sqrt(250) * 100
    df["vol250"] = r.rolling(250).std() * np.sqrt(250) * 100
    df["vol_ratio"] = df["vol20"] / df["vol250"]
    df["dev200"] = (c / c.rolling(200).mean() - 1) * 100

    # ADX（screen.py と同じ計算を使う）
    import screen
    df["adx"] = screen.calc_adx(h, l, c)

    # 各日について「その後HORIZON_DAYS営業日以内に暴落が始まったか」
    future = pd.Series(False, index=c.index)
    pos = {d: i for i, d in enumerate(c.index)}
    for s in starts:
        if s not in pos:
            continue
        i = pos[s]
        future.iloc[max(0, i - HORIZON_DAYS):i] = True
    df["crash_next"] = future

    base = df["crash_next"].mean() * 100
    print(f"=== 全体：その後{HORIZON_DAYS}営業日以内に暴落が始まる日の割合 "
          f"= {base:.1f}% ===")
    print("（各指標の帯でこれを明確に上回れば「予兆」になる）\n")

    def report(col, bands, title):
        sub = df[df[col].notna()]
        rows = {}
        for lo, hi, lab in bands:
            m = (sub[col] > lo) & (sub[col] <= hi)
            if m.sum() < 200:
                continue
            rate = sub.loc[m, "crash_next"].mean() * 100
            rows[lab] = {"日数": int(m.sum()), "暴落前の割合%": round(rate, 1),
                         "全体比": round(rate / base, 2)}
        if rows:
            print(f"=== {title} ===")
            print(pd.DataFrame(rows).T.to_string())
            print()

    report("ret1y", [(-1e9, -10, "-10%以下"), (-10, 0, "-10〜0%"),
                     (0, 10, "0〜+10%"), (10, 25, "+10〜+25%"),
                     (25, 50, "+25〜+50%"), (50, 1e9, "+50%以上")],
           "日経平均の過去1年騰落率")
    report("dd", [(-1e9, -15, "-15%以下"), (-15, -10, "-15〜-10%"),
                  (-10, -5, "-10〜-5%"), (-5, -2, "-5〜-2%"),
                  (-2, 0.1, "高値圏(-2%〜)")],
           "直近1年高値からの下落率")
    report("vol_ratio", [(0, 0.7, "0.7未満(静か)"), (0.7, 1.0, "0.7-1.0"),
                         (1.0, 1.3, "1.0-1.3"), (1.3, 1.8, "1.3-1.8"),
                         (1.8, 1e9, "1.8以上(急変)")],
           "ボラティリティ比（20日÷250日）")
    report("vol20", [(0, 12, "12%未満"), (12, 18, "12-18%"), (18, 25, "18-25%"),
                     (25, 35, "25-35%"), (35, 1e9, "35%以上")],
           "20日ボラティリティ（年率）")
    report("dev200", [(-1e9, -10, "-10%以下"), (-10, -3, "-10〜-3%"),
                      (-3, 3, "-3〜+3%"), (3, 10, "+3〜+10%"),
                      (10, 1e9, "+10%以上")],
           "200日移動平均からの乖離")
    report("adx", [(0, 15, "15未満"), (15, 20, "15-20"), (20, 25, "20-25"),
                   (25, 35, "25-35"), (35, 1e9, "35以上")],
           "ADX（トレンドの強さ）")

    print("読み方：「全体比」が1.0なら予兆なし（全体と同じ確率）。")
    print("        2.0以上なら、その水準では暴落前である確率が2倍ということ。")
    print("⚠️ ただし暴落は17件しかない。帯ごとの日数が多くても、")
    print("   元になるイベント数が少ないので過信しないこと。")


if __name__ == "__main__":
    main()
