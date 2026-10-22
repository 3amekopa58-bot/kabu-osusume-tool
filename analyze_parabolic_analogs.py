"""
「倍になって高値を更新した銘柄」の過去事例を集め、その後どうなったかを測る

発端は7552（ハピネット）の分析。120日で+110%・200日線から+84%・上場来高値
という状態で「過去に似た値動きをした株はあるか」を問われたのが起点。

出すもの:
  ① 条件該当事例の事後リターン（20/60/120/250日）とベースラインとの比較
  ② 3期間（重複なし）での一貫性
  ③ 業績（純利益成長率）で分けた場合の違い
  ④ 天井を打った日の分布（＝いつ下がり始めたか）
  ⑤ 上昇の起点から数えた天井日の分布とハザード

⚠️ 判定はすべて**中央値**。4.4-56 / 4.4-65 で、平均は裾の1件（8105.Tの
   +1,189%など）に支配されて結論が反転すると確認済み。

⚠️ ユニバースは**現存銘柄のみ**＝生存者バイアスが乗る。上場廃止になった
   事例が抜けているので、実際の事後成績はここで出る数字より悪い。
   このスクリプトの結論は「不利」方向なので、バイアスは結論を覆さない
   （覆すには欠落が逆方向に働く必要があるが、生存者バイアスは逆方向）。

⚠️ 「上場来高値」は price_cache の開始時点（多くは2000〜2001年）からの
   最高値。それ以前に高値がある古い銘柄では条件が実際より緩くなる。

⚠️ 起点A（直前250日の最安値）は遡りを250日で打ち切っているので、
   それより前から上げている銘柄では起点が実際より後ろにずれる＝日数は
   過小評価側。

前提: data/price_cache/max/ が作られていること（price_cache.py）。
使い方: python3 analyze_parabolic_analogs.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
CACHE = BASE_DIR / "data" / "price_cache" / "max"
OUT = BASE_DIR / "output" / "analog_events.csv"

# --- 該当条件（7552の2026-10-22時点の値を丸めたもの）
RET_LOOKBACK = 120      # 騰落を測る遡り日数
MIN_RET = 1.00          # +100%以上
MIN_DEV200 = 0.60       # 200日線から+60%以上
MAX_ATH_GAP = -0.05     # 高値から-5%以内
COOLDOWN = 250          # 同じ銘柄の同じ波を重複カウントしない
MIN_TURNOVER_OKU = 5.0  # 売買代金フィルター（7552は約21億円/日）
HORIZONS = [20, 60, 120, 250]
FWD = 250
LOOKBACK = 250


def load_names():
    names = {}
    path = BASE_DIR / "universe.csv"
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            names[code] = name
    return names


def collect_events(names):
    """条件に該当した銘柄・日付と、その後のリターンを集める。"""
    rows, baseline = [], []
    files = sorted(CACHE.glob("*.csv"))
    for n, path in enumerate(files):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            continue
        if len(df) < 500 or "Close" not in df:
            continue
        close = df["Close"].astype(float)
        high = df["High"].astype(float) if "High" in df else close
        vol = df["Volume"].astype(float) if "Volume" in df else pd.Series(0.0, index=close.index)

        ret = close / close.shift(RET_LOOKBACK) - 1
        dev200 = close / close.rolling(200).mean() - 1
        ath_gap = close / high.cummax() - 1
        turnover = (close * vol).rolling(20).mean()
        arr = close.values
        size = len(arr)

        for j in range(LOOKBACK, size - FWD):
            baseline.append([arr[j + h] / arr[j] - 1 for h in HORIZONS])

        j = LOOKBACK
        while j < size - 1:
            hit = (ret.iloc[j] >= MIN_RET and dev200.iloc[j] >= MIN_DEV200
                   and ath_gap.iloc[j] >= MAX_ATH_GAP)
            if not hit:
                j += 1
                continue
            # 上昇の起点：直前250日の最安値（起点A）と200日線の上抜け（起点B）
            low_i = j - LOOKBACK + int(np.argmin(arr[j - LOOKBACK:j + 1]))
            above = (close > close.rolling(200).mean()).values
            cross = next((k for k in range(j, max(j - LOOKBACK, 200), -1)
                          if above[k] and not above[k - 1]), None)
            peak_i = j + int(np.argmax(arr[j:j + FWD + 1])) if j + FWD < size else np.nan
            rows.append({
                "code": path.stem,
                "name": names.get(path.stem, ""),
                "date": close.index[j].date(),
                "price": arr[j],
                "ret120": ret.iloc[j],
                "dev200": dev200.iloc[j],
                "turnover_oku": turnover.iloc[j] / 1e8,
                "rise_days": j - low_i,
                "rise_days_cross": (j - cross) if cross else np.nan,
                "peak_from_event": (peak_i - j) if peak_i == peak_i else np.nan,
                "peak_from_low": (peak_i - low_i) if peak_i == peak_i else np.nan,
                "peak_gain": (arr[int(peak_i)] / arr[j] - 1) if peak_i == peak_i else np.nan,
                **{f"fwd{h}": (arr[j + h] / arr[j] - 1 if j + h < size else np.nan)
                   for h in HORIZONS},
            })
            j += COOLDOWN
        if n % 300 == 0:
            print(f"  ...{n}/{len(files)}")
    return pd.DataFrame(rows), pd.DataFrame(baseline, columns=[f"fwd{h}" for h in HORIZONS])


def show_forward(df, label):
    print(f"--- {label}（{len(df)}件）")
    print(f"{'期間':>6}{'中央値':>9}{'平均':>9}{'勝率':>8}{'-30%以下':>10}{'+50%以上':>10}")
    for h in HORIZONS:
        s = df[f"fwd{h}"].dropna()
        if s.empty:
            continue
        print(f"{h:>4}日{s.median()*100:>+8.1f}%{s.mean()*100:>+8.1f}%"
              f"{(s > 0).mean()*100:>7.1f}%{(s <= -0.30).mean()*100:>9.1f}%"
              f"{(s >= 0.50).mean()*100:>9.1f}%")
    print()


def profit_growth_at(fin, code, date):
    """その時点で開示済みだった最新本決算の純利益成長率（先読みなし）。"""
    rec = fin.get(code)
    if not rec:
        return np.nan
    avail = sorted(
        [v for v in rec.values()
         if v.get("available_from") and pd.Timestamp(v["available_from"]) <= date],
        key=lambda v: v["period_end"])
    if len(avail) < 2:
        return np.nan
    cur, prev = avail[-1], avail[-2]
    if not prev.get("net_income") or prev["net_income"] <= 0 or cur.get("net_income") is None:
        return np.nan
    return cur["net_income"] / prev["net_income"] - 1


def show_hist(series, label, bins, labels):
    s = series.dropna()
    print(f"=== {label}（{len(s)}件）")
    counts = pd.cut(s, bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels)
    for lab, cnt, width in zip(labels, counts.values, np.diff(bins)):
        density = cnt / len(s) * 100 / width
        print(f"  {lab:<12}{cnt:>5}件{cnt/len(s)*100:>6.1f}%  密度{density:>5.2f}%/日 "
              f"{'#' * int(density * 12)}")
    print(f"  中央値 {s.median():.0f}日 / 平均 {s.mean():.0f}日\n")


def main():
    names = load_names()
    ev, baseline = collect_events(names)
    OUT.parent.mkdir(exist_ok=True)
    ev.to_csv(OUT, index=False)
    ev["date"] = pd.to_datetime(ev["date"])
    liq = ev[ev.turnover_oku >= MIN_TURNOVER_OKU].copy()
    print(f"\n該当 {len(ev)}件 / {ev['code'].nunique()}銘柄"
          f"（売買代金{MIN_TURNOVER_OKU:.0f}億以上 {len(liq)}件 / "
          f"ベースライン {len(baseline):,}日）→ {OUT}\n")

    # ① 事後リターン
    show_forward(ev, "条件該当（全件）")
    show_forward(liq, f"うち売買代金{MIN_TURNOVER_OKU:.0f}億円/日以上")
    show_forward(baseline, "ベースライン（全営業日）")

    # ② 3期間で一貫するか
    print("=== 重複しない3期間（売買代金フィルター後）")
    print(f"{'期間':<12}{'件数':>6}{'中60日':>9}{'中250日':>10}{'勝率250':>9}{'-30%以下':>10}")
    for lab, start, end in [("2000-2008", "2000-01-01", "2008-12-31"),
                            ("2009-2017", "2009-01-01", "2017-12-31"),
                            ("2018-2025", "2018-01-01", "2025-12-31")]:
        w = liq[(liq.date >= start) & (liq.date <= end)]
        f250, f60 = w["fwd250"].dropna(), w["fwd60"].dropna()
        if f250.empty:
            continue
        print(f"{lab:<12}{len(w):>6}{f60.median()*100:>+8.1f}%{f250.median()*100:>+9.1f}%"
              f"{(f250 > 0).mean()*100:>8.1f}%{(f250 <= -0.30).mean()*100:>9.1f}%")

    # ③ 業績の裏付けで分ける
    fin = json.load(open(BASE_DIR / "data" / "edinet_financials.json"))["data"]
    liq["pg"] = [profit_growth_at(fin, c, d) for c, d in zip(liq.code, liq.date)]
    have = liq.dropna(subset=["pg"])
    print(f"\n=== 業績の裏付けで分ける（純利益成長率が取れた{len(have)}件）")
    print(f"{'区分':<20}{'件数':>6}{'中60日':>9}{'中250日':>10}{'勝率250':>9}{'-30%以下':>10}")
    for lab, sub in [("純利益 +30%以上", have[have.pg >= 0.30]),
                     ("純利益 0〜+30%", have[(have.pg >= 0) & (have.pg < 0.30)]),
                     ("純利益 減益", have[have.pg < 0])]:
        f250, f60 = sub["fwd250"].dropna(), sub["fwd60"].dropna()
        if f250.empty:
            continue
        print(f"{lab:<20}{len(sub):>6}{f60.median()*100:>+8.1f}%{f250.median()*100:>+9.1f}%"
              f"{(f250 > 0).mean()*100:>8.1f}%{(f250 <= -0.30).mean()*100:>9.1f}%")
    print()

    # ④ 天井を打った日（イベント日から）
    done = liq.dropna(subset=["peak_from_event"])
    bins = [0, 5, 10, 20, 40, 60, 90, 120, 180, 251]
    labels = ["0〜5日", "6〜10日", "11〜20日", "21〜40日", "41〜60日",
              "61〜90日", "91〜120日", "121〜180日", "181〜250日"]
    show_hist(done.peak_from_event, "高値更新日から天井まで", bins, labels)
    print(f"  ※ピーク時の上昇は中央値 {done.peak_gain.median()*100:+.1f}%\n")

    # ⑤ 上昇の起点から数えた天井日とハザード
    bins2 = [0, 60, 90, 120, 150, 180, 210, 250, 300, 360, 500]
    labels2 = ["〜60日", "61〜90", "91〜120", "121〜150", "151〜180",
               "181〜210", "211〜250", "251〜300", "301〜360", "361〜500"]
    show_hist(done.peak_from_low, "起点A（直前250日の最安値）から天井まで", bins2, labels2)

    print("=== ハザード：ここまで天井を打たずに来た場合、次の20日で天井を打つ確率")
    print(f"{'経過日数':<10}{'生存数':>8}{'次20日で天井':>14}")
    for t in range(60, 381, 20):
        alive = done[done.peak_from_low >= t]
        if len(alive) < 15:
            print(f"  {t}日以上の生存が{len(alive)}件まで減ったので打ち切り")
            break
        topped = alive[alive.peak_from_low < t + 20]
        print(f"{t:>4}日目{'':<4}{len(alive):>8}{len(topped)/len(alive)*100:>13.1f}%")

    # ⑥ イベント時点の上昇成熟度で分ける（速い上昇ほど早く終わるか）
    print("\n=== イベント時点で上昇が何日続いていたかで分ける")
    print(f"{'経過日数':<14}{'件数':>6}{'天井まで':>10}{'天井の上げ':>11}{'60日後':>9}{'1年後':>9}")
    for lab, sub in [("〜120日", done[done.rise_days <= 120]),
                     ("121〜180日", done[(done.rise_days > 120) & (done.rise_days <= 180)]),
                     ("181〜220日", done[(done.rise_days > 180) & (done.rise_days <= 220)]),
                     ("221日以上", done[done.rise_days > 220])]:
        if len(sub) < 10:
            continue
        print(f"{lab:<14}{len(sub):>6}{sub.peak_from_event.median():>9.0f}日"
              f"{sub.peak_gain.median()*100:>+10.1f}%{sub.fwd60.median()*100:>+8.1f}%"
              f"{sub.fwd250.median()*100:>+8.1f}%")

    print("\n=== どちらのものさしがブレが小さいか（天井日の標準偏差）")
    print(f"  高値更新日から数える : {done.peak_from_event.std():.0f}日")
    print(f"  上昇の起点から数える : {done.peak_from_low.std():.0f}日")


if __name__ == "__main__":
    main()
