"""
暴落に周期性があるのかを検証する

「暴落は周期的に起きる」という見方があるが、このツールでは未検証。
日経平均の61年分（1965年〜）で、暴落の発生間隔を実際に測る。

⚠️ 周期性の検証は**後付けでパターンを見つけやすい**典型例。
   「4年周期に見える」といった印象は、少数のイベントからいくらでも作れる。
   そこで：
     ・暴落の定義を先に決める（後から閾値を動かさない）
     ・発生間隔の**ばらつき**を見る（規則的なら間隔の分散が小さいはず）
     ・**ランダムに起きた場合**と比べて、本当に規則的かを判定する

使い方:
    python3 analyze_crash_cycles.py [下落率の閾値]
      省略時は -20（高値から20%下落を暴落とみなす）
"""

import sys

import numpy as np
import pandas as pd

from price_cache import fetch_histories


def find_crashes(close: pd.Series, threshold: float = -20.0,
                 peak_window: int = 250) -> pd.DataFrame:
    """
    高値からthreshold%以上下落した局面を1件ずつ拾う。

    同じ下落局面を何度も数えないよう、**いったん高値を更新し直すまでを
    1つの局面**として扱う。

    ⚠️ `peak_window` は高値をどこまで遡って見るか（営業日）。
       **史上最高値（cummax）を使うと、日本株では機能しない。**
       日経は1989年の高値を34年間更新できず、その間の下落（ITバブル崩壊・
       リーマン・コロナ）がすべて「1つの34年間の暴落」に飲み込まれるため
       （2026-09-02に実測）。既定の250日＝約1年の高値を基準にする。
       peak_window=None を渡すと従来どおり史上最高値を使う。
    """
    peak = close.cummax() if peak_window is None else close.rolling(
        peak_window, min_periods=20).max()
    dd = (close / peak - 1) * 100
    events, in_crash, start, trough_v, trough_d = [], False, None, None, None
    for d, v in dd.items():
        if not in_crash and v <= threshold:
            in_crash, start = True, d
            trough_v, trough_d = v, d
        elif in_crash:
            if v < trough_v:
                trough_v, trough_d = v, d
            if v >= 0:                     # 高値を更新＝この局面は終了
                events.append({"開始": start, "底": trough_d, "回復": d,
                               "最大下落%": trough_v,
                               "底までの日数": (trough_d - start).days,
                               "回復までの日数": (d - start).days})
                in_crash = False
    if in_crash:                           # 未回復のまま現在に至る場合
        events.append({"開始": start, "底": trough_d, "回復": pd.NaT,
                       "最大下落%": trough_v,
                       "底までの日数": (trough_d - start).days,
                       "回復までの日数": np.nan})
    return pd.DataFrame(events)


def main():
    th = float(sys.argv[1]) if len(sys.argv) > 1 else -20.0
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    nk = fetch_histories(["^N225"], period="max", verbose=False).get("^N225")
    c = nk["Close"]
    if c.index.tz is not None:
        c.index = c.index.tz_localize(None)
    print(f"日経平均: {c.index[0].date()} 〜 {c.index[-1].date()}"
          f"（{(c.index[-1] - c.index[0]).days / 365.25:.0f}年）")
    print(f"暴落の定義: 直近{win}営業日（約{win/250:.0f}年）の高値から"
          f"{th}%以上の下落\n")

    ev = find_crashes(c, th, win)
    print(f"=== 検出した暴落局面: {len(ev)}件 ===")
    show = ev.copy()
    for col in ("開始", "底", "回復"):
        show[col] = show[col].dt.date.astype(str)
    print(show.to_string(index=False))
    print()

    if len(ev) < 3:
        print("件数が少なすぎて周期性は判定できない")
        return

    gaps = ev["開始"].diff().dt.days.dropna() / 365.25
    print("=== 発生間隔（前回の暴落の開始からの年数）===")
    print(f"  {[f'{g:.1f}年' for g in gaps]}")
    print(f"  平均 {gaps.mean():.1f}年 / 中央値 {gaps.median():.1f}年 / "
          f"標準偏差 {gaps.std():.1f}年")
    print(f"  最短 {gaps.min():.1f}年 / 最長 {gaps.max():.1f}年")
    cv = gaps.std() / gaps.mean()
    print(f"\n  変動係数（標準偏差÷平均）= {cv:.2f}")
    print("    完全に規則的なら0に近づく。")
    print("    ランダム（ポアソン過程）なら理論値は1.0。")
    if cv < 0.5:
        print("    → 0.5未満。規則性がある可能性がある")
    elif cv < 0.9:
        print("    → 0.5〜0.9。ランダムよりはややまとまっているが規則的とは言えない")
    else:
        print("    → 0.9以上。**ランダムと区別がつかない**")

    # ランダムに同じ回数だけ起きた場合と比べる
    rng = np.random.default_rng(0)
    span = (c.index[-1] - c.index[0]).days / 365.25
    sims = []
    for _ in range(10000):
        pts = np.sort(rng.uniform(0, span, len(ev)))
        g = np.diff(pts)
        if len(g) > 1 and g.mean() > 0:
            sims.append(g.std() / g.mean())
    sims = np.array(sims)
    pct = (sims <= cv).mean() * 100
    print(f"\n  ランダムに{len(ev)}回起きた場合の変動係数と比べると、"
          f"実際の値は下位{pct:.0f}%タイル")
    print("    （小さいほど規則的。50%前後ならランダムと変わらない）")

    print("\n=== 暦年ごとの暴落の有無 ===")
    years = sorted(set(ev["開始"].dt.year))
    print(f"  暴落が始まった年: {years}")
    if len(years) > 2:
        d = np.diff(years)
        print(f"  年の間隔: {list(d)}")


if __name__ == "__main__":
    main()
