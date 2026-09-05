"""
テクニカルスコアの各項目が、実際にリターンを予測しているのかを検証する

`screen.py` は推奨順位を「割安度スコア×0.5 ＋ テクニカルスコア×0.5」で決めている。
割安度の側は 4.4-10 で検証済みだが、**テクニカルスコアの側は未検証**だった。
順位付けの半分が根拠不明のまま、という 4.4-10 と同じ構図が残っていた。

やること：バックテストのトレード明細について、**エントリー日までで切った
株価履歴**を screen.py の指標関数にそのまま通し、当時のスコアと各項目を
再現する。そのうえで、項目ごとに「成立した組」と「しなかった組」の
成績を比べる。

⚠️ screen.py の関数をそのまま呼ぶ（式を書き写さない）。写すと本体と
   ずれたときに気づけないため。

⚠️ 加点の大きい項目ほど順位への影響が大きい。効いていない項目に
   大きな配点があれば、それは順位付けのノイズになっている。

使い方:
    python3 analyze_technical_score.py [トレード明細CSV] [--limit N]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import screen
from price_cache import fetch_histories

BASE_DIR = Path(__file__).parent
SUSPICIOUS_RETURN_THRESHOLD = 500.0
# 再現した指標をここに保存する。全13,633件の再現に約8分かかるので、
# 配点を変えて試すたびに待たなくて済むようにキャッシュする。
# （株価から導いた中間データなのでコミットはしない＝.gitignore）
CACHE_PATH = BASE_DIR / "data" / "technical_rows.csv"
# ⚠️ 中間ファイルを名指しで固定すると、コードや母集団を変えたときに
# 追随せず、古い成果物で測り続けることになる（2026-09-05に
# analyze_targets.py と analyze_sectors.py で実際に起きた＝4.4-56）。
# 現行ルールの正規のトレード明細を指す。無ければ次で作る:
#   python3 backtest.py timesl either trend marketadx volume rs sl10 max \
#           --tickers universe.csv
DEFAULT_TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"


def row_at(hist: pd.DataFrame) -> dict:
    """
    screen.py の第1段階と同じ計算を、渡された履歴の**末尾時点**で行う。
    エントリー日までで切った履歴を渡せば、当時の値が再現できる。
    """
    if len(hist) < screen.MIN_HISTORY_DAYS:
        return None
    close, open_, low = hist["Close"], hist["Open"], hist["Low"]
    volume = hist["Volume"]
    sma = {n: close.rolling(n).mean() for n in screen.MA_PERIODS}
    r = {"last_close": close.iloc[-1], "last_open": open_.iloc[-1],
         "rsi14": screen.calc_rsi(close)}
    for n in screen.MA_PERIODS:
        r[f"sma{n}"] = sma[n].iloc[-1]

    vol_avg20 = volume.rolling(20).mean().iloc[-1]
    ratio = (volume.iloc[-1] / vol_avg20) if vol_avg20 and vol_avg20 > 0 else None
    r["volume_confirmed"] = bool(ratio is not None and ratio >= 1.5)

    up = sum(sma[screen.MA_PERIODS[i]].iloc[-1] > sma[screen.MA_PERIODS[i + 1]].iloc[-1]
             for i in range(len(screen.MA_PERIODS) - 1))
    r["ppp_matches"] = up
    r["trend_filter_pass"] = (bool(up >= 3 and r["last_close"] > sma[100].iloc[-1])
                              if pd.notna(sma[100].iloc[-1]) else False)

    s5 = sma[5]
    r["sma5_slope_up"] = s5.iloc[-1] > s5.iloc[-5]
    prev_close, prev_s5 = close.iloc[-2], s5.iloc[-2]
    bullish = r["last_close"] > r["last_open"]
    r["kahanshin"] = bool(prev_close <= prev_s5 and r["last_close"] > r["sma5"]
                          and bullish and r["sma5_slope_up"])
    s20 = sma[20].iloc[-1]
    r["pullback"] = bool(pd.notna(s20)
                         and low.iloc[-1] <= s20 * (1 + screen.PULLBACK_TOLERANCE_PCT / 100)
                         and bullish and r["last_close"] > s20)

    # 9の法則（下落レグの本数）
    _, buy = screen.swing_leg_count(close, s5)
    r["td_buy"] = buy
    # くちばし・ものわかれ・節目
    r["kuchibashi_signal"] = screen.detect_kuchibashi(close).get("signal")
    r["monowakare_signal"] = screen.detect_monowakare(close, sma).get("signal")
    r["fushime_breakout_level"] = screen.detect_fushime(close).get("breakout_level")
    return r


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    w = sub[sub["return_pct"] > 0]["return_pct"].sum()
    l = abs(sub[sub["return_pct"] <= 0]["return_pct"].sum())
    return {"件数": len(sub),
            "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
            "平均%": round(sub["return_pct"].mean(), 2),
            "PF": round(w / l, 2) if l else float("inf")}


def main():
    argv = sys.argv[1:]
    # --limit の「値」を位置引数と取り違えないように、フラグとその値を除く
    skip = set()
    for i, a in enumerate(argv):
        if a == "--limit":
            skip.update({i, i + 1})
    args = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("--")]
    path = Path(args[0]) if args else BASE_DIR / DEFAULT_TRADES
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None

    df = pd.read_csv(path)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    print(f"対象: {path.name} / {len(df):,}トレード")

    codes = sorted(df["code"].unique())
    print(f"株価を読み込みます（{len(codes)}銘柄）…")
    hist_map = fetch_histories(codes, period="max", verbose=False)

    rows = []
    for i, (_, t) in enumerate(df.iterrows(), 1):
        h = hist_map.get(t["code"])
        if h is None or h.empty:
            continue
        entry = t["entry_date"]
        if h.index.tz is not None and entry.tz is None:
            entry = entry.tz_localize(h.index.tz)
        sliced = h.loc[:entry]
        try:
            r = row_at(sliced)
        except (IndexError, KeyError, ValueError):
            continue
        if r is None:
            continue
        r["score"] = screen.technical_score(r)
        r["return_pct"] = t["return_pct"]
        r["entry_date"] = t["entry_date"]
        rows.append(r)
        if i % 2000 == 0:
            print(f"  {i:,}/{len(df):,} 件")

    r = pd.DataFrame(rows)
    if not limit:
        # 全件で回したときだけキャッシュを更新する（サンプルで上書きしない）
        r.to_csv(CACHE_PATH, index=False)
        print(f"\n再現結果を保存しました: {CACHE_PATH.name}")
    print(f"再現できたトレード: {len(r):,}件")
    print(f"期間: {r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}\n")

    print("=== テクニカルスコアの帯別（スコアが高いほど良いはず）===")
    r["帯"] = pd.qcut(r["score"], 5, labels=["最低", "低", "中", "高", "最高"],
                     duplicates="drop")
    g = {str(k): stats(v) for k, v in r.groupby("帯", observed=True)}
    print(pd.DataFrame({k: v for k, v in g.items() if v}).T.to_string())
    corr = r["score"].corr(r["return_pct"])
    print(f"\n  スコアとリターンの相関: {corr:+.3f}")
    print()

    print("=== 加点項目ごとの寄与（成立した組 vs しなかった組）===")
    items = [
        ("RSI 40〜70（+0.20）", (r["rsi14"] >= 40) & (r["rsi14"] <= 70)),
        ("9の法則 td_buy>=7（+0.10〜0.35）", r["td_buy"] >= 7),
        ("9の法則 td_buy>=9（+0.20〜）", r["td_buy"] >= 9),
        ("くちばし成立（+0.30）", r["kuchibashi_signal"] == "up"),
        ("節目突破（+0.15）", r["fushime_breakout_level"].notna()),
        ("ものわかれ（+0.05）", r["monowakare_signal"] == "up"),
        ("PPP完成（4/4）（+0.25）", r["ppp_matches"] == 4),
        ("出来高1.5倍（+0.10）", r["volume_confirmed"]),
        ("トレンド条件（+0.08）", r["trend_filter_pass"]),
    ]
    out = {}
    for label, mask in items:
        a, b = stats(r[mask]), stats(r[~mask])
        if not a or not b or a["件数"] < 50 or b["件数"] < 50:
            continue
        out[label] = {"成立 件数": a["件数"], "成立 PF": a["PF"], "成立 勝率%": a["勝率%"],
                      "不成立 PF": b["PF"], "不成立 勝率%": b["勝率%"],
                      "PF差": round(a["PF"] - b["PF"], 2)}
    print(pd.DataFrame(out).T.to_string())
    print()
    print("PF差がマイナスの項目は、加点しているのに成績が悪い＝順位付けのノイズ。")
    print("配点の見直しを検討すること（REQUIREMENTS に記録）。")


if __name__ == "__main__":
    main()
