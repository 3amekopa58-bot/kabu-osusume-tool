"""
指定した銘柄の「チャートのくせ」を、ユニバース内での位置づけとともに出す

⚠️ **これは選別の道具ではない。** 4.4-51〜4.4-53 で、銘柄ごとのくせを
   使って「買う銘柄を選ぶ」ことはできないと確認済み：
     ・銘柄ごとの成績は次の期間に持続しない（むしろ逆転する）
     ・持続する性質（荒さ・日経連動）は成績を予測しないか、相場の方向に従属する
     ・14種類×4回の引き継ぎで総当たりしても、多重検定に耐える候補はゼロ

   ここで出すのは**その銘柄がどういう値動きをする銘柄なのかの説明**であって、
   将来の成績の予測ではない。「この性質だから買う／避ける」には使えない。

出すもの:
  ・14種類の性質と、ユニバース944銘柄の中での順位（パーセンタイル）
  ・5年窓ごとの推移（その性質が安定しているのか、変わってきたのか）
  ・現行ルールがその銘柄で出したトレードの実績（26年）
  ・ユニバースの上位/下位10%に入る、際立った特徴

使い方:
    python3 stock_profile.py 7974            # 証券コード
    python3 stock_profile.py 7974.T 6758     # 複数可
    python3 stock_profile.py 任天堂           # 銘柄名の一部でも可
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fetch_nikkei_close
from price_cache import fetch_histories
from trade_data import load_trades

BASE_DIR = Path(__file__).parent
UNIVERSE = BASE_DIR / "universe.csv"

WINDOWS = [
    ("2000-2005", "2000-01-01", "2004-12-31"),
    ("2005-2010", "2005-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2014-12-31"),
    ("2015-2020", "2015-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
]
MIN_DAYS = 400

# (名前, 説明, 高いとどういう銘柄か)
TRAITS = [
    ("ボラ",       "年率ボラティリティ(%)",        "値動きが荒い"),
    ("効率比",     "|正味変化|÷|日々の変化の合計|", "一方向に素直に動く"),
    ("自己相関1",  "1日リターンのラグ1自己相関",    "前日の動きを引き継ぐ"),
    ("分散比20",   "20日分散÷(20×1日分散)",        "トレンドが伸びる"),
    ("回帰半減期", "20日線への戻りの半減期(日)",    "戻るのが遅い"),
    ("ジャンプ率", "1日±5%超の日の割合(%)",        "材料でよく飛ぶ"),
    ("歪度",       "日次リターンの歪み",            "たまに大きく上がる"),
    ("尖度",       "とがり具合",                    "ふだん静かでたまに飛ぶ"),
    ("日中値幅",   "(高値−安値)÷終値 の平均(%)",   "ザラ場の振れが大きい"),
    ("出来高変動", "出来高の標準偏差÷平均",         "出来高が不安定"),
    ("日経β",     "日経に対する感応度",            "日経に大きく連動"),
    ("個別要因",   "1−R²（日経で説明できない部分）", "独自の材料で動く"),
    ("上昇連続",   "連続して上がる日数の平均",      "上げ続けやすい"),
]


def compute(o, h, l, c, v, nk):
    r = c.pct_change().dropna()
    if len(r) < MIN_DAYS:
        return None
    ann = np.sqrt(252)
    k = 20
    # 株価に0や欠損があると log が警告を出すので正の値だけで計算する
    cp = c[c > 0]
    rk = np.log(cp).diff(k).dropna() if len(cp) > k else pd.Series(dtype=float)
    vr = (rk.var() / (k * r.var())) if (len(rk) > 10 and r.var() > 0) else np.nan
    ma20 = c.rolling(20).mean()
    dev = ((c - ma20) / ma20).dropna()
    rho = dev.autocorr(lag=1) if len(dev) > 60 else np.nan
    half = (-np.log(2) / np.log(rho)) if (rho is not None and 0 < rho < 1) else np.nan
    nka = nk.reindex(c.index, method="ffill")
    both = pd.concat([r, nka.pct_change()], axis=1).dropna()
    beta = idio = np.nan
    if len(both) > 100 and both.iloc[:, 1].var() > 0:
        beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
        corr = both.iloc[:, 0].corr(both.iloc[:, 1])
        idio = 1 - corr ** 2 if pd.notna(corr) else np.nan
    up = (r > 0).astype(int).values
    runs, cur = [], 0
    for x in up:
        if x:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    net, path = abs(c.iloc[-1] - c.iloc[0]), c.diff().abs().sum()
    return {
        "ボラ": r.std() * ann * 100,
        "効率比": net / path if path else np.nan,
        "自己相関1": r.autocorr(lag=1),
        "分散比20": vr,
        "回帰半減期": half,
        "ジャンプ率": (r.abs() > 0.05).mean() * 100,
        "歪度": r.skew(),
        "尖度": r.kurt(),
        "日中値幅": ((h - l) / c).mean() * 100,
        "出来高変動": v.std() / v.mean() if v.mean() else np.nan,
        "日経β": beta,
        "個別要因": idio,
        "上昇連続": float(np.mean(runs)) if runs else np.nan,
    }


def load_universe():
    with open(UNIVERSE, encoding="utf-8-sig") as f:
        return {r["code"]: r["name"] for r in csv.DictReader(f)}


def resolve(arg, names):
    """コード・コード.T・名前の一部 のどれでも受ける"""
    a = arg.strip()
    if a in names:
        return [a]
    if f"{a}.T" in names:
        return [f"{a}.T"]
    hits = [c for c, n in names.items() if a in n]
    return hits


def bar(pct):
    """パーセンタイルを見た目で示す"""
    n = int(round(pct / 10))
    return "▁" * max(0, n) + "█" + "▁" * max(0, 10 - n)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        return
    names = load_universe()
    targets = []
    for a in args:
        hits = resolve(a, names)
        if not hits:
            print(f"⚠️ 「{a}」に該当する銘柄がユニバースにありません")
        targets.extend(hits)
    if not targets:
        return

    all_codes = sorted(names)
    hist = fetch_histories(all_codes, period="max")
    nk = fetch_nikkei_close("max")
    if getattr(nk.index, "tz", None) is not None:
        nk.index = nk.index.tz_localize(None)

    px = {}
    for code in all_codes:
        d = hist.get(code)
        if d is None or len(d) < MIN_DAYS:
            continue
        idx = d.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        px[code] = pd.DataFrame(
            {"o": d["Open"].values, "h": d["High"].values, "l": d["Low"].values,
             "c": d["Close"].values, "v": d["Volume"].values}, index=idx)

    # 窓ごとに全銘柄の性質（パーセンタイルを出すために全体が要る）
    T = {}
    for lab, lo, hi in WINDOWS:
        rows = {}
        for code, d in px.items():
            w = d[(d.index >= pd.Timestamp(lo)) & (d.index <= pd.Timestamp(hi))]
            if len(w) < MIN_DAYS:
                continue
            t = compute(w["o"], w["h"], w["l"], w["c"], w["v"], nk)
            if t:
                rows[code] = t
        T[lab] = pd.DataFrame(rows).T

    tr = load_trades()
    latest = WINDOWS[-1][0]

    for code in targets:
        print("\n" + "=" * 72)
        print(f"  {names[code]}（{code}）")
        print("=" * 72)

        if code not in T[latest].index:
            print("  直近5年の株価データが足りず、性質を出せません")
            continue

        cur = T[latest].loc[code]
        pop = T[latest]
        print(f"\n【直近5年（{latest}）の性質】"
              f"  ユニバース{len(pop)}銘柄の中での位置")
        print(f"  {'性質':<10}{'値':>9}  {'順位':>5}  "
              f"{'低い ← → 高い':<13} 高いとどういう銘柄か")
        highlights = []
        for name, desc, high_means in TRAITS:
            v = cur.get(name)
            if not np.isfinite(v):
                continue
            pct = (pop[name] < v).mean() * 100
            print(f"  {name:<10}{v:>9.2f}  {pct:>4.0f}%  {bar(pct):<13} {high_means}")
            if pct >= 90:
                highlights.append(f"**{name}が上位10%**（{high_means}）")
            elif pct <= 10:
                highlights.append(f"**{name}が下位10%**（{high_means}の逆）")

        print(f"\n【際立った特徴】")
        if highlights:
            for h in highlights:
                print(f"  ・{h}")
        else:
            print("  ・ユニバースの上位/下位10%に入る性質はなし＝平均的な値動き")

        print(f"\n【5年窓ごとの推移】その性質が安定しているか")
        wl = [w[0] for w in WINDOWS]
        print(f"  {'性質':<10}" + "".join(f"{w:>11}" for w in wl))
        for name, _, _ in TRAITS:
            line = f"  {name:<10}"
            for w in wl:
                if code in T[w].index and np.isfinite(T[w].loc[code, name]):
                    line += f"{T[w].loc[code, name]:>11.2f}"
                else:
                    line += f"{'-':>11}"
            print(line)

        s = tr[tr["code"] == code]
        print(f"\n【現行ルールでの実績（26年）】")
        if len(s) == 0:
            print("  この銘柄ではシグナルが出ていません")
        else:
            g = s["return_pct"][s["return_pct"] > 0].sum()
            l_ = -s["return_pct"][s["return_pct"] < 0].sum()
            print(f"  トレード {len(s)}件 / 勝率 {(s['return_pct']>0).mean()*100:.1f}% "
                  f"/ 平均 {s['return_pct'].mean():+.2f}% / "
                  f"PF {g/l_ if l_ else float('inf'):.2f}")
            print(f"  最大の勝ち {s['return_pct'].max():+.1f}% / "
                  f"最大の負け {s['return_pct'].min():+.1f}%")
            print(f"  （全体: 勝率51.8% / 平均+3.14% / PF1.73）")
            print(f"\n  ⚠️ この実績は**将来の成績を示さない**。4.4-51 で、"
                  f"銘柄ごとの成績は\n"
                  f"     次の期間に持続せず、むしろ逆転する（前半最下位の"
                  f"銘柄群が後半最良）\n"
                  f"     ことを確認済み。銘柄選別の根拠には使えない。")


if __name__ == "__main__":
    main()
