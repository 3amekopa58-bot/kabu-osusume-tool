"""区分を総当たりで検証し、偶然で説明できないものだけを残す。

⚠️ **この道具の主目的は「良い区分を見つけること」ではなく、
   「見つけたつもりになるのを防ぐこと」。**
   区分を何十通りも切れば、26年のデータでも偶然で必ずいくつか当たる。
   4.4-5（業種）4.4-47（多重検定）4.4-53（銘柄のくせ）4.4-62（月別）は
   すべて「集計すると差はあるが将来に使えない」で終わっている。

判定のしかた（プロジェクトの採用基準そのまま）:
  ① 重複しない3期間すべてで、全体平均より良い（または悪い）こと
  ② ①を満たす区分の数が、**偶然の期待個数を明確に上回る**こと
     期待個数は各期間の「全体超えの割合」の実測から出す
     （符号を五分五分と仮定すると過小評価になる＝4.4-62 で踏んだ罠）
  ③ 件数が足りること（MIN_TRADES未満は判断しない）

使い方:
    ./venv/bin/python segment_lab.py            # 全区分を検証
    ./venv/bin/python segment_lab.py 業種        # 1区分だけ
出力:
    output/segment_lab_summary.csv   区分×バケツの全成績
    output/segment_lab_survivors.csv 3期間を通過したものだけ
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
TRADES = B / "output" / "_universe_max_trades.csv"
FEATURES = B / "output" / "segment_features.csv"   # build_segment_features.py が作る
SECTORS = B / "data" / "sectors.json"

# 株価だけで作れる区分は26年を3等分できる
PERIODS_PRICE = [("第1期 2000-2008", 2000, 2008),
                 ("第2期 2009-2017", 2009, 2017),
                 ("第3期 2018-2026", 2018, 2026)]

# ⚠️ ファンダメンタルは2013年より前が存在しない（EDINETのAPIが約10年
#    ローリング＋有報の5年表で2012年頃が限界）。26年で3分割すると
#    第1期・第2期が空になり**判定不能**になるので、データのある範囲を
#    3等分する。期間は短くなるが、3つとも本物の期間外検証になる（4.4-64）。
PERIODS_FUND = [("F1期 2013-2017", 2013, 2017),
                ("F2期 2018-2021", 2018, 2021),
                ("F3期 2022-2026", 2022, 2026)]

# ファンダメンタル由来＝短い期間を使う区分
FUND_COLS = {"時価総額帯", "PBR帯", "PER帯"}


def periods_for(col: str):
    return PERIODS_FUND if col in FUND_COLS else PERIODS_PRICE
MIN_TRADES = 120          # 1バケツ・1期間あたりの最低件数
MIN_TRADES_PERIOD = 30    # 各期間での最低件数


def pf(s: pd.Series) -> float:
    w = s[s > 0].sum()
    l = -s[s < 0].sum()
    return w / l if l > 0 else np.nan


def load() -> pd.DataFrame:
    d = pd.read_csv(TRADES)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["年"] = d["entry_date"].dt.year

    # --- 区分①〜③: トレード明細だけで作れるもの（追加データ不要） ---
    d["月"] = d["entry_date"].dt.month
    d["曜日"] = d["entry_date"].dt.dayofweek.map(
        {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"})
    d["エントリー価格帯"] = pd.cut(
        d["entry_price"], [0, 500, 1000, 2000, 5000, np.inf],
        labels=["〜500円", "500-1000", "1000-2000", "2000-5000", "5000円〜"])

    if SECTORS.exists():
        sec = json.loads(SECTORS.read_text(encoding="utf-8"))
        d["業種"] = d["code"].map(sec).fillna("Unknown")

    if FEATURES.exists():
        f = pd.read_csv(FEATURES)
        f["entry_date"] = pd.to_datetime(f["entry_date"])
        d = d.merge(f, on=["code", "entry_date"], how="left")
    return d


def evaluate(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """1つの区分について、バケツごとの成績を3期間ぶん出す。

    期間の定義は区分によって違う（ファンダは2013年以降しか無いため）。
    """
    P = periods_for(col)
    # ファンダ区分は、その値が取れているトレードだけを母集団にする。
    # 取れていない行を混ぜると「全体平均」が別物になり比較にならない。
    if col in FUND_COLS:
        d = d[d[col].notna()]
    base = {lab: d[(d["年"] >= a) & (d["年"] <= b)]["return_pct"].mean()
            for lab, a, b in P}
    base_all = d["return_pct"].mean()

    rows = []
    for name, g in d.groupby(col, observed=True):
        if len(g) < MIN_TRADES:
            continue
        r = {"区分": col, "バケツ": name, "件数": len(g),
             "平均%": g["return_pct"].mean(), "勝率%": (g["return_pct"] > 0).mean() * 100,
             "PF": pf(g["return_pct"]), "全体との差": g["return_pct"].mean() - base_all}
        ok_up = ok_dn = True
        enough = True
        for lab, a, b in P:
            s = g[(g["年"] >= a) & (g["年"] <= b)]["return_pct"]
            if len(s) < MIN_TRADES_PERIOD:
                enough = False
                r[lab] = np.nan
                continue
            diff = s.mean() - base[lab]
            r[lab] = diff
            ok_up &= diff > 0
            ok_dn &= diff < 0
        r["期間定義"] = "／".join(lab for lab, _, _ in P)
        r["3期間とも上"] = bool(enough and ok_up)
        r["3期間とも下"] = bool(enough and ok_dn)
        r["判定可"] = enough
        rows.append(r)
    return pd.DataFrame(rows)


def monotonic_p(t: pd.DataFrame, col: str) -> float:
    """順序のある区分（Q1..Q4 のような帯）で、成績が単調に並ぶ確率。

    個数の一致より強い証拠になる。k個のバケツが偶然に単調（昇順or降順）に
    並ぶ確率は 2/k! なので、それを返す。順序が無い区分（業種・曜日）には
    使えないので None を返す。
    """
    import math
    if not col.endswith("帯"):
        return float("nan")
    t = t[t["判定可"]] if "判定可" in t else t
    names = [str(x) for x in t["バケツ"]]
    if not all("_Q" in n for n in names) or len(names) < 3:
        return float("nan")
    order = sorted(range(len(names)), key=lambda i: int(names[i].split("_Q")[1]))
    vals = [t["平均%"].iloc[i] for i in order]
    up = all(a < b for a, b in zip(vals, vals[1:]))
    dn = all(a > b for a, b in zip(vals, vals[1:]))
    if not (up or dn):
        return float("nan")
    return 2 / math.factorial(len(vals))


def chance_expectation(t: pd.DataFrame, col: str = "") -> tuple:
    """偶然でいくつ通過するかの期待個数。

    各期間で「全体平均を上回るバケツ」の実測割合を p_i とし、
    独立と仮定して p1*p2*p3 × バケツ数。符号を0.5と決め打ちしない。
    """
    t = t[t["判定可"]]
    if len(t) == 0:
        return 0.0, 0.0, 0
    ps = [(t[lab] > 0).mean() for lab, _, _ in periods_for(col)]
    n = len(t)
    return n * np.prod(ps), n * np.prod([1 - p for p in ps]), n


def control_for_price(d: pd.DataFrame, col: str) -> None:
    """通過した区分を、既知の交絡（株価の水準）で層別して見直す。

    ⚠️ 4.4-49：yfinance の調整後株価では「過去に安かった株」＝その後分割した株
       ＝上がった株になる。**低位株効果は後知恵。** 新しい区分が当たったとき、
       それが低位株の言い換えでないかを必ず確認する。
       効果が価格帯を上がるにつれ縮む／反転するなら、独立した効果ではない。
    """
    # ⚠️ 両端の差に意味があるのは**順序のある帯だけ**。
    #    月・曜日・業種に「1月と12月の差」という概念はないので当てない。
    if not col.endswith("帯") or col == "エントリー価格帯":
        return
    if col not in d.columns or "entry_price" not in d.columns:
        return
    band = pd.cut(d["entry_price"], [0, 500, 1000, 2000, np.inf],
                  labels=["〜500円", "500-1000", "1000-2000", "2000円〜"])
    t = d.assign(価格帯=band).pivot_table(
        index="価格帯", columns=col, values="return_pct",
        aggfunc="mean", observed=True)
    if t.shape[1] < 2:
        return
    first, last = t.columns[0], t.columns[-1]
    print(f"  【交絡チェック】株価の水準で層別（{col} の両端の差）")
    diffs = []
    for b in t.index:
        if pd.notna(t.loc[b, first]) and pd.notna(t.loc[b, last]):
            gap = t.loc[b, first] - t.loc[b, last]
            diffs.append(gap)
            print(f"    {b}: {gap:+.2f}pt")
    # ⚠️ 判定は**向き**を見る。低位株の後知恵（4.4-49）の特徴は
    #    「最も安い帯で効果が最大、価格が上がるにつれ縮む／反転する」。
    #    逆に安い帯で効果が無く高い帯で強いなら、それは低位株の言い換えではない。
    #    符号が反転したというだけで不採用にしない（2026-09-20に踏んだ誤判定）。
    if len(diffs) < 3:
        print("    → 層が足りず判定できない")
        return
    biggest_is_cheapest = abs(diffs[0]) >= max(abs(x) for x in diffs) - 1e-9
    decays = abs(diffs[-1]) < abs(diffs[0]) / 2 or diffs[0] * diffs[-1] < 0
    if biggest_is_cheapest and decays:
        print("    → **最も安い帯で効果が最大、価格とともに縮む／反転する"
              "＝低位株の言い換え。独立した効果ではない（4.4-49）。不採用。**")
    elif biggest_is_cheapest:
        print("    → 安い帯で最大だが減衰は弱い。低位株との重なりを要確認。")
    else:
        print("    → **安い帯で最大ではない＝低位株の言い換えではない。**"
              "この交絡では否定できない＝追加検証に進む価値あり。")


def main() -> None:
    d = load()
    want = sys.argv[1] if len(sys.argv) > 1 else None

    cands = ["月", "曜日", "エントリー価格帯", "業種", "時価総額帯", "売買代金帯",
             "PBR帯", "PER帯", "ボラ帯", "200日線乖離帯", "1年騰落帯", "相場環境"]
    cands = [c for c in cands if c in d.columns]
    if want:
        cands = [c for c in cands if c == want] or [want]

    print(f"トレード {len(d):,}件 / 検証する区分 {len(cands)}個: {cands}\n")
    print(f"全体の平均リターン {d['return_pct'].mean():+.2f}% / "
          f"勝率 {(d['return_pct']>0).mean()*100:.1f}% / PF {pf(d['return_pct']):.2f}\n")

    all_t, survivors = [], []
    for c in cands:
        t = evaluate(d, c)
        if len(t) == 0:
            print(f"--- {c}: 件数不足で判定できず ---\n")
            continue
        exp_up, exp_dn, n = chance_expectation(t, c)
        up = t["3期間とも上"].sum()
        dn = t["3期間とも下"].sum()
        verdict = ("**偶然を上回る**" if up > exp_up + 1 or dn > exp_dn + 1
                   else "偶然の範囲")
        P = periods_for(c)
        print(f"--- {c}（判定可 {n}バケツ／期間 "
              f"{'／'.join(lab for lab, _, _ in P)}）---")
        print(f"  3期間とも上: {up}個（偶然の期待 {exp_up:.1f}個）／"
              f"3期間とも下: {dn}個（期待 {exp_dn:.1f}個） → {verdict}")
        mp = monotonic_p(t, c)
        if mp == mp:   # NaNでない＝単調に並んでいる
            print(f"  ⭐ バケツが**単調**に並んでいる（偶然にこうなる確率 {mp*100:.1f}%）"
                  f"＝個数の一致より強い証拠")
        show = t.sort_values("全体との差", ascending=False)
        cols = ["バケツ", "件数", "平均%", "勝率%", "PF", "全体との差",
                "3期間とも上", "3期間とも下"]
        print(show[cols].to_string(index=False, float_format="%.2f"))
        print()
        if verdict.startswith("**") or (mp == mp):
            control_for_price(d, c)
            print()
        all_t.append(t)
        s = t[(t["3期間とも上"]) | (t["3期間とも下"])].copy()
        s["偶然の期待_上"] = exp_up
        s["偶然の期待_下"] = exp_dn
        survivors.append(s)

    if all_t:
        pd.concat(all_t).to_csv(B / "output" / "segment_lab_summary.csv",
                                index=False, encoding="utf-8-sig")
        sv = pd.concat(survivors) if survivors else pd.DataFrame()
        sv.to_csv(B / "output" / "segment_lab_survivors.csv",
                  index=False, encoding="utf-8-sig")
        print(f"通過したバケツ: 合計 {len(sv)}個 → segment_lab_survivors.csv")
        print("\n⚠️ 通過＝採用ではない。各区分の『偶然の期待』と見比べ、"
              "明確に上回っていなければ採用しない。")


if __name__ == "__main__":
    main()
