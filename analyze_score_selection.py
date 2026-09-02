"""
スコアで銘柄を絞ったら成績が上がるのか／配点を変えると良くなるのかを測る

`screen.py` の総合スコアは推奨の**並び順**を決めているが、
4.4-26 の検証で「効いているのは最高帯だけ」「RSIとPPP完成は逆効果」
と分かった。ただし**配点を変えても良くなったか判定する枠組みが無い**ため、
変更を見送っていた。その枠組みをここで用意する。

考え方：
  実際の運用では「その日の候補のうち上位N件を買う」という使い方をする。
  そこで各トレードを**エントリー日ごとにグループ化してスコア順に並べ、
  上位N件だけを買った場合**の成績を測る。資金シミュレーション
  （portfolio_sim.py）は経路依存が強く点推定に意味がないが、
  この方法は決定的（同じ入力なら同じ答え）なので配点の比較に使える。

  ⚠️ ただし「その日の候補」＝バックテストでシグナルが出た銘柄に限られる。
     実際の screen.py は全944銘柄を並べるので、母集団は完全には一致しない。

使い方:
    python3 analyze_score_selection.py            # 現行配点で上位N件の効果
    python3 analyze_score_selection.py --weights  # 配点を変えた場合の比較
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "data" / "technical_rows.csv"


def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        print("キャッシュがありません。先に "
              "python3 analyze_technical_score.py（全件）を実行してください。")
        sys.exit(1)
    r = pd.read_csv(CACHE_PATH)
    r["entry_date"] = pd.to_datetime(r["entry_date"])
    return r


def score_with(r: pd.DataFrame, w: dict) -> pd.Series:
    """
    配点を差し替えてスコアを計算し直す。
    screen.py の technical_score と同じ構造だが、各項目の重みを引数で渡せる。
    重みを0にすればその項目を無効化した場合が測れる。
    """
    s = (r["ppp_matches"] / 4) * w["ppp"]
    sig = r["kahanshin"].astype(bool) | r["pullback"].astype(bool)
    bonus = pd.Series(0.0, index=r.index)
    bonus += w["signal"]
    bonus += r["trend_filter_pass"].astype(bool) * w["trend"]
    bonus += r["volume_confirmed"].astype(bool) * w["volume"]
    bonus += (r["monowakare_signal"] == "up") * w["monowakare"]
    s = s + sig * bonus
    s += ((r["rsi14"] >= 40) & (r["rsi14"] <= 70)) * w["rsi"]
    # 9の法則は段階的な加点
    td = r["td_buy"]
    s += (td == 23) * w["td23"] + (td == 17) * w["td17"] + (td == 9) * w["td9"] \
        + td.isin([7, 8]) * w["td78"]
    s += (r["kuchibashi_signal"] == "up") * w["kuchibashi"]
    s += r["fushime_breakout_level"].notna() * w["fushime"]
    return s


# screen.py の現行配点
CURRENT = {"ppp": 0.25, "signal": 0.10, "trend": 0.08, "volume": 0.10,
           "monowakare": 0.05, "rsi": 0.20, "td23": 0.35, "td17": 0.25,
           "td9": 0.20, "td78": 0.10, "kuchibashi": 0.30, "fushime": 0.15}


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    w = sub[sub["return_pct"] > 0]["return_pct"].sum()
    l = abs(sub[sub["return_pct"] <= 0]["return_pct"].sum())
    return {"件数": len(sub),
            "勝率%": round((sub["return_pct"] > 0).mean() * 100, 1),
            "平均%": round(sub["return_pct"].mean(), 2),
            "PF": round(w / l, 2) if l else float("inf")}


def top_n_per_day(r: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """エントリー日ごとに col の降順で上位n件だけ残す"""
    return (r.sort_values(col, ascending=False)
             .groupby("entry_date", group_keys=False).head(n))


def eras(r: pd.DataFrame):
    edges = [r["entry_date"].quantile(x) for x in (1 / 3, 2 / 3)]
    def era(d):
        return 0 if d <= edges[0] else (1 if d <= edges[1] else 2)
    return r["entry_date"].apply(era)


def main():
    r = load_cache()
    r["era"] = eras(r)
    r["score_cur"] = score_with(r, CURRENT)
    print(f"対象: {len(r):,}トレード / "
          f"{r['entry_date'].min().date()} 〜 {r['entry_date'].max().date()}")
    print(f"1日あたりの候補数: 中央値 "
          f"{r.groupby('entry_date').size().median():.0f}件\n")

    if "--weights" not in sys.argv:
        print("=== スコア上位N件だけを買った場合（現行配点）===")
        out = {"全件（絞らない）": stats(r)}
        for n in (1, 2, 3, 5, 10):
            out[f"上位{n}件/日"] = stats(top_n_per_day(r, "score_cur", n))
        print(pd.DataFrame(out).T.to_string())
        print("\n=== 重複しない3期間（上位3件/日）===")
        for i in range(3):
            p = r[r.era == i]
            a, b = stats(p), stats(top_n_per_day(p, "score_cur", 3))
            if a and b:
                print(f"  {p.entry_date.min().date()}〜{p.entry_date.max().date()}  "
                      f"全件 PF{a['PF']:.2f}({a['件数']:,}件) → "
                      f"上位3件 PF{b['PF']:.2f}({b['件数']:,}件)  "
                      f"勝率 {a['勝率%']:.1f}%→{b['勝率%']:.1f}%")
        return

    print("=== 配点を変えたときの比較（上位3件/日で評価）===")
    variants = {
        "現行": CURRENT,
        "RSIを0に": {**CURRENT, "rsi": 0.0},
        "PPP完成を0に": {**CURRENT, "ppp": 0.0},
        "RSIとPPPを0に": {**CURRENT, "rsi": 0.0, "ppp": 0.0},
        "効く項目を厚く": {**CURRENT, "rsi": 0.0, "ppp": 0.0,
                     "kuchibashi": 0.40, "fushime": 0.25},
    }
    rows = {}
    for label, w in variants.items():
        r["_s"] = score_with(r, w)
        sel = top_n_per_day(r, "_s", 3)
        st = stats(sel)
        per_era = []
        ok = True
        for i in range(3):
            p = r[r.era == i]
            e = stats(top_n_per_day(p, "_s", 3))
            per_era.append(e["PF"] if e else float("nan"))
        rows[label] = {**st, "第1期PF": per_era[0], "第2期PF": per_era[1],
                       "第3期PF": per_era[2]}
    print(pd.DataFrame(rows).T.to_string())
    print("\n採用基準：重複しない3期間すべてで現行を上回って初めて配点を変える。")


if __name__ == "__main__":
    main()
