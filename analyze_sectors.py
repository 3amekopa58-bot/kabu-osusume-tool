"""
業種別の分析：シグナルの偏り・業種ごとの成績・時代による変化を調べる

backtest.py が出力したトレード明細に業種情報を紐付け、
  ①シグナルがどの業種に偏っているか
  ②業種によって勝率・リターンがどう違うか
  ③時代（年代）によってその傾向がどう変わるか
を集計する。業種はyfinanceから取得して data/sectors.json にキャッシュする
（初回のみ225銘柄ぶんのAPI呼び出しが走るので数分かかる）。

使い方:
    python analyze_sectors.py [トレード明細CSVのパス]
      省略時は27年・採用ルールの明細を使う
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
SECTOR_CACHE = BASE_DIR / "data" / "sectors.json"
# ⚠️ 中間ファイルを名指しで固定すると、コードや母集団を変えたときに
# 追随せず、古い成果物で測り続けることになる（2026-09-05に
# analyze_targets.py と analyze_sectors.py で実際に起きた＝4.4-56）。
# 現行ルールの正規のトレード明細を指す。無ければ次で作る:
#   python3 backtest.py timesl either trend marketadx volume rs sl10 max \
#           --tickers universe.csv
DEFAULT_TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"

# 異常値（分割データ不整合）を除外する閾値
SUSPICIOUS_RETURN_THRESHOLD = 500.0
# 集計対象とする最低トレード数（これ未満の業種は統計的に判断できない）
MIN_TRADES_FOR_STATS = 30

# yfinanceの英語セクター名を日本語に読み替える
SECTOR_JA = {
    "Industrials": "資本財・工業",
    "Consumer Cyclical": "一般消費財",
    "Consumer Defensive": "生活必需品",
    "Technology": "テクノロジー",
    "Financial Services": "金融",
    "Basic Materials": "素材",
    "Healthcare": "ヘルスケア",
    "Communication Services": "通信サービス",
    "Energy": "エネルギー",
    "Utilities": "公益",
    "Real Estate": "不動産",
    "Unknown": "不明",
}


def load_tickers():
    with open(TICKERS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_or_fetch_sectors(codes) -> dict:
    """業種をキャッシュから読み、無い銘柄だけyfinanceに問い合わせる。"""
    SECTOR_CACHE.parent.mkdir(exist_ok=True)
    cache = {}
    if SECTOR_CACHE.exists():
        cache = json.loads(SECTOR_CACHE.read_text(encoding="utf-8"))

    missing = [c for c in codes if c not in cache]
    if missing:
        print(f"{len(missing)}銘柄の業種をyfinanceから取得します（数分かかります）…")
        for i, code in enumerate(missing, 1):
            try:
                cache[code] = yf.Ticker(code).info.get("sector") or "Unknown"
            except Exception:
                cache[code] = "Unknown"
            if i % 25 == 0:
                print(f"  {i}/{len(missing)}件")
                SECTOR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        SECTOR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"  → {SECTOR_CACHE} にキャッシュしました")
    return cache


def stats(d: pd.Series) -> tuple:
    """勝率・PF・平均リターンを返す。"""
    w = d[d > 0].sum()
    l = d[d <= 0].sum()
    pf = w / abs(l) if l != 0 else float("inf")
    return (d > 0).mean() * 100, pf, d.mean()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRADES
    if not path.exists():
        print(f"トレード明細が見つかりません: {path}")
        return

    df = pd.read_csv(path)
    before = len(df)
    df = df[df["return_pct"].abs() <= SUSPICIOUS_RETURN_THRESHOLD]
    if len(df) < before:
        print(f"⚠️  異常値{before - len(df)}件を除外しました（分割データ不整合の疑い）")

    tickers = load_tickers()
    sectors = load_or_fetch_sectors([t["code"] for t in tickers])
    df["sector"] = df["code"].map(lambda c: SECTOR_JA.get(sectors.get(c, "Unknown"),
                                                          sectors.get(c, "Unknown")))
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year

    # 母集団の業種構成（銘柄数ベース）と、シグナルの業種構成を比べる
    universe = pd.Series([SECTOR_JA.get(sectors.get(t["code"], "Unknown"),
                                        sectors.get(t["code"], "Unknown"))
                          for t in tickers]).value_counts()

    print(f"\n対象: {path.name}")
    print(f"トレード数: {len(df)}件 / 期間: {df['year'].min()}〜{df['year'].max()}年\n")

    print("=" * 78)
    print("① シグナルの業種偏り（母集団の構成比とシグナルの構成比を比較）")
    print("=" * 78)
    sig_share = df["sector"].value_counts()
    print(f'{"業種":<14}{"銘柄数":>6}{"母集団比":>9}{"トレード数":>8}{"シグナル比":>10}{"偏り":>8}')
    rows = []
    for sec in sig_share.index:
        n_uni = universe.get(sec, 0)
        uni_pct = n_uni / len(tickers) * 100
        sig_pct = sig_share[sec] / len(df) * 100
        bias = sig_pct / uni_pct if uni_pct > 0 else float("nan")
        rows.append((sec, n_uni, uni_pct, sig_share[sec], sig_pct, bias))
    for sec, n_uni, uni_pct, n_sig, sig_pct, bias in sorted(rows, key=lambda r: -r[5]):
        print(f"{sec:<14}{n_uni:6d}{uni_pct:8.1f}%{n_sig:8d}{sig_pct:9.1f}%{bias:8.2f}倍")
    print("※偏り = シグナル比 ÷ 母集団比。1.0なら中立、大きいほどその業種に偏っている")

    print("\n" + "=" * 78)
    print(f"② 業種別の成績（{MIN_TRADES_FOR_STATS}件以上のもののみ）")
    print("=" * 78)
    print(f'{"業種":<14}{"件数":>6}{"勝率":>8}{"PF":>7}{"平均R":>9}')
    perf = []
    for sec, g in df.groupby("sector"):
        if len(g) < MIN_TRADES_FOR_STATS:
            continue
        wr, pf, avg = stats(g["return_pct"])
        perf.append((sec, len(g), wr, pf, avg))
    for sec, n, wr, pf, avg in sorted(perf, key=lambda r: -r[4]):
        print(f"{sec:<14}{n:6d}{wr:7.1f}%{pf:7.2f}{avg:+8.2f}%")
    all_wr, all_pf, all_avg = stats(df["return_pct"])
    print(f'{"（全体）":<14}{len(df):6d}{all_wr:7.1f}%{all_pf:7.2f}{all_avg:+8.2f}%')

    print("\n" + "=" * 78)
    print("③ 時代による変化（年代別の業種別平均リターン）")
    print("=" * 78)
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "年代"
    pivot = df.pivot_table(index="sector", columns="decade",
                           values="return_pct", aggfunc="mean")
    counts = df.pivot_table(index="sector", columns="decade",
                            values="return_pct", aggfunc="count")
    # 件数が少ないセルは信頼できないので伏せる
    pivot = pivot.where(counts >= 10)
    print(pivot.round(2).to_string(na_rep="  −"))
    print("※各セルは平均リターン(%)。10件未満のセルは「−」で伏せている")

    print("\n" + "=" * 78)
    print("④ 年代別の全体成績（相場環境の違い）")
    print("=" * 78)
    print(f'{"年代":<10}{"件数":>6}{"勝率":>8}{"PF":>7}{"平均R":>9}')
    for dec, g in df.groupby("decade"):
        wr, pf, avg = stats(g["return_pct"])
        print(f"{dec:<10}{len(g):6d}{wr:7.1f}%{pf:7.2f}{avg:+8.2f}%")

    print("\n" + "=" * 78)
    print("⑤ 業種の優劣は持続するか（アウトオブサンプル検証）")
    print("=" * 78)
    print("全期間の集計で業種を選ぶのは過学習になる（同じデータで作って同じ")
    print("データで検証することになる）。そこで前半期間だけで業種をランク付けし、")
    print("それが後半期間でも通用するかを検証する。\n")

    for split_year in (2010, 2013, 2016):
        train = df[df["year"] < split_year]
        test = df[df["year"] >= split_year]
        if len(train) < 200 or len(test) < 200:
            continue
        # 前半期間で平均リターン順に業種をランク付け（件数が少ない業種は除外）
        tr_perf = train.groupby("sector")["return_pct"].agg(["mean", "count"])
        tr_perf = tr_perf[tr_perf["count"] >= MIN_TRADES_FOR_STATS]
        ranked = tr_perf.sort_values("mean", ascending=False).index.tolist()
        if len(ranked) < 4:
            continue
        top_half = set(ranked[: len(ranked) // 2])

        base_wr, base_pf, base_avg = stats(test["return_pct"])
        sel = test[test["sector"].isin(top_half)]
        sel_wr, sel_pf, sel_avg = stats(sel["return_pct"])

        print(f"--- 学習期間 〜{split_year - 1}年 / 検証期間 {split_year}年〜 ---")
        print(f"  学習期間で上位だった業種: {'、'.join(ranked[: len(ranked) // 2])}")
        print(f"  検証期間の成績:")
        print(f"    絞らない        : {len(test):5d}件 勝率{base_wr:5.1f}% "
              f"PF{base_pf:5.2f} 平均{base_avg:+6.2f}%")
        print(f"    上位業種のみ    : {len(sel):5d}件 勝率{sel_wr:5.1f}% "
              f"PF{sel_pf:5.2f} 平均{sel_avg:+6.2f}%")
        verdict = "改善" if sel_avg > base_avg else "悪化"
        print(f"    → 平均リターンは{verdict}（{sel_avg - base_avg:+.2f}pt）\n")

    # 年代間で業種の順位がどれだけ保たれるか（順位相関）
    print("--- 年代間の業種順位の相関（1.0なら順位が完全に保たれる）---")
    decs = sorted(df["decade"].unique())
    piv = df.pivot_table(index="sector", columns="decade",
                         values="return_pct", aggfunc="mean")
    cnt = df.pivot_table(index="sector", columns="decade",
                         values="return_pct", aggfunc="count")
    piv = piv.where(cnt >= 10)
    for i in range(len(decs) - 1):
        a, b = decs[i], decs[i + 1]
        pair = piv[[a, b]].dropna()
        if len(pair) >= 4:
            # 順位に変換してからピアソン相関を取る＝スピアマンの順位相関。
            # scipyを増やしたくないので自前で計算する
            rho = pair[a].rank().corr(pair[b].rank())
            print(f"  {a} → {b}: {rho:+.2f}（{len(pair)}業種で比較）")


if __name__ == "__main__":
    main()
