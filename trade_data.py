"""
バックテストのトレード明細を読む共通ローダー（汚染データの検知つき）

⚠️ **なぜこれが要るか（2026-09-05の事故）**

`output/_universe_max_trades.csv` は 2026-08-29 23:02 に生成されたが、
`backtest.py` に汚染データ除外を入れたコミットは同日 **23:03:43**。
**1分違いで修正前のファイル**だった。yfinanceの株式分割データ不整合により
4765.T のエントリー価格が0.045円と記録され、1トレードで
**+119,453%** というリターンが入っていた。

このファイルを使った分析は平均リターンとPFが壊れる。実際に
4.4-45／4.4-46／4.4-47 をこの汚染データで測ってしまい、やり直した。
**気づけたのは PF 32.53・平均+116% という「あり得ない数字」が出たから**で、
もっともらしい値だったら見逃していた。

そこで読み込み時に必ず検査し、**見つけたら止まる**ようにする。
黙って除外すると、次に同じことが起きても気づけないため。
"""

from pathlib import Path

import pandas as pd

from backtest import SUSPICIOUS_RETURN_THRESHOLD

BASE_DIR = Path(__file__).parent
DEFAULT_TRADES = BASE_DIR / "output" / "_universe_max_trades.csv"

# ⚠️ 「極端だが実在する値動き」と「あり得ない値」を区別する。
# 8105.T は2025年に53円→890円（60日で+1,189%）と動いたが、これは
# 暗号資産関連への転換による**実在の急騰**で、1日80%超の日は0日だった。
# こういう本物を弾いてしまうと検証対象が歪むので、止めるのは
# データ不整合以外に説明がつかない水準だけにする（4765.Tは+119,453%）。
IMPOSSIBLE_RETURN_PCT = 2000.0


def load_trades(path=None) -> pd.DataFrame:
    """
    トレード明細を読み、汚染が残っていれば RuntimeError で止める。

    汚染を黙って落とさないのは、**ファイルそのものを作り直すべき**だから。
    除外して先に進むと、古い汚染ファイルを使い続けることになる。
    """
    p = Path(path) if path else DEFAULT_TRADES
    if not p.exists():
        raise FileNotFoundError(
            f"{p} が見つかりません。次のコマンドで作り直してください:\n"
            "  python3 backtest.py timesl either trend marketadx volume rs "
            "sl10 max --tickers universe.csv")
    df = pd.read_csv(p, encoding="utf-8-sig")
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    impossible = df[df["return_pct"].abs() > IMPOSSIBLE_RETURN_PCT]
    if len(impossible):
        worst = impossible.loc[impossible["return_pct"].abs().idxmax()]
        raise RuntimeError(
            f"{p.name} に汚染データが残っています（|リターン|>"
            f"{IMPOSSIBLE_RETURN_PCT:.0f}%が{len(impossible)}件）。\n"
            f"  最悪の例: {worst['name']}({worst['code']}) "
            f"{worst['entry_date'].date()} → {worst['return_pct']:,.0f}%\n"
            "  yfinanceの株式分割データ不整合です。次で作り直してください:\n"
            "    python3 backtest.py timesl either trend marketadx volume rs "
            "sl10 max --tickers universe.csv\n"
            "  （backtest.py は MAX_PLAUSIBLE_DAILY_MOVE で該当銘柄を除外します）")

    # 止めはしないが、目立つように必ず知らせる。集計は少数の極値に
    # 引っ張られるので、結果を読むときに頭に入れておく必要がある
    extreme = df[df["return_pct"].abs() > SUSPICIOUS_RETURN_THRESHOLD]
    if len(extreme):
        print(f"⚠️ |リターン|>{SUSPICIOUS_RETURN_THRESHOLD:.0f}% のトレードが"
              f"{len(extreme)}件あります（データ不整合ではなく実在の急騰の"
              f"可能性が高いが、平均やPFを押し上げる）:")
        for _, r in extreme.iterrows():
            print(f"     {r['name']}({r['code']}) "
                  f"{r['entry_date'].date()} → {r['return_pct']:+,.0f}%")
    return df
