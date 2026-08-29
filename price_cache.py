"""
株価データのローカルキャッシュ

バックテストやスクリーニングは同じ銘柄の同じ期間を何度も取得するが、
過去の株価は変わらないので毎回ダウンロードするのは完全な無駄だった
（225銘柄×26年の取得だけで毎回5〜15分かかっていた）。
このモジュールは取得済みデータを data/price_cache/ に保存し、
2回目以降はディスクから読む。

未取得の銘柄は yf.download でまとめて取りに行く。1銘柄ずつ
yf.Ticker().history() を呼ぶより桁違いに速い。

使い方:
    from price_cache import fetch_histories
    hist_map = fetch_histories(["7203.T", "6758.T"], period="5y")
    # -> {"7203.T": DataFrame, ...}
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "data" / "price_cache"
BATCH_SIZE = 200
# 直近データがこの日数より古ければ取り直す（週末・連休を考慮して余裕を持たせる）
STALE_DAYS = 4


def _cache_path(code: str, period: str) -> Path:
    return CACHE_DIR / period / f"{code}.csv"


def _is_fresh(path: Path, stale_days: int) -> bool:
    """キャッシュが十分新しいか（最終データ日がstale_days以内か）を判定する。"""
    if not path.exists():
        return False
    try:
        # 末尾だけ読めば最終日が分かるが、CSVは行数が可変なので素直に読む
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return False
        last = df.index[-1]
        if last.tzinfo is not None:
            last = last.tz_localize(None)
        return (dt.datetime.now() - last).days <= stale_days
    except Exception:
        return False


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # バックテスト側はtz-naiveなindexを前提にしている箇所があるため揃える
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_histories(codes, period: str = "5y", verbose: bool = True,
                    stale_days: int = STALE_DAYS) -> dict:
    """
    複数銘柄の株価をまとめて返す。キャッシュにあるものはディスクから、
    無いものだけを yf.download で一括取得してキャッシュに保存する。
    取得できなかった銘柄は結果に含まれない。

    stale_days: キャッシュを使ってよい古さの上限（日）。バックテストは
      既定値でよいが、当日の推奨を出す screen.py は直近の株価が要るので
      0 を渡して毎回取り直すこと。
    """
    cache_dir = CACHE_DIR / period
    cache_dir.mkdir(parents=True, exist_ok=True)

    result, missing = {}, []
    for code in codes:
        path = _cache_path(code, period)
        if _is_fresh(path, stale_days):
            try:
                result[code] = _read(path)
                continue
            except Exception:
                pass
        missing.append(code)

    if verbose:
        print(f"株価データ: キャッシュ{len(result)}銘柄 / 新規取得{len(missing)}銘柄")

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        try:
            data = yf.download(batch, period=period, group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            if verbose:
                print(f"  バッチ取得失敗: {e}")
            continue

        for code in batch:
            try:
                # group_by="ticker" のとき、yfinanceは1銘柄だけを指定した場合も
                # 列を (ティッカー, 項目) の2階層で返す。銘柄数で分岐すると
                # 1銘柄のときに Close 列を見つけられず取りこぼす
                d = data[code] if isinstance(data.columns, pd.MultiIndex) else data
                d = d.dropna(subset=["Close"])
                if d.empty:
                    continue
                if d.index.tz is not None:
                    d.index = d.index.tz_localize(None)
                # 有効数字を明示しないと、CSVに書き出す段階で株価の下位桁が
                # 落ちる。1e-4円程度のわずかな差でも、日次で資金を回す
                # portfolio_sim.py では買う銘柄の選択が変わり、27年で
                # 20pt以上リターンがずれる（2026-08-29に実測）
                d.to_csv(_cache_path(code, period), float_format="%.12g")
                result[code] = d
            except Exception:
                pass

        if verbose and missing:
            print(f"  取得 {min(i + BATCH_SIZE, len(missing))}/{len(missing)}銘柄")

    return result


def fetch_history(code: str, period: str = "5y",
                  stale_days: int = STALE_DAYS) -> pd.DataFrame:
    """1銘柄ぶん。内部では fetch_histories を使う。"""
    return fetch_histories([code], period=period, verbose=False,
                           stale_days=stale_days).get(code)


def clear_cache(period: str = None):
    """キャッシュを削除する（データがおかしくなったときの復旧用）。"""
    import shutil
    target = CACHE_DIR / period if period else CACHE_DIR
    if target.exists():
        shutil.rmtree(target)
        print(f"削除しました: {target}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_cache(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        # 動作確認
        h = fetch_histories(["7203.T", "6758.T"], period="1y")
        for k, v in h.items():
            print(f"{k}: {len(v)}行 {v.index[0].date()}〜{v.index[-1].date()}")
