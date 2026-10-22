"""
今この瞬間、7552と同じ形（倍になって高値更新）になっている銘柄を数える

条件は analyze_parabolic_analogs.py と同一。4.4-86 で測った事後分布
（1年後の中央値-11.4%・暴落率は基準の2.9倍）がそのまま当てはまる側の
リストなので、**買い候補ではなく注意リスト**。

使い方: python3 screen_parabolic_now.py
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
CACHE = BASE_DIR / "data" / "price_cache" / "max"
OUT = BASE_DIR / "output" / "parabolic_now.csv"

RET_LOOKBACK = 120
MIN_RET = 1.00
MIN_DEV200 = 0.60
MAX_ATH_GAP = -0.05
BATCH = 200

names = {}
for line in (BASE_DIR / "universe.csv").read_text(encoding="utf-8-sig").splitlines()[1:]:
    if "," in line:
        c, n = line.split(",", 1)
        names[c] = n
codes = [p.stem for p in sorted(CACHE.glob("*.csv"))]
print(f"キャッシュ {len(codes)}銘柄 / 直近分を取り直します…")

# 直近3か月を取り直してキャッシュに継ぎ足す
fresh = {}
for i in range(0, len(codes), BATCH):
    batch = codes[i:i + BATCH]
    try:
        d = yf.download(batch, period="3mo", auto_adjust=False,
                        group_by="ticker", progress=False, threads=True)
    except Exception as e:
        print(f"  バッチ{i}失敗: {e}")
        continue
    for c in batch:
        try:
            sub = d[c].dropna(how="all")
            if not sub.empty:
                fresh[c] = sub
        except Exception:
            pass
    print(f"  {min(i+BATCH, len(codes))}/{len(codes)}")

rows = []
asof = []
for c in codes:
    try:
        df = pd.read_csv(CACHE / f"{c}.csv", index_col=0, parse_dates=True)
    except Exception:
        continue
    if len(df) < 300 or "Close" not in df:
        continue
    df.index = df.index.tz_localize(None)
    if c in fresh:
        f = fresh[c].copy()
        f.index = pd.DatetimeIndex(f.index).tz_localize(None)
        df = pd.concat([df[~df.index.isin(f.index)], f]).sort_index()
    close = df["Close"].astype(float).dropna()
    high = df["High"].astype(float).dropna() if "High" in df else close
    vol = df["Volume"].astype(float).fillna(0) if "Volume" in df else pd.Series(0.0, index=close.index)
    if len(close) < 300:
        continue
    px = close.iloc[-1]
    ret = px / close.iloc[-1 - RET_LOOKBACK] - 1
    dev200 = px / close.rolling(200).mean().iloc[-1] - 1
    ath = high.cummax().iloc[-1]
    gap = px / ath - 1
    asof.append(close.index[-1])
    if ret >= MIN_RET and dev200 >= MIN_DEV200 and gap >= MAX_ATH_GAP:
        rows.append({
            "code": c, "name": names.get(c, ""), "price": px,
            "ret120_pct": ret * 100, "dev200_pct": dev200 * 100,
            "ath_gap_pct": gap * 100,
            "turnover_oku": (close * vol).rolling(20).mean().iloc[-1] / 1e8,
        })

res = pd.DataFrame(rows).sort_values("turnover_oku", ascending=False)
OUT.parent.mkdir(exist_ok=True)
res.to_csv(OUT, index=False)
latest = pd.Series(asof).max()
print(f"\n最新データ日 {latest.date()} / 判定対象 {len(asof)}銘柄")
print(f"=== 該当 {len(res)}銘柄（うち売買代金5億円/日以上 "
      f"{(res.turnover_oku >= 5).sum()}銘柄）→ {OUT}\n")
print(f"{'コード':<9}{'銘柄名':<24}{'株価':>8}{'120日':>8}{'200線':>8}{'高値差':>7}{'代金':>8}")
for _, r in res[res.turnover_oku >= 5].iterrows():
    print(f"{r.code:<9}{(r['name'] or '')[:11]:<24}{r.price:>8.0f}{r.ret120_pct:>+7.0f}%"
          f"{r.dev200_pct:>+7.0f}%{r.ath_gap_pct:>+6.1f}%{r.turnover_oku:>7.1f}億")
