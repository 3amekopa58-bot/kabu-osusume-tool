"""各トレードのエントリー時点の特徴量を作る（segment_lab.py の入力）。

⚠️ **先読み厳禁。** すべての特徴量は entry_date の**前日まで**の情報で作る。
   ファンダメンタルは `fundamental_history.json` の `available_from`
   （決算の開示日）以降のものしか使わない。
   4.4-39／4.4-46 で「結果依存の条件切り」を2回踏んでいるので、
   ここは entry_date 時点で確定している値だけに限る。

使い方: ./venv/bin/python build_segment_features.py
出力  : output/segment_features.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from price_cache import fetch_histories

B = Path(__file__).resolve().parent
TRADES = B / "output" / "_universe_max_trades.csv"
# ⚠️ fundamental_history.json は2021年9月以降しか無く、1期間しか作れない
#    ＝採用基準の3期間検証が成立しない（4.4-63）。edinet_financials.json は
#    2013年2月から14年分あるので、こちらを使う。
#    （EDINETのAPIは約10年ローリングだが、有報の「主要な経営指標等の推移」に
#      5年分の過去数値が載るため、最古の有報から2012年頃まで遡れる）
FUND = B / "data" / "edinet_financials.json"
# 実際の開示日（決算短信）。EDINET側の available_from は period_end+92日の
# 合成値で、実際は中央値41日・0.9%が92日超だった（4.4-66）。
# こちらがあれば差し替える。⚠️ J-Quants のデータは規約上コミット禁止。
JQ = B / "data" / "jquants_summary.json"
OUT = B / "output" / "segment_features.csv"


def load_disclosure_dates() -> dict:
    """(銘柄, 決算期末) → 実際の開示日（決算短信の発表日）。

    ⚠️ 値そのものは EDINET の有報（確定値）、開示日は短信のもの、という
       組み合わせになる。短信の速報値と有報の確定値はほぼ一致するが、
       修正が入った場合はそのぶん先読みになる。
       それでも period_end+92日 の合成値より実態に近い（4.4-69）。
    """
    if not JQ.exists():
        return {}
    raw = json.loads(JQ.read_text(encoding="utf-8"))["data"]
    out = {}
    for code, recs in raw.items():
        for r in recs:
            if r.get("CurPerType") != "FY":
                continue
            d, e = r.get("DiscDate"), r.get("CurFYEn")
            if not (d and e):
                continue
            dd, ee = pd.Timestamp(d), pd.Timestamp(e)
            if dd < ee:          # 期末前＝通期予想の行。実際の開示ではない
                continue
            k = (code, ee.date())
            if k not in out or dd < out[k]:
                out[k] = dd
    return out


def available_at(rec: dict, disc: dict, code: str) -> pd.Timestamp:
    """その決算が実際に使えるようになった日。

    実開示日があればそれを使い、無ければ EDINET の合成値（+92日）に戻す。
    決算期末の記録が数日ずれることがあるので前後10日を許容する。
    """
    pe = rec.get("period_end")
    if pe:
        base = pd.Timestamp(pe).date()
        for off in range(0, 11):
            for sign in (1, -1):
                k = (code, base + pd.Timedelta(days=off * sign))
                if k in disc:
                    return disc[k]
    return pd.Timestamp(rec["available_from"])


def fundamentals_asof(hist, when: pd.Timestamp) -> dict:
    """when 時点で**開示済み**の最新決算を返す（先読み防止）。

    edinet_financials.json は {決算期: {...}} の辞書、
    fundamental_history.json は [{...}] のリスト。どちらでも動くようにする。
    """
    items = sorted(hist.values(), key=lambda x: x.get("period_end") or "") \
        if isinstance(hist, dict) else list(hist)
    ok = [h for h in items
          if h.get("available_from") and pd.Timestamp(h["available_from"]) <= when]
    return ok[-1] if ok else {}


def fundamentals_asof_real(hist, when: pd.Timestamp, disc: dict, code: str) -> dict:
    """実開示日ベースで、when 時点で開示済みの最新決算を返す。"""
    items = sorted(hist.values(), key=lambda x: x.get("period_end") or "") \
        if isinstance(hist, dict) else list(hist)
    ok = [h for h in items
          if h.get("available_from") and available_at(h, disc, code) <= when]
    return ok[-1] if ok else {}


def main() -> None:
    d = pd.read_csv(TRADES)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    codes = sorted(d["code"].unique())
    print(f"{len(d):,}トレード / {len(codes)}銘柄の株価を用意中…")
    hist = fetch_histories(codes, period="max")

    fund = json.loads(FUND.read_text(encoding="utf-8"))["data"]
    disc = load_disclosure_dates()
    print(f"実際の開示日: {len(disc):,}件（無ければ period_end+92日に戻す）")

    # 相場環境（日経のADX>20 かつ 100日線上）。全トレード共通なので先に1本作る
    regime = bt.fetch_market_regime_adx(period="max")
    regime.index = pd.to_datetime(regime.index).tz_localize(None)

    rows = []
    for i, (code, g) in enumerate(d.groupby("code"), 1):
        h = hist.get(code)
        if h is None or len(h) == 0:
            continue
        h = h.copy()
        h.index = pd.to_datetime(h.index).tz_localize(None)
        close = h["Close"]
        sma200 = close.rolling(200).mean()
        vol20 = close.pct_change().rolling(20).std() * 100
        tv20 = (close * h["Volume"]).rolling(20).mean() / 1e8
        fh = fund.get(code) or []

        for _, t in g.iterrows():
            e = t["entry_date"]
            # entry_date の**前日まで**に限る
            past = close.index[close.index < e]
            if len(past) < 200:
                continue
            p = past[-1]
            px = float(close.loc[p])
            f = fundamentals_asof_real(fh, e, disc, code)
            shares = f.get("shares")
            bps, eps = f.get("bps"), f.get("eps")
            r = {
                "code": code, "entry_date": e,
                "ボラ%": float(vol20.loc[p]) if pd.notna(vol20.loc[p]) else np.nan,
                "売買代金億": float(tv20.loc[p]) if pd.notna(tv20.loc[p]) else np.nan,
                "200日線乖離%": (px / float(sma200.loc[p]) - 1) * 100
                if pd.notna(sma200.loc[p]) else np.nan,
                "1年騰落%": (px / float(close.loc[past[-245]]) - 1) * 100
                if len(past) >= 245 else np.nan,
                "時価総額億": px * shares / 1e8 if shares else np.nan,
                "PBR": px / bps if bps and bps > 0 else np.nan,
                "PER": px / eps if eps and eps > 0 else np.nan,
                "相場環境": ("◆本命（ADX20超＋100日線上）"
                         if bool(regime.get(p, False)) else "◇参考"),
            }
            rows.append(r)
        if i % 200 == 0:
            print(f"  {i}/{len(codes)}銘柄")

    f = pd.DataFrame(rows)
    # --- 帯に切る（境界はデータの分位点で。恣意的な線引きを避ける）---
    for col, name, q in [("時価総額億", "時価総額帯", 4), ("売買代金億", "売買代金帯", 4),
                         ("PBR", "PBR帯", 4), ("PER", "PER帯", 4),
                         ("ボラ%", "ボラ帯", 4), ("200日線乖離%", "200日線乖離帯", 4),
                         ("1年騰落%", "1年騰落帯", 4)]:
        try:
            f[name] = pd.qcut(f[col], q, labels=[f"{col}_Q{i+1}" for i in range(q)],
                              duplicates="drop")
        except Exception:
            f[name] = np.nan

    f.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n{len(f):,}行 → {OUT}")
    print(f"欠損率: " + " / ".join(
        f"{c} {f[c].isna().mean()*100:.0f}%"
        for c in ["時価総額億", "PBR", "PER", "ボラ%", "200日線乖離%"]))


if __name__ == "__main__":
    main()
