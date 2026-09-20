"""区分検証（segment_lab.py）が使う入力データを検査する。

⚠️ **なぜ必要か。** このプロジェクトは「成果物を作り直したら結論が変わった」
   事故を繰り返している（4.4-48 汚染トレード／4.4-56 古い母集団／
   4.4-57 取得失敗がデータ不足に化けた／4.4-61 本番ユニバースが壊れていた）。
   中間ファイルには「いつ・どのコードで作ったか」が残らないので、
   結論を出す前に毎回入力を検査する。

検査するもの:
  ① トレード明細   件数・重複・日付の整合・リターンの再計算・鮮度
  ② 特徴量         結合で落ちた/増えた行・先読みの有無・値の妥当性
  ③ セクター       欠損・Unknown
  ④ ファンダ       bps/eps の妥当性・available_from の整合
  ⑤ ユニバース     トレード銘柄との対応

使い方: ./venv/bin/python audit_segment_inputs.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
NG = []


def ok(msg):
    print(f"  ✓ {msg}")


def ng(msg):
    print(f"  ✗ **{msg}**")
    NG.append(msg)


def warn(msg):
    print(f"  ⚠ {msg}")


def mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def audit_trades() -> pd.DataFrame:
    print("\n① トレード明細 output/_universe_max_trades.csv")
    p = B / "output" / "_universe_max_trades.csv"
    d = pd.read_csv(p)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    print(f"  ファイル更新: {mtime(p)} / {len(d):,}件 / {d['code'].nunique()}銘柄")

    dup = d.duplicated(["code", "entry_date"]).sum()
    (ok if dup == 0 else ng)(f"同一銘柄・同一エントリー日の重複 {dup}件")

    bad = (d["exit_date"] < d["entry_date"]).sum()
    (ok if bad == 0 else ng)(f"手仕舞いがエントリーより前 {bad}件")

    hd = (d["exit_date"] - d["entry_date"]).dt.days
    gap = (hd - d["holding_days"]).abs()
    (ok if (gap > 5).sum() == 0 else warn)(
        f"holding_days と日付の差が5日超 {(gap>5).sum()}件（営業日/暦日の違いで多少は出る）")

    calc = (d["exit_price"] / d["entry_price"] - 1) * 100
    err = (calc - d["return_pct"]).abs()
    (ok if err.max() < 0.01 else ng)(
        f"return_pct を価格から再計算した最大誤差 {err.max():.6f}%")

    neg = (d[["entry_price", "exit_price"]] <= 0).any(axis=1).sum()
    (ok if neg == 0 else ng)(f"価格が0以下 {neg}件")

    ext = (d["return_pct"].abs() > 500).sum()
    warn(f"|リターン|>500% が {ext}件（実在の急騰を含む。中央値判定なら影響は限定的）")

    bad_lo = (d["return_pct"] < -100).sum()
    (ok if bad_lo == 0 else ng)(f"リターン -100%未満（ありえない）{bad_lo}件")

    print(f"  期間: {d['entry_date'].min():%Y-%m-%d} 〜 {d['entry_date'].max():%Y-%m-%d}")
    return d


def audit_freshness(d: pd.DataFrame):
    print("\n①-b 鮮度：トレード明細は現在のコード・ユニバースと整合するか")
    uni = pd.read_csv(B / "universe.csv", encoding="utf-8-sig")
    ucodes = set(uni["code"])
    tcodes = set(d["code"].unique())
    out = tcodes - ucodes
    (ok if len(out) == 0 else ng)(
        f"トレードにあってユニバースに無い銘柄 {len(out)}件"
        + (f" 例:{sorted(out)[:5]}" if out else ""))
    print(f"  ユニバース {len(ucodes):,}銘柄 / トレードに登場 {len(tcodes):,}銘柄")

    tp = (B / "output" / "_universe_max_trades.csv").stat().st_mtime
    for f in ["backtest.py", "universe.csv"]:
        s = (B / f).stat().st_mtime
        if s > tp:
            ng(f"{f} がトレード明細より新しい（{mtime(B/f)} > {mtime(B/'output'/'_universe_max_trades.csv')}）"
               "＝作り直しが必要")
        else:
            ok(f"{f} はトレード明細より古い（整合）")


def audit_features(d: pd.DataFrame):
    print("\n② 特徴量 output/segment_features.csv")
    p = B / "output" / "segment_features.csv"
    f = pd.read_csv(p)
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    print(f"  ファイル更新: {mtime(p)} / {len(f):,}行")

    dup = f.duplicated(["code", "entry_date"]).sum()
    (ok if dup == 0 else ng)(f"重複 {dup}件")

    key_t = set(zip(d["code"], d["entry_date"]))
    key_f = set(zip(f["code"], f["entry_date"]))
    extra = key_f - key_t
    lost = key_t - key_f
    (ok if len(extra) == 0 else ng)(f"トレードに存在しない行 {len(extra)}件")
    warn(f"特徴量が作れなかったトレード {len(lost)}件"
         f"（{len(lost)/len(d)*100:.1f}%。200日未満の履歴などで発生）")

    m = d.merge(f, on=["code", "entry_date"], how="left")
    (ok if len(m) == len(d) else ng)(
        f"結合後の件数 {len(m):,}（元 {len(d):,}）＝行が増えていない")

    for c, lo, hi in [("ボラ%", 0, 50), ("PBR", 0, 100), ("PER", 0, 1000),
                      ("売買代金億", 0, 100000), ("時価総額億", 0, 1000000)]:
        if c not in f:
            continue
        s = f[c].dropna()
        bad = ((s < lo) | (s > hi)).sum()
        (ok if bad == 0 else warn)(
            f"{c}: 範囲外 {bad}件 / 中央値 {s.median():.2f} / 欠損 {f[c].isna().mean()*100:.0f}%")


def audit_lookahead(d: pd.DataFrame):
    print("\n②-b 先読みの検査（ファンダは開示日以降のものだけか）")
    fin = json.loads((B / "data" / "edinet_financials.json").read_text(encoding="utf-8"))["data"]
    f = pd.read_csv(B / "output" / "segment_features.csv")
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    sub = f[f["PBR"].notna()].sample(min(400, f["PBR"].notna().sum()), random_state=0)
    bad = 0
    for _, r in sub.iterrows():
        h = fin.get(r["code"])
        if not h:
            continue
        avail = [pd.Timestamp(v["available_from"]) for v in h.values()
                 if v.get("available_from") and pd.Timestamp(v["available_from"]) <= r["entry_date"]]
        if not avail:
            bad += 1
    (ok if bad == 0 else ng)(
        f"無作為400件で、エントリー日時点で未開示の決算を使っていた疑い {bad}件")


def audit_sectors(d: pd.DataFrame):
    print("\n③ セクター data/sectors.json")
    p = B / "data" / "sectors.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    print(f"  ファイル更新: {mtime(p)} / {len(s):,}銘柄")
    codes = set(d["code"].unique())
    miss = codes - set(s)
    (ok if len(miss) == 0 else ng)(f"セクター未取得の銘柄 {len(miss)}件")
    unk = sum(1 for c in codes if s.get(c) in (None, "Unknown"))
    (ok if unk == 0 else warn)(
        f"Unknown扱い {unk}銘柄（{unk/len(codes)*100:.1f}%）")


def audit_fundamentals():
    print("\n④ ファンダ data/edinet_financials.json")
    p = B / "data" / "edinet_financials.json"
    fin = json.loads(p.read_text(encoding="utf-8"))["data"]
    print(f"  ファイル更新: {mtime(p)} / {len(fin):,}銘柄")
    bps, eps, bad_av = [], [], 0
    for code, h in fin.items():
        for k, v in h.items():
            if v.get("bps"):
                bps.append(v["bps"])
            if v.get("eps"):
                eps.append(v["eps"])
            pe, av = v.get("period_end"), v.get("available_from")
            if pe and av and pd.Timestamp(av) < pd.Timestamp(pe):
                bad_av += 1
    (ok if bad_av == 0 else ng)(f"開示日が決算期末より前 {bad_av}件（ありえない）")
    b = pd.Series(bps)
    ok(f"BPS: 中央値 {b.median():,.0f} / 0以下 {(b<=0).sum()}件 / {len(b):,}件")
    e = pd.Series(eps)
    ok(f"EPS: 中央値 {e.median():,.1f} / 0以下 {(e<=0).sum()}件 / {len(e):,}件")
    lag = []
    for h in fin.values():
        for v in h.values():
            if v.get("period_end") and v.get("available_from"):
                lag.append((pd.Timestamp(v["available_from"]) - pd.Timestamp(v["period_end"])).days)
    l = pd.Series(lag)
    ok(f"開示までの日数: 中央値 {l.median():.0f}日 / 最小 {l.min()}日 / 最大 {l.max()}日")


def main():
    print("=" * 70)
    print("区分検証の入力データ検査")
    print("=" * 70)
    d = audit_trades()
    audit_freshness(d)
    audit_features(d)
    audit_lookahead(d)
    audit_sectors(d)
    audit_fundamentals()
    print("\n" + "=" * 70)
    if NG:
        print(f"**問題 {len(NG)}件**")
        for x in NG:
            print(f"  ✗ {x}")
    else:
        print("致命的な問題は見つからなかった（⚠ は仕様上の欠損・要把握）")
    print("=" * 70)


if __name__ == "__main__":
    main()
