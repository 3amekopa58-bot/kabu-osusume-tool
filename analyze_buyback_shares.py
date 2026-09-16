"""自社株買い（発行済株式数の減少）と、その後の株価リターンの関係を測る。

4.4-58 で「220（自己株券買付状況報告書）は直近1年しか索引に無いので
検証できない」と書いたが、それは誤り。EDINET の有報から取った
`shares`（発行済株式数）が14年分あるので、**株式数が減ったかどうか**
という形なら測れる。本スクリプトはその再現用。

⚠️ shares は多くが `shares_estimated: true`＝純資産÷BPS からの推計。
   直接開示値ではない。分割・併合も同じ列に混ざる（後述の交絡の原因）。

使い方: ./venv/bin/python analyze_buyback_shares.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

B = Path(__file__).resolve().parent
FIN = B / "data" / "edinet_financials.json"
HOLD_YEARS = 3          # 株式数の変化を見たあと、何年のリターンを測るか


def load_shares_change() -> pd.DataFrame:
    """銘柄ごとに「最初の年→最後の年」の発行済株式数の変化率を出す。"""
    raw = json.load(open(FIN, encoding="utf-8"))
    comp = raw["data"]

    rows = []
    for code, years in comp.items():
        ys = sorted(years)
        if len(ys) < 3:
            continue
        s0 = years[ys[0]].get("shares")
        s1 = years[ys[-1]].get("shares")
        if not s0 or not s1:
            continue
        est = sum(bool(years[y].get("shares_estimated")) for y in ys)
        rows.append({
            "code": code,
            "株式数変化%": (s1 / s0 - 1) * 100,
            "開始": ys[0][:4], "終了": ys[-1][:4], "年数": len(ys),
            "推計比率": est / len(ys),
        })
    return pd.DataFrame(rows)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """同期間の株価リターンを付ける。

    ⚠️ yfinance の Close は auto_adjust に関わらず**常に分割調整済み**。
    「分割ぶんを株価に残して打ち消す」ことはできないので、分割銘柄は
    後段でシェア数の急増から検出して除外する（exclude_splits）。
    """
    out = {}
    codes = list(df["code"])
    for i in range(0, len(codes), 50):
        b = codes[i:i + 50]
        try:
            x = yf.download(b, period="max", group_by="ticker",
                            auto_adjust=False, progress=False, threads=True)
        except Exception:
            continue
        for c in b:
            try:
                s = x[c]["Close"].dropna()
                if len(s) < 250 * HOLD_YEARS:
                    continue
                out[c] = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100
            except Exception:
                pass
    df["リターン%"] = df["code"].map(out)
    return df.dropna(subset=["リターン%"])


def exclude_splits(raw_comp: dict, df: pd.DataFrame) -> pd.DataFrame:
    """分割・大型増資らしき銘柄を落とす。

    株式数が前年比 +40% 超になった年が1度でもあれば分割等とみなす。
    自社株買い（株式数の減少）の効果を、分割の後知恵から切り離すため。
    """
    split = set()
    for code, years in raw_comp.items():
        ys = sorted(years)
        for a, b in zip(ys, ys[1:]):
            s0, s1 = years[a].get("shares"), years[b].get("shares")
            if s0 and s1 and s1 / s0 - 1 > 0.40:
                split.add(code)
                break
    kept = df[~df["code"].isin(split)]
    print(f"分割・大型増資らしき {len(split)}銘柄を除外 → {len(kept)}銘柄\n")
    return kept


def main() -> None:
    df = add_returns(load_shares_change())
    out = B / "output" / "buyback_shares_vs_return.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"対象 {len(df)}銘柄（推計の shares を含む） → {out.name}\n")

    # scipy が無い環境なので順位相関は rank() で出す（spearman と同値）
    sp = df["株式数変化%"].rank().corr(df["リターン%"].rank())
    pe = df["株式数変化%"].corr(df["リターン%"])
    print(f"株式数変化 × リターン  順位相関 {sp:+.3f} / 相関 {pe:+.3f}")
    print("→ ゼロ近辺なら『自社株買いをした銘柄が上がる』とは言えない\n")

    df["群"] = pd.cut(df["株式数変化%"], [-np.inf, -5, 0, 5, np.inf],
                      labels=["-5%超 減（買い戻し）", "0〜-5% 減",
                              "0〜+5% 増", "+5%超 増（分割等）"])
    g = df.groupby("群", observed=True)["リターン%"].agg(["median", "mean", "count"])
    print("=== 株式数の増減で分けたリターン ===")
    print(g.to_string(float_format="%.1f"))
    print("\n⚠️ 『増えた群が最良』なら、それは自社株買いの否定ではなく")
    print("   分割の後知恵（上がった株ほど分割している）＝4.4-49 と同じ型。")

    # --- 分割を除いて測り直す ---
    raw_comp = json.load(open(FIN, encoding="utf-8"))["data"]
    d2 = exclude_splits(raw_comp, df)
    print("=== 分割を除いた場合 ===")
    sp2 = d2["株式数変化%"].rank().corr(d2["リターン%"].rank())
    print(f"順位相関 {sp2:+.3f}")
    g2 = d2.groupby("群", observed=True)["リターン%"].agg(
        ["median", "mean", "count"])
    print(g2.to_string(float_format="%.1f"))


if __name__ == "__main__":
    main()
