"""
EDINETのBPS/EPSを、株価データ（分割・併合調整済み）と同じ基準に揃える

有価証券報告書のBPS・EPSは「その決算期時点の株数」で計算されているが、
yfinanceの株価は株式分割・併合を遡って調整している。そのまま組み合わせると
PER・PBRが桁でずれる。

  例）古河電気工業(5801) は2024年に10株→1株の併合を実施
      EDINETのBPS 4,288円 / yfinance実績 428.5円 → ちょうど10倍
      補正しないとPBRが1/10になり「超割安」と誤判定される

やり方：yfinanceの財務データ（data/fundamental_history.json、こちらは株価と
同じ調整基準）と決算期が重なる年で BPS の比率を取り、それを銘柄ごとの
補正係数として全期間に適用する。分割履歴を別途調べなくても、重複期間の
実データから係数が直接求まる。

⚠️ 限界：重複期間（yfinanceが持つ直近5期）より古い時点で追加の分割があっても
検出できない。ただしEDINET側は「各期の株数ベース」で一貫しているため、
同一銘柄内での期間比較（増収率・増益率）は補正なしでも正しい。
影響を受けるのは株価と組み合わせる指標（PER・PBR）だけ。

使い方:
    python3 scripts/adjust_edinet_split.py
      data/edinet_financials.json を読み、調整済みの
      data/edinet_financials_adjusted.json を書き出す
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
EDINET_PATH = ROOT / "data" / "edinet_financials.json"
YF_PATH = ROOT / "data" / "fundamental_history.json"
OUT_PATH = ROOT / "data" / "edinet_financials_adjusted.json"

# 比率がこの範囲に収まっていれば「分割・併合なし」とみなす（端数処理の誤差を吸収）
NO_ADJUST_LO, NO_ADJUST_HI = 0.98, 1.02
# 比率がこれだけ食い違う年が混ざっていたら、係数が信用できないので調整しない
RATIO_SPREAD_LIMIT = 0.05


def _is_plausible_split(f: float, tol: float = 0.03) -> bool:
    """
    株式分割・併合として説明できる比率か。分割比は 2:1, 5:1, 10:1 や
    1:2（＝0.5）のように整数か整数の逆数になるので、それに近いかで見る。
    """
    if f <= 0:
        return False
    for cand in (f, 1 / f):
        n = round(cand)
        if n >= 2 and abs(cand - n) / n <= tol:
            return True
    return False


def main():
    if not EDINET_PATH.exists():
        print(f"{EDINET_PATH} がありません。"
              "先に scripts/build_edinet_financials.py を実行してください。")
        return
    edinet = json.loads(EDINET_PATH.read_text(encoding="utf-8"))["data"]
    yf_data = json.loads(YF_PATH.read_text(encoding="utf-8"))["data"] if YF_PATH.exists() else {}

    adjusted, stats = {}, {"調整なし": 0, "調整あり": 0, "係数不明": 0,
                           "比率がばらつく": 0, "半端な比率のため未調整": 0}
    factors = {}

    for code, periods in edinet.items():
        # yfinance 側の同じ決算期の BPS を集める
        yf_bps = {}
        for rec in yf_data.get(code, []):
            if rec.get("bps"):
                yf_bps[rec["period_end"]] = rec["bps"]

        ratios = []
        for pe, r in periods.items():
            a, b = r.get("bps"), yf_bps.get(pe)
            # 債務超過などでBPSが負の年は比率が無意味になる（実際に千代田化工建設
            # (6366)で係数-2.33が出た）。正の値どうしでしか比を取らない
            if a and b and a > 0 and b > 0:
                ratios.append(a / b)

        factor = 1.0
        if not ratios:
            stats["係数不明"] += 1
        elif max(ratios) / min(ratios) > 1 + RATIO_SPREAD_LIMIT:
            # 重複期間の途中で分割があると年ごとに比率が変わる。
            # その場合は一律の係数にできないので調整を諦める（誤補正より安全）
            stats["比率がばらつく"] += 1
        else:
            f = statistics.median(ratios)
            if NO_ADJUST_LO <= f <= NO_ADJUST_HI:
                stats["調整なし"] += 1
            elif not _is_plausible_split(f):
                # 株式分割・併合の比率は整数か単純な分数になる。半端な値
                # （0.94など）は自己株式の増減等を拾っただけの可能性が高く、
                # 補正すると逆に歪むので触らない
                stats["半端な比率のため未調整"] += 1
            else:
                factor = f
                stats["調整あり"] += 1
                factors[code] = round(f, 4)

        out = {}
        for pe, r in periods.items():
            rec = dict(r)
            if factor != 1.0:
                for k in ("bps", "eps"):
                    if rec.get(k) is not None:
                        rec[k] = rec[k] / factor
                rec["split_factor"] = round(factor, 4)
            out[pe] = rec
        adjusted[code] = out

    OUT_PATH.write_text(json.dumps({
        "note": "EDINETのBPS/EPSを株価（分割調整済み）と同じ基準に揃えたもの。"
                "available_from 以降でのみ使うこと",
        "split_factors": factors,
        "data": adjusted,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"=== 株式分割・併合の調整 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}銘柄")
    print(f"\n保存: {OUT_PATH}")
    if factors:
        print(f"\n調整した銘柄（係数が1から離れているもの）:")
        for code, f in sorted(factors.items(), key=lambda x: -abs(x[1] - 1))[:15]:
            name = next(iter(edinet[code].values())).get("filer_name", "")
            print(f"  {code} {name[:24]:26s} 係数 {f}")


if __name__ == "__main__":
    main()
