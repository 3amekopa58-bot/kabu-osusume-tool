"""
通知に埋め込んだ実測値が、現在のコードと一致しているかを照合する

⚠️ **なぜ必要か。** 通知の数字は**それぞれ別の時期に別のコマンドで導出**
   されており、コードを直しても自動では追随しない。REQUIREMENTS 6.5 の
   棚卸し手順の項目1（数値の出どころを遡る／コード修正後に測り直して
   いない数値が残っていないか）がまさにこれで、2026-09-05 に
   `EXPECTED_PCT` のズレを実際に検出した。

   さらに同日、**1分違いで修正前に作られた中間ファイル**を使って3節を
   測っていたことも判明している（4.4-48）。中間ファイルには
   「いつ・どのコードで作ったか」が残らないので、定期的な照合が要る。

照合する定数:
  ・STOP_HIT_RATE / EXPECTED_PCT      … 現行ルールの26年バックテスト
  ・PARTIAL_WIN_RATE / _EXPECTED_PCT / _PF … ◇参考枠（相場環境で分けた実測）

  ⚠️ 「100日線上」は**個別銘柄ではなく日経平均**の条件（相場環境の話）。
     個別銘柄の100日線上は `trend` フィルターに含まれており、
     ADX条件を外したファイルでも全トレードが満たしている。

使い方:
    python3 audit_notify_constants.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from backtest import fetch_market_regime_adx, fetch_nikkei_close
from trade_data import load_trades
import notify

BASE_DIR = Path(__file__).parent
# 現行ルール（ADX条件あり）＝ notify の EXPECTED_PCT / STOP_HIT_RATE の出どころ
MAIN = BASE_DIR / "output" / "_universe_max_trades.csv"
# ADX条件を外したもの＝ ◇参考枠の出どころ
NO_ADX = (BASE_DIR / "output" /
          "backtest_trades_timesl10d60_either_trend_volume_rs_universe_max_20260903.csv")

TOL = 0.5   # この差までは「一致」とみなす（丸めや期間の伸びぶん）


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()
    return gain / loss if loss else np.nan


def check(name, current, recomputed, unit="", tol=TOL):
    d = recomputed - current
    mark = "一致" if abs(d) <= tol else "**ズレ**"
    print(f"  {name:<24} 設定 {current:>8.2f}{unit}  実測 {recomputed:>8.2f}{unit}"
          f"  差 {d:+6.2f}  {mark}")
    return abs(d) > tol


def main():
    issues = []

    print("=" * 76)
    print("① 現行ルールの26年バックテスト（STOP_HIT_RATE / EXPECTED_PCT）")
    print("=" * 76)
    tr = load_trades(MAIN)
    r = tr["return_pct"]
    print(f"  対象: {len(tr):,}件 "
          f"（{tr['entry_date'].min().date()}〜{tr['entry_date'].max().date()}）")
    print(f"  ※コメントには「26年13,633トレード」とある\n")
    if check("STOP_HIT_RATE", notify.STOP_HIT_RATE, (r <= -9.9).mean() * 100, "%"):
        issues.append("STOP_HIT_RATE")
    if check("EXPECTED_PCT", notify.EXPECTED_PCT, r.mean(), "%", tol=0.05):
        issues.append("EXPECTED_PCT")
    print(f"  （参考）勝率 {(r>0).mean()*100:.2f}% / PF {pf(r):.2f}")

    print("\n" + "=" * 76)
    print("② ◇参考枠（PARTIAL_*）")
    print("=" * 76)
    if not NO_ADX.exists():
        print(f"  ⚠️ {NO_ADX.name} が無いので照合できません")
        print("     再作成: python3 backtest.py timesl either trend volume rs "
              "sl10 max --tickers universe.csv")
    else:
        t2 = load_trades(NO_ADX)
        print(f"  対象: {len(t2):,}件（ADX条件を外したもの）")
        print(f"  ※コメントには「26年30,304トレード」とある")

        adx = fetch_market_regime_adx("max")
        nk = fetch_nikkei_close("max")
        for s in (adx, nk):
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        nk_above = (nk > nk.rolling(100).mean())

        ed = t2["entry_date"]
        a = adx.reindex(ed, method="ffill").fillna(False).to_numpy()
        b = nk_above.reindex(ed, method="ffill").fillna(False).to_numpy()
        honmei = a & b
        t2 = t2.assign(本命=honmei, adx=a, above=b)

        h, p_ = t2[t2["本命"]]["return_pct"], t2[~t2["本命"]]["return_pct"]
        print(f"\n  ◆本命（日経ADX20超 かつ 日経が100日線上）: {len(h):,}件"
              f"  ※記録では 11,835件")
        print(f"     勝率 {(h>0).mean()*100:.1f}% / 平均 {h.mean():+.2f}% "
              f"/ PF {pf(h):.2f}   ※記録では 51.6% / +3.09% / 1.71")
        print(f"  ◇参考: {len(p_):,}件  ※記録では 18,468件")
        if check("PARTIAL_WIN_RATE", notify.PARTIAL_WIN_RATE,
                 (p_ > 0).mean() * 100, "%"):
            issues.append("PARTIAL_WIN_RATE")
        if check("PARTIAL_EXPECTED_PCT", notify.PARTIAL_EXPECTED_PCT,
                 p_.mean(), "%", tol=0.05):
            issues.append("PARTIAL_EXPECTED_PCT")
        if check("PARTIAL_PF", notify.PARTIAL_PF, pf(p_), "", tol=0.03):
            issues.append("PARTIAL_PF")

        sub_adx = t2[~t2["adx"]]["return_pct"]
        sub_ma = t2[~t2["above"]]["return_pct"]
        print(f"\n  内訳（重なりあり）")
        print(f"    ADX20未満のみ    : {len(sub_adx):,}件 "
              f"勝率{(sub_adx>0).mean()*100:.1f}% 平均{sub_adx.mean():+.2f}% "
              f"PF{pf(sub_adx):.2f}   ※記録では 14,648件 50.0% +1.83% 1.44")
        print(f"    日経が100日線下のみ: {len(sub_ma):,}件 "
              f"勝率{(sub_ma>0).mean()*100:.1f}% 平均{sub_ma.mean():+.2f}% "
              f"PF{pf(sub_ma):.2f}   ※記録では 8,166件 47.7% +1.32% 1.30")

    print("\n" + "=" * 76)
    print("結果")
    print("=" * 76)
    if issues:
        print(f"  **ズレが {len(issues)}件**: {'、'.join(issues)}")
        print("  → notify.py の該当定数を実測値に直すこと")
    else:
        print("  すべて一致")
    print("\n  ⚠️ TARGET_HIT_RATE_BANDS / DAYS_TO_TARGET_* は "
          "`python3 analyze_targets.py`、")
    print("     MARK_STATS / KATAYAMA_VARIANTS は各 analyze_*.py で別途照合する")


if __name__ == "__main__":
    main()
