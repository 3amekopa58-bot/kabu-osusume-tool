"""
J-Quantsから四半期の実績と会社予想を集める（進捗率の検証用）

片山晃『5年で1億貯める株式投資』PART 5 の**進捗率**
（＝通期の会社予想に対する四半期実績の達成率）を検証するために使う。

進捗率には**会社予想**が要る。会社予想は決算短信にしか載らず、
決算短信はTDnetにしかない。TDnetは robots.txt が全面クロール禁止なので、
JPX公式のJ-Quantsが唯一の機械的な取得経路（REQUIREMENTS 4.4-20/22）。

2026-09-02にStandardプラン（遅延なし・10年分・120件/分）へ移行したので、
ようやく検証に足る期間が取れるようになった。

⚠️ 先読みバイアス：`DiscDate`（開示日）をそのまま available_from として使う。
   これは実際に市場へ出た日なので、近似が要らない。

使い方:
    python3 scripts/build_jquants_financials.py [銘柄CSV]
      銘柄CSV 省略時は universe.csv
      **中断しても再実行すれば続きから進む**（取得済みの銘柄は飛ばす）
"""

import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "jquants_summary.json"

# 進捗率と増収率の計算に要る項目だけを残す（全項目だと100列以上あり無駄に重い）
KEEP = ("DiscDate", "CurPerType", "CurFYEn", "CurPerSt", "CurPerEn",
        "Sales", "OP", "OdP", "NP", "FSales", "FOP", "EPS", "FEPS")


def load_tickers(path: Path) -> list:
    with path.open(encoding="utf-8-sig") as f:
        return [r["code"] for r in csv.DictReader(f) if r.get("code")]


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from jquants import JQuantsClient

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tickers_path = Path(args[0]) if args else ROOT / "universe.csv"
    codes = load_tickers(tickers_path)

    store = {"data": {}}
    if OUT_PATH.exists():
        store = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"  既存データ: {len(store['data'])}銘柄。続きから取得します")

    cli = JQuantsClient()
    todo = [c for c in codes if c not in store["data"]]
    print(f"対象 {len(codes)}銘柄 / 未取得 {len(todo)}銘柄"
          f"（120件/分なので約{len(todo) * 0.6 / 60:.0f}分）")

    def save():
        store["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        OUT_PATH.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

    failed = 0
    for i, code in enumerate(todo, 1):
        try:
            rows = cli.summary(code)
            store["data"][code] = [
                {k: r.get(k) for k in KEEP if r.get(k) not in (None, "")}
                for r in rows
            ]
        except Exception as e:
            failed += 1
            print(f"  {code}: 取得失敗 ({str(e)[:60]})")
        if i % 100 == 0:
            n = sum(len(v) for v in store["data"].values())
            print(f"  [{i}/{len(todo)}] {len(store['data'])}銘柄 / {n:,}レコード")
            save()
    save()
    n = sum(len(v) for v in store["data"].values())
    print(f"\n完了: {OUT_PATH}")
    print(f"  {len(store['data'])}銘柄 / 合計{n:,}レコード / 失敗{failed}件")


if __name__ == "__main__":
    main()
