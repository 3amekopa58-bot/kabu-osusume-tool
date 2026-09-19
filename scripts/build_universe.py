"""
売買可能な銘柄ユニバースを作る（日経225からの拡大用）

tickers.csv は日経225の大型株だけだが、予算100万円・100株単位という制約では
むしろ中小型株のほうが買いやすい。一方で小型株には流動性の問題があり、
バックテストが前提にしている「終値で100株買える」「往復コスト0.2%」という
仮定が崩れる。そこで全上場3,822銘柄から以下の条件で絞り込む：

  ①買える      : 株価×100株 が予算以内
  ②流動性がある: 直近の平均売買代金が閾値以上
                  （1単元の売買が1日の出来高に占める割合が小さいこと）
  ③データがある: 十分な期間の株価データが取得できる
  ④汚染がない  : 1日で±80%超の異常な値動きがない（分割データ不整合の疑い）

yfinanceの一括ダウンロード（yf.download）を使うため、3,822銘柄でも
個別取得より大幅に速い。

使い方:
    python3 scripts/build_universe.py [出力先CSV]
      省略時は universe.csv
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
ALL_TICKERS_JSON = BASE_DIR / "data" / "all_listed_tickers.json"
DEFAULT_OUTPUT = BASE_DIR / "universe.csv"

BUDGET = 1_000_000
LOT_SIZE = 100
# 1日の平均売買代金がこれ未満の銘柄は、100株の売買でも値が動いてしまうため除外。
# 1単元（数十万円）が1日の売買代金の1%未満に収まる目安として1億円とした。
MIN_DAILY_TURNOVER = 100_000_000
MIN_DAYS = 120                  # スクリーニングに必要な最低営業日数
# ⚠️ 2026-09-19: period="6mo" は約126営業日しか返らず、MIN_DAYS との差が
# わずか6日だった。レート制限で少し切り詰められるだけで120日を割り、
# 「データ不足」という正常に見える分類に落ちてキユーピー等が消えた。
# period を1年に広げたうえで、バッチの中で極端に短い応答は
# 「基準を満たさない」ではなく「取れていない」として再取得に回す。
TRUNCATED_RATIO = 0.5           # バッチ中央値のこの割合未満なら切り詰められた応答とみなす
MAX_PLAUSIBLE_DAILY_MOVE = 0.8  # 1日で±80%超は分割データ不整合の疑い
BATCH_SIZE = 100                # 200だとレート制限に当たりやすい（2026-09-13）
BATCH_WAIT_SEC = 1.0            # バッチ間で一息入れる
RETRY_WAIT_SEC = 20             # レート制限に当たったときの待ち時間
RETRY_BATCH_SIZE = 20           # 拾い直しは小分けにする
MAX_ALLOWED_FAILURES = 30       # これを超えたら書き出さずに中止する
MAX_DROP_RATIO_PCT = 5          # 前回のユニバースからの脱落がこの割合を超えたら中止


def _batch_median_len(data, batch) -> float:
    """このバッチが返してきた営業日数の中央値。切り詰め検出の物差しに使う。"""
    lens = []
    for code in batch:
        try:
            d = data[code] if len(batch) > 1 else data
            n = len(d.dropna(subset=["Close"]))
            if n:
                lens.append(n)
        except Exception:
            pass
    if not lens:
        return 0.0
    lens.sort()
    return float(lens[len(lens) // 2])


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    if not ALL_TICKERS_JSON.exists():
        print(f"{ALL_TICKERS_JSON} がありません。先に "
              "python3 scripts/build_all_listed_tickers.py を実行してください。")
        return

    listed = json.loads(ALL_TICKERS_JSON.read_text(encoding="utf-8"))
    codes = [t["code"] for t in listed]
    names = {t["code"]: t["name"] for t in listed}
    print(f"上場{len(codes)}銘柄について、直近1年の株価を一括取得します…")

    kept, stats = [], {"データ不足": 0, "予算オーバー": 0, "流動性不足": 0,
                       "データ汚染": 0, "取得失敗": 0}
    failed = []          # 取得できなかった銘柄（データ不足とは別物）

    def fetch(batch, tries=3):
        """レート制限に当たったら待って数回やり直す"""
        for attempt in range(tries):
            try:
                return yf.download(batch, period="1y", group_by="ticker",
                                   auto_adjust=True, progress=False,
                                   threads=True)
            except Exception as e:
                if attempt == tries - 1:
                    print(f"    取得失敗（{tries}回）: {str(e)[:80]}")
                    return None
                wait = RETRY_WAIT_SEC * (attempt + 1)
                print(f"    失敗。{wait}秒待って再試行… ({str(e)[:60]})")
                time.sleep(wait)
        return None

    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        data = fetch(batch)
        if data is None:
            # ⚠️ バッチごと落とすと「データ不足」に紛れて気づけない。
            # 取得失敗として記録し、あとで個別に拾い直す
            failed.extend(batch)
            stats["取得失敗"] += len(batch)
            continue
        time.sleep(BATCH_WAIT_SEC)   # レート制限を避けるため一息入れる

        # このバッチが「どれくらいの長さで返ってきたか」の目安。
        # 個々の銘柄がこれより極端に短ければ、基準割れではなく切り詰め。
        med = _batch_median_len(data, batch)

        for code in batch:
            try:
                d = data[code] if len(batch) > 1 else data
                d = d.dropna(subset=["Close"])
                if len(d) == 0:
                    # 1行も無いのは「基準を満たさない」ではなく「取れていない」
                    failed.append(code)
                    stats["取得失敗"] += 1
                    continue
                if med and len(d) < med * TRUNCATED_RATIO:
                    # バッチの他銘柄より極端に短い＝取れていない。
                    # 「データ不足」に混ぜると気づけないので再取得へ回す
                    failed.append(code)
                    stats["取得失敗"] += 1
                    continue
                if len(d) < MIN_DAYS:
                    stats["データ不足"] += 1
                    continue
                price = float(d["Close"].iloc[-1])
                if price * LOT_SIZE > BUDGET:
                    stats["予算オーバー"] += 1
                    continue
                turnover = float((d["Close"] * d["Volume"]).tail(20).mean())
                if turnover < MIN_DAILY_TURNOVER:
                    stats["流動性不足"] += 1
                    continue
                if (d["Close"].pct_change().abs() > MAX_PLAUSIBLE_DAILY_MOVE).any():
                    stats["データ汚染"] += 1
                    continue
                kept.append({"code": code, "name": names.get(code, code),
                             "price": round(price, 1),
                             "turnover_oku": round(turnover / 1e8, 2)})
            except KeyError:
                # その銘柄の列自体が返ってきていない＝取得失敗
                failed.append(code)
                stats["取得失敗"] += 1
            except Exception:
                stats["データ不足"] += 1

        print(f"  {min(i + BATCH_SIZE, len(codes))}/{len(codes)}銘柄 "
              f"→ 通過{len(kept)}銘柄 / 取得失敗{stats['取得失敗']}銘柄")

    # 取得に失敗したぶんを小さなバッチで拾い直す
    if failed:
        print(f"\n取得に失敗した{len(failed)}銘柄を小分けで再取得します…")
        retry_list, failed = list(dict.fromkeys(failed)), []
        for i in range(0, len(retry_list), RETRY_BATCH_SIZE):
            batch = retry_list[i:i + RETRY_BATCH_SIZE]
            data = fetch(batch, tries=4)
            if data is None:
                failed.extend(batch)
                continue
            time.sleep(BATCH_WAIT_SEC)
            for code in batch:
                try:
                    d = data[code] if len(batch) > 1 else data
                    d = d.dropna(subset=["Close"])
                    if len(d) < MIN_DAYS:
                        stats["データ不足"] += 1
                        stats["取得失敗"] -= 1
                        continue
                    price = float(d["Close"].iloc[-1])
                    if price * LOT_SIZE > BUDGET:
                        stats["予算オーバー"] += 1
                        stats["取得失敗"] -= 1
                        continue
                    turnover = float((d["Close"] * d["Volume"]).tail(20).mean())
                    if turnover < MIN_DAILY_TURNOVER:
                        stats["流動性不足"] += 1
                        stats["取得失敗"] -= 1
                        continue
                    if (d["Close"].pct_change().abs()
                            > MAX_PLAUSIBLE_DAILY_MOVE).any():
                        stats["データ汚染"] += 1
                        stats["取得失敗"] -= 1
                        continue
                    kept.append({"code": code, "name": names.get(code, code),
                                 "price": round(price, 1),
                                 "turnover_oku": round(turnover / 1e8, 2)})
                    stats["取得失敗"] -= 1
                except Exception:
                    failed.append(code)
        print(f"  再取得後に残った失敗: {len(failed)}銘柄")

    # ⚠️ **既存のユニバースから大量に脱落していたら書き出さない。**
    # 「取得失敗」は検知できるようにしたが、**部分的にしか返ってこなかった**
    # 場合（120日ぶん要るのに30日しか来ない等）は「データ不足」に見えるので
    # 素通りしてしまう。前回入っていた銘柄が急に大量に落ちるのは、
    # 基準を満たさなくなったのではなく**取れていない**可能性が高い。
    #
    # ⚠️ 2026-09-19: 比較元を out_path にしていたため、出力先を新しいパスに
    # 変えて試すと `out_path.exists()` が False になり、**このガードごと
    # スキップされていた**。安全確認のつもりで別ファイルに書く操作が
    # 保護を外す、という逆向きの作りだった。比較元は常に本番の
    # universe.csv に固定する。
    baseline = DEFAULT_OUTPUT
    if baseline.exists():
        import csv as _csv
        with baseline.open(encoding="utf-8-sig") as f:
            _rows = list(_csv.DictReader(f))
        prev = {r["code"] for r in _rows}
        prev_name = {r["code"]: r.get("name", "") for r in _rows}
        now = {k["code"] for k in kept}
        dropped = prev - now
        ratio = len(dropped) / len(prev) * 100 if prev else 0
        print(f"\n前回のユニバース {len(prev)}銘柄 との比較:")
        print(f"  引き続き入る: {len(prev & now)} / 脱落: {len(dropped)}"
              f"（{ratio:.1f}%）/ 新規: {len(now - prev)}")
        if ratio > MAX_DROP_RATIO_PCT:
            print(f"\n⚠️ 前回入っていた銘柄の{ratio:.1f}%が脱落しました"
                  f"（許容{MAX_DROP_RATIO_PCT}%）。")
            print("   株価や流動性が実際に変わったのか、単に取得できて")
            print("   いないだけなのかを確かめてください。**中止します。**")
            print(f"   脱落した銘柄の例: {sorted(dropped)[:15]}")
            # ⚠️ 2026-09-19: 中止すると1時間かけた取得結果が丸ごと消え、
            # 調べ直すのにまた1時間かかっていた。**中止＝本番に書かない**
            # であって、捨てる必要はない。調査用に退避しておく。
            rej = out_path.with_suffix(".rejected.csv")
            pd.DataFrame(kept).to_csv(rej, index=False, encoding="utf-8-sig")
            (rej.with_name(rej.stem + "_dropped.txt")).write_text(
                "\n".join(f"{c},{prev_name.get(c, '')}" for c in sorted(dropped)),
                encoding="utf-8")
            print(f"   → 調査用に退避: {rej.name} / {rej.stem}_dropped.txt")
            return

    # ⚠️ **失敗が多いまま書き出さない。**
    # 2026-09-13に、レート制限で失敗した銘柄が「データ不足」に紛れ、
    # メルカリ・DeNA等が黙って脱落した universe.csv が作られた。
    # 銘柄リストはこのプロジェクトのほぼ全ての土台なので、
    # 不完全なものを書くくらいなら止めるほうがよい。
    if len(failed) > MAX_ALLOWED_FAILURES:
        print(f"\n⚠️ 取得できなかった銘柄が{len(failed)}件あり、"
              f"許容({MAX_ALLOWED_FAILURES}件)を超えました。")
        print("   不完全なユニバースを書き出すと、本来入るべき銘柄が")
        print("   黙って脱落します。**書き込みを中止します。**")
        print("   時間をおいて再実行してください。")
        print(f"   失敗した銘柄の例: {failed[:15]}")
        rej = out_path.with_suffix(".rejected.csv")
        pd.DataFrame(kept).to_csv(rej, index=False, encoding="utf-8-sig")
        print(f"   → 調査用に退避: {rej.name}")
        return

    df = pd.DataFrame(kept).sort_values("turnover_oku", ascending=False)
    df[["code", "name"]].to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 絞り込み結果 ===")
    print(f"上場全銘柄        : {len(codes)}")
    for k, v in stats.items():
        print(f"  {k}で除外       : {v}")
    print(f"売買可能なユニバース: {len(df)}銘柄 → {out_path}")
    if len(df):
        print(f"\n売買代金の分布（億円/日）:")
        q = df["turnover_oku"].describe(percentiles=[0.25, 0.5, 0.75])
        print(f"  中央値 {q['50%']:.1f} / 上位25% {q['75%']:.1f} / 下位25% {q['25%']:.1f}")
        print(f"\n株価の分布（円）:")
        p = pd.DataFrame(kept)["price"].describe(percentiles=[0.5])
        print(f"  中央値 {p['50%']:,.0f}円（100株で{p['50%']*100:,.0f}円）")


if __name__ == "__main__":
    main()
