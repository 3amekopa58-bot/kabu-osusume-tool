"""
GitHub Actions から呼ばれる通知スクリプト。
output/recommend_*.csv の最新ファイルを読み、上位のおすすめ銘柄を
ntfy.sh 経由でプッシュ通知する。CSVが無い（screen.pyが失敗した）場合は
エラー通知を送る。
"""

import glob
import os
import urllib.request

TOPIC = os.environ.get("NTFY_TOPIC", "").strip()

PICK_COUNT = 3
LOT_SIZE = 100

# 手仕舞いルール（screen.py の HOLDING_DAYS_LIMIT / STOP_LOSS_PCT と揃えること）
HOLDING_DAYS_LIMIT = 60
STOP_LOSS_PCT = 10.0

# 利確目標は「エントリー価格＋ATR×3」で銘柄ごとに算出する。
# ATR（Average True Range）はその銘柄の値動きの荒さなので、荒い銘柄ほど
# 目標が遠く、穏やかな銘柄ほど近くなる。全銘柄一律の固定%より実態に合う。
#
# analyze_targets.py で4方式を過去トレードに当てはめて検証した結果：
#   直近高値      目標+1.6%  到達率90.5%  ← 目標が近すぎて意味をなさない
#   ATR×3        目標+7.2%  到達率65.7%  ← 採用
#   フィボ127.2%  目標+7.6%  到達率62.5%
#   固定+10.3%    目標+10.3% 到達率52.6%
# 目標の高さと到達率は逆相関する（高い目標ほど届かない）ので「どれが優秀か」
# ではなくバランスの問題。ATR×3は現実的な高さと到達率を両立し、かつ
# フィボナッチよりスイングの取り方に左右されず安定して計算できる。
ATR_PERIOD = 14
ATR_MULTIPLE = 3.0
TARGET_HIT_RATE = 65.7     # ATR×3の目標に到達した割合（保有期間中の高値ベース）

# 損切りと期待値は backtest.py の採用ルールを過去26年3,152トレードで集計した値。
# ルールを変更したら必ず取り直すこと：
#   python backtest.py timesl either trend marketadx volume rs sl10 max
STOP_HIT_RATE = 30.5       # 損切り-10%に到達した割合
EXPECTED_PCT = 2.87        # 1トレードあたりの期待値（平均リターン）


def fetch_atr(code: str) -> float:
    """
    銘柄のATR（値動きの荒さ）を返す。取得できなければ None。
    利確目標を銘柄ごとに変えるために使う。
    """
    try:
        import pandas as pd
        import yfinance as yf

        hist = yf.Ticker(code).history(period="3mo")
        if len(hist) < ATR_PERIOD + 5:
            return None
        prev_close = hist["Close"].shift(1)
        tr = pd.concat([
            hist["High"] - hist["Low"],
            (hist["High"] - prev_close).abs(),
            (hist["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().iloc[-1]
        return float(atr) if atr == atr and atr > 0 else None
    except Exception:
        return None


def format_pick(r) -> str:
    """
    1銘柄ぶんの通知テキスト。現物買いを前提とし、利確目安・損切り・期限を示す。
    「利確目安」は予測ではなく過去実績の中央値であり、そこに到達する割合は
    TARGET_HIT_RATE にすぎない。損切りに到達する割合のほうが高いため、
    利益側だけを見せて楽観的に映らないよう損失側も必ず併記する。
    """
    code = str(r["code"]).replace(".T", "")
    price = float(r["price"])
    cost = price * LOT_SIZE

    # 利確目標は銘柄ごとのATRから決める（荒い銘柄ほど目標が遠くなる）
    atr = fetch_atr(str(r["code"]))
    if atr is None:
        target_line = "  利確目安 算出不可（値動きデータ不足）"
    else:
        target_price = price + ATR_MULTIPLE * atr
        target_pct = (target_price - price) / price * 100
        target_gain = (target_price - price) * LOT_SIZE
        target_line = (
            f"  利確目安 {target_price:,.0f}円(+{target_pct:.1f}%) = +{target_gain:,.0f}円"
            f" ※到達{TARGET_HIT_RATE:.0f}%"
        )

    stop_price = price * (1 - STOP_LOSS_PCT / 100)
    stop_loss = cost * STOP_LOSS_PCT / 100

    # どのシグナルで拾ったかを示す（下半身＝ブレイク、押し目買い＝押し目）
    tags = []
    if r.get("kahanshin") is True or str(r.get("kahanshin")).lower() == "true":
        tags.append("下半身")
    if r.get("pullback") is True or str(r.get("pullback")).lower() == "true":
        tags.append("押し目")
    tag_txt = f"[{'/'.join(tags)}]" if tags else ""

    return (
        f"[{code}]{r['name']}{tag_txt} 現物買い\n"
        f"  買い {price:,.0f}円 × {LOT_SIZE}株 = {cost:,.0f}円\n"
        f"{target_line}\n"
        f"  損切り  {stop_price:,.0f}円(-{STOP_LOSS_PCT:.0f}%) = -{stop_loss:,.0f}円"
        f" ※到達{STOP_HIT_RATE:.0f}%\n"
        f"  期限 保有{HOLDING_DAYS_LIMIT}日で手仕舞い"
    )


def build_message() -> str:
    files = sorted(glob.glob("output/recommend_*.csv"))
    if not files:
        return "株おすすめツール: 本日はスクリーニングに失敗し、結果を取得できませんでした"

    import pandas as pd

    df = pd.read_csv(files[-1]).sort_values("total_score", ascending=False)

    picks = []
    weak_regime = False
    for _, r in df.iterrows():
        buy = r.get("buy_timing")
        if isinstance(buy, str) and "買いタイミング" in buy:
            picks.append(format_pick(r))
            if "地合いが弱い" in buy:
                weak_regime = True
        if len(picks) >= PICK_COUNT:
            break

    if not picks:
        top = df.iloc[0]
        return (
            f"本日は買いシグナルなし。上位候補: "
            f"[{str(top['code']).replace('.T', '')}]{top['name']}"
            f"({top['price']:,.0f}円) {top['trend_label']}"
        )

    header = "本日のおすすめ（すべて現物買い・空売りではない）"
    footer = (
        f"※利確目安は銘柄ごとのATR×{ATR_MULTIPLE:.0f}（値動きの荒さ）から算出。"
        f"到達率は過去トレードでの実測値であり予測ではない。"
        f"期待値は1トレードあたり+{EXPECTED_PCT:.1f}%"
    )
    if weak_regime:
        footer += "\n⚠️日経がレンジ相場（ADX20未満）。この局面は過去の成績が落ちるため慎重に"

    return "\n\n".join([header] + picks + [footer])


def main():
    if not TOPIC:
        print("NTFY_TOPIC が設定されていません。通知をスキップします。")
        return

    message = build_message()
    print(f"送信メッセージ:\n{message}")

    req = urllib.request.Request(
        f"https://ntfy.sh/{TOPIC}",
        data=message.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"ntfy応答: {resp.status}")


if __name__ == "__main__":
    main()
