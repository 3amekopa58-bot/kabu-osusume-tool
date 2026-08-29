"""
GitHub Actions から呼ばれる通知スクリプト。
output/recommend_*.csv の最新ファイルを読み、上位のおすすめ銘柄を
ntfy.sh 経由でプッシュ通知する。CSVが無い（screen.pyが失敗した）場合は
エラー通知を送る。
"""

import glob
import json
import os
import urllib.request
from pathlib import Path

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

# 到達率は目標の高さによって変わる（高い目標ほど届かない）ため、
# 全銘柄一律ではなく銘柄ごとの目標の高さに応じて出し分ける。
# analyze_targets.py の「ATR×3の目標の高さ別・到達率」の実測値：
#   〜6%: 70.7%(150件) / 6-8%: 66.7%(192件) / 8-10%: 62.7%(110件) / 10%〜: 54.8%(84件)
# （〜4%は10件しかないため4-6%帯に統合している）
TARGET_HIT_RATE_BANDS = [
    (6.0, 71),
    (8.0, 67),
    (10.0, 63),
    (float("inf"), 55),
]


def hit_rate_for(target_pct: float) -> int:
    """目標の高さ（%）に対応する到達率を返す。"""
    for upper, rate in TARGET_HIT_RATE_BANDS:
        if target_pct < upper:
            return rate
    return TARGET_HIT_RATE_BANDS[-1][1]

# 損切りと期待値は backtest.py の採用ルールを過去26年3,152トレードで集計した値。
# ルールを変更したら必ず取り直すこと：
#   python backtest.py timesl either trend marketadx volume rs sl10 max
STOP_HIT_RATE = 30.5       # 損切り-10%に到達した割合
EXPECTED_PCT = 2.87        # 1トレードあたりの期待値（平均リターン）


def load_japanese_names() -> dict:
    """
    証券コード -> 日本語社名 の対応表。yfinanceは日本株でも英語名しか
    返さないため、EDINET由来の対応表を使う
    （scripts/build_japanese_names.py で生成）。
    """
    path = Path(__file__).parent.parent / "data" / "japanese_names.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def display_name(code: str, fallback: str, names: dict) -> str:
    """通知用の銘柄名。日本語名があれば使い、冗長な「株式会社」は落とす。"""
    ja = names.get(code)
    if not ja:
        return str(fallback)
    for token in ("株式会社", "(株)", "（株）"):
        ja = ja.replace(token, "")
    return ja.replace("　", "").strip() or str(fallback)


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


def format_pick(r, names: dict) -> str:
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
            f" ※到達{hit_rate_for(target_pct)}%"
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
        f"[{code}]{display_name(str(r['code']), r['name'], names)}{tag_txt} 現物買い\n"
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

    names = load_japanese_names()
    picks = []
    weak_regime = False
    for _, r in df.iterrows():
        buy = r.get("buy_timing")
        if isinstance(buy, str) and "買いタイミング" in buy:
            picks.append(format_pick(r, names))
            if "地合いが弱い" in buy:
                weak_regime = True
        if len(picks) >= PICK_COUNT:
            break

    if not picks:
        top = df.iloc[0]
        return (
            f"本日は買いシグナルなし。上位候補: "
            f"[{str(top['code']).replace('.T', '')}]"
            f"{display_name(str(top['code']), top['name'], names)}"
            f"({top['price']:,.0f}円) {top['trend_label']}"
        )

    header = "本日のおすすめ（すべて現物買い）"
    footer = (
        f"※利確目安は銘柄ごとのATR×{ATR_MULTIPLE:.0f}（値動きの荒さ）から算出。"
        f"到達率は目標の高さ別の実測値であり予測ではない。"
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
