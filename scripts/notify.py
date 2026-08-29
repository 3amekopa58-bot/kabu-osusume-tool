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

# 過去実績の統計（backtest.py の採用ルールを過去26年・3,152トレードで
# 集計した値）。これは「予測」ではなく実績の分布であることに注意。
# 直近5年（強気相場）だと +10%到達32.1% / 損切り到達23.3% とかなり良く
# 見えるが、26年で見ると下記のとおり損切り到達のほうが多い。楽観的に
# 見せないため長期の数字を採用している。
# ルールを変更したら必ず取り直すこと：
#   python backtest.py timesl either trend marketadx volume rs sl10 max
TARGET_PCT = 10.3          # 勝ちトレードのリターン中央値
TARGET_HIT_RATE = 26.7     # +10%以上に到達した割合
STOP_HIT_RATE = 30.5       # 損切り-10%に到達した割合
EXPECTED_PCT = 2.87        # 1トレードあたりの期待値（平均リターン）


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

    target_price = price * (1 + TARGET_PCT / 100)
    target_gain = cost * TARGET_PCT / 100
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
        f"  利確目安 {target_price:,.0f}円(+{TARGET_PCT:.1f}%) = +{target_gain:,.0f}円"
        f" ※到達{TARGET_HIT_RATE:.0f}%\n"
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
        f"※利確/損切りの%は過去26年3,152トレードの実績分布であり予測ではない。"
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
