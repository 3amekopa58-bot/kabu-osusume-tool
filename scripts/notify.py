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


def format_pick(r) -> str:
    """
    「売り目安」は空売りの新規シグナルではなく、この現物買いで取得した株を
    いつ手仕舞う（保有株を売却する）かの目安。購入価格より上で手仕舞えれば
    利益確定、下なら損切りになる（どちらになるかは結果次第）。
    """
    code = str(r["code"]).replace(".T", "")
    sma5, sma20 = r.get("sma5"), r.get("sma20")
    if sma5 == sma5 and sma20 == sma20:  # NaNでない
        sell_txt = f"5日線({sma5:.0f})が20日線({sma20:.0f})を下抜けたら手仕舞い（保有株を売却）"
    else:
        sell_txt = "算出不可"
    return (
        f"[{code}]{r['name']} 現物買い（空売りではない）"
        f"購入{r['price']:.0f}円(100株={r['lot_cost']:,.0f}円) "
        f"売り目安:{sell_txt}"
    )


def build_message() -> str:
    files = sorted(glob.glob("output/recommend_*.csv"))
    if not files:
        return "株おすすめツール: 本日はスクリーニングに失敗し、結果を取得できませんでした"

    import pandas as pd

    df = pd.read_csv(files[-1]).sort_values("total_score", ascending=False)

    picks = []
    for _, r in df.iterrows():
        buy = r.get("buy_timing")
        if isinstance(buy, str) and "買いタイミング" in buy:
            picks.append(format_pick(r))
        if len(picks) >= 3:
            break

    if picks:
        msg = "本日のおすすめ（すべて現物買い）\n" + "\n".join(picks)
    else:
        top = df.iloc[0]
        msg = (
            f"本日は買いシグナルなし。上位候補: "
            f"[{str(top['code']).replace('.T', '')}]{top['name']}"
            f"({top['price']:.0f}円) {top['trend_label']}"
        )

    return msg


def main():
    if not TOPIC:
        print("NTFY_TOPIC が設定されていません。通知をスキップします。")
        return

    message = build_message()
    print(f"送信メッセージ: {message}")

    req = urllib.request.Request(
        f"https://ntfy.sh/{TOPIC}",
        data=message.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"ntfy応答: {resp.status}")


if __name__ == "__main__":
    main()
