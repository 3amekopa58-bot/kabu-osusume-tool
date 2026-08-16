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
            picks.append(f"{r['name']}({r['price']:.0f}円)")
        if len(picks) >= 3:
            break

    if picks:
        msg = "本日のおすすめ: " + "、".join(picks)
    else:
        top = df.iloc[0]
        msg = f"本日は買いシグナルなし。上位候補: {top['name']}({top['price']:.0f}円) {top['trend_label']}"

    return msg[:200]


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
