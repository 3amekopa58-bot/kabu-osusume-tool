"""
EDINET APIへの接続テスト。.env の EDINET_API_KEY を使って、
直近の平日の書類一覧を取得できるか確認する（キーの値は一切出力しない）。
"""

import datetime as dt
import os
from pathlib import Path

import requests

ENV_PATH = Path(__file__).parent.parent / ".env"


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("EDINET_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".envにEDINET_API_KEYが見つかりません")


def main():
    api_key = load_api_key()
    if not api_key:
        print("EDINET_API_KEYが空です。.envを確認してください。")
        return

    # 直近の平日を対象日にする（土日は書類提出が無いことが多いため）
    target = dt.date.today()
    while target.weekday() >= 5:
        target -= dt.timedelta(days=1)
    target -= dt.timedelta(days=1)  # 前営業日
    while target.weekday() >= 5:
        target -= dt.timedelta(days=1)

    url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
    params = {"date": target.isoformat(), "type": 2, "Subscription-Key": api_key}

    resp = requests.get(url, params=params, timeout=15)
    print(f"HTTPステータス: {resp.status_code}")

    if resp.status_code != 200:
        print("接続に失敗しました。APIキーが正しいか確認してください。")
        return

    data = resp.json()
    count = len(data.get("results", []))
    print(f"対象日: {target}")
    print(f"取得できた書類件数: {count}")
    if count > 0:
        sample = data["results"][0]
        print("サンプル1件目:")
        print(f"  提出者名: {sample.get('filerName')}")
        print(f"  書類種別: {sample.get('docTypeCode')}")
        print(f"  書類名: {sample.get('docDescription')}")
    print("\n接続成功です。")


if __name__ == "__main__":
    main()
