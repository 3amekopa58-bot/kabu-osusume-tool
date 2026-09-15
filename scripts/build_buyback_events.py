"""
自己株券買付状況報告書（EDINET docTypeCode=220）の提出イベントを集める

⚠️ **本文は取らない。** 一覧API（documents.json）に secCode と提出日が
   入っているので、「いつ・どの銘柄が自社株買いを報告したか」は
   一覧だけで分かる。本文まで取ると10年で約12万件＝17時間かかるため、
   まず二値（実施したか否か）で効果を確かめ、効きそうなら金額を取りに行く。

出力: data/buyback_events.json  { "コード": ["提出日", ...] }

使い方: python3 scripts/build_buyback_events.py [開始年] [終了年]
"""
import datetime as dt, json, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "buyback_events.json"
URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
SLEEP = 0.25

def main():
    a = [x for x in sys.argv[1:] if x.isdigit()]
    y0 = int(a[0]) if a else 2016
    y1 = int(a[1]) if len(a) > 1 else dt.date.today().year
    key = [l.split("=", 1)[1].strip()
           for l in (ROOT / ".env").read_text(encoding="utf-8").splitlines()
           if l.startswith("EDINET_API_KEY=")][0]

    ev = {}
    if OUT.exists():
        ev = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"既存データ: {len(ev)}銘柄")
    done = set()
    d, end = dt.date(y0, 1, 1), dt.date(y1, 12, 31)
    end = min(end, dt.date.today())
    n_days = ok = 0
    while d <= end:
        if d.weekday() < 5:
            n_days += 1
            try:
                r = requests.get(URL, params={"date": d.isoformat(), "type": 2,
                                              "Subscription-Key": key}, timeout=30)
                if r.status_code == 200:
                    ok += 1
                    for doc in r.json().get("results", []):
                        if doc.get("docTypeCode") == "220" and doc.get("secCode"):
                            code = doc["secCode"][:4] + ".T"
                            ev.setdefault(code, [])
                            if d.isoformat() not in ev[code]:
                                ev[code].append(d.isoformat())
            except Exception:
                pass
            time.sleep(SLEEP)
            if n_days % 200 == 0:
                print(f"  {d} まで: {n_days}営業日 / {len(ev)}銘柄 / 成功{ok}日")
                OUT.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
        d += dt.timedelta(days=1)
    OUT.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
    tot = sum(len(v) for v in ev.values())
    print(f"\n完了: {len(ev)}銘柄 / 延べ{tot:,}件 / {n_days}営業日を照会（成功{ok}）")
    print(f"保存: {OUT}")

if __name__ == "__main__":
    main()
