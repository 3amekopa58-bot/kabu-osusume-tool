"""
J-Quants API（JPX公式）のクライアント v2

片山晃 PART 7 のNGポイント②「上場5年以内に下方修正2回以上」を判定するには
**会社予想の修正履歴**が要る。TDnetの適時開示は直近1か月ぶんしか公開されて
いないが（2026-08-30に実測）、J-Quants の /v2/fins/summary には各開示時点の
**会社予想値**（FSales/FOP/FOdP/FNP など）が入っているので、同じ決算期に
ついて時系列で並べれば下方修正を検出できる。

⚠️ 認証は **v2のAPIキー方式**（ヘッダー x-api-key）。
   v1のメール＋パスワード→リフレッシュトークン方式は使えない
   （2026-08-31に実測。/v1/token/auth_user は 403 Forbidden）。

   .env に置く（gitignore済み。このファイルには書かない）：
     JQUANTS_API_KEY=（マイページで発行される43文字のキー）

⚠️ 無料プランの制約：
   - データは「直近12週間を除く2年分」
   - APIコールは5件/分

主なフィールド（/v2/fins/summary）：
  Sales/OP/OdP/NP/EPS  実績（売上・営業利益・経常利益・純利益・EPS）
  FSales/FOP/FOdP/FNP  **当期の会社予想**（四半期決算に入る）
  NxFSales/NxFOP/...   翌期の会社予想（通期決算に入る）
  CurFYEn              当期の決算期末（予想の対象期を識別するのに使う）
  ROE, BPS, ShOutFY    ROE・BPS・発行済株式数
"""

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"

BASE = "https://api.jquants.com/v2"
# 無料プランは5件/分。安全側に倒して1リクエストごとに待つ
MIN_INTERVAL_SEC = 13.0


def _load_env() -> dict:
    if not ENV_PATH.exists():
        return {}
    out = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _num(v):
    """J-Quants は欠損を空文字で返すので、数値にできないものは None にする"""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class JQuantsClient:
    def __init__(self, min_interval: float = MIN_INTERVAL_SEC):
        key = _load_env().get("JQUANTS_API_KEY")
        if not key:
            raise RuntimeError(
                ".env に JQUANTS_API_KEY がありません。\n"
                "  https://jpx-jquants.com/ のマイページで発行し、\n"
                "  echo 'JQUANTS_API_KEY=（キー）' >> .env")
        self._headers = {"x-api-key": key}
        self.min_interval = min_interval
        self._last_call = 0.0

    def _get(self, path: str, params: dict) -> dict:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        r = requests.get(f"{BASE}{path}", params=params,
                         headers=self._headers, timeout=60)
        self._last_call = time.time()
        if r.status_code != 200:
            raise RuntimeError(f"{path} が HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def summary(self, code: str) -> list:
        """
        財務サマリー。code は5桁（証券コード4桁＋'0'）で渡す。
        "6750.T" や "6750" を渡しても内部で 67500 に直す。
        """
        c = code.replace(".T", "")
        if len(c) == 4:
            c += "0"
        return self._get("/fins/summary", {"code": c}).get("data", [])

    def daily_bars(self, code: str, date: str) -> list:
        """日足。動作確認用"""
        c = code.replace(".T", "")
        if len(c) == 4:
            c += "0"
        return self._get("/equities/bars/daily", {"code": c, "date": date}).get("data", [])


def count_downward_revisions(rows: list, key: str = "FOP") -> dict:
    """
    会社予想の下方修正・上方修正の回数を数える。

    同じ決算期（CurFYEn）ごとに開示を日付順に並べ、当期予想（既定は営業利益
    FOP）が前回より下がっていれば下方修正、上がっていれば上方修正とみなす。
    通期決算では当期予想が空になり翌期予想（NxF...）に入るので、当期予想が
    ある開示だけを対象にする。

    戻り値: {"down": 下方修正の回数, "up": 上方修正の回数,
             "periods": 判定できた決算期数, "detail": [...]}
    """
    by_period = {}
    for r in rows:
        fy = r.get("CurFYEn")
        v = _num(r.get(key))
        if not fy or v is None:
            continue
        by_period.setdefault(fy, []).append((r.get("DiscDate"), v))

    down = up = 0
    detail = []
    for fy, items in by_period.items():
        items.sort()
        for (d0, v0), (d1, v1) in zip(items, items[1:]):
            if v1 < v0:
                down += 1
                detail.append({"fy": fy, "date": d1, "from": v0, "to": v1, "dir": "down"})
            elif v1 > v0:
                up += 1
                detail.append({"fy": fy, "date": d1, "from": v0, "to": v1, "dir": "up"})
    return {"down": down, "up": up, "periods": len(by_period), "detail": detail}


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "6750"
    cli = JQuantsClient()
    rows = cli.summary(code)
    print(f"{code}: 開示 {len(rows)}件"
          f"（{min(r['DiscDate'] for r in rows)} 〜 {max(r['DiscDate'] for r in rows)}）\n")
    print(f"{'開示日':12s} {'期':4s} {'当期末':12s} {'当期予想 営業利益':>18s}")
    for r in sorted(rows, key=lambda x: x["DiscDate"]):
        v = _num(r.get("FOP"))
        nv = _num(r.get("NxFOP"))
        s = f"{v/1e8:,.0f}億" if v is not None else (f"(翌期 {nv/1e8:,.0f}億)" if nv else "—")
        print(f"{r['DiscDate']:12s} {str(r.get('CurPerType')):4s} {str(r.get('CurFYEn')):12s} {s:>18s}")
    res = count_downward_revisions(rows)
    print(f"\n下方修正 {res['down']}回 / 上方修正 {res['up']}回（{res['periods']}決算期で判定）")
    for d in res["detail"]:
        print(f"  {d['date']} {d['dir']}: {d['from']/1e8:,.0f}億 → {d['to']/1e8:,.0f}億")
