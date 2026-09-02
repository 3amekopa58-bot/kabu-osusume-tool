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
# プラン別のレート制限（公式 https://jpx-jquants.com/ja/spec/rate-limits）
#   Free 5回/分 / Light 60回/分 / Standard 120回/分 / Premium 500回/分
# 2026-09-02にStandardへ移行。表向きは120回/分（0.5秒間隔）だが、
# **0.6秒間隔で944銘柄を回したら849件が HTTP 429 になった**（同日に実測）。
# 公称値どおりには出ないので、実測に合わせて1.2秒（≒50回/分）にしている。
# それでも無料プラン（13秒）の20倍以上速い。
# ⚠️ 大幅に超過し続けると5分程度アクセスが遮断される。
# プランを変えたらこの値も直すこと（既定引数なので import 時に束縛される。
# 実行中に定数だけ書き換えても効かない）
MIN_INTERVAL_SEC = 1.2
# 429（レート超過）が返ったときの待ち時間と再試行回数
RATE_LIMIT_BACKOFF_SEC = 60.0
RATE_LIMIT_RETRIES = 3


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
        # 429が返ったら少し待って数回だけやり直す。一度詰まると数分間
        # 遮断されるので、失敗のたびに待ち時間を伸ばす
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            r = requests.get(f"{BASE}{path}", params=params,
                             headers=self._headers, timeout=60)
            self._last_call = time.time()
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SEC * (attempt + 1))
                self._last_call = time.time()
                continue
            raise RuntimeError(f"{path} が HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"{path}: レート制限で {RATE_LIMIT_RETRIES} 回試して失敗")

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


def count_downward_revisions(rows: list, key: str = "FOP",
                             since: str = None) -> dict:
    """
    会社予想の下方修正・上方修正の回数を数える。

    同じ決算期（CurFYEn）ごとに開示を日付順に並べ、当期予想（既定は営業利益
    FOP）が前回より下がっていれば下方修正、上がっていれば上方修正とみなす。
    通期決算では当期予想が空になり翌期予想（NxF...）に入るので、当期予想が
    ある開示だけを対象にする。

    `since`（"YYYY-MM-DD"）を渡すと、その日以降の開示だけを数える。
    片山晃 PART 7 のNGポイント②は**「上場5年以内に下方修正2回以上」**と
    期間が限定されているため、上場日を渡して窓を切るのに使う。
    ⚠️ これを渡さないと、取得できた全期間（Standardプランでは10年）の
    累計になり、古い会社ほど不利になる。

    戻り値: {"down": 下方修正の回数, "up": 上方修正の回数,
             "periods": 判定できた決算期数, "detail": [...]}
    """
    by_period = {}
    for r in rows:
        fy = r.get("CurFYEn")
        v = _num(r.get(key))
        d = r.get("DiscDate")
        if not fy or v is None:
            continue
        if since and (not d or d < since):
            continue
        by_period.setdefault(fy, []).append((d, v))

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


_ORDER = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}


def quarterly_revenue_growth(rows: list) -> dict:
    """
    直近の四半期決算の「前年同期比 増収率」を返す。

    片山晃『5年で1億貯める株式投資』PART 6 は
    **「四半期決算ごとに前年同期比『売上高10%増』が目安」**と書いている。
    このツールが持っているEDINETの決算データは**有価証券報告書＝年次**なので
    年次の増収率しか出せない。四半期の前年同期比はここで補う。

    J-Quantsの `Sales` は**期首からの累計**（1Q→2Q→3Q→FYと積み上がる）。
    そこで2通りを出す：
      cumulative … 累計どうしの前年同期比（日本の決算発表の「前年同期比」）
      standalone … その四半期"単独"の前年同期比
                   （累計の差分。直近3か月の勢いが出るので、
                     累計では見えない失速・加速が分かる）

    戻り値: {"cumulative": %, "standalone": % or None, "period": "3Q",
             "cur_fy": ..., "prev_fy": ...} / 比較できなければ None
    """
    # {決算期: {四半期種別: 累計売上}}
    by_fy, disc = {}, {}
    for r in rows:
        t, fy = r.get("CurPerType"), r.get("CurFYEn")
        v = _num(r.get("Sales"))
        if not t or not fy or v is None:
            continue
        by_fy.setdefault(fy, {})[t] = v
        disc[(fy, t)] = r.get("DiscDate")

    fys = sorted(by_fy)
    if len(fys) < 2:
        return None
    cur_fy, prev_fy = fys[-1], fys[-2]
    cur, prev = by_fy[cur_fy], by_fy[prev_fy]

    # 当期でいちばん進んだ四半期のうち、前年にも同じものがあるものを使う
    common = [t for t in cur if t in prev]
    if not common:
        return None
    t = max(common, key=lambda x: _ORDER.get(x, 0))
    if not prev[t] or prev[t] <= 0:
        return None

    def standalone(d, kind):
        """累計から1つ前の四半期の累計を引いて、その四半期単独の売上を出す"""
        order = _ORDER.get(kind, 0)
        if order <= 1:
            return d.get(kind)          # 1Qは累計＝単独
        pre = next((k for k, o in sorted(_ORDER.items(), key=lambda kv: -kv[1])
                    if o == order - 1 and k in d), None)
        if pre is None or d.get(kind) is None:
            return None
        return d[kind] - d[pre]

    sa_cur, sa_prev = standalone(cur, t), standalone(prev, t)
    sa = ((sa_cur - sa_prev) / sa_prev * 100
          if sa_cur is not None and sa_prev and sa_prev > 0 else None)

    return {"cumulative": (cur[t] - prev[t]) / prev[t] * 100,
            "standalone": sa, "period": t,
            "cur_fy": cur_fy, "prev_fy": prev_fy,
            "disc_date": disc.get((cur_fy, t))}


def progress_rate(rows: list) -> dict:
    """
    直近の四半期決算の「進捗率」＝通期の会社予想に対する達成率を返す。

    片山晃『5年で1億貯める株式投資』PART 5 より：

      進捗率 = 四半期決算の業績が通期予想の何%を達成しているか。
      1年を4分割している四半期決算ごとに25%ずつ達成すれば通期100%。
      「まあまあ好決算」＝第1四半期で約25%、第2四半期で50%を
      わずかに下回る49%といった**中途半端な数字**。

    著者は「売上高や営業利益より進捗率で見たほうがわかりやすい」と書いている。
    理由は、期初に会社が出した予想への期待はすでに株価に織り込まれており、
    「広げた風呂敷に対して実際どうだったか」が進捗率だから。

    ⚠️ **J-Quants無料プランのデータは約16〜17週（4か月）遅れる**
       （2026-09-01に実測：最新開示が2026-05-15）。そのため8月発表の1Qは
       12月ごろまで入らない。表示には必ず開示日を添えること。
    ⚠️ 通期決算（CurPerType="FY"）は当期予想が空になるので進捗率は出ない。
    ⚠️ 著者自身が「不動産株など四半期ごとのブレが大きい会社では、進捗率が
       悪いからと売るのは誤った判断になりうる」と注意している。

    戻り値: {"sales": 売上の進捗率%, "op": 営業利益の進捗率%,
             "expected": 目安%（25×四半期数）, "period": "2Q"} / 出せなければ None
    """
    # いちばん新しい開示を探す。それが通期決算（FY）なら、その年度は
    # 終わっているので進捗率に意味は無い（古い3Qの数字を出すと誤解を招く）
    latest = None
    for r in rows:
        t, fy = r.get("CurPerType"), r.get("CurFYEn")
        if not t or not fy:
            continue
        key = (fy, _ORDER.get(t, 0), r.get("DiscDate") or "")
        if latest is None or key > latest[0]:
            latest = (key, r)
    if latest is None or latest[1].get("CurPerType") not in ("1Q", "2Q", "3Q"):
        return None

    r = latest[1]
    t = r["CurPerType"]
    expected = 25.0 * _ORDER.get(t, 0)

    def rate(actual_key, forecast_key):
        a, f = _num(r.get(actual_key)), _num(r.get(forecast_key))
        if a is None or not f or f <= 0:
            return None
        return a / f * 100

    sales, op = rate("Sales", "FSales"), rate("OP", "FOP")
    if sales is None and op is None:
        return None
    return {"sales": sales, "op": op, "expected": expected,
            "period": t, "fy": r.get("CurFYEn"),
            "disc_date": r.get("DiscDate")}


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
