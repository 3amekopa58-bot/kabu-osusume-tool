"""
GitHub Actions から呼ばれる通知スクリプト。
output/recommend_*.csv の最新ファイルを読み、上位のおすすめ銘柄を
ntfy.sh 経由でプッシュ通知する。CSVが無い（screen.pyが失敗した）場合は
エラー通知を送る。
"""

import datetime as dt
import glob
import json
import os
import urllib.request
from pathlib import Path

TOPIC = os.environ.get("NTFY_TOPIC", "").strip()

# 1枠あたりに出す銘柄数。スコア上位から何件を採るかで成績が変わることを
# 26年13,630トレードで実測した（REQUIREMENTS 4.4-28）：
#   上位1件/日 PF1.98 / 2件 1.91 / 3件 1.86 / 5件 1.80 / 10件 1.73
#   （絞らない場合は1.71。絞るほど良くなり、重複しない3期間すべてで改善）
# 3件は「選べる幅を残しつつ成績も確保する」ための妥協点。
# より確度を上げたいなら1〜2件に減らす（そのぶん候補が出ない日が増える）
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
# analyze_targets.py の「ATR×3の目標の高さ別・到達率」の実測値
# （2026-08-30、944銘柄ユニバース・26年13,634トレードで取り直した）：
#   〜4%: 70.1%(87件) / 4-6%: 68.8%(638件) / 6-8%: 62.4%(822件)
#   8-10%: 59.6%(446件) / 10%〜: 55.2%(518件)
# 日経225の225銘柄で測っていた旧値（71/67/63/55）より2〜5pt低い。
# 中小型株を含めたぶん値動きが荒く、ATR×3の目標が遠くなるため。
TARGET_HIT_RATE_BANDS = [
    (6.0, 69),
    (8.0, 62),
    (10.0, 60),
    (float("inf"), 55),
]


# 目標に到達するまでの営業日数（到達したトレードのみで集計）。
# analyze_targets.py の実測値：全体で中央値10営業日・平均12.8営業日。
# 目標の高さ別に見ても〜6%:11日 / 6-8%:10日 / 8-10%:10日 / 10%〜:8日 と
# ほとんど差がなかったため（目標が遠いほど時間がかかるわけではない。
# 急騰銘柄は一気に届くため10%超の帯がむしろ最短）、帯別には出し分けない。
DAYS_TO_TARGET_MEDIAN = 10
DAYS_TO_TARGET_Q1 = 4
DAYS_TO_TARGET_Q3 = 18


def hit_rate_for(target_pct: float) -> int:
    """目標の高さ（%）に対応する到達率を返す。"""
    for upper, rate in TARGET_HIT_RATE_BANDS:
        if target_pct < upper:
            return rate
    return TARGET_HIT_RATE_BANDS[-1][1]

# 損切りと期待値は backtest.py の採用ルールを過去26年13,633トレードで集計した値。
# **売買ルールを変えたときも、対象銘柄リストを変えたときも必ず取り直すこと**：
#   python backtest.py timesl either trend marketadx volume rs sl10 max --tickers universe.csv
STOP_HIT_RATE = 30.4       # 損切り-10%に到達した割合
EXPECTED_PCT = 3.04        # 1トレードあたりの期待値（平均リターン）

# --- 片山流「新高値ブレイク投資」（別系統。REQUIREMENTS 4.4-14）---
# 14年576トレードの実測値。現行ルールとは損切り幅も期限の有無も違うので
# 同じ枠に混ぜず、専用の表示にする
KATAYAMA_STOP_LOSS_PCT = 8.0    # 損切り-8%（片山氏が「-20〜30%は甘すぎる」と明言）
# 書籍に明記された条件と、このツールの検証で最も成績が良かった条件は
# PERで食い違う（著者は「PER30倍台まで買い」、検証では30倍台がPF0.50）。
# どちらが実際に機能するかを見るため、両方を別々に出す（REQUIREMENTS 4.4-15）
KATAYAMA_VARIANTS = {
    "katayama_book": {
        "label": "書籍版（増収10%↑・増益20%↑・ROE10%↑・PER39倍以下）",
        "stats": "14年388件で勝率46%・PF2.96・平均+5.3%",
    },
    "katayama_tested": {
        "label": "検証版（増収10%↑・増益30%↑・PER20倍未満）",
        "stats": "14年576件で勝率51%・PF3.04・平均+4.8%",
    },
    # PART 6「中小型株の中長期投資」。増益を条件にしないのが上2つとの違い。
    # ⚠️ 長期保有が前提。短期で切ると増益条件を付けたほうが良い
    "katayama_long": {
        "label": "長期版（増収10%↑・利益不問・PER39倍以下／長期保有前提）",
        "stats": "500日保有の検証で3期間すべて勝率最良（66/45/81%・平均+26/+13/+56%）",
    },
}


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
            f"  利確目安 {target_price:,.0f}円(+{target_pct:.1f}%) = +{target_gain:,.0f}円\n"
            f"       到達率{hit_rate_for(target_pct)}% / 到達なら約{DAYS_TO_TARGET_MEDIAN}営業日"
            f"（{DAYS_TO_TARGET_Q1}〜{DAYS_TO_TARGET_Q3}日）"
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
        f"[{code}]{display_name(str(r['code']), r['name'], names)}{tag_txt}\n"
        f"  買い {price:,.0f}円 × {LOT_SIZE}株 = {cost:,.0f}円\n"
        f"{target_line}\n"
        f"  損切り  {stop_price:,.0f}円(-{STOP_LOSS_PCT:.0f}%) = -{stop_loss:,.0f}円"
        f" ※到達{STOP_HIT_RATE:.0f}%\n"
        f"  期限 保有{HOLDING_DAYS_LIMIT}日で手仕舞い"
    )


def format_katayama_pick(r, names: dict) -> str:
    """
    片山流の1銘柄ぶん。現行ルールと違い**利確目標も保有期限も置かない**
    （上昇が続く限り持ち、-8%で損切り）ので、その旨を明示する。
    """
    code = str(r["code"]).replace(".T", "")
    price = float(r["price"])
    cost = price * LOT_SIZE
    stop_price = price * (1 - KATAYAMA_STOP_LOSS_PCT / 100)
    stop_loss = cost * KATAYAMA_STOP_LOSS_PCT / 100

    def pct(v):
        # NaN は float() を通ってしまうので明示的に弾く（"+nan%" と出るのを防ぐ）
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "?"
        return "?" if f != f else f"{f:+.0f}%"

    roe = r.get("roe")
    roe_txt = f" / ROE{float(roe):.0f}%" if roe is not None and roe == roe else ""
    # カップ・ウィズ・ハンドル。この形が完成した新高値は成績が良い
    # （重複しない3期間すべてでPF改善／REQUIREMENTS 4.4-17）
    cwh_txt = "【カップ】" if bool(r.get("cup_with_handle")) else ""
    # 時価総額300億円未満。重複しない3期間すべてでPFが最良だった帯
    # （REQUIREMENTS 4.4-25）。必須条件ではないので印として出す
    cap_txt = "【小型】" if bool(r.get("small_cap")) else ""
    cap = r.get("market_cap_oku")
    cap_line = (f"\n  時価総額 {float(cap):,.0f}億円"
                if (cap is not None and cap == cap) else "")
    # 片山晃 PART 7 のOKポイント①②。上場から5年/10年以内は成長余地が大きい
    yrs = r.get("years_since_listing")
    if yrs is not None and yrs == yrs:
        mark = "★" if float(yrs) <= 5 else ("☆" if float(yrs) <= 10 else "")
        listing_txt = f"\n  上場から{float(yrs):.1f}年{mark}"
    else:
        listing_txt = ""

    # 四半期の前年同期比（PART 6「四半期決算ごとに前年同期比 売上高10%増が目安」）。
    # このツールのEDINETデータは年次なので、J-Quantsで補った値を併記する。
    # ⚠️ 年次の条件（検証済み）を置き換えるものではなく、追加情報として出す。
    # 累計が10%を超えていても直近の四半期単独で失速していれば分かるようにする
    q_cum, q_sa, q_per = (r.get("q_revenue_growth"), r.get("q_revenue_growth_sa"),
                          r.get("q_period"))
    # J-Quants無料プランは約4か月遅れるので、いつ時点の数字かを必ず添える
    disc = r.get("jq_disc_date")
    asof = f"（{disc}時点）" if (disc and disc == disc) else ""
    q_txt = ""
    if q_cum is not None and q_cum == q_cum:
        sa = f"／直近四半期単独{pct(q_sa)}" if (q_sa is not None and q_sa == q_sa) else ""
        warn = ""
        if q_sa is not None and q_sa == q_sa and float(q_cum) >= 10 > float(q_sa):
            warn = " ⚠️失速"
        q_txt = f"\n  {q_per or '四半期'}累計 増収{pct(q_cum)}{sa}{warn}{asof}"

    # 進捗率（PART 5）。四半期ごとに25%ずつ達成すれば通期100%が目安。
    # 著者は「売上高や営業利益より進捗率で見たほうがわかりやすい」と書いている
    # （期初予想への期待はすでに株価に織り込まれているため）
    ps, po, pe = (r.get("progress_sales"), r.get("progress_op"),
                  r.get("progress_expected"))
    prog_txt = ""
    if pe is not None and pe == pe and any(v is not None and v == v for v in (ps, po)):
        def rate(v):
            return f"{float(v):.0f}%" if (v is not None and v == v) else "?"
        # 目安を下回る＝著者のいう「まあまあ好決算」の水準
        behind = "  ※目安割れ" if (po is not None and po == po
                                  and float(po) < float(pe)) else ""
        prog_txt = (f"\n  進捗率 売上{rate(ps)}／営業利益{rate(po)}"
                    f"（目安{float(pe):.0f}%）{behind}{asof}")

    return (
        f"[{code}]{display_name(str(r['code']), r['name'], names)}[新高値]{cwh_txt}{cap_txt}\n"
        f"  買い {price:,.0f}円 × {LOT_SIZE}株 = {cost:,.0f}円\n"
        f"  増収{pct(r.get('revenue_growth'))} / 増益{pct(r.get('profit_growth'))}"
        f" / PER{float(r['per']):.1f}倍{roe_txt}{cap_line}{q_txt}{prog_txt}{listing_txt}\n"
        f"  損切り  {stop_price:,.0f}円(-{KATAYAMA_STOP_LOSS_PCT:.0f}%) = -{stop_loss:,.0f}円\n"
        f"  利確目標なし・期限なし（上昇が続く限り持つ）"
    )


# 採用ルールの各条件と、満たしていないときに通知へ出す短い説明
CONDITION_LABELS = [
    ("cond_signal", "買いシグナル"),
    ("cond_trend", "PPP3/4以上＋100日線上"),
    ("cond_volume", "出来高1.5倍以上"),
    ("cond_rs", "日経をアウトパフォーム"),
    ("cond_regime", "日経がADX20超の上昇"),
]


def missing_note(row) -> str:
    """満たしていない条件を列挙した1行を返す。"""
    missing = [label for key, label in CONDITION_LABELS if not bool(row.get(key))]
    met = len(CONDITION_LABELS) - len(missing)
    return f"  条件 {met}/{len(CONDITION_LABELS)} ・未達: {' / '.join(missing)}"


def build_message() -> str:
    files = sorted(glob.glob("output/recommend_*.csv"))
    if not files:
        return "株おすすめツール: 本日はスクリーニングに失敗し、結果を取得できませんでした"

    # ⚠️ ワークフローは notify を if:always() で呼ぶので、screen.py が落ちた日でも
    # ここが動く。日付を確認しないと**前日の推奨が今日のものとして届く**。
    # 2026-08-30、screen.py がPER列の型エラーで落ちた際に実際にそうなった
    latest = Path(files[-1]).stem.replace("recommend_", "")
    today = dt.date.today().strftime("%Y%m%d")
    if latest != today:
        return ("⚠️ 株おすすめツール: 本日のスクリーニングが完了していません。\n"
                f"最新の結果は {latest[:4]}-{latest[4:6]}-{latest[6:]} 時点のものです。\n"
                "古い推奨で売買しないでください。GitHub Actionsのログを確認してください。")

    import pandas as pd

    df = pd.read_csv(files[-1]).sort_values("total_score", ascending=False)

    names = load_japanese_names()

    # 買いシグナルが点灯している行だけを候補にする
    cand = [r for _, r in df.iterrows()
            if isinstance(r.get("buy_timing"), str) and "買いタイミング" in r["buy_timing"]]

    # 採用ルールの全条件を満たす銘柄と、一部しか満たさない銘柄を分ける。
    # バックテストの成績は全条件が揃った場合の数字で、一部しか満たさない
    # 銘柄に当てはめてはいけないため、混ぜて出さない。
    # ⚠️ 期間で数字が大きく違う（2026-09-02に全期間を実測）：
    #     5年 2,632件 勝率59.8% PF2.52 平均+4.97%
    #    10年 5,542件 勝率52.5% PF1.69 平均+2.73%
    #    26年13,633件 勝率51.8% PF1.71 平均+3.04%
    # 通知に出す EXPECTED_PCT / STOP_HIT_RATE は**26年の値**を使っている。
    # 5年の数字（勝率60.6%）は直近相場に偏っているので通知には出さないこと
    full = [r for r in cand if bool(r.get("conditions_all"))]
    partial = sorted([r for r in cand if not bool(r.get("conditions_all"))],
                     key=lambda r: -int(r.get("conditions_met", 0)))

    if not cand and not any(bool(r.get("katayama")) for _, r in df.iterrows()):
        top = df.iloc[0]
        return (
            f"本日は買いシグナルなし。上位候補: "
            f"[{str(top['code']).replace('.T', '')}]"
            f"{display_name(str(top['code']), top['name'], names)}"
            f"({top['price']:,.0f}円) {top['trend_label']}"
        )

    parts = []
    if full:
        parts.append(f"◆本命（採用ルールの条件をすべて満たす）{len(full)}件")
        parts += [format_pick(r, names) for r in full[:PICK_COUNT]]
    else:
        parts.append("◆本命（全条件を満たす銘柄）: 本日はなし")

    if partial:
        parts.append(f"◇参考（条件を一部満たす）※成績の裏付けは弱い")
        for r in partial[:PICK_COUNT]:
            parts.append(format_pick(r, names) + "\n" + missing_note(r))

    # 片山流（別系統）。現行ルールと買う位置が正反対なので混ぜず末尾に別枠で出す。
    # 書籍版と検証版はPERの条件が食い違うので、それぞれ分けて出す
    any_kata = False
    for key, spec in KATAYAMA_VARIANTS.items():
        picks = [r for _, r in df.iterrows() if bool(r.get(key))]
        if not picks:
            continue
        any_kata = True
        parts.append(f"★片山流・{spec['label']} {len(picks)}件")
        parts += [format_katayama_pick(r, names) for r in picks[:PICK_COUNT]]
        parts.append(f"　※{spec['stats']}")
    # 押し目帯の注記（相場環境なので銘柄ごとではなく全体に1行追加する）
    dip_note = ""
    if not df.empty and bool(df.iloc[0].get("market_dip_band")):
        dd = df.iloc[0].get("nikkei_dd_pct")
        dd_txt = f"（高値から{float(dd):+.1f}%）" if (dd is not None and dd == dd) else ""
        dip_note = (f"\n📉 本日は日経が押し目帯{dd_txt}。"
                    "過去26年の実測では、この局面のトレードは"
                    "勝率63.9%・PF3.10（全体は51.8%・1.71）。"
                    "ただし該当するのは全体の約1割の期間で、"
                    "ADXが強いことが前提の数字です")

    if any_kata:
        parts.append(
            "　※片山流は現行ルールより当たり外れが大きい（2018/2021/2022年は負け越し）。"
            "書籍版と検証版はPERの条件が逆で、どちらが機能するか検証中。"
            "長期版は増益を見ない代わりに長期保有が前提（短期で切るなら書籍版/検証版）。"
            "★=上場5年以内 ☆=10年以内（伸びしろが大きい）。"
            "【カップ】=カップ・ウィズ・ハンドル完成（この形の新高値は"
            "重複しない3期間すべてでPFが改善＝優先度が高い）。"
            "【小型】=時価総額300億円未満（同じく3期間すべてで最良の帯。"
            "片山流の条件と重ねるとPF2.17→3.32/1.65→2.10/3.36→6.60。"
            "ただし件数が44〜121件と少ない）。"
            "増収率は年次（検証済みの条件）。四半期の前年同期比と進捗率は著者が"
            "本来見ている粒度で、参考情報として併記している"
            "（J-Quants無料プランは約4か月遅れるので時点を併記。"
            "進捗率は通期決算が最新の期には出ない。"
            "不動産株など四半期のブレが大きい業種では目安割れでも判断材料にしない）"
        )

    footer = (
        f"※利確目安は銘柄ごとのATR×{ATR_MULTIPLE:.0f}（値動きの荒さ）から算出。"
        f"到達率は目標の高さ別の実測値であり予測ではない。"
        f"期待値+{EXPECTED_PCT:.1f}%は◆本命の条件で検証した値"
    )
    if cand and not bool(cand[0].get("cond_regime", True)):
        footer += "\n⚠️日経がレンジ相場（ADX20未満）。この局面は過去の成績が落ちるため慎重に"
    footer += dip_note

    return "\n\n".join(["本日のおすすめ（すべて現物買い）"] + parts + [footer])


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
