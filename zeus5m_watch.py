#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeus5m_watch.py

BingX 무기한 선물 전 종목을 대상으로
  1) 5분봉에서 '제우스 스위칭라인' 롱/숏 신호를 계산하고
  2) 1시간 또는 4시간봉에서 200EMA의 지지(롱) / 저항(숏)을 받고 있는 종목만 남겨
  3) 텔레그램으로 알림 + CSV 로그를 쌓는다.

Pine "Zeus Switching Line v5"의 기본 설정을 그대로 이식했다.
  볼린저 63 / 2.4, 이탈=꼬리, 복귀=깊이 0.25, 직전 룩백 3, 이탈 극단값 래치

환경변수
  TELEGRAM_TOKEN      (필수)
  TELEGRAM_CHAT_ID    (필수)
  TOP_N               스캔할 거래대금 상위 종목 수      (기본 150)
  HTF_MAX_DIST_ATR    상위TF 종가~200EMA 최대 거리      (기본 1.2 ATR)
  HTF_TOUCH_BARS      터치 확인 룩백 봉수               (기본 6)
  HTF_TOUCH_ATR       터치 인정 허용 오차               (기본 0.35 ATR)
  REQUIRE_ALIGN       50/200EMA 정배열까지 강제  1/0    (기본 0)
  COOLDOWN_MIN        동일 종목·방향 재알림 금지 분      (기본 90)
  WORKERS             동시 요청 수                      (기본 6)
  DRY_RUN             1이면 텔레그램 전송 없이 콘솔만
"""

import os
import sys
import csv
import math
import time
import traceback
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
BASE = "https://open-api.bingx.com"
KST = timezone(timedelta(hours=9))

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TOP_N = int(os.getenv("TOP_N", "150"))
HTF_MAX_DIST_ATR = float(os.getenv("HTF_MAX_DIST_ATR", "1.2"))
HTF_TOUCH_BARS = int(os.getenv("HTF_TOUCH_BARS", "6"))
HTF_TOUCH_ATR = float(os.getenv("HTF_TOUCH_ATR", "0.35"))
REQUIRE_ALIGN = os.getenv("REQUIRE_ALIGN", "0") == "1"
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "90"))
WORKERS = int(os.getenv("WORKERS", "6"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

LTF = "5m"
LTF_LIMIT = 400          # 제우스 상태머신 워밍업용
HTF_LIST = ["1h", "4h"]
HTF_LIMIT = 600          # 200EMA 워밍업용
BAR_DELAY_SEC = 20       # 5분봉 마감 후 대기 (거래소 집계 지연 흡수)

LOG_PATH = os.getenv("LOG_PATH", "logs/zeus5m_signals.csv")

# ── 제우스 파라미터 (Pine 기본값) ──
BB_LEN, BB_MULT = 63, 2.4
MIN_OUT = 1
IN_DEPTH = 0.25
IN_CONFIRM = 0
PRE_UP, PRE_DN = 3, 3
OFF_UP, OFF_DN = 0.0, 0.0
MIN_JUMP = 0.0
NEED_CLOSE = True
FLAT_CONFIRM = 1

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "zeus5m-watch/1.0"})


def now_kst():
    return datetime.now(KST)


def log(msg):
    print(f"[{now_kst():%H:%M:%S}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# BingX API
# ──────────────────────────────────────────────────────────────
def api_get(path, params=None, retry=3):
    for i in range(retry):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=15)
            if r.status_code != 200:
                time.sleep(0.6 * (i + 1))
                continue
            j = r.json()
            if str(j.get("code", 0)) not in ("0", "None"):
                time.sleep(0.6 * (i + 1))
                continue
            return j.get("data")
        except Exception:
            time.sleep(0.8 * (i + 1))
    return None


def get_universe():
    """거래대금 상위 TOP_N 종목"""
    data = api_get("/openApi/swap/v2/quote/ticker")
    if not data:
        return []
    rows = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("-USDT"):
            continue
        try:
            qv = float(d.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            qv = 0.0
        rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:TOP_N]]


def get_klines(symbol, interval, limit):
    data = api_get("/openApi/swap/v3/quote/klines",
                   {"symbol": symbol, "interval": interval, "limit": limit})
    if not data or len(data) < 60:
        return None
    rows = []
    for d in data:
        try:
            rows.append((int(d["time"]), float(d["open"]), float(d["high"]),
                         float(d["low"]), float(d["close"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(rows) < 60:
        return None
    rows.sort(key=lambda x: x[0])
    t = np.array([r[0] for r in rows], dtype=np.int64)
    o = np.array([r[1] for r in rows], dtype=float)
    h = np.array([r[2] for r in rows], dtype=float)
    lw = np.array([r[3] for r in rows], dtype=float)
    c = np.array([r[4] for r in rows], dtype=float)
    return t, o, h, lw, c


# ──────────────────────────────────────────────────────────────
# 지표
# ──────────────────────────────────────────────────────────────
def ema(src, length):
    n = len(src)
    out = np.empty(n)
    a = 2.0 / (length + 1.0)
    out[0] = src[0]
    for i in range(1, n):
        out[i] = a * src[i] + (1 - a) * out[i - 1]
    return out


def atr_rma(h, l, c, length=14):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(n, np.nan)
    if n < length:
        return out
    out[length - 1] = tr[:length].mean()
    for i in range(length, n):
        out[i] = (out[i - 1] * (length - 1) + tr[i]) / length
    return out


def bollinger(c, length, mult):
    n = len(c)
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    for i in range(length - 1, n):
        w = c[i - length + 1:i + 1]
        m = w.mean()
        sd = w.std()          # Pine ta.stdev 기본은 모집단 표준편차
        mid[i] = m
        up[i] = m + mult * sd
        lo[i] = m - mult * sd
    return mid, up, lo


# ──────────────────────────────────────────────────────────────
# 제우스 스위칭라인 (Pine v5 이식)
# ──────────────────────────────────────────────────────────────
def zeus(h, l, c):
    """마지막 봉 기준 (sig_long, sig_short, sw_line, state) 반환"""
    n = len(c)
    mid, bup, blo = bollinger(c, BB_LEN, BB_MULT)
    atr14 = atr_rma(h, l, c, 14)

    pre_high = np.full(n, np.nan)
    pre_low = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - PRE_UP)
        b = max(0, i - PRE_DN)
        pre_high[i] = h[a:i + 1].max()
        pre_low[i] = l[b:i + 1].min()

    state = 0
    ext = float("nan")
    out_bars = 0
    in_bars = 0
    sw = float("nan")
    sw_side = 0
    prev_sw = float("nan")
    bars_since_not_flat = -1

    sig_long = np.zeros(n, dtype=bool)
    sig_short = np.zeros(n, dtype=bool)

    for i in range(n):
        lock_evt = False
        no_data = math.isnan(bup[i]) or math.isnan(blo[i])

        if not no_data:
            band_w = bup[i] - blo[i]
            is_out_up = h[i] > bup[i]      # 이탈 기준 = 꼬리
            is_out_dn = l[i] < blo[i]
            in_th_up = bup[i] - band_w * IN_DEPTH
            in_th_dn = blo[i] + band_w * IN_DEPTH

            if state == 0:
                if is_out_up:
                    state, ext, out_bars, in_bars = 1, pre_high[i], 1, 0
                elif is_out_dn:
                    state, ext, out_bars, in_bars = -1, pre_low[i], 1, 0
            else:
                ext = max(ext, h[i]) if state == 1 else min(ext, l[i])
                back_in = (c[i] <= in_th_up) if state == 1 else (c[i] >= in_th_dn)
                if back_in:
                    in_bars += 1
                else:
                    out_bars += 1
                    in_bars = 0

                if back_in and in_bars > IN_CONFIRM and out_bars >= MIN_OUT:
                    lock_val = ext
                    off = (atr14[i] * OFF_UP) if state == 1 else (-atr14[i] * OFF_DN)
                    if not math.isnan(off):
                        lock_val += off
                    big_enough = True
                    if not math.isnan(sw) and MIN_JUMP > 0 and sw != 0:
                        big_enough = abs(lock_val - sw) / sw * 100.0 >= MIN_JUMP
                    if big_enough:
                        sw = lock_val
                        sw_side = state
                        lock_evt = True
                    state, out_bars, in_bars = 0, 0, 0

            if state != 0:                 # 이탈 중 극단값 추종
                sw = ext

            if math.isnan(sw):
                sw = mid[i]

        is_flat = (not math.isnan(sw) and not math.isnan(prev_sw) and sw == prev_sw)
        bars_since_not_flat = bars_since_not_flat + 1 if is_flat else 0
        flat_held = bars_since_not_flat >= FLAT_CONFIRM - 1

        if lock_evt and flat_held:
            if sw_side == -1 and (not NEED_CLOSE or c[i] > sw):
                sig_long[i] = True
            elif sw_side == 1 and (not NEED_CLOSE or c[i] < sw):
                sig_short[i] = True

        prev_sw = sw

    return sig_long, sig_short, sw, state


# ──────────────────────────────────────────────────────────────
# 상위 TF 200EMA 지지/저항 판정
# ──────────────────────────────────────────────────────────────
def htf_check(kl, side):
    """
    통과 조건
      방향 일치 : 롱=종가가 200EMA 위 / 숏=종가가 200EMA 아래
      근접     : |종가-200EMA| <= HTF_MAX_DIST_ATR * ATR
      터치     : 최근 HTF_TOUCH_BARS 봉 중 저가(롱)/고가(숏)가 200EMA 허용오차 안까지 들어옴
    """
    t, o, h, l, c = kl
    if len(c) < 220:
        return None
    e200 = ema(c, 200)
    e50 = ema(c, 50)
    a = atr_rma(h, l, c, 14)

    px, em, atr = c[-1], e200[-1], a[-1]
    if math.isnan(atr) or atr <= 0:
        return None

    dist = abs(px - em) / atr
    if dist > HTF_MAX_DIST_ATR:
        return None

    n = min(HTF_TOUCH_BARS, len(c))
    tol = HTF_TOUCH_ATR * atr

    if side == "long":
        if px <= em:
            return None
        touched = bool(np.any(l[-n:] <= e200[-n:] + tol))
        aligned = e50[-1] > e200[-1]
    else:
        if px >= em:
            return None
        touched = bool(np.any(h[-n:] >= e200[-n:] - tol))
        aligned = e50[-1] < e200[-1]

    if not touched:
        return None
    if REQUIRE_ALIGN and not aligned:
        return None

    return {"dist": dist, "aligned": aligned, "ema200": em, "ema50": e50[-1]}


# ──────────────────────────────────────────────────────────────
# 종목 1개 처리
# ──────────────────────────────────────────────────────────────
def scan_symbol(symbol):
    kl = get_klines(symbol, LTF, LTF_LIMIT)
    if kl is None:
        return None
    t, o, h, l, c = kl

    # 형성 중인 봉 제거 → 직전 마감봉으로 판정
    now_ms = int(time.time() * 1000)
    if t[-1] + 5 * 60 * 1000 > now_ms:
        if len(c) < 260:
            return None
        t, o, h, l, c = t[:-1], o[:-1], h[:-1], l[:-1], c[:-1]

    sig_long, sig_short, sw, state = zeus(h, l, c)
    if sig_long[-1]:
        side = "long"
    elif sig_short[-1]:
        side = "short"
    else:
        return None

    matched = []
    for tf in HTF_LIST:
        hk = get_klines(symbol, tf, HTF_LIMIT)
        if hk is None:
            continue
        res = htf_check(hk, side)
        if res:
            res["tf"] = tf
            matched.append(res)

    if not matched:
        return None

    return {
        "symbol": symbol,
        "side": side,
        "price": float(c[-1]),
        "sw": float(sw) if not math.isnan(sw) else None,
        "bar_time": int(t[-1]),
        "matched": matched,
    }


# ──────────────────────────────────────────────────────────────
# 알림 / 로그
# ──────────────────────────────────────────────────────────────
def tv_link(symbol):
    return f"https://www.tradingview.com/chart/?symbol=BINGX%3A{symbol.replace('-', '')}.P"


def build_msg(r):
    head = "🟢 롱" if r["side"] == "long" else "🔴 숏"
    role = "200EMA 지지" if r["side"] == "long" else "200EMA 저항"
    sym = r["symbol"].replace("-USDT", "")
    lines = [
        f"{head}  {sym}",
        f"가격 {r['price']:.8g}",
    ]
    for m in r["matched"]:
        al = "정배열" if m["aligned"] else "역배열"
        lines.append(
            f"{m['tf'].upper()} {role} · 이격 {m['dist']:.2f}ATR · {al} "
            f"(EMA200 {m['ema200']:.8g})"
        )
    lines.append(tv_link(r["symbol"]))
    return "\n".join(lines)


def send_tg(text):
    if DRY_RUN or not TG_TOKEN or not TG_CHAT:
        print(text)
        print("-" * 40)
        return
    try:
        SESSION.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text,
                  "disable_web_page_preview": True},
            timeout=15,
        )
    except Exception as e:
        log(f"텔레그램 전송 실패: {e}")


def write_log(r):
    try:
        d = os.path.dirname(LOG_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["시각", "심볼", "방향", "가격", "충족TF",
                            "이격ATR", "정배열", "EMA200"])
            m = r["matched"][0]
            w.writerow([
                f"{now_kst():%Y-%m-%d %H:%M}", r["symbol"], r["side"],
                f"{r['price']:.10g}",
                "+".join(x["tf"] for x in r["matched"]),
                f"{m['dist']:.3f}", int(m["aligned"]), f"{m['ema200']:.10g}",
            ])
    except Exception as e:
        log(f"로그 기록 실패: {e}")


# ──────────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────────
_last_alert = {}     # (symbol, side) -> datetime
_universe = []
_universe_at = 0.0


def universe():
    global _universe, _universe_at
    if not _universe or time.time() - _universe_at > 3600:
        u = get_universe()
        if u:
            _universe = u
            _universe_at = time.time()
            log(f"유니버스 갱신: {len(u)}종목")
    return _universe


def cycle():
    syms = universe()
    if not syms:
        log("유니버스 조회 실패")
        return

    hits = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(lambda s: safe_scan(s), syms):
            if r:
                hits.append(r)

    sent = 0
    now = now_kst()
    for r in hits:
        key = (r["symbol"], r["side"])
        last = _last_alert.get(key)
        if last and (now - last).total_seconds() < COOLDOWN_MIN * 60:
            continue
        _last_alert[key] = now
        send_tg(build_msg(r))
        write_log(r)
        sent += 1

    log(f"스캔 {len(syms)}종목 · 조건충족 {len(hits)} · 발송 {sent}")


def safe_scan(s):
    try:
        return scan_symbol(s)
    except Exception:
        return None


def sleep_to_next_bar():
    now = time.time()
    nxt = (int(now // 300) + 1) * 300 + BAR_DELAY_SEC
    time.sleep(max(5, nxt - now))


def main():
    once = "--once" in sys.argv
    log(f"제우스 5분봉 감시 시작 (상위 {TOP_N}종목 · 1H/4H 200EMA 필터)")
    if not DRY_RUN and (not TG_TOKEN or not TG_CHAT):
        log("경고: 텔레그램 환경변수 없음 → 콘솔 출력으로 대체")
    while True:
        try:
            cycle()
        except Exception:
            traceback.print_exc()
        if once:
            break
        sleep_to_next_bar()


if __name__ == "__main__":
    main()
