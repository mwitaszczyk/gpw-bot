import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "")

POLAND_TZ = ZoneInfo("Europe/Warsaw")
REPORT_TIMES = {"10:15", "13:00", "16:00"}

WATCHLIST = [
    "PKN.WA", "PKO.WA", "PEO.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA", "CDR.WA",
    "LPP.WA", "CCC.WA", "MBK.WA", "SPL.WA", "JSW.WA", "ACP.WA", "KTY.WA", "BDX.WA",
    "OPL.WA", "TPE.WA", "ENA.WA", "MIL.WA", "XTB.WA", "11B.WA", "TEN.WA", "CAR.WA",
    "NEU.WA", "DOM.WA", "WPL.WA", "APR.WA", "GPW.WA", "BHW.WA", "ASB.WA", "ATT.WA",
    "LWB.WA", "CPS.WA", "MAB.WA", "ING.WA", "MRC.WA", "KRU.WA", "BXD.WA", "TOR.WA",
    "RBW.WA", "AMC.WA", "TXT.WA", "PLW.WA", "STP.WA", "PXM.WA", "PCR.WA", "ABE.WA",
    "DAT.WA", "VOX.WA", "MRB.WA", "KGN.WA"
]

LIQUIDITY_THRESHOLD_PLN = 1_000_000


def is_business_day(now: datetime) -> bool:
    return now.weekday() < 5


def should_send_report() -> bool:
    now = datetime.now(POLAND_TZ)

    # Twarda blokada weekendów — także przy ręcznym uruchomieniu.
    if not is_business_day(now):
        return False

    # Ręczne uruchomienie jest dozwolone tylko w dni robocze.
    if GITHUB_EVENT_NAME == "workflow_dispatch":
        return True

    return now.strftime("%H:%M") in REPORT_TIMES


def send(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Brak TELEGRAM_TOKEN lub TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
        },
        timeout=30,
    )
    response.raise_for_status()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()

    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def download_data(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Close", "High", "Volume"]
    for col in needed:
        if col not in df.columns:
            return None

    df = df.copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed)

    if len(df) < 60:
        return None

    return df


def analyze(symbol: str) -> dict | None:
    df = download_data(symbol)
    if df is None:
        return None

    df["VALUE"] = df["Close"] * df["Volume"]
    df["LIQ20"] = df["VALUE"].rolling(20).mean()

    last_liq20 = df["LIQ20"].iloc[-1]
    if pd.isna(last_liq20) or float(last_liq20) < LIQUIDITY_THRESHOLD_PLN:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])
    df["MOM5"] = df["Close"].pct_change(5)
    df["MOM20"] = df["Close"].pct_change(20)
    df["VOLAVG20"] = df["Volume"].rolling(20).mean()
    df["HIGH20"] = df["High"].rolling(20).max()

    df = df.dropna(subset=["SMA20", "SMA50", "RSI", "MOM5", "MOM20", "VOLAVG20", "HIGH20"])
    if df.empty:
        return None

    last = df.iloc[-1]

    close = float(last["Close"])
    sma20 = float(last["SMA20"])
    sma50 = float(last["SMA50"])
    rsi_val = float(last["RSI"])
    mom5 = float(last["MOM5"])
    mom20 = float(last["MOM20"])
    volume = float(last["Volume"])
    volavg20 = float(last["VOLAVG20"])
    high20 = float(last["HIGH20"])
    liq20 = float(last["LIQ20"])

    # Twarde filtry — ma być rzadki, mocny układ.
    if not (close > sma20 > sma50):
        return None

    if mom20 <= 0.05:
        return None

    if mom5 <= 0.02:
        return None

    if volume <= 1.5 * volavg20:
        return None

    if close < 0.98 * high20:
        return None

    if not (55 < rsi_val < 72):
        return None

    vol_spike = volume / volavg20 if volavg20 > 0 else 0.0
    breakout_distance = ((close / high20) - 1) * 100 if high20 > 0 else 0.0

    score = 0
    score += 20  # trend
    score += 20  # mom20
    score += 10  # mom5
    score += 20  # volume
    score += 20  # blisko wybicia
    score += 10  # RSI

    reasons = [
        f"momentum 20d: {mom20 * 100:.1f}%",
        f"momentum 5d: {mom5 * 100:.1f}%",
        f"wolumen: {vol_spike:.2f}x średniej 20d",
    ]

    extra = [
        f"RSI: {rsi_val:.1f}",
        f"odległość od 20d high: {breakout_distance:.1f}%",
        f"śr. obrót 20d: {liq20:,.0f} PLN".replace(",", " "),
    ]

    return {
        "symbol": symbol,
        "score": score,
        "price": round(close, 2),
        "reasons": reasons,
        "extra": extra,
    }


def build_message(results: list[dict]) -> str:
    now_str = datetime.now(POLAND_TZ).strftime("%H:%M")

    if not results:
        return (
            "GPW SMART MONEY BOT\n\n"
            f"Raport: {now_str}\n\n"
            "DECYZJA:\n"
            "BRAK SYGNAŁU\n\n"
            "Dziś żadna płynna spółka\n"
            "nie spełnia twardych kryteriów sygnału."
        )

    best = results[0]

    msg = (
        "GPW SMART MONEY BOT\n\n"
        f"Raport: {now_str}\n\n"
        "DECYZJA:\n"
        f"Najmocniejszy sygnał → {best['symbol']}\n\n"
        "Dlaczego:\n"
    )

    for item in best["reasons"]:
        msg += f"• {item}\n"

    msg += f"\nCena: {best['price']} PLN\n"
    msg += f"Score: {best['score']}\n"

    for item in best["extra"]:
        msg += f"{item}\n"

    if len(results) > 1:
        msg += "\nAlternatywy:\n"
        for alt in results[1:3]:
            msg += (
                f"{alt['symbol']} — "
                f"{alt['reasons'][0]}, {alt['reasons'][2]}\n"
            )

    return msg


def run() -> None:
    if not should_send_report():
        print("Poza dniem/godziną raportu. Koniec.")
        return

    results = []

    for symbol in WATCHLIST:
        result = analyze(symbol)
        if result is not None:
            results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    msg = build_message(results)
    send(msg)


if __name__ == "__main__":
    run()
