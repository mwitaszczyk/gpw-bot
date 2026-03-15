import os
import requests
import pandas as pd
import yfinance as yf

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Płynny koszyk GPW
WATCHLIST = [
"PKN.WA","PKO.WA","PEO.WA","PZU.WA","KGH.WA","DNP.WA","ALE.WA","CDR.WA",
"LPP.WA","CCC.WA","MBK.WA","SPL.WA","JSW.WA","ACP.WA","KTY.WA","BDX.WA",
"OPL.WA","TPE.WA","ENA.WA","MIL.WA","XTB.WA","11B.WA","TEN.WA","CAR.WA",
"NEU.WA","DOM.WA","WPL.WA","APR.WA","GPW.WA","BHW.WA","ASB.WA","BFT.WA"
]

LIQUIDITY_THRESHOLD = 1_000_000


def send(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    })


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def analyze(symbol):

    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    except:
        return None

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["VALUE"] = df["Close"] * df["Volume"]
    liquidity = df["VALUE"].rolling(20).mean().iloc[-1]

    if liquidity < LIQUIDITY_THRESHOLD:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])

    df["MOM5"] = df["Close"].pct_change(5)
    df["MOM20"] = df["Close"].pct_change(20)

    df["VOLAVG"] = df["Volume"].rolling(20).mean()

    df["HIGH20"] = df["High"].rolling(20).max()

    df = df.dropna()

    if df.empty:
        return None

    last = df.iloc[-1]

    reasons = []
    score = 0

    if last["Close"] > last["SMA20"] > last["SMA50"]:
        score += 20
        reasons.append("trend wzrostowy")

    if last["MOM20"] > 0.05:
        score += 20
        reasons.append("silne momentum")

    if last["MOM5"] > 0.02:
        score += 10
        reasons.append("krótkoterminowy wzrost")

    if last["Volume"] > 1.5 * last["VOLAVG"]:
        score += 20
        reasons.append("napływ kapitału")

    if last["Close"] >= last["HIGH20"] * 0.98:
        score += 20
        reasons.append("blisko wybicia")

    if 55 < last["RSI"] < 72:
        score += 10
        reasons.append("zdrowy RSI")

    if score < 60:
        return None

    return {
        "symbol": symbol,
        "score": score,
        "price": round(last["Close"], 2),
        "reasons": reasons[:3]
    }


def run():

    results = []

    for s in WATCHLIST:

        r = analyze(s)

        if r:
            results.append(r)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if not results:

        msg = """GPW SMART MONEY BOT

DECYZJA:
BRAK SYGNAŁU

Dziś żadna płynna spółka
nie spełnia kryteriów napływu kapitału.
"""

        send(msg)
        return

    best = results[0]

    msg = "GPW SMART MONEY BOT\n\n"

    msg += f"DECYZJA:\nNajmocniejszy sygnał → {best['symbol']}\n\n"

    msg += "Dlaczego:\n"

    for r in best["reasons"]:
        msg += f"• {r}\n"

    msg += f"\nCena: {best['price']} PLN\n\n"

    if len(results) > 1:

        msg += "Alternatywy:\n"

        for r in results[1:3]:
            msg += f"{r['symbol']} — {', '.join(r['reasons'])}\n"

    send(msg)


if __name__ == "__main__":
    run()
