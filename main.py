import os
import requests
import pandas as pd
import yfinance as yf

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCHLIST = [
    "PKN.WA", "PKO.WA", "PEO.WA", "PZU.WA", "KGH.WA", "DNP.WA",
    "ALE.WA", "CDR.WA", "LPP.WA", "CCC.WA", "MBK.WA", "SPL.WA",
    "JSW.WA", "ACP.WA", "KTY.WA", "BDX.WA", "OPL.WA", "TPE.WA",
    "ENA.WA", "MIL.WA", "XTB.WA", "11B.WA", "TEN.WA"
]

def send(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Brak TELEGRAM_TOKEN lub TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        },
        timeout=30
    )
    r.raise_for_status()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()

    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def analyze(symbol: str):
    try:
        df = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Close"]
    for col in required_cols:
        if col not in df.columns:
            return None

    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    if len(df) < 50:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])

    df = df.dropna(subset=["SMA20", "SMA50", "RSI"])
    if df.empty:
        return None

    last = df.iloc[-1]

    close = float(last["Close"])
    sma20 = float(last["SMA20"])
    sma50 = float(last["SMA50"])
    rsi_val = float(last["RSI"])

    score = 0

    if close > sma20:
        score += 10

    if sma20 > sma50:
        score += 10

    if 50 < rsi_val < 70:
        score += 10

    return {
        "symbol": symbol,
        "price": round(close, 2),
        "score": score,
        "rsi": round(rsi_val, 1)
    }

def run():
    results = []

    for symbol in WATCHLIST:
        result = analyze(symbol)
        if result is not None:
            results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if not results:
        send("GPW BOT\n\nBrak danych do raportu.")
        return

    msg = "GPW BOT\n\n"

    for r in results[:5]:
        msg += f"{r['symbol']}  score:{r['score']}  rsi:{r['rsi']}  price:{r['price']}\n"

    send(msg)

if __name__ == "__main__":
    run()
