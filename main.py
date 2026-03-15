import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCHLIST = [
"PKN.WA","PKO.WA","PEO.WA","PZU.WA","KGH.WA","DNP.WA",
"ALE.WA","CDR.WA","LPP.WA","CCC.WA","MBK.WA","SPL.WA",
"JSW.WA","ACP.WA","KTY.WA","BDX.WA","OPL.WA","TPE.WA",
"ENA.WA","MIL.WA","XTB.WA","11B.WA","TEN.WA"
]

POLAND = ZoneInfo("Europe/Warsaw")

def send(msg):

    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    requests.post(url,data={
        "chat_id":TELEGRAM_CHAT_ID,
        "text":msg
    })

def rsi(series,period=14):

    delta=series.diff()

    up=delta.clip(lower=0)

    down=-delta.clip(upper=0)

    ma_up=up.ewm(com=period-1,adjust=False).mean()

    ma_down=down.ewm(com=period-1,adjust=False).mean()

    rs=ma_up/ma_down

    return 100-(100/(1+rs))

def analyze(symbol):

    df=yf.download(symbol,period="6mo",interval="1d",progress=False)

    if df.empty:

        return None

    df["SMA20"]=df["Close"].rolling(20).mean()

    df["SMA50"]=df["Close"].rolling(50).mean()

    df["RSI"]=rsi(df["Close"])

    last=df.iloc[-1]

    score=0

    if last["Close"]>last["SMA20"]:

        score+=10

    if last["SMA20"]>last["SMA50"]:

        score+=10

    if 50<last["RSI"]<70:

        score+=10

    return {

        "symbol":symbol,

        "price":round(last["Close"],2),

        "score":score,

        "rsi":round(last["RSI"],1)

    }

def run():

    results=[]

    for s in WATCHLIST:

        r=analyze(s)

        if r:

            results.append(r)

    results=sorted(results,key=lambda x:x["score"],reverse=True)

    msg="GPW BOT\n\n"

    for r in results[:5]:

        msg+=f"{r['symbol']}  score:{r['score']}  rsi:{r['rsi']}  price:{r['price']}\n"

    send(msg)

if __name__=="__main__":

    run()
