"""
GPW SMART MONEY BOT
===================
Wysyła raport na Telegram przy każdym uruchomieniu w dni robocze.
Harmonogram uruchamiania kontroluje GitHub Actions (cron).
Sygnały: KUPUJ / CZEKAJ / UNIKAJ z prostym uzasadnieniem po polsku.

Wymagania:
  pip install yfinance pandas requests
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

POLAND_TZ         = ZoneInfo("Europe/Warsaw")
LIQUIDITY_MIN_PLN = 2_000_000
TOP_N             = 5
MIN_DATA_DAYS     = 60
BENCHMARK         = "^WIG20"

# Pełna lista płynnych spółek GPW
WATCHLIST = [
    "PKN.WA", "PKO.WA", "PEO.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA",
    "CDR.WA", "LPP.WA", "CCC.WA", "MBK.WA", "SPL.WA", "JSW.WA", "ACP.WA",
    "KTY.WA", "BDX.WA", "OPL.WA", "TPE.WA", "ENA.WA", "MIL.WA", "XTB.WA",
    "11B.WA", "TEN.WA", "CAR.WA", "NEU.WA", "DOM.WA", "WPL.WA", "APR.WA",
    "GPW.WA", "BHW.WA", "ASB.WA", "ATT.WA", "LWB.WA", "CPS.WA", "MAB.WA",
    "ING.WA", "MRC.WA", "KRU.WA", "BXD.WA", "TOR.WA", "RBW.WA", "AMC.WA",
    "TXT.WA", "PLW.WA", "STP.WA", "PXM.WA", "PCR.WA", "ABE.WA", "DAT.WA",
    "VOX.WA", "MRB.WA", "KGN.WA", "ACG.WA", "AGO.WA", "ALR.WA", "ALT.WA",
    "ARH.WA", "BAH.WA", "BCM.WA", "BFT.WA", "BMC.WA", "BNP.WA", "BOS.WA",
    "BRS.WA", "CAV.WA", "CBF.WA", "CFI.WA", "CIG.WA", "CLN.WA", "CMR.WA",
    "CNT.WA", "COG.WA", "CPG.WA", "CRC.WA", "CRJ.WA", "EAT.WA", "ELT.WA",
    "EMC.WA", "ENI.WA", "ERB.WA", "ERG.WA", "ETL.WA", "EUR.WA", "FMF.WA",
    "FOR.WA", "GNB.WA", "GTC.WA", "HMI.WA", "HRP.WA", "HRS.WA", "HUG.WA",
    "IFC.WA", "IMC.WA", "IPO.WA", "IZO.WA", "KER.WA", "KPL.WA", "KSW.WA",
    "LBW.WA", "LEN.WA", "LTS.WA", "MAK.WA", "MCF.WA", "MDI.WA", "MEX.WA",
    "MFO.WA", "MGT.WA", "MLG.WA", "MOJ.WA", "MPW.WA", "MSW.WA", "MTV.WA",
    "MWT.WA", "MXC.WA", "NNG.WA", "NTT.WA", "NVT.WA", "OAT.WA", "OBL.WA",
    "OND.WA", "ONO.WA", "OPF.WA", "PAL.WA", "PCX.WA", "PEP.WA", "PHN.WA",
    "PKP.WA", "PMP.WA", "PNT.WA", "POZ.WA", "PRM.WA", "PRF.WA", "PSW.WA",
    "PTC.WA", "PTG.WA", "PVT.WA", "QMK.WA", "RAB.WA", "RFK.WA", "RLP.WA",
    "RNK.WA", "SAF.WA", "SAN.WA", "SCO.WA", "SEK.WA", "SGN.WA", "SHG.WA",
    "SKH.WA", "SKT.WA", "SLV.WA", "SNK.WA", "SNT.WA", "SNW.WA", "SOK.WA",
    "STF.WA", "STX.WA", "SUN.WA", "SVE.WA", "SWG.WA", "TAR.WA", "TBL.WA",
    "TIM.WA", "TME.WA", "TOA.WA", "TRK.WA", "TXM.WA", "UNT.WA", "URS.WA",
    "VGO.WA", "VTI.WA", "WAS.WA", "WLT.WA", "WRN.WA", "WWL.WA", "ZEP.WA",
    "ZUE.WA", "ZWC.WA",
]


# ---------------------------------------------------------------------------
# Wskaźniki techniczne
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta   = series.diff()
    up      = delta.clip(lower=0)
    down    = -delta.clip(upper=0)
    ma_up   = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Pobieranie danych
# ---------------------------------------------------------------------------

def download(symbol: str) -> pd.DataFrame | None:
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

    needed = ["Close", "High", "Low", "Volume"]
    for col in needed:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed)
    return df if len(df) >= MIN_DATA_DAYS else None


def download_benchmark() -> pd.Series | None:
    df = download(BENCHMARK)
    if df is None:
        return None
    return df["Close"].pct_change()


# ---------------------------------------------------------------------------
# Analiza spółki
# ---------------------------------------------------------------------------

def analyze(symbol: str, benchmark_returns: pd.Series | None) -> dict | None:
    df = download(symbol)
    if df is None:
        return None

    df["VALUE"] = df["Close"] * df["Volume"]
    df["LIQ20"] = df["VALUE"].rolling(20).mean()
    if df["LIQ20"].iloc[-1] < LIQUIDITY_MIN_PLN:
        return None

    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["RSI"]    = calc_rsi(df["Close"])
    df["MOM5"]   = df["Close"].pct_change(5)
    df["MOM20"]  = df["Close"].pct_change(20)
    df["VOL20"]  = df["Volume"].rolling(20).mean()
    df["HIGH20"] = df["High"].rolling(20).max()
    df["OBV"]    = calc_obv(df["Close"], df["Volume"])

    df = df.dropna()
    if df.empty:
        return None

    last = df.iloc[-1]

    close     = float(last["Close"])
    sma20     = float(last["SMA20"])
    sma50     = float(last["SMA50"])
    rsi       = float(last["RSI"])
    mom5      = float(last["MOM5"])
    mom20     = float(last["MOM20"])
    volume    = float(last["Volume"])
    vol20     = float(last["VOL20"])
    high20    = float(last["HIGH20"])
    liq20     = float(last["LIQ20"])

    vol_ratio = volume / vol20 if vol20 > 0 else 0

    # Relative Strength vs WIG20
    rs_vs_wig = 0.0
    if benchmark_returns is not None:
        stock_ret = df["Close"].pct_change(20).dropna()
        bench     = benchmark_returns.reindex(stock_ret.index).dropna()
        stock_ret = stock_ret.reindex(bench.index)
        if not stock_ret.empty:
            rs_vs_wig = float(stock_ret.iloc[-1]) - float(bench.iloc[-1])

    # OBV trend
    obv_rosnace = False
    if len(df) >= 10:
        obv_trend   = float(df["OBV"].iloc[-1]) - float(df["OBV"].iloc[-10])
        avg_vol     = float(df["Volume"].tail(20).mean())
        obv_rosnace = obv_trend > avg_vol * 2

    # --- Sygnały ---
    powody_kupuj  = []
    powody_czekaj = []
    powody_unikaj = []

    if close > sma20 > sma50:
        powody_kupuj.append("kurs powyżej średnich — trend wzrostowy")

    if mom20 > 0.08:
        powody_kupuj.append(f"wzrost o {mom20*100:.0f}% w ciągu miesiąca")
    elif mom20 > 0.04:
        powody_kupuj.append(f"umiarkowany wzrost {mom20*100:.0f}% w miesiącu")

    if vol_ratio >= 2.0 and mom5 > 0:
        powody_kupuj.append(f"duże zainteresowanie — wolumen {vol_ratio:.1f}x powyżej średniej")
    elif vol_ratio >= 1.5 and mom5 > 0:
        powody_kupuj.append(f"rosnące zainteresowanie — wolumen {vol_ratio:.1f}x średniej")

    if (close / high20 - 1) * 100 >= -1.5:
        powody_kupuj.append("kurs blisko 20-dniowego maksimum — możliwe wybicie")

    if 55 <= rsi <= 70:
        powody_kupuj.append(f"RSI {rsi:.0f} — spółka ma siłę, ale nie jest wykupiona")

    if obv_rosnace and mom20 < 0.05:
        powody_kupuj.append("instytucje mogą akumulować — wolumen rośnie przy spokojnym kursie")

    if rs_vs_wig > 0.05:
        powody_kupuj.append(f"mocniejsza od WIG20 o {rs_vs_wig*100:.0f}pp w ostatnim miesiącu")

    if close < sma50:
        powody_unikaj.append("kurs poniżej długiej średniej — trend spadkowy")
    elif close < sma20:
        powody_czekaj.append("kurs poniżej krótkoterminowej średniej — brak trendu")

    if rsi > 75:
        powody_unikaj.append(f"RSI {rsi:.0f} — spółka silnie wykupiona, ryzyko korekty")
    elif rsi > 70:
        powody_czekaj.append(f"RSI {rsi:.0f} — spółka blisko wykupienia")

    if rsi < 35:
        powody_czekaj.append(f"RSI {rsi:.0f} — możliwe odbicie, ale brak sygnału kupna")

    if mom20 < -0.05:
        powody_unikaj.append(f"spadek o {abs(mom20)*100:.0f}% w ciągu miesiąca")

    if vol_ratio >= 1.8 and mom5 < -0.02:
        powody_unikaj.append(f"duży wolumen przy spadku — wyprzedaż")

    if rs_vs_wig < -0.08:
        powody_unikaj.append(f"słabsza od WIG20 o {abs(rs_vs_wig)*100:.0f}pp")

    # --- Decyzja ---
    kupuj_score  = len(powody_kupuj)
    unikaj_score = len(powody_unikaj)

    if unikaj_score >= 2:
        decyzja = "UNIKAJ"
        powody  = powody_unikaj[:3]
    elif kupuj_score >= 3 and unikaj_score == 0:
        decyzja = "KUPUJ"
        powody  = powody_kupuj[:4]
    elif kupuj_score >= 2 and unikaj_score <= 1:
        decyzja = "CZEKAJ"
        powody  = powody_kupuj[:2] + powody_czekaj[:1]
    else:
        decyzja = "CZEKAJ"
        powody  = (powody_czekaj or powody_kupuj or ["brak wyraźnych sygnałów"])[:2]

    return {
        "symbol":      symbol.replace(".WA", ""),
        "decyzja":     decyzja,
        "kupuj_score": kupuj_score,
        "cena":        round(close, 2),
        "liq20":       liq20,
        "powody":      powody,
        "mom20":       mom20,
        "rsi":         rsi,
    }


# ---------------------------------------------------------------------------
# Budowanie wiadomości
# ---------------------------------------------------------------------------

def format_pln(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M PLN"
    return f"{value/1_000:.0f}K PLN"


def build_message(kupuj: list, czekaj: list, unikaj_count: int) -> str:
    now_str = datetime.now(POLAND_TZ).strftime("%d.%m.%Y %H:%M")
    lines   = ["📊 GPW SMART MONEY BOT", now_str, "─" * 28]

    if not kupuj:
        lines += [
            "",
            "🟡 BRAK SYGNAŁÓW KUPNA",
            "",
            "Żadna płynna spółka nie spełnia",
            "kryteriów do zakupu.",
            "Najlepiej poczekać na lepszy moment.",
        ]
    else:
        lines += ["", f"🟢 DO ROZWAŻENIA ({len(kupuj)} spółek):", ""]
        for r in kupuj:
            lines.append(f"🟢 {r['symbol']}  |  {r['cena']} PLN")
            for p in r["powody"]:
                lines.append(f"   · {p}")
            lines.append(f"   Obrót śr.: {format_pln(r['liq20'])}")
            lines.append("")

    if czekaj:
        lines.append(
            "🟡 OBSERWUJ: " + ", ".join(r["symbol"] for r in czekaj[:8])
        )
        lines.append("")

    lines.append(f"🔴 W trendzie spadkowym: {unikaj_count} spółek")
    lines.append("")
    lines.append("⚠️ To nie jest porada inwestycyjna.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Brak TELEGRAM_TOKEN lub TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
        timeout=30,
    )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"Start: {datetime.now(POLAND_TZ).strftime('%d.%m.%Y %H:%M')} Warsaw")
    print(f"Analiza {len(WATCHLIST)} spółek...")

    benchmark_returns = download_benchmark()
    print(f"Benchmark {'OK' if benchmark_returns is not None else 'NIEDOSTĘPNY'}")

    kupuj        = []
    czekaj       = []
    unikaj_count = 0

    for i, symbol in enumerate(WATCHLIST):
        print(f"  [{i+1}/{len(WATCHLIST)}] {symbol}...")
        try:
            result = analyze(symbol, benchmark_returns)
        except Exception as e:
            print(f"    Błąd: {e}")
            continue

        if result is None:
            continue

        if result["decyzja"] == "KUPUJ":
            kupuj.append(result)
        elif result["decyzja"] == "CZEKAJ":
            czekaj.append(result)
        else:
            unikaj_count += 1

        time.sleep(0.3)

    kupuj.sort(key=lambda x: (x["kupuj_score"], x["mom20"]), reverse=True)
    kupuj = kupuj[:TOP_N]
    czekaj.sort(key=lambda x: x["symbol"])

    print(f"Wyniki: KUPUJ={len(kupuj)}, CZEKAJ={len(czekaj)}, UNIKAJ={unikaj_count}")

    msg = build_message(kupuj, czekaj, unikaj_count)
    print("\n--- WIADOMOŚĆ ---")
    print(msg)

    send_telegram(msg)
    print("Wysłano na Telegram.")


if __name__ == "__main__":
    run()
