"""
GPW SMART MONEY BOT
===================
Wysyła raport na Telegram 3x dziennie w dni robocze (10:15, 13:00, 16:00 Warsaw).
Pobiera dynamicznie listę spółek z GPW przez stooq.pl
Sygnały: KUPUJ / CZEKAJ / UNIKAJ z prostym uzasadnieniem po polsku.

Wymagania:
  pip install yfinance pandas requests beautifulsoup4

Zmienne środowiskowe (GitHub Secrets):
  TELEGRAM_TOKEN   — token bota Telegram
  TELEGRAM_CHAT_ID — ID chatu
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "")

POLAND_TZ    = ZoneInfo("Europe/Warsaw")
REPORT_TIMES = {"10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00"}
BENCHMARK         = "^WIG20"
LIQUIDITY_MIN_PLN = 2_000_000   # minimalny średni dzienny obrót (PLN)
TOP_N             = 5           # ile spółek KUPUJ w raporcie
MIN_DATA_DAYS     = 60          # minimalna historia do analizy


# ---------------------------------------------------------------------------
# Harmonogram
# ---------------------------------------------------------------------------

def is_business_day(now: datetime) -> bool:
    return now.weekday() < 5


def should_send_report() -> bool:
    now = datetime.now(POLAND_TZ)
    if not is_business_day(now):
        return False
    if GITHUB_EVENT_NAME == "workflow_dispatch":
        return True
    return now.strftime("%H:%M") in REPORT_TIMES


# ---------------------------------------------------------------------------
# Pobieranie listy spółek GPW ze stooq.pl
# ---------------------------------------------------------------------------

def get_gpw_tickers() -> list[str]:
    """
    Pobiera listę spółek z GPW ze stooq.pl.
    Zwraca listę tickerów w formacie yfinance (np. PKN.WA).
    """
    tickers = []
    try:
        url = "https://stooq.pl/t/?i=513"  # lista spółek GPW główny rynek
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Tabela z tickerami
        table = soup.find("table", {"id": "fth1"})
        if table is None:
            raise ValueError("Nie znaleziono tabeli na stooq.pl")

        for row in table.find_all("tr")[1:]:  # pomijamy nagłówek
            cols = row.find_all("td")
            if len(cols) >= 2:
                ticker_raw = cols[1].get_text(strip=True).upper()
                if ticker_raw:
                    tickers.append(f"{ticker_raw}.WA")

    except Exception as e:
        print(f"Błąd pobierania tickerów ze stooq.pl: {e}")
        # Fallback — podstawowa lista jeśli stooq niedostępny
        tickers = [
            "PKN.WA", "PKO.WA", "PEO.WA", "PZU.WA", "KGH.WA", "DNP.WA",
            "ALE.WA", "CDR.WA", "LPP.WA", "CCC.WA", "MBK.WA", "SPL.WA",
            "JSW.WA", "ACP.WA", "KTY.WA", "BDX.WA", "OPL.WA", "TPE.WA",
            "ENA.WA", "MIL.WA", "XTB.WA", "11B.WA", "TEN.WA", "CAR.WA",
            "NEU.WA", "DOM.WA", "WPL.WA", "APR.WA", "GPW.WA", "BHW.WA",
            "ASB.WA", "ATT.WA", "LWB.WA", "CPS.WA", "MAB.WA", "ING.WA",
            "MRC.WA", "KRU.WA", "BXD.WA", "TOR.WA", "RBW.WA", "AMC.WA",
            "TXT.WA", "PLW.WA", "STP.WA", "PXM.WA", "PCR.WA", "ABE.WA",
            "DAT.WA", "VOX.WA", "MRB.WA", "KGN.WA",
        ]

    print(f"Pobrano {len(tickers)} spółek do analizy.")
    return tickers


# ---------------------------------------------------------------------------
# Wskaźniki techniczne
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ma_up   = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Pobieranie danych OHLCV
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
# Główna analiza — decyzja KUPUJ / CZEKAJ / UNIKAJ
# ---------------------------------------------------------------------------

def analyze(symbol: str, benchmark_returns: pd.Series | None) -> dict | None:
    df = download(symbol)
    if df is None:
        return None

    # --- Filtr płynności ---
    df["VALUE"] = df["Close"] * df["Volume"]
    df["LIQ20"] = df["VALUE"].rolling(20).mean()
    if df["LIQ20"].iloc[-1] < LIQUIDITY_MIN_PLN:
        return None

    # --- Wskaźniki ---
    df["SMA20"]   = df["Close"].rolling(20).mean()
    df["SMA50"]   = df["Close"].rolling(50).mean()
    df["RSI"]     = calc_rsi(df["Close"])
    df["MOM5"]    = df["Close"].pct_change(5)
    df["MOM20"]   = df["Close"].pct_change(20)
    df["VOL20"]   = df["Volume"].rolling(20).mean()
    df["HIGH20"]  = df["High"].rolling(20).max()
    df["OBV"]     = calc_obv(df["Close"], df["Volume"])
    df["BB_MID"]  = df["Close"].rolling(20).mean()
    df["BB_STD"]  = df["Close"].rolling(20).std()
    df["BB_UP"]   = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW"]  = df["BB_MID"] - 2 * df["BB_STD"]

    df = df.dropna()
    if df.empty:
        return None

    last = df.iloc[-1]

    close   = float(last["Close"])
    sma20   = float(last["SMA20"])
    sma50   = float(last["SMA50"])
    rsi     = float(last["RSI"])
    mom5    = float(last["MOM5"])
    mom20   = float(last["MOM20"])
    volume  = float(last["Volume"])
    vol20   = float(last["VOL20"])
    high20  = float(last["HIGH20"])
    liq20   = float(last["LIQ20"])
    bb_low  = float(last["BB_LOW"])
    bb_up   = float(last["BB_UP"])
    bb_mid  = float(last["BB_MID"])

    vol_ratio = volume / vol20 if vol20 > 0 else 0

    # --- Relative Strength vs WIG20 ---
    rs_vs_wig = 0.0
    if benchmark_returns is not None:
        stock_ret = df["Close"].pct_change(20).dropna()
        bench     = benchmark_returns.reindex(stock_ret.index).dropna()
        stock_ret = stock_ret.reindex(bench.index)
        if not stock_ret.empty:
            rs_vs_wig = float(stock_ret.iloc[-1]) - float(bench.iloc[-1])

    # --- OBV trend (ostatnie 10 dni) ---
    obv_trend = float(df["OBV"].iloc[-1]) - float(df["OBV"].iloc[-10])
    avg_vol   = float(df["Volume"].tail(20).mean())
    obv_rosnace = obv_trend > avg_vol * 2  # OBV rośnie wyraźnie

    # =========================================================
    # LOGIKA DECYZYJNA
    # =========================================================

    powody_kupuj  = []
    powody_czekaj = []
    powody_unikaj = []

    # --- SYGNAŁY POZYTYWNE (KUPUJ) ---

    # 1. Trend wzrostowy
    if close > sma20 > sma50:
        powody_kupuj.append("kurs powyżej średnich — trend wzrostowy")

    # 2. Silny momentum
    if mom20 > 0.08:
        powody_kupuj.append(f"wzrost o {mom20*100:.0f}% w ciągu miesiąca")
    elif mom20 > 0.04:
        powody_kupuj.append(f"umiarkowany wzrost {mom20*100:.0f}% w miesiącu")

    # 3. Wolumen potwierdza ruch
    if vol_ratio >= 2.0 and mom5 > 0:
        powody_kupuj.append(f"duże zainteresowanie — wolumen {vol_ratio:.1f}x powyżej średniej")
    elif vol_ratio >= 1.5 and mom5 > 0:
        powody_kupuj.append(f"rosnące zainteresowanie — wolumen {vol_ratio:.1f}x średniej")

    # 4. Blisko wybicia z oporu
    breakout_dist = (close / high20 - 1) * 100
    if breakout_dist >= -1.5:
        powody_kupuj.append("kurs blisko 20-dniowego maksimum — możliwe wybicie")

    # 5. RSI w strefie siły (nie wykupiony)
    if 55 <= rsi <= 70:
        powody_kupuj.append(f"RSI {rsi:.0f} — spółka ma siłę, ale nie jest wykupiona")

    # 6. Smart money — OBV rośnie
    if obv_rosnace and mom20 < 0.05:
        powody_kupuj.append("instytucje mogą akumulować — wolumen rośnie przy spokojnym kursie")

    # 7. Relative Strength
    if rs_vs_wig > 0.05:
        powody_kupuj.append(f"mocniejsza od WIG20 o {rs_vs_wig*100:.0f}pp w ostatnim miesiącu")

    # --- SYGNAŁY OSTRZEGAWCZE (CZEKAJ/UNIKAJ) ---

    # Trend spadkowy
    if close < sma50:
        powody_unikaj.append("kurs poniżej długiej średniej — trend spadkowy")
    elif close < sma20:
        powody_czekaj.append("kurs poniżej krótkoterminowej średniej — brak trendu")

    # Wykupienie
    if rsi > 75:
        powody_unikaj.append(f"RSI {rsi:.0f} — spółka silnie wykupiona, ryzyko korekty")
    elif rsi > 70:
        powody_czekaj.append(f"RSI {rsi:.0f} — spółka blisko wykupienia")

    # Wyprzedanie
    if rsi < 35:
        powody_czekaj.append(f"RSI {rsi:.0f} — możliwe odbicie, ale brak sygnału kupna")

    # Słaby momentum
    if mom20 < -0.05:
        powody_unikaj.append(f"spadek o {abs(mom20)*100:.0f}% w ciągu miesiąca")

    # Wolumen przy spadku
    if vol_ratio >= 1.8 and mom5 < -0.02:
        powody_unikaj.append(f"duży wolumen przy spadku — wyprzedaż")

    # Słabsza od rynku
    if rs_vs_wig < -0.08:
        powody_unikaj.append(f"słabsza od WIG20 o {abs(rs_vs_wig)*100:.0f}pp — inwestorzy omijają")

    # =========================================================
    # DECYZJA KOŃCOWA
    # =========================================================

    kupuj_score = len(powody_kupuj)
    unikaj_score = len(powody_unikaj)

    if unikaj_score >= 2:
        decyzja = "UNIKAJ"
        powody = powody_unikaj[:3]
    elif kupuj_score >= 3 and unikaj_score == 0:
        decyzja = "KUPUJ"
        powody = powody_kupuj[:4]
    elif kupuj_score >= 2 and unikaj_score <= 1:
        decyzja = "CZEKAJ"
        powody = powody_kupuj[:2] + powody_czekaj[:1]
    elif kupuj_score <= 1:
        decyzja = "CZEKAJ"
        powody = powody_czekaj[:2] if powody_czekaj else ["brak wyraźnych sygnałów"]
    else:
        decyzja = "CZEKAJ"
        powody = (powody_kupuj[:1] + powody_czekaj[:1] + powody_unikaj[:1])[:3]

    if not powody:
        return None

    return {
        "symbol": symbol.replace(".WA", ""),
        "decyzja": decyzja,
        "kupuj_score": kupuj_score,
        "cena": round(close, 2),
        "liq20": liq20,
        "powody": powody,
        "mom20": mom20,
        "rsi": rsi,
    }


# ---------------------------------------------------------------------------
# Budowanie wiadomości
# ---------------------------------------------------------------------------

EMOJI = {
    "KUPUJ":  "🟢",
    "CZEKAJ": "🟡",
    "UNIKAJ": "🔴",
}


def format_pln(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M PLN"
    return f"{value/1_000:.0f}K PLN"


def build_message(kupuj: list, czekaj: list, unikaj_count: int) -> str:
    now_str = datetime.now(POLAND_TZ).strftime("%d.%m.%Y %H:%M")
    lines = [f"📊 GPW SMART MONEY BOT", f"{now_str}", "─" * 28]

    if not kupuj:
        lines += [
            "",
            "🟡 BRAK SYGNAŁÓW KUPNA",
            "",
            "Dziś żadna płynna spółka nie spełnia",
            "kryteriów do zakupu.",
            "Najlepiej poczekać na lepszy moment.",
        ]
    else:
        lines += ["", f"🟢 DO ROZWAŻENIA ({len(kupuj)} spółek):", ""]
        for r in kupuj:
            lines.append(
                f"{EMOJI['KUPUJ']} {r['symbol']}  |  {r['cena']} PLN"
            )
            for p in r["powody"]:
                lines.append(f"   · {p}")
            lines.append(
                f"   Obrót śr.: {format_pln(r['liq20'])}"
            )
            lines.append("")

    if czekaj:
        lines.append(f"🟡 OBSERWUJ ({len(czekaj)} spółek): " +
                     ", ".join(r["symbol"] for r in czekaj[:8]))
        lines.append("")

    lines.append(f"🔴 Odfiltrowano: {unikaj_count} spółek w trendzie spadkowym")
    lines.append("")
    lines.append("⚠️ To nie jest porada inwestycyjna.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wysyłanie na Telegram
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
    if not should_send_report():
        print("Poza dniem/godziną raportu. Koniec.")
        return

    print("Pobieranie listy spółek GPW...")
    tickers = get_gpw_tickers()

    print("Pobieranie benchmarku (WIG20)...")
    benchmark_returns = download_benchmark()

    kupuj  = []
    czekaj = []
    unikaj_count = 0

    for i, symbol in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] Analizuję {symbol}...")
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

        # Małe opóźnienie żeby nie przeciążać yfinance
        time.sleep(0.3)

    # Sortuj KUPUJ po sile (liczba powodów * momentum)
    kupuj.sort(key=lambda x: (x["kupuj_score"], x["mom20"]), reverse=True)
    kupuj = kupuj[:TOP_N]

    # Sortuj CZEKAJ alfabetycznie
    czekaj.sort(key=lambda x: x["symbol"])

    msg = build_message(kupuj, czekaj, unikaj_count)
    print("\n--- WIADOMOŚĆ ---")
    print(msg)

    send_telegram(msg)
    print("\nWysłano na Telegram.")


if __name__ == "__main__":
    run()
