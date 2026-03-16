"""
GPW SMART MONEY BOT
===================
Wysyła raport na Telegram 3x dziennie w dni robocze (10:15, 13:00, 16:00 Warsaw).
Skanuje spółki z watchlisty pod kątem 4 setupów swingowych:
  1. Volume Breakout
  2. Momentum Surge (z Relative Strength vs WIG20)
  3. Accumulation Setup (OBV)
  4. Volatility Squeeze + Expansion (Bollinger Bands)

Wymagania:
  pip install yfinance pandas requests

Zmienne środowiskowe (GitHub Secrets):
  TELEGRAM_TOKEN   — token bota Telegram
  TELEGRAM_CHAT_ID — ID chatu / kanału
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "")

POLAND_TZ = ZoneInfo("Europe/Warsaw")
REPORT_TIMES = {"10:15", "13:00", "16:00"}

# Watchlista — spółki GPW do skanowania
WATCHLIST = [
    "PKN.WA", "PKO.WA", "PEO.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA", "CDR.WA",
    "LPP.WA", "CCC.WA", "MBK.WA", "SPL.WA", "JSW.WA", "ACP.WA", "KTY.WA", "BDX.WA",
    "OPL.WA", "TPE.WA", "ENA.WA", "MIL.WA", "XTB.WA", "11B.WA", "TEN.WA", "CAR.WA",
    "NEU.WA", "DOM.WA", "WPL.WA", "APR.WA", "GPW.WA", "BHW.WA", "ASB.WA", "ATT.WA",
    "LWB.WA", "CPS.WA", "MAB.WA", "ING.WA", "MRC.WA", "KRU.WA", "BXD.WA", "TOR.WA",
    "RBW.WA", "AMC.WA", "TXT.WA", "PLW.WA", "STP.WA", "PXM.WA", "PCR.WA", "ABE.WA",
    "DAT.WA", "VOX.WA", "MRB.WA", "KGN.WA",
]

BENCHMARK = "^WIG20"          # indeks odniesienia dla Relative Strength
LIQUIDITY_MIN_PLN = 1_000_000  # minimalny średni dzienny obrót (PLN)
TOP_N = 5                      # ile spółek w raporcie


# ---------------------------------------------------------------------------
# Wagi sygnałów (suma = 100)
# ---------------------------------------------------------------------------
WEIGHT_VOLUME_BREAKOUT  = 30
WEIGHT_MOMENTUM_SURGE   = 25
WEIGHT_ACCUMULATION     = 25
WEIGHT_SQUEEZE          = 20


# ---------------------------------------------------------------------------
# Pomocnicze: harmonogram
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
# Pomocnicze: wskaźniki techniczne
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up   = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def calc_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # szerokość względna
    return sma, upper, lower, width


# ---------------------------------------------------------------------------
# Pobieranie danych
# ---------------------------------------------------------------------------

def download(symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period=period,
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
    return df if len(df) >= 60 else None


def download_benchmark() -> pd.Series | None:
    """Zwraca serię dziennych zwrotów benchmarku (WIG20)."""
    df = download(BENCHMARK)
    if df is None:
        return None
    return df["Close"].pct_change()


# ---------------------------------------------------------------------------
# Sygnały
# ---------------------------------------------------------------------------

def signal_volume_breakout(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Volume Breakout:
    - kurs przebija 20-dniowe maksimum (lub jest bardzo blisko)
    - wolumen >= 2x średniej 20d
    - RSI w zakresie 50–75 (nie wykupiony, ale z siłą)
    Zwraca (score 0–100, lista powodów).
    """
    df["VOLAVG20"] = df["Volume"].rolling(20).mean()
    df["HIGH20"]   = df["High"].rolling(20).max()
    df["RSI"]      = calc_rsi(df["Close"])
    df = df.dropna()
    if df.empty:
        return 0.0, []

    last = df.iloc[-1]
    close    = float(last["Close"])
    high20   = float(last["HIGH20"])
    volume   = float(last["Volume"])
    volavg20 = float(last["VOLAVG20"])
    rsi_val  = float(last["RSI"])

    vol_ratio = volume / volavg20 if volavg20 > 0 else 0
    breakout  = close / high20 if high20 > 0 else 0

    if vol_ratio < 1.5 or breakout < 0.97 or not (48 <= rsi_val <= 78):
        return 0.0, []

    # Scoring ciągły: im bliżej/powyżej high20 i wyższy wolumen, tym lepiej
    score = 0.0
    score += min(vol_ratio / 3.0, 1.0) * 50      # do 50 pkt za wolumen (max przy 3x)
    score += min((breakout - 0.97) / 0.05, 1.0) * 30  # do 30 pkt za bliskość wybicia
    score += (1 - abs(rsi_val - 62) / 20) * 20   # do 20 pkt za RSI w optimum ~62

    reasons = [
        f"wolumen {vol_ratio:.1f}x średniej",
        f"kurs {(breakout-1)*100:+.1f}% od 20d high",
        f"RSI {rsi_val:.0f}",
    ]
    return min(score, 100.0), reasons


def signal_momentum_surge(
    df: pd.DataFrame, benchmark_returns: pd.Series | None
) -> tuple[float, list[str]]:
    """
    Momentum Surge + Relative Strength:
    - silny ruch 5d i 20d
    - spółka mocniejsza niż WIG20 w tym samym oknie
    """
    df["MOM5"]  = df["Close"].pct_change(5)
    df["MOM20"] = df["Close"].pct_change(20)
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df = df.dropna()
    if df.empty:
        return 0.0, []

    last = df.iloc[-1]
    mom5  = float(last["MOM5"])
    mom20 = float(last["MOM20"])
    close = float(last["Close"])
    sma20 = float(last["SMA20"])
    sma50 = float(last["SMA50"])

    # Twarde filtry
    if mom20 < 0.04 or mom5 < 0.01 or close <= sma20 or sma20 <= sma50:
        return 0.0, []

    # Relative Strength vs benchmark
    rs_bonus = 0.0
    rs_label = ""
    if benchmark_returns is not None:
        # Wyrównaj indeksy
        shared = df["Close"].pct_change(20).dropna()
        bench  = benchmark_returns.reindex(shared.index).dropna()
        shared = shared.reindex(bench.index)
        if not shared.empty:
            stock_ret = float(shared.iloc[-1])
            bench_ret = float(bench.iloc[-1])
            rs = stock_ret - bench_ret
            if rs > 0:
                rs_bonus = min(rs / 0.10, 1.0) * 30  # max 30 pkt przy RS > 10pp
                rs_label = f"RS vs WIG20: {rs*100:+.1f}pp"

    score = 0.0
    score += min(mom20 / 0.15, 1.0) * 40   # do 40 pkt (optimum: +15%)
    score += min(mom5  / 0.06, 1.0) * 30   # do 30 pkt (optimum: +6%)
    score += rs_bonus

    reasons = [
        f"momentum 20d: {mom20*100:+.1f}%",
        f"momentum 5d: {mom5*100:+.1f}%",
    ]
    if rs_label:
        reasons.append(rs_label)

    return min(score, 100.0), reasons


def signal_accumulation(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Accumulation Setup (OBV):
    - OBV rośnie przez ostatnie N dni gdy kurs jest relatywnie spokojny
    - dywergencja: OBV idzie w górę szybciej niż kurs → akumulacja
    """
    df["OBV"] = calc_obv(df["Close"], df["Volume"])
    df = df.dropna()
    if len(df) < 20:
        return 0.0, []

    # Okno 10 dni
    window = 10
    recent = df.iloc[-window:]

    obv_change   = (float(recent["OBV"].iloc[-1]) - float(recent["OBV"].iloc[0]))
    price_change = (float(recent["Close"].iloc[-1]) / float(recent["Close"].iloc[0])) - 1

    # OBV musi rosnąć
    if obv_change <= 0:
        return 0.0, []

    # Normalizuj OBV change do % (vs średni wolumen)
    avg_vol = float(df["Volume"].tail(20).mean())
    obv_pct = obv_change / (avg_vol * window) if avg_vol > 0 else 0

    # Akumulacja: OBV rośnie, kurs nie uciekł jeszcze (price_change < 0.08)
    if price_change > 0.10:
        return 0.0, []  # kurs już wybił — to nie jest "pre-breakout" accumulation

    # Im więcej OBV rośnie przy spokojnym kursie, tym lepszy sygnał
    divergence = obv_pct / max(abs(price_change), 0.005)

    score = min(divergence * 20, 60.0)  # max 60 za samą dywergencję
    score += min(obv_pct * 200, 40.0)   # max 40 za bezwzględny wzrost OBV

    if score < 15:
        return 0.0, []

    reasons = [
        f"OBV +{obv_pct*100:.1f}% (10d, vs avg vol)",
        f"cena {price_change*100:+.1f}% w tym czasie",
        "akumulacja bez ucieczki kursu",
    ]
    return min(score, 100.0), reasons


def signal_squeeze(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Volatility Squeeze + Expansion:
    - Bollinger Bands bardzo wąskie (historycznie niski width) → cisza
    - ostatni dzień: nagły wzrost wolumenu i ceny → potencjalny kierunek
    """
    sma, upper, lower, width = calc_bollinger(df["Close"])
    df["BB_WIDTH"] = width
    df["VOLAVG20"] = df["Volume"].rolling(20).mean()
    df = df.dropna()
    if len(df) < 30:
        return 0.0, []

    last     = df.iloc[-1]
    bb_width = float(last["BB_WIDTH"])
    volume   = float(last["Volume"])
    volavg20 = float(last["VOLAVG20"])
    close    = float(last["Close"])
    sma_val  = float(sma.iloc[-1])

    # Historyczne percentyle szerokości BB (ostatnie 60 sesji)
    hist_width = df["BB_WIDTH"].tail(60)
    pct10 = float(hist_width.quantile(0.10))
    pct30 = float(hist_width.quantile(0.30))

    # Squeeze: obecna szerokość w dolnych 30% historii
    if bb_width > pct30:
        return 0.0, []

    # Ekspansja: wolumen skok i kurs powyżej SMA
    vol_ratio = volume / volavg20 if volavg20 > 0 else 0
    if vol_ratio < 1.3 or close <= sma_val:
        return 0.0, []

    # Im głębszy squeeze i silniejsza ekspansja, tym lepiej
    squeeze_depth = 1 - (bb_width / pct30)  # 0–1, im niżej tym lepiej
    score = squeeze_depth * 50 + min((vol_ratio - 1.3) / 1.7, 1.0) * 50

    reasons = [
        f"BB width w {bb_width/pct10*10:.0f}. percentylu (60d)",
        f"wolumen {vol_ratio:.1f}x średniej przy ekspansji",
        "squeeze przed potencjalnym ruchem",
    ]
    return min(score, 100.0), reasons


# ---------------------------------------------------------------------------
# Główna analiza spółki
# ---------------------------------------------------------------------------

def analyze(symbol: str, benchmark_returns: pd.Series | None) -> dict | None:
    df = download(symbol)
    if df is None:
        return None

    # Filtr płynności
    df["VALUE"]  = df["Close"] * df["Volume"]
    df["LIQ20"]  = df["VALUE"].rolling(20).mean()
    if df.empty or pd.isna(df["LIQ20"].iloc[-1]):
        return None
    if float(df["LIQ20"].iloc[-1]) < LIQUIDITY_MIN_PLN:
        return None

    liq20 = float(df["LIQ20"].iloc[-1])
    price = float(df["Close"].iloc[-1])

    # Każdy sygnał zwraca (score 0-100, powody)
    s1, r1 = signal_volume_breakout(df.copy())
    s2, r2 = signal_momentum_surge(df.copy(), benchmark_returns)
    s3, r3 = signal_accumulation(df.copy())
    s4, r4 = signal_squeeze(df.copy())

    # Ważony score końcowy
    total_score = (
        s1 * WEIGHT_VOLUME_BREAKOUT / 100
        + s2 * WEIGHT_MOMENTUM_SURGE / 100
        + s3 * WEIGHT_ACCUMULATION  / 100
        + s4 * WEIGHT_SQUEEZE       / 100
    )

    if total_score < 8:
        return None

    # Zbieramy aktywne sygnały (score > 0)
    active_signals = []
    if s1 > 0:
        active_signals.append(("📈 Volume Breakout", s1, r1))
    if s2 > 0:
        active_signals.append(("🚀 Momentum Surge", s2, r2))
    if s3 > 0:
        active_signals.append(("🏦 Akumulacja (OBV)", s3, r3))
    if s4 > 0:
        active_signals.append(("🔥 Squeeze", s4, r4))

    if not active_signals:
        return None

    return {
        "symbol": symbol,
        "score": round(total_score, 1),
        "price": round(price, 2),
        "liq20": liq20,
        "signals": active_signals,
    }


# ---------------------------------------------------------------------------
# Budowanie wiadomości
# ---------------------------------------------------------------------------

def format_pln(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M PLN"
    return f"{value/1_000:.0f}K PLN"


def build_message(results: list[dict]) -> str:
    now_str = datetime.now(POLAND_TZ).strftime("%d.%m.%Y %H:%M")

    header = f"📊 GPW SMART MONEY BOT\n{now_str}\n{'─'*28}\n"

    if not results:
        return (
            header
            + "❌ BRAK SYGNAŁU\n\n"
            + "Żadna spółka z watchlisty\n"
            + "nie spełnia kryteriów swingowych.\n"
            + "Rynek bez wyraźnych setupów."
        )

    lines = [header]

    for i, r in enumerate(results[:TOP_N], 1):
        lines.append(f"{'🥇' if i==1 else f'{i}.'} {r['symbol']}  |  score: {r['score']:.0f}/100")
        lines.append(f"   Cena: {r['price']} PLN  |  Obrót śr.: {format_pln(r['liq20'])}")

        for sig_name, sig_score, sig_reasons in r["signals"]:
            lines.append(f"   {sig_name}  ({sig_score:.0f}pkt)")
            for reason in sig_reasons:
                lines.append(f"     · {reason}")

        lines.append("")

    lines.append("⚠️ To nie jest porada inwestycyjna.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wysyłanie na Telegram
# ---------------------------------------------------------------------------

def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Brak TELEGRAM_TOKEN lub TELEGRAM_CHAT_ID w env")

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

    print("Pobieranie benchmarku...")
    benchmark_returns = download_benchmark()

    results = []
    for symbol in WATCHLIST:
        print(f"  Analizuję {symbol}...")
        result = analyze(symbol, benchmark_returns)
        if result is not None:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    msg = build_message(results)
    print("\n--- WIADOMOŚĆ ---")
    print(msg)

    send_telegram(msg)
    print("Wysłano na Telegram.")


if __name__ == "__main__":
    run()
