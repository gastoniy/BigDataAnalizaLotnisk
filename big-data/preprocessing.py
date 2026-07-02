import os
import sqlite3

import numpy as np
import pandas as pd
import airportsdata

# ścieżki domyślne
_TU = os.path.dirname(os.path.abspath(__file__))
DOMYSLNA_BAZA = os.path.join(_TU, "..", "data", "baza_lotow.db")
DOMYSLNE_WYJSCIE = os.path.join(_TU, "dataset_eda_ready.csv")

# współrzędne lotniska Kraków (EPKK)
KRK_LAT = np.radians(50.0777)
KRK_LON = np.radians(19.7848)

# mapowanie dni tygodnia w jednym miejscu (0 = poniedziałek)
DNI_PL = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
PORY_DNIA = ["Rano", "Popołudnie", "Wieczór", "Noc"]
KATEGORIE_OPOZNIENIA = ["Wcześniej/punkt.", "1-15", "15-30", "30-60", "60+"]

# baza danych lotnisk wczytywana raz (klucz = kod IATA)
_AERO = airportsdata.load("IATA")


def wczytaj_z_bazy(db_path=DOMYSLNA_BAZA):
    """Wczytuje surowe loty z SQLite (te same statusy co transform.py)."""
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM loty_odloty "
            "WHERE status LIKE 'Wystartował%' OR status LIKE 'Odwołany%'",
            con,
        )
    finally:
        con.close()
    return df


def _kod_iata(kierunek):
    """Wyciąga kod IATA z napisu typu 'MONACHIUM (MUC)'."""
    try:
        return kierunek.split("(")[-1].replace(")", "").strip().upper()
    except Exception:
        return None


def _wspolrzedne(kod):
    inf = _AERO.get(kod) if kod else None
    if inf:
        return inf["lat"], inf["lon"]
    return np.nan, np.nan


def _haversine_z_krakowa(lat_deg, lon_deg):
    """Odległość wielkokołowa (km) od KRK dla wektorów współrzędnych w stopniach."""
    lat2 = np.radians(lat_deg)
    lon2 = np.radians(lon_deg)
    dlat = lat2 - KRK_LAT
    dlon = lon2 - KRK_LON
    a = np.sin(dlat / 2) ** 2 + np.cos(KRK_LAT) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _przypisz_pore_dnia(godzina):
    if pd.isna(godzina):
        return np.nan
    if 5 <= godzina < 12:
        return "Rano"
    if 12 <= godzina < 17:
        return "Popołudnie"
    if 17 <= godzina < 22:
        return "Wieczór"
    return "Noc"


def preprocess(df, prog_opoznienia=15):
    """
    Przekształca surowy DataFrame lotów w zbiór gotowy pod EDA.

    Args:
        df: surowe dane.
        prog_opoznienia: próg w minutach dla flagi czy_opozniony.

    Returns:
        DataFrame z cechami analitycznymi.
    """
    df = df.copy()

    # Deduplikacja — zachowujemy najświeższy wpis dla danego lotu
    if "ostatnia_aktualizacja" in df.columns:
        df = (
            df.sort_values("ostatnia_aktualizacja")
            .drop_duplicates(subset=["data_lotu", "numer_lotu", "czas_planowany"], keep="last")
            .reset_index(drop=True)
        )

    # Flaga odwołania - przpisanie do odzielnej "klasy"
    df["czy_odwolany"] = (
        df["status"].astype(str).str.lower().str.contains("odwołany").astype(int)
    )

    # Parsowanie dat
    df["czas_planowany"] = pd.to_datetime(df["czas_planowany"], errors="coerce")
    df["czas_rzeczywisty"] = pd.to_datetime(df["czas_rzeczywisty"], errors="coerce")
    df["data_lotu"] = pd.to_datetime(df["data_lotu"], errors="coerce")

    # Opóźnienie z korektą przejścia przez północ
    delta = (df["czas_rzeczywisty"] - df["czas_planowany"]).dt.total_seconds() / 60.0
    delta = delta.mask(delta < -720, delta + 1440)
    df["opoznienie_surowe"] = delta                       # ze znakiem — do rozkładów
    df["opoznienie_minuty"] = delta.clip(lower=0)         # wyzerowane — metryki operacyjne

    # Cechy czasowe
    df["godzina_planowana"] = df["czas_planowany"].dt.hour
    df["dzien_tygodnia_num"] = df["czas_planowany"].dt.dayofweek
    df["dzien_tygodnia"] = pd.Categorical.from_codes(
        df["dzien_tygodnia_num"].fillna(-1).astype(int),
        categories=DNI_PL,
        ordered=True,
    )
    df["czy_weekend"] = df["dzien_tygodnia_num"].isin([5, 6]).astype(int)
    df["pora_dnia"] = pd.Categorical(
        df["godzina_planowana"].apply(_przypisz_pore_dnia),
        categories=PORY_DNIA,
        ordered=True,
    )
    df["miesiac"] = df["czas_planowany"].dt.month
    df["numer_tygodnia"] = df["czas_planowany"].dt.isocalendar().week.astype("Int64")

    # Standaryzacja kategorii tekstowych
    df["linia_lotnicza"] = df["linia_lotnicza"].astype(str).str.strip().str.upper()
    df["kierunek"] = df["kierunek"].astype(str).str.strip().str.upper()

    # Geografia: kod IATA, współrzędne, dystans od KRK
    df["kod_kierunku"] = df["kierunek"].apply(_kod_iata)
    wsp = df["kod_kierunku"].apply(_wspolrzedne)
    df[["lat", "lon"]] = pd.DataFrame(wsp.tolist(), index=df.index)
    df["dystans_km"] = _haversine_z_krakowa(df["lat"], df["lon"])

    # Kategoria opóźnienia (kubełki) + flaga opóźnienia operacyjnego
    df["kategoria_opoznienia"] = pd.cut(
        df["opoznienie_surowe"],
        bins=[-np.inf, 0, 15, 30, 60, np.inf],
        labels=KATEGORIE_OPOZNIENIA,
    )
    df["czy_opozniony"] = np.where(
        df["opoznienie_minuty"].notna(),
        (df["opoznienie_minuty"] > prog_opoznienia).astype(float),
        np.nan,
    )

    # Detekcja anomalii — reguła IQR
    df["czy_anomalia"] = 0
    zr = df["opoznienie_minuty"].dropna()
    if len(zr) > 0:
        q1, q3 = zr.quantile([0.25, 0.75])
        prog_anom = q3 + 1.5 * (q3 - q1)
        df["czy_anomalia"] = (df["opoznienie_minuty"] > prog_anom).fillna(False).astype(int)

    # Usunięcie kolumn surowych/technicznych niepotrzebnych w EDA
    do_usuniecia = [
        "id", "ostatnia_aktualizacja", "numer_lotu",
        "czas_planowany", "czas_rzeczywisty", "status", "lat", "lon",
    ]
    df = df.drop(columns=[c for c in do_usuniecia if c in df.columns])

    return df


def run(db_path=DOMYSLNA_BAZA, csv_path=None, wyjscie=DOMYSLNE_WYJSCIE, prog_opoznienia=15):
    """Wczytuje dane (z bazy lub CSV), przetwarza i zapisuje zbiór gotowy pod EDA."""
    if csv_path:
        print(f"Wczytuję dane z CSV: {csv_path}")
        surowe = pd.read_csv(csv_path)
    else:
        print(f"Wczytuję dane z bazy: {db_path}")
        surowe = wczytaj_z_bazy(db_path)
    print(f"  rekordów surowych: {len(surowe)}")

    ready = preprocess(surowe, prog_opoznienia=prog_opoznienia)
    print(f"  rekordów po przetworzeniu: {len(ready)} "
          f"(odwołane: {int(ready['czy_odwolany'].sum())}, "
          f"anomalie: {int(ready['czy_anomalia'].sum())})")

    ready.to_csv(wyjscie, index=False, encoding="utf-8")
    print(f"Zapisano: {wyjscie}")
    return ready


if __name__ == "__main__":
    run()
