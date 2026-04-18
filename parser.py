from bs4 import BeautifulSoup
import sqlite3
import re
from datetime import datetime, timedelta
import os

PLIK_HTML = 'odloty_krk_20260417_2254.html'
NAZWA_BAZY = 'baza_lotow_test1.db'
SZUKANA_DATA_HTML = '17/04/2026' # Data wyświetlana na stronie
DATA_Z_PLIKU = '2026-04-17'

def init_db():
    """Inicjalizuje bazę danych z klauzulą UNIQUE dla UPSERT."""
    conn = sqlite3.connect(NAZWA_BAZY)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loty_odloty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_lotu DATE NOT NULL,
            numer_lotu TEXT NOT NULL,
            linia_lotnicza TEXT,
            kierunek TEXT,
            czas_planowany DATETIME,
            czas_rzeczywisty DATETIME,
            status TEXT,
            ostatnia_aktualizacja DATETIME,
            UNIQUE(data_lotu, numer_lotu)
        )
    ''')
    conn.commit()
    return conn

def parse_time_to_datetime(time_str, date_str):
    """Łączy podaną datę (YYYY-MM-DD) z godziną (HH:MM) w obiekt Datetime."""
    try:
        dt_str = f"{date_str} {time_str}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return None

def process_local_html(conn):
    print(f"Otwieranie pliku: {PLIK_HTML}...")
    
    try:
        with open(PLIK_HTML, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{PLIK_HTML}'.")
        return

    print("Parsowanie struktury HTML...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Szukamy nagłówka (dividera) z wybraną datą
    dividers = soup.find_all('p', class_='table-departures-arrivals__divider')
    target_divider = None
    
    for divider in dividers:
        if SZUKANA_DATA_HTML in divider.text:
            target_divider = divider
            break
            
    if not target_divider:
        print(f"Nie znaleziono sekcji dla daty {SZUKANA_DATA_HTML} w pliku HTML.")
        return

    # Znajdujemy pierwszą tabelę po tym nagłówku
    table = target_divider.find_next('table')
    if not table:
        print("Błąd: Nie znaleziono tabeli pod podaną datą.")
        return

    rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')
    
    cursor = conn.cursor()
    processed_count = 0
    added_or_updated = 0

    print(f"Rozpoczynam wprowadzanie lotów z dnia {SZUKANA_DATA_HTML}...")

    try:
        for row in rows:
            cols = row.find_all(['td', 'th'])
            
            # Jeśli wiersz ma kolumny i nie jest informacją o braku lotów
            if len(cols) >= 4 and "Brak lotów" not in cols[0].text:
                czas_planowany_str = cols[0].text.strip()
                kierunek = cols[1].text.strip()
                numer_lotu = cols[2].text.strip()
                status_text = cols[3].text.strip()
                
                if not numer_lotu or "Numer lotu" in numer_lotu:
                    continue
                    
                processed_count += 1
                
                # Mapowanie na zgodne ze strukturą 
                linia_lotnicza = numer_lotu.split(' ')[0] if ' ' in numer_lotu else "Nieznana"
                
                czas_planowany_dt = parse_time_to_datetime(czas_planowany_str, DATA_Z_PLIKU)
                czas_rzeczywisty_dt = None
                
                start_match = re.search(r'Wystartował\s+(\d{2}:\d{2})', status_text, re.IGNORECASE)
                if start_match:
                    czas_rzeczywisty_dt = parse_time_to_datetime(start_match.group(1), DATA_Z_PLIKU)
                    
                    # Zabezpieczenie przed przejściem przez północ (np. plan 23:10, start 01:05)
                    # Jeśli czas rzeczywisty jest wcześnie rano, a planowany był późno wieczorem, dodaj 1 dzień
                    if czas_rzeczywisty_dt and czas_planowany_dt:
                        if czas_rzeczywisty_dt < czas_planowany_dt - timedelta(hours=12):
                            czas_rzeczywisty_dt += timedelta(days=1)

                czas_planowany_db = czas_planowany_dt.strftime("%Y-%m-%d %H:%M:%S") if czas_planowany_dt else None
                czas_rzeczywisty_db = czas_rzeczywisty_dt.strftime("%Y-%m-%d %H:%M:%S") if czas_rzeczywisty_dt else None
                
                # Upsert do bazy
                cursor.execute('''
                    INSERT INTO loty_odloty (
                        data_lotu, numer_lotu, linia_lotnicza, kierunek, 
                        czas_planowany, czas_rzeczywisty, status, ostatnia_aktualizacja
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(data_lotu, numer_lotu) 
                    DO UPDATE SET 
                        czas_rzeczywisty = excluded.czas_rzeczywisty,
                        status = excluded.status,
                        ostatnia_aktualizacja = CURRENT_TIMESTAMP
                ''', (DATA_Z_PLIKU, numer_lotu, linia_lotnicza, kierunek, 
                    czas_planowany_db, czas_rzeczywisty_db, status_text))
                
                added_or_updated += 1

        conn.commit()
        print("-" * 40)
        print("ZAKOŃCZONO PARSOWANIE")
        print(f"Zidentyfikowano lotów w tabeli: {processed_count}")
        print(f"Pomyślnie zaktualizowano w bazie: {added_or_updated}")

        # --- SEKCJA CZYSZCZENIA ---
        if added_or_updated > 0:
            os.remove(PLIK_HTML)
            print(f"PLIK USUNIĘTY: {PLIK_HTML} (dane są już bezpieczne w bazie)")
        else:
            print("Uwaga: Nie znaleziono danych, plik pozostawiono do inspekcji.")

    except Exception as e:
        conn.rollback() # W razie błędu wycofujemy zmiany w bazie
        print(f"BŁĄD krytyczny: {e}. Plik {PLIK_HTML} NIE został usunięty.")

if __name__ == "__main__":
    conn = init_db()
    process_local_html(conn)
    conn.close()