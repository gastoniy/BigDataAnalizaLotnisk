import sqlite3
import pandas as pd
from datetime import datetime

def export_to_csv():
    # Połączenie z bazą
    # Używamy uri=True i trybu ro (read-only)
    conn = sqlite3.connect('file:baza_lotow.db?mode=ro', uri=True)
    
    print("Pobieranie danych z bazy...")
    
    # Modyfikacja zapytania SQL:
    # Pobieramy zarówno loty zrealizowane, jak i odwołane.
    # Status "Szum" (oraz potencjalne inne śmieci) jest automatycznie ignorowany.
    zapytanie_sql = '''
        SELECT * FROM loty_odloty 
        WHERE status LIKE 'Wystartował%' 
           OR status LIKE 'Odwołany%'
    '''
    
    df = pd.read_sql_query(zapytanie_sql, conn)
    
    if df.empty:
        print("Baza danych jest pusta lub nie znaleziono pasujących lotów!")
        return

    # Zapis do CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nazwa_pliku = f"dataset_loty_krakow_{timestamp}.csv"
    
    # Opcja index=False zapobiega tworzeniu dodatkowej, pustej kolumny z numeracją wierszy
    df.to_csv(nazwa_pliku, index=False, encoding='utf-8')
    
    print(f"Sukces! Wyeksportowano {len(df)} rekordów do pliku {nazwa_pliku}")
    
    # Zmieniono podgląd, aby pokazywał kolumnę status, ułatwiając weryfikację
    print("\nPróbka pobranych danych (z uwzględnieniem statusu):")
    print(df[['numer_lotu', 'kierunek', 'czas_planowany', 'status']].head())
    
    # Dodatkowe podsumowanie pokazujące, ile lotów wystartowało, a ile odwołano
    print("\nPodsumowanie wyeksportowanych statusów:")
    print(df['status'].value_counts().head())
    
    conn.close()

if __name__ == "__main__":
    export_to_csv()