import sqlite3
import pandas as pd
from datetime import datetime

def export_to_csv():
    # Połączenie z bazą
    # Używamy uri=True i trybu ro (read-only)
    conn = sqlite3.connect('file:baza_lotow.db?mode=ro', uri=True)
    
    print("Pobieranie danych z bazy...")
    
    # Pobranie całej tabeli odlotów
    df = pd.read_sql_query('SELECT * FROM loty_odloty WHERE status LIKE "Wystartował%"', conn)
    
    if df.empty:
        print("Baza danych jest pusta!")
        return

    # Zapis do CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nazwa_pliku = f"dataset_loty_krakow_{timestamp}.csv"
    
    # Opcja index=False zapobiega tworzeniu dodatkowej, pustej kolumny z numeracją wierszy
    df.to_csv(nazwa_pliku, index=False, encoding='utf-8')
    
    print(f"Sukces! Wyeksportowano {len(df)} rekordów do pliku {nazwa_pliku}")
    print("\nPróbka pobranych danych:")
    print(df[['numer_lotu', 'kierunek', 'czas_planowany', 'czas_rzeczywisty']].head())
    
    conn.close()

if __name__ == "__main__":
    export_to_csv()