import argparse
import re
import sqlite3
import sys
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialises the database and returns a connection."""
    conn = sqlite3.connect(db_path)
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
            ostatnia_aktualizacja DATETIME
        )
    ''')

    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_loty_odloty_data_numer 
        ON loty_odloty(data_lotu, numer_lotu)
    ''')
    
    conn.commit()
    return conn


def parse_time_to_datetime(time_str: str, date_str: str) -> datetime | None:
    """Combines a date (YYYY-MM-DD) with a time (HH:MM) into a datetime object."""
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def process_html(conn: sqlite3.Connection, html_path: str, date_display: str, date_db: str):
    """
    Parses the HTML file and upserts flight rows into the database.

    Args:
        conn:         Open SQLite connection.
        html_path:    Path to the saved HTML file.
        date_display: Date as shown in the HTML divider, e.g. "26/04/2026".
        date_db:      Date in DB format, e.g. "2026-04-26".
    """
    print(f"[parser] Opening: {html_path}", flush=True)

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"[parser] ERROR: File not found: '{html_path}'", file=sys.stderr)
        sys.exit(1)

    print("[parser] Parsing HTML...", flush=True)
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the divider matching our date
    target_divider = None
    for divider in soup.find_all("p", class_="table-departures-arrivals__divider"):
        if date_display in divider.text:
            target_divider = divider
            break

    if not target_divider:
        print(f"[parser] ERROR: No section found for date {date_display}.", file=sys.stderr)
        sys.exit(1)

    table = target_divider.find_next("table")
    if not table:
        print("[parser] ERROR: No table found under the date divider.", file=sys.stderr)
        sys.exit(1)

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    cursor = conn.cursor()
    processed = 0
    upserted = 0

    print(f"[parser] Inserting flights for {date_display}...", flush=True)

    for row in rows:
        cols = row.find_all(["td", "th"])

        if len(cols) < 4 or "Brak lotów" in cols[0].text:
            continue

        czas_planowany_str = cols[0].text.strip()
        kierunek          = cols[1].text.strip()
        numer_lotu        = cols[2].text.strip()
        status_text       = cols[3].text.strip()

        if not numer_lotu or "Numer lotu" in numer_lotu:
            continue

        processed += 1
        linia_lotnicza = numer_lotu.split(" ")[0] if " " in numer_lotu else "Nieznana"

        czas_planowany_dt  = parse_time_to_datetime(czas_planowany_str, date_db)
        czas_rzeczywisty_dt = None

        match = re.search(r"Wystartował\s+(\d{2}:\d{2})", status_text, re.IGNORECASE)
        if match:
            czas_rzeczywisty_dt = parse_time_to_datetime(match.group(1), date_db)
            # Guard against midnight crossover (planned 23:10, actual 01:05 next day)
            if czas_rzeczywisty_dt and czas_planowany_dt:
                if czas_rzeczywisty_dt < czas_planowany_dt - timedelta(hours=12):
                    czas_rzeczywisty_dt += timedelta(days=1)

        czas_planowany_db   = czas_planowany_dt.strftime("%Y-%m-%d %H:%M:%S")  if czas_planowany_dt   else None
        czas_rzeczywisty_db = czas_rzeczywisty_dt.strftime("%Y-%m-%d %H:%M:%S") if czas_rzeczywisty_dt else None

        cursor.execute('''
            INSERT INTO loty_odloty (
                data_lotu, numer_lotu, linia_lotnicza, kierunek,
                czas_planowany, czas_rzeczywisty, status, ostatnia_aktualizacja
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(data_lotu, numer_lotu)
            DO UPDATE SET
                czas_rzeczywisty    = excluded.czas_rzeczywisty,
                status              = excluded.status,
                ostatnia_aktualizacja = CURRENT_TIMESTAMP
        ''', (date_db, numer_lotu, linia_lotnicza, kierunek,
              czas_planowany_db, czas_rzeczywisty_db, status_text))

        upserted += 1

    conn.commit()
    print("-" * 40, flush=True)
    print(f"[parser] Flights found in table : {processed}", flush=True)
    print(f"[parser] Rows inserted/updated  : {upserted}", flush=True)


def parse_date_arg(value: str) -> datetime:
    """Accepts DD/MM/YYYY or YYYY-MM-DD."""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date: '{value}'. Use DD/MM/YYYY or YYYY-MM-DD."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Kraków Airport HTML file and store flights in SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parser.py --html odloty_krk_20260426.html --db baza_lotow.db --date 2026-04-26
  python parser.py --html odloty_krk_20260425.html --db baza_lotow.db --date 25/04/2026
        """,
    )
    parser.add_argument(
        "--html",
        required=True,
        help="Path to the scraped HTML file.",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite database file (created if missing).",
    )
    parser.add_argument(
        "--date",
        type=parse_date_arg,
        default=None,
        help="Date of the flights in the file (DD/MM/YYYY or YYYY-MM-DD). "
             "Defaults to yesterday.",
    )
    args = parser.parse_args()

    target_date  = args.date if args.date else (datetime.now() - timedelta(days=1))
    date_display = target_date.strftime("%d/%m/%Y")  # e.g. 26/04/2026
    date_db      = target_date.strftime("%Y-%m-%d")  # e.g. 2026-04-26

    print(f"[parser] Date   : {date_display}", flush=True)
    print(f"[parser] HTML   : {args.html}", flush=True)
    print(f"[parser] DB     : {args.db}", flush=True)

    conn = init_db(args.db)
    try:
        process_html(conn, args.html, date_display, date_db)
    finally:
        conn.close()


if __name__ == "__main__":
    main()