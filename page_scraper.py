import time
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

def scrape_krakow_airport():
    # Obliczenie wczorajszej daty
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%d/%m/%Y")
    
    # Obliczenie dzisiejszej daty
    today = datetime.now()
    today_str = today.strftime("%d/%m/%Y")
    
    # Format pliku do zapisu
    filename = f"odloty_krk_{yesterday.strftime('%Y%m%d_%H%M')}.html"

    print(f"Wyszukuję loty dla daty: {yesterday_str}...")

    # Przygotowanie kodu HTML z dividerem (potrzebne dla następnego etapu)
    divider_html = f"""
    <p class="table-departures-arrivals__divider" style="margin-top: 20px;">
        <svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="23.709" height="23.86" viewBox="0 0 23.709 23.86">
            <path d="M450.576,645.235h-3.6v-.465c0-.576-.208-1.183-.375-1.183h-.976c-.187,0-.374.607-.374,1.182v.466H434.132v-.465c0-.576-.209-1.183-.375-1.183h-.976c-.187,0-.374.607-.374,1.182v.466h-3.814a.847.847,0,0,0-.863.862v20.488a.847.847,0,0,0,.863.862h21.982a.847.847,0,0,0,.863-.862V646.1A.865.865,0,0,0,450.576,645.235Zm-17.306,18.53A1.658,1.658,0,0,1,431.6,662.1a1.7,1.7,0,0,1,1.669-1.637,1.675,1.675,0,0,1,1.669,1.637A1.64,1.64,0,0,1,433.27,663.765Zm0-6.12a1.659,1.659,0,0,1-1.669-1.67,1.7,1.7,0,0,1,1.669-1.636,1.675,1.675,0,0,1,1.669,1.636A1.64,1.64,0,0,1,433.27,657.645Zm6.422,6.12a1.658,1.658,0,0,1-1.669-1.669,1.7,1.7,0,0,1,1.669-1.637,1.675,1.675,0,0,1,1.669,1.637A1.64,1.64,0,0,1,439.692,663.765Zm0-6.12a1.659,1.659,0,0,1-1.669-1.67,1.7,1.7,0,0,1,1.669-1.636,1.675,1.675,0,0,1,1.669,1.636A1.64,1.64,0,0,1,439.692,657.645Zm6.421,6.12a1.658,1.658,0,0,1-1.669-1.669,1.7,1.7,0,0,1,1.669-1.637,1.675,1.675,0,0,1,1.669,1.637A1.64,1.64,0,0,1,446.113,663.765Zm0-6.12a1.659,1.659,0,0,1-1.669-1.67,1.7,1.7,0,0,1,1.669-1.636,1.675,1.675,0,0,1,1.669,1.636A1.64,1.64,0,0,1,446.113,657.645Zm3.6-7.64v.345H429.457v-3.391h2.95v.245c0,.576.185,1.28.374,1.28h.976c.166,0,.374-.7.375-1.28v-.245H445.25v.245c0,.576.185,1.28.374,1.28h.976c.166,0,.374-.7.375-1.28v-.245h2.739Z" transform="translate(-427.731 -643.587)" fill="#657baa"></path>
        </svg>
        &nbsp; {yesterday_str}
    </p>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Przejście na stronę odlotów
        page.goto("https://krakowairport.pl/pl/pasazer/loty/polaczenia/odloty")
        page.wait_for_selector('#flight_date')

        # Akceptacja ciasteczek (jeśli się pojawią)
        try:
            page.click("button:has-text('Akceptuję')", timeout=3000)
        except:
            pass 

        # Ustawienie daty wczorajszej w ukrytym polu
        page.evaluate(f"document.getElementById('flight_date').value = '{yesterday_str}'")
        page.evaluate("document.getElementById('flight_date').dispatchEvent(new Event('input'))")
        page.evaluate("document.getElementById('flight_date').dispatchEvent(new Event('change'))")

        # Ustawienie godziny na 00:30
        page.evaluate("document.getElementById('flight_time').value = '00:30'")
        page.evaluate("document.getElementById('flight_time').dispatchEvent(new Event('change'))")

        # Kliknięcie "Pokaż"
        page.click("button.btn-primary:has-text('Pokaż')")

        # Oczekiwanie na przeładowanie i wyrenderowanie danych
        page.wait_for_load_state("networkidle")
        time.sleep(2) 

        # Wstrzyknięcie dividera przez JavaScript
        js_inject_code = """(htmlStr) => {
            const firstTableWrap = document.querySelector('.table-responsive.table-departures-arrivals');
            if (firstTableWrap) {
                firstTableWrap.insertAdjacentHTML('beforebegin', htmlStr);
            }
        }"""
        page.evaluate(js_inject_code, divider_html)

        # Pobranie całego wygenerowanego obszaru
        table_locator = page.locator(".departures_table__table-inner-wrap")
        table_html = table_locator.inner_html()

        # Zapis z odrobiną stylów CSS, żeby tabela wyglądała czytelnie bez stylów z lotniska
        with open(filename, "w", encoding="utf-8") as file:
            file.write("""<!DOCTYPE html>
            <html>
            <head>
                <meta charset='utf-8'>
                <title>Odloty</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .table-departures-arrivals__divider { font-weight: bold; color: #657baa; font-size: 18px; display: flex; align-items: center; }
                    .table-departures-arrivals__divider svg { margin-right: 8px; }
                    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                    th, td { border-bottom: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #f4f4f4; }
                </style>
            </head>
            <body>\n""")
            file.write(table_html) # Tak
            file.write("\n</body></html>")

        print(f"Dodano divider '{yesterday_str}' i zapisano dane do pliku: {filename}")
        
        browser.close()

if __name__ == "__main__":
    scrape_krakow_airport()