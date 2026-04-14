# Opis projektu i inne informacje
Patrz plik PDF udostępniony na GitHub
# Algorytm Pobierania Danych ze strony lotniska w Krakowie

1. Skrypt - `page_scraper.py` codziennie o wybranej godzinie (np. 19-20) pobiera ze strony lotniska uproszczoną stronę `html`, która zawiera 2 tabeli:
	- pierwsza: WSZYSTKIE loty za dzień wczorajszy z czasem odlotu
	- druga: wszystkie loty za dzień dzisiejszy z czasem odlotu dla prawie wszystkich lotów
2. Skrypt - `parser.py` jest używany do wyciągnięcia niezbędnych danych z pliku `html` i zapisania tych danych do SQLite3
3. (Optional) Skrypt - `transform.py` jest używany do umieszczenia danych z bd do pliku `csv` np. dla analizy wizualnej (przydatne dla troubleshooting i podobnego)

Po wykonaniu wszystkich skryptów uzyskuje się taki przykładowy plik `csv`:
```cs
id,data_lotu,numer_lotu,linia_lotnicza,kierunek,czas_planowany,czas_rzeczywisty,status,ostatnia_aktualizacja
1,2026-04-13,LO 3910,LO,WARSZAWA (WAW),2026-04-13 05:40:00,2026-04-13 05:49:00,Wystartował 05:49,2026-04-14 17:49:32
2,2026-04-13,W6 2035,W6,BERGEN (BGO),2026-04-13 05:40:00,2026-04-13 05:47:00,Wystartował 05:47,2026-04-14 17:49:32
3,2026-04-13,FR 6216,FR,OSLO TORP (TRF),2026-04-13 05:50:00,2026-04-13 05:53:00,Wystartował 05:53,2026-04-14 17:49:32
4,2026-04-13,FR 1902,FR,DUBLIN (DUB),2026-04-13 06:00:00,2026-04-13 06:55:00,Wystartował 06:55,2026-04-14 17:49:32
5,2026-04-13,FR 5412,FR,MARSYLIA (MRS),2026-04-13 06:00:00,2026-04-13 06:14:00,Wystartował 06:14,2026-04-14 17:49:32
6,2026-04-13,LH 1627,LH,MONACHIUM (MUC),2026-04-13 06:00:00,,Odwołany,2026-04-14 17:49:32
7,2026-04-13,W6 2007,W6,LONDYN GATWICK (LGW),2026-04-13 06:00:00,2026-04-13 06:13:00,Wystartował 06:13,2026-04-14 17:49:32
8,2026-04-13,W6 2047,W6,BARCELONA (BCN),2026-04-13 06:00:00,2026-04-13 06:21:00,Wystartował 06:21,2026-04-14 17:49:32
9,2026-04-13,LH 1371,LH,FRANKFURT (FRA),2026-04-13 06:10:00,,Odwołany,2026-04-14 17:49:32
...