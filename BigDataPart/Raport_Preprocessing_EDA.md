# Raport: Preprocessing i Eksploracyjna Analiza Danych (EDA)

**Projekt:** Analiza punktualności odlotów z lotniska Kraków-Balice (KRK)
**Zakres raportu:** przygotowanie danych do analizy (preprocessing) oraz eksploracyjna analiza danych (EDA)
**Moduły:** `BigDataPart/preprocessing.py`, `BigDataPart/eda.py`

---

## 1. Wprowadzenie i cel

Celem tej części projektu jest przekształcenie surowych, automatycznie pozyskanych danych o odlotach w zbiór analityczny oraz przeprowadzenie eksploracyjnej analizy danych. EDA ma odpowiedzieć na pytania badawcze projektu: **które czynniki wpływają na opóźnienia** (linia lotnicza, pora dnia, godzina, kierunek, odległość) oraz **jak punktualność zmienia się w czasie**. Analiza stanowi również podstawę pod dalszą część predykcyjną (model klasyfikacji opóźnień).

---

## 2. Źródło i charakterystyka danych

Dane pochodzą z bazy SQLite `baza_lotow.db` (tabela `loty_odloty`), zasilanej codziennie przez potok web-scrapingu tablicy odlotów portu lotniczego. Surowy rekord opisuje pojedynczy odlot:

| Kolumna | Typ | Opis |
|---|---|---|
| `id` | INTEGER | klucz techniczny |
| `data_lotu` | DATE | data lotu |
| `numer_lotu` | TEXT | numer rejsu (np. `LO 3910`) |
| `linia_lotnicza` | TEXT | kod przewoźnika (np. `FR`, `W6`, `LO`) |
| `kierunek` | TEXT | lotnisko docelowe z kodem IATA, np. `MONACHIUM (MUC)` |
| `czas_planowany` | DATETIME | rozkładowy czas wylotu |
| `czas_rzeczywisty` | DATETIME | faktyczny czas wylotu (NULL dla odwołanych) |
| `status` | TEXT | status (np. `Wystartował 06:55`, `Odwołany`) |
| `ostatnia_aktualizacja` | DATETIME | znacznik czasu zapisu/aktualizacji |

**Filtr istotności:** do analizy trafiają wyłącznie loty o statusie `Wystartował%` lub `Odwołany%`; pozostałe wartości to szum z tablicy odlotów (statusy pośrednie typu „Odprawa", „Gate zamknięty").

**Charakterystyka zbioru analitycznego (po przetworzeniu):**
- **6 863 rekordów**, zakres **13.04.2026 – 02.06.2026** (51 dni),
- **40 linii lotniczych**, **131 kierunków**,
- **94 loty odwołane (1,37%)** — silna nierównowaga klas,
- dominacja przewoźnika **FR (Ryanair, ~50% ruchu)**,
- część dni ma dane syntetyczne (uzupełnienie braków: 18, 19, 21–23.04.2026).

---

## 3. Preprocessing (`preprocessing.py`)

Moduł wczytuje dane bezpośrednio z bazy (ten sam filtr statusów co potok produkcyjny) i przekształca je w zbiór gotowy pod EDA (`dataset_eda_ready.csv`). Poszczególne kroki:

### 3.1. Deduplikacja
Scraper działa w trybie INSERT/UPDATE i uruchamiany jest cyklicznie, więc ten sam lot może pojawić się w bazie wielokrotnie. Rekordy są deduplikowane po naturalnym kluczu `(data_lotu, numer_lotu, czas_planowany)`, z zachowaniem **najświeższego** wpisu (sortowanie po `ostatnia_aktualizacja`, `keep='last'`). Zapobiega to zawyżaniu wolumenów i przesuwaniu średnich.

### 3.2. Flaga odwołania
Zamiast usuwać odwołane loty, tworzymy zmienną binarną `czy_odwolany` (1, gdy status zawiera „odwołany"). Dzięki temu odwołania pozostają przedmiotem analizy (osobny wymiar jakości operacyjnej), a nie są tracone.

### 3.3. Obliczenie opóźnienia z korektą przejścia przez północ
Opóźnienie liczone jest jako różnica `czas_rzeczywisty − czas_planowany` w minutach. **Kluczowa poprawka:** lot planowany np. na 23:40, startujący po północy, dawałby pozornie ogromne opóźnienie ujemne (różnica dat). Dlatego wartości `< −720 min` korygujemy o `+1440 min` (jedna doba). Zachowujemy dwie wersje zmiennej:
- `opoznienie_surowe` — wartość ze znakiem (umożliwia analizę odlotów przed czasem i kształtu rozkładu),
- `opoznienie_minuty` — wersja z odcięciem wartości ujemnych do 0 (metryka operacyjna; wcześniejszy odlot to z punktu widzenia pasażera „punktualność").

### 3.4. Cechy czasowe
Z `czas_planowany` wyprowadzamy: `godzina_planowana` (0–23), `dzien_tygodnia_num` (0–6) oraz `dzien_tygodnia` (uporządkowana kategoria polska — bez zależności od locale systemu), `czy_weekend`, `pora_dnia` (Rano 5–12, Popołudnie 12–17, Wieczór 17–22, Noc 22–5), `miesiac`, `numer_tygodnia`.

### 3.5. Standaryzacja kategorii i geografia
Kody linii i nazwy kierunków normalizowane są do wielkich liter bez białych znaków (eliminacja duplikatów w wykresach). Z `kierunek` wyciągamy kod IATA, a przez bibliotekę `airportsdata` pobieramy współrzędne lotniska docelowego i liczymy `dystans_km` — **odległość wielkokołową (Haversine)** od Krakowa. Pozwala to badać wpływ długości trasy na opóźnienie.

### 3.6. Kategoryzacja i etykieta opóźnienia
- `kategoria_opoznienia` — kubełki: `Wcześniej/punkt. (≤0)`, `1–15`, `15–30`, `30–60`, `60+`,
- `czy_opozniony` — etykieta biznesowa: 1, gdy opóźnienie **> 15 min** (dla odwołanych: brak wartości).

### 3.7. Detekcja anomalii (reguła IQR)
Wartości skrajne (np. opóźnienia kilkugodzinne z powodu pogody/awarii) oznaczamy flagą `czy_anomalia` regułą **IQR** (`> Q3 + 1,5·IQR`). W odróżnieniu od metod losowych (np. Isolation Forest) jest to próg **deterministyczny i interpretowalny**. W zbiorze flaga objęła **515 lotów (7,5%)**. Anomalie są wyłączane z wykresów ukazujących „typowe" zachowanie operacyjne, ale zachowane w zbiorze.

**Zbiór wynikowy** zawiera kolumny analityczne i jest zapisywany do `dataset_eda_ready.csv`. Uwaga: uporządkowanie kategorii (`pora_dnia`, `dzien_tygodnia`, `kategoria_opoznienia`) jest tracone przy zapisie CSV i **przywracane przy wczytaniu** w `eda.py`.

---

## 4. Eksploracyjna Analiza Danych (`eda.py`)

EDA generuje **15 wykresów** (`wykresy/01–15_*.png`). Przyjęte zasady projektowe:
- **mediana zamiast średniej** dla opóźnień — rozkład jest silnie prawoskośny (pojedyncze ekstrema zawyżają średnią),
- **odsetki (%) zamiast liczb bezwzględnych** przy porównaniach grup o różnej wielkości,
- **próg minimalnej liczby lotów (≥30) i top-N** dla linii/kierunków — eliminacja mylących skoków przy małych próbkach,
- `opoznienie_surowe` do rozkładów, `opoznienie_minuty` do metryk operacyjnych.

Poniżej szczegółowy opis każdego wykresu: **co pokazuje**, **dlaczego dany typ**, **jakiej wiedzy o zbiorze dostarcza**.

### Wykres 01 — Rozkład opóźnień (histogram + KDE)
- **Co pokazuje:** rozkład `opoznienie_surowe` lotów zrealizowanych, z nałożoną estymacją gęstości (KDE), liniami punktualności (0 min) i progu opóźnienia (15 min). Oś X przycięta do (−60, 180) min dla czytelności.
- **Dlaczego histogram + KDE:** to podstawowy wykres do oceny **kształtu, środka i rozproszenia** jednej zmiennej ciągłej; KDE wygładza i uwypukla skośność.
- **Informacje o zbiorze:** rozkład jest **prawoskośny** — większość lotów ma niewielkie opóźnienie skupione wokół mediany **10 min** (średnia 14,9 min — różnica potwierdza skośność), z długim ogonem do kilkuset minut. Ok. **10,3%** lotów odlatuje przed czasem lub o czasie (`≤0`). Uzasadnia to wybór mediany jako miary tendencji centralnej w dalszych wykresach.

### Wykres 02 — Dystrybuanta empiryczna (ECDF)
- **Co pokazuje:** skumulowany odsetek lotów o opóźnieniu ≤ x.
- **Dlaczego ECDF:** pozwala **odczytać dowolny percentyl** bez arbitralnego binowania i precyzyjnie oszacować udziały (np. „jaki % lotów mieści się w progu 15 min").
- **Informacje o zbiorze:** ~**67%** lotów ma opóźnienie ≤ 15 min (a więc **32,7%** przekracza próg), 90. percentyl to ~36 min, 95. ~51 min, 99. ~102 min. Ogon jest „cienki", ale istotny operacyjnie.

### Wykres 03 — Mediana opóźnienia wg linii (top-15, 95% CI)
- **Co pokazuje:** poziome słupki mediany opóźnienia dla 15 najczęściej operujących linii, posortowane malejąco, z 95% przedziałem ufności.
- **Dlaczego słupki poziome + mediana + CI:** poziomy układ mieści długie etykiety; mediana jest odporna na ekstrema; CI pokazuje **niepewność** estymacji (czy różnice są wiarygodne).
- **Informacje o zbiorze:** istnieje **wyraźne zróżnicowanie operacyjne między przewoźnikami** — od ~4–6 min mediany (DY/OS/D8) po ~20 min (LX/SWISS). Wskazuje to linię lotniczą jako istotny czynnik opóźnień.

### Wykres 04 — Odsetek lotów opóźnionych (%) wg linii (top-15)
- **Co pokazuje:** udział lotów opóźnionych (>15 min) na linię, z adnotacją liczby lotów `n`.
- **Dlaczego odsetek (nie średnia):** przy bardzo różnych wolumenach (FR ~50% ruchu vs niszowi przewoźnicy) **proporcja opóźnień jest uczciwszą** miarą porównawczą niż średnie minuty; adnotacja `n` chroni przed nadinterpretacją małych prób.
- **Informacje o zbiorze:** rozpiętość jest duża — od ~**8–15%** (DY, OS, D8) do ~**60–64%** (LX, SN, EN). Komplementarnie do wykresu 03 potwierdza, że jakość operacyjna silnie zależy od przewoźnika.

### Wykres 05 — Rozkład opóźnień wg linii (boxplot, top-15)
- **Co pokazuje:** pełen rozkład `opoznienie_surowe` na linię (mediana, kwartyle, wąsy, wartości odstające), z linią 0 min. **Anomalie są celowo widoczne** (`showfliers`).
- **Dlaczego boxplot:** uzupełnia wykresy 03/04 o **rozrzut i stabilność** — dwie linie o tej samej medianie mogą różnić się przewidywalnością. Wycinanie anomalii w boxplocie byłoby błędem (od tego są fliery).
- **Informacje o zbiorze:** widać różnice w **rozrzucie** — niektórzy przewoźnicy są „stabilni" (wąskie pudełka), inni mają szerokie rozkłady i liczne ekstrema. Część rejsów odlatuje przed czasem (wartości poniżej 0).

### Wykres 06 — Odsetek opóźnionych wg pory dnia
- **Co pokazuje:** odsetek lotów opóźnionych w czterech porach dnia z 95% CI i adnotacją `n`.
- **Dlaczego pointplot z CI (zamiast countplotu liczb):** liczby bezwzględne mylą — pora o największym ruchu wygląda „najgorzej". **Proporcja** odpowiada na pytanie o ryzyko opóźnienia.
- **Informacje o zbiorze:** wyraźny wzorzec narastania w ciągu doby: **Rano 22,3% → Popołudnie 37,0% → Wieczór 39,9% → Noc 50,5%**. To pierwszy sygnał **efektu kaskadowego** (opóźnienia kumulują się wraz z dniem).

### Wykres 07 — Struktura kategorii opóźnień wg pory dnia (stacked 100%)
- **Co pokazuje:** procentowy udział kategorii (`≤0`, `1–15`, `15–30`, `30–60`, `60+`) w obrębie każdej pory dnia.
- **Dlaczego stacked 100%:** pokazuje **kompozycję** punktualności, nie tylko binarny podział — widać, czy rośnie udział umiarkowanych czy ciężkich opóźnień.
- **Informacje o zbiorze:** w skali całości dominuje kategoria `1–15` (~57%), a `60+` to ~3,4%. W porach późniejszych rośnie udział kategorii cięższych (`30–60`, `60+`) kosztem punktualnych — potwierdza pogłębianie się opóźnień wieczorem i nocą.

### Wykres 08 — Efekt kaskadowości: opóźnienia wg godziny (boxplot + mediana)
- **Co pokazuje:** rozkład opóźnień dla każdej z 24 godzin planowanego wylotu, z nałożoną linią mediany.
- **Dlaczego boxplot per godzina (zamiast scatter):** ~6,8 tys. punktów na dyskretnej osi godzinowej tworzy nieczytelne pasy; boxplot porządkuje obraz i pokazuje trend oraz rozrzut jednocześnie.
- **Informacje o zbiorze:** mediana rośnie z **~6 min (5–8 rano)** do **~15–16 min (21–23)** — kwantyfikacja efektu kaskadowego. Korelacja Spearmana godzina↔opóźnienie wynosi **ρ ≈ 0,21 (p < 10⁻⁶⁰)**: zależność słaba do umiarkowanej, ale wysoce istotna.

### Wykres 09 — Obciążenie ruchem vs mediana opóźnienia (dwie osie Y)
- **Co pokazuje:** liczba lotów na godzinę (słupki) zestawiona z medianą opóźnienia (linia).
- **Dlaczego dwie osie:** pozwala sprawdzić hipotezę „więcej ruchu = większe opóźnienia" przez bezpośrednie porównanie dwóch metryk w jednej skali czasu.
- **Informacje o zbiorze:** opóźnienie **nie rośnie wprost proporcjonalnie do wolumenu** — szczyt ruchu przypada na środek dnia, lecz mediana opóźnień najwyższa jest wieczorem/nocą. Sugeruje to, że kaskada jest funkcją **akumulacji w ciągu dnia**, a nie wyłącznie chwilowego natężenia.

### Wykres 10 — Mapa cieplna: dzień tygodnia × godzina (mediana)
- **Co pokazuje:** mediana opóźnienia w siatce dzień (7) × godzina (24); komórki z <5 lotów są maskowane. Anomalie wyłączone.
- **Dlaczego heatmapa:** ujawnia **dwuwymiarowe wzorce**, których nie widać w analizie jednej zmiennej (np. konkretne „gorące" okna czasowe).
- **Informacje o zbiorze:** potwierdza wzorzec godzinowy w każdym dniu (chłodniejsze poranki, cieplejsze wieczory) oraz pozwala wychwycić pojedyncze „gorące" sloty (np. wieczory wybranych dni). Brak silnego, jednolitego efektu „złego dnia tygodnia".

### Wykres 11 — Top 10 kierunków wg odsetka odwołań (min. 30 lotów)
- **Co pokazuje:** kierunki o najwyższym udziale odwołań, z adnotacją `n`.
- **Dlaczego współczynnik (nie liczba):** liczba bezwzględna faworyzuje kierunki o dużym ruchu; **odsetek** wskazuje trasy realnie najbardziej zawodne (próg `n≥30` chroni przed artefaktami).
- **Informacje o zbiorze:** odwołania koncentrują się na wybranych trasach (głównie europejskie huby z dużym udziałem LH) — sygnał, że odwołania są **specyficzne dla przewoźnika/trasy**, nie rozłożone losowo.

### Wykres 12 — Odsetek odwołań wg linii (min. 30 lotów)
- **Co pokazuje:** udział odwołanych lotów na przewoźnika.
- **Dlaczego osobny wymiar:** odwołanie to inna kategoria zdarzenia niż opóźnienie; wymaga oddzielnej oceny jakości operacyjnej.
- **Informacje o zbiorze:** dominującym wnioskiem jest **bardzo wysoki odsetek odwołań LH (Lufthansa) ≈ 12,5%**, wielokrotnie powyżej pozostałych (~2–4%). To wyraźny, pojedynczy czynnik odpowiadający za większość odwołań w zbiorze.

### Wykres 13 — Top 15 kierunków wg mediany opóźnienia (min. 30 lotów)
- **Co pokazuje:** trasy o najwyższej medianie opóźnienia (loty zrealizowane, bez anomalii), z adnotacją `n`.
- **Dlaczego mediana + próg n:** identyfikuje kierunki systematycznie „trudne" operacyjnie, odporne na pojedyncze ekstrema.
- **Informacje o zbiorze:** ujawnia geograficzny wymiar opóźnień — które połączenia są typowo mniej punktualne (uzupełnienie analizy per linia o perspektywę trasy/portu docelowego).

### Wykres 14 — Opóźnienie względem odległości trasy
- **Co pokazuje:** punkty (dystans, opóźnienie) z nałożoną medianą w przedziałach co 250 km.
- **Dlaczego scatter + mediana binowana:** przy silnym zagęszczeniu punktów linia mediany w binach czytelnie pokazuje ewentualny trend.
- **Informacje o zbiorze:** **brak praktycznie istotnej zależności** — mediana jest niemal płaska, a korelacja Spearmana wynosi zaledwie **ρ ≈ 0,09**. Odległość trasy **nie jest** istotnym predyktorem opóźnienia w tym zbiorze (ważna wskazówka dla części predykcyjnej).

### Wykres 15 — Szereg czasowy: mediana opóźnień i liczba odwołań
- **Co pokazuje:** dzienna mediana opóźnienia (cienka linia), **krocząca mediana 7-dniowa** (gruba linia) oraz liczba odwołań (słupki na drugiej osi). Dni z danymi syntetycznymi są zacieniowane.
- **Dlaczego szereg + wygładzanie + oznaczenie syntetyki:** surowe wartości dzienne są zaszumione; krocząca mediana ujawnia **trend**; oznaczenie danych syntetycznych zapewnia uczciwość interpretacji.
- **Informacje o zbiorze:** pozwala ocenić **stabilność punktualności w czasie** (tendencje wzrostowe/spadkowe, dni nietypowe) i powiązać skoki odwołań z konkretnymi datami. Zacieniowane okno (18–23.04) należy interpretować ostrożnie ze względu na pochodzenie danych.

---

## 5. Podsumowanie wniosków z EDA

1. **Rozkład opóźnień jest prawoskośny** — mediana 10 min, ale 32,7% lotów przekracza próg 15 min; ~3,4% to ciężkie opóźnienia (60+ min).
2. **Pora dnia / godzina to najsilniejszy zaobserwowany czynnik** (efekt kaskadowy): ryzyko opóźnienia rośnie z 22% rano do 50% nocą; mediana z 6 do 16 min (ρ ≈ 0,21).
3. **Linia lotnicza silnie różnicuje punktualność** — od ~8% do ~64% lotów opóźnionych.
4. **Odwołania są zdominowane przez Lufthansę (≈12,5%)** i koncentrują się na wybranych trasach — zjawisko niemal binarne, silnie niezbalansowane (1,37% całości).
5. **Odległość trasy nie ma znaczenia praktycznego** (ρ ≈ 0,09) — wbrew intuicji nie jest predyktorem.
6. **Obciążenie ruchem samo w sobie nie tłumaczy opóźnień** — kluczowa jest akumulacja w ciągu doby, a nie chwilowe natężenie.

Te obserwacje wskazują zmienne czasowe (godzina, pora dnia) oraz tożsamość przewoźnika jako najbardziej obiecujące cechy dla części predykcyjnej projektu, przy jednoczesnej potrzebie obsługi silnej nierównowagi klasy odwołań.
