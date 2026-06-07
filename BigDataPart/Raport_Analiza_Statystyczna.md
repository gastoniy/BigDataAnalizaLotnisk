# Raport: Analiza statystyczna opóźnień odlotów (KRK)

**Projekt:** Analiza punktualności odlotów z lotniska Kraków-Balice (KRK)
**Zakres:** wnioskowanie statystyczne — weryfikacja hipotez i rozmiary efektu
**Moduł:** `BigDataPart/analiza_statystyczna.py`
**Dane wejściowe:** `dataset_eda_ready.csv` (6 863 rekordy, 13.04–02.06.2026)

---

## 1. Wprowadzenie i cel

Eksploracyjna analiza danych (EDA) ujawniła wzorce w opóźnieniach, ale wykres sam w sobie nie rozstrzyga, czy obserwowane różnice są **istotne statystycznie**, czy mogą wynikać z przypadku. Celem tej części jest formalna weryfikacja hipotez postawionych w EDA oraz oszacowanie **rozmiaru efektu** (praktycznego znaczenia różnic). Analiza dostarcza podstaw do selekcji cech w części predykcyjnej (model ML).

## 2. Dane i metody

Analiza obejmuje loty zrealizowane (`czy_odwolany = 0`); odwołania traktowane są jako osobny wymiar (testy dot. odwołań na pełnym zbiorze). Porównania między liniami ograniczono do przewoźników z **N ≥ 30 lotów** (22 linie), aby uniknąć artefaktów małych prób. Poziom istotności **α = 0,05**.

**Dobór testów — uzasadnienie nieparametryczne.** Rozkład opóźnień jest skrajnie prawoskośny (patrz §3.2), dlatego zamiast testów parametrycznych (ANOVA, test t Studenta) stosujemy ich odpowiedniki rangowe:

| Pytanie badawcze | Test | Rozmiar efektu |
|---|---|---|
| Czy rozkład jest normalny? | D'Agostino-Pearson, Shapiro-Wilk | skośność, kurtoza |
| Czy mediany opóźnień różnią się między liniami? | Kruskal-Wallis (+ post-hoc Dunna, Bonferroni) | ε² |
| Czy zmienne kategoryczne są zależne? | chi-kwadrat niezależności | Craméra V |
| Czy weekend różni się od dni roboczych? | Mann-Whitney U | korelacja rangowo-biserialna |
| Siła zależności monotonicznych | korelacja Spearmana | ρ |
| Niepewność estymacji średniej | bootstrap (2000 prób) | 95% CI |

Testy Dunna, Craméra V, korelację rangowo-biserialną oraz bootstrap zaimplementowano samodzielnie (zależności: tylko `scipy`, `numpy`, `pandas`).

---

## 3. Wyniki

### 3.1. Statystyki opisowe per linia (N ≥ 30)

Pełna tabela: `analiza/statystyki_opisowe_linie.csv`. Wybrane przewoźniki (posortowane wg mediany opóźnienia):

| Linia | N | mediana | średnia | IQR | % opóźn. | % odwołań |
|---|---|---|---|---|---|---|
| LX (SWISS) | 100 | 20,0 | 22,1 | 15,3 | 64,0 | 2,9 |
| LS | 67 | 19,0 | 25,9 | 24,5 | 53,7 | 0,0 |
| SN | 34 | 19,0 | 23,3 | 17,5 | 61,8 | 0,0 |
| EZY (easyJet) | 182 | 15,0 | 21,8 | 19,8 | 46,2 | 0,0 |
| KL (KLM) | 234 | 14,0 | 18,7 | 15,0 | 44,4 | 0,4 |
| LH (Lufthansa) | 294 | 12,0 | 17,4 | 18,0 | 43,5 | **12,5** |
| FR (Ryanair) | 3415 | 9,0 | 16,5 | 15,0 | 31,1 | 0,6 |
| W6 (Wizz Air) | 850 | 8,0 | 12,5 | 10,0 | 21,8 | 0,8 |
| DY (Norwegian) | 172 | 4,0 | 6,9 | 9,0 | 15,1 | 0,0 |
| OS (Austrian) | 49 | 5,0 | 7,4 | 9,0 | 8,2 | 0,0 |

**Obserwacje:** mediana opóźnienia waha się od **4 min (DY)** do **20 min (LX)** — pięciokrotna różnica. Średnia jest wyraźnie wyższa od mediany u każdego przewoźnika (np. FR: 9 vs 16,5), co potwierdza prawoskośność i zasadność mediany. Lufthansa wyróżnia się skrajnie wysokim **odsetkiem odwołań (12,5%)** przy umiarkowanym opóźnieniu.

### 3.2. Normalność rozkładu opóźnień

| Miara | Wartość |
|---|---|
| Skośność | **12,15** (silnie prawoskośny) |
| Kurtoza | **263,5** (ekstremalnie ciężki ogon) |
| D'Agostino-Pearson | K²=11 840, **p ≈ 0** |
| Shapiro-Wilk (podpróba 5000) | W=0,446, **p = 2,1·10⁻⁸²** |

Oba testy jednoznacznie odrzucają hipotezę o normalności. **Wniosek:** zastosowanie testów nieparametrycznych jest w pełni uzasadnione.

### 3.3. Kruskal-Wallis: różnice między liniami

> **H = 390,5; df = 21; p = 8,7·10⁻⁷⁰; ε² = 0,056**

Wynik **wysoce istotny** — mediany opóźnień **różnią się między liniami**. Rozmiar efektu ε² = 0,056 oznacza, że przynależność do linii wyjaśnia ~5,6% zmienności rang opóźnień: efekt **mały, ale realny** (typowe przy dużym N — istotność statystyczna ≠ duże znaczenie praktyczne).

**Post-hoc Dunna (Bonferroni), top 10 linii** — najistotniejsze różnice par (pełne: `analiza/dunn_posthoc_linie.csv`):

| Para | z | p (Bonferroni) |
|---|---|---|
| KL vs DY | 10,84 | < 10⁻¹² |
| LO vs DY | 10,59 | < 10⁻¹² |
| DY vs LX | −10,41 | < 10⁻¹² |
| EZY vs DY | 9,02 | < 10⁻¹² |
| LH vs DY | 8,89 | < 10⁻¹² |
| FR vs DY | 7,80 | 2,9·10⁻¹³ |

**DY (Norwegian)** jest „kotwicą punktualności" — różni się istotnie od niemal wszystkich dużych przewoźników. Różnice wewnątrz grupy „opóźnionych" (KL, LO, LH) są mniej wyraźne.

### 3.4. Testy chi-kwadrat niezależności

| Test | χ² | dof | p | Craméra V | Siła |
|---|---|---|---|---|---|
| pora_dnia × czy_opozniony | 232,2 | 3 | 4,7·10⁻⁵⁰ | **0,185** | słaby |
| czy_weekend × czy_opozniony | 6,8 | 1 | 9,1·10⁻³ | **0,032** | znikomy |
| linia × czy_odwolany | 343,7 | 21 | 3,7·10⁻⁶⁰ | **0,227** | słaby |

- **Pora dnia ↔ opóźnienie:** zależność istotna i najsilniejsza spośród czynników czasowych (V=0,185) — potwierdza efekt kaskadowy z EDA.
- **Weekend ↔ opóźnienie:** formalnie istotne (efekt dużej próby), ale V=0,032 to siła **znikoma** — praktycznie bez znaczenia.
- **Linia ↔ odwołanie:** najsilniejszy związek (V=0,227); odwołania nie są losowe — koncentrują się u określonych przewoźników (LH).

### 3.5. Mann-Whitney U: weekend vs dzień roboczy

> Mediana: roboczy = 10,0 (n=4823) vs weekend = 10,0 (n=1946)
> **U = 4 652 734; p = 0,58; rank-biserial = 0,009**

Wynik **nieistotny**, rozmiar efektu znikomy. W połączeniu z §3.4 daje to spójny wniosek: **dzień tygodnia / weekend nie różnicują opóźnień w sposób praktyczny**. (Pozorna istotność chi-kwadrat z §3.4 to artefakt wielkości próby.)

### 3.6. Korelacje Spearmana

| Para | ρ | p | Interpretacja |
|---|---|---|---|
| godzina ↔ opóźnienie | **0,209** | 1,8·10⁻⁶⁷ | słaby, istotny — efekt kaskadowy |
| dystans ↔ opóźnienie | **0,086** | 1,7·10⁻¹² | znikomy |

Godzina planowanego wylotu jest najsilniejszym pojedynczym, ciągłym korelatem opóźnienia. **Odległość trasy jest praktycznie nieistotna** mimo statystycznej istotności (efekt dużej próby).

### 3.7. Macierz korelacji (`analiza/16_macierz_korelacji.png`)

Macierz Spearmana potwierdza obraz: jedyną cechą zewnętrzną wyraźnie skorelowaną z opóźnieniem jest `godzina_planowana` (ρ=0,21). `dystans_km`, `miesiac`, `dzien_tygodnia_num`, `czy_weekend` mają z opóźnieniem korelacje ≈ 0. Wysokie wartości `czy_opozniony`↔`opoznienie` (0,81) i `czy_anomalia`↔`opoznienie` (0,46) są z definicji (cechy pochodne). Para `dzien_tygodnia_num`↔`czy_weekend` (0,79) to oczekiwana redundancja.

### 3.8. Bootstrapowe 95% przedziały ufności (średnie opóźnienie)

Pełne wyniki: `analiza/bootstrap_ci_linie.csv`. Skrajne przykłady:

| Linia | N | średnia | 95% CI |
|---|---|---|---|
| EZS | 38 | 26,2 | [12,1 ; 45,8] |
| LS | 67 | 25,9 | [20,5 ; 32,2] |
| LX | 100 | 22,1 | [19,1 ; 25,8] |
| DY | 172 | 6,9 | [5,6 ; 8,5] |
| OS | 49 | 7,4 | [5,0 ; 9,9] |

Szerokość przedziału odzwierciedla wielkość próby i wariancję: dla **EZS** (mała próba, duży rozrzut) CI jest bardzo szeroki [12–46], więc jego wysokiej „średniej" nie należy nadinterpretować. Dla **DY/OS** (niska wariancja) przedziały są wąskie i nie nakładają się na przedziały najgorszych linii — różnica jest wiarygodna.

---

## 4. Synteza wniosków

1. **Rozkład opóźnień jest skrajnie nienormalny** (skośność 12,2; kurtoza 264) — analizy oparto słusznie na metodach nieparametrycznych i medianie.
2. **Linia lotnicza istotnie różnicuje opóźnienia** (Kruskal-Wallis p≈10⁻⁷⁰), choć efekt jest umiarkowanie mały (ε²=0,056). DY/OS są najpunktualniejsze, LX/LS/SN najgorsze.
3. **Pora dnia / godzina to najsilniejszy zidentyfikowany czynnik opóźnień** (V=0,185; ρ=0,21) — potwierdzenie efektu kaskadowego.
4. **Odwołania są silnie związane z przewoźnikiem** (V=0,227), zdominowane przez Lufthansę (12,5%).
5. **Dzień tygodnia/weekend oraz odległość trasy nie mają praktycznego znaczenia** — istotności statystyczne to artefakty dużej próby (V≤0,03, ρ≤0,09).

**Implikacje dla części predykcyjnej (ML):** najbardziej obiecujące cechy to **godzina/pora dnia** oraz **tożsamość linii lotniczej**; cechy `dystans_km`, `czy_weekend`, `miesiac` wnoszą niewiele i mogą być kandydatami do pominięcia. Predykcja odwołań wymaga obsługi silnej nierównowagi klas (1,37%).

## 5. Ograniczenia

- Krótki horyzont (51 dni) i obecność danych syntetycznych (18–23.04) ograniczają wnioski sezonowe.
- Próba dotyczy wyłącznie odlotów z KRK — brak przylotów i innych portów.
- Małe próby niektórych linii (np. EZS, SN) dają szerokie przedziały ufności — ich rankingi traktować ostrożnie.
- Istotność statystyczna przy ~6,8 tys. obserwacji nie oznacza istotności praktycznej — dlatego konsekwentnie raportowano rozmiary efektu.

---

### Pliki wynikowe (`BigDataPart/analiza/`)
- `statystyki_opisowe_linie.csv` / `.md` — pełna tabela opisowa per linia
- `dunn_posthoc_linie.csv` — wszystkie pary post-hoc Dunna
- `bootstrap_ci_linie.csv` — przedziały ufności średnich
- `16_macierz_korelacji.png` — macierz korelacji Spearmana
