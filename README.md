# Analiza punktualności odlotów z lotniska Kraków-Balice (KRK)

Projekt uczelniany „Big Data" analizujący **punktualność odlotów z lotniska Kraków-Balice (KRK)**.
Repozytorium samodzielnie pozyskuje dane z publicznej tablicy odlotów lotniska, przechowuje je w SQLite
i zasila dwie równoległe części analityczne: **analizę eksploracyjną + statystyczną (Big Data / EDA)**
oraz **uczenie maszynowe** przewidujące opóźnienia (klasyfikacja binarna).

> **Definicja opóźnienia (wspólna dla obu części):** lot jest opóźniony, gdy
> `opóźnienie = czas_rzeczywisty − czas_planowany > 15 min` → target `czy_opozniony ∈ {0, 1}`.

Kod, komentarze, nazwy kolumn i raporty są **po polsku** — to konwencja projektu.

---

## English summary

This is a university "Big Data" project on the **departure punctuality of Kraków-Balice airport (KRK)**.
A scraper harvests the airport's public departure board daily (Playwright), a parser upserts the rows into
a SQLite database (`data/baza_lotow.db`), and the data then feeds two independent analysis tracks:

- **Big Data / EDA** (`big-data/`) — exploratory analysis (15 charts) plus a full non-parametric
  **statistical study** (Kruskal-Wallis, Dunn post-hoc, chi-square, Spearman, proportion tests, robustness control).
- **Machine Learning** (`machine-learning/`) — a delay classifier (`delay > 15 min`) with **two experiment paths**:
  (1) **model comparison** — 5 models × 2 airline encodings × 4 resampling strategies × 5 thresholds, and
  (2) **Random Forest tuning** — hyper-parameter optimisation, imbalance strategies, and k-fold vs. out-of-time validation.

Everything is written in Polish. Each script anchors its paths to its own location, so scripts run regardless
of the current working directory. See **[Jak uruchomić](#jak-uruchomić)** below.

---

## Struktura repozytorium

```
BigDataAnalizaLotnisk/
├── README.md                      # ten plik
├── Pipfile / Pipfile.lock         # zależności (pipenv, Python 3.11+/3.14)
│
├── data/                          # wspólne artefakty danych
│   ├── baza_lotow.db              # kanoniczna baza SQLite (tablica odlotów)
│   ├── dataset_loty_krakow_20260610_165348.csv   # wejście ścieżki Random Forest
│   └── dataset_loty_krakow_20260521_213240.csv   # wejście porównania modeli
│
├── data-scrapping/                # ETAP 1 — pozyskiwanie danych (wspólne)
│   ├── page_scraper.py            # Playwright → html/<data>.html
│   ├── parser.py                  # BeautifulSoup → upsert do baza_lotow.db
│   ├── transform.py               # baza_lotow.db → dataset_loty_*.csv (debug/inspekcja)
│   ├── run_flights.sh             # spina powyższe (uruchamiane z cron)
│   └── html/                      # zrzuty HTML (lokalne, ignorowane przez git)
│
├── big-data/                      # CZĘŚĆ 1 — Big Data / EDA + statystyka
│   ├── preprocessing.py           # baza_lotow.db → dataset_eda_ready.csv
│   ├── eda.py                     # → wykresy/01..15_*.png
│   ├── analiza_statystyczna_final.py   # testy statystyczne → analiza_finalna/
│   ├── dataset_eda_ready.csv      # oczyszczony, wzbogacony zbiór do analiz
│   ├── wykresy/                   # 15 wykresów EDA
│   └── analiza_finalna/           # CSV z wynikami testów + wyniki_finalne.md
│
├── machine-learning/              # CZĘŚĆ 2 — Machine Learning
│   ├── data_transform.py          # WSPÓLNY FlightsTransform (cała logika cech ML)
│   ├── model-comparison/          # ŚCIEŻKA 1 — porównanie modeli
│   │   ├── train_modeltests.py    # 5 modeli × kodowania × resamplery × progi → traininglogs/
│   │   ├── training.py            # TrainingClass — wcześniejszy orkiestrator CV
│   │   ├── plot_traininglogs.py, plot_mean_combinations.py   # traininglogs/ → trainingplots/
│   │   ├── find_best_f1_threshold15.py   # najlepszy średni F1 dla progu
│   │   ├── traininglogs/          # 122 logi JSON (wyniki eksperymentów)
│   │   └── trainingplots/         # wykresy porównawcze
│   └── random-forest/             # ŚCIEŻKA 2 — strojenie Random Forest
│       ├── rf_experiments.py      # A: HPO, B: niezbalansowanie, C: kfold vs OOT → rf_logs/
│       ├── rf_plots.py            # rf_logs/ → wykresy + rf_summary.csv
│       ├── rf_pca.py              # PCA 2D — rozdzielność klas
│       ├── rf_logs/               # 30 logów JSON
│       └── rf_plots/              # wykresy + rf_summary.csv
│
└── archive/                       # materiały stare/eksploracyjne/robocze (poza głównym flow)
    ├── ml-project-viz/            # pierwotna wizualizacja ML (vizual.py + helpers)
    ├── notes/                     # notatki robocze, prezentacja (.pptx), RF_Pipeline.md
    ├── old_dbs/, baza_lotow.db.bak, cron.log
    └── bigtest.py, create_tsne.py, test.py, *.joblib
```

---

## ETAP 1 — pozyskiwanie danych (`data-scrapping/`)

Dane pochodzą z publicznej **tablicy odlotów** lotniska Kraków-Balice, zbieranej codziennie (cron):

```
page_scraper.py  →  html/<data>.html      (Playwright renderuje tablicę odlotów, wypisuje OUTPUT_FILE=)
parser.py        →  data/baza_lotow.db     (BeautifulSoup, UPSERT do tabeli loty_odloty)
transform.py     →  data/dataset_loty_*.csv (opcjonalny eksport DB→CSV do inspekcji)
```

```bash
cd data-scrapping
./run_flights.sh                       # scrape wczoraj → parse → upsert do ../data/baza_lotow.db
./run_flights.sh --date 2026-05-01     # backfill konkretnej daty (DD/MM/YYYY lub YYYY-MM-DD)
./run_flights.sh --no-keep             # usuń pośredni HTML po zakończeniu
```

Tabela `loty_odloty` (kolumny PL): `id, data_lotu, numer_lotu, linia_lotnicza, kierunek, czas_planowany,
czas_rzeczywisty, status, ostatnia_aktualizacja`. Kluczowe fakty o danych:

- Lot **odwołany**: `status LIKE 'Odwołany%'` przy `czas_rzeczywisty` = NULL.
- Analizowane są tylko wiersze `status LIKE 'Wystartował%' OR 'Odwołany%'` (reszta to szum scrapera).
- `kierunek` zawiera **kod IATA celu w nawiasach**, np. `"MONACHIUM (MUC)"` — używany z `airportsdata`
  do wyliczenia współrzędnych i dystansu Haversine od KRK.
- Parser **upsertuje** codziennie, więc ten sam lot bywa wielokrotnie aktualizowany → część kodu deduplikuje.
- Część dat jest **syntetyczna** (wygenerowana): 2026-04-{18,19,21-23}. Ryanair (FR) to ~50% ruchu.

---

## CZĘŚĆ 1 — Big Data / EDA + statystyka (`big-data/`)

```bash
cd big-data
python3 preprocessing.py    # baza_lotow.db → dataset_eda_ready.csv
python3 eda.py              # dataset_eda_ready.csv → wykresy/01..15_*.png
python3 analiza_statystyczna_final.py   # → analiza_finalna/*.csv + wyniki_finalne.md
```

**Kluczowe decyzje `preprocessing.py`** (naprawiają błędy wcześniejszych wersji):

- **Rollover północy**: odlot po północy dawałby duże ujemne opóźnienie; delty `< −720 min` dostają `+1440`.
- **Deduplikacja** po `(data_lotu, numer_lotu, czas_planowany)` — zostaje najnowsza `ostatnia_aktualizacja`.
- Zachowane oba warianty: `opoznienie_surowe` (ze znakiem, do rozkładów) i `opoznienie_minuty` (clip ≥ 0, do metryk operacyjnych).
- Flaga anomalii regułą **IQR** (`> Q3 + 1.5·IQR`).
- Wzbogacenie: `dystans_km` (Haversine od KRK), `kategoria_opoznienia`, dzień tygodnia, pora dnia.

**Konwencje EDA:** mediana zamiast średniej (rozkład prawoskośny), odsetki/% zamiast liczności do porównań
międzygrupowych, top-N linii z progiem minimalnej liczby lotów.

**Wybrane wyniki statystyczne** (`analiza_finalna/wyniki_finalne.md`, N = 6769 lotów zrealizowanych, α = 0.05):

- Rozkład opóźnień **nie-normalny** (Shapiro-Wilk p ≈ 1e-84) → metody **rangowe**.
- **Kruskal-Wallis (linie):** istotny (p ≈ 8.7e-70), ale efekt mały (ε² = 0.059). Mediana od 4 min (DY) do 20 min (LX).
- **Kruskal-Wallis (pora dnia):** istotny (p ≈ 1.4e-50); najgorsza noc (mediana 16 min), najlepszy poranek (8 min).
- **Weekend vs dzień roboczy:** Mann-Whitney **nieistotny** (p = 0.58) — mimo że test parametryczny sugeruje inaczej
  (przykład, dlaczego ufamy testom rangowym).
- **Korelacje Spearmana:** godzina ↔ opóźnienie ρ = 0.21 (słaba), dystans ↔ opóźnienie ρ = 0.09 (znikoma).
- **Chi-kwadrat:** pora dnia × opóźnienie (V = 0.185), linia × odwołanie (V = 0.227). LH odwołuje 12.5% vs 0.8% reszta.
- **Strażnik efektu:** wiele wyników istotnych, lecz praktycznie trywialnych (małe rozmiary efektu).

---

## CZĘŚĆ 2 — Machine Learning (`machine-learning/`)

Cel: klasyfikacja binarna `czy_opozniony = (czas_rzeczywisty − czas_planowany) > 15 min`.

### Wspólny fundament — `data_transform.py`

Klasa **`FlightsTransform`** to jedyne źródło całej logiki danych ML:

- IATA z `kierunek` → współrzędne → **dystans Haversine** od KRK,
- **cykliczne kodowanie czasu** (sin/cos: godzina, dzień tygodnia, miesiąc),
- kodowanie linii: **one-hot** lub **label/ordinal**,
- skalowanie per-kolumna (std / robust / log / minmax),
- **resampling** (SMOTE / undersample / SMOTEENN / SMOTETomek).

> **Krytyczne (brak wycieku danych):** `load_xy()` celowo **nie** skaluje ani nie resampluje — robi to
> wywołujący **per fold CV**, żeby uniknąć data leakage.

Oba katalogi ścieżek importują ten plik z katalogu nadrzędnego (`machine-learning/`) poprzez wstawkę do
`sys.path` na początku skryptów — dzięki temu działają niezależnie od katalogu roboczego.

### Ścieżka 1 — porównanie modeli (`model-comparison/`)

`train_modeltests.py` uruchamia **5 modeli** (Random Forest, AdaBoost, XGBoost, Gaussian NB, MLP) w siatce
konfiguracji: **2 kodowania × 4 resamplery × 5 progów (5/10/15/20/25 min) × 3 przebiegi**, 5-fold CV
(skalowanie i resampling per fold). Każda konfiguracja → jeden log JSON w `traininglogs/` (F1 + balanced accuracy).

```bash
cd machine-learning/model-comparison
python3 train_modeltests.py                 # pełna siatka → traininglogs/  (domyślne wejście: data/…20260521…csv)
python3 plot_traininglogs.py                # traininglogs/ → trainingplots/ + aggregated_metrics.csv
python3 plot_mean_combinations.py           # średnie po kombinacjach → trainingplots/mean_combinations/
python3 find_best_f1_threshold15.py --threshold 15 --top 3   # najlepszy średni F1 dla progu
```

### Ścieżka 2 — strojenie Random Forest (`random-forest/`)

`rf_experiments.py` skupia się na Random Forest i wykonuje trzy bloki (próg = 15 min):

- **A. HPO** — RandomizedSearchCV / GridSearchCV po przestrzeni hiperparametrów RF,
- **B. Niezbalansowanie** — 7 strategii (class_weight none/balanced/balanced_subsample + SMOTE/undersample/SMOTEENN/SMOTETomek), 5-fold CV,
- **C. Walidacja** — k-fold vs **out-of-time** (podział chronologiczny, OOT = 20%).

Każdy model → log JSON w `rf_logs/`.

```bash
cd machine-learning/random-forest
python3 rf_experiments.py                   # A + B + C → rf_logs/  (domyślne wejście: data/…20260610…csv)
python3 rf_plots.py                         # rf_logs/ → rf_plots/ + rf_summary.csv
python3 rf_pca.py --encoding label          # PCA 2D rozdzielności klas → rf_plots/
```

---

## Dwie definicje opóźnienia — z założenia

Obie części niezależnie przeliczają opóźnienie z `czas_rzeczywisty − czas_planowany`; target binarny to
`> 15 min`. EDA przycina/zachowuje wartości ze znakiem do wykresów; ML podaje surowe cechy do modeli. Przy
zmianie semantyki opóźnienia należy aktualizować tylko właściwą część — celowo się rozjeżdżają.

---

## Jak uruchomić

Zależności zadeklarowane w `Pipfile` (pipenv, Python 3.11+). Wymagane pakiety: `pandas`, `numpy`,
`scikit-learn`, `imbalanced-learn`, `xgboost`, `seaborn`, `matplotlib`, `airportsdata`, `playwright`,
`beautifulsoup4`.

```bash
pipenv install            # środowisko z Pipfile
pipenv shell              # albo poprzedzaj polecenia „pipenv run"
playwright install        # jednorazowo: binaria przeglądarki dla scrapera (tylko ETAP 1)
```

Skrypty przyjmują argumenty CLI (`--help`) i domyślnie czytają/zapisują względem własnej lokalizacji, więc
można je uruchamiać z dowolnego katalogu. Brak frameworka testowego — zmiany weryfikuje się uruchomieniem
skryptu i obejrzeniem konsoli / wygenerowanych PNG / CSV.

---

## Archiwum (`archive/`)

Materiały historyczne, eksploracyjne i robocze — **nie są częścią głównego przepływu**, zachowane dla
kontekstu: pierwotna wizualizacja ML (`ml-project-viz/`), notatki i prezentacja (`notes/`), stare bazy
(`old_dbs/`, `.bak`), log crona oraz skrypty-szkice (`bigtest.py`, `create_tsne.py`, `test.py`).
