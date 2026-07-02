# Notatki mówcy — `ML_prezentacja.pptx`

Ścieżka Random Forest · predykcja opóźnień odlotów KRK · klasyfikacja `> 15 min`.
Liczby pochodzą z `rf_logs/` (seed 42, próg 15 min, ~6 863 rekordy po dedup, onehot = 49 cech).

---

## Slajd 1 — Tytuł
- Temat: przewidywanie opóźnień odlotów z Kraków-Balice (KRK).
- Ramka jednym zdaniem: **„jeden model dogłębnie" (Random Forest), nie szeroka siatka 5 modeli.**

## Slajd 2–3 — Cel i opis pracy
- Zadanie: **klasyfikacja binarna** — czy lot opóźniony `> 15 min`.
- Etykieta: `czy_opozniony = (czas_rzeczywisty − czas_planowany) > 15 min`.
- **Dlaczego F1, nie accuracy?** Klasy niezbalansowane → accuracy myli. Patrzymy na F1 klasy mniejszościowej (opóźnione) + balanced accuracy.
- Świadoma decyzja: zamiast 5 modeli — jeden estymator + 3 eksperymenty (A/B/C).

## Slajd 4–6 — Dane
- Źródło: **publiczna tablica odlotów KRK**, renderowana w JS → dlatego Playwright, nie zwykły request.
- Potok codzienny (cron): `page_scraper.py` (Playwright→HTML) → `parser.py` (BeautifulSoup→SQLite `loty_odloty`).
- **Pułapka do wspomnienia:** parser robi **upsert** → ten sam lot bywa zapisany wielokrotnie → **deduplikacja** konieczna (po `data_lotu, numer_lotu, czas_planowany`).
- Po deduplikacji ~**6 863 rekordy**, okno ~kwiecień–maj 2026 (wąskie okno — wróci przy wynikach).
- **13 cech**, onehot daje 49 kolumn: dystans Haversine od KRK (IATA→współrzędne via `airportsdata`), cykliczne sin/cos (godzina, dzień tygodnia, miesiąc), `jest_weekend`, `dzien_miesiaca`, kodowanie linii (one-hot LUB label).
- **Najważniejsze zdanie sekcji:** `load_xy` **NIE skaluje i NIE resampluje** — to robione **per-fold**, by uniknąć **wycieku danych**. Decyzja architektury, nie zaniedbanie.

## Slajd 7–9 — Teoria Random Forest
- Ensemble drzew + **bagging** (próbki bootstrap) + **losowy podzbiór cech** przy podziale → dekorelacja drzew.
- Predykcja = głosowanie / średnie prawdopodobieństwo.
- `predict_proba` → pozwala stroić **próg** i liczyć **ROC-AUC**.
- Hiperparametry: zapowiedz, że **`class_weight` i `max_features` są najważniejsze**, a `n_estimators` prawie bez wpływu (wniosek z eksperymentu A).

---

# CZĘŚĆ EKSPERYMENTALNA (rozbudowana)

## Slajd 10–11 — Uwagi do kodu / architektura
- `FlightsTransform` = **jedyne** źródło logiki danych; skrypty eksperymentalne jej nie duplikują.
- Wspólny obiekt wszystkich eksperymentów — **leakage-safe `imblearn.Pipeline`**:
  ```
  Pipeline([
    ("scaler",    ft.get_scaler()),          # ColumnTransformer, fit per fold
    ("resampler", <sampler> | passthrough),  # imblearn: TYLKO fold treningowy
    ("rf",        RandomForestClassifier),
  ])
  ```
- Kluczowe: imblearn stosuje `resampler` **tylko na foldzie treningowym** → CV/search wokół tego obiektu jest odporne na wyciek **bez** ręcznej pętli per-fold. Scaler też re-fitowany na każdym foldzie.
- Dwa tryby badania niezbalansowania:
  - badamy `class_weight` → `resampler = passthrough`, zmieniamy `rf__class_weight`;
  - badamy resampling → `resampler = <sampler>`, `rf__class_weight = None`.
- Skalowanie **per-kolumna** wg rozkładu cechy (Standard / Robust / log1p / MinMax).
- Logi JSON **schema-kompatybilne** z `train_modeltests.py` → stare plotery działają na `rf_logs/` bez zmian (RF jest jedynym modelem w logu).
- Wspólne dla A/B/C: próg 15 min, metryka główna **F1**, druga **balanced accuracy**, seed bazowy 42.

## Slajd 12 — Eksperymenty: przegląd
- Stałe założenia: próg 15 min, tylko RF, 2 kodowania linii (`onehot`, `label`).
- **A — HPO:** `RandomizedSearchCV`, 40 kandydatów, 5-fold stratyfikowany, `refit = f1`.
- **B — strategia niezbalansowania:** 7 strategii pod identycznym `StratifiedKFold(5)`.
- **C — schemat walidacji:** te same 7 strategii w `StratifiedKFold` vs **chronologiczny OOT 80/20**.
- Wszystko → logi JSON → wykresy w `rf_plots/`.

---

## Slajd 13 (A) — HPO: szczegóły

**Procedura**
- Pipeline z `resampler = passthrough` → niezbalansowanie zaszyte w przestrzeni przeszukiwania jako `class_weight` (HPO i obsługa niezbalansowania **strojone łącznie**).
- `RandomizedSearchCV`: 40 kandydatów, `cv = StratifiedKFold(5, shuffle=True, seed=42)`, `scoring = {f1, balanced_accuracy}`, `refit = f1`.
- Uruchamiane raz na próg 15, osobno dla każdego kodowania.

**Przestrzeń przeszukiwania**

| Parametr | Wartości |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | 10, 15, 20, 25, None |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | "sqrt", "log2", 0.5 |
| `criterion` | gini, entropy |
| `class_weight` | None, balanced, balanced_subsample |

**Najlepsze parametry (z logów)**
- **onehot:** F1 = **0.5813** (±0.0225), bal_acc = 0.6872
  `n_estimators=200, max_depth=25, min_samples_leaf=4, max_features=log2, criterion=entropy, class_weight=balanced_subsample`
- **label:** F1 = **0.5827** (±0.0210), bal_acc = 0.6865
  `n_estimators=500, max_depth=10, min_samples_leaf=4, max_features=sqrt, criterion=gini, class_weight=balanced_subsample`

**Wniosek liczbowy (mocna pointa):**
- **Najlepszy** kandydat F1 ≈ 0.581; **najgorszy** F1 ≈ **0.369**.
- **Cała ta różnica to `class_weight`:** każdy z 3 najgorszych kandydatów ma `class_weight=None`. Najgorszy: `class_weight=None` → 0.369.
- Najlepsze konfiguracje wśród bardzo różnych `n_estimators` (100, 200, 500) i `max_depth` (10 vs 25) dają niemal identyczne F1 (0.580–0.581) → **liczba drzew i głębokość praktycznie bez wpływu**.
- Wniosek: strojenie tylko **przesuwa kompromis precision/recall**, nie tworzy nowego sygnału. Realne dźwignie to **`class_weight` i `max_features`**.

> Wartości najlepszych params różnią się między onehot i label (np. głębokość 25 vs 10) — bo to różne kandydaty z randomized search; **ważne, że F1 niemal identyczne** (~0.58). To potwierdza płaskie plateau, a nie czułość na konkretny zestaw.

---

## Slajd 14 (B) — Strategia niezbalansowania: szczegóły

**Procedura**
- RF zafiksowany na `best_params` z A (bez `class_weight`, który staje się zmienną).
- 7 strategii pod identycznym `StratifiedKFold(5, shuffle=True, seed=42)`:

| # | resampler | `class_weight` |
|---|---|---|
| 1 | passthrough | None (baseline) |
| 2 | passthrough | balanced |
| 3 | passthrough | balanced_subsample |
| 4 | smote | None |
| 5 | undersample | None |
| 6 | smoteenn | None |
| 7 | smotetomek | None |

**Wyniki — ranking F1 (onehot, kfold):**

| Strategia | F1 | std | bal_acc |
|---|---|---|---|
| cw_balanced_subsample | **0.5813** | 0.0225 | 0.6872 |
| cw_balanced | 0.5798 | 0.0204 | 0.6860 |
| undersample | 0.5772 | 0.0161 | 0.6774 |
| smoteenn | 0.5722 | 0.0198 | 0.6748 |
| smotetomek | 0.5403 | 0.0207 | 0.6631 |
| smote | 0.5402 | 0.0230 | 0.6630 |
| **cw_none** | **0.4145** | 0.0358 | 0.6133 |

(label bardzo podobnie: zwycięzca `cw_balanced_subsample` 0.5827; `cw_none` najgorszy 0.4009.)

**Wnioski do powiedzenia:**
- **`cw_none` zdecydowanie najsłabszy** (0.41 vs 0.58) — brak obsługi niezbalansowania to realny koszt ~0.17 F1.
- **`class_weight ≈ resampling`** w wynikach, ale `class_weight` jest **prostsze i tańsze** (brak generowania syntetycznych próbek) → rekomendacja.
- Czyste **SMOTE / SMOTETomek wyraźnie słabsze** (~0.54) od `balanced` (~0.58) — przeważanie wagą bije nadpróbkowanie interpolacją.
- `undersample` zaskakująco dobry (0.577) mimo wyrzucania danych — i ma **najniższą wariancję** (std 0.016).

---

## Slajd 15 (C) — Schemat walidacji: kfold vs OOT

**Procedura OOT (uczciwa, „predykcja przyszłości")**
- Osobno wczytujemy surowy CSV, parsujemy `czas_planowany` (bo `load_xy` go usuwa, a `data_transform.py` zostaje nietknięty).
- Datę reindeksujemy na `X.index` (preprocess zachowuje oryginalny indeks) → sortujemy chronologicznie.
- **Najwcześniejsze 80% = trening (5 490 wierszy), najpóźniejsze 20% = test (1 373 wiersze)** — pojedyncze cięcie, bez foldów. Scaler + resampler fitowane tylko na treningu.

**Wyniki — ranking F1 (onehot, OOT):**

| Strategia | F1 (OOT) | bal_acc |
|---|---|---|
| undersample | **0.6088** | 0.6584 |
| smoteenn | 0.5939 | 0.6518 |
| cw_balanced_subsample | 0.5922 | 0.6648 |
| cw_balanced | 0.5920 | 0.6651 |
| smotetomek | 0.5508 | 0.6471 |
| smote | 0.5456 | 0.6408 |
| **cw_none** | **0.3394** | 0.5787 |

**Porównanie Δ = kfold − OOT (najważniejsza interpretacja, onehot):**

| Strategia | kfold F1 | OOT F1 | Δ |
|---|---|---|---|
| cw_none | 0.4145 | 0.3394 | **+0.075** (duży spadek) |
| cw_balanced_subsample | 0.5813 | 0.5922 | −0.011 |
| cw_balanced | 0.5798 | 0.5920 | −0.012 |
| undersample | 0.5772 | 0.6088 | −0.032 |
| smoteenn | 0.5722 | 0.5939 | −0.022 |

**Wnioski — powiedz to wprost (i uczciwie):**
- **`cw_none` jako jedyna strategia wyraźnie traci na OOT** (0.41→0.34) — bez obsługi niezbalansowania model najgorzej generalizuje w czasie.
- **Dobre strategie generalizują dobrze** — Δ bliskie zera, a część (undersample, balanced) ma na OOT **F1 nawet wyższe** niż na kfold.
- **Ważny niuans / bądź gotów na pytanie:** OOT NIE jest tu jednolicie niższy od kfold — naiwna narracja „kfold zawsze optymistyczny" się nie potwierdza. Powód → **caveat poniżej**.
- **CAVEAT (musi paść):** część dat jest **syntetyczna** (`note.txt`: 2026-04-18, 19, 21–23). Sztuczne dni zaburzają prawdziwy sygnał chronologiczny, więc do OOT podchodzimy ostrożnie — to dlatego wynik „przyszłości" potrafi być wyższy niż CV.
- Ranking strategii jest **spójny** w obu schematach (cw_none zawsze ostatni, balanced/undersample na czele) → wniosek o niezbalansowaniu jest odporny na wybór walidacji.

---

## Slajd 16 — co podsumować po eksperymentach (mostek do wyników)
- Trzy eksperymenty zbiegają się do jednej liczby: **F1 ≈ 0.57–0.58, niezależnie od** parametrów, strategii i schematu walidacji.
- To nie zbieg okoliczności — to **plateau**. Przejście do slajdu „dlaczego sufit".

## Slajd 17–18 — Wyniki i diagnoza (najważniejszy slajd)
- Liczby: **F1 ≈ 0.56–0.58**, balanced accuracy ≈ 0.67–0.69, **ROC-AUC = 0.74**.
- AUC 0.74 → sygnał **realny, ale umiarkowany**; model ~80% powyżej losowego baseline (F1 ≈ 0.30).
- **Główna teza: to sufit CECH, a nie modelu** — żadne HPO go nie przebije (pokazaliśmy: różne params → ta sama F1).
- Dowody: ważności cech **płaskie** (top ≈ 0.15, linia ≈ 0.06), cechy `miesiac_*` martwe (wąskie 2-miesięczne okno).
- Czego brakuje danym: **opóźnienie rotacyjne** (spóźniony samolot przylotowy — najsilniejszy predyktor w literaturze), **pogoda**, **ATFM**. Tablica odlotów tego nie rejestruje.

## Slajd 19–20 — Dalsze kroki
- **Dźwignia = bogatsze źródło, nie większy model** — to puenta.
- Zalecane: **US DOT BTS** (On-Time Performance / Delay Causes) — ma dokładnie brakujące kolumny: `LateAircraftDelay`, `WeatherDelay`, `NASDelay`, `CarrierDelay`; miliony lotów, 20+ lat → realna sezonowość.
- Alternatywa bliżej KRK: **EUROCONTROL ADRR** (dane europejskie, ale trajektoryjne, więcej obróbki).
- Łatwe rozszerzenia: strojenie progu prawdopodobieństwa, XGBoost (hook gotowy), cechy operacyjne (kroczące opóźnienie / zatłoczenie). Ten sam pipeline `FlightsTransform → RF` przenosi się 1:1; zmienia się tylko mapowanie kolumn.

## Slajd 21 — Q&A: przygotowane odpowiedzi
- *„Czemu nie głębsze drzewa / więcej estymatorów?"* → testowane w A; `n_estimators` (100 vs 500) i `max_depth` (10 vs 25) bez wpływu na F1. Problem to cechy.
- *„Czemu nie inny model / sieć?"* → ROC-AUC 0.74 to sufit danych; inny model nie wyciągnie sygnału, którego nie ma.
- *„Skąd pewność, że to sufit cech?"* → płaskie ważności + identyczne F1 dla bardzo różnych konfiguracji + martwe cechy miesiąca.
- *„Czemu OOT czasem wyższy niż kfold?"* → daty syntetyczne (note.txt) zaburzają sygnał czasowy; dlatego OOT raportujemy z caveatem.
- *„class_weight vs SMOTE?"* → w naszych danych `class_weight=balanced(_subsample)` ≈ undersample, a oba biją czyste SMOTE; `class_weight` prostsze → rekomendacja.
- *„Wyciek danych?"* → scaler i resampler w `imblearn.Pipeline` fitowane per-fold (resampler tylko na treningu), nigdy na całości.
