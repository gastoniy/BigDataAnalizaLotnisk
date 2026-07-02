# Wyniki: finalna analiza statystyczna (auto-generowane)

Poziom istotności α = 0.05. Loty zrealizowane: N = 6769. Metoda: rangowa (opóźnienie) + chi-kwadrat (kategorie).


## 1. Normalność rozkładu (uzasadnienie metod rangowych)

- skośność = 0.27, kurtoza = 185.2
- Shapiro-Wilk (n=5000): W = 0.384, p = 1.03e-84 → rozkład **NIE-normalny**


## 2. Statystyki opisowe per linia (mediana + bootstrap 95% CI)

Ogółem: mediana = 10.0 min, średnia = 16.2 min.

| Linia | N | mediana | 95% CI | IQR | % opóźn. | % odwołań |
|---|---|---|---|---|---|---|
| LX | 100 | 20.0 | [17.0; 22.5] | 15.2 | 64.0 | 2.9 |
| LS | 67 | 19.0 | [14.0; 24.1] | 24.5 | 53.7 | 0.0 |
| SN | 34 | 19.0 | [14.5; 27.5] | 17.5 | 61.8 | 0.0 |
| EN | 48 | 16.0 | [9.0; 20.0] | 18.2 | 50.0 | 2.0 |
| W4 | 40 | 16.0 | [10.0; 21.0] | 16.5 | 50.0 | 0.0 |
| EZY | 182 | 15.0 | [12.0; 17.0] | 19.8 | 46.2 | 0.0 |
| TK | 52 | 15.0 | [13.0; 18.5] | 12.2 | 46.2 | 3.7 |
| KL | 234 | 14.0 | [13.0; 16.0] | 15.0 | 44.4 | 0.4 |
| EJU | 94 | 13.5 | [11.0; 16.5] | 20.2 | 42.6 | 0.0 |
| EW | 49 | 13.0 | [9.0; 18.0] | 17.0 | 40.8 | 3.9 |
| AF | 30 | 13.0 | [8.0; 16.5] | 12.8 | 36.7 | 3.2 |
| LO | 513 | 13.0 | [12.0; 13.0] | 12.0 | 38.0 | 1.5 |
| EZS | 38 | 12.5 | [5.0; 15.0] | 14.2 | 31.6 | 0.0 |
| LH | 294 | 12.0 | [11.0; 15.0] | 18.0 | 43.5 | 12.5 |
| BA | 48 | 10.0 | [7.0; 15.0] | 15.0 | 35.4 | 0.0 |
| FR | 3415 | 9.0 | [8.0; 9.0] | 15.0 | 31.1 | 0.6 |
| W6 | 850 | 8.0 | [8.0; 9.0] | 10.0 | 21.8 | 0.8 |
| SK | 85 | 7.0 | [5.0; 9.0] | 11.0 | 14.1 | 2.3 |
| AY | 127 | 6.0 | [4.0; 9.0] | 16.5 | 26.8 | 1.6 |
| D8 | 71 | 5.0 | [2.0; 6.0] | 9.5 | 14.1 | 0.0 |
| OS | 49 | 5.0 | [4.0; 7.0] | 9.0 | 8.2 | 0.0 |
| DY | 172 | 4.0 | [2.0; 5.0] | 9.0 | 15.1 | 0.0 |


## 3. Kruskal-Wallis: różnice między liniami

H = 390.5; df = 21; p = 8.73e-70; ε² = 0.059 (efekt mały) → **istotny**


Post-hoc Dunna (Bonferroni), najistotniejsze pary:

| Para | mediana A | mediana B | p (Bonferroni) | werdykt |
|---|---|---|---|---|
| DY vs KL | 4.0 | 14.0 | 9.21e-25 | istotny |
| DY vs LO | 4.0 | 13.0 | 1.44e-23 | istotny |
| DY vs LX | 4.0 | 20.0 | 6.98e-23 | istotny |
| DY vs EZY | 4.0 | 15.0 | 5.72e-17 | istotny |
| DY vs LH | 4.0 | 12.0 | 1.81e-16 | istotny |
| DY vs LS | 4.0 | 19.0 | 1.82e-16 | istotny |
| D8 vs LX | 5.0 | 20.0 | 1.54e-13 | istotny |
| DY vs FR | 4.0 | 9.0 | 1.90e-12 | istotny |
| KL vs W6 | 14.0 | 8.0 | 2.75e-12 | istotny |
| AY vs LX | 6.0 | 20.0 | 4.36e-12 | istotny |


## 4. Kruskal-Wallis: różnice między porami dnia

H = 234.6; df = 3; p = 1.38e-50; ε² = 0.035 (efekt mały) → **istotny**


| Para | mediana A | mediana B | p (Bonferroni) | werdykt |
|---|---|---|---|---|
| Rano vs Wieczór | 8.0 | 12.0 | 5.05e-38 | istotny |
| Popołudnie vs Rano | 11.0 | 8.0 | 3.05e-25 | istotny |
| Noc vs Rano | 16.0 | 8.0 | 7.93e-19 | istotny |
| Noc vs Popołudnie | 16.0 | 11.0 | 0.000612 | istotny |
| Noc vs Wieczór | 16.0 | 12.0 | 0.0412 | istotny |
| Popołudnie vs Wieczór | 11.0 | 12.0 | 0.123 | nieistotny |


## 5. Mann-Whitney U: weekend vs dzień roboczy

- mediana: roboczy = 10.0 (n=4823) vs weekend = 10.0 (n=1946)
- U = 4652734; p = 0.582 → **nieistotny**
- rank-biserial = -0.009 (efekt znikomy)


## 6. Korelacje Spearmana

| Para | ρ | 95% CI | p | interpretacja | werdykt |
|---|---|---|---|---|---|
| godzina_planowana ↔ opóźnienie | 0.209 | [0.19 0.23] | 1.78e-67 | słaba | istotny |
| dystans_km ↔ opóźnienie | 0.086 | [0.06 0.11] | 1.71e-12 | znikoma | istotny |


## 7. Testy chi-kwadrat niezależności

| Test | χ² | dof | p | V Craméra | siła | werdykt |
|---|---|---|---|---|---|---|
| pora_dnia × czy_opozniony | 232.17 | 3 | 4.69e-50 | 0.185 | słaby | istotny |
| czy_weekend × czy_opozniony | 6.81 | 1 | 0.00906 | 0.032 | znikomy | istotny |
| linia × czy_odwolany | 343.71 | 21 | 3.69e-60 | 0.227 | słaby | istotny |


## 8. Testy dla dwóch proporcji

| Porównanie | grupa 1 (%) | grupa 2 (%) | z | p | werdykt |
|---|---|---|---|---|---|
| % opóźnionych: weekend vs roboczy | 35.0 | 31.7 | 2.638 | 0.00833 | istotny |
| % odwołań: LH vs pozostałe | 12.5 | 0.8 | 18.0 | 1.96e-72 | istotny |


## 9. Kontrola odporności (parametryczny vs rangowy)

| Pytanie | Test parametryczny | Test rangowy | Rozbieżność |
|---|---|---|---|
| Weekend vs roboczy | t-Studenta: istotny (p=0.000941) | Mann-Whitney: nieistotny (p=0.582) | TAK — ufaj rangowemu |
| Pora dnia | ANOVA: istotny (p=3.92e-29) | Kruskal-Wallis: istotny (p=1.38e-50) | nie |
| Godzina ↔ opóźnienie | Pearson: istotny (p=1.07e-32) | Spearman: istotny (p=1.78e-67) | nie |
| Dystans ↔ opóźnienie | Pearson: istotny (p=0.000206) | Spearman: istotny (p=1.71e-12) | nie |


## 10. Strażnik efektu — wyniki istotne, lecz praktycznie trywialne

- Kruskal (linie): istotny, ale ε²=0.059 (mały).
- Kruskal (pora dnia): istotny, ale ε²=0.035 (mały).
- Spearman (dystans_km ↔ opóźnienie): istotny, ale ρ=0.086 (znikomy).
- Chi² (czy_weekend × czy_opozniony): istotny, ale V=0.032 (znikomy).
