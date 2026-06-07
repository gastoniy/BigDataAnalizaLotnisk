"""
Analiza statystyczna opóźnień odlotów z lotniska Kraków-Balice (KRK).

Wchodzi krok dalej niż EDA: weryfikuje hipotezy testami istotności i podaje
rozmiary efektu. Ze względu na silnie skośny rozkład opóźnień oraz niezbalansowane
klasy stosowane są metody NIEPARAMETRYCZNE.

Wejście:  dataset_eda_ready.csv (produkt preprocessing.py)
Wyjście:  analiza/ — tabele CSV/MD oraz macierz korelacji (PNG),
          pełny zrzut wyników wypisywany na konsolę.

Zależności: pandas, numpy, scipy, matplotlib, seaborn (bez statsmodels /
scikit-posthocs / tabulate — test Dunna, Craméra V, korelacja rangowo-biserialna,
bootstrap oraz eksport Markdown zaimplementowane samodzielnie).
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

_TU = os.path.dirname(os.path.abspath(__file__))
DOMYSLNE_WEJSCIE = os.path.join(_TU, "dataset_eda_ready.csv")
FOLDER_WYJSCIA = os.path.join(_TU, "analiza")

MIN_LOTOW = 30          # próg minimalnej liczby lotów dla porównań per linia
TOP_N_DUNN = 10         # ile linii porównywać parami w teście post-hoc Dunna
ALPHA = 0.05
RNG = np.random.default_rng(42)

PORY_DNIA = ["Rano", "Popołudnie", "Wieczór", "Noc"]


# --------------------------------------------------------------------------- #
#  Pomocnicze: eksport Markdown, miary i testy (implementacje własne)
# --------------------------------------------------------------------------- #
def _df_to_md(df):
    """Prosty eksport DataFrame -> tabela Markdown (bez biblioteki tabulate)."""
    cols = list(df.columns)
    naglowek = "| " + " | ".join(map(str, cols)) + " |"
    linia = "| " + " | ".join("---" for _ in cols) + " |"
    wiersze = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([naglowek, linia, *wiersze])


def cramers_v(tab):
    """Craméra V — siła związku dla tablicy kontyngencji."""
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def interpretuj_v(v):
    if v < 0.1:
        return "znikomy"
    if v < 0.3:
        return "słaby"
    if v < 0.5:
        return "umiarkowany"
    return "silny"


def interpretuj_rho(rho):
    a = abs(rho)
    if a < 0.1:
        return "znikomy"
    if a < 0.3:
        return "słaby"
    if a < 0.5:
        return "umiarkowany"
    return "silny"


def rank_biserial(x, y, U):
    """Korelacja rangowo-biserialna jako rozmiar efektu dla Manna-Whitneya."""
    return 1 - (2 * U) / (len(x) * len(y))


def dunn_test(df, val_col, group_col, groups):
    """Test post-hoc Dunna z poprawką Bonferroniego (po istotnym Kruskalu-Wallisie)."""
    sub = df[df[group_col].isin(groups)][[val_col, group_col]].dropna().copy()
    sub["rank"] = sub[val_col].rank()
    N = len(sub)

    _, counts = np.unique(sub[val_col].to_numpy(), return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    sigma2 = (N * (N + 1)) / 12.0 - ties / (12.0 * (N - 1))

    grp = sub.groupby(group_col, observed=True)["rank"]
    Rbar, n = grp.mean(), grp.size()

    pary = list(combinations(groups, 2))
    m = len(pary)
    wyniki = []
    for a, b in pary:
        se = np.sqrt(sigma2 * (1 / n[a] + 1 / n[b]))
        z = (Rbar[a] - Rbar[b]) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        wyniki.append((a, b, round(z, 3), p, min(p * m, 1.0)))
    out = pd.DataFrame(wyniki, columns=["linia_A", "linia_B", "z", "p", "p_bonferroni"])
    return out.sort_values("p_bonferroni")


def bootstrap_ci(x, n_boot=2000, ci=95, stat=np.mean):
    """Przedział ufności bootstrapowy dla wskazanej statystyki."""
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return stat(x), lo, hi


# --------------------------------------------------------------------------- #
#  Wczytanie danych
# --------------------------------------------------------------------------- #
def wczytaj_dane(path=DOMYSLNE_WEJSCIE):
    df = pd.read_csv(path)
    df["pora_dnia"] = pd.Categorical(df["pora_dnia"], categories=PORY_DNIA, ordered=True)
    return df


def _linie_z_progiem(df):
    zr = df[df["czy_odwolany"] == 0]
    liczby = zr["linia_lotnicza"].value_counts()
    return liczby[liczby >= MIN_LOTOW].index.tolist()


# --------------------------------------------------------------------------- #
#  1. Statystyki opisowe per linia
# --------------------------------------------------------------------------- #
def statystyki_opisowe(df, output_dir):
    zr = df[df["czy_odwolany"] == 0]
    opis = (
        zr.groupby("linia_lotnicza")
        .agg(
            N=("opoznienie_minuty", "size"),
            srednia=("opoznienie_minuty", "mean"),
            mediana=("opoznienie_minuty", "median"),
            std=("opoznienie_minuty", "std"),
            q1=("opoznienie_minuty", lambda s: s.quantile(0.25)),
            q3=("opoznienie_minuty", lambda s: s.quantile(0.75)),
            pct_opozn=("czy_opozniony", "mean"),
        )
    )
    odwol = df.groupby("linia_lotnicza")["czy_odwolany"].mean().rename("pct_odwol")
    opis = opis.join(odwol)
    opis["IQR"] = opis["q3"] - opis["q1"]
    opis["pct_opozn"] *= 100
    opis["pct_odwol"] *= 100
    opis = opis[opis["N"] >= MIN_LOTOW].sort_values("mediana", ascending=False).round(2)

    opis.to_csv(os.path.join(output_dir, "statystyki_opisowe_linie.csv"), encoding="utf-8")
    with open(os.path.join(output_dir, "statystyki_opisowe_linie.md"), "w", encoding="utf-8") as f:
        f.write(_df_to_md(opis.reset_index()))
    return opis


# --------------------------------------------------------------------------- #
#  2. Sprawdzenie normalności rozkładu opóźnień
# --------------------------------------------------------------------------- #
def test_normalnosci(df):
    x = df.loc[df["czy_odwolany"] == 0, "opoznienie_minuty"].dropna().to_numpy()
    skosnosc = stats.skew(x)
    kurt = stats.kurtosis(x)
    # D'Agostino-Pearson na pełnej próbie
    k2, p_norm = stats.normaltest(x)
    # Shapiro-Wilk na podpróbie (ograniczenie metody)
    proba = RNG.choice(x, size=min(5000, len(x)), replace=False)
    w, p_sh = stats.shapiro(proba)
    return {
        "skosnosc": skosnosc, "kurtoza": kurt,
        "dagostino_k2": k2, "dagostino_p": p_norm,
        "shapiro_w": w, "shapiro_p": p_sh,
    }


# --------------------------------------------------------------------------- #
#  3. Kruskal-Wallis (różnice opóźnień między liniami) + Dunn
# --------------------------------------------------------------------------- #
def kruskal_linie(df, output_dir):
    zr = df[df["czy_odwolany"] == 0]
    linie = _linie_z_progiem(df)
    grupy = [zr.loc[zr["linia_lotnicza"] == l, "opoznienie_minuty"].dropna() for l in linie]
    H, p = stats.kruskal(*grupy)
    # epsilon^2 jako rozmiar efektu
    n = sum(len(g) for g in grupy)
    eps2 = (H - len(grupy) + 1) / (n - len(grupy))

    top = (
        zr[zr["linia_lotnicza"].isin(linie)]
        .groupby("linia_lotnicza")["opoznienie_minuty"]
        .size().sort_values(ascending=False).head(TOP_N_DUNN).index.tolist()
    )
    dunn = dunn_test(zr, "opoznienie_minuty", "linia_lotnicza", top)
    dunn.to_csv(os.path.join(output_dir, "dunn_posthoc_linie.csv"), index=False, encoding="utf-8")
    return {"H": H, "p": p, "eps2": eps2, "k": len(grupy)}, dunn


# --------------------------------------------------------------------------- #
#  4. Testy chi-kwadrat niezależności + Craméra V
# --------------------------------------------------------------------------- #
def testy_chi2(df):
    wyniki = []

    # pora_dnia × czy_opozniony
    zr = df[df["czy_odwolany"] == 0]
    t1 = pd.crosstab(zr["pora_dnia"], zr["czy_opozniony"])
    chi, p, dof, _ = stats.chi2_contingency(t1)
    wyniki.append(("pora_dnia × czy_opozniony", chi, p, dof, cramers_v(t1)))

    # czy_weekend × czy_opozniony (2x2 -> korekta Yatesa)
    t2 = pd.crosstab(zr["czy_weekend"], zr["czy_opozniony"])
    chi, p, dof, _ = stats.chi2_contingency(t2)
    wyniki.append(("czy_weekend × czy_opozniony", chi, p, dof, cramers_v(t2)))

    # linia × czy_odwolany (tylko linie z progiem)
    linie = df["linia_lotnicza"].value_counts()
    linie = linie[linie >= MIN_LOTOW].index
    t3 = pd.crosstab(df.loc[df["linia_lotnicza"].isin(linie), "linia_lotnicza"],
                     df.loc[df["linia_lotnicza"].isin(linie), "czy_odwolany"])
    chi, p, dof, _ = stats.chi2_contingency(t3)
    wyniki.append(("linia × czy_odwolany", chi, p, dof, cramers_v(t3)))

    return pd.DataFrame(wyniki, columns=["test", "chi2", "p", "dof", "cramers_v"])


# --------------------------------------------------------------------------- #
#  5. Mann-Whitney U: weekend vs dzień roboczy
# --------------------------------------------------------------------------- #
def mann_whitney_weekend(df):
    zr = df[df["czy_odwolany"] == 0]
    rob = zr.loc[zr["czy_weekend"] == 0, "opoznienie_minuty"].dropna()
    wknd = zr.loc[zr["czy_weekend"] == 1, "opoznienie_minuty"].dropna()
    U, p = stats.mannwhitneyu(rob, wknd, alternative="two-sided")
    rrb = rank_biserial(rob, wknd, U)
    return {
        "U": U, "p": p, "rank_biserial": rrb,
        "mediana_roboczy": rob.median(), "mediana_weekend": wknd.median(),
        "n_roboczy": len(rob), "n_weekend": len(wknd),
    }


# --------------------------------------------------------------------------- #
#  6. Korelacje Spearmana
# --------------------------------------------------------------------------- #
def korelacje_spearman(df):
    zr = df[df["czy_odwolany"] == 0].dropna(subset=["opoznienie_minuty"])
    out = {}
    rho, p = stats.spearmanr(zr["godzina_planowana"], zr["opoznienie_minuty"])
    out["godzina↔opoznienie"] = (rho, p)
    zr2 = zr.dropna(subset=["dystans_km"])
    rho, p = stats.spearmanr(zr2["dystans_km"], zr2["opoznienie_minuty"])
    out["dystans↔opoznienie"] = (rho, p)
    return out


# --------------------------------------------------------------------------- #
#  7. Macierz korelacji (Spearman) — heatmapa
# --------------------------------------------------------------------------- #
def macierz_korelacji(df, output_dir):
    kolumny = ["opoznienie_minuty", "opoznienie_surowe", "godzina_planowana",
               "dzien_tygodnia_num", "czy_weekend", "miesiac", "dystans_km",
               "czy_opozniony", "czy_anomalia"]
    kolumny = [c for c in kolumny if c in df.columns]
    corr = df[kolumny].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=.5, cbar_kws={"label": "ρ Spearmana"})
    plt.title("Macierz korelacji Spearmana cech numerycznych", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "16_macierz_korelacji.png"), dpi=120)
    plt.close()
    return corr


# --------------------------------------------------------------------------- #
#  8. Bootstrapowe przedziały ufności średniego opóźnienia per linia
# --------------------------------------------------------------------------- #
def bootstrap_linie(df, output_dir):
    zr = df[df["czy_odwolany"] == 0]
    linie = _linie_z_progiem(df)
    rzedy = []
    for l in linie:
        x = zr.loc[zr["linia_lotnicza"] == l, "opoznienie_minuty"].dropna().to_numpy()
        sr, lo, hi = bootstrap_ci(x)
        rzedy.append((l, len(x), round(sr, 2), round(lo, 2), round(hi, 2)))
    out = pd.DataFrame(rzedy, columns=["linia_lotnicza", "N", "srednia", "ci_dol", "ci_gora"])
    out = out.sort_values("srednia", ascending=False)
    out.to_csv(os.path.join(output_dir, "bootstrap_ci_linie.csv"), index=False, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
#  Orkiestracja
# --------------------------------------------------------------------------- #
def main(path=DOMYSLNE_WEJSCIE, output_dir=FOLDER_WYJSCIA):
    os.makedirs(output_dir, exist_ok=True)
    df = wczytaj_dane(path)
    print(f"Wczytano {len(df)} rekordów\n")

    print("=" * 70)
    print("1. STATYSTYKI OPISOWE PER LINIA (min. %d lotów)" % MIN_LOTOW)
    print("=" * 70)
    opis = statystyki_opisowe(df, output_dir)
    print(opis.to_string())

    print("\n" + "=" * 70)
    print("2. NORMALNOŚĆ ROZKŁADU OPÓŹNIEŃ")
    print("=" * 70)
    norm = test_normalnosci(df)
    print(f"Skośność: {norm['skosnosc']:.3f}  |  Kurtoza: {norm['kurtoza']:.3f}")
    print(f"D'Agostino-Pearson: K2={norm['dagostino_k2']:.1f}, p={norm['dagostino_p']:.2e}")
    print(f"Shapiro-Wilk (podpróba): W={norm['shapiro_w']:.3f}, p={norm['shapiro_p']:.2e}")
    print("-> rozkład NIE jest normalny => testy nieparametryczne uzasadnione")

    print("\n" + "=" * 70)
    print("3. KRUSKAL-WALLIS: różnice opóźnień między liniami")
    print("=" * 70)
    kw, dunn = kruskal_linie(df, output_dir)
    print(f"H={kw['H']:.1f}, df={kw['k']-1}, p={kw['p']:.2e}, epsilon^2={kw['eps2']:.3f}")
    print(f"Wynik: {'ISTOTNE' if kw['p'] < ALPHA else 'nieistotne'} "
          f"-> mediany opóźnień różnią się między liniami")
    print("\nPost-hoc Dunna (Bonferroni) — najistotniejsze pary (top %d linii):" % TOP_N_DUNN)
    print(dunn.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("4. TESTY CHI-KWADRAT NIEZALEŻNOŚCI (+ Craméra V)")
    print("=" * 70)
    chi = testy_chi2(df)
    for _, r in chi.iterrows():
        print(f"{r['test']:<32} chi2={r['chi2']:.1f}, dof={int(r['dof'])}, "
              f"p={r['p']:.2e}, V={r['cramers_v']:.3f} ({interpretuj_v(r['cramers_v'])})")

    print("\n" + "=" * 70)
    print("5. MANN-WHITNEY U: weekend vs dzień roboczy")
    print("=" * 70)
    mw = mann_whitney_weekend(df)
    print(f"Mediana roboczy={mw['mediana_roboczy']:.1f} (n={mw['n_roboczy']}), "
          f"weekend={mw['mediana_weekend']:.1f} (n={mw['n_weekend']})")
    print(f"U={mw['U']:.0f}, p={mw['p']:.2e}, rank-biserial={mw['rank_biserial']:.3f}")
    print(f"Wynik: {'ISTOTNE' if mw['p'] < ALPHA else 'nieistotne'} "
          f"(rozmiar efektu {interpretuj_rho(mw['rank_biserial'])})")

    print("\n" + "=" * 70)
    print("6. KORELACJE SPEARMANA")
    print("=" * 70)
    for nazwa, (rho, p) in korelacje_spearman(df).items():
        print(f"{nazwa:<24} rho={rho:.3f}, p={p:.2e} ({interpretuj_rho(rho)})")

    print("\n" + "=" * 70)
    print("7. MACIERZ KORELACJI -> 16_macierz_korelacji.png")
    print("=" * 70)
    macierz_korelacji(df, output_dir)
    print("zapisano")

    print("\n" + "=" * 70)
    print("8. BOOTSTRAP 95% CI średniego opóźnienia per linia")
    print("=" * 70)
    boot = bootstrap_linie(df, output_dir)
    print(boot.head(8).to_string(index=False))
    print("...")
    print(boot.tail(4).to_string(index=False))

    print(f"\nArtefakty zapisane w: {output_dir}")


if __name__ == "__main__":
    main()
