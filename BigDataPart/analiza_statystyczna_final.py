import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import scikit_posthocs as sp
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

_TU = os.path.dirname(os.path.abspath(__file__))
DOMYSLNE_WEJSCIE = os.path.join(_TU, "dataset_eda_ready.csv")
FOLDER_WYNIKOW = os.path.join(_TU, "analiza_finalna")

# stałe spójne z eda.py / preprocessing.py
DNI_PL = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
PORY_DNIA = ["Rano", "Popołudnie", "Wieczór", "Noc"]
KATEGORIE_OPOZNIENIA = ["Wcześniej/punkt.", "1-15", "15-30", "30-60", "60+"]

ALFA = 0.05
MIN_LOTOW = 30
N_BOOTSTRAP = 2000
SEED = 42

def wczytaj_dane(file_path=DOMYSLNE_WEJSCIE):
    """Wczytuje zbiór EDA i przywraca uporządkowane typy kategoryczne"""
    df = pd.read_csv(file_path)
    df["data_lotu"] = pd.to_datetime(df["data_lotu"])
    df["pora_dnia"] = pd.Categorical(df["pora_dnia"], categories=PORY_DNIA, ordered=True)
    df["dzien_tygodnia"] = pd.Categorical(df["dzien_tygodnia"], categories=DNI_PL, ordered=True)
    df["kategoria_opoznienia"] = pd.Categorical(
        df["kategoria_opoznienia"], categories=KATEGORIE_OPOZNIENIA, ordered=True
    )
    return df

def _werdykt(p, alfa=ALFA):
    return "istotny" if p < alfa else "nieistotny"


def _epsilon_kwadrat(H, n):
    """ε² dla testu Kruskala-Wallisa (Tomczak & Tomczak): ε² = H / (n - 1)."""
    return float(H / (n - 1)) if n > 1 else np.nan


def _opis_eps2(e):
    if e < 0.01:
        return "znikomy"
    if e < 0.06:
        return "mały"
    if e < 0.14:
        return "umiarkowany"
    return "duży"


def _opis_rbc(r):
    ar = abs(r)
    if ar < 0.1:
        return "znikomy"
    if ar < 0.3:
        return "mały"
    if ar < 0.5:
        return "średni"
    return "duży"


def _opis_cramera_v(v):
    if v < 0.1:
        return "znikomy"
    if v < 0.3:
        return "słaby"
    if v < 0.5:
        return "umiarkowany"
    return "silny"


def _opis_korelacji(r):
    ar = abs(r)
    if ar < 0.1:
        return "znikoma"
    if ar < 0.3:
        return "słaba"
    if ar < 0.5:
        return "umiarkowana"
    if ar < 0.7:
        return "silna"
    return "bardzo silna"


def _bootstrap_ci_mediana(x, n_resamples=N_BOOTSTRAP, seed=SEED):
    """95% przedział ufności dla MEDIANY metodą bootstrap (scipy)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan, np.nan
    res = stats.bootstrap((x,), np.median, confidence_level=0.95,
                          n_resamples=n_resamples, method="percentile",
                          random_state=seed)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _fmt_p(p):
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"


#  1. Badanie normalności (uzasadnienie metod rangowych)
def badanie_normalnosci(df, max_probka=5000, seed=SEED):
    zr = df[df["czy_odwolany"] == 0]
    x = zr["opoznienie_surowe"].dropna().to_numpy()

    rng = np.random.default_rng(seed)
    probka = rng.choice(x, size=min(max_probka, x.size), replace=False)
    norm = pg.normality(probka)  # Shapiro-Wilk: W, pval, normal

    return {
        "n": int(x.size),
        "skosnosc": float(stats.skew(x)),
        "kurtoza": float(stats.kurtosis(x)),
        "shapiro_W": float(norm["W"].iloc[0]),
        "shapiro_p": float(norm["pval"].iloc[0]),
        "n_probka": int(probka.size),
        "normalny": bool(norm["normal"].iloc[0]),
    }


#  2. Statystyki opisowe per linia (mediana, IQR, % opóźn./odwołań, bootstrap CI)
def statystyki_opisowe(df):
    zr = df[df["czy_odwolany"] == 0].copy()
    rekordy = []
    for linia, grupa in df.groupby("linia_lotnicza", observed=True):
        g_zr = grupa[grupa["czy_odwolany"] == 0]
        op = g_zr["opoznienie_minuty"].dropna()
        if op.size < MIN_LOTOW:
            continue
        ci_low, ci_high = _bootstrap_ci_mediana(op)
        rekordy.append({
            "linia": linia,
            "n": int(op.size),
            "mediana": round(float(op.median()), 1),
            "mediana_ci_low": round(ci_low, 1),
            "mediana_ci_high": round(ci_high, 1),
            "srednia": round(float(op.mean()), 1),
            "iqr": round(float(op.quantile(0.75) - op.quantile(0.25)), 1),
            "pct_opozn": round(float(g_zr["czy_opozniony"].mean() * 100), 1),
            "pct_odwol": round(float(grupa["czy_odwolany"].mean() * 100), 1),
        })
    tab = pd.DataFrame(rekordy).sort_values("mediana", ascending=False).reset_index(drop=True)

    ogolne = {
        "n": int(zr["opoznienie_minuty"].notna().sum()),
        "mediana": float(zr["opoznienie_minuty"].median()),
        "srednia": float(zr["opoznienie_minuty"].mean()),
    }
    return ogolne, tab


#  3. Kruskal-Wallis + ε² + post-hoc Dunna (zmienna grupująca kategoryczna)
def kruskal_z_dunnem(df, grupa_kol, ogranicz_min_lotow=False, top_dunn=10):
    """
    Kruskal-Wallis dla opoznienie_minuty wg grupa_kol (loty zrealizowane).
    Zwraca słownik wyniku + DataFrame najistotniejszych par post-hoc Dunna.
    """
    zr = df[df["czy_odwolany"] == 0].dropna(subset=["opoznienie_minuty", grupa_kol]).copy()
    if ogranicz_min_lotow:
        liczby = zr[grupa_kol].value_counts()
        zr = zr[zr[grupa_kol].isin(liczby[liczby >= MIN_LOTOW].index)]
    zr[grupa_kol] = zr[grupa_kol].astype(str)  # czyste etykiety dla Dunna

    kw = pg.kruskal(data=zr, dv="opoznienie_minuty", between=grupa_kol)
    H = float(kw["H"].iloc[0])
    p = float(kw["p_unc"].iloc[0])
    df_b = int(kw["ddof1"].iloc[0])
    n = int(zr.shape[0])
    eps2 = _epsilon_kwadrat(H, n)

    wynik = {
        "grupa": grupa_kol, "H": H, "df": df_b, "p": p, "n": n,
        "eps2": eps2, "rozmiar_efektu": _opis_eps2(eps2), "werdykt": _werdykt(p),
    }

    # post-hoc Dunna z korektą Bonferroniego -> macierz p; wyciągamy pary
    dunn = sp.posthoc_dunn(zr, val_col="opoznienie_minuty",
                           group_col=grupa_kol, p_adjust="bonferroni")
    mediany = zr.groupby(grupa_kol, observed=True)["opoznienie_minuty"].median()
    pary = []
    kol = list(dunn.columns)
    for i in range(len(kol)):
        for j in range(i + 1, len(kol)):
            a, b = kol[i], kol[j]
            pary.append({
                "para": f"{a} vs {b}",
                "mediana_a": round(float(mediany[a]), 1),
                "mediana_b": round(float(mediany[b]), 1),
                "p_bonferroni": float(dunn.loc[a, b]),
                "werdykt": _werdykt(dunn.loc[a, b]),
            })
    pary_df = pd.DataFrame(pary).sort_values("p_bonferroni").reset_index(drop=True)
    return wynik, pary_df.head(top_dunn), pary_df


#  4. Mann-Whitney U: weekend vs dzień roboczy (+ rank-biserial)
def mannwhitney_weekend(df):
    zr = df[df["czy_odwolany"] == 0]
    rob = zr.loc[zr["czy_weekend"] == 0, "opoznienie_minuty"].dropna()
    wek = zr.loc[zr["czy_weekend"] == 1, "opoznienie_minuty"].dropna()
    res = pg.mwu(rob, wek)  # U_val, p_val, RBC, CLES
    rbc = float(res["RBC"].iloc[0])
    p = float(res["p_val"].iloc[0])
    return {
        "n_roboczy": int(rob.size), "n_weekend": int(wek.size),
        "mediana_roboczy": float(rob.median()), "mediana_weekend": float(wek.median()),
        "U": float(res["U_val"].iloc[0]), "p": p,
        "rank_biserial": rbc, "rozmiar_efektu": _opis_rbc(rbc),
        "werdykt": _werdykt(p),
    }


#  5. Korelacje Spearmana (z 95% CI)
def korelacje_spearman(df):
    zr = df[df["czy_odwolany"] == 0]
    pary = [("godzina_planowana ↔ opóźnienie", "godzina_planowana"),
            ("dystans_km ↔ opóźnienie", "dystans_km")]
    rekordy = []
    for nazwa, kol in pary:
        d = zr.dropna(subset=[kol, "opoznienie_minuty"])
        res = pg.corr(d[kol], d["opoznienie_minuty"], method="spearman")
        r = float(res["r"].iloc[0])
        rekordy.append({
            "para": nazwa, "rho": round(r, 3),
            "ci95": str(res["CI95"].iloc[0]),
            "p": float(res["p_val"].iloc[0]),
            "interpretacja": _opis_korelacji(r), "werdykt": _werdykt(res["p_val"].iloc[0]),
        })
    return pd.DataFrame(rekordy)


#  6. Testy chi-kwadrat niezależności (+ V Craméra) — pingouin
def testy_chi_kwadrat(df):
    zr = df[df["czy_odwolany"] == 0]
    konfig = [
        ("pora_dnia × czy_opozniony", zr, "pora_dnia", "czy_opozniony"),
        ("czy_weekend × czy_opozniony", zr, "czy_weekend", "czy_opozniony"),
        ("linia × czy_odwolany", df, "linia_lotnicza", "czy_odwolany"),
    ]
    rekordy = []
    for nazwa, dane, x, y in konfig:
        d = dane.dropna(subset=[x, y]).copy()
        if x == "linia_lotnicza":
            duze = d["linia_lotnicza"].value_counts()
            d = d[d["linia_lotnicza"].isin(duze[duze >= MIN_LOTOW].index)]
        d[x] = d[x].astype(str)
        d[y] = d[y].astype(str)
        _, _, st = pg.chi2_independence(d, x=x, y=y)
        st = st[st["test"] == "pearson"].iloc[0]
        v = float(st["cramer"])
        rekordy.append({
            "test": nazwa, "chi2": round(float(st["chi2"]), 2), "dof": int(st["dof"]),
            "p": float(st["pval"]), "cramera_v": round(v, 3),
            "sila": _opis_cramera_v(v), "werdykt": _werdykt(st["pval"]),
        })
    return pd.DataFrame(rekordy)


#  7. Testy dla dwóch proporcji (statsmodels)
def testy_proporcji(df):
    rekordy = []
    zr = df[df["czy_odwolany"] == 0].dropna(subset=["czy_opozniony"])
    wek = zr[zr["czy_weekend"] == 1]["czy_opozniony"]
    rob = zr[zr["czy_weekend"] == 0]["czy_opozniony"]
    z, p = proportions_ztest([wek.sum(), rob.sum()], [len(wek), len(rob)])
    rekordy.append({
        "porownanie": "% opóźnionych: weekend vs roboczy",
        "pct_grupa1": round(wek.mean() * 100, 1), "n1": len(wek),
        "pct_grupa2": round(rob.mean() * 100, 1), "n2": len(rob),
        "z": round(float(z), 3), "p": float(p), "werdykt": _werdykt(p),
    })

    lh = df[df["linia_lotnicza"] == "LH"]["czy_odwolany"]
    reszta = df[df["linia_lotnicza"] != "LH"]["czy_odwolany"]
    if len(lh) > 0:
        z, p = proportions_ztest([lh.sum(), reszta.sum()], [len(lh), len(reszta)])
        rekordy.append({
            "porownanie": "% odwołań: LH vs pozostałe",
            "pct_grupa1": round(lh.mean() * 100, 1), "n1": len(lh),
            "pct_grupa2": round(reszta.mean() * 100, 1), "n2": len(reszta),
            "z": round(float(z), 3), "p": float(p), "werdykt": _werdykt(p),
        })
    return pd.DataFrame(rekordy)


#  8. Kontrola odporności: wersja parametryczna vs rangowa (flaga rozbieżności)
def kontrola_odpornosci(df):
    """
    Dla kluczowych pytań zestawia werdykt testu parametrycznego i rangowego.
    Flaga 'ROZBIEŻNOŚĆ' = testy dają różne werdykty -> ufaj wersji rangowej.
    """
    zr = df[df["czy_odwolany"] == 0]
    rekordy = []

    # weekend: t-Studenta vs Mann-Whitney
    rob = zr.loc[zr["czy_weekend"] == 0, "opoznienie_minuty"].dropna()
    wek = zr.loc[zr["czy_weekend"] == 1, "opoznienie_minuty"].dropna()
    _, p_t = stats.ttest_ind(rob, wek, equal_var=False)
    p_mwu = float(pg.mwu(rob, wek)["p_val"].iloc[0])
    rekordy.append(_porownaj("Weekend vs roboczy", "t-Studenta", p_t, "Mann-Whitney", p_mwu))

    # pora dnia: ANOVA vs Kruskal-Wallis
    grupy = [zr.loc[zr["pora_dnia"] == p, "opoznienie_minuty"].dropna() for p in PORY_DNIA]
    grupy = [g for g in grupy if g.size > 0]
    _, p_anova = stats.f_oneway(*grupy)
    _, p_kw = stats.kruskal(*grupy)
    rekordy.append(_porownaj("Pora dnia", "ANOVA", p_anova, "Kruskal-Wallis", p_kw))

    # korelacje: Pearson vs Spearman
    for nazwa, kol in [("Godzina ↔ opóźnienie", "godzina_planowana"),
                       ("Dystans ↔ opóźnienie", "dystans_km")]:
        d = zr.dropna(subset=[kol, "opoznienie_minuty"])
        _, p_pe = stats.pearsonr(d[kol], d["opoznienie_minuty"])
        _, p_sp = stats.spearmanr(d[kol], d["opoznienie_minuty"])
        rekordy.append(_porownaj(nazwa, "Pearson", p_pe, "Spearman", p_sp))

    return pd.DataFrame(rekordy)


def _porownaj(pytanie, nazwa_param, p_param, nazwa_rang, p_rang):
    w_param = _werdykt(p_param)
    w_rang = _werdykt(p_rang)
    return {
        "pytanie": pytanie,
        "test_parametryczny": f"{nazwa_param}: {w_param} (p={_fmt_p(p_param)})",
        "test_rangowy": f"{nazwa_rang}: {w_rang} (p={_fmt_p(p_rang)})",
        "rozbieznosc": "TAK — ufaj rangowemu" if w_param != w_rang else "nie",
    }


#  9. Wykrywanie wyników "istotnych, lecz trywialnych" (strażnik efektu)
def straznik_efektu(kw_linie, kw_pora, mwu, spearman_df, chi2_df):
    """Lista wyników istotnych statystycznie, ale o znikomym/małym rozmiarze efektu."""
    ostrzezenia = []
    if kw_linie["werdykt"] == "istotny" and kw_linie["eps2"] < 0.06:
        ostrzezenia.append(f"Kruskal (linie): istotny, lecz ε²={kw_linie['eps2']:.3f} "
                           f"({kw_linie['rozmiar_efektu']}) — efekt mały praktycznie.")
    if kw_pora["werdykt"] == "istotny" and kw_pora["eps2"] < 0.06:
        ostrzezenia.append(f"Kruskal (pora dnia): istotny, lecz ε²={kw_pora['eps2']:.3f} "
                           f"({kw_pora['rozmiar_efektu']}).")
    if mwu["werdykt"] == "istotny" and abs(mwu["rank_biserial"]) < 0.1:
        ostrzezenia.append(f"Mann-Whitney (weekend): istotny, lecz rank-biserial="
                           f"{mwu['rank_biserial']:.3f} (znikomy) — bez znaczenia praktycznego.")
    for _, r in spearman_df.iterrows():
        if r["werdykt"] == "istotny" and abs(r["rho"]) < 0.1:
            ostrzezenia.append(f"Spearman ({r['para']}): istotny, lecz ρ={r['rho']} (znikomy).")
    for _, r in chi2_df.iterrows():
        if r["werdykt"] == "istotny" and r["cramera_v"] < 0.1:
            ostrzezenia.append(f"Chi² ({r['test']}): istotny, lecz V={r['cramera_v']} (znikomy).")
    return ostrzezenia


#  Zapis wyników (CSV + zbiorczy markdown)
def zapisz_wyniki(output_dir, ogolne, opis, norm, kw_lin, dunn_lin_top, dunn_lin_all,
                  kw_pora, dunn_pora_top, mwu, spearman_df, chi2_df, prop_df,
                  odpornosc_df, ostrzezenia):
    os.makedirs(output_dir, exist_ok=True)

    opis.to_csv(os.path.join(output_dir, "statystyki_opisowe_linie.csv"), index=False)
    dunn_lin_all.to_csv(os.path.join(output_dir, "dunn_posthoc_linie.csv"), index=False)
    dunn_pora_top.to_csv(os.path.join(output_dir, "dunn_posthoc_pora.csv"), index=False)
    spearman_df.to_csv(os.path.join(output_dir, "korelacje_spearman.csv"), index=False)
    chi2_df.to_csv(os.path.join(output_dir, "chi_kwadrat.csv"), index=False)
    prop_df.to_csv(os.path.join(output_dir, "testy_proporcji.csv"), index=False)
    odpornosc_df.to_csv(os.path.join(output_dir, "kontrola_odpornosci.csv"), index=False)

    L = []
    L.append("# Wyniki: finalna analiza statystyczna (auto-generowane)\n")
    L.append(f"Poziom istotności α = {ALFA}. Loty zrealizowane: N = {ogolne['n']}. "
             f"Metoda: rangowa (opóźnienie) + chi-kwadrat (kategorie).\n")

    L.append("\n## 1. Normalność rozkładu (uzasadnienie metod rangowych)\n")
    L.append(f"- skośność = {norm['skosnosc']:.2f}, kurtoza = {norm['kurtoza']:.1f}")
    L.append(f"- Shapiro-Wilk (n={norm['n_probka']}): W = {norm['shapiro_W']:.3f}, "
             f"p = {_fmt_p(norm['shapiro_p'])} → rozkład "
             f"**{'normalny' if norm['normalny'] else 'NIE-normalny'}**\n")

    L.append("\n## 2. Statystyki opisowe per linia (mediana + bootstrap 95% CI)\n")
    L.append(f"Ogółem: mediana = {ogolne['mediana']:.1f} min, średnia = {ogolne['srednia']:.1f} min.\n")
    L.append("| Linia | N | mediana | 95% CI | IQR | % opóźn. | % odwołań |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in opis.iterrows():
        L.append(f"| {r['linia']} | {r['n']} | {r['mediana']} | "
                 f"[{r['mediana_ci_low']}; {r['mediana_ci_high']}] | {r['iqr']} | "
                 f"{r['pct_opozn']} | {r['pct_odwol']} |")

    L.append("\n\n## 3. Kruskal-Wallis: różnice między liniami\n")
    L.append(f"H = {kw_lin['H']:.1f}; df = {kw_lin['df']}; p = {_fmt_p(kw_lin['p'])}; "
             f"ε² = {kw_lin['eps2']:.3f} (efekt {kw_lin['rozmiar_efektu']}) → **{kw_lin['werdykt']}**\n")
    L.append("\nPost-hoc Dunna (Bonferroni), najistotniejsze pary:\n")
    L.append("| Para | mediana A | mediana B | p (Bonferroni) | werdykt |")
    L.append("|---|---|---|---|---|")
    for _, r in dunn_lin_top.iterrows():
        L.append(f"| {r['para']} | {r['mediana_a']} | {r['mediana_b']} | "
                 f"{_fmt_p(r['p_bonferroni'])} | {r['werdykt']} |")

    L.append("\n\n## 4. Kruskal-Wallis: różnice między porami dnia\n")
    L.append(f"H = {kw_pora['H']:.1f}; df = {kw_pora['df']}; p = {_fmt_p(kw_pora['p'])}; "
             f"ε² = {kw_pora['eps2']:.3f} (efekt {kw_pora['rozmiar_efektu']}) → **{kw_pora['werdykt']}**\n")
    L.append("\n| Para | mediana A | mediana B | p (Bonferroni) | werdykt |")
    L.append("|---|---|---|---|---|")
    for _, r in dunn_pora_top.iterrows():
        L.append(f"| {r['para']} | {r['mediana_a']} | {r['mediana_b']} | "
                 f"{_fmt_p(r['p_bonferroni'])} | {r['werdykt']} |")

    L.append("\n\n## 5. Mann-Whitney U: weekend vs dzień roboczy\n")
    L.append(f"- mediana: roboczy = {mwu['mediana_roboczy']:.1f} (n={mwu['n_roboczy']}) vs "
             f"weekend = {mwu['mediana_weekend']:.1f} (n={mwu['n_weekend']})")
    L.append(f"- U = {mwu['U']:.0f}; p = {_fmt_p(mwu['p'])} → **{mwu['werdykt']}**")
    L.append(f"- rank-biserial = {mwu['rank_biserial']:.3f} (efekt {mwu['rozmiar_efektu']})\n")

    L.append("\n## 6. Korelacje Spearmana\n")
    L.append("| Para | ρ | 95% CI | p | interpretacja | werdykt |")
    L.append("|---|---|---|---|---|---|")
    for _, r in spearman_df.iterrows():
        L.append(f"| {r['para']} | {r['rho']} | {r['ci95']} | {_fmt_p(r['p'])} | "
                 f"{r['interpretacja']} | {r['werdykt']} |")

    L.append("\n\n## 7. Testy chi-kwadrat niezależności\n")
    L.append("| Test | χ² | dof | p | V Craméra | siła | werdykt |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in chi2_df.iterrows():
        L.append(f"| {r['test']} | {r['chi2']} | {r['dof']} | {_fmt_p(r['p'])} | "
                 f"{r['cramera_v']} | {r['sila']} | {r['werdykt']} |")

    L.append("\n\n## 8. Testy dla dwóch proporcji\n")
    L.append("| Porównanie | grupa 1 (%) | grupa 2 (%) | z | p | werdykt |")
    L.append("|---|---|---|---|---|---|")
    for _, r in prop_df.iterrows():
        L.append(f"| {r['porownanie']} | {r['pct_grupa1']} | {r['pct_grupa2']} | "
                 f"{r['z']} | {_fmt_p(r['p'])} | {r['werdykt']} |")

    L.append("\n\n## 9. Kontrola odporności (parametryczny vs rangowy)\n")
    L.append("| Pytanie | Test parametryczny | Test rangowy | Rozbieżność |")
    L.append("|---|---|---|---|")
    for _, r in odpornosc_df.iterrows():
        L.append(f"| {r['pytanie']} | {r['test_parametryczny']} | "
                 f"{r['test_rangowy']} | {r['rozbieznosc']} |")

    L.append("\n\n## 10. Strażnik efektu — wyniki istotne, lecz praktycznie trywialne\n")
    if ostrzezenia:
        for o in ostrzezenia:
            L.append(f"- ⚠️ {o}")
    else:
        L.append("- Brak: każdy istotny wynik ma niezerowy rozmiar efektu.")

    with open(os.path.join(output_dir, "wyniki_finalne.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


#  Główny przebieg
def main(file_path=DOMYSLNE_WEJSCIE, output_dir=FOLDER_WYNIKOW):
    print("Wczytywanie zbioru...")
    df = wczytaj_dane(file_path)
    print(f"  rekordów: {len(df)} (odwołane: {int(df['czy_odwolany'].sum())})")

    print("\n[1/9] Normalność rozkładu...")
    norm = badanie_normalnosci(df)
    print(f"      skośność={norm['skosnosc']:.2f}, Shapiro p={_fmt_p(norm['shapiro_p'])} "
          f"→ metody RANGOWE uzasadnione")

    print("[2/9] Statystyki opisowe per linia (bootstrap CI median)...")
    ogolne, opis = statystyki_opisowe(df)
    print(f"      mediana ogólna = {ogolne['mediana']:.1f} min; linii (N≥{MIN_LOTOW}): {len(opis)}")

    print("[3/9] Kruskal-Wallis: linie + Dunn...")
    kw_lin, dunn_lin_top, dunn_lin_all = kruskal_z_dunnem(df, "linia_lotnicza",
                                                          ogranicz_min_lotow=True)
    print(f"      H={kw_lin['H']:.1f}, p={_fmt_p(kw_lin['p'])}, ε²={kw_lin['eps2']:.3f}")

    print("[4/9] Kruskal-Wallis: pora dnia + Dunn...")
    kw_pora, dunn_pora_top, _ = kruskal_z_dunnem(df, "pora_dnia")
    print(f"      H={kw_pora['H']:.1f}, p={_fmt_p(kw_pora['p'])}, ε²={kw_pora['eps2']:.3f}")

    print("[5/9] Mann-Whitney: weekend vs roboczy...")
    mwu = mannwhitney_weekend(df)
    print(f"      p={_fmt_p(mwu['p'])}, rank-biserial={mwu['rank_biserial']:.3f} → {mwu['werdykt']}")

    print("[6/9] Korelacje Spearmana...")
    spearman_df = korelacje_spearman(df)
    for _, r in spearman_df.iterrows():
        print(f"      {r['para']}: ρ={r['rho']} ({r['interpretacja']})")

    print("[7/9] Testy chi-kwadrat...")
    chi2_df = testy_chi_kwadrat(df)
    for _, r in chi2_df.iterrows():
        print(f"      {r['test']}: V={r['cramera_v']} ({r['sila']})")

    print("[8/9] Testy proporcji + kontrola odporności...")
    prop_df = testy_proporcji(df)
    odpornosc_df = kontrola_odpornosci(df)
    n_rozb = (odpornosc_df["rozbieznosc"].str.startswith("TAK")).sum()
    print(f"      rozbieżności parametryczny↔rangowy: {n_rozb}")

    print("[9/9] Strażnik efektu...")
    ostrzezenia = straznik_efektu(kw_lin, kw_pora, mwu, spearman_df, chi2_df)
    print(f"      wyników istotnych-lecz-trywialnych: {len(ostrzezenia)}")

    print("\nZapis wyników...")
    zapisz_wyniki(output_dir, ogolne, opis, norm, kw_lin, dunn_lin_top, dunn_lin_all,
                  kw_pora, dunn_pora_top, mwu, spearman_df, chi2_df, prop_df,
                  odpornosc_df, ostrzezenia)
    print(f"Gotowe! Wyniki zapisano w: {output_dir}")
    print(f"  -> wyniki_finalne.md (zbiorcze) + pliki CSV")


if __name__ == "__main__":
    main()