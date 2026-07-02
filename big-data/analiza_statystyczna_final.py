import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import scikit_posthocs as sp
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DOMYSLNE_WEJSCIE = os.path.join(HERE, "dataset_eda_ready.csv")
FOLDER_WYNIKOW = os.path.join(HERE, "analiza_finalna")

DNI_PL = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
PORY_DNIA = ["Rano", "Popołudnie", "Wieczór", "Noc"]
KATEGORIE_OPOZNIENIA = ["Wcześniej/punkt.", "1-15", "15-30", "30-60", "60+"]

ALFA = 0.05
MIN_LOTOW = 30          # ponizej tego nie porownuje linii - za male proby
N_BOOTSTRAP = 2000
SEED = 42


def wczytaj_dane(file_path=DOMYSLNE_WEJSCIE):
    # CSV gubi typy kategoryczne, wiec przywracam je recznie
    df = pd.read_csv(file_path)
    df["data_lotu"] = pd.to_datetime(df["data_lotu"])
    df["pora_dnia"] = pd.Categorical(df["pora_dnia"], categories=PORY_DNIA, ordered=True)
    df["dzien_tygodnia"] = pd.Categorical(df["dzien_tygodnia"], categories=DNI_PL, ordered=True)
    df["kategoria_opoznienia"] = pd.Categorical(
        df["kategoria_opoznienia"], categories=KATEGORIE_OPOZNIENIA, ordered=True)
    return df


def werdykt(p):
    return "istotny" if p < ALFA else "nieistotny"


def fmt_p(p):
    # bardzo male p ladniej wyglada w notacji wykladniczej
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"


# progi rozmiarow efektu (Cohen / Tomczak)

def opis_eps2(e):
    if e < 0.01: return "znikomy"
    if e < 0.06: return "mały"
    if e < 0.14: return "umiarkowany"
    return "duży"


def opis_rbc(r):
    r = abs(r)
    if r < 0.1: return "znikomy"
    if r < 0.3: return "mały"
    if r < 0.5: return "średni"
    return "duży"


def opis_v(v):
    if v < 0.1: return "znikomy"
    if v < 0.3: return "słaby"
    if v < 0.5: return "umiarkowany"
    return "silny"


def opis_rho(r):
    r = abs(r)
    if r < 0.1: return "znikoma"
    if r < 0.3: return "słaba"
    if r < 0.5: return "umiarkowana"
    if r < 0.7: return "silna"
    return "bardzo silna"


def bootstrap_ci_mediana(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan, np.nan
    res = stats.bootstrap((x,), np.median, confidence_level=0.95,
                          n_resamples=N_BOOTSTRAP, method="percentile", random_state=SEED)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def badanie_normalnosci(df, max_probka=5000):
    # Shapiro jest przeczulony przy duzym N, wiec losuje podprobe
    x = df.loc[df["czy_odwolany"] == 0, "opoznienie_surowe"].dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    probka = rng.choice(x, size=min(max_probka, x.size), replace=False)
    norm = pg.normality(probka)
    return {
        "n": int(x.size),
        "skosnosc": float(stats.skew(x)),
        "kurtoza": float(stats.kurtosis(x)),
        "shapiro_W": float(norm["W"].iloc[0]),
        "shapiro_p": float(norm["pval"].iloc[0]),
        "n_probka": int(probka.size),
        "normalny": bool(norm["normal"].iloc[0]),
    }


def statystyki_opisowe(df):
    rekordy = []
    for linia, grupa in df.groupby("linia_lotnicza", observed=True):
        zrealizowane = grupa[grupa["czy_odwolany"] == 0]
        op = zrealizowane["opoznienie_minuty"].dropna()
        if op.size < MIN_LOTOW:
            continue
        ci_low, ci_high = bootstrap_ci_mediana(op)
        rekordy.append({
            "linia": linia,
            "n": int(op.size),
            "mediana": round(float(op.median()), 1),
            "mediana_ci_low": round(ci_low, 1),
            "mediana_ci_high": round(ci_high, 1),
            "srednia": round(float(op.mean()), 1),
            "iqr": round(float(op.quantile(0.75) - op.quantile(0.25)), 1),
            "pct_opozn": round(float(zrealizowane["czy_opozniony"].mean() * 100), 1),
            "pct_odwol": round(float(grupa["czy_odwolany"].mean() * 100), 1),
        })
    tab = pd.DataFrame(rekordy).sort_values("mediana", ascending=False).reset_index(drop=True)

    zr = df[df["czy_odwolany"] == 0]["opoznienie_minuty"]
    ogolne = {
        "n": int(zr.notna().sum()),
        "mediana": float(zr.median()),
        "srednia": float(zr.mean()),
    }
    return ogolne, tab


def kruskal_z_dunnem(df, grupa_kol, ogranicz_min_lotow=False, top_dunn=10):
    zr = df[df["czy_odwolany"] == 0].dropna(subset=["opoznienie_minuty", grupa_kol]).copy()
    if ogranicz_min_lotow:
        liczby = zr[grupa_kol].value_counts()
        zr = zr[zr[grupa_kol].isin(liczby[liczby >= MIN_LOTOW].index)]
    zr[grupa_kol] = zr[grupa_kol].astype(str)

    kw = pg.kruskal(data=zr, dv="opoznienie_minuty", between=grupa_kol)
    H = float(kw["H"].iloc[0])
    p = float(kw["p_unc"].iloc[0])
    n = len(zr)
    eps2 = H / (n - 1)          # Tomczak & Tomczak

    wynik = {
        "grupa": grupa_kol, "H": H, "df": int(kw["ddof1"].iloc[0]), "p": p, "n": n,
        "eps2": eps2, "rozmiar_efektu": opis_eps2(eps2), "werdykt": werdykt(p),
    }

    dunn = sp.posthoc_dunn(zr, val_col="opoznienie_minuty",
                           group_col=grupa_kol, p_adjust="bonferroni")
    mediany = zr.groupby(grupa_kol, observed=True)["opoznienie_minuty"].median()

    pary = []
    nazwy = list(dunn.columns)
    for i in range(len(nazwy)):
        for j in range(i + 1, len(nazwy)):
            a, b = nazwy[i], nazwy[j]
            pary.append({
                "para": f"{a} vs {b}",
                "mediana_a": round(float(mediany[a]), 1),
                "mediana_b": round(float(mediany[b]), 1),
                "p_bonferroni": float(dunn.loc[a, b]),
                "werdykt": werdykt(dunn.loc[a, b]),
            })
    pary = pd.DataFrame(pary).sort_values("p_bonferroni").reset_index(drop=True)
    return wynik, pary.head(top_dunn), pary


def mannwhitney_weekend(df):
    zr = df[df["czy_odwolany"] == 0]
    rob = zr.loc[zr["czy_weekend"] == 0, "opoznienie_minuty"].dropna()
    wek = zr.loc[zr["czy_weekend"] == 1, "opoznienie_minuty"].dropna()
    res = pg.mwu(rob, wek)
    rbc = float(res["RBC"].iloc[0])
    p = float(res["p_val"].iloc[0])
    return {
        "n_roboczy": len(rob), "n_weekend": len(wek),
        "mediana_roboczy": float(rob.median()), "mediana_weekend": float(wek.median()),
        "U": float(res["U_val"].iloc[0]), "p": p,
        "rank_biserial": rbc, "rozmiar_efektu": opis_rbc(rbc), "werdykt": werdykt(p),
    }


def korelacje_spearman(df):
    zr = df[df["czy_odwolany"] == 0]
    rekordy = []
    for nazwa, kol in [("godzina_planowana ↔ opóźnienie", "godzina_planowana"),
                       ("dystans_km ↔ opóźnienie", "dystans_km")]:
        d = zr.dropna(subset=[kol, "opoznienie_minuty"])
        res = pg.corr(d[kol], d["opoznienie_minuty"], method="spearman")
        rho = float(res["r"].iloc[0])
        p = float(res["p_val"].iloc[0])
        rekordy.append({
            "para": nazwa, "rho": round(rho, 3), "ci95": str(res["CI95"].iloc[0]),
            "p": p, "interpretacja": opis_rho(rho), "werdykt": werdykt(p),
        })
    return pd.DataFrame(rekordy)


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
        st = pg.chi2_independence(d, x=x, y=y)[2]
        st = st[st["test"] == "pearson"].iloc[0]
        v = float(st["cramer"])
        rekordy.append({
            "test": nazwa, "chi2": round(float(st["chi2"]), 2), "dof": int(st["dof"]),
            "p": float(st["pval"]), "cramera_v": round(v, 3),
            "sila": opis_v(v), "werdykt": werdykt(st["pval"]),
        })
    return pd.DataFrame(rekordy)


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
        "z": round(float(z), 3), "p": float(p), "werdykt": werdykt(p),
    })

    lh = df[df["linia_lotnicza"] == "LH"]["czy_odwolany"]
    reszta = df[df["linia_lotnicza"] != "LH"]["czy_odwolany"]
    if len(lh):
        z, p = proportions_ztest([lh.sum(), reszta.sum()], [len(lh), len(reszta)])
        rekordy.append({
            "porownanie": "% odwołań: LH vs pozostałe",
            "pct_grupa1": round(lh.mean() * 100, 1), "n1": len(lh),
            "pct_grupa2": round(reszta.mean() * 100, 1), "n2": len(reszta),
            "z": round(float(z), 3), "p": float(p), "werdykt": werdykt(p),
        })
    return pd.DataFrame(rekordy)


def _porownaj(pytanie, nazwa_param, p_param, nazwa_rang, p_rang):
    w1, w2 = werdykt(p_param), werdykt(p_rang)
    return {
        "pytanie": pytanie,
        "test_parametryczny": f"{nazwa_param}: {w1} (p={fmt_p(p_param)})",
        "test_rangowy": f"{nazwa_rang}: {w2} (p={fmt_p(p_rang)})",
        "rozbieznosc": "TAK — ufaj rangowemu" if w1 != w2 else "nie",
    }


def kontrola_odpornosci(df):
    # Sedno: licze obie wersje testu i sprawdzam, czy daja te sama odpowiedz.
    # Jak sie roznia (jak na weekendzie) - winna jest niestabilna srednia.
    zr = df[df["czy_odwolany"] == 0]
    out = []

    rob = zr.loc[zr["czy_weekend"] == 0, "opoznienie_minuty"].dropna()
    wek = zr.loc[zr["czy_weekend"] == 1, "opoznienie_minuty"].dropna()
    p_t = stats.ttest_ind(rob, wek, equal_var=False).pvalue
    p_mwu = float(pg.mwu(rob, wek)["p_val"].iloc[0])
    out.append(_porownaj("Weekend vs roboczy", "t-Studenta", p_t, "Mann-Whitney", p_mwu))

    grupy = [zr.loc[zr["pora_dnia"] == p, "opoznienie_minuty"].dropna() for p in PORY_DNIA]
    grupy = [g for g in grupy if len(g)]
    p_anova = stats.f_oneway(*grupy).pvalue
    p_kw = stats.kruskal(*grupy).pvalue
    out.append(_porownaj("Pora dnia", "ANOVA", p_anova, "Kruskal-Wallis", p_kw))

    for nazwa, kol in [("Godzina ↔ opóźnienie", "godzina_planowana"),
                       ("Dystans ↔ opóźnienie", "dystans_km")]:
        d = zr.dropna(subset=[kol, "opoznienie_minuty"])
        p_pe = stats.pearsonr(d[kol], d["opoznienie_minuty"])[1]
        p_sp = stats.spearmanr(d[kol], d["opoznienie_minuty"])[1]
        out.append(_porownaj(nazwa, "Pearson", p_pe, "Spearman", p_sp))

    return pd.DataFrame(out)


def straznik_efektu(kw_lin, kw_pora, mwu, spearman_df, chi2_df):
    # wylapuje wyniki istotne, ale z tak malym efektem, ze praktycznie nic nie znacza
    flagi = []
    if kw_lin["werdykt"] == "istotny" and kw_lin["eps2"] < 0.06:
        flagi.append(f"Kruskal (linie): istotny, ale ε²={kw_lin['eps2']:.3f} ({kw_lin['rozmiar_efektu']}).")
    if kw_pora["werdykt"] == "istotny" and kw_pora["eps2"] < 0.06:
        flagi.append(f"Kruskal (pora dnia): istotny, ale ε²={kw_pora['eps2']:.3f} ({kw_pora['rozmiar_efektu']}).")
    if mwu["werdykt"] == "istotny" and abs(mwu["rank_biserial"]) < 0.1:
        flagi.append(f"Mann-Whitney (weekend): istotny, ale rank-biserial={mwu['rank_biserial']:.3f} (znikomy).")
    for _, r in spearman_df.iterrows():
        if r["werdykt"] == "istotny" and abs(r["rho"]) < 0.1:
            flagi.append(f"Spearman ({r['para']}): istotny, ale ρ={r['rho']} (znikomy).")
    for _, r in chi2_df.iterrows():
        if r["werdykt"] == "istotny" and r["cramera_v"] < 0.1:
            flagi.append(f"Chi² ({r['test']}): istotny, ale V={r['cramera_v']} (znikomy).")
    return flagi


def zapisz_wyniki(output_dir, ogolne, opis, norm, kw_lin, dunn_lin_top, dunn_lin_all,
                  kw_pora, dunn_pora_top, mwu, spearman_df, chi2_df, prop_df,
                  odpornosc_df, flagi):
    os.makedirs(output_dir, exist_ok=True)

    opis.to_csv(os.path.join(output_dir, "statystyki_opisowe_linie.csv"), index=False)
    dunn_lin_all.to_csv(os.path.join(output_dir, "dunn_posthoc_linie.csv"), index=False)
    dunn_pora_top.to_csv(os.path.join(output_dir, "dunn_posthoc_pora.csv"), index=False)
    spearman_df.to_csv(os.path.join(output_dir, "korelacje_spearman.csv"), index=False)
    chi2_df.to_csv(os.path.join(output_dir, "chi_kwadrat.csv"), index=False)
    prop_df.to_csv(os.path.join(output_dir, "testy_proporcji.csv"), index=False)
    odpornosc_df.to_csv(os.path.join(output_dir, "kontrola_odpornosci.csv"), index=False)

    def tabela(naglowek, df_out, kolumny):
        wiersze = ["| " + " | ".join(naglowek) + " |", "|" + "---|" * len(naglowek)]
        for _, r in df_out.iterrows():
            wiersze.append("| " + " | ".join(str(r[k]) for k in kolumny) + " |")
        return "\n".join(wiersze)

    L = ["# Wyniki: finalna analiza statystyczna (auto-generowane)\n",
         f"Poziom istotności α = {ALFA}. Loty zrealizowane: N = {ogolne['n']}. "
         f"Metoda: rangowa (opóźnienie) + chi-kwadrat (kategorie).\n"]

    L.append("\n## 1. Normalność rozkładu (uzasadnienie metod rangowych)\n")
    L.append(f"- skośność = {norm['skosnosc']:.2f}, kurtoza = {norm['kurtoza']:.1f}")
    L.append(f"- Shapiro-Wilk (n={norm['n_probka']}): W = {norm['shapiro_W']:.3f}, "
             f"p = {fmt_p(norm['shapiro_p'])} → rozkład "
             f"**{'normalny' if norm['normalny'] else 'NIE-normalny'}**\n")

    L.append("\n## 2. Statystyki opisowe per linia (mediana + bootstrap 95% CI)\n")
    L.append(f"Ogółem: mediana = {ogolne['mediana']:.1f} min, średnia = {ogolne['srednia']:.1f} min.\n")
    opis_md = opis.assign(ci=lambda d: "[" + d["mediana_ci_low"].astype(str) + "; "
                          + d["mediana_ci_high"].astype(str) + "]")
    L.append(tabela(["Linia", "N", "mediana", "95% CI", "IQR", "% opóźn.", "% odwołań"],
                    opis_md, ["linia", "n", "mediana", "ci", "iqr", "pct_opozn", "pct_odwol"]))

    L.append("\n\n## 3. Kruskal-Wallis: różnice między liniami\n")
    L.append(f"H = {kw_lin['H']:.1f}; df = {kw_lin['df']}; p = {fmt_p(kw_lin['p'])}; "
             f"ε² = {kw_lin['eps2']:.3f} (efekt {kw_lin['rozmiar_efektu']}) → **{kw_lin['werdykt']}**\n")
    L.append("\nPost-hoc Dunna (Bonferroni), najistotniejsze pary:\n")
    dl = dunn_lin_top.assign(p=lambda d: d["p_bonferroni"].map(fmt_p))
    L.append(tabela(["Para", "mediana A", "mediana B", "p (Bonferroni)", "werdykt"],
                    dl, ["para", "mediana_a", "mediana_b", "p", "werdykt"]))

    L.append("\n\n## 4. Kruskal-Wallis: różnice między porami dnia\n")
    L.append(f"H = {kw_pora['H']:.1f}; df = {kw_pora['df']}; p = {fmt_p(kw_pora['p'])}; "
             f"ε² = {kw_pora['eps2']:.3f} (efekt {kw_pora['rozmiar_efektu']}) → **{kw_pora['werdykt']}**\n")
    dp = dunn_pora_top.assign(p=lambda d: d["p_bonferroni"].map(fmt_p))
    L.append("\n" + tabela(["Para", "mediana A", "mediana B", "p (Bonferroni)", "werdykt"],
                           dp, ["para", "mediana_a", "mediana_b", "p", "werdykt"]))

    L.append("\n\n## 5. Mann-Whitney U: weekend vs dzień roboczy\n")
    L.append(f"- mediana: roboczy = {mwu['mediana_roboczy']:.1f} (n={mwu['n_roboczy']}) vs "
             f"weekend = {mwu['mediana_weekend']:.1f} (n={mwu['n_weekend']})")
    L.append(f"- U = {mwu['U']:.0f}; p = {fmt_p(mwu['p'])} → **{mwu['werdykt']}**")
    L.append(f"- rank-biserial = {mwu['rank_biserial']:.3f} (efekt {mwu['rozmiar_efektu']})\n")

    L.append("\n## 6. Korelacje Spearmana\n")
    L.append(tabela(["Para", "ρ", "95% CI", "p", "interpretacja", "werdykt"],
                    spearman_df.assign(p=lambda d: d["p"].map(fmt_p)),
                    ["para", "rho", "ci95", "p", "interpretacja", "werdykt"]))

    L.append("\n\n## 7. Testy chi-kwadrat niezależności\n")
    L.append(tabela(["Test", "χ²", "dof", "p", "V Craméra", "siła", "werdykt"],
                    chi2_df.assign(p=lambda d: d["p"].map(fmt_p)),
                    ["test", "chi2", "dof", "p", "cramera_v", "sila", "werdykt"]))

    L.append("\n\n## 8. Testy dla dwóch proporcji\n")
    L.append(tabela(["Porównanie", "grupa 1 (%)", "grupa 2 (%)", "z", "p", "werdykt"],
                    prop_df.assign(p=lambda d: d["p"].map(fmt_p)),
                    ["porownanie", "pct_grupa1", "pct_grupa2", "z", "p", "werdykt"]))

    L.append("\n\n## 9. Kontrola odporności (parametryczny vs rangowy)\n")
    L.append(tabela(["Pytanie", "Test parametryczny", "Test rangowy", "Rozbieżność"],
                    odpornosc_df, ["pytanie", "test_parametryczny", "test_rangowy", "rozbieznosc"]))

    L.append("\n\n## 10. Strażnik efektu — wyniki istotne, lecz praktycznie trywialne\n")
    if flagi:
        L += [f"- {f}" for f in flagi]
    else:
        L.append("- Brak: każdy istotny wynik ma niezerowy rozmiar efektu.")

    with open(os.path.join(output_dir, "wyniki_finalne.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(file_path=DOMYSLNE_WEJSCIE, output_dir=FOLDER_WYNIKOW):
    df = wczytaj_dane(file_path)
    print(f"Wczytano {len(df)} lotów ({int(df['czy_odwolany'].sum())} odwołanych)\n")

    norm = badanie_normalnosci(df)
    print(f"Normalność: skośność={norm['skosnosc']:.2f}, kurtoza={norm['kurtoza']:.0f}, "
          f"Shapiro p={fmt_p(norm['shapiro_p'])}")

    ogolne, opis = statystyki_opisowe(df)
    print(f"Opis: mediana ogólna {ogolne['mediana']:.1f} min, {len(opis)} linii (N>={MIN_LOTOW})")

    kw_lin, dunn_lin_top, dunn_lin_all = kruskal_z_dunnem(df, "linia_lotnicza", ogranicz_min_lotow=True)
    print(f"Kruskal linie: H={kw_lin['H']:.1f}, p={fmt_p(kw_lin['p'])}, eps2={kw_lin['eps2']:.3f}")

    kw_pora, dunn_pora_top, _ = kruskal_z_dunnem(df, "pora_dnia")
    print(f"Kruskal pora: H={kw_pora['H']:.1f}, p={fmt_p(kw_pora['p'])}, eps2={kw_pora['eps2']:.3f}")

    mwu = mannwhitney_weekend(df)
    print(f"Mann-Whitney weekend: p={fmt_p(mwu['p'])}, rbc={mwu['rank_biserial']:.3f}")

    spearman_df = korelacje_spearman(df)
    chi2_df = testy_chi_kwadrat(df)
    prop_df = testy_proporcji(df)
    odpornosc_df = kontrola_odpornosci(df)
    flagi = straznik_efektu(kw_lin, kw_pora, mwu, spearman_df, chi2_df)

    rozbieznosci = odpornosc_df["rozbieznosc"].str.startswith("TAK").sum()
    print(f"Rozbieżności param/rangowy: {rozbieznosci}; wyników trywialnych: {len(flagi)}")

    zapisz_wyniki(output_dir, ogolne, opis, norm, kw_lin, dunn_lin_top, dunn_lin_all,
                  kw_pora, dunn_pora_top, mwu, spearman_df, chi2_df, prop_df, odpornosc_df, flagi)
    print(f"\nZapisano w {output_dir} (wyniki_finalne.md + CSV)")


if __name__ == "__main__":
    main()