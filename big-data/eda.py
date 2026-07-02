import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

_TU = os.path.dirname(os.path.abspath(__file__))
DOMYSLNE_WEJSCIE = os.path.join(_TU, "dataset_eda_ready.csv")
FOLDER_WYKRESOW = os.path.join(_TU, "wykresy")

DNI_PL = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
PORY_DNIA = ["Rano", "Popołudnie", "Wieczór", "Noc"]
KATEGORIE_OPOZNIENIA = ["Wcześniej/punkt.", "1-15", "15-30", "30-60", "60+"]

DATY_SYNTETYCZNE = ["2026-04-18", "2026-04-19", "2026-04-21", "2026-04-22", "2026-04-23"]

TOP_N = 15        # ile linii pokazywać na wykresach per przewoźnik
MIN_LOTOW = 30    # próg minimalnej liczby lotów dla porównań


def read_data(file_path=DOMYSLNE_WEJSCIE):
    """Wczytuje zbiór EDA i przywraca uporządkowane typy kategoryczne"""
    df = pd.read_csv(file_path)
    df["data_lotu"] = pd.to_datetime(df["data_lotu"])
    df["pora_dnia"] = pd.Categorical(df["pora_dnia"], categories=PORY_DNIA, ordered=True)
    df["dzien_tygodnia"] = pd.Categorical(df["dzien_tygodnia"], categories=DNI_PL, ordered=True)
    df["kategoria_opoznienia"] = pd.Categorical(
        df["kategoria_opoznienia"], categories=KATEGORIE_OPOZNIENIA, ordered=True
    )
    return df


def _top_linie(df, top_n=TOP_N, min_lotow=MIN_LOTOW):
    """Zwraca kody top-N linii (wg liczby zrealizowanych lotów, próg min_lotow)."""
    zr = df[df["czy_odwolany"] == 0]
    liczby = zr["linia_lotnicza"].value_counts()
    return liczby[liczby >= min_lotow].head(top_n).index.tolist()


def _zapisz(fig_name, output_dir):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fig_name), dpi=120)
    plt.close()


#  1. Rozkład opóźnień (histogram + KDE)
def plot_01_rozklad_opoznien(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    plt.figure(figsize=(11, 6))
    sns.histplot(data=d, x="opoznienie_surowe", bins=80, kde=True, color="royalblue")
    plt.axvline(0, color="gray", ls="--", lw=1, label="Punktualnie (0 min)")
    plt.axvline(15, color="red", ls="--", lw=1, label="Próg opóźnienia (15 min)")
    plt.xlim(-60, 180)  # ucięcie skrajnych wartości odstających dla czytelności
    plt.title("Rozkład opóźnień odlotów (z KDE)", fontsize=14)
    plt.xlabel("Opóźnienie (min)")
    plt.ylabel("Liczba lotów")
    plt.legend()
    _zapisz("01_rozklad_opoznien.png", output_dir)


#  2. ECDF opóźnień
def plot_02_ecdf_opoznien(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    plt.figure(figsize=(11, 6))
    sns.ecdfplot(data=d, x="opoznienie_surowe", color="darkblue")
    plt.axvline(15, color="red", ls="--", lw=1, label="Próg 15 min")
    plt.xlim(-60, 180)
    plt.title("Dystrybuanta empiryczna (ECDF) opóźnień", fontsize=14)
    plt.xlabel("Opóźnienie (min)")
    plt.ylabel("Odsetek lotów ≤ x")
    plt.legend()
    _zapisz("02_ecdf_opoznien.png", output_dir)


#  3. Mediana opóźnienia wg linii (top-N, CI)
def plot_03_mediana_opoznienia_linii(df, output_dir):
    top = _top_linie(df)
    d = df[(df["czy_odwolany"] == 0) & (df["linia_lotnicza"].isin(top))]
    kolejnosc = (
        d.groupby("linia_lotnicza", observed=True)["opoznienie_minuty"]
        .median().sort_values(ascending=False).index
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(data=d, y="linia_lotnicza", x="opoznienie_minuty", order=kolejnosc,
                estimator=np.median, errorbar=("ci", 95), color="royalblue")
    plt.title(f"Mediana opóźnienia wg linii (top {len(top)} wg liczby lotów)", fontsize=14)
    plt.xlabel("Mediana opóźnienia (min, 95% CI)")
    plt.ylabel("Linia lotnicza")
    _zapisz("03_mediana_opoznienia_linii.png", output_dir)


#  4. Odsetek opóźnionych (%) wg linii (top-N)
def plot_04_odsetek_opoznionych_linii(df, output_dir):
    top = _top_linie(df)
    d = df[(df["czy_odwolany"] == 0) & (df["linia_lotnicza"].isin(top))]
    agg = (
        d.groupby("linia_lotnicza", observed=True)
        .agg(odsetek=("czy_opozniony", "mean"), n=("czy_opozniony", "size"))
        .sort_values("odsetek", ascending=False)
    )
    agg["odsetek"] *= 100
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=agg.reset_index(), y="linia_lotnicza", x="odsetek",
                     hue="linia_lotnicza", palette="Reds_r", legend=False)
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["odsetek"] + 0.5, i, f"n={int(row['n'])}", va="center", fontsize=8)
    plt.title(f"Odsetek lotów opóźnionych (>15 min) wg linii (min. {MIN_LOTOW} lotów)", fontsize=14)
    plt.xlabel("% lotów opóźnionych")
    plt.ylabel("Linia lotnicza")
    _zapisz("04_odsetek_opoznionych_linii.png", output_dir)


#  5. Boxplot opóźnień wg linii (top-N, z anomaliami)
def plot_05_boxplot_linii(df, output_dir):
    top = _top_linie(df)
    d = df[(df["czy_odwolany"] == 0) & (df["linia_lotnicza"].isin(top))]
    kolejnosc = (
        d.groupby("linia_lotnicza", observed=True)["opoznienie_surowe"]
        .median().sort_values(ascending=False).index
    )
    plt.figure(figsize=(13, 7))
    sns.boxplot(data=d, x="linia_lotnicza", y="opoznienie_surowe", order=kolejnosc,
                hue="linia_lotnicza", palette="Set3", legend=False, showfliers=True,
                flierprops={"marker": ".", "markersize": 3})
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.ylim(-40, 200)
    plt.title(f"Rozkład opóźnień wg linii (top {len(top)}, z wartościami odstającymi)", fontsize=14)
    plt.xlabel("Linia lotnicza")
    plt.ylabel("Opóźnienie surowe (min)")
    plt.xticks(rotation=45)
    _zapisz("05_boxplot_linii.png", output_dir)


#  6. Odsetek opóźnionych wg pory dnia
def plot_06_odsetek_opoznionych_pora(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    plt.figure(figsize=(9, 6))
    sns.pointplot(data=d, x="pora_dnia", y="czy_opozniony", errorbar=("ci", 95),
                  color="salmon", capsize=0.15)
    licz = d.groupby("pora_dnia", observed=True)["czy_opozniony"].agg(["mean", "size"])
    ax = plt.gca()
    for i, (_, row) in enumerate(licz.iterrows()):
        ax.text(i, row["mean"] + 0.01, f"n={int(row['size'])}", ha="center", fontsize=9)
    plt.title("Odsetek lotów opóźnionych (>15 min) wg pory dnia", fontsize=14)
    plt.xlabel("Pora dnia")
    plt.ylabel("Odsetek opóźnionych")
    _zapisz("06_odsetek_opoznionych_pora.png", output_dir)


#  7. Udział kategorii opóźnień wg pory dnia (stacked 100%)
def plot_07_udzial_kategorii_pora(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    tab = pd.crosstab(d["pora_dnia"], d["kategoria_opoznienia"], normalize="index") * 100
    tab = tab[KATEGORIE_OPOZNIENIA]
    ax = tab.plot(kind="bar", stacked=True, figsize=(10, 6),
                  colormap="RdYlGn_r", edgecolor="white")
    plt.title("Struktura kategorii opóźnień wg pory dnia (udział %)", fontsize=14)
    plt.xlabel("Pora dnia")
    plt.ylabel("Udział lotów (%)")
    plt.xticks(rotation=0)
    ax.legend(title="Kategoria opóźnienia", bbox_to_anchor=(1.02, 1), loc="upper left")
    _zapisz("07_udzial_kategorii_pora.png", output_dir)


#  8. Kaskadowość: rozkład opóźnień per godzina
def plot_08_kaskadowosc_godzina(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=d, x="godzina_planowana", y="opoznienie_minuty",
                color="steelblue", showfliers=False)
    mediany = d.groupby("godzina_planowana", observed=True)["opoznienie_minuty"].median()
    plt.plot(range(len(mediany)), mediany.values, color="darkred", marker="o",
             lw=2, label="Mediana opóźnienia")
    plt.title("Efekt kaskadowości: opóźnienia wg planowanej godziny odlotu", fontsize=14)
    plt.xlabel("Planowana godzina wylotu (0-23)")
    plt.ylabel("Opóźnienie (min)")
    plt.legend()
    _zapisz("08_kaskadowosc_godzina.png", output_dir)


#  9. Wolumen godzinowy vs mediana opóźnienia (dwie osie Y)
def plot_09_wolumen_vs_opoznienie(df, output_dir):
    d = df[df["czy_odwolany"] == 0]
    agg = d.groupby("godzina_planowana", observed=True).agg(
        wolumen=("opoznienie_minuty", "size"),
        mediana=("opoznienie_minuty", "median"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(agg["godzina_planowana"], agg["wolumen"], color="lightsteelblue",
            label="Liczba lotów")
    ax1.set_xlabel("Planowana godzina wylotu (0-23)")
    ax1.set_ylabel("Liczba lotów", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(range(0, 24))

    ax2 = ax1.twinx()
    ax2.plot(agg["godzina_planowana"], agg["mediana"], color="darkred",
             marker="o", lw=2, label="Mediana opóźnienia")
    ax2.set_ylabel("Mediana opóźnienia (min)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    plt.title("Obciążenie ruchem vs mediana opóźnienia wg godziny", fontsize=14)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper left")
    _zapisz("09_wolumen_vs_opoznienie.png", output_dir)


#  10. Heatmapa dzień tygodnia × godzina (mediana)
def plot_10_heatmapa_dzien_godzina(df, output_dir):
    d = df[(df["czy_odwolany"] == 0) & (df["czy_anomalia"] == 0)]
    pivot = d.pivot_table(index="dzien_tygodnia", columns="godzina_planowana",
                          values="opoznienie_minuty", aggfunc="median", observed=True)
    liczby = d.pivot_table(index="dzien_tygodnia", columns="godzina_planowana",
                           values="opoznienie_minuty", aggfunc="size", observed=True)
    pivot = pivot.mask(liczby < 5)  # maskowanie komórek o małej liczbie obserwacji
    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=.5,
                cbar_kws={"label": "Mediana opóźnienia (min)"})
    plt.title("Mediana opóźnienia wg dnia tygodnia i godziny (komórki <5 lotów ukryte)", fontsize=14)
    plt.xlabel("Godzina wylotu")
    plt.ylabel("Dzień tygodnia")
    _zapisz("10_heatmapa_dzien_godzina.png", output_dir)


#  11. Współczynnik odwołań wg kierunku (rate)
def plot_11_wspolczynnik_odwolan_kierunek(df, output_dir, top_n=10):
    agg = (
        df.groupby("kierunek")
        .agg(rate=("czy_odwolany", "mean"), n=("czy_odwolany", "size"))
        .query("n >= @MIN_LOTOW")
        .sort_values("rate", ascending=False)
        .head(top_n)
    )
    agg["rate"] *= 100
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agg.reset_index(), y="kierunek", x="rate",
                     hue="kierunek", palette="Reds_r", legend=False)
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["rate"] + 0.1, i, f"n={int(row['n'])}", va="center", fontsize=8)
    plt.title(f"Top {top_n} kierunków wg odsetka odwołań (min. {MIN_LOTOW} lotów)", fontsize=14)
    plt.xlabel("% odwołanych lotów")
    plt.ylabel("Kierunek")
    _zapisz("11_wspolczynnik_odwolan_kierunek.png", output_dir)


#  12. Współczynnik odwołań wg linii (rate)
def plot_12_wspolczynnik_odwolan_linii(df, output_dir):
    agg = (
        df.groupby("linia_lotnicza")
        .agg(rate=("czy_odwolany", "mean"), n=("czy_odwolany", "size"))
        .query("n >= @MIN_LOTOW")
        .sort_values("rate", ascending=False)
        .head(TOP_N)
    )
    agg["rate"] *= 100
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=agg.reset_index(), y="linia_lotnicza", x="rate",
                     hue="linia_lotnicza", palette="OrRd_r", legend=False)
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["rate"] + 0.05, i, f"n={int(row['n'])}", va="center", fontsize=8)
    plt.title(f"Odsetek odwołań wg linii (min. {MIN_LOTOW} lotów)", fontsize=14)
    plt.xlabel("% odwołanych lotów")
    plt.ylabel("Linia lotnicza")
    _zapisz("12_wspolczynnik_odwolan_linii.png", output_dir)


#  13. Top kierunki wg mediany opóźnienia
def plot_13_top_kierunki_opoznienie(df, output_dir, top_n=15):
    d = df[(df["czy_odwolany"] == 0) & (df["czy_anomalia"] == 0)]
    agg = (
        d.groupby("kierunek")
        .agg(mediana=("opoznienie_minuty", "median"), n=("opoznienie_minuty", "size"))
        .query("n >= @MIN_LOTOW")
        .sort_values("mediana", ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=agg.reset_index(), y="kierunek", x="mediana",
                     hue="kierunek", palette="viridis", legend=False)
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["mediana"] + 0.1, i, f"n={int(row['n'])}", va="center", fontsize=8)
    plt.title(f"Top {top_n} kierunków wg mediany opóźnienia (min. {MIN_LOTOW} lotów)", fontsize=14)
    plt.xlabel("Mediana opóźnienia (min)")
    plt.ylabel("Kierunek")
    _zapisz("13_top_kierunki_opoznienie.png", output_dir)


#  14. Opóźnienie vs odległość
def plot_14_opoznienie_vs_dystans(df, output_dir):
    d = df[(df["czy_odwolany"] == 0) & (df["czy_anomalia"] == 0)].dropna(subset=["dystans_km"])
    biny = pd.cut(d["dystans_km"], bins=range(0, int(d["dystans_km"].max()) + 250, 250))
    agg = d.groupby(biny, observed=True)["opoznienie_minuty"].median().reset_index()
    agg["srodek_km"] = agg["dystans_km"].apply(lambda x: x.mid)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=d, x="dystans_km", y="opoznienie_minuty",
                    alpha=0.15, color="steelblue", s=15)
    plt.plot(agg["srodek_km"], agg["opoznienie_minuty"], color="darkred",
             marker="o", lw=2, label="Mediana w przedziale 250 km")
    plt.ylim(0, 120)
    plt.title("Opóźnienie względem odległości trasy od KRK", fontsize=14)
    plt.xlabel("Dystans od Krakowa (km)")
    plt.ylabel("Opóźnienie (min)")
    plt.legend()
    _zapisz("14_opoznienie_vs_dystans.png", output_dir)


#  15. Szereg czasowy + krocząca mediana 7-dniowa
def plot_15_trend_czasowy(df, output_dir):
    trend = df.groupby("data_lotu").agg(
        mediana_opoznienia=("opoznienie_minuty", "median"),
        liczba_odwolan=("czy_odwolany", "sum"),
    ).reset_index()
    trend["mediana_7d"] = (
        trend["mediana_opoznienia"].rolling(7, center=True, min_periods=3).median()
    )

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(trend["data_lotu"], trend["mediana_opoznienia"], color="cornflowerblue",
             marker=".", lw=1, alpha=0.6, label="Mediana dzienna")
    ax1.plot(trend["data_lotu"], trend["mediana_7d"], color="navy", lw=2.5,
             label="Mediana krocząca 7 dni")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Mediana opóźnienia (min)", color="navy")
    ax1.tick_params(axis="y", labelcolor="navy")
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    for d in pd.to_datetime(DATY_SYNTETYCZNE):
        ax1.axvspan(d, d + pd.Timedelta(days=1), color="orange", alpha=0.12)

    ax2 = ax1.twinx()
    ax2.bar(trend["data_lotu"], trend["liczba_odwolan"], color="red", alpha=0.25,
            width=0.8, label="Liczba odwołań")
    ax2.set_ylabel("Liczba odwołanych lotów", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Szereg czasowy: mediana opóźnień i liczba odwołań (pomarańczowe = dane syntetyczne)",
              fontsize=13)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper left")
    _zapisz("15_trend_czasowy.png", output_dir)


WYKRESY = [
    plot_01_rozklad_opoznien,
    plot_02_ecdf_opoznien,
    plot_03_mediana_opoznienia_linii,
    plot_04_odsetek_opoznionych_linii,
    plot_05_boxplot_linii,
    plot_06_odsetek_opoznionych_pora,
    plot_07_udzial_kategorii_pora,
    plot_08_kaskadowosc_godzina,
    plot_09_wolumen_vs_opoznienie,
    plot_10_heatmapa_dzien_godzina,
    plot_11_wspolczynnik_odwolan_kierunek,
    plot_12_wspolczynnik_odwolan_linii,
    plot_13_top_kierunki_opoznienie,
    plot_14_opoznienie_vs_dystans,
    plot_15_trend_czasowy,
]


def main(file_path=DOMYSLNE_WEJSCIE, output_dir=FOLDER_WYKRESOW):
    os.makedirs(output_dir, exist_ok=True)
    print("Wczytywanie zbioru EDA...")
    df = read_data(file_path)
    print(f"  rekordów: {len(df)}")
    print("Generowanie wykresów...")
    for fn in WYKRESY:
        fn(df, output_dir)
        print(f"  [OK] {fn.__name__}")
    print(f"Gotowe! {len(WYKRESY)} wykresów zapisano w: {output_dir}")


if __name__ == "__main__":
    main()
