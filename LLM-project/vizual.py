import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

DB_PATH = "baza_lotow.db.link"
PROG_OPOZNIENIA = 15
PROG_OPOZNIENIA_PROCENT_LOTY = 20
PROG_OPOZNIENIA_PROCENT_LINIE = 25

plt.rcParams.update({
    "figure.facecolor": "#F8F7F4",
    "axes.facecolor":   "#F8F7F4",
    "axes.edgecolor":   "#CCCBC3",
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.color":       "#E0DED7",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "xtick.color":      "#5F5E5A",
    "ytick.color":      "#5F5E5A",
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

KOLOR_OK       = "#1D9E75"   # zielony – na czas
KOLOR_MALY     = "#EF9F27"   # żółty – małe opóźnienie
KOLOR_DUZY     = "#D85A30"   # pomarańczowy – duże opóźnienie
KOLOR_LINIA    = "#378ADD"   # niebieski – linia trendu
KOLOR_PROG     = "#E24B4A"   # czerwony – linia progu

def wczytaj_dane(db_path: str):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM loty_odloty", conn)
    conn.close()

    df["data_lotu"]        = pd.to_datetime(df["data_lotu"])
    df["czas_planowany"]   = pd.to_datetime(df["czas_planowany"])
    df["czas_rzeczywisty"] = pd.to_datetime(df["czas_rzeczywisty"])

    df["opoznienie"] = (
        (df["czas_rzeczywisty"] - df["czas_planowany"])
        .dt.total_seconds() / 60
    )
    df["godzina"]   = df["czas_planowany"].dt.hour
    df["dzien_tyg"] = df["czas_planowany"].dt.dayofweek  # 0 = Pon

    df["odwołany"] = df["status"].str.startswith("Odwołany")
    df_valid = df[~df["odwołany"]].copy()
    df_valid["delayed"] = (df_valid["opoznienie"] >= PROG_OPOZNIENIA).astype(int)

    return df, df_valid


def wykres_rozklad(df_valid: pd.DataFrame):
    bins   = [-200, -1, 0, 5, 15, 30, 60, 200, 1000]
    labels = ["< 0", "0", "1–5", "6–15", "16–30", "31–60", "61–200", "> 200"]
    kolory = [KOLOR_OK, KOLOR_OK, KOLOR_OK, KOLOR_MALY,
              KOLOR_DUZY, KOLOR_DUZY, KOLOR_DUZY, KOLOR_DUZY]

    counts = pd.cut(df_valid["opoznienie"], bins=bins, labels=labels).value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts.values, color=kolory, edgecolor="white",
                  linewidth=0.8, zorder=3, width=0.7)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(val), ha="center", va="bottom", fontsize=9, color="#5F5E5A")

    ax.set_xlabel("Opóźnienie (minuty)", labelpad=8)
    ax.set_ylabel("Liczba lotów", labelpad=8)
    ax.set_title("Rozkład opóźnień lotów – KRK", fontsize=14, fontweight="bold",
                 color="#2C2C2A", pad=12)

    legend_patches = [
        mpatches.Patch(color=KOLOR_OK,   label="Na czas / wcześniej"),
        mpatches.Patch(color=KOLOR_MALY, label="Małe opóźnienie (1–15 min)"),
        mpatches.Patch(color=KOLOR_DUZY, label="Duże opóźnienie (> 15 min)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.7,
              edgecolor="#CCCBC3", loc="upper right")

    fig.tight_layout()
    fig.savefig("wykres1_rozklad_opoznien.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres1_rozklad_opoznien.png")


def wykres_godzina(df_valid: pd.DataFrame):
    by_hour = (
        df_valid.groupby("godzina")["delayed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_hour = by_hour[by_hour["godzina"] >= 5].copy()
    by_hour["pct"] = by_hour["mean"] * 100

    kolory = [KOLOR_DUZY if p >= 35 else KOLOR_MALY if p >= 20 else KOLOR_OK
              for p in by_hour["pct"]]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(by_hour["godzina"], by_hour["pct"],
                   color=kolory, alpha=0.85, zorder=3, width=0.7,
                   edgecolor="white", linewidth=0.8)
    ax2.plot(by_hour["godzina"], by_hour["count"],
             color="#888780", linewidth=1.5, linestyle=":",
             marker="o", markersize=4, zorder=4, label="Liczba lotów")

    ax1.axhline(y=PROG_OPOZNIENIA_PROCENT_LOTY, color=KOLOR_PROG, linewidth=1.2,
                linestyle="--", zorder=5)
    ax1.text(by_hour["godzina"].iloc[-1] + 0.2, PROG_OPOZNIENIA_PROCENT_LOTY + 0.5,
             f"{PROG_OPOZNIENIA_PROCENT_LOTY}%", color=KOLOR_PROG, fontsize=9)

    ax1.set_xlabel("Godzina planowanego odlotu", labelpad=8)
    ax1.set_ylabel("Odsetek opóźnionych (%)", color="#2C2C2A", labelpad=8)
    ax2.set_ylabel("Liczba lotów", color="#888780", labelpad=8)
    ax2.tick_params(axis="y", colors="#888780")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax1.set_xticks(by_hour["godzina"])
    ax1.set_xticklabels([f"{h:02d}:00" for h in by_hour["godzina"]], rotation=45)
    ax1.set_title("% opóźnionych lotów wg godziny odlotu – KRK",
                  fontsize=14, fontweight="bold", color="#2C2C2A", pad=12)

    legend_patches = [
        mpatches.Patch(color=KOLOR_OK,   label="< 20%"),
        mpatches.Patch(color=KOLOR_MALY, label="20–35%"),
        mpatches.Patch(color=KOLOR_DUZY, label="≥ 35%"),
        plt.Line2D([0], [0], color="#888780", linestyle=":", marker="o",
                   markersize=4, label="Liczba lotów"),
    ]
    ax1.legend(handles=legend_patches, fontsize=9, framealpha=0.7,
               edgecolor="#CCCBC3", loc="upper left")

    fig.tight_layout()
    fig.savefig("wykres2_godzina_odlotu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres2_godzina_odlotu.png")


def wykres_dzien_tygodnia(df_valid: pd.DataFrame):
    nazwy_dni = ["Poniedziałek", "Wtorek", "Środa", "Czwartek",
                 "Piątek", "Sobota", "Niedziela"]

    by_dow = (
        df_valid.groupby("dzien_tyg")["delayed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_dow["pct"] = by_dow["mean"] * 100

    kolory = [KOLOR_DUZY if d >= 4 else KOLOR_MALY for d in by_dow["dzien_tyg"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(by_dow["dzien_tyg"], by_dow["pct"],
                  color=kolory, zorder=3, width=0.6,
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, by_dow["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#5F5E5A")

    ax.axhline(y=by_dow["pct"].mean(), color=KOLOR_LINIA, linewidth=1.4,
               linestyle="--", zorder=4)
    ax.text(6.5, by_dow["pct"].mean() + 0.3,
            f"śr. {by_dow['pct'].mean():.1f}%", color=KOLOR_LINIA, fontsize=9)

    ax.set_xticks(range(7))
    ax.set_xticklabels(nazwy_dni, rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 45)
    ax.set_ylabel("Odsetek opóźnionych (%)", labelpad=8)
    ax.set_title("% opóźnionych lotów wg dnia tygodnia – KRK",
                 fontsize=14, fontweight="bold", color="#2C2C2A", pad=12)

    legend_patches = [
        mpatches.Patch(color=KOLOR_MALY, label="Dni robocze"),
        mpatches.Patch(color=KOLOR_DUZY, label="Piątek / Weekend"),
        plt.Line2D([0], [0], color=KOLOR_LINIA, linestyle="--", label="Średnia"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.7,
              edgecolor="#CCCBC3", loc="upper left")

    fig.tight_layout()
    fig.savefig("wykres3_dzien_tygodnia.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres3_dzien_tygodnia.png")


def wykres_linia_lotnicza(df_valid: pd.DataFrame):
    MIN_LOTOW = 10

    by_airline = (
        df_valid.groupby("linia_lotnicza")["delayed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_airline = by_airline[by_airline["count"] >= MIN_LOTOW].copy()
    by_airline["pct"] = by_airline["mean"] * 100
    by_airline = by_airline.sort_values("pct", ascending=True)

    kolory = [KOLOR_DUZY if p >= 40 else KOLOR_MALY if p >= 25 else KOLOR_OK
              for p in by_airline["pct"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(by_airline["linia_lotnicza"], by_airline["pct"],
                   color=kolory, zorder=3, height=0.65,
                   edgecolor="white", linewidth=0.8)

    for bar, val, cnt in zip(bars, by_airline["pct"], by_airline["count"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%  (n={int(cnt)})",
                va="center", ha="left", fontsize=9, color="#5F5E5A")

    ax.axvline(x=PROG_OPOZNIENIA_PROCENT_LINIE, color=KOLOR_PROG, linewidth=1.4,
               linestyle="--", zorder=4)
    ax.text(PROG_OPOZNIENIA_PROCENT_LINIE + 0.5, len(by_airline) - 0.3,
            f"próg {PROG_OPOZNIENIA_PROCENT_LINIE}%", color=KOLOR_PROG, fontsize=9)

    ax.set_xlim(0, 85)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Odsetek opóźnionych (%)", labelpad=8)
    ax.set_title(f"% opóźnionych lotów wg linii lotniczej (min. {MIN_LOTOW} lotów) – KRK",
                 fontsize=13, fontweight="bold", color="#2C2C2A", pad=12)

    legend_patches = [
        mpatches.Patch(color=KOLOR_OK,   label="< 25%"),
        mpatches.Patch(color=KOLOR_MALY, label="25–40%"),
        mpatches.Patch(color=KOLOR_DUZY, label="≥ 40%"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.7,
              edgecolor="#CCCBC3", loc="lower right")

    fig.tight_layout()
    fig.savefig("wykres4_linie_lotnicze.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres4_linie_lotnicze.png")


def wykres_kierunki(df_valid: pd.DataFrame):
    MIN_LOTOW = 10
    TOP_N     = 20

    by_route = (
        df_valid.groupby("kierunek")["delayed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_route = by_route[by_route["count"] >= MIN_LOTOW].copy()
    by_route["pct"] = by_route["mean"] * 100
    by_route = by_route.sort_values("pct", ascending=False).head(TOP_N)
    by_route = by_route.sort_values("pct", ascending=True)   # oś Y: rosnąco

    # Skróć długie nazwy kierunków
    by_route["label"] = by_route["kierunek"].str.replace(r"\s*\(.*\)", "", regex=True)

    kolory = [KOLOR_DUZY if p >= 55 else KOLOR_MALY for p in by_route["pct"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(by_route["label"], by_route["pct"],
                   color=kolory, zorder=3, height=0.65,
                   edgecolor="white", linewidth=0.8)

    for bar, val, cnt in zip(bars, by_route["pct"], by_route["count"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%  (n={int(cnt)})",
                va="center", ha="left", fontsize=9, color="#5F5E5A")

    ax.set_xlim(0, 90)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Odsetek opóźnionych (%)", labelpad=8)
    ax.set_title(f"Top {TOP_N} kierunków wg % opóźnień (min. {MIN_LOTOW} lotów) – KRK",
                 fontsize=13, fontweight="bold", color="#2C2C2A", pad=12)

    legend_patches = [
        mpatches.Patch(color=KOLOR_MALY, label="40–55%"),
        mpatches.Patch(color=KOLOR_DUZY, label="≥ 55%"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.7,
              edgecolor="#CCCBC3", loc="lower right")

    fig.tight_layout()
    fig.savefig("wykres5_kierunki.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres5_kierunki.png")


def wykres_dashboard(df: pd.DataFrame, df_valid: pd.DataFrame):
    n_total     = len(df)
    n_cancelled = df["odwołany"].sum()
    n_delayed   = df_valid["delayed"].sum()
    pct_delayed = df_valid["delayed"].mean() * 100
    mean_delay = df_valid["opoznienie"].mean()
    n_airlines  = df_valid["linia_lotnicza"].nunique()
    n_routes    = df_valid["kierunek"].nunique()

    # Zmniejszono nieco rozmiar figury, ponieważ mamy mniej wykresów
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle("Analiza P1 – Predykcja opóźnień lotów, KRK (kwiecień–maj 2026)",
                 fontsize=15, fontweight="bold", color="#2C2C2A", y=0.98)

    # ── KPI boxes ──
    kpi_ax = fig.add_axes([0.0, 0.75, 1.0, 0.20])
    kpi_ax.axis("off")
    kpis = [
        ("Wszystkich lotów",   f"{n_total:,}",        KOLOR_LINIA),
        ("Opóźnionych ≥15 min",f"{pct_delayed:.1f}%", KOLOR_DUZY),
        ("Odwołanych",         f"{n_cancelled}",      KOLOR_MALY),
        ("Śr. opóźnienie",     f"+{mean_delay:.1f} min", KOLOR_MALY),
        ("Linii lotniczych",   f"{n_airlines}",       "#5F5E5A"),
        ("Kierunków",          f"{n_routes}",         "#5F5E5A"),
    ]
    
    for i, (label, val, col) in enumerate(kpis):
        x = 0.035 + i * 0.16
        rect = mpatches.FancyBboxPatch((x, 0.08), 0.14, 0.84,
                                       boxstyle="round,pad=0.02",
                                       facecolor="#EFEDE8", edgecolor="#D3D1C7",
                                       linewidth=0.8, transform=kpi_ax.transAxes,
                                       clip_on=False)
        kpi_ax.add_patch(rect)
        kpi_ax.text(x + 0.07, 0.72, label, ha="center", va="center",
                    fontsize=8.5, color="#888780", transform=kpi_ax.transAxes)
        kpi_ax.text(x + 0.07, 0.32, val, ha="center", va="center",
                    fontsize=16, fontweight="bold", color=col,
                    transform=kpi_ax.transAxes)

    # ── Pie: rozkład klas ──
    # Wykres wyśrodkowany i powiększony
    ax_pie = fig.add_axes([0.25, 0.05, 0.50, 0.65])
    n_ok = len(df_valid) - n_delayed
    
    wedges, texts, autotexts = ax_pie.pie(
        [n_ok, n_delayed, n_cancelled],
        labels=["Na czas", "Opóźniony", "Odwołany"],
        colors=[KOLOR_OK, KOLOR_DUZY, KOLOR_MALY],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
        
    ax_pie.set_title("Rozkład klas", fontsize=13, fontweight="bold",
                     color="#2C2C2A", pad=8)

    fig.savefig("wykres6_dashboard_P1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ wykres6_dashboard_P1.png")


if __name__ == "__main__":
    print(f"Wczytywanie danych z: {DB_PATH}")
    df, df_valid = wczytaj_dane(DB_PATH)
    print(f"  Lotów łącznie: {len(df)}, ważnych: {len(df_valid)}, "
          f"opóźnionych: {df_valid['delayed'].sum()} "
          f"({df_valid['delayed'].mean()*100:.1f}%)\n")

    print("Generowanie wykresów...")
    wykres_rozklad(df_valid)
    wykres_godzina(df_valid)
    wykres_dzien_tygodnia(df_valid)
    wykres_linia_lotnicza(df_valid)
    wykres_kierunki(df_valid)
    wykres_dashboard(df, df_valid)

    print("\nGotowe. Pliki PNG zapisane w bieżącym katalogu.")

    print("Konwertowanie df_valid do csv...")
    df_valid.to_csv("loty_valid.csv", index=False, sep=',', encoding='utf-8-sig')