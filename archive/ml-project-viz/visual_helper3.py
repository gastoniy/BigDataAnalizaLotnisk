import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytanie danych z pliku CSV
df = pd.read_csv('sanitized_pandas_20260528_094039.csv')

# 2. Ustawienie estetycznego stylu wykresu
sns.set_theme(style="ticks")

# 3. Wygenerowanie macierzy wykresów (scatter plot matrix)
# hue='czy_opozniony' - koloruje punkty wg opóźnienia
# plot_kws - parametry punktów: alpha (przezroczystość) i s (rozmiar)
# corner=True - ukrywa górny trójkąt wykresów, żeby macierz była bardziej przejrzysta
g = sns.pairplot(
    df, 
    hue='czy_opozniony', 
    plot_kws={'alpha': 0.5, 's': 10}, 
    corner=True
)

# 4. Dodanie ogólnego tytułu
g.fig.suptitle('Macierz wykresów punktowych dla wszystkich cech', y=1.02)

# 5. Zapisanie wykresu do pliku (z zachowaniem wysokiej rozdzielczości)
plt.savefig('scatter_matrix.png', dpi=150, bbox_inches='tight')

# Wyświetlenie wykresu (przydatne przy uruchamianiu skryptu lokalnie)
plt.show()