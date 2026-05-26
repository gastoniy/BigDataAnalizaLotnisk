import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sanitized_pandas_20260523_203936.csv')

if 'czy_opozniony' in df.columns:
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    # Filter data
    df_0 = df[df['czy_opozniony'] == 0]
    df_1 = df[df['czy_opozniony'] == 1]

    # Subplot 1: Not Delayed
    axes[0].scatter(df_0['dzien_miesiaca'], df_0['lat'], c='blue', alpha=0.4, s=10)
    axes[0].set_title(f'Not Delayed (n={len(df_0)})')
    axes[0].grid(True)

    # Subplot 2: Delayed
    axes[1].scatter(df_1['dzien_miesiaca'], df_1['lat'], c='red', alpha=0.6, s=10)
    axes[1].set_title(f'Delayed (n={len(df_1)})')
    axes[1].grid(True)

    # Main Title
    plt.suptitle('Geographic Distribution: Side-by-Side Comparison of Delay Classes', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('scatter_geo_parallel.png', bbox_inches='tight')
    plt.close()
    
    print("Parallel plots generated successfully.")
else:
    print("Column 'czy_opozniony' not found.")