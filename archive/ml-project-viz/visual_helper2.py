import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sanitized_pandas_20260525_155930.csv', on_bad_lines="warn")

# Verify columns exist
required_cols = ['dzien_miesiaca', 'lat', 'godzina_sin', 'czy_opozniony']
if all(col in df.columns for col in required_cols):
    # Filter the dataset
    df_0 = df[df['czy_opozniony'] == 0]
    df_1 = df[df['czy_opozniony'] == 1]

    # Calculate global limits for consistent comparison
    x_min, x_max = df['dzien_miesiaca'].min(), df['dzien_miesiaca'].max()
    y_min, y_max = df['lat'].min(), df['lat'].max()
    z_min, z_max = df['godzina_sin'].min(), df['godzina_sin'].max()

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # Subplot 1: Not Delayed
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df_0['dzien_miesiaca'], df_0['lat'], df_0['godzina_sin'], 
                c='blue', alpha=0.3, s=10)
    ax1.set_title(f'Not Delayed (0)\nn={len(df_0)}')
    ax1.set_xlabel('Day of Month (dzien_miesiaca)')
    ax1.set_ylabel('Latitude (lat)')
    ax1.set_zlabel('Hour Sine (godzina_sin)')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    # Subplot 2: Delayed
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(df_1['dzien_miesiaca'], df_1['lat'], df_1['godzina_sin'], 
                c='red', alpha=0.5, s=10)
    ax2.set_title(f'Delayed (1)\nn={len(df_1)}')
    ax2.set_xlabel('Day of Month (dzien_miesiaca)')
    ax2.set_ylabel('Latitude (lat)')
    ax2.set_zlabel('Hour Sine (godzina_sin)')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    # Add a main title
    fig.suptitle('3D Scatter Plots: Day of Month vs Latitude vs Hour Sine by Delay Class', fontsize=16)

    # Save the plot
    plt.savefig('scatter_3d_parallel-4.png', bbox_inches='tight')
    plt.close()
    
    print("3D parallel plots successfully generated.")
else:
    print("Some required columns are missing.")