import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

# Importy Twoich 5 modeli
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Import klasy transformującej (upewnij się, że plik nazywa się flights_transform.py)
from data_transform import FlightsTransform

# 1. Przygotowanie danych
print("Przetwarzanie danych wejściowych...")
transformer = FlightsTransform("dataset_loty_krakow_20260513_183527.csv")
sanitized_df = transformer.transform(threshold_minutes=15)

X = sanitized_df.drop(['czy_opozniony'], axis=1)
y = sanitized_df['czy_opozniony']

# 2. Definicja i konfiguracja (Setup) Twoich 5 modeli
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=2, class_weight='balanced', random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=150, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=2.4, random_state=42
    ),
    "Gaussian NB (GNB)": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42
    )
}

# 3. Walidacja krzyżowa (Rundy / Foldy)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- ROZPOCZĘCIE EWALUACJI (5-Fold Cross-Validation) ---")
for name, model in classifiers.items():
    # Liczymy F1-score dla każdego z 5 modeli
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print(f"{name:22} -> Średni F1-Score: {scores.mean():.4f} (+/- {scores.std():.3f})")

# 4. Finalny trening i zapis stanu modeli do plików .joblib
print("\n--- TRENOWANIE FINALNE I ZAPIS STANU ---")
for name, model in classifiers.items():
    print(f"Trenowanie i zamrażanie modelu: {name}...")
    model.fit(X, y)
    
    # Tworzenie bezpiecznej nazwy pliku (bez spacji i nawiasów)
    safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f'model_{safe_filename}.joblib')

# Zapisujemy koder One-Hot, aby skrypt strumieniowy pasował do każdego z tych 5 modeli
joblib.dump(transformer.encoder, 'airline_one_hot_encoder.joblib')
print("\nWszystkie modele oraz encoder zostały pomyślnie zapisane na dysku!")