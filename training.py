import joblib
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from data_transform import FlightsTransform # transformation module 

# initial transform of the raw data block 
transformer = FlightsTransform("dataset_loty_krakow_20260513_183527.csv")
sanitized_df = transformer.transform(threshold_minutes=15) # since the model itself is going to be saved, there's no point in savng csv now

# dropping the unwanted stuff to prevent data leakage
X = sanitized_df.drop(['czy_opozniony', 'czas_rzeczywisty_unix'], axis=1)
y = sanitized_df['czy_opozniony']


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_eval = RandomForestClassifier(class_weight='balanced', random_state=42)


# since data is imbalanced, using f1 score to accurately access 
cv_scores = cross_val_score(model_eval, X, y, cv=cv, scoring='f1')
print(f"F1 results for each fold: {cv_scores}")
print(f"Mean validation f1: {cv_scores.mean():.4f}")

# final fit on the whole data: 
final_model = RandomForestClassifier(class_weight='balanced', random_state=42)
final_model.fit(X, y)

# since we need to preserve trained model's state 
# and hot encoder, to not confuse model later on
# we are saving their state using joblib module 
joblib.dump(final_model, 'flights_random_forest.joblib')
joblib.dump(transformer.encoder, "airline_hot_encoder.joblib")