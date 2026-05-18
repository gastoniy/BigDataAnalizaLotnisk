import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from data_transform import FlightsTransform

# transformacja
transformer = FlightsTransform("dataset_loty_krakow_20260513_183527.csv")
sanitized_df = transformer.transform(threshold_minutes=15)

# drop klasyfikacji do trenowania 
X = sanitized_df.drop(['czy_opozniony'], axis=1)
y = sanitized_df['czy_opozniony']

# kontola jakości za pomocą f1 score i rskf
cv = StratifiedKFold(n_splits=5, shuffle=True)
model_eval = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    min_samples_leaf=15, 
    class_weight='balanced', 
)

cv_scores = cross_val_score(model_eval, X, y, cv=cv, scoring='f1')
print(f"F1 results for each fold: {cv_scores}")
print(f"Mean validation f1: {cv_scores.mean():.4f}")

# finalny trening
final_model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    min_samples_leaf=15, 
    class_weight='balanced', 
)
final_model.fit(X, y)

# zrzucenie modelu oraz 
joblib.dump(final_model, 'flights_random_forest.joblib')
joblib.dump(transformer.encoder, 'airline_one_hot_encoder.joblib')
