import joblib
import numpy as np
import pandas as pd

# splitters / metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

# classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# data pipeline — all preprocessing, encoding, scaling, and resampling
# is handled by FlightsTransform; training.py contains no data logic
from data_transform import FlightsTransform


class TrainingClass:
    def __init__(self, data_path: str, threshold: int, encoding: str = 'onehot'):
        """
        Args:
            data_path: Path to the raw flights CSV.
            threshold: Delay threshold in minutes passed to FlightsTransform.load_xy().
            encoding:  'onehot' (default) or 'label' — forwarded to load_xy().
        """
        self.PATH = data_path
        self.threshold = threshold
        self.encoding = encoding

        self.MODEL_SELECTION = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=150, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            ),
            "Gaussian NB (GNB)": GaussianNB(),
            "Neural Network (MLP)": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42
            ),
        }

    def train_test_cv(self, balance: bool, resampling: str = 'smote') -> None:
        """
        All data manipulation (preprocessing, encoding, per-fold scaling,
        per-fold resampling) is delegated to FlightsTransform — this method
        contains only training and evaluation logic.

        Args:
            balance:    if ture -- resmaple in each fold 
            resampling: forwards to FlightsTransform.get_resampler().
                        either 'smote', 'undersample', 'smoteenn', 'smotetomek'.
                        only used when balanced = True
        """
        # loading data -- no resampling/scaling yet 
        ft = FlightsTransform(self.PATH)
        X, y = ft.load_xy(threshold_minutes=self.threshold, encoding=self.encoding)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        global_results = {}

        for model_name, model in self.MODEL_SELECTION.items():
            print(f"\nEvaluating: {model_name} via 5-fold stratified CV")

            f1_list, bal_acc_list = [], []

            # if resampling is off turn balancing on for tree based models in order not to break them entirely
            if not balance:
                if "Random Forest" in model_name:
                    model.set_params(class_weight='balanced')
                elif "XGBoost" in model_name:
                    model.set_params(scale_pos_weight=2.4)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # per-fold scaling 
                scaler = ft.get_scaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=scaler.get_feature_names_out(),
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=scaler.get_feature_names_out(),
                )

                # per-fold resampling -- of course only for training
                if balance:
                    resampler = ft.get_resampler(resampling)
                    X_train, y_train = resampler.fit_resample(X_train, y_train)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1      = f1_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1_list.append(f1)
                bal_acc_list.append(bal_acc)

                print(f"  fold {fold + 1} | f1: {f1:.3f} | balanced acc: {bal_acc:.3f}")

            global_results[model_name] = {
                "f1_history":      f1_list,
                "f1_mean":         np.mean(f1_list),
                "bal_acc_history": bal_acc_list,
                "bal_acc_mean":    np.mean(bal_acc_list),
            }

        # cross validation summary 
        print("\n" + "=" * 50)
        print("FINAL CROSS-VALIDATION SUMMARY")
        print("=" * 50)
        for model_name, metrics in global_results.items():
            print(f"Model: {model_name}")
            print(f"  * Mean F1-Score:          {metrics['f1_mean']:.4f}")
            print(f"  * Mean Balanced Accuracy: {metrics['bal_acc_mean']:.4f}")
            print("-" * 30)

        # -select best model and train the "prod" 
        best_name = max(global_results, key=lambda k: global_results[k]['f1_mean'])
        print(f"\nBest model based on F1: {best_name}. Training production version...")

        final_model = self.MODEL_SELECTION[best_name]

        X_final, y_final = X, y
        final_scaler = ft.get_scaler()
        X_final = pd.DataFrame(
            final_scaler.fit_transform(X_final),
            columns=final_scaler.get_feature_names_out(),
        )
        if balance:
            X_final, y_final = ft.get_resampler(resampling).fit_resample(X_final, y_final)

        final_model.fit(X_final, y_final)
        
        # dumping model AND SCALER to joblib 
        # joblib.dump({'model': final_model, 'scaler': final_scaler}, 'flights_production_model.joblib')
        # print("Production model saved as 'flights_production_model.joblib'!")


if __name__ == "__main__":
    trainer = TrainingClass(
        data_path="dataset_loty_krakow_20260521_213240.csv",
        threshold=25,
        encoding='label',
    )
    trainer.train_test_cv(balance=True, resampling='smoteenn')
    # The best for now is onehot + balance=true + undersample (Random Forest)
    # Maybe OOT or another skf costs trying