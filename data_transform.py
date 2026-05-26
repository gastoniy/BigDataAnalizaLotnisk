import pandas as pd
import numpy as np
from datetime import datetime 
import airportsdata
<<<<<<< HEAD
from sklearn.preprocessing import OneHotEncoder
=======
from time import sleep

from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder,
    StandardScaler, RobustScaler, MinMaxScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
>>>>>>> dev-branch-ihor

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath
        self.df = pd.read_csv(datapath)
        self.aerodata = airportsdata.load('IATA')

<<<<<<< HEAD
=======
        # defining top 37 airlines based on long-term observation
>>>>>>> dev-branch-ihor
        self.top_airlines = ['LS', 'D8', 'TK', 'IZ', 'EZY', 'A3', 'FH', '4M', 'LX',
                             'RK', 'TOM', 'DY', 'AF', 'EJU', 'LH', 'EN', 'LG', 'PC',
                             'ENT', 'EW', 'W4', 'FR', 'LO', 'W6', 'SK', 'OS', 'RR',
                             'BA', 'XQ', 'LY', 'KLJ', 'MGH', 'SN', 'JU', 'EZS', 'AY',
                             'KL']

<<<<<<< HEAD
        # 1hot encoder (mamy to 37 linii, inne ignorowane )
        self.encoder = OneHotEncoder(categories=[self.top_airlines],
                                     handle_unknown='ignore',
                                     sparse_output=False)
        
=======
        # OHE encoder initialization
        self.encoder = OneHotEncoder(
            categories=[self.top_airlines],
            handle_unknown='ignore',  # unknown airlines get all 0s
            sparse_output=False
        )

        # OrdinalEncoder for label encoding:
        # known airlines get a stable 0-based index, unknowns are mapped to -1
        self.label_encoder = OrdinalEncoder(
            categories=[self.top_airlines],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=int
        )

        # fit both encoders on the fixed airline list so columns are stable
        # regardless of what appears in the CSV
>>>>>>> dev-branch-ihor
        fake_data = pd.DataFrame({'linia_lotnicza': self.top_airlines})
        self.encoder.fit(fake_data)
        self.label_encoder.fit(fake_data)

<<<<<<< HEAD
    # zamiast nazwy celu otrzymujemy koordynaty
=======
        # populated by scale(); exposed so callers can pickle/reuse the fitted scaler
        self.scaler: ColumnTransformer | None = None

    # Kraków airport coordinates (EPKK)
    _KRK_LAT = np.radians(50.0777)
    _KRK_LON = np.radians(19.7848)

>>>>>>> dev-branch-ihor
    def _convert_icao(self, old_dest: str):
        try:
            iata = old_dest.split('(')[-1].replace(')', '').strip()
            inf = self.aerodata.get(iata)
            if inf:
                return inf['lat'], inf['lon'], inf['elevation']
        except:
            pass
<<<<<<< HEAD
        return None, None, None 

    def transform(self, threshold_minutes: int = 15):
        needed_cols = ['numer_lotu', 'linia_lotnicza', 'kierunek', 'czas_planowany', 'czas_rzeczywisty']
        self.df = self.df[needed_cols].copy()

        # konwersja na współrzędne
=======
        return None, None, None

    def _haversine_from_krakow(self, lat_deg: pd.Series, lon_deg: pd.Series) -> pd.Series:
        """Return great-circle distance in km between each point and Kraków airport."""
        lat2 = np.radians(lat_deg)
        lon2 = np.radians(lon_deg)
        dlat = lat2 - self._KRK_LAT
        dlon = lon2 - self._KRK_LON
        a = np.sin(dlat / 2) ** 2 + np.cos(self._KRK_LAT) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def preprocess(self, threshold_minutes: int = 15) -> pd.DataFrame:
        """
        Step 1 — sanitize columns, extract coordinates, compute distance from Kraków,
        build cyclic datetime features, derive the delay label, and drop raw columns.
        Leaves 'linia_lotnicza' in the DataFrame so either encoding method can
        be applied afterwards.

        Args:
            threshold_minutes: Delay threshold in minutes for the 'czy_opozniony' label.

        Returns:
            Preprocessed DataFrame with 'linia_lotnicza' still present.
        """
        # reload from source so preprocess() is idempotent and transform() is safely repeatable
        self.df = pd.read_csv(self.datapath)

        needed_cols = ['numer_lotu', 'linia_lotnicza', 'kierunek', 'czas_planowany', 'czas_rzeczywisty']
        self.df = self.df[needed_cols].copy()

        # --- coordinates & distance ---
>>>>>>> dev-branch-ihor
        result = self.df['kierunek'].apply(self._convert_icao)
        self.df[['lat', 'lon', 'elev']] = pd.DataFrame(result.tolist(), index=self.df.index)
        self.df['dystans_km'] = self._haversine_from_krakow(self.df['lat'], self.df['lon'])
        self.df = self.df.drop('kierunek', axis=1)

<<<<<<< HEAD
        # zapisujemy dystans do krakowa zamiast destynacji 
        lat1, lon1 = np.radians(50.0777), np.radians(19.7848)
        lat2, lon2 = np.radians(self.df['lat']), np.radians(self.df['lon'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        self.df['dystans_km'] = 6371 * c

        # konwersja datetime
        self.df['czas_planowany'] = pd.to_datetime(self.df['czas_planowany'])
        self.df['czas_rzeczywisty'] = pd.to_datetime(self.df['czas_rzeczywisty'])

        # cze
        dzien_tygodnia = self.df['czas_planowany'].dt.weekday
        miesiac = self.df['czas_planowany'].dt.month
        godzina = self.df['czas_planowany'].dt.hour
        
        # Zostawiamy flagi binarne i dni miesiąca (bo dni miesiąca 1-31 nie są idealnie cykliczne)
        self.df['jest_weekend'] = (dzien_tygodnia >= 5).astype(int)
        self.df['dzien_miesiaca'] = self.df['czas_planowany'].dt.day

        # kodowanie godziny jako sin i cos dla zachowania cylkiczności 
        self.df['godzina_sin'] = np.sin(2 * np.pi * godzina / 24.0)
        self.df['godzina_cos'] = np.cos(2 * np.pi * godzina / 24.0)

        # dzień tygodnia
        self.df['dzien_tyg_sin'] = np.sin(2 * np.pi * dzien_tygodnia / 7.0)
        self.df['dzien_tyg_cos'] = np.cos(2 * np.pi * dzien_tygodnia / 7.0)

        # miesiąc 
        self.df['miesiac_sin'] = np.sin(2 * np.pi * miesiac / 12.0)
        self.df['miesiac_cos'] = np.cos(2 * np.pi * miesiac / 12.0)

        # wyliczanie label
        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

        # 1 hot encoding linii lotniczej 
        airline_encoded = self.encoder.transform(self.df[['linia_lotnicza']])
        encoded_cols = self.encoder.get_feature_names_out(['linia_lotnicza'])
        df_encoded = pd.DataFrame(airline_encoded, columns=encoded_cols, index=self.df.index)
        self.df = pd.concat([self.df, df_encoded], axis=1)

        # czyszczenie niepotrzebnych danych
        self.df = self.df.drop(['linia_lotnicza', 'numer_lotu', 'czas_planowany', 'czas_rzeczywisty'], axis=1)
=======
        # --- datetime parsing ---
        self.df['czas_planowany'] = pd.to_datetime(self.df['czas_planowany'])
        self.df['czas_rzeczywisty'] = pd.to_datetime(self.df['czas_rzeczywisty'])

        # --- raw time components (used for cyclic encoding below) ---
        godzina      = self.df['czas_planowany'].dt.hour
        dzien_tygodnia = self.df['czas_planowany'].dt.weekday
        miesiac      = self.df['czas_planowany'].dt.month

        # --- binary / ordinal features ---
        self.df['jest_weekend']   = (dzien_tygodnia >= 5).astype(int)
        self.df['dzien_miesiaca'] = self.df['czas_planowany'].dt.day

        # --- cyclic encoding (sin + cos preserve distance between periodic values) ---
        self.df['godzina_sin']    = np.sin(2 * np.pi * godzina      / 24.0)
        self.df['godzina_cos']    = np.cos(2 * np.pi * godzina      / 24.0)
        self.df['dzien_tyg_sin']  = np.sin(2 * np.pi * dzien_tygodnia /  7.0)
        self.df['dzien_tyg_cos']  = np.cos(2 * np.pi * dzien_tygodnia /  7.0)
        self.df['miesiac_sin']    = np.sin(2 * np.pi * miesiac       / 12.0)
        self.df['miesiac_cos']    = np.cos(2 * np.pi * miesiac       / 12.0)

        # --- classification label ---
        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

        # --- drop raw columns (airline encoding happens in a separate step) ---
        self.df = self.df.drop(['numer_lotu', 'czas_planowany', 'czas_rzeczywisty'], axis=1)
>>>>>>> dev-branch-ihor
        self.df = self.df.dropna()

        return self.df

    def one_hot_encode(self) -> pd.DataFrame:
        """
        Step 2a — one-hot encode the 'linia_lotnicza' column.
        Produces one binary column per airline in top_airlines; unknown airlines
        get all zeros (handle_unknown='ignore').
        Requires preprocess() to have been called first.

        Returns:
            DataFrame with 'linia_lotnicza' replaced by OHE columns.
        """
        if 'linia_lotnicza' not in self.df.columns:
            raise ValueError("Column 'linia_lotnicza' not found. Run preprocess() first.")

        airline_encoded = self.encoder.transform(self.df[['linia_lotnicza']])
        encoded_cols = self.encoder.get_feature_names_out(['linia_lotnicza'])
        df_encoded = pd.DataFrame(airline_encoded, columns=encoded_cols, index=self.df.index)

        self.df = pd.concat([self.df.drop('linia_lotnicza', axis=1), df_encoded], axis=1)

        return self.df

    def label_encode(self) -> pd.DataFrame:
        """
        Step 2b — label-encode the 'linia_lotnicza' column using OrdinalEncoder.
        Known airlines receive a stable integer index (0-based, matching top_airlines order);
        unknowns are mapped to -1 (via handle_unknown='use_encoded_value').
        Requires preprocess() to have been called first.

        Returns:
            DataFrame with 'linia_lotnicza' replaced by 'linia_lotnicza_label' (int).
        """
        if 'linia_lotnicza' not in self.df.columns:
            raise ValueError("Column 'linia_lotnicza' not found. Run preprocess() first.")

        self.df['linia_lotnicza_label'] = self.label_encoder.transform(
            self.df[['linia_lotnicza']]
        ).astype(int)
        self.df = self.df.drop('linia_lotnicza', axis=1)

        return self.df

    # columns whose distributions and roles call for specific scaling strategies
    _STD_COLS    = ['lat']               # near-normal, few outliers → z-score
    _ROBUST_COLS = ['lon', 'elev']       # skewed / many outliers → median + IQR
    _LOG_COLS    = ['dystans_km']        # extreme right skew (skew +4.2) → log1p then z-score
    _MM_COLS     = ['dzien_miesiaca']    # bounded uniform, no outliers → [0, 1]
    # cyclic sin/cos columns, linia_lotnicza_label, OHE columns, and the target
    # are intentionally excluded — scaling them would corrupt their meaning

    def get_scaler(self) -> ColumnTransformer:
        """
        Factory — returns a fresh, unfitted ColumnTransformer with the column-specific
        scaling strategy for this dataset.

        Intended use: call once per CV fold so each fold fits its own scaler on
        training data only, preventing leakage from the test split.

            scaler = ft.get_scaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train), columns=scaler.get_feature_names_out()
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test), columns=scaler.get_feature_names_out()
            )

        Scaling strategy:
            StandardScaler   lat            — near-normal (skew −0.14), few outliers
            RobustScaler     lon, elev      — skewed / heavy outliers; median+IQR
            log1p→Standard   dystans_km     — extreme right skew (+4.20); log compression
                                              via FunctionTransformer inside a Pipeline
            MinMaxScaler     dzien_miesiaca — bounded [1, 30], uniform, zero outliers

        Columns left at remainder='passthrough' (intentionally unscaled):
            sin/cos cyclic features  — already in [−1, 1]
            linia_lotnicza_label     — nominal integer; scaling implies false ordinality
            OHE airline columns      — binary indicators
            czy_opozniony            — target; never scale

        Returns:
            Unfitted ColumnTransformer ready for fit_transform / transform.
        """
        log_pipeline = Pipeline([
            ('log1p',   FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
            ('standard', StandardScaler()),
        ])

        # guard: only include a group if all its columns exist in the current df
        # (allows calling get_scaler() before or after encoding without errors)
        def _present(cols):
            return [c for c in cols if c in self.df.columns]

        transformers = []
        if _present(self._STD_COLS):
            transformers.append(('standard', StandardScaler(),  _present(self._STD_COLS)))
        if _present(self._ROBUST_COLS):
            transformers.append(('robust',   RobustScaler(),    _present(self._ROBUST_COLS)))
        if _present(self._LOG_COLS):
            transformers.append(('log_std',  log_pipeline,      _present(self._LOG_COLS)))
        if _present(self._MM_COLS):
            transformers.append(('minmax',   MinMaxScaler(),    _present(self._MM_COLS)))

        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose_feature_names_out=False,
        )

    def get_resampler(self, method: str, random_state: int = 42):
        """
        Factory — returns a fresh, unfitted imblearn sampler for the given method.

        Intended use: call once per CV fold so each fold's sampler is independent.

            resampler = ft.get_resampler('smote')
            X_train, y_train = resampler.fit_resample(X_train, y_train)

        Args:
            method:       One of 'smote', 'undersample', 'smoteenn', 'smotetomek'.
            random_state: Seed for reproducibility (default 42).

        Returns:
            Unfitted sampler ready for fit_resample.
        """
        if method not in self._RESAMPLE_METHODS:
            raise ValueError(
                f"Unknown resampling method '{method}'. "
                f"Choose one of: {self._RESAMPLE_METHODS}."
            )
        return {
            'smote':       SMOTE(random_state=random_state),
            'undersample': RandomUnderSampler(random_state=random_state),
            'smoteenn':    SMOTEENN(random_state=random_state),
            'smotetomek':  SMOTETomek(random_state=random_state),
        }[method]

    def load_xy(
        self,
        threshold_minutes: int = 15,
        encoding: str = 'onehot',
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Convenience loader — runs preprocess() and the chosen encoding step, then
        splits the result into features X and target y.

        Scaling and resampling are intentionally excluded: both must be fitted
        inside each CV fold on training data only to prevent leakage.

        Args:
            threshold_minutes: Passed through to preprocess().
            encoding:          'onehot' (default) or 'label'.

        Returns:
            (X, y) where X is the feature DataFrame and y is the binary target Series.
        """
        if encoding not in ('onehot', 'label'):
            raise ValueError(f"Unknown encoding '{encoding}'. Choose 'onehot' or 'label'.")

        self.preprocess(threshold_minutes=threshold_minutes)

        if encoding == 'onehot':
            self.one_hot_encode()
        else:
            self.label_encode()

        return self.df.drop('czy_opozniony', axis=1), self.df['czy_opozniony']

    def scale(self) -> pd.DataFrame:
        """
        Step 3 (optional) — apply column-specific scaling to the full dataset in-place.
        Uses get_scaler() internally so the strategy is defined in one place.
        Stores the fitted scaler in self.scaler for serialisation or reuse.

        Note: this method is for whole-dataset transformations (e.g. saving a
        pre-scaled CSV). For CV training use get_scaler() per fold instead.

        Returns:
            DataFrame with scaled continuous columns; all other columns unchanged.
        """
        if 'linia_lotnicza' in self.df.columns:
            raise ValueError(
                "Raw 'linia_lotnicza' column still present. "
                "Run one_hot_encode() or label_encode() before scale()."
            )

        self.scaler = self.get_scaler()

        scaled_array = self.scaler.fit_transform(self.df)
        self.df = pd.DataFrame(
            scaled_array,
            columns=self.scaler.get_feature_names_out(),
            index=self.df.index,
        ).astype({col: int for col in ['czy_opozniony'] if col in self.scaler.get_feature_names_out()})

        return self.df

    _RESAMPLE_METHODS = ('smote', 'undersample', 'smoteenn', 'smotetomek')

    def resample(self, method: str, random_state: int = 42) -> pd.DataFrame:
        """
        Step 3 (optional) — rebalance the dataset by over- or under-sampling.
        Must be called after an encoding step (one_hot_encode or label_encode),
        because all features must be numeric before resampling.

        Available methods:
            'smote'       — SMOTE oversampling: synthesises new minority-class
                            rows by interpolating between existing neighbours.
                            Increases dataset size.
            'undersample' — RandomUnderSampler: randomly removes majority-class
                            rows until classes are balanced.
                            Decreases dataset size.
            'smoteenn'    — SMOTEENN: SMOTE followed by Edited Nearest Neighbours
                            cleaning. Adds synthetic minority samples then removes
                            noisy / borderline samples from both classes.
            'smotetomek'  — SMOTETomek: SMOTE followed by Tomek Links removal.
                            Gentler cleaning than ENN; keeps more samples overall.

        Args:
            method:       One of 'smote', 'undersample', 'smoteenn', 'smotetomek'.
            random_state: Seed for reproducibility (default 42).

        Returns:
            Resampled DataFrame; self.df is updated in-place.
        """
        if method not in self._RESAMPLE_METHODS:
            raise ValueError(
                f"Unknown resampling method '{method}'. "
                f"Choose one of: {self._RESAMPLE_METHODS}."
            )
        if 'linia_lotnicza' in self.df.columns:
            raise ValueError(
                "Raw 'linia_lotnicza' column still present. "
                "Run one_hot_encode() or label_encode() before resample()."
            )
        if 'czy_opozniony' not in self.df.columns:
            raise ValueError("Target column 'czy_opozniony' not found. Run preprocess() first.")

        X = self.df.drop('czy_opozniony', axis=1)
        y = self.df['czy_opozniony']

        samplers = {
            'smote':       SMOTE(random_state=random_state),
            'undersample': RandomUnderSampler(random_state=random_state),
            'smoteenn':    SMOTEENN(random_state=random_state),
            'smotetomek':  SMOTETomek(random_state=random_state),
        }

        X_res, y_res = samplers[method].fit_resample(X, y)

        self.df = pd.DataFrame(X_res, columns=X.columns)
        self.df['czy_opozniony'] = y_res.values

        return self.df

    def transform(
        self,
        threshold_minutes: int = 15,
        encoding: str = 'onehot',
        scaling: bool = False,
        resampling: str = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Convenience orchestrator that runs the full pipeline in one call.

        Pipeline order:
            preprocess() -> encode() -> [scale()] -> [resample()]

        Scaling is applied before resampling so that SMOTE interpolates in the
        already-normalised feature space, which produces more realistic synthetic
        samples for distance-sensitive columns such as dystans_km and elev.

        Args:
            threshold_minutes: Passed through to preprocess().
            encoding:          'onehot' (default) or 'label'.
            scaling:           If True, run scale() after encoding (default False).
            resampling:        Optional resampling step. One of 'smote',
                               'undersample', 'smoteenn', 'smotetomek', or None
                               (default — no resampling).
            random_state:      Seed forwarded to resample() when resampling is set.

        Returns:
            Fully transformed DataFrame.
        """
        if encoding not in ('onehot', 'label'):
            raise ValueError(f"Unknown encoding '{encoding}'. Choose 'onehot' or 'label'.")
        if resampling is not None and resampling not in self._RESAMPLE_METHODS:
            raise ValueError(
                f"Unknown resampling method '{resampling}'. "
                f"Choose one of: {self._RESAMPLE_METHODS}, or None."
            )

        self.preprocess(threshold_minutes=threshold_minutes)

        if encoding == 'onehot':
            self.one_hot_encode()
        else:
            self.label_encode()

        if scaling:
            self.scale()

        if resampling is not None:
            self.resample(method=resampling, random_state=random_state)

        return self.df

    def save(self):
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sanitized_pandas_{savedate}.csv"
        self.df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Data saved as: {filename}")

if __name__ == "__main__":
<<<<<<< HEAD
    transformer = FlightsTransform("dataset_loty_krakow_20260513_183527.csv")
    data = transformer.transform(threshold_minutes=15)
    transformer.save()
    
    print("Data sample:")
    print(data[['dystans_km', 'godzina_sin', 'godzina_cos', 'czy_opozniony']].head(3))
=======
    # --- OHE, no scaling, no resampling (default) ---
    transformer = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    data_ohe = transformer.transform(threshold_minutes=15, encoding='onehot')
    # transformer.save()
    print("OHE sample:")
    print(data_ohe.head(3))

    # --- label encoding + scaling (step-by-step) ---
    transformer2 = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    transformer2.preprocess(threshold_minutes=15)
    transformer2.label_encode()
    data_scaled = transformer2.scale()
    # transformer2.save()
    print("\nLabel + scaled sample (continuous cols):")
    print(data_scaled[['lat', 'lon', 'elev', 'dystans_km', 'dzien_miesiaca']].describe().round(3))

    # --- OHE + scaling + Under (full pipeline via transform) ---
    transformer3 = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    data_full = transformer3.transform(
        threshold_minutes=15,
        encoding='onehot',
        scaling=True,
        resampling='undersample',
    )
    transformer3.save()
    print("\nOHE + scaled + Under class distribution:")
    print(data_full['czy_opozniony'].value_counts())
    sleep(1)

    # --- label encoding + undersampling, no scaling ---
    transformer4 = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    transformer4.preprocess(threshold_minutes=15)
    transformer4.label_encode()
    data_under = transformer4.resample(method='undersample')
    transformer4.save()
    print("\nLabel + undersample class distribution:")
    print(data_under['czy_opozniony'].value_counts())
    sleep(1)
>>>>>>> dev-branch-ihor
