import pandas as pd
import numpy as np
from datetime import datetime 
from typing import Literal
import airportsdata
from time import sleep
from pathlib import Path

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

class FlightsTransform:
    Encoding = Literal['onehot', 'label']
    ResamplingMethod = Literal['smote', 'undersample', 'smoteenn', 'smotetomek']
    ENCODING_OPTIONS = ('onehot', 'label')
    RESAMPLING_OPTIONS = ('smote', 'undersample', 'smoteenn', 'smotetomek')

    def __init__(self, datapath: str):
        self.datapath = datapath
        self.df = pd.read_csv(datapath)
        self.aerodata = airportsdata.load('IATA')

        # defining top 37 airlines based on long-term observation
        self.top_airlines = ['LS', 'D8', 'TK', 'IZ', 'EZY', 'A3', 'FH', '4M', 'LX',
                             'RK', 'TOM', 'DY', 'AF', 'EJU', 'LH', 'EN', 'LG', 'PC',
                             'ENT', 'EW', 'W4', 'FR', 'LO', 'W6', 'SK', 'OS', 'RR',
                             'BA', 'XQ', 'LY', 'KLJ', 'MGH', 'SN', 'JU', 'EZS', 'AY',
                             'KL']

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
        fake_data = pd.DataFrame({'linia_lotnicza': self.top_airlines})
        self.encoder.fit(fake_data)
        self.label_encoder.fit(fake_data)

        # populated by scale(); exposed so callers can pickle/reuse the fitted scaler
        self.scaler: ColumnTransformer | None = None

    # Kraków airport coordinates (EPKK)
    _KRK_LAT = np.radians(50.0777)
    _KRK_LON = np.radians(19.7848)

    def _convert_icao(self, old_dest: str):
        try:
            iata = old_dest.split('(')[-1].replace(')', '').strip()
            inf = self.aerodata.get(iata)
            if inf:
                return inf['lat'], inf['lon'], inf['elevation']
        except:
            pass
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
        result = self.df['kierunek'].apply(self._convert_icao)
        self.df[['lat', 'lon', 'elev']] = pd.DataFrame(result.tolist(), index=self.df.index)
        self.df['dystans_km'] = self._haversine_from_krakow(self.df['lat'], self.df['lon'])
        self.df = self.df.drop('kierunek', axis=1)

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
        self.df = self.df.dropna()

        return self.df

    def one_hot_encode(self) -> pd.DataFrame:

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

    def get_resampler(self, method: ResamplingMethod, random_state: int = 42):

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
        encoding: Encoding = 'onehot',
    ) -> tuple[pd.DataFrame, pd.Series]:

        if encoding not in ('onehot', 'label'):
            raise ValueError(f"Unknown encoding '{encoding}'. Choose 'onehot' or 'label'.")

        self.preprocess(threshold_minutes=threshold_minutes)

        if encoding == 'onehot':
            self.one_hot_encode()
        else:
            self.label_encode()

        return self.df.drop('czy_opozniony', axis=1), self.df['czy_opozniony']

    def scale(self) -> pd.DataFrame:

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

    def resample(self, method: ResamplingMethod, random_state: int = 42) -> pd.DataFrame:
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
        encoding: Encoding = 'onehot',
        scaling: bool = False,
        resampling: ResamplingMethod | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:

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

    def save(
        self,
        encoding: Encoding,
        resampling: ResamplingMethod,
        threshold: int,
        path: str | Path = ".",
    ) -> None:
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sanitized_pandas_{savedate}_{encoding}_{resampling}_{threshold}.csv"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

        full_path = path / filename

        self.df.to_csv(full_path, index=False, encoding="utf-8")
        print(f"Data saved as: {full_path}")

if __name__ == "__main__":
    tester = FlightsTransform("dataset_loty_krakow_20260521_213240.csv")
    iterations = 3 # each undersampling iteration will get different random state for objective variability assessment, while other encoding/resampling combinations will be repeated for consistency check
    stepup = 5
    for i in range(5,30,5):
        print(f"Iteration for threshold {i} minutes")
        for encoding_option in tester.ENCODING_OPTIONS:
            for resampling_option in tester.RESAMPLING_OPTIONS:
                print(f"  with encoding={encoding_option} and resampling={resampling_option}")
                for _ in range(iterations):
                    tester.transform(
                        threshold_minutes=i,
                        encoding=encoding_option,
                        scaling=True,
                        resampling=resampling_option,
                        random_state=None,  # use different random seed each time for undersampling 
                    )
                    tester.save(encoding=encoding_option, resampling=resampling_option, threshold=i, path="modeltests")
                    sleep(1)
        