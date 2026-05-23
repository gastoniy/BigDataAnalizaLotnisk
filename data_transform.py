import pandas as pd
import numpy as np
from datetime import datetime 
import airportsdata
from time import sleep

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class FlightsTransform:
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
        """Return great-circle distance in km between each point and Krakow airport."""
        lat2 = np.radians(lat_deg)
        lon2 = np.radians(lon_deg)
        dlat = lat2 - self._KRK_LAT
        dlon = lon2 - self._KRK_LON
        a = np.sin(dlat / 2) ** 2 + np.cos(self._KRK_LAT) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def preprocess(self, threshold_minutes: int = 15) -> pd.DataFrame:
        """
        Step 1 — sanitize columns, extract coordinates, compute distance,
        build cyclic datetime features, derive the delay label, and drop raw columns.

        Args:
            threshold_minutes: Delay threshold in minutes for the 'czy_opozniony' label.

        Returns:
            Preprocessed DataFrame with 'linia_lotnicza' still present.
        """
        # reload from source so preprocess() is idempotent and transform() is safely repeatable
        self.df = pd.read_csv(self.datapath)

        needed_cols = ['numer_lotu', 'linia_lotnicza', 'kierunek', 'czas_planowany', 'czas_rzeczywisty']
        self.df = self.df[needed_cols].copy()

        # coordinates & distance 
        result = self.df['kierunek'].apply(self._convert_icao)
        self.df[['lat', 'lon', 'elev']] = pd.DataFrame(result.tolist(), index=self.df.index)
        self.df['dystans_km'] = self._haversine_from_krakow(self.df['lat'], self.df['lon'])
        self.df = self.df.drop('kierunek', axis=1)

        # datetime parsing
        self.df['czas_planowany'] = pd.to_datetime(self.df['czas_planowany'])
        self.df['czas_rzeczywisty'] = pd.to_datetime(self.df['czas_rzeczywisty'])

        # raw time components (used for cyclic encoding below)
        godzina      = self.df['czas_planowany'].dt.hour
        dzien_tygodnia = self.df['czas_planowany'].dt.weekday
        miesiac      = self.df['czas_planowany'].dt.month

        # binary / ordinal features
        self.df['jest_weekend']   = (dzien_tygodnia >= 5).astype(int)
        self.df['dzien_miesiaca'] = self.df['czas_planowany'].dt.day

        # cyclic encoding (sin + cos preserve distance between periodic values)
        self.df['godzina_sin']    = np.sin(2 * np.pi * godzina      / 24.0)
        self.df['godzina_cos']    = np.cos(2 * np.pi * godzina      / 24.0)
        self.df['dzien_tyg_sin']  = np.sin(2 * np.pi * dzien_tygodnia /  7.0)
        self.df['dzien_tyg_cos']  = np.cos(2 * np.pi * dzien_tygodnia /  7.0)
        self.df['miesiac_sin']    = np.sin(2 * np.pi * miesiac       / 12.0)
        self.df['miesiac_cos']    = np.cos(2 * np.pi * miesiac       / 12.0)

        # classification label
        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

        # drop raw columns (airline encoding happens in a separate step)
        self.df = self.df.drop(['numer_lotu', 'czas_planowany', 'czas_rzeczywisty'], axis=1)
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

    def transform(self, threshold_minutes: int = 15, encoding: str = 'onehot') -> pd.DataFrame:
        """
        Convenience orchestrator that runs the full pipeline in one call.

        Args:
            threshold_minutes: Passed through to preprocess().
            encoding: 'onehot' (default) or 'label' — selects the encoding step.

        Returns:
            Fully transformed DataFrame.
        """
        if encoding not in ('onehot', 'label'):
            raise ValueError(f"Unknown encoding '{encoding}'. Choose 'onehot' or 'label'.")

        self.preprocess(threshold_minutes=threshold_minutes)

        if encoding == 'onehot':
            return self.one_hot_encode()
        else:
            return self.label_encode()

    def save(self):
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sanitized_pandas_{savedate}.csv"
        self.df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Data saved as: {filename}")

if __name__ == "__main__":
    transformer = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    data_ohe = transformer.transform(threshold_minutes=15, encoding='onehot')
    transformer.save()
    print("OHE sample:")
    print(data_ohe.head(3))

    sleep(1)

    transformer2 = FlightsTransform("dataset_loty_krakow_20260523_195511.csv")
    data_label = transformer2.transform(threshold_minutes=15, encoding='label')
    transformer2.save()
    print("\nLabel encoding sample:")
    print(data_label.head(3))