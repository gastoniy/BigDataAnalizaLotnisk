import pandas as pd
from datetime import datetime 
import airportsdata

from sklearn.preprocessing import OneHotEncoder

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath

        self.df = pd.read_csv(datapath)

        self.aerodata = airportsdata.load('IATA')

        # defining top 37 airlines based on long-term observiation
        self.top_airlines = ['LS', 'D8', 'TK', 'IZ', 'EZY', 'A3', 'FH', '4M', 'LX',
                             'RK', 'TOM', 'DY', 'AF', 'EJU', 'LH', 'EN', 'LG', 'PC',
                             'ENT', 'EW', 'W4', 'FR', 'LO', 'W6', 'SK', 'OS', 'RR',
                             'BA', 'XQ', 'LY', 'KLJ', 'MGH', 'SN', 'JU', 'EZS', 'AY',
                             'KL']

        # encoder initialization
        self.encoder = OneHotEncoder(categories=[self.top_airlines],
                                      handle_unknown='ignore', # if new airline would suddenly appear (which is going to be unpopular) it's going to have all 0's 
                                      sparse_output=False # do not create csr matrix 
                                      )
        
        # creating and fitting clasifier on the fake data so no matter what happens in CSV 
        # we are getting all the columns 
        fake_data = pd.DataFrame({'linia_lotnicza': self.top_airlines})
        self.encoder.fit(fake_data)

        # label encoding map: known airlines get a stable index, unknowns get -1
        self.label_map = {airline: idx for idx, airline in enumerate(self.top_airlines)}


    def _convert_icao(self,old_dest:str):
        try:
            iata = old_dest.split('(')[-1].replace(')', '').strip()
            inf = self.aerodata.get(iata)
            if inf:
                return inf['lat'], inf['lon'], inf['elevation']
        except:
            pass
        return None, None, None 
            

    def preprocess(self, threshold_minutes: int = 15) -> pd.DataFrame:
        """
        Step 1 — sanitize columns, extract coordinates, build datetime features,
        compute unix timestamps, derive the delay label, and drop raw columns.
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

        # coordinate extraction from the destination
        result = self.df['kierunek'].apply(self._convert_icao)
        self.df[['lat', 'lon', 'elev']] = pd.DataFrame(result.tolist(), index=self.df.index)
        self.df = self.df.drop('kierunek', axis=1)

        # converting to datetime
        self.df['czas_planowany'] = pd.to_datetime(self.df['czas_planowany'])
        self.df['czas_rzeczywisty'] = pd.to_datetime(self.df['czas_rzeczywisty'])

        # getting important fields
        self.df['dzien_tygodnia'] = self.df['czas_planowany'].dt.weekday
        self.df['jest_weekend'] = (self.df['dzien_tygodnia'] >= 5).astype(int)
        self.df['miesiac'] = self.df['czas_planowany'].dt.month

        # unix timestamp conversion
        self.df['czas_planowany_unix'] = self.df['czas_planowany'].astype('int64') // 10**9
        self.df['czas_rzeczywisty_unix'] = self.df['czas_rzeczywisty'].astype('int64') // 10**9

        # classification label
        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

        # drop columns no longer needed (airline encoding happens in a separate step)
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
        Step 2b — label-encode the 'linia_lotnicza' column.
        Known airlines receive a stable integer index (0-based); unknowns are mapped to -1.
        Requires preprocess() to have been called first.

        Returns:
            DataFrame with 'linia_lotnicza' replaced by 'linia_lotnicza_label' (int).
        """
        if 'linia_lotnicza' not in self.df.columns:
            raise ValueError("Column 'linia_lotnicza' not found. Run preprocess() first.")

        self.df['linia_lotnicza_label'] = (
            self.df['linia_lotnicza'].map(self.label_map).fillna(-1).astype(int)
        )
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
    transformer = FlightsTransform("dataset_loty_krakow_20260518_183551.csv")
    data_ohe = transformer.transform(threshold_minutes=15, encoding='onehot')
    transformer.save()
    print("OHE sample:")
    print(data_ohe.head(3))

    transformer2 = FlightsTransform("dataset_loty_krakow_20260518_183551.csv")
    data_label = transformer2.transform(threshold_minutes=15, encoding='label')
    transformer2.save()
    print("\nLabel encoding sample:")
    print(data_label.head(3))