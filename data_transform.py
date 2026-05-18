import pandas as pd
import numpy as np
from datetime import datetime 
import airportsdata
from sklearn.preprocessing import OneHotEncoder

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath
        self.df = pd.read_csv(datapath)
        self.aerodata = airportsdata.load('IATA')

        self.top_airlines = ['LS', 'D8', 'TK', 'IZ', 'EZY', 'A3', 'FH', '4M', 'LX',
                             'RK', 'TOM', 'DY', 'AF', 'EJU', 'LH', 'EN', 'LG', 'PC',
                             'ENT', 'EW', 'W4', 'FR', 'LO', 'W6', 'SK', 'OS', 'RR',
                             'BA', 'XQ', 'LY', 'KLJ', 'MGH', 'SN', 'JU', 'EZS', 'AY',
                             'KL']

        # 1hot encoder (mamy to 37 linii, inne ignorowane )
        self.encoder = OneHotEncoder(categories=[self.top_airlines],
                                     handle_unknown='ignore',
                                     sparse_output=False)
        
        fake_data = pd.DataFrame({'linia_lotnicza': self.top_airlines})
        self.encoder.fit(fake_data)

    # zamiast nazwy celu otrzymujemy koordynaty
    def _convert_icao(self, old_dest: str):
        try:
            iata = old_dest.split('(')[-1].replace(')', '').strip()
            inf = self.aerodata.get(iata)
            if inf:
                return inf['lat'], inf['lon'], inf['elevation']
        except:
            pass
        return None, None, None 

    def transform(self, threshold_minutes: int = 15):
        needed_cols = ['numer_lotu', 'linia_lotnicza', 'kierunek', 'czas_planowany', 'czas_rzeczywisty']
        self.df = self.df[needed_cols].copy()

        # konwersja na współrzędne
        result = self.df['kierunek'].apply(self._convert_icao)
        self.df[['lat', 'lon', 'elev']] = pd.DataFrame(result.tolist(), index=self.df.index)
        self.df = self.df.drop('kierunek', axis=1)

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
        self.df = self.df.dropna()

        return self.df
    
    def save(self):
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sanitized_pandas_{savedate}.csv"
        self.df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Data saved as: {filename}")

if __name__ == "__main__":
    transformer = FlightsTransform("dataset_loty_krakow_20260513_183527.csv")
    data = transformer.transform(threshold_minutes=15)
    transformer.save()
    
    print("Data sample:")
    print(data[['dystans_km', 'godzina_sin', 'godzina_cos', 'czy_opozniony']].head(3))