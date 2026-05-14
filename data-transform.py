import pandas as pd
from datetime import datetime 
import airportsdata

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath

        self.df = pd.read_csv(datapath)

        self.aerodata = airportsdata.load('IATA')


    def _convert_icao(self,old_dest:str):
        try:
            iata = old_dest.split('(')[-1].replace(')', '').strip()
            inf = self.aerodata.get(iata)
            if inf:
                return inf['lat'], inf['lon'], inf['elevation']
        except:
            pass
        return None, None, None 
            

    def transform(self, threshold_minutes: int = 15):
        # initial sanitization
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

        # classifcation
        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

 
        # unix timestamp conversion
        self.df['czas_planowany_unix'] = self.df['czas_planowany'].astype('int64') // 10**9
        self.df['czas_rzeczywisty_unix'] = self.df['czas_rzeczywisty'].astype('int64') // 10**9

        # removing old columns
        self.df = self.df.drop(['numer_lotu', 'czas_planowany', 'czas_rzeczywisty'], axis=1)

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
    
    print("\Data sample:")
    print(data.head(3))