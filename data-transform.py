import pandas as pd
from datetime import datetime 

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath

        self.df = pd.read_csv(datapath)

    def transform(self, threshold_minutes: int = 15):
        # sanitization and addition of new columns 
        needed_cols = ['numer_lotu','linia_lotnicza','kierunek','czas_planowany','czas_rzeczywisty']
        self.df = self.df[needed_cols].copy()

        self.df['czas_planowany'] = pd.to_datetime(self.df['czas_planowany'])
        self.df['czas_rzeczywisty'] = pd.to_datetime(self.df['czas_rzeczywisty'])


        self.df['dzien_tygodnia'] = self.df['czas_planowany'].dt.weekday
        self.df['jest_weekend'] = (self.df['dzien_tygodnia'] >= 5).astype(int)
        self.df['miesiac'] = self.df['czas_planowany'].dt.month

        delay_seconds = (self.df['czas_rzeczywisty'] - self.df['czas_planowany']).dt.total_seconds()
        self.df['czy_opozniony'] = (delay_seconds > (threshold_minutes * 60)).astype(int)

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