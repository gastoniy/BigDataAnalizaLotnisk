import numpy as np
from datetime import datetime

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath
        # Input structure 
        self.input_dt = np.dtype([
            ('id', 'int'), ('data', 'U10'), ('nr_lotu', 'U10'), 
            ('linia', 'U3'), ('kierunek', 'U30'), ('planowany', 'U20'),
            ('rzeczywisty', 'U20'), ('status', 'U20'), ('aktualizacja', 'U20')
        ])
        
        # output structure
        self.output_dt = np.dtype([
            ('nr_lotu', 'U10'), ('linia', 'U3'), ('kierunek', 'U30'), 
            ('planowany', 'U20'), ('rzeczywisty', 'U20'),
            ('dzien_tygodnia', 'int'), ('jest_weekend', 'int'), ('miesiac', 'int')
        ])

        self.data = np.loadtxt(
            self.datapath, delimiter=",", skiprows=1, 
            dtype=self.input_dt, encoding="utf-8"
        )

    def transform(self):
        # sanitizing and adding support fields 
        n_rows = len(self.data)
        # create new table of needed structure, filled with zeroes for now
        self.processed_data = np.zeros(n_rows, dtype=self.output_dt)

        # copy basic fields 
        for col in ['nr_lotu', 'linia', 'kierunek', 'planowany', 'rzeczywisty']:
            self.processed_data[col] = self.data[col]

        # calculate and add support fields
        for i, plan_str in enumerate(self.data['planowany']):
            dt_obj = datetime.strptime(plan_str, "%Y-%m-%d %H:%M:%S")
            
            self.processed_data['dzien_tygodnia'][i] = dt_obj.weekday()
            self.processed_data['jest_weekend'][i] = 1 if dt_obj.weekday() >= 5 else 0
            self.processed_data['miesiac'][i] = dt_obj.month

        return self.processed_data
    
    def save(self):
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sanitized_{savedate}.csv"
        
        # Nagłówek brany bezpośrednio z nazw zdefiniowanych w output_dt
        header_line = ",".join(self.processed_data.dtype.names)
        
        np.savetxt(
            filename, 
            self.processed_data.tolist(), # Konwersja na listę usuwa problemy z paddingiem bajtów
            delimiter=",", 
            fmt='%s',           
            header=header_line,
            comments="",       
            encoding="utf-8"
        )
        print(f"Przetworzone dane zapisano w: {filename}")

if __name__ == "__main__":
    test = FlightsTransform("dataset_loty_krakow_20260512_110908.csv")
    test.transform()
    test.save()
    
    print("\nPróbka danych (3 pierwsze wiersze):")
    print(test.processed_data[:3])