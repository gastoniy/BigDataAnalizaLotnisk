import numpy as np
from datetime import datetime

class FlightsTransform:
    def __init__(self, datapath: str):
        self.datapath = datapath
        # Definicja struktury - musi pasować 1:1 do Twojego CSV
        self.dt = np.dtype([
            ('id', 'int'),
            ('data', 'U10'), 
            ('nr_lotu', 'U10'), 
            ('linia', 'U3'), 
            ('kierunek', 'U30'), 
            ('planowany', 'U20'),
            ('rzeczywisty', 'U20'),
            ('status', 'U20'),
            ('aktualizacja', 'U20')
        ])
        
        self.unsanitized_data = np.loadtxt(
            self.datapath,
            delimiter=",",
            skiprows=1,
            dtype=self.dt,
            encoding="utf-8"
        )

    def sanitize(self):
        # mask of what data fields do we want to keep
        needed_columns = ['nr_lotu', 'linia', 'kierunek', 'planowany', 'rzeczywisty']
        
        # create new table here aplying the mask 
        self.sanitized_data = self.unsanitized_data[needed_columns]
        return self.sanitized_data
    
    def save(self):
        savedate = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename = f"sanitized_{savedate}.csv"
        np.savetxt(
            filename, 
            self.sanitized_data, 
            delimiter=",", 
            fmt='%s',           # Ważne: fmt='%s' obsłuży stringi w tablicy strukturalnej
            header="nr_lotu,linia,kierunek,planowany,rzeczywisty",
            comments="",        # Usuwa domyślny znak '#' przed nagłówkiem
            encoding="utf-8"
        )

if __name__ == "__main__":
    test = FlightsTransform("dataset_loty_krakow_20260512_110908.csv")
    clean_data = test.sanitize()
    test.save()
    
    print("data sample after sanitizaion:")
    print(clean_data[:3])