import pandas 
from collections import Counter

table = pandas.read_csv('sanitized_pandas_20260515_105307.csv')

airlines = table['czy_opozniony']

print(Counter(airlines).total)