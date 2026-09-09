import pandas as pd
import numpy as np

data = {
    'age':    [25, None, 35, 40, None, 29],
    'salary': [50000, 60000, None, 80000, 55000, 62000],
    'city':   ['Tashkent', 'Samarkand', 'Tashkent', None, 'Bukhara', 'Samarkand'],
    'bought': [1, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
df=df["age", "salary"].fillna(df["age", "salary"].median())
