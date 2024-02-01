from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 1000)
ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)
synthetic_data = ctgan.sample(n=10)
fair_data = ctgan.fairness_ensure(synthetic_data)

print(fair_data)

