from ctgan import CTGAN
from ctgan.synthesizers.fairgan import FAIRGAN
import pandas as pd

real_data = pd.read_csv("data/adult.csv")

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
protected_columns = 'sex'   # 保护属性

epochs = 5
sample_num = 1000

ctgan = CTGAN(epochs=epochs)
fairgan = FAIRGAN(epochs=epochs)

ctgan.fit(real_data, discrete_columns, epochs=epochs)
fairgan.fit(real_data, discrete_columns, protected_columns, epochs=epochs)

# Create synthetic data
synthetic_data = ctgan.sample(sample_num)
synthetic_data2 = fairgan.sample(sample_num)

synthetic_data.to_csv("ctgan_test.csv")
synthetic_data2.to_csv("fairgan_test.csv")