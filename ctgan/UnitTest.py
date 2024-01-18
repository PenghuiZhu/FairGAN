import json
import numpy as np
import pandas as pd

import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from ctgan import data

# Load the dataset
df = pd.read_csv('/Users/penghuizhu/Desktop/Workspace/LocalTestCTGAN/examples/csv/adult.csv')

# Assuming the decision variable is 'income' and the protected attribute is 'sex'
# and 'sex' is binary coded as 0 or 1 for simplicity
features = df.drop(['income', 'sex'], axis=1)
decision = df['income']
protected_attr = df['sex']

print(features,decision,protected_attr)
# Preprocess the dataset
# Identify categorical and numerical columns
categorical_columns = features.select_dtypes(include=['object']).columns
numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessor with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Fit the preprocessor and transform the data
X = preprocessor.fit_transform(features)
Y = decision.apply(lambda x: 1 if x == '>50K' else 0).values
S = protected_attr.values

X = X.toarray()
S = protected_attr.map({' Male': 1, ' Female': 0}).values.astype(float)


# Convert the data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for Y
S_tensor = torch.tensor(S, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for S


# the dataset in the [x, y, s] format ready to be used in a PyTorch model
S_tensor = S_tensor[:500]
S_size = S_tensor.size()
print(S_tensor,S_size)

mean = torch.zeros(500, 128)
"standard deviation, used for generate noise vector"
std = mean + 1
fakez = torch.normal(mean=mean, std=std)
print('noise z' , format(fakez), fakez.size())

protected_attr = data.data_loader('/Users/penghuizhu/Desktop/Workspace/LocalTestCTGAN/examples/csv/adult.csv')


print(type(data))
print(dir(data))
