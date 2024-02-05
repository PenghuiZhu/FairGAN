import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ctgan import CTGAN, load_demo


class Metrics:

    # def __init__(self, mandatory_param, optional_param1='default1', optional_param2=None):
    #     pass

    def binary_fair_data_generation_metrics(self, df, protected_attr, target_attr):
        """
        Calculate fair data generation metrics for binary protected attributes.
        """
        # Convert DataFrame to Torch tensors
        protected_tensor = torch.tensor(df[protected_attr].values)
        target_tensor = torch.tensor(df[target_attr].values)

        # Calculate Risk Difference
        positive_outcome_protected = target_tensor[protected_tensor == 1].float().mean()
        positive_outcome_unprotected = target_tensor[protected_tensor == 0].float().mean()
        rd = positive_outcome_protected - positive_outcome_unprotected

        # Calculate Balanced Error Rate (Placeholder, implement as needed)
        ber = torch.tensor(0.0)  # Placeholder for BER calculation

        metrics_dict = {"Risk Difference": rd.item(), "Balanced Error Rate": ber.item()}
        return metrics_dict

    def euclidean_distance(self, df_synthetic, df_real, columns):
        """
        Calculate euclidean distance of joint probability distributions between two datasets.
        """
        synthetic_tensor = torch.tensor(df_synthetic[columns].values, dtype=torch.float32)
        real_tensor = torch.tensor(df_real[columns].values, dtype=torch.float32)

        # Euclidean distance calculation
        distance = torch.norm(synthetic_tensor - real_tensor, p=2) / synthetic_tensor.shape[0]
        return {"Euclidean Distance": distance.item()}

    def binary_fair_classification_metrics(self, X_real, y_real, y_pred):
        """
        Calculate fair data classification metrics for binary protected attributes.
        """
        # Convert inputs to numpy arrays for sklearn metrics
        y_real = y_real.to_numpy() if isinstance(y_real, pd.Series) else np.array(y_real)
        y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.array(y_pred)

        acc = accuracy_score(y_real, y_pred)
        f1 = f1_score(y_real, y_pred)

        metrics_dict = {"Accuracy": acc, "F1 Score": f1}
        return metrics_dict

metrics = Metrics()
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

df_synthetic = synthetic_data  # Your synthetic DataFrame
df_real = pd.read_csv('/Users/penghuizhu/Desktop/Workspace/FairGAN/examples/csv/adult.csv')  # Your real DataFrame
protected_attr = 'sex'  # Example protected attribute column name
target_attr = 'income'  # Example target/outcome column name

fairness_metrics = metrics.binary_fair_data_generation_metrics(df_synthetic, protected_attr, target_attr)
print(fairness_metrics)
distance_metrics = metrics.euclidean_distance(df_synthetic, df_real)
print(distance_metrics)
