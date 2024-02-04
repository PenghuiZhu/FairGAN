import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class Classifier:
    def __init__(self, test_size=0.2, random_seed=42):
        self.test_size = test_size
        self.random_seed = random_seed

    def logistic_regression(self, df, df_real):
        """
        Train a logistic regression model on synthetic data and predict on real data.
        """
        # Assuming the last column is the target variable
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_real = df_real.iloc[:, :-1]
        y_real = df_real.iloc[:, -1]

        # Train-test split for evaluation (optional, can be skipped or modified as needed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)

        # Logistic Regression in PyTorch
        model = nn.Sequential(nn.Linear(X_train.shape[1], 1), nn.Sigmoid())
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        # Training loop
        for epoch in range(100):  # Number of epochs can be adjusted
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Predict on real data
        model.eval()
        X_real_tensor = torch.tensor(X_real.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_real_tensor).squeeze().numpy()
            y_pred_class = np.round(y_pred)  # Convert probabilities to class labels

        return X_real, y_real.values, y_pred_class

    def random_forest(self, df, df_real):
        """
        Train a random forest model on synthetic data and predict on real data.
        """
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_real = df_real.iloc[:, :-1]
        y_real = df_real.iloc[:, -1]

        # Random Forest Classifier
        clf = RandomForestClassifier(random_state=self.random_seed)
        clf.fit(X, y)
        y_pred = clf.predict(X_real)

        return X_real, y_real.values, y_pred
