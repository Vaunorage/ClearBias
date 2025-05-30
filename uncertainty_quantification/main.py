import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class AleatoricUncertainty(BaseEstimator):

    def __init__(self, method: str = 'entropy', temperature: float = 1.0):
        self.method = method
        self.temperature = temperature
        self.scaler = StandardScaler()
        self._check_method()

    def _check_method(self):
        valid_methods = ['entropy', 'probability_margin', 'label_smoothing']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, X: np.ndarray, y: np.ndarray):

        # Ensure labels are consecutive integers starting from 0
        unique_labels = np.unique(y)
        self.label_encoder = {old: new for new, old in enumerate(unique_labels)}
        self.reverse_encoder = {new: old for old, new in self.label_encoder.items()}
        encoded_y = np.array([self.label_encoder[label] for label in y])
        
        self.n_classes = len(unique_labels)
        self.X = self.scaler.fit_transform(X)
        self.y = encoded_y

        # Train a base classifier (Random Forest in this case)
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(self.X, self.y)
        return self

    def _apply_temperature_scaling(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        log_probs = np.log(probs) / self.temperature
        scaled_probs = np.exp(log_probs)
        return scaled_probs / scaled_probs.sum(axis=1, keepdims=True)

    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute entropy of probability distributions."""
        return entropy(probs.T)

    def _compute_probability_margin(self, probs: np.ndarray) -> np.ndarray:
        """Compute margin between top two probabilities."""
        sorted_probs = np.sort(probs, axis=1)
        return 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])

    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)

        # Apply temperature scaling
        scaled_probs = self._apply_temperature_scaling(probs)

        # Calculate uncertainty based on chosen method
        if self.method == 'entropy':
            uncertainty = self._compute_entropy(scaled_probs)
        elif self.method == 'probability_margin':
            uncertainty = self._compute_probability_margin(scaled_probs)
        else:  # label_smoothing
            uncertainty = 1 - np.max(scaled_probs, axis=1)

        return scaled_probs, uncertainty


class EpistemicUncertainty(BaseEstimator):

    def __init__(self,
                 method: str = 'ensemble',
                 n_estimators: int = 5,
                 dropout_rate: float = 0.5,
                 n_forward_passes: int = 30):

        self.method = method
        self.n_estimators = n_estimators
        self.dropout_rate = dropout_rate
        self.n_forward_passes = n_forward_passes
        self.scaler = StandardScaler()
        self._check_method()

    def _check_method(self):
        valid_methods = ['ensemble', 'mc_dropout', 'evidential']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Encode labels to be zero-based and contiguous
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_encoded = np.array([label_map[label] for label in y])
        self.n_classes = len(unique_labels)
        self.X = self.scaler.fit_transform(X)
        self.y = y_encoded

        if self.method == 'ensemble':
            self.models = []
            for _ in range(self.n_estimators):
                model = RandomForestClassifier()
                model.fit(self.X, self.y)
                self.models.append(model)

        elif self.method == 'mc_dropout':
            input_dim = X.shape[1]
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(32, self.n_classes)
            )
            # Train the model (simplified for demonstration)
            self._train_dropout_model(X, self.y)

        elif self.method == 'evidential':
            input_dim = X.shape[1]
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_classes * 2)  # Output both logits and uncertainty
            )
            # Train the model (simplified for demonstration)
            self._train_evidential_model(X, self.y)

        return self

    def _train_dropout_model(self, X: np.ndarray, y: np.ndarray):
        """Train the dropout model using PyTorch."""
        # Simplified training loop
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        assert y_tensor.min() >= 0 and y_tensor.max() < self.n_classes, f"Target label out of bounds: [{y_tensor.min()}, {y_tensor.max()}], n_classes={self.n_classes}"

        for _ in range(100):  # epochs
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def _train_evidential_model(self, X: np.ndarray, y: np.ndarray):
        """Train the evidential model using PyTorch."""
        assert np.min(y) >= 0 and np.max(y) < self.n_classes, f"Target label out of bounds: [{np.min(y)}, {np.max(y)}], n_classes={self.n_classes}"

        class EvidentialNet(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, n_classes)
                self.evidence = nn.Linear(64, n_classes)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                logits = self.fc2(x)
                evidence = torch.exp(self.evidence(x)) + 1  # Ensure positive evidence
                return logits, evidence

        input_dim = X.shape[1]
        self.model = EvidentialNet(input_dim, self.n_classes)

        def evidential_loss(logits, evidence, targets):
            # Convert targets to one-hot
            y_onehot = torch.zeros(targets.size(0), self.n_classes)
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)

            # Compute Dirichlet parameters (alpha)
            alpha = evidence + 1

            # Compute expected probability
            prob = alpha / alpha.sum(dim=1, keepdim=True)

            # Classification loss
            class_loss = nn.CrossEntropyLoss()(logits, targets)

            # Evidence regularization
            reg_loss = torch.mean(torch.sum(evidence, dim=1))

            return class_loss + 0.1 * reg_loss

        optimizer = torch.optim.Adam(self.model.parameters())

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        self.model.train()
        for epoch in range(100):  # epochs
            optimizer.zero_grad()
            logits, evidence = self.model(X_tensor)
            loss = evidential_loss(logits, evidence, y_tensor)
            loss.backward()
            optimizer.step()

    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        X_scaled = self.scaler.transform(X)
        # Store the label encoder/decoder if not already stored
        if not hasattr(self, 'label_encoder'):
            self.label_encoder = {i: i for i in range(self.n_classes)}
            self.reverse_encoder = {i: i for i in range(self.n_classes)}

        if self.method == 'ensemble':
            # Ensemble code remains the same
            predictions = np.zeros((X_scaled.shape[0], self.n_classes, self.n_estimators))
            for i, model in enumerate(self.models):
                predictions[:, :, i] = model.predict_proba(X_scaled)

            mean_probs = np.mean(predictions, axis=2)
            uncertainty = np.mean(np.var(predictions, axis=2), axis=1)

        elif self.method == 'mc_dropout':
            # MC Dropout code remains the same
            predictions = np.zeros((X_scaled.shape[0], self.n_classes, self.n_forward_passes))
            self.model.train()  # Enable dropout

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                for i in range(self.n_forward_passes):
                    outputs = self.model(X_tensor)
                    predictions[:, :, i] = torch.softmax(outputs, dim=1).numpy()

            mean_probs = np.mean(predictions, axis=2)
            uncertainty = np.mean(np.var(predictions, axis=2), axis=1)

        else:  # evidential
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                logits, evidence = self.model(X_tensor)

                # Compute Dirichlet parameters (alpha)
                alpha = evidence.numpy() + 1

                # Compute expected probability
                mean_probs = alpha / np.sum(alpha, axis=1, keepdims=True)

                # Compute uncertainty as total variance
                # Higher evidence means lower uncertainty
                uncertainty = self.n_classes / np.sum(evidence.numpy(), axis=1)

        return mean_probs, uncertainty


class UncertaintyRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=10, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    def predict_with_uncertainty(self, X):
        # Remove feature names before prediction
        X_array = X.values if hasattr(X, 'values') else X

        # Get predictions from all trees
        predictions = []
        for tree in self.estimators_:
            # Get probabilistic predictions
            leaf_id = tree.apply(X_array)
            tree_pred = tree.tree_.value[leaf_id].reshape(-1, self.n_classes_)
            # Normalize predictions
            tree_pred = tree_pred / np.sum(tree_pred, axis=1, keepdims=True)
            predictions.append(tree_pred[:, 1])  # Get probability of positive class

        predictions = np.array(predictions).T  # Shape: (n_samples, n_estimators)

        # Calculate mean predictions
        mean_pred = np.mean(predictions, axis=1)

        # Calculate uncertainties
        epistemic = np.var(predictions, axis=1)  # Between-model variance
        aleatoric = np.mean(predictions * (1 - predictions), axis=1)  # Within-model variance

        return mean_pred.reshape(-1, 1), epistemic, aleatoric
