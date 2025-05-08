import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Tuple, Optional
import numpy as np
import numpy.typing as npt

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
    
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'TorchClassifierWrapper':
        # Convert to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> npt.NDArray:
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> npt.NDArray:
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy() 