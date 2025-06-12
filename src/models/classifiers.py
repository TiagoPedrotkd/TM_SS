import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Always use bidirectional for better performance
        )
        # Account for bidirectional in final layer (hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size * 2)
        
        # Get final output
        final_out = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        final_out = self.dropout(final_out)
        
        # Project to output classes
        logits = self.fc(final_out)
        return logits

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

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_focal_loss: bool = True,  # Added parameter
        focal_gamma: float = 2.0      # Added parameter
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
    
    def compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights based on class frequencies with additional scaling."""
        classes = np.unique(y)
        class_weights = []
        n_samples = len(y)
        
        # Count samples per class
        class_counts = np.array([np.sum(y == c) for c in classes])
        
        # Compute weights with additional scaling for minority classes
        weights = n_samples / (len(classes) * class_counts)
        
        # Apply additional scaling to minority classes
        max_weight = np.max(weights)
        scaled_weights = np.where(
            weights > max_weight * 0.5,  # For minority classes
            weights * 1.2,  # Increase weight by 20%
            weights
        )
        
        return torch.FloatTensor(scaled_weights).to(self.device)
    
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
        
        # Compute class weights
        class_weights = self.compute_class_weights(y.cpu().numpy())
        
        # Use Focal Loss or weighted CrossEntropyLoss
        if self.use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=self.focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            perm = torch.randperm(len(X))
            X = X[perm]
            y = y[perm]
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
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