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
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers
        lstm_out_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(lstm_out_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # shape: (batch_size, 1, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # Get last output
        if self.bidirectional:
            # Concatenate last outputs from both directions
            last_output = torch.cat((lstm_out[:, -1, :self.hidden_size],
                                   lstm_out[:, 0, self.hidden_size:]), dim=1)
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Feed through linear layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # shape: (batch_size, 1, input_size)
        
        # Project input to hidden size
        x = self.input_proj(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Pool sequence dimension (use mean pooling)
        x = torch.mean(x, dim=1)
        
        # Apply final classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
    
    def compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights based on class frequencies."""
        classes = np.unique(y)
        class_weights = []
        n_samples = len(y)
        
        for c in classes:
            class_weights.append(n_samples / (len(classes) * np.sum(y == c)))
        
        return torch.FloatTensor(class_weights).to(self.device)
    
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
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
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
            
            avg_loss = epoch_loss / num_batches
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
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