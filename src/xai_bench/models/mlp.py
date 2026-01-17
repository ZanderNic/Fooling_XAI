# std-lib imports
from __future__ import annotations
from typing import Optional, Sequence, Union, Literal

# 3 party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# projekt imports
from xai_bench.models.base_model import BaseModel


def _to_numpy(X) -> np.ndarray:
    """Accepts numpy array / torch tensor / list-like and returns float32 numpy."""
    if isinstance(X, np.ndarray):
        arr = X
    elif torch.is_tensor(X):
        arr = X.detach().cpu().numpy()
    else:
        arr = np.asarray(X)
    if arr.dtype == object:
        raise TypeError("X must be numeric, got dtype=object.")
    return arr.astype(np.float32, copy=False)


def _to_torch(X, device: torch.device) -> torch.Tensor:
    """Convert input to float32 torch tensor on device."""
    if torch.is_tensor(X):
        t = X
    else:
        t = torch.from_numpy(_to_numpy(X))
    return t.to(device=device, dtype=torch.float32)


def _check_1d_vector(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D of shape (n,).")
    return y




class TorchMLP(BaseModel):
    """
        Simple fully-connected network.
        - classification: outputs logits -> softmax in predict_proba
        - regression: outputs scalar -> predict_scalar
    """

    def __init__(
        self,
        task: Literal["classification","regression"],
        *,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        num_classes: Optional[int] = None,  # required for classification
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 30,
        batch_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ):
        super().__init__(task,stats=(self,"TorchMLP"))

        if self.task == "classification":
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("For classification, num_classes must be an int >= 2.")
            out_dim = num_classes
        else:
            out_dim = 1

        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers).to(self.device)

        self.num_classes = num_classes

    def fit(self, X, y):
        self.stats("fit",X)
        Xn = _to_numpy(X)
        yn = _check_1d_vector(y)

        X_t = torch.from_numpy(Xn).to(self.device, dtype=torch.float32)

        if self.task == "classification":
            y_t = torch.from_numpy(yn.astype(np.int64, copy=False)).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_t = torch.from_numpy(yn.astype(np.float32, copy=False)).to(self.device)
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n = X_t.shape[0]
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]

                optimizer.zero_grad(set_to_none=True)
                out = self.net(xb)

                if self.task == "classification":
                    loss = criterion(out, yb)
                else:
                    loss = criterion(out.squeeze(1), yb)

                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.stats("predict_proba",X)
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only defined for classification models.")
        
        self.net.eval()
        
        Xt = _to_torch(X, self.device)
        logits = self.net(Xt)  # (n, C)
        proba = F.softmax(logits, dim=1).detach().cpu().numpy()
        
        if proba.ndim != 2:
            raise ValueError("predict_proba must return an array of shape (n, C)")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("predict_proba rows must sum to 1.")
        
        return proba

    @torch.no_grad()
    def predict_scalar(self, X) -> np.ndarray:
        self.stats("predict_scalar",X)
        if self.task != "regression":
            raise NotImplementedError("predict_scalar is only defined for regression models.")
        
        self.net.eval()
        Xt = _to_torch(X, self.device)
        
        out = self.net(Xt).squeeze(1)  # (n,)
        pred = out.detach().cpu().numpy().astype(np.float64, copy=False)
        
        if pred.ndim != 1:
            raise ValueError("predict_scalar must return 1D array of shape (n,)")
        
        return pred