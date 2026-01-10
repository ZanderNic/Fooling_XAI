# std-lib imports
from __future__ import annotations
from typing import Optional, Sequence, Union, Literal

# 3 party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# projekt imports
from xai_bench.base import BaseModel


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



class TorchCNN1D(BaseModel):
    """
    1D-CNN for sequence-like input.
    Expected X shape:
        - (n, L)  -> will be treated as (n, 1, L)
        - (n, C, L)
    """

    def __init__(
        self,
        task: Literal["classification","regression"],
        *,
        in_channels: int = 1,
        seq_len: int,
        num_classes: Optional[int] = None,  # required for classification
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 30,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ):
        super().__init__(task)

        if self.task == "classification":
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("For classification, num_classes must be an int >= 2.")
            out_dim = num_classes
        else:
            out_dim = 1

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.in_channels = in_channels

        # Build conv stack
        convs: list[nn.Module] = []
        c_prev = in_channels
        padding = kernel_size // 2  # keep length
        for c in channels:
            convs += [
                nn.Conv1d(c_prev, c, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ]
            c_prev = c
        self.conv = nn.Sequential(*convs)

        # compute flattened size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self.net = nn.Sequential(self.conv, nn.Flatten(), self.head).to(self.device)

    def _prepare_X(self, X) -> torch.Tensor:
        Xn = _to_numpy(X)
        if Xn.ndim == 2:
            # (n, L) -> (n, 1, L)
            Xn = Xn[:, None, :]
        if Xn.ndim != 3:
            raise ValueError("CNN1D expects X of shape (n, L) or (n, C, L).")
        if Xn.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {Xn.shape[1]}.")
        if Xn.shape[2] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {Xn.shape[2]}.")
        return torch.from_numpy(Xn).to(self.device, dtype=torch.float32)

    def fit(self, X, y) -> "TorchCNN1D":
        Xt = self._prepare_X(X)
        yn = _check_1d_vector(y)

        if self.task == "classification":
            y_t = torch.from_numpy(yn.astype(np.int64, copy=False)).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_t = torch.from_numpy(yn.astype(np.float32, copy=False)).to(self.device)
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n = Xt.shape[0]
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = Xt[idx]
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
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only defined for classification models.")
        self.net.eval()
        Xt = self._prepare_X(X)
        logits = self.net(Xt)  # (n, C)
        proba = F.softmax(logits, dim=1).detach().cpu().numpy()
        if proba.ndim != 2:
            raise ValueError("predict_proba must return an array of shape (n, C)")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("predict_proba rows must sum to 1.")
        return proba

    @torch.no_grad()
    def predict_scalar(self, X) -> np.ndarray:
        if self.task != "regression":
            raise NotImplementedError("predict_scalar is only defined for regression models.")
        self.net.eval()
        Xt = self._prepare_X(X)
        out = self.net(Xt).squeeze(1)
        pred = out.detach().cpu().numpy().astype(np.float64, copy=False)
        if pred.ndim != 1:
            raise ValueError("predict_scalar must return 1D array of shape (n,)")
        return pred