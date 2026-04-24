"""
Transformer forecaster — drop-in replacement for LGBMForecaster.

Implements the same predict(features_t, idx, horizon) interface so
MPCController works without any modification.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger

from forecast.transformer_model import TransformerEPF
from forecast.transformer_config import TransformerConfig
from forecast.transformer_dataset import EPFDataset


class TransformerForecaster:
    """
    Drop-in replacement for LGBMForecaster.

    Interface contract (from mpc_controller.py):
        forecaster.predict(features_t, idx, horizon=96) -> np.ndarray[horizon]

    Set these externally before calling predict():
        _full_prices:    (N,) rt_price array
        _full_da_prices: (N,) da_price array (optional, zeros if missing)
        _full_exo:       (N, n_exo) exogenous features array
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
        model: TransformerEPF | None = None,
    ):
        self.config = config or TransformerConfig()
        # Auto-detect: cuda > mps > cpu
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = TransformerEPF(self.config).to(self.device)

        self.model.eval()

        # Set externally, same pattern as LGBMForecaster._full_prices
        self._full_prices: np.ndarray | None = None
        self._full_da_prices: np.ndarray | None = None
        self._full_exo: np.ndarray | None = None
        self._full_seq: np.ndarray | None = None  # (N, 8) full sequence features

    def fit(
        self,
        train_dataset: EPFDataset,
        val_dataset: EPFDataset | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train the Transformer model.

        Args:
            train_dataset: EPFDataset for training
            val_dataset:   EPFDataset for validation (early stopping)
            verbose:       log training progress

        Returns:
            dict with training history (train_loss, val_loss per epoch)
        """
        config = self.config
        model = self.model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0
        )
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        history = {"train_loss": [], "val_loss": [], "val_mae": []}
        best_val_mae = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(config.max_epochs):
            # ---- Training ----
            model.train()
            train_losses = []
            for price_seq, exo, target, norm_stats in train_loader:
                price_seq = price_seq.to(self.device)
                exo = exo.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                pred = model(price_seq, exo)
                loss = criterion(pred, target)
                loss.backward()

                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # ---- Validation ----
            val_mae = float("inf")
            avg_val_loss = float("inf")
            if val_loader is not None:
                model.eval()
                val_losses = []
                val_maes = []
                with torch.no_grad():
                    for price_seq, exo, target, norm_stats in val_loader:
                        price_seq = price_seq.to(self.device)
                        exo = exo.to(self.device)
                        target = target.to(self.device)
                        norm_stats = norm_stats.to(self.device)

                        pred = model(price_seq, exo)
                        loss = criterion(pred, target)
                        val_losses.append(loss.item())

                        # Denormalize for MAE in real units
                        mean = norm_stats[:, 0:1]  # (B, 1)
                        std = norm_stats[:, 1:2]    # (B, 1)
                        pred_real = pred * std + mean
                        target_real = target * std + mean
                        mae = torch.abs(pred_real - target_real).mean().item()
                        val_maes.append(mae)

                avg_val_loss = np.mean(val_losses)
                val_mae = np.mean(val_maes)
                history["val_loss"].append(avg_val_loss)
                history["val_mae"].append(val_mae)

                # Early stopping
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.patience:
                    if verbose:
                        logger.info(f"  Early stopping at epoch {epoch+1} (best val MAE: {best_val_mae:.2f})")
                    break

            if verbose and (epoch + 1) % 5 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.6f}, "
                    f"val_loss={avg_val_loss:.6f}, val_MAE={val_mae:.2f}, lr={lr:.2e}"
                )

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            if verbose:
                logger.info(f"  Restored best model (val MAE: {best_val_mae:.2f})")

        model.eval()
        self.model = model
        return history

    def predict(self, features_t: np.ndarray, idx: int, horizon: int = 96) -> np.ndarray:
        """
        Predict next `horizon` prices — identical signature to LGBMForecaster.

        Args:
            features_t: (D,) feature vector at time t — used only for exo extraction fallback
            idx:        current index in the FULL price array
            horizon:    number of steps to predict (default 96)

        Returns:
            (horizon,) predicted prices in real units (元/MWh)
        """
        W = self.config.window_size
        norm_w = self.config.norm_window

        if self._full_prices is None:
            raise RuntimeError("Set _full_prices before calling predict()")

        # ---- Build sequence window ----
        win_start = max(0, idx - W + 1)
        win_end = idx + 1

        # Use full sequence features if available (8 channels)
        if self._full_seq is not None:
            seq_win = self._full_seq[win_start:win_end].astype(np.float32)
        else:
            # Fallback: build from prices only (3 channels)
            rt_win = self._full_prices[win_start:win_end].astype(np.float32)
            da_win = self._full_da_prices[win_start:win_end].astype(np.float32) if self._full_da_prices is not None else np.zeros_like(rt_win)
            sp_win = da_win - rt_win
            seq_win = np.stack([rt_win, da_win, sp_win], axis=-1)

        # Pad if not enough history
        if len(seq_win) < W:
            pad_len = W - len(seq_win)
            seq_win = np.pad(seq_win, ((pad_len, 0), (0, 0)), mode="edge")

        # ---- Normalize price channels ----
        rt_col = seq_win[:, 0]
        norm_slice = rt_col[-norm_w:]
        mean = norm_slice.mean()
        std = norm_slice.std()
        if std < 1e-6:
            std = 1.0

        seq_win = seq_win.copy()
        seq_win[:, 0] = (seq_win[:, 0] - mean) / std  # rt_price
        seq_win[:, 1] = (seq_win[:, 1] - mean) / std  # da_price
        seq_win[:, 2] = seq_win[:, 2] / std            # spread

        price_seq = seq_win  # (W, n_price_features)

        # ---- Exogenous features ----
        if self._full_exo is not None and idx < len(self._full_exo):
            exo = self._full_exo[idx].astype(np.float32)
        else:
            # Fallback: use features_t as exo (truncate/pad to n_exo_features)
            exo = np.zeros(self.config.n_exo_features, dtype=np.float32)
            n = min(len(features_t), self.config.n_exo_features)
            exo[:n] = features_t[:n]

        # ---- Inference ----
        with torch.no_grad():
            price_tensor = torch.from_numpy(price_seq).unsqueeze(0).to(self.device)
            exo_tensor = torch.from_numpy(exo).unsqueeze(0).to(self.device)
            pred_norm = self.model(price_tensor, exo_tensor).cpu().numpy()[0]

        # ---- Denormalize ----
        pred = pred_norm * std + mean
        pred = np.clip(pred, -500, 50000)

        # Return requested horizon (usually 96)
        if horizon <= len(pred):
            return pred[:horizon]
        else:
            # Pad with last value if horizon > 96
            return np.pad(pred, (0, horizon - len(pred)), mode="edge")

    def save(self, path: str | Path):
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"  Saved model to {path}")

    def load(self, path: str | Path):
        """Load model weights."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "config" in checkpoint:
            self.config = checkpoint["config"]
            self.model = TransformerEPF(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        logger.info(f"  Loaded model from {path}")
